import os
import pdb
import sys
from sched import scheduler

import torch
import numpy as np
import yaml

import datetime
import logging
import importlib
import argparse

from pathlib import Path

from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from Dataset.ModelNetDataset_dat import ModelNet40_GFT_dat


import torch.nn as nn
from pytorch3d.ops import sample_farthest_points,knn_gather,knn_points
from utils.visualization import visualize_two_point_clouds,plot_frequency_spectrum_bar


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def hausdorff_loss_py(adv_pc, ori_pc):
    adv_KNN = knn_points(adv_pc, ori_pc, K=1)
    hd_loss_adv_to_ori = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0]
    ori_KNN = knn_points(ori_pc, adv_pc, K=1)
    hd_loss_ori_to_adv = ori_KNN.dists.contiguous().squeeze(-1).max(-1)[0]
    hd_loss = torch.max(hd_loss_adv_to_ori, hd_loss_ori_to_adv)
    return hd_loss.mean()

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--name', type=str, default='Test', help='name of the experiment')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointPN_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='choose data set [modelnet40, shapenet]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40, 16], help='training on ModelNet40/shapenet')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    # parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--index_file', type=str, default='random_numbers_train.npy', help='index file')
    parser.add_argument('--target_label', type=int, default=8, help='the attacker-specified target label')
    parser.add_argument('--seed', type=int, default=256, help='random seed')

    parser.add_argument('--K', type=int, default=10, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--patch_size', type=int, default=32, help='patch size')
    parser.add_argument('--poison_weight_attack', type=float, default=1.0, help='weight of attack loss')
    parser.add_argument('--L2_weight', type=float, default=0, help='weight of L2 loss')
    parser.add_argument('--chamfer_weight', type=float, default=0, help='weight of chamfer loss')
    parser.add_argument('--Hausdorff_weight', type=float, default=0, help='weight of Hausdorff loss')
    parser.add_argument('--initial_noise_level',type=float,default=0.1,help='initial noise level')
    parser.add_argument('--attack_lr', default=0.01, type=float, help='learning rate for attacks')
    parser.add_argument('--topk', type=int, default=16, help='topk')

    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')

    parser.add_argument('--target_model', default='log/modelnet40_PointPN_cls/v51_1L10C1H_selectedPatch_patchsize32_topk16/2025-02-11_10-16/',
                        type=str, help='log of the experiment')
    parser.add_argument('--pth_choice', default='best', type=str, help='choose best or final model')

    return parser.parse_args()

def create_bd(points_bd, U, sorted_neighborhood, center, sorted_idx_group,  patch_num, GFT_noise,indices ):
    B, N, _ = points_bd.shape
    _, G, M, _ = sorted_neighborhood.shape

    batch_indices = torch.arange(B).unsqueeze(1)
    selected_group = sorted_neighborhood[batch_indices, indices, :, :].contiguous()
    selected_U = U[batch_indices, indices, :, :].contiguous()

    assert selected_group.shape[1] == patch_num
    selected_group = selected_group.view(B*patch_num, M, 3)
    selected_U = selected_U.view(B*patch_num, M, M)
    freq_points = torch.einsum('bij,bjk->bik', selected_U.transpose(1, 2), selected_group)
    freq_points = freq_points + GFT_noise
    selected_group_bd = torch.einsum('bij,bjk->bik', selected_U, freq_points)
    selected_group_bd = selected_group_bd.view(B, patch_num * M, 3)
    selected_group = selected_group.view(B, patch_num * M, 3)

    selected_idx_group = sorted_idx_group[batch_indices, indices, :].contiguous()
    selected_idx_group = selected_idx_group.view(B, patch_num * M)

    batch_indices = torch.arange(points_bd.size(0)).unsqueeze(1).expand_as(selected_idx_group)
    points_bd[batch_indices, selected_idx_group] = selected_group_bd
    return points_bd,selected_group,selected_group_bd

def test(classifier,  testDataLoader, GFT_noise,args):
    mean_correct_clean = 0
    mean_correct_bd = 0
    class_acc = np.zeros((args.num_classes, 3))
    classifier.eval()

    patch_num = args.num_point//args.patch_size

    all_l2_loss = 0
    all_chamfer_loss = 0
    all_Hausdorff_loss=0


    print(args.target_model)
    plot_frequency_spectrum_bar(GFT_noise.transpose(0, 1))

    for _, (points, labels, conduct_poison,U,neighborhood, center, idx_group, kappa_group) in tqdm(enumerate(testDataLoader),total=len(testDataLoader)):
        points, labels = points.cuda(), labels.cuda()
        U, neighborhood, center, idx_group, kappa_group = U.cuda(),neighborhood.cuda(), center.cuda(), idx_group.cuda(), kappa_group.cuda()
        GFT_noise = GFT_noise.cuda()
        B, N, _ = points.shape
        nn_kappa_std = knn_gather(kappa_group.unsqueeze(2), idx_group).squeeze(3)
        nn_kappa_std = nn_kappa_std.mean(2)
        vales,indices = torch.topk(nn_kappa_std, k=args.topk, dim=1, largest=True)
        testing_points_bd,selected_group,selected_group_bd = create_bd(points.detach().clone(), U, neighborhood, center, idx_group, args.topk, GFT_noise,indices)
        target_labels = torch.ones_like(labels) * args.target_label

        visualize_two_point_clouds(points[0].detach().cpu().numpy(), testing_points_bd[0].detach().cpu().numpy())

        if args.model == 'PointTransformer_cls'or args.model=='PCT_cls':
            pred_clean = classifier(points)
            pred_bd = classifier(testing_points_bd)
        else:
            pred_clean = classifier(points.permute(0,2,1).contiguous())
            pred_bd = classifier(testing_points_bd.permute(0,2,1).contiguous())

        pred_choice = pred_clean.data.max(1)[1]
        pred_choice_bd = pred_bd.data.max(1)[1]

        l2_loss = torch.norm(points-testing_points_bd,p=2,dim=(1,2)).mean()
        all_l2_loss += l2_loss.item()*B
        chamfer_loss = chamfer_distance(points, testing_points_bd, batch_reduction="mean", point_reduction="mean")[0]
        all_chamfer_loss += chamfer_loss.item()*B

        Hausdorff_loss = hausdorff_loss_py(points, testing_points_bd)
        all_Hausdorff_loss += Hausdorff_loss.item()*B

        for cat in np.unique(labels.cpu()):
            classacc = pred_choice[labels == cat].eq(labels[labels == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(
                points[labels == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(labels.long().data).cpu().sum()
        correct_bd = pred_choice_bd.eq(target_labels.long().data).cpu().sum()
        mean_correct_clean += correct.item()
        mean_correct_bd += correct_bd.item()
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = mean_correct_clean / len(testDataLoader.dataset)
    instance_ASR = mean_correct_bd / len(testDataLoader.dataset)
    print('l2_loss:',all_l2_loss/len(testDataLoader.dataset))
    print('chamfer_loss:',all_chamfer_loss/len(testDataLoader.dataset))
    print('Hausdorff_loss:',all_Hausdorff_loss/len(testDataLoader.dataset))

    return instance_acc, class_acc, instance_ASR
    
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log') / f"{args.dataset}_{args.model}" / str(args.name) /  timestr
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    import shutil
    current_file_path = os.path.abspath(__file__)
    shutil.copy2(current_file_path, exp_dir)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    with open(os.path.join(log_dir, 'data.yaml'), 'w') as file:
        yaml.dump(args, file, default_flow_style=False, allow_unicode=True)

    log_string('Load dataset ...')

    if 'modelnet' in args.dataset:
        DATA_PATH = './data/modelnet40_normal_resampled/'
        TEST_DATASET = ModelNet40_GFT_dat(root=DATA_PATH, args=args, split='test')
    else:
        raise Exception("no such dataset")
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of training data is: %d" % len(TEST_DATASET))

    if args.model == 'DGCNN_cls':
        args.k = 20
        args.emb_dims = 1024
        args.dropout = 0.5
        model = importlib.import_module(args.model)
        classifier = model.get_model(args,args.num_classes).cuda()
    elif args.model =='PointTransformer_cls':
        args.model_nblocks = 4
        args.model_nneighbor= 16
        args.model_input_dim = 3
        args.model_transformer_dim= 512
        model = importlib.import_module(args.model)
        classifier = model.get_model(args).cuda()
    else:
        model = importlib.import_module(args.model)
        classifier = model.get_model(args.num_classes).cuda()

    if args.pth_choice == 'best':
        checkpoint = torch.load(os.path.join(args.target_model, 'checkpoints_best', 'best_model.pth'))
    else:
        checkpoint = torch.load(os.path.join(args.target_model, 'checkpoints', 'last_model.pth'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    GFT_noise = checkpoint['GFT_noise']

    GFT_noise = torch.tensor(GFT_noise, dtype=torch.float32).cuda()
    trigger_optimizer = torch.optim.Adam([GFT_noise], lr=args.attack_lr,weight_decay=1e-4)
    scheduler_noise = torch.optim.lr_scheduler.CosineAnnealingLR(trigger_optimizer, args.epoch, eta_min=1e-4)

    with torch.no_grad():
        instance_acc, class_acc, instance_ASR = test(classifier,  testDataLoader, GFT_noise,args)
        log_string('Test Instance Accuracy: %f, Instance ASR: %f, Class Accuracy: %f' % (
        instance_acc, instance_ASR, class_acc))

if __name__=='__main__':
    args = parse_args()
    main(args)