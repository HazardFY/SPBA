import os
import pdb
import sys

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
from Dataset.ShapeNetPart_dat import ShapeNetDataset_GFT_dat
from utils.loss import get_loss_cls

import matplotlib.pyplot as plt
from knn_cuda import KNN
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points,knn_gather,knn_points

import sklearn.metrics as metrics
from utils.visualization import visualize_two_point_clouds,plot_frequency_spectrum_bar


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--name', type=str, default='SPBA', help='name of the experiment')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNext_cls', help='model name [default: pointnet_cls,DGCNN_cls, pointnet2_cls]')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='choose data set [modelnet40, shapenet]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[ 40, 16], help='training on ModelNet40/shapenet')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--index_file', type=str, default='random_numbers_train.npy', help='index file')
    parser.add_argument('--target_label', type=int, default=8, help='the attacker-specified target label')
    parser.add_argument('--seed', type=int, default=256, help='random seed')

    parser.add_argument('--K', type=int, default=10, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--patch_size', type=int, default=32, help='patch size')
    parser.add_argument('--poison_weight_attack', type=float, default=1.0, help='weight of attack loss')
    parser.add_argument('--L2_weight', type=float, default=1.0, help='weight of L2 loss')
    parser.add_argument('--chamfer_weight', type=float, default=10.0, help='weight of chamfer loss')
    parser.add_argument('--Hausdorff_weight', type=float, default=1.0, help='weight of Hausdorff loss')
    parser.add_argument('--initial_noise_level',type=float,default=0.3,help='initial noise level')
    parser.add_argument('--attack_lr', default=0.01, type=float, help='learning rate for attacks')
    parser.add_argument('--topk', type=int, default=16, help='topk')

    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')


    return parser.parse_args()


def fps(data, number):
    fps_data,_ = sample_farthest_points(data, K=number)
    fps_data = fps_data.contiguous()
    return fps_data

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center,_ = sample_farthest_points(xyz, K=self.num_group)
        center = center.contiguous()
        _, idx_group = self.knn(xyz, center)
        assert idx_group.size(1) == self.num_group
        assert idx_group.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx_group + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        return neighborhood, center, idx_group


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


def hausdorff_loss_py(adv_pc, ori_pc):
    adv_KNN = knn_points(adv_pc, ori_pc, K=1)
    hd_loss_adv_to_ori = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0]

    ori_KNN = knn_points(ori_pc, adv_pc, K=1)
    hd_loss_ori_to_adv = ori_KNN.dists.contiguous().squeeze(-1).max(-1)[0]

    hd_loss = torch.max(hd_loss_adv_to_ori, hd_loss_ori_to_adv)
    return hd_loss.mean()


def train(classifier, trainDataLoader, optimizer, criterion_cls,  trigger_optimizer, GFT_noise, args, log_string, global_epoch):
    train_loss = 0.0
    count = 0.0
    train_pred = []
    train_labels = []

    attack_train_pred = []
    attack_train_labels = []

    all_l2_loss = 0
    all_chamfer_loss = 0
    all_Hausdorff_loss=0

    for _,(points,labels,conduct_poison,U,neighborhood, center, idx_group, kappa_group) in tqdm(enumerate(trainDataLoader),total=len(trainDataLoader)):
        points,labels = points.cuda(),labels.cuda()
        U, neighborhood, center, idx_group, kappa_group = U.cuda(),neighborhood.cuda(), center.cuda(), idx_group.cuda(), kappa_group.cuda()
        GFT_noise = GFT_noise.cuda()
        B,N,_ = points.shape

        classifier.eval()
        trigger_optimizer.zero_grad()
        GFT_noise.requires_grad_(True)
        GFT_noise.retain_grad()
        target_labels = torch.ones_like(labels) * args.target_label

        nn_kappa_std = knn_gather(kappa_group.unsqueeze(2), idx_group).squeeze(3)
        nn_kappa_std = nn_kappa_std.mean(2)
        vales,indices = torch.topk(nn_kappa_std, k=args.topk, dim=1, largest=True)
        training_points_bd,selected_group,selected_group_bd = create_bd(points.detach().clone(), U, neighborhood, center, idx_group, args.topk, GFT_noise,indices)

        logits_1 = classifier(training_points_bd.permute(0,2,1).contiguous())
        attack_loss = criterion_cls(logits_1, target_labels.long())
        attack_preds = logits_1.data.max(1)[1]

        attack_train_pred.append(attack_preds.detach().cpu().numpy())
        attack_train_labels.append(target_labels.cpu().numpy())

        l2_loss = torch.norm(points-training_points_bd,p=2,dim=(1,2)).mean()
        all_l2_loss += l2_loss.item()*B

        chamfer_loss = chamfer_distance(points, training_points_bd, batch_reduction="mean", point_reduction="mean")[0]
        all_chamfer_loss += chamfer_loss.item()*B

        Hausdorff_loss = hausdorff_loss_py(points, training_points_bd)

        all_Hausdorff_loss += Hausdorff_loss.item()*B

        all_loss = args.poison_weight_attack * attack_loss + args.L2_weight*l2_loss +args.chamfer_weight*chamfer_loss + args.Hausdorff_weight*Hausdorff_loss
        all_loss.backward()
        trigger_optimizer.step()

        poison_index = np.where(conduct_poison==True)
        if poison_index[0].size != 0:
            training_points_partbd,selected_group,selected_group_bd = create_bd(points[poison_index].detach().clone(), U[poison_index],neighborhood[poison_index], center[poison_index], idx_group[poison_index], args.topk, GFT_noise,indices[poison_index])
            points[poison_index] = training_points_partbd
            labels[poison_index] = args.target_label
        classifier.train()
        optimizer.zero_grad()

        logits_2 = classifier(points.permute(0,2,1).contiguous())
        benign_loss = criterion_cls(logits_2, labels.long())

        loss = benign_loss
        loss.backward()

        optimizer.step()

        preds = logits_2.max(1)[1]
        count += B
        train_loss += loss.item() * B
        train_labels.append(labels.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    train_labels = np.concatenate(train_labels)
    train_pred = np.concatenate(train_pred)
    attack_train_labels = np.concatenate(attack_train_labels)
    attack_train_pred = np.concatenate(attack_train_pred)

    outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (global_epoch,
                                                                             train_loss * 1.0 / count,
                                                                             metrics.accuracy_score(
                                                                                 train_labels, train_pred),
                                                                             metrics.balanced_accuracy_score(
                                                                                 train_labels, train_pred))
    log_string(outstr)
    log_string('indivadual l2_loss:{}'.format(all_l2_loss / len(train_labels)))
    log_string('indivadual chamfer_loss:{}'.format(all_chamfer_loss / len(train_labels)))
    log_string('indivadual Hausdorff_loss:{}'.format(all_Hausdorff_loss / len(train_labels)))
    log_string('train attack acc:{}'.format(metrics.accuracy_score(attack_train_labels, attack_train_pred)))


def test(classifier,  testDataLoader, GFT_noise,args):
    mean_correct_clean = 0
    mean_correct_bd = 0
    class_acc = np.zeros((args.num_classes, 3))
    classifier.eval()

    all_l2_loss = 0
    all_chamfer_loss = 0
    all_Hausdorff_loss=0
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
        mean_correct_clean+= correct.item()

        mean_correct_bd += correct_bd.item()

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = mean_correct_clean / len(testDataLoader.dataset)
    instance_ASR = mean_correct_bd / len(testDataLoader.dataset)

    test_l2_loss = all_l2_loss / len(testDataLoader.dataset)
    test_chamfer_loss = all_chamfer_loss / len(testDataLoader.dataset)
    test_Hausdorff_loss = all_Hausdorff_loss / len(testDataLoader.dataset)

    return instance_acc, class_acc, instance_ASR, test_l2_loss, test_chamfer_loss, test_Hausdorff_loss


def plot_frequency_spectrum_bar(freq_points, save_path="frequency_spectrum.png"):

    if freq_points.is_cuda:
        freq_points = freq_points.detach().cpu().numpy()
    else:
        freq_points = freq_points.detach().numpy()

    frequencies = freq_points

    x_freq = frequencies[0]
    y_freq = frequencies[1]
    z_freq = frequencies[2]

    freq_indices = np.arange(len(x_freq))

    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(freq_indices - bar_width, x_freq, width=bar_width, color='red', alpha=0.7, label='X-axis')
    plt.bar(freq_indices, y_freq, width=bar_width, color='green', alpha=0.7, label='Y-axis')
    plt.bar(freq_indices + bar_width, z_freq, width=bar_width, color='blue', alpha=0.7, label='Z-axis')

    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Point Cloud (Bar Chart)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(save_path, format="png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f" saving at: {save_path}")

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
        TRAIN_DATASET = ModelNet40_GFT_dat(root=DATA_PATH, args=args, split='train')
        TEST_DATASET = ModelNet40_GFT_dat(root=DATA_PATH, args=args, split='test')
    elif 'shapenet' in args.dataset:
        DATA_PATH = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
        TRAIN_DATASET = ShapeNetDataset_GFT_dat(root=DATA_PATH, args=args, split='trainval', class_choice=None, normal_channel=args.use_normals)
        TEST_DATASET = ShapeNetDataset_GFT_dat(root=DATA_PATH, args=args, split='test', class_choice=None, normal_channel=args.use_normals)
    else:
        raise Exception("no such dataset")
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))

    if args.model == 'DGCNN_cls':
        args.k = 20
        args.emb_dims = 1024
        args.dropout = 0.5
        model = importlib.import_module(args.model)
        classifier = model.get_model(args,args.num_classes).cuda()
    else:
        model = importlib.import_module(args.model)
        classifier = model.get_model(args.num_classes).cuda()

    criterion_cls = get_loss_cls().cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate*100, momentum=0.9,weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-4)

    GFT_noise = np.random.uniform(low=-args.initial_noise_level, high=args.initial_noise_level, size=(args.patch_size, 3))

    GFT_noise = torch.tensor(GFT_noise, dtype=torch.float32).cuda()
    trigger_optimizer = torch.optim.Adam([GFT_noise], lr=args.attack_lr,weight_decay=1e-4)
    scheduler_noise = torch.optim.lr_scheduler.CosineAnnealingLR(trigger_optimizer, args.epoch, eta_min=1e-4)

    max_instance_acc = 0
    max_ASR=0

    tmp_best_instance_acc = 0
    tmp_best_ASR = 0
    tmp_l2_loss, tmp_chamfer_loss, tmp_Hausdorff_loss = 0, 0, 0
    logger.info('Start training...')
    for global_epoch in range(0,args.epoch):
        scheduler.step()
        scheduler_noise.step()

        log_string('Epoch %d/%s:' % (global_epoch + 1, args.epoch))

        train(classifier, trainDataLoader, optimizer, criterion_cls, trigger_optimizer, GFT_noise, args, log_string, global_epoch)

        with torch.no_grad():
            instance_acc, class_acc, instance_ASR,test_l2_loss, test_chamfer_loss, test_Hausdorff_loss = test(classifier,  testDataLoader, GFT_noise,args)
            log_string('Test Instance Accuracy: %f, Instance ASR: %f,  Class Accuracy: %f, l2_loss: %f, chamfer_loss: %f, Hausdorff_loss: %f' % (
            instance_acc, instance_ASR, class_acc, test_l2_loss, test_chamfer_loss, test_Hausdorff_loss))

            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/last_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'instance_ASR': instance_ASR,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'GFT_noise': GFT_noise.detach(),

            }
            torch.save(state, savepath)
            png_file = str(checkpoints_dir) + '/frequency_spectrum_last.png'
            plot_frequency_spectrum_bar(GFT_noise.transpose(0, 1), save_path=png_file)

            if instance_ASR > 0.9 and instance_acc > 0.8 and test_l2_loss < 1.7:

                if instance_acc>=tmp_best_instance_acc-0.005 and instance_ASR>=tmp_best_ASR-0.01:
                    tmp_best_instance_acc = instance_acc
                    tmp_best_ASR = instance_ASR
                    tmp_l2_loss, tmp_chamfer_loss, tmp_Hausdorff_loss = test_l2_loss, test_chamfer_loss, test_Hausdorff_loss
                    savepath_best = str(checkpoints_dir) + '/best_model.pth'
                    state = {
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'instance_ASR': instance_ASR,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'GFT_noise': GFT_noise.detach(),
                    }

                    torch.save(state, savepath_best)
                    png_best_file = str(checkpoints_dir) + '/frequency_spectrum_best.png'
                    plot_frequency_spectrum_bar(GFT_noise.transpose(0, 1), save_path=png_best_file)

            max_instance_acc = max(instance_acc, max_instance_acc)
            max_ASR = max(instance_ASR, max_ASR)
            log_string('Best Instance Accuracy: %f, Instance ASR: %f; tmp Best Instance Acc:%f, tmp Best ASR: %f, tmp_l2_loss: %f, tmp_chamfer_loss: %f, tmp_Hausdorff_loss: %f' % (max_instance_acc, max_ASR,tmp_best_instance_acc,tmp_best_ASR,tmp_l2_loss, tmp_chamfer_loss, tmp_Hausdorff_loss))
        global_epoch += 1

    logger.info('End of training...')


if __name__=='__main__':
    args = parse_args()
    main(args)
