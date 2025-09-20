import copy
import os
import pdb

import numpy as np
import warnings
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points,knn_gather
import torch.nn as nn

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import visualize_two_point_clouds,plot_frequency_spectrum_bar
from knn_cuda import KNN

from Dataset.WLT import WLT

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
def pc_normalize_tensor(pc):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid

    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))

    pc = pc / m
    return pc


class ShapeNetDataset_dat(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=None, split='train', class_choice=None, normal_channel=False):
        self.npoints = args.num_point
        self.num_category = args.num_category
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.save_path = os.path.join(root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))

        if not os.path.exists(self.save_path):
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)


            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cat = self.datapath[index][0]
                label = self.classes[cat]
                label = np.array([label]).astype(np.int32)
                data = np.loadtxt(fn[1]).astype(np.float32)
                point_set = data[:, 0:3]

                point_set = torch.tensor(point_set).unsqueeze(0).cuda()
                FPS_points, idx = sample_farthest_points(point_set, K=self.npoints,
                                                         random_start_point=False)
                point_set = FPS_points.squeeze().cpu().numpy()

                self.list_of_points[index] = point_set
                self.list_of_labels[index] = label
            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)
        return point_set, label[0]

    def __len__(self):
        return len(self.datapath)
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

def get_Laplace_from_pc(data_ori, K):
    data = data_ori.detach().clone()
    b, n, _ = data.shape
    knn = KNN(k=K,transpose_mode=True)
    _,idx = knn(data,data)

    idx0 = torch.arange(0,b,device=data.device).reshape((b,1)).expand(-1,n*K).reshape((1,b*n*K))
    idx1 = torch.arange(0,n,device=data.device).reshape((1,n,1)).expand(b,n,K).reshape((1,b*n*K))
    idx = idx.reshape((1,b*n*K))
    idx = torch.cat([idx0, idx1, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=bool, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n)).to_dense()
    A = A | A.transpose(1, 2)
    A = A.float()
    deg = torch.diag_embed(torch.sum(A, dim=2))
    laplacian = deg - A
    u, v = torch.linalg.eigh(laplacian)
    return u.real.to(data_ori), v.real.to(data_ori)

def normalize( input, p=2, dim=3, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def get_kappa_std_ori( pc, normal, k=10):
    b,n,_ = pc.shape
    k_num = k+1
    knn = KNN(k=k_num, transpose_mode=True)
    knn_dists,knn_idx = knn(pc, pc)
    nn_pts = knn_gather(pc, knn_idx)[:, :, 1:, :].contiguous()
    vectors = nn_pts - pc.unsqueeze(2)
    vectors = normalize(vectors)
    kappa_ori = torch.abs((vectors * normal.unsqueeze(2)).sum(3)).mean(2)
    nn_kappa = knn_gather(kappa_ori.unsqueeze(2), knn_idx)[:, :, 1:, :].contiguous()
    std_kappa = torch.std(nn_kappa.squeeze(3), dim=2)
    return std_kappa

class ShapeNetDataset_GFT_dat(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=None, split='train', class_choice=None, normal_channel=False):
        self.npoints = args.num_point
        self.num_category = args.num_classes
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_normals = args.use_normals
        self.split=split
        self.poison_train_index = np.load("random_numbers_train_ShapeNetPart.npy")

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.save_path = os.path.join(root, 'shapenet%d_%s_%dpts_%dpatchsize_fps_GFT_v2.dat' % (self.num_category, split, self.npoints, args.patch_size))

        if not os.path.exists(self.save_path):
            print('Extracting GFT information...')
            patch_num = args.num_point // args.patch_size
            group_driver = Group(patch_num, args.patch_size)
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)
            self.list_of_U = [None] * len(self.datapath)
            self.list_of_neighborhood = [None] * len(self.datapath)
            self.list_of_center = [None] * len(self.datapath)
            self.list_of_idx_group = [None] * len(self.datapath)
            self.list_of_kappa_group = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cat = self.datapath[index][0]
                cls = self.classes[cat]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1]).astype(np.float32)
                point_set = point_set[:,:6]

                if point_set.shape[0] <= args.num_point:
                    sum_temp = point_set.shape[0]
                    print('%s, Point number is %d less than %d, padding...' % (fn,sum_temp,args.num_point))
                    extra_indices = np.random.choice(sum_temp, args.num_point-sum_temp, replace=False)
                    extra_points = point_set[extra_indices]

                    point_set = np.concatenate([point_set, extra_points], axis=0)
                    point_set = torch.tensor(point_set).unsqueeze(0).cuda()

                else:
                    point_set = torch.tensor(point_set).unsqueeze(0).cuda()
                    _, idx = sample_farthest_points(point_set[:, :, :3], K=self.npoints,random_start_point=False)
                    point_set = point_set.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, point_set.size(-1)))

                point_set = point_set.squeeze(0)
                point_set[:, :3] = pc_normalize_tensor(point_set[:, :3])
                point_set = point_set.unsqueeze(0)

                ori_kappa_std = get_kappa_std_ori(point_set[:, :, :3], point_set[:, :, 3:], k=10)
                neighborhood, center, idx_group = group_driver(
                    point_set[:, :,:3])

                Evs, u = get_Laplace_from_pc(neighborhood.squeeze(0), args.K)

                point_set = point_set.squeeze(0).cpu().numpy()
                u = u.cpu().numpy()
                neighborhood = neighborhood.squeeze(0).cpu().numpy()
                center = center.squeeze(0).cpu().numpy()
                idx_group = idx_group.squeeze(0).cpu().numpy()
                ori_kappa_std = ori_kappa_std.squeeze(0).cpu().numpy()

                self.list_of_points[index] = point_set
                self.list_of_labels[index] = cls
                self.list_of_U[index] = u
                self.list_of_neighborhood[index] = neighborhood
                self.list_of_center[index] = center
                self.list_of_idx_group[index] = idx_group
                self.list_of_kappa_group[index] = ori_kappa_std

            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels,self.list_of_U,self.list_of_neighborhood,self.list_of_center,self.list_of_idx_group,self.list_of_kappa_group], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels,self.list_of_U,self.list_of_neighborhood,self.list_of_center,self.list_of_idx_group,self.list_of_kappa_group = pickle.load(f)


    def __getitem__(self, index):
        conduct_poison = False
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        u,neighborhood,center,idx_group = self.list_of_U[index],self.list_of_neighborhood[index],self.list_of_center[index],self.list_of_idx_group[index]
        kappa_group = self.list_of_kappa_group[index]
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        if self.split == 'trainval':
            if index in self.poison_train_index:
                conduct_poison = True

        return point_set, label[0], conduct_poison, u, neighborhood,center,idx_group,kappa_group

    def __len__(self):
        return len(self.datapath)

class ShapeNetDataset_GFT_dat_pure(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=None, split='train', class_choice=None, normal_channel=False,conduct_poison=True):
        self.npoints = args.num_point
        self.num_category = args.num_classes
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_normals = args.use_normals
        self.split=split

        self.conduct_poison = conduct_poison
        self.target_label = args.target_label
        self.poison_train_index = np.load("random_numbers_train_ShapeNetPart.npy")

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.save_path = os.path.join(root, 'shapenet%d_%s_%dpts_%dpatchsize_fps_GFT_v2.dat' % (self.num_category, split, self.npoints, args.patch_size))


        print('Load processed data from %s...' % self.save_path)
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels,_,_,_,_,_ = pickle.load(f)
        self.add_WLT_trigger = WLT(args)
        if self.conduct_poison==True:
            self.add_trigger()
    def add_trigger(self):
        tri_list_of_points, tri_list_of_labels = [None] * len(self.list_of_labels), [None] * len(self.list_of_labels)
        for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
            point_set,label = self.list_of_points[index][:,:3],self.list_of_labels[index]
            if self.split == 'trainval':
                if index in self.poison_train_index:
                    _,point_set = self.add_WLT_trigger(point_set)
                    label = np.array([self.target_label]).astype(np.int32)
            elif self.split == 'test':
                _,point_set = self.add_WLT_trigger(point_set)
                label = np.array([self.target_label]).astype(np.int32)

            tri_list_of_points[index] = point_set
            tri_list_of_labels[index] = label

        self.list_of_points, self.list_of_labels = np.array(tri_list_of_points), np.array(tri_list_of_labels)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        if not self.use_normals:
            point_set = point_set[:, 0:3]
        point_set = pc_normalize(point_set)
        return point_set.astype(np.float32), label[0]

    def __len__(self):
        return len(self.datapath)


class ShapeNetDataset_GFT_dat_PointBA(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=None, split='trainval',
                 class_choice=None, normal_channel=False, conduct_poison=True):
        self.npoints = args.num_point
        self.num_category = args.num_classes
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_normals = args.use_normals
        self.split = split

        self.conduct_poison = conduct_poison
        self.target_label = args.target_label
        self.poison_train_index = np.load("random_numbers_train_ShapeNetPart.npy")

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.save_path = os.path.join(root, 'shapenet%d_%s_%dpts_%dpatchsize_fps_GFT_v2.dat' % (
        self.num_category, split, self.npoints, args.patch_size))

        print('Load processed data from %s...' % self.save_path)
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels, _, _, _, _, _ = pickle.load(f)

        if self.conduct_poison==True:
            self.attack_type = args.attack_type
            self.poison_num = args.poison_num
            self.ball_points = self.fibonacci_sphere(self.poison_num)
            self.angle = args.angle
            self.add_trigger()

    def fibonacci_sphere(self,num_points, radius=0.05, center=(0.5, 0.5, 0.5)):
        indices = np.arange(0, num_points, dtype=float) + 0.5
        phi = np.pi * (1 + 5 ** 0.5) * indices
        costheta = 1 - 2 * indices / num_points
        theta = np.arccos(costheta)

        x = radius * np.sin(theta) * np.cos(phi) + center[0]
        y = radius * np.sin(theta) * np.sin(phi) + center[1]
        z = radius * np.cos(theta) + center[2]

        return np.vstack((x, y, z)).T
    def rotate_point_cloud_around_z(self, points, angle_degrees):
        angle_radians = np.radians(angle_degrees)

        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])

        rotated_points = np.dot(points, rotation_matrix.T)

        return rotated_points

    def add_trigger(self):
        tri_list_of_points, tri_list_of_labels = [None] * len(self.list_of_labels), [None] * len(self.list_of_labels)
        for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
            point_set,label = self.list_of_points[index][:,:3],self.list_of_labels[index]
            point_set_ori = copy.copy(point_set)
            if self.split == 'trainval':
                if index in self.poison_train_index:
                    assert self.attack_type in ['BAI','BAO']
                    if self.attack_type == 'BAI':
                        point_set[-self.poison_num:,] = self.ball_points
                    elif self.attack_type == 'BAO':
                        point_set = self.rotate_point_cloud_around_z(point_set, self.angle)
                    label = np.array([self.target_label]).astype(np.int32)
            elif self.split == 'test':
                assert self.attack_type in ['BAI', 'BAO']
                if self.attack_type == 'BAI':
                    point_set[-self.poison_num:, ] = self.ball_points
                elif self.attack_type == 'BAO':
                    point_set = self.rotate_point_cloud_around_z(point_set, self.angle)

                label = np.array([self.target_label]).astype(np.int32)
            tri_list_of_points[index] = point_set
            tri_list_of_labels[index] = label

        self.list_of_points, self.list_of_labels = np.array(tri_list_of_points), np.array(tri_list_of_labels)


    def __getitem__(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set.astype(np.float32), label[0]

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser=argparse.ArgumentParser('training')
        parser.add_argument('--num_point',type=int,default=1024,help='Point Number')
        parser.add_argument('--num_classes',type=int,default=16,help='ModelNet10 or ModelNet40')
        parser.add_argument('--use_uniform_sample',action='store_true',default=True,help='use uniform sampiling')
        parser.add_argument('--use_normals',action='store_true',default=False,help='use normals')
        parser.add_argument('--patch_size', type=int, default=32, help='patch size')
        parser.add_argument('--K', type=int, default=10, help='knn kernal')
        return parser.parse_args()

    args=parse_args()
    data = ShapeNetDataset_GFT_dat('../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=args,
                                   split='trainval', class_choice=None, normal_channel=args.use_normals)
    print(len(data))
    DataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False,num_workers=4)
    for point, label, conduct_poison, u, neighborhood, center, idx_group, kappa_group in DataLoader:
        B, patch_num, M,_ = neighborhood.shape

        print(torch.max(point[:, :3]))
        nn_kappa_std = knn_gather(kappa_group.unsqueeze(2), idx_group).squeeze(3)
        nn_kappa_std = nn_kappa_std.mean(2)
        values,indices = torch.topk(nn_kappa_std, k=1, dim=1, largest=True)

        print(nn_kappa_std[0])
        indice = indices[0].item()
        value = values[0].item()
        print('The patch with the highest curvature is: %d, value: %f' %(indice,value))

        neighborhood = neighborhood.view(B*patch_num,M,3)
        u= u.view(B*patch_num,M,M)
        freq_points = torch.einsum('bij,bjk->bik', u.transpose(1, 2), neighborhood)
        selected_group_bd = torch.einsum('bij,bjk->bik', u, freq_points)
        freq_points = freq_points.view(B,patch_num,M,3)
        selected_group_bd = selected_group_bd.view(B,patch_num,M,3)

        visualize_two_point_clouds(point[0, :, :3].detach().cpu().numpy(), selected_group_bd[0, 0, :,:].numpy())
        visualize_two_point_clouds(point[0, :, :3].detach().cpu().numpy(), selected_group_bd[0, indice, :, :].numpy())
        plot_frequency_spectrum_bar(freq_points[0,0,:,:].transpose(0,1))
        plot_frequency_spectrum_bar(freq_points[0, indice, :, :].transpose(0, 1))