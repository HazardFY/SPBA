import copy
import os
import numpy as np
import warnings
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points,knn_gather
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import visualize_two_point_clouds,plot_frequency_spectrum_bar
from knn_cuda import KNN

from Dataset.WLT import WLT

warnings.filterwarnings('ignore')

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



class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group # 64
        self.group_size = group_size    # 32
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3 (G=64,M=32)
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center,_ = sample_farthest_points(xyz, K=self.num_group)
        center = center.contiguous()
        # knn to get the neighborhood
        _, idx_group = self.knn(xyz, center)  # [B, G, M]
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
    idx = torch.cat([idx0, idx1, idx], dim=0) # (3, b*n*K)

    ones = torch.ones(idx.shape[1], dtype=bool, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n)).to_dense() # (b,n,n)
    A = A | A.transpose(1, 2)
    A = A.float()  
    deg = torch.diag_embed(torch.sum(A, dim=2))  
    laplacian = deg - A
    u, v = torch.linalg.eigh(laplacian) 
    return u.real.to(data_ori), v.real.to(data_ori)



class ModelNet40_GFT_dat(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.npoints = args.num_point
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_classes = args.num_classes
        self.split = split

        self.poison_train_index = np.load(args.index_file)

        if self.num_classes == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat)))) # {'airplane':0,'bathtub':1...}

        shape_ids = {}
        if self.num_classes == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_%dpatchsize_fps_GFT_v2.dat' % (self.num_classes, split, self.npoints, args.patch_size))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_%dpatchsize_GFT_v2.dat' % (self.num_classes, split, self.npoints, args.patch_size))


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
                cls = self.classes[self.datapath[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

               
                point_set = torch.tensor(point_set).unsqueeze(0).cuda()
                if self.uniform:
                    FPS_points, idx = sample_farthest_points(point_set[:, :, :3], K=self.npoints,
                                                             random_start_point=False) 
                    point_set = point_set.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, point_set.size(-1))) 
                else:
                    print('no FPS algorithm')
                    exit(0)
                    point_set = point_set[0:self.npoints, :]

                point_set = point_set.squeeze(0)
                point_set[:,:3] = pc_normalize_tensor(point_set[:,:3]) 
                point_set = point_set.unsqueeze(0)

                ori_kappa_std = get_kappa_std_ori(point_set[:, :, :3], point_set[:, :, 3:], k=10)  # [1,N]
                
                neighborhood, center, idx_group = group_driver(
                    point_set[:,:,:3])  

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

        print('Load processed data from %s...' % self.save_path)
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels,self.list_of_U,self.list_of_neighborhood,self.list_of_center,self.list_of_idx_group,self.list_of_kappa_group = pickle.load(f)

    def __len__(self):
        return len(self.list_of_points)

    def _get_item(self, index):
        conduct_poison = False
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        u,neighborhood,center,idx_group = self.list_of_U[index],self.list_of_neighborhood[index],self.list_of_center[index],self.list_of_idx_group[index]
        kappa_group = self.list_of_kappa_group[index]
        
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        if self.split == 'train':
            if index in self.poison_train_index:
                conduct_poison = True

        return point_set, label[0], conduct_poison, u, neighborhood,center,idx_group,kappa_group

    def __getitem__(self, index):
        return self._get_item(index)

def normalize( input, p=2, dim=3, eps=1e-12): 
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def get_kappa_std_ori( pc, normal, k=10):
    
    # b, _, n = pc.size()
    b,n,_ = pc.shape    # b, n, 3
    
    k_num = k+1
    knn = KNN(k=k_num, transpose_mode=True)
    knn_dists,knn_idx = knn(pc, pc)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = knn_gather(pc, knn_idx)[:, :, 1:, :].contiguous()  # [b,  n ,k, 3]
    
    vectors = nn_pts - pc.unsqueeze(2)  # [b, n, k, 3]
    vectors = normalize(vectors)  # [b, n, k, 3] 
    kappa_ori = torch.abs((vectors * normal.unsqueeze(2)).sum(3)).mean(2)  
    
    nn_kappa = knn_gather(kappa_ori.unsqueeze(2), knn_idx)[:, :, 1:, :].contiguous()  # [b, n ,k,1]
    
    std_kappa = torch.std(nn_kappa.squeeze(3), dim=2)  
    return std_kappa

class ModelNet40_GFT_dat_pure(Dataset):
    def __init__(self, root, args, split='train', process_data=False,conduct_poison=True):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_classes = args.num_classes
        self.split = split
        self.conduct_poison = conduct_poison 
        self.target_label = args.target_label
        
        self.poison_train_index = np.load(args.index_file)

        if self.num_classes == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat)))) # {'airplane':0,'bathtub':1...}

        shape_ids = {}
        if self.num_classes == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_%dpatchsize_fps_GFT_v2.dat' % (self.num_classes, split, self.npoints, args.patch_size))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_%dpatchsize_GFT_v2.dat' % (self.num_classes, split, self.npoints, args.patch_size))


        print('Load processed data from %s...' % self.save_path)
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels,_,_,_,_,_ = pickle.load(f)
        self.add_WLT_trigger = WLT(args)

        if self.conduct_poison==True:
            self.add_trigger()


    def __len__(self):
        return len(self.list_of_points)

    def add_trigger(self):
        tri_list_of_points, tri_list_of_labels = [None] * len(self.list_of_labels), [None] * len(self.list_of_labels)
        for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):

            point_set,label = self.list_of_points[index][:,:3],self.list_of_labels[index]
            if self.split == 'train':
                if index in self.poison_train_index:
                    
                    _,point_set = self.add_WLT_trigger(point_set)
                    label = np.array([self.target_label]).astype(np.int32)
            elif self.split == 'test':
                
                _,point_set = self.add_WLT_trigger(point_set)
                label = np.array([self.target_label]).astype(np.int32)

            tri_list_of_points[index] = point_set
            tri_list_of_labels[index] = label

        self.list_of_points, self.list_of_labels = np.array(tri_list_of_points), np.array(tri_list_of_labels)
    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index] 

        if not self.use_normals:
            point_set = point_set[:, 0:3]
        point_set = pc_normalize(point_set) 

        return point_set.astype(np.float32), label[0]

    def __getitem__(self, index):
        return self._get_item(index)


class ModelNet40_GFT_dat_PointBA(Dataset):

    def __init__(self, root, args, split='train', process_data=False,conduct_poison=True):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_classes = args.num_classes
        self.split = split
        self.conduct_poison = conduct_poison 
        self.target_label = args.target_label
        
        self.poison_train_index = np.load(args.index_file)

        if self.num_classes == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat)))) # {'airplane':0,'bathtub':1...}

        shape_ids = {}
        if self.num_classes == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_%dpatchsize_fps_GFT_v2.dat' % (self.num_classes, split, self.npoints, args.patch_size))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_%dpatchsize_GFT_v2.dat' % (self.num_classes, split, self.npoints, args.patch_size))


        print('Load processed data from %s...' % self.save_path)
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels,_,_,_,_,_ = pickle.load(f)

        if self.conduct_poison==True:
            self.attack_type = args.attack_type
            self.poison_num = args.poison_num 
            self.ball_points = self.fibonacci_sphere(self.poison_num)
            self.angle = args.angle
            self.add_trigger()


    def __len__(self):
        return len(self.list_of_points)

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
            
            if self.split == 'train':
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
    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index] 
  
        if not self.use_normals:
            point_set = point_set[:, 0:3]   
        return point_set.astype(np.float32), label[0]

    def __getitem__(self, index):
        return self._get_item(index)



if __name__ == '__main__':
    import argparse
    def parse_args():
        parser=argparse.ArgumentParser('training')
        parser.add_argument('--num_point',type=int,default=1024,help='Point Number')
        parser.add_argument('--num_classes',type=int,default=40,help='ModelNet10 or ModelNet40')
        parser.add_argument('--use_uniform_sample',action='store_true',default=True,help='use uniform sampiling')
        parser.add_argument('--use_normals',action='store_true',default=True,help='use normals')
        parser.add_argument('--patch_size', type=int, default=64, help='patch size')
        parser.add_argument('--K', type=int, default=10, help='knn kernal')
        return parser.parse_args()


    args = parse_args()

    data = ModelNet40_GFT_dat_1024('../data/modelnet40_normal_resampled/', args=args, split='train')
    data = ModelNet40_GFT_dat_1024('../data/modelnet40_normal_resampled/', args=args, split='test')

   
    DataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

    for point, label, conduct_poison, u, neighborhood,center,idx_group,kappa_group in DataLoader:
        B, patch_num, M,_ = neighborhood.shape

        print(torch.max(point[:, :3]))




        ori_data = point[:, :, :3].cuda()
        ori_normal = point[:, :, 3:].cuda()
        idx_group = idx_group.cuda()
        ori_kappa_std = get_kappa_std_ori(ori_data, ori_normal, k=10)  # [b, n]
        nn_kappa_std = knn_gather(ori_kappa_std.unsqueeze(2), idx_group).squeeze(3)  # [b, n ,1], [b, G, M] -> [b, G, M, 1]->[b, G, M]
        nn_kappa_std = nn_kappa_std.mean(2)  # [b, G]
        vales,indices = torch.topk(nn_kappa_std, k=1, dim=1, largest=True)  # [b], [b]

        print(nn_kappa_std[0]) 

        indice = indices[0].item()




        neighborhood = neighborhood.view(B*patch_num,M,3)
        u= u.view(B*patch_num,M,M)
        freq_points = torch.einsum('bij,bjk->bik', u.transpose(1, 2), neighborhood) # [B*patch_num,M,M],[B*patch_num,M,3],[B*patch_num,M,3]
        selected_group_bd = torch.einsum('bij,bjk->bik', u, freq_points)
        freq_points = freq_points.view(B,patch_num,M,3)
        selected_group_bd = selected_group_bd.view(B,patch_num,M,3)

        visualize_two_point_clouds(point[0, :, :3].detach().cpu().numpy(), selected_group_bd[0, 1, :,:].numpy())
        visualize_two_point_clouds(point[0, :, :3].detach().cpu().numpy(), selected_group_bd[0, indice, :, :].numpy())
        plot_frequency_spectrum_bar(freq_points[0,0,:,:].transpose(0,1))
        plot_frequency_spectrum_bar(freq_points[0, indice, :, :].transpose(0, 1))

