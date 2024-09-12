import os
from os.path import join
import itertools
import logging
import random

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import trimesh
import torch
from torch.utils.data import Dataset
from einops import rearrange, repeat
import open3d as o3d
from data.utils import to_o3d_pcd, to_array, get_correspondences

def save_pc(filename:str, pcd_tensors:list):
    pcds = []
    for tensor_ in pcd_tensors:
        if tensor_.size()[0] == 1:
            tensor_ = tensor_.squeeze(0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tensor_.cpu().numpy())
        pcd.paint_uniform_color([random.uniform(0, 1) for _ in range(3)])
        pcds.append(pcd)
    combined_cloud = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_cloud += pcd
    o3d.io.write_point_cloud(filename, combined_cloud)

class DatasetBreakingBad(Dataset):
    def __init__(self, datapath, data_category, sub_category, n_pts, split, scale, visualize=False):
        self.datapath = datapath
        self.data_category = data_category # ['everyday', 'artifact']
        self.split = split
        self.sub_category = sub_category
        self.n_pts = n_pts
        self.visualize = visualize

        self.min_n_pts = 256
        self.min_part = 2
        self.max_part = 2

        self.anchor_idx = 0

        # Read fracture path list
        if scale == 'overfitting':
            filepaths = join('./data/data_list', f"{data_category}_{split}_one.txt")
        elif scale == 'full':
            filepaths = join('./data/data_list', f"{data_category}_{split}.txt")
        else:
            filepaths = join('./data/data_list', f"{data_category}_{split}_small.txt")
        print(filepaths)
        
        with open(filepaths, 'r') as f:
            self.filepaths = [x.strip() for x in f.readlines() if x.strip()]

        self.filepaths = [x for x in self.filepaths if self.min_part <= int(x.split()[0]) <= self.max_part]
        if self.sub_category != 'all': self.filepaths = [x for x in self.filepaths if x.split()[1].split('/')[1] == self.sub_category]

        self.n_frac = [int(x.split()[0]) for x in self.filepaths]
        self.filepaths = [x.split()[1] for x in self.filepaths]

        self.overlap_radius = 0.018
        
    def __len__(self):
        return len(self.filepaths)

    def _translate(self, mesh, pcd):
        gt_trans = [p.mean(dim=0) for p in pcd]
        pcd_t, mesh_t = [], [m.copy() for m in mesh]
        for idx, trans in enumerate(gt_trans):
            pcd_t.append(pcd[idx] - trans)
            if self.visualize: mesh_t[idx].vertices -= trans.numpy()
        return pcd_t, mesh_t, gt_trans

    def _rotate(self, mesh, pcd):
        gt_rotat = [torch.tensor(R.random().as_matrix(), dtype=torch.float) for _ in pcd]
        # gt_rotat = [torch.tensor([[0.26726124, -0.57735027,  0.77151675],
        #           [0.53452248, -0.57735027, -0.6172134],
        #           [0.80178373,  0.57735027,  0.15430335]], dtype=torch.float32) for _ in pcd]
        pcd_t, mesh_t = [], [m.copy() for m in mesh]
        for idx, rotat in enumerate(gt_rotat):
            pcd_t.append(torch.einsum('x y, n y -> n x', rotat, pcd[idx]))
            if self.visualize: mesh_t[idx].vertices = torch.einsum('x y, n y -> n x', rotat, torch.tensor(mesh_t[idx].vertices).float()).numpy()
        return pcd_t, mesh_t, gt_rotat

    def _compute_relative_transform(self, trans, rotat):
        permut_relative_transform = {}
        for src_idx, trg_idx in itertools.permutations(range(len(trans)), 2):
            # Compute relative rotation and translation
            trans0, trans1 = trans[src_idx], trans[trg_idx]
            rotat0, rotat1 = rotat[src_idx], rotat[trg_idx]
            relative_rotat = rotat1 @ rotat0.T
            relative_trans = - (rotat1 @ (trans0 - trans1))

            # Save relative transformation between each pairs
            key = f"{src_idx}-{trg_idx}"
            permut_relative_transform[key] = relative_rotat, relative_trans

        if self.split in ['train', 'val']: return {'0-1':permut_relative_transform['0-1']}
        else: return permut_relative_transform

    def __getitem__(self, idx):
        # Fix randomness
        if self.split in ['train', 'val', 'test']: np.random.seed(idx)

        # Read mesh, point cloud of a fractured object
        logger = logging.getLogger("trimesh")
        logger.setLevel(logging.ERROR)
        mesh, pcd = self.read_obj_data(idx)
        
        # Get ground-truth correspondences
        matching_inds = get_correspondences(to_o3d_pcd(pcd[0]), to_o3d_pcd(pcd[1]), self.overlap_radius)
        
        # Apply random transformation to sampled points
        pcd_t, mesh_t, gt_trans = self._translate(mesh, pcd)
        pcd_t, mesh_t, gt_rotat = self._rotate(mesh_t, pcd_t)
        gt_relative_trsfm = self._compute_relative_transform(gt_trans, gt_rotat)

        batch = {
                'eval_idx': idx,
                'filepath': self.filepaths[idx],
                'obj_class': self.filepaths[idx].split('/')[1],

                'pcd_t': pcd_t,
                'pcd': pcd,
                'n_frac': self.n_frac[idx],
                'anchor_idx': self.anchor_idx,

                'gt_trans': gt_trans,
                'gt_rotat': gt_rotat,
                'gt_rotat_inv': [R.T for R in gt_rotat],
                'gt_trans_inv': [-t for t in gt_trans],
                'relative_trsfm': gt_relative_trsfm,

                'gt_correspondence': matching_inds,
                }

        return batch

    def read_obj_data(self, idx):
        if self.split in ['val', 'test']: random.seed(idx)
        # np.seterr(divide='ignore', invalid='ignore')
        
        filepath = self.filepaths[idx]
        n_frac = self.n_frac[idx]

        # Load N-part meshes and calculate each area
        base_path = join(self.datapath, filepath)
        obj_paths = [join(base_path, x) for x in os.listdir(base_path)]
        meshes = [trimesh.load_mesh(x) for x in obj_paths]
        mesh_areas = [mesh_.area for mesh_ in meshes]

        # Set anchor fracture and sum all of areas
        self.anchor_idx, total_area = mesh_areas.index(max(mesh_areas)), sum(mesh_areas)

        # Sample N-part point clouds from meshes
        pcds = []
        for mesh in meshes:
            n_pts = int(self.n_pts * mesh.area / total_area)
            if self.split in ['val', 'test']: sampled_pts = torch.tensor(trimesh.sample.sample_surface_even(mesh, n_pts, seed=idx)[0]).float()
            else: sampled_pts = torch.tensor(trimesh.sample.sample_surface_even(mesh, n_pts)[0]).float()

            if sampled_pts.size(0) < self.min_n_pts:
                if self.split in ['val', 'test']: extra_pts, _ = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0), seed=idx)
                else: extra_pts, _ = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0))
                sampled_pts = torch.cat([sampled_pts, torch.tensor(extra_pts).float()], dim=0)
            
            pcds.append(sampled_pts)

        # Augment train dataset
        if self.split == 'train' and random.random() > 0.5:
            meshes.reverse()
            pcds.reverse()
            
        return meshes, pcds