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
from data.utils import to_o3d_pcd, to_array, get_correspondences, check_mesh_intersection, are_meshes_connected

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")

class DatasetBreakingBad(Dataset):
    def __init__(self, datapath, data_category, sub_category, min_part, max_part, n_pts, split, scale, visualize=False):
        self.datapath = datapath
        self.data_category = data_category # ['everyday', 'artifact']
        self.split = split
        self.sub_category = sub_category
        self.n_pts = n_pts
        self.visualize = visualize

        self.min_n_pts = 256
        self.min_part = min_part
        self.max_part = max_part
        self.mpa = True if self.max_part > 2 else False
        self.anchor_idx = 0

        if self.mpa and self.split in ['train', 'val']:
            filepaths = join('./data/data_list', f"mpa_{data_category}_{split}.txt")
        else:
            if self.split == 'test': split = 'val'
            # Read fracture path list
            if scale == 'overfitting':
                filepaths = join('./data/data_list', f"{data_category}_{split}_one.txt")
            elif scale == 'full':
                filepaths = join('./data/data_list', f"{data_category}_{split}.txt")
            elif scale == 'ransac':
                filepaths = join('./data/data_list', f"{data_category}_test.txt")
                # filepaths = join('./data/data_list', f"{data_category}_{split}.txt")
                # filepaths = join('./data/data_list', f"ransac_analysis.txt")
            else:
                filepaths = join('./data/data_list', f"{data_category}_{split}_small.txt")
        print(filepaths)
        
        with open(filepaths, 'r') as f:
            self.filepaths = [x.strip() for x in f.readlines() if x.strip()]

        self.filepaths = [x for x in self.filepaths if self.min_part <= int(x.split()[0]) <= self.max_part]
        if self.sub_category != 'all': self.filepaths = [x for x in self.filepaths if x.split()[1].split('/')[1] == self.sub_category]

        if self.mpa and self.split in ['train', 'val']:
            self.frac0 = [x.split()[2] for x in self.filepaths]
            self.frac1 = [x.split()[3] for x in self.filepaths]

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
            mesh_t[idx].vertices -= trans.numpy()
        return pcd_t, mesh_t, gt_trans

    def _rotate(self, mesh, pcd):
        gt_rotat = [torch.tensor(R.random().as_matrix(), dtype=torch.float) for _ in pcd]
        pcd_t, mesh_t = [], [m.copy() for m in mesh]
        for idx, rotat in enumerate(gt_rotat):
            pcd_t.append(torch.einsum('x y, n y -> n x', rotat, pcd[idx]))
            mesh_t[idx].vertices = torch.einsum('x y, n y -> n x', rotat, torch.tensor(mesh_t[idx].vertices).float()).numpy()
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
        if self.split in ['val', 'test']: np.random.seed(idx)

        # Read mesh, point cloud of a fractured object
        logger = logging.getLogger("trimesh")
        logger.setLevel(logging.ERROR)
        mesh, pcd, face = self.read_obj_data(idx)

        # Get ground-truth correspondences
        # matching_inds = get_correspondences(to_o3d_pcd(pcd[0]), to_o3d_pcd(pcd[1]), self.overlap_radius)

        pair_indices = list(itertools.permutations([i for i in range(self.n_frac[idx])], 2))
        cls_gt_matrix = np.full((self.n_frac[idx], self.n_frac[idx]), False, dtype=bool)
        matching_inds_matrix = {}
        gt_mating_surface = {} # dict of meshes
        for pair_idx in pair_indices:
            pair_idx0, pair_idx1 = pair_idx
            # is_touching = check_mesh_intersection(mesh[pair_idx0], mesh[pair_idx1])
            # cls_gt_matrix[pair_idx0][pair_idx1] = is_touching
            # matching_inds_matrix[f'{pair_idx0}-{pair_idx1}'] = get_correspondences(to_o3d_pcd(pcd[pair_idx0]), to_o3d_pcd(pcd[pair_idx1]), self.overlap_radius)
            connected, shared_faces_0, shared_faces_1 = are_meshes_connected(mesh[pair_idx0], mesh[pair_idx1])
            cls_gt_matrix[pair_idx0][pair_idx1] = len(connected) > 0
            # breakpoint()
            # shared_faces_0 = shared_faces_0[face[pair_idx0]]
            # shared_faces_1 = shared_faces_1[face[pair_idx1]]
            # shared_faces_0 = shared_faces_0[mesh[pair_idx0].faces]
            # shared_faces_1 = shared_faces_1[mesh[pair_idx1].faces]
            gt_mating_surface[f'{pair_idx0}-{pair_idx1}'] = [shared_faces_0, shared_faces_1]

        # Sampling only on mating surface Designed for 2-part           
        if self.n_pts != 5001:
            gt_mating_surface_idx = [np.where(gt_mating_surface['0-1'][0])[0], np.where(gt_mating_surface['0-1'][1])[0]]
            mating_mesh = [mesh[0].submesh([np.where(gt_mating_surface['0-1'][0])[0]], append=True), mesh[1].submesh([np.where(gt_mating_surface['0-1'][1])[0]], append=True)]
            for i, mesh_roi in enumerate(mating_mesh):
                # n_pts = pcd[i].shape[0]
                roi_n_pts = mesh_roi.faces.shape[0]
                # remove_n_pts = gt_mating_surface['0-1'][i][face[i]].sum()
                # breakpoint()
                # mesh_rest = mesh[i].difference(mesh_roi)
                if self.split in ['val', 'train']:
                    # sampled_pts_rest, face_idx_rest = trimesh.sample.sample_surface_even(mesh_rest, n_pts - remove_n_pts)
                    sampled_pts_roi, face_idx_roi = trimesh.sample.sample_surface_even(mesh_roi, roi_n_pts)
                else:
                    # sampled_pts_rest, face_idx_rest = trimesh.sample.sample_surface_even(mesh_rest, n_pts - remove_n_pts)
                    sampled_pts_roi, face_idx_roi = trimesh.sample.sample_surface_even(mesh_roi, roi_n_pts)
                
                sampled_pts_roi = torch.tensor(sampled_pts_roi).float()
                pcd[i] = torch.cat([pcd[i], sampled_pts_roi], dim=0)
                face[i] = np.concatenate([face[i], gt_mating_surface_idx[i][face_idx_roi]], axis=0)
                # breakpoint()
        
        for pair_idx in pair_indices:
            pair_idx0, pair_idx1 = pair_idx
            # Adaptive distance thresholding for GT mating surface
            dists_0 = torch.cdist(pcd[0], pcd[0], p=2) + torch.eye(pcd[0].size(0)) * 999
            dists_1 = torch.cdist(pcd[1], pcd[1], p=2) + torch.eye(pcd[1].size(0)) * 999
            self.overlap_radius = min(dists_0.min(dim=0)[0].mean(), dists_1.min(dim=0)[0].mean())

            matching_inds_matrix[f'{pair_idx0}-{pair_idx1}'] = get_correspondences(to_o3d_pcd(pcd[pair_idx0]), to_o3d_pcd(pcd[pair_idx1]), self.overlap_radius)
            

        # Apply random transformation to sampled points
        pcd_t, mesh_t, gt_trans = self._translate(mesh, pcd)
        pcd_t, mesh_t, gt_rotat = self._rotate(mesh_t, pcd_t)
        gt_relative_trsfm = self._compute_relative_transform(gt_trans, gt_rotat)
        
        ## gt surface normal
        gt_normals = [
            mesh[i].face_normals[face_i] for i, face_i in enumerate(face)
        ]
        rotated_gt_normals = []
        for i, rotat in enumerate(gt_rotat):
            # rotated_gt_normals.append(torch.einsum('x y, n y -> n x', rotat, gt_normals[idx]))
            rotated_gt_normals.append((rotat @ gt_normals[i].T).T)
        
        batch = {
                'eval_idx': idx,
                'filepath': self.filepaths[idx],
                'obj_class': self.filepaths[idx].split('/')[1],

                'mesh': [torch.tensor(_mesh.vertices).float() for _mesh in mesh],
                'mesh_t': [torch.tensor(_mesh.vertices).float() for _mesh in mesh_t],
                'pcd_t': pcd_t,
                'pcd': pcd,
                'n_frac': self.n_frac[idx],
                'anchor_idx': self.anchor_idx,

                'gt_trans': gt_trans,
                'gt_rotat': gt_rotat,
                'gt_rotat_inv': [R.T for R in gt_rotat],
                'gt_trans_inv': [-t for t in gt_trans],
                'relative_trsfm': gt_relative_trsfm,

                'gt_correspondence': [matching_inds_matrix],
                'cls_gt': cls_gt_matrix,

                'gt_normals': rotated_gt_normals,
                'gt_mating_surface': gt_mating_surface,
                }

        return batch

    def read_obj_data(self, idx):
        if self.split in ['val', 'test']: random.seed(idx)
        # np.seterr(divide='ignore', invalid='ignore')
        
        filepath = self.filepaths[idx]
        n_frac = self.n_frac[idx]

        # Load N-part meshes and calculate each area
        base_path = join(self.datapath, filepath)
        if self.mpa and self.split in ['train', 'val']:
            obj_paths = [join(base_path, x) for x in [self.frac0[idx], self.frac1[idx]]]
        else: obj_paths = [join(base_path, x) for x in os.listdir(base_path)]

        meshes = [trimesh.load_mesh(x) for x in obj_paths]
        mesh_areas = [mesh_.area for mesh_ in meshes]

        # Set anchor fracture and sum all of areas
        self.anchor_idx, total_area = mesh_areas.index(max(mesh_areas)), sum(mesh_areas)

        # Sample N-part point clouds from meshes
        pcds = []
        faces = []
        for mesh in meshes:
            n_pts = int(self.n_pts * mesh.area / total_area)
            if self.split in ['val', 'test']: sampled_pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_pts, seed=idx)
            else: sampled_pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_pts)

            sampled_pts = torch.tensor(sampled_pts).float()

            if sampled_pts.size(0) < self.min_n_pts:
                if self.split in ['val', 'test']: extra_pts, extra_face_idx = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0), seed=idx)
                else: extra_pts, extra_face_idx = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0))
                sampled_pts = torch.cat([sampled_pts, torch.tensor(extra_pts).float()], dim=0)
                face_idx = np.concatenate([face_idx, extra_face_idx], axis=0)
            
            pcds.append(sampled_pts)
            faces.append(face_idx)

        # if self.mpa and len(pcds)>2:
        #     # Randomly select one point cloud
        #     src_idx = random.randint(0, n_frac-1)
        #     src_pcd, src_mesh = pcds[src_idx], meshes[src_idx]
            
        #     # Set target to be mostly mated with source pcd
        #     other_pcd = pcds[:src_idx] + pcds[src_idx+1:]
        #     other_mesh = meshes[:src_idx] + meshes[src_idx+1:]
        #     # n_fracture_points = [self._extract_fracture_points(src_pcd, x).size(0) for x in other_pcd]
        #     n_fracture_points = [get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(x), self.overlap_radius).size(0) for x in other_pcd]
        #     trg_idx = n_fracture_points.index(max(n_fracture_points))

        #     # Return 2-part pc, mesh
        #     pcds = [src_pcd, other_pcd[trg_idx]]
        #     meshes = [src_mesh, other_mesh[trg_idx]]
        
        # Augment train dataset
        if self.split == 'train' and random.random() > 0.5:
            meshes.reverse()
            pcds.reverse()
        
        return meshes, pcds, faces