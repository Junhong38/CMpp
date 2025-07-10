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

import glob
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")

class DatasetFantasticBreaks(Dataset):
    def __init__(self, datapath, n_pts, visualize=False):
        self.datapath = datapath
        self.n_pts = n_pts
        self.min_n_pts = 256

        # Read fracture path list
        self.filepaths, self.assmpaths = [], []
        fb_cls_dir = sorted(glob.glob('../../data/FantasticBreaks/*'))
        for cls_dir in fb_cls_dir:
            fb_sample_dir = sorted(glob.glob('%s/*' % cls_dir))
            for sample_dir in fb_sample_dir:
                self.filepaths.append(['%s/model_r_0.ply' % sample_dir, '%s/model_b_0.ply' % sample_dir])
                self.assmpaths.append('%s/model_c.ply' % sample_dir)

        self.overlap_radius = 0.018
        self.visualize = visualize

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

        return {'0-1':permut_relative_transform['0-1']}

    def __getitem__(self, idx):
        # Fix randomness
        np.random.seed(idx)

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
                'filepath': os.path.dirname(self.filepaths[idx][0]),
                'obj_class': self.filepaths[idx][0].split('/')[3],

                'pcd_t': pcd_t,
                'pcd': pcd,
                'n_frac': 2,

                'gt_trans': gt_trans,
                'gt_rotat': gt_rotat,
                'gt_rotat_inv': [R.T for R in gt_rotat],
                'gt_trans_inv': [-t for t in gt_trans],
                'relative_trsfm': gt_relative_trsfm,

                'gt_correspondence': matching_inds,
                }

        return batch

    def read_obj_data(self, idx):
        obj_paths = self.filepaths[idx]
        assm_path = self.assmpaths[idx]

        mesh_all = [trimesh.load_mesh(obj_path) for obj_path in obj_paths]
        # mesh_all = [mesh_.apply_transform(transform) for mesh_ in mesh_all]
        mesh_areas = [mesh_.area for mesh_ in mesh_all]
        total_area = sum(mesh_areas)


        pcd_all = []
        for mesh in mesh_all:
            n_pts = int(self.n_pts * mesh.area / total_area)
            sampled_pts = torch.tensor(trimesh.sample.sample_surface_even(mesh, n_pts, seed=idx)[0]).float()

            if sampled_pts.size(0) < self.min_n_pts:
                extra_pts, _ = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0), seed=idx)
                sampled_pts = torch.cat([sampled_pts, torch.tensor(extra_pts).float()], dim=0)
            
            pcd_all.append(sampled_pts)

        # Normalize the sampled points
        combined_pcd = torch.cat(pcd_all, dim=0)
        centroid = torch.mean(combined_pcd, dim=0)
        combined_pcd -= centroid
        furthest_distance = torch.max(torch.norm(combined_pcd, dim=1))
        combined_pcd /= furthest_distance
        normalized_pcd_all = []
        start_idx = 0
        for pcd in pcd_all:
            end_idx = start_idx + pcd.shape[0]
            normalized_pcd_all.append(combined_pcd[start_idx:end_idx])
            start_idx = end_idx
        
        pcd_all = normalized_pcd_all
        
        return mesh_all, pcd_all