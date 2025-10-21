import os
from os.path import join
import itertools
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh

import torch
from torch.utils.data import Dataset

from data.utils import to_o3d_pcd, get_correspondences


class DatasetBreakingBad(Dataset):
    def __init__(self, datapath, data_category, sub_category, min_part, max_part, n_pts, split, scale, multiplicity, visualize=False):
        """Dataset for Breaking Bad

        Args:
            datapath (str): path to the dataset
            data_category (str): ['everyday', 'artifact', 'synthetic'], candidates are fixed by argparse
            sub_category (str): ['all', 'xxx', ...]
            min_part (int): minimum number of parts
            max_part (int): maximum number of parts
            n_pts (int): number of points to sample
            split (str): ['train', 'val', 'test']
            scale (str): ['full', 'small', 'overfitting', 'tiny'], candidates are fixed by argparse
            multiplicity (int): multiplicity of the dataset
            visualize (bool, optional): whether to visualize the dataset. Defaults to False.
        """
        # Assertion
        assert split in ['train', 'val', 'test'], f"split must be in ['train', 'val', 'test'], but got {split}"

        self.datapath = datapath
        self.data_category = data_category 
        self.sub_category = sub_category

        self.min_n_pts = 256
        self.min_part = min_part
        self.max_part = max_part
        self.n_pts = n_pts

        self.split = split
        
        self.multiplicity = multiplicity if split == 'train' else 1
        self.visualize = visualize

        self.mpa = True if self.max_part > 2 else False
        self.anchor_idx = 0

        if self.mpa and self.split in ['train', 'val']:
            filepaths = join('./data/data_list', f"mpa_{data_category}_{split}.txt")
        else:
            if self.split == 'test': 
                split = 'val'
            
            # Read fracture path list
            if scale in ['overfitting', 'tiny']:
                filepaths = join('./data/data_list', f"{data_category}_{split}_{scale}.txt")
            elif scale == 'full':
                filepaths = join('./data/data_list', f"{data_category}_{split}.txt")
            else:
                filepaths = join('./data/data_list', f"{data_category}_{split}_small.txt")

        with open(filepaths, 'r') as f:
            self.filepaths = [x.strip() for x in f.readlines() if x.strip()]

        self.filepaths = [x for x in self.filepaths if self.min_part <= int(x.split()[0]) <= self.max_part]
        if self.sub_category != 'all': 
            self.filepaths = [x for x in self.filepaths if x.split()[1].split('/')[1] == self.sub_category]

        if self.mpa and self.split in ['train', 'val']:
            self.frac0 = [x.split()[2] for x in self.filepaths]
            self.frac1 = [x.split()[3] for x in self.filepaths]

        self.n_frac = [int(x.split()[0]) for x in self.filepaths]
        self.filepaths = [x.split()[1] for x in self.filepaths]
        self.len_filepaths = len(self.filepaths)
        
        self.overlap_radius = 0.018

        print("================================================")
        print(f"DATASET INITIALIZATION for {self.split}")
        print(f"datapath: {self.datapath}")
        print(f"data_category: {self.data_category}")
        print(f"split: {self.split}")
        print(f"sub_category: {self.sub_category}")
        print(f"n_pts: {self.n_pts}")
        print(f"visualize: {self.visualize}")
        print(f"min_n_pts: {self.min_n_pts}")
        print(f"min_part: {self.min_part}")
        print(f"max_part: {self.max_part}")
        print(f"mpa: {self.mpa}")
        print(f"anchor_idx: {self.anchor_idx}")
        print(f"overlap_radius: {self.overlap_radius}") 
        print(f"scale: {scale}")
        print(f"multiplicity: {self.multiplicity}")

        print(f"n_frac: {self.n_frac}")
        print(f"filepaths: {self.filepaths}")

        if self.mpa:
            print(f"frac0: {self.frac0}")
            print(f"frac1: {self.frac1}")
        print("================================================")
        

    def __len__(self):
        return self.len_filepaths * self.multiplicity


    def _translate(self, mesh, pcd):
        """Apply random translation to sampled points

        Args:
            mesh (list): list of meshes
            pcd (list): list of point clouds

        Returns:
            tuple: (translated_pcd, translated_mesh, gt_translations)
                - translated_pcd (list): list of translated point clouds (centered at origin)
                - translated_mesh (list): list of translated meshes (centered at origin)
                - gt_translations (list): list of translation vectors used for centering
        """
        gt_trans = [p.mean(dim=0) for p in pcd]
        pcd_t, mesh_t = [], [m.copy() for m in mesh]
        for idx, trans in enumerate(gt_trans):
            pcd_t.append(pcd[idx] - trans)
            mesh_t[idx].vertices -= trans.numpy()
        return pcd_t, mesh_t, gt_trans


    def _rotate(self, mesh, pcd):
        """Apply random rotation to sampled points

        Args:
            mesh (list): list of meshes
            pcd (list): list of point clouds

        Returns:
            tuple: (rotated_pcd, rotated_mesh, gt_rotations)
                - rotated_pcd (list): list of randomly rotated point clouds
                - rotated_mesh (list): list of randomly rotated meshes
                - gt_rotations (list): list of rotation matrices used for rotation
        """
        gt_rotat = [torch.tensor(R.random().as_matrix(), dtype=torch.float) for _ in pcd]
        pcd_t, mesh_t = [], [m.copy() for m in mesh]
        for idx, rotat in enumerate(gt_rotat):
            pcd_t.append(torch.einsum('x y, n y -> n x', rotat, pcd[idx]))
            mesh_t[idx].vertices = torch.einsum('x y, n y -> n x', rotat, torch.tensor(mesh_t[idx].vertices).float()).numpy()
        return pcd_t, mesh_t, gt_rotat


    def _compute_relative_transform(self, trans, rotat):
        """Compute relative transformation between each pairs

        Args:
            trans (list): list of translation vectors for each part
            rotat (list): list of rotation matrices for each part

        Returns:
            dict: Dictionary containing relative transformations between all pairs
                - Key format: "{src_idx}-{trg_idx}" (e.g., "0-1", "1-0")
                - Value: tuple of (relative_rotation, relative_translation)
                - relative_rotation: rotation matrix from src to trg coordinate system
                - relative_translation: translation vector from src to trg coordinate system
        """
        permut_relative_transform = {}
        for src_idx, trg_idx in itertools.permutations(range(len(trans)), 2):
            # Compute relative rotation and translation
            trans0, trans1 = trans[src_idx], trans[trg_idx]
            rotat0, rotat1 = rotat[src_idx], rotat[trg_idx]


            # From _pairwise_mating, we will move src_pcd to trg_pcd -> self._transform(src_pcd.squeeze(0), rotat, -trans, True)
            # src_pcd_t = R_0 * (src_pcd - T_0)
            # trg_pcd_t = R_1 * (trg_pcd - T_1)
            # R_0 * (src_pcd - T_0) -> R_1 * R_0^T * R_0 * (src_pcd - T_0) = R_1 * src_pcd - R_1 * T_0
            # - R_1 * T_0 + R' = - R_1 * T_1, So R' = R_1 * T_0 - R_1 * T_1 = R_1 * (T_0 - T_1)
            # In short, R_1 * R_0^T, R_1 * (T_0 - T_1)
            # Hence, in self._transform, R_1 * R_0^T * R_0 * (src_pcd - T_0) + R_1 * (T_0 - T_1) = R_1 * src_pcd - R_1 * T_0 + R_1 * T_0 - R_1 * T_1 = R_1 * src_pcd - R_1 * T_1 = R_1 * (src_pcd - T_1)

            relative_rotat = rotat1 @ rotat0.T
            relative_trans = (rotat1 @ (trans0 - trans1))

            # Save relative transformation between each pairs
            key = f"{src_idx}-{trg_idx}"
            permut_relative_transform[key] = relative_rotat, relative_trans

        if self.split in ['train', 'val']: 
            return {'0-1':permut_relative_transform['0-1']}
        else: 
            return permut_relative_transform

    
    def _extract_gt_normals(self, mesh, face, filepath):
        """Extract ground-truth normals from meshes and point clouds

        Args:
            mesh (list): list of meshes
            face (list): list of faces
            filepath (str): filepath of the object, self.filepaths[idx]

        Returns:
            list: list of ground-truth normals
        """
        gt_normals = []
        for i, mesh_ in enumerate(mesh):
            # Check if the mesh is watertight
            assert mesh_.is_watertight, f"[{filepath}] mesh_{i} is not watertight"

            # trimesh documentation
            # For face normals ensure that vectors are consistently pointed outwards, 
            # and that self.faces is wound in the correct direction for all connected components.
            mesh_.fix_normals()

            if mesh_.volume < 0: # Normal is pointing inward
                mesh_.invert()

            assert mesh_.is_winding_consistent, f"[{filepath}] mesh_{i} is not winding consistent"
            assert mesh_.volume > 0, f"[{filepath}] mesh_{i} has negative volume"

            gt_normals.append(mesh_.face_normals[face[i]])

        return gt_normals



    def __getitem__(self, idx):
        """
        [TODO] After finishing debugging, we should remove this comment
        For debugging, we already fix randomness from main.py by seed_everything(42, workers=True)
        So we don't need to fix randomness here for debugging purpose
        # Fix randomness
        if self.split in ['val', 'test']: 
            np.random.seed(idx)
            random.seed(idx)
        """
        idx = idx % self.len_filepaths

        # Read mesh, point cloud of a fractured object
        mesh, pcd, face = self.read_obj_data(idx)


        # Get all possible pairs. If two parts, then [0,1], [1,0]
        pair_indices = list(itertools.permutations([i for i in range(self.n_frac[idx])], 2))
        

        # Get ground-truth correspondences
        if self.split in ['train', 'val']:
            matching_inds = get_correspondences(to_o3d_pcd(pcd[0]), to_o3d_pcd(pcd[1]), self.overlap_radius)
        else:
            matching_inds = {}
            for pair_idx in pair_indices:
                pair_idx0, pair_idx1 = pair_idx
                matching_inds[f'{pair_idx0}-{pair_idx1}'] = get_correspondences(to_o3d_pcd(pcd[pair_idx0]), to_o3d_pcd(pcd[pair_idx1]), self.overlap_radius)
            matching_inds = [matching_inds]
        

        # Apply random transformation to sampled points
        pcd_t, mesh_t, gt_trans = self._translate(mesh, pcd)
        pcd_t, mesh_t, gt_rotat = self._rotate(mesh_t, pcd_t)
        gt_relative_trsfm = self._compute_relative_transform(gt_trans, gt_rotat)


        gt_normals = self._extract_gt_normals(mesh_t, face, self.filepaths[idx])
        

        batch = {
                'eval_idx': idx, # integer e.g. 0
                'filepath': self.filepaths[idx], # string e.g. 'everyday/BeerBottle/2927d6c8438f6e24fe6460d8d9bd16c6/fractured_37'
                'obj_class': self.filepaths[idx].split('/')[1], # string e.g. 'BeerBottle'

                'mesh': [torch.tensor(_mesh.vertices).float() for _mesh in mesh], # list of torch.Tensor, (N', 3)
                'mesh_t': [torch.tensor(_mesh.vertices).float() for _mesh in mesh_t], # list of torch.Tensor, (N', 3)
                
                'pcd': pcd, # list of torch.Tensor, (N, 3)
                'pcd_t': pcd_t, # list of torch.Tensor, (N, 3)

                'n_frac': self.n_frac[idx], # integer e.g. 2
                'anchor_idx': self.anchor_idx, # integer e.g. 0

                'gt_trans': gt_trans, # list of torch.Tensor, (3, )
                'gt_rotat': gt_rotat, # list of torch.Tensor, (3, 3)

                'gt_trans_inv': [-t for t in gt_trans], # list of torch.Tensor, (3, )
                'gt_rotat_inv': [R.T for R in gt_rotat], # list of torch.Tensor, (3, 3)
                
                'relative_trsfm': gt_relative_trsfm, # dict, key: string e.g. '0-1', value: tuple of (torch.Tensor (3, 3), torch.Tensor (3, ))

                'gt_normals': gt_normals, # list of torch.Tensor, (N, 3)
                'gt_correspondence': matching_inds, # if test then dict, key: string e.g. '0-1', value: torch.Tensor, (P, 2) else torch.Tensor, (P, 2)
                }
    
        return batch

    def read_obj_data(self, idx):        
        filepath = self.filepaths[idx]
        n_frac = self.n_frac[idx]

        # Load N-part meshes and calculate each area
        base_path = join(self.datapath, filepath)
        
        if self.mpa and self.split in ['train', 'val']:
            obj_paths = [join(base_path, x) for x in [self.frac0[idx], self.frac1[idx]]]
        else: 
            obj_paths = [join(base_path, x) for x in os.listdir(base_path)]


        # Load meshes, obj files
        meshes = [trimesh.load_mesh(x) for x in obj_paths] # If two parts, then length is 2
        mesh_areas = [mesh_.area for mesh_ in meshes] # area -> Summed area of all triangles in the current mesh


        # Set anchor fracture and sum all of areas
        self.anchor_idx, total_area = mesh_areas.index(max(mesh_areas)), sum(mesh_areas)


        # Sample N-part point clouds from meshes
        pcds = []
        faces = []
        for mesh in meshes:
            n_pts = int(self.n_pts * mesh.area / total_area)
            if self.split in ['val', 'test']: 
                sampled_pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_pts, seed=idx) # (N, 3), (N, )
            else: 
                sampled_pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_pts) # (N, 3), (N, )

            sampled_pts = torch.tensor(sampled_pts).float() # (N, 3)

            if sampled_pts.size(0) < self.min_n_pts: # if the number of points is less than the minimum number of points, sample more points
                if self.split in ['val', 'test']: 
                    extra_pts, extra_face_idx = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0), seed=idx) # (N', 3), (N', )
                else: 
                    extra_pts, extra_face_idx = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0)) # (N', 3), (N', )
                sampled_pts = torch.cat([sampled_pts, torch.tensor(extra_pts).float()], dim=0) # (N + N', 3)
                face_idx = np.concatenate([face_idx, extra_face_idx], axis=0) # (N + N', )
            
            pcds.append(sampled_pts)
            faces.append(face_idx)
        
        
        
        # [TODO] Implement MPA part after finishing two parts matching
        assert not self.mpa or len(pcds) <= 2, f"len(pcds): {len(pcds)}, mpa is blocked now"
        

        # Augment train dataset
        if self.split == 'train' and random.random() > 0.5:
            meshes.reverse()
            pcds.reverse()
            faces.reverse()
        
        return meshes, pcds, faces