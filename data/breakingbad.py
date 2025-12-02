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
from common.misc import offset2batch

class DatasetBreakingBad(Dataset):
    def __init__(self, datapath, data_category, sub_category, split, scale='full', multiplicity=1,
                 min_part=2, max_part=2, min_n_pts=256, n_pts=5000, overlap_radius=0.018):
        """Dataset for Breaking Bad

        Args:
            datapath (str): path to the dataset
            data_category (str): ['everyday', 'artifact'], candidates are fixed by argparse
            sub_category (str): ['all']
            split (str): ['train', 'val', 'test']
            scale (str): ['full', 'small', 'overfitting', 'tiny'], candidates are fixed by argparse
            multiplicity (int): multiplicity of the dataset
            min_part (int): minimum number of parts
            max_part (int): maximum number of parts
            min_n_pts (int): minimum number of points to sample
            n_pts (int): number of points to sample
            overlap_radius (float): overlap radius for correspondence
        """
        # Assertion
        assert split in ['train', 'val', 'test'], f"split must be in ['train', 'val', 'test'], but got {split}"

        self.datapath = datapath
        self.data_category = data_category 
        self.sub_category = sub_category
        self.split = split
        self.multiplicity = multiplicity if split == 'train' else 1
        
        self.min_part = min_part
        self.max_part = max_part
        self.min_n_pts = min_n_pts
        self.n_pts = n_pts

        self.overlap_radius = overlap_radius

        if self.split == 'test': 
            split = 'val'
            
        # Read fracture path list
        if scale in ['overfitting', 'tiny', 'small']:
            filepaths = join('./data/data_list', f"{data_category}_{split}_{scale}.txt")
        elif scale == 'full':
            filepaths = join('./data/data_list', f"{data_category}_{split}.txt")

        with open(filepaths, 'r') as f:
            self.filepaths = [x.strip() for x in f.readlines() if x.strip()]

        self.filepaths = [x for x in self.filepaths if self.min_part <= int(x.split()[0]) <= self.max_part]
        if self.sub_category != 'all': 
            self.filepaths = [x for x in self.filepaths if x.split()[1].split('/')[1] == self.sub_category]

        self.n_frac = [int(x.split()[0]) for x in self.filepaths]
        self.filepaths = [x.split()[1] for x in self.filepaths]
        self.len_filepaths = len(self.filepaths)
        
        

        print("================================================")
        print(f"DATASET INITIALIZATION for {self.split}")
        print(f"datapath: {self.datapath} | data_category: {self.data_category} | sub_category: {self.sub_category}")
        print(f"scale: {scale} | multiplicity: {self.multiplicity}")
        print(f"min_part: {self.min_part} | max_part: {self.max_part} | min_n_pts: {self.min_n_pts} | n_pts: {self.n_pts}")
        print(f"overlap_radius: {self.overlap_radius}")
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

            # From _pairwise_mating, we will move src_pcd to trg_pcd -> self._transform(src_pcd.squeeze(0), rotat, trans)
            # src_pcd_t = R_0 * (src_pcd - T_0)
            # trg_pcd_t = R_1 * (trg_pcd - T_1)
            # R_0 * (src_pcd - T_0) -> R_1 * R_0^T * R_0 * (src_pcd - T_0) = R_1 * src_pcd - R_1 * T_0
            # - R_1 * T_0 + T' = - R_1 * T_1, So T' = R_1 * T_0 - R_1 * T_1 = R_1 * (T_0 - T_1)
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

    
    def _extract_gt_normals(self, mesh, face):
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
            # trimesh documentation
            # For face normals ensure that vectors are consistently pointed outwards, 
            # and that self.faces is wound in the correct direction for all connected components.
            mesh_.fix_normals()
            gt_normals.append(torch.tensor(mesh_.face_normals[face[i]]))

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
        filepath, n_frac, anchor_idx, mesh, pcd, face = self.read_obj_data(idx)

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
        gt_normals = self._extract_gt_normals(mesh_t, face)
        
        concat_pcd = torch.cat(pcd, dim=0) # (total_N, 3)
        concat_pcd_t = torch.cat(pcd_t, dim=0) # (total_N, 3)
        concat_gt_normals = torch.cat(gt_normals, dim=0) # (total_N, 3)
        pcd_batch_info = offset2batch(torch.tensor([len(pcd_) for pcd_ in pcd])) # (total_N, )

        batch = {
                'eval_idx': idx, # integer e.g. 0
                'filepath': filepath, # string e.g. 'everyday/BeerBottle/2927d6c8438f6e24fe6460d8d9bd16c6/fractured_37'
                'obj_class': filepath.split('/')[1], # string e.g. 'BeerBottle'
                'n_frac': n_frac, # integer e.g. 2
                'anchor_idx': anchor_idx, # integer e.g. 0
                
                'pcd': concat_pcd, # torch.Tensor, (total_N, 3)
                'pcd_t': concat_pcd_t, # torch.Tensor, (total_N, 3)
                'gt_normals': concat_gt_normals, # torch.Tensor, (total_N, 3)
                'pcd_batch_info': pcd_batch_info, # torch.Tensor, (total_N, ), batch index of the point cloud
                'gt_correspondence': matching_inds, # if test then dict, key: string e.g. '0-1', value: torch.Tensor, (Corr, 2) else torch.Tensor, (Corr, 2)
                }
        
        if self.split in ['val', 'test']:
            eval_dict = {
                # Eval
                'mesh': [torch.tensor(_mesh.vertices).float() for _mesh in mesh], # list of torch.Tensor, (N', 3)
                'mesh_t': [torch.tensor(_mesh.vertices).float() for _mesh in mesh_t], # list of torch.Tensor, (N', 3)
                'mesh_faces': [torch.tensor(_mesh.faces) for _mesh in mesh], # list of torch.Tensor, (F, 3)
                'relative_trsfm': gt_relative_trsfm, # dict, key: string e.g. '0-1', value: tuple of (torch.Tensor (3, 3), torch.Tensor (3, ))
            }
            batch.update(eval_dict)
    
        return batch
    

    def read_obj_data(self, idx):        
        filepath = self.filepaths[idx]
        n_frac = self.n_frac[idx]

        # Load N-part meshes and calculate each area
        base_path = join(self.datapath, filepath)
        obj_paths = [join(base_path, x) for x in os.listdir(base_path)]
        
        # Load meshes, obj files
        meshes = [trimesh.load_mesh(x) for x in obj_paths] # If two parts, then length is 2
        mesh_areas = [mesh_.area for mesh_ in meshes] # area -> Summed area of all triangles in the current mesh

        # Set anchor fracture and sum all of areas
        anchor_idx, total_area = mesh_areas.index(max(mesh_areas)), sum(mesh_areas)

        # Calculate number of points for each part
        remaining_points = self.n_pts - self.min_n_pts * len(meshes)
        counts = (self.min_n_pts + (remaining_points * (mesh_areas / total_area)).astype(int)).tolist()
        diff = self.n_pts - sum(counts)
        counts[np.argmax(counts)] += diff


        # Sample N-part point clouds from meshes
        pcds = []
        faces = []
        for mesh, n_pts in zip(meshes, counts):
            if self.split in ['val', 'test']: 
                sampled_pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_pts, seed=idx) # (N, 3), (N, )
            else: 
                sampled_pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_pts) # (N, 3), (N, )

            sampled_pts = torch.tensor(sampled_pts).float() # (N, 3)

            if sampled_pts.size(0) < n_pts: # if the number of points is less than the number of points to sample, sample more points
                if self.split in ['val', 'test']: 
                    extra_pts, extra_face_idx = trimesh.sample.sample_surface(mesh, n_pts - sampled_pts.size(0), seed=idx) # (N', 3), (N', )
                else: 
                    extra_pts, extra_face_idx = trimesh.sample.sample_surface(mesh, n_pts - sampled_pts.size(0)) # (N', 3), (N', )
                sampled_pts = torch.cat([sampled_pts, torch.tensor(extra_pts).float()], dim=0) # (N + N', 3)
                face_idx = np.concatenate([face_idx, extra_face_idx], axis=0) # (N + N', )
            
            pcds.append(sampled_pts)
            faces.append(face_idx)


        # Augment train dataset
        if self.split == 'train' and random.random() > 0.5:
            meshes.reverse()
            pcds.reverse()
            faces.reverse()
        
        return filepath, n_frac, anchor_idx, meshes, pcds, faces


def collate_fn(batch):
    result_batch = {}

    for batch_key in batch[0].keys():
        if batch_key in ['eval_idx', 'n_frac', 'anchor_idx']: # Single numerical value
            result_batch[batch_key] = torch.tensor([a_batch[batch_key] for a_batch in batch]) # (B, )

        elif batch_key in ['filepath', 'obj_class']: # Single string value
            result_batch[batch_key] = [a_batch[batch_key] for a_batch in batch] # (B, )

        elif batch_key in ['pcd', 'pcd_t', 'gt_normals', 'pcd_batch_info']:
            result_batch[batch_key] = torch.stack([a_batch[batch_key] for a_batch in batch], dim=0) # (B, total_N, 3) or (B, total_N, )
        
        elif batch_key in ['gt_correspondence']:
            if isinstance(batch[0][batch_key], dict):
                assert len(batch) == 1, f"len(batch): {len(batch)}, batch size must be 1 for evaluation"
                result_batch[batch_key] = batch[0][batch_key] # (Corr, 2)
            else:
                list_of_gt_correspondence = [a_batch[batch_key] for a_batch in batch]
                gt_corr_offset_info = torch.tensor([len(gt_corr) for gt_corr in list_of_gt_correspondence]) # (B, )
                result_batch[batch_key] = torch.cat(list_of_gt_correspondence, dim=0) # (total_Corr, 2)
                result_batch['gt_corr_offset_info'] = gt_corr_offset_info
        
        elif batch_key in ['mesh', 'mesh_t', 'mesh_faces', 'relative_trsfm']: # Only for evaluation, So batch size must be 1
            assert len(batch) == 1, f"len(batch): {len(batch)}, batch size must be 1 for evaluation"
            result_batch[batch_key] = batch[0][batch_key]
    
    return result_batch