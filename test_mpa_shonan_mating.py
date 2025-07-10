import os
import sys
import pwd
import argparse
import importlib
import time
import gc
from distutils.dir_util import copy_tree

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset import GADataset
from common import utils
import open3d as o3d

from model.equiassem_mating import EquiAssem

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

import itertools
import gtsam
import numpy as np
import random
import open3d as o3d
import trimesh

from html_vis_matches import MatchVisualizer

from chamfer_distance import ChamferDistance as chamfer_dist

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

def save_pc(filename:str, pcd_tensors:list, mating_idx:list=None):
    colors = [
        [1, 0.996, 0.804],
        [0.804, 0.98, 1],
        [1, 0.376, 0],
        [0, 0.055, 1]
    ]

    mating_colors = [
        [0.85, 0.80, 0.5],
        [0.4, 0.7, 0.85],
        [0.75, 0.2, 0],
        [0, 0.03, 0.7]
    ]

    pcds = []
    for i, tensor_ in enumerate(pcd_tensors):
        if tensor_.size()[0] == 1:
            tensor_ = tensor_.squeeze(0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tensor_.cpu().numpy())
        pcd.paint_uniform_color(colors[i % len(colors)])
        pcds.append(pcd)
        if mating_idx is not None and len(pcd_tensors) == len(mating_idx):
            mating_pcd = o3d.geometry.PointCloud()
            mating_pcd.points = o3d.utility.Vector3dVector(tensor_[mating_idx[i]].cpu().numpy())
            mating_pcd.paint_uniform_color(mating_colors[i % len(mating_colors)])
            pcds.append(mating_pcd)
    combined_cloud = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_cloud += pcd
    o3d.io.write_point_cloud(filename, combined_cloud)

def save_mesh(in_dict, out_dict, crd, cd, pa_crd, pa_cd, folder_name):
    base_path = '../../../data/bbad_v2/'
    obj_paths = [os.path.join(base_path+in_dict['filepath'], x) for x in os.listdir(base_path+in_dict['filepath'])]
    mesh = [trimesh.load_mesh(x) for x in obj_paths]
    idx = 0
    for rotat, trans in zip(in_dict['gt_rotat'], in_dict['gt_trans']):
        mesh[idx].vertices = (rotat[0].cpu() @ (torch.tensor(mesh[idx].vertices).float() - trans.cpu()).T).T
        idx+=1

    idx = 0
    for rotat, trans in zip(out_dict['aligned_pred_rotat'], out_dict['aligned_pred_trans']):
    # for rotat, trans in zip(in_dict['aligned_gt_rotat'], in_dict['aligned_gt_trans']): # for gt save
        mesh[idx].vertices = (rotat.inverse().cpu() @ (torch.tensor(mesh[idx].vertices).float() + trans.cpu()).T).T
        idx+=1
    
    # Create directories if they don't exist
    save_dir = os.path.join('vis', folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{in_dict["eval_idx"][0].item()}_{len(in_dict["gt_rotat"])}part_crd{round(crd,2)}_cd{round(cd,2)}_pacrd{round(pa_crd,2)}_pacd{round(pa_cd,2)}_'+in_dict['filepath'].replace('/','_'))
    # save_path = os.path.join('vis','artifact_gt',f'{len(in_dict["gt_rotat"])}part_'+in_dict['filepath'].replace('/','_'))

    os.mkdir(save_path)
    for idx, _mesh in enumerate(mesh):
        _mesh.export(os.path.join(save_path,f'{idx}_fracture.obj'), file_type='obj')
    sum(mesh).export(os.path.join(save_path,f'assemble.obj'), file_type='obj')

def save_ransaced_mesh(in_dict, out_dict, crd, cd, pa_crd, pa_cd, folder_name):
    base_path = '../../../data/bbad_v2/'
    obj_paths = [os.path.join(base_path+in_dict['filepath'], x) for x in os.listdir(base_path+in_dict['filepath'])]
    mesh = [trimesh.load_mesh(x) for x in obj_paths]
    idx = 0
    for rotat, trans in zip(in_dict['gt_rotat'], in_dict['gt_trans']):
        mesh[idx].vertices = (rotat[0].cpu() @ (torch.tensor(mesh[idx].vertices).float() - trans.cpu()).T).T
        idx+=1

    idx = 0
    # for rotat, trans in zip(out_dict['aligned_pred_rotat'], out_dict['aligned_pred_trans']):
    for rotat, trans in zip(in_dict['aligned_gt_rotat'], in_dict['aligned_gt_trans']): # for gt save
        mesh[idx].vertices = (rotat.inverse().cpu() @ (torch.tensor(mesh[idx].vertices).float() + trans.cpu()).T).T
        idx+=1

    # Create directories if they don't exist
    save_dir = os.path.join('vis', folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{in_dict["eval_idx"][0].item()}_{len(in_dict["gt_rotat"])}part_crd{round(crd,2)}_cd{round(cd,2)}_pacrd{round(pa_crd,2)}_pacd{round(pa_cd,2)}_'+in_dict['filepath'].replace('/','_'))
    # save_path = os.path.join('vis','artifact_gt',f'{len(in_dict["gt_rotat"])}part_'+in_dict['filepath'].replace('/','_'))

    os.mkdir(save_path)
    for idx, _mesh in enumerate(mesh):
        _mesh.export(os.path.join(save_path,f'{idx}_fracture.obj'), file_type='obj')
    sum(mesh).export(os.path.join(save_path,f'assemble.obj'), file_type='obj')

def _transform(pcd, rotat=None, trans=None, rotate_first=True):
    if rotat == None: rotat = torch.eye(3, 3)
    if trans == None: trans = torch.zeros(3)

    rotat = rotat.to(pcd.device)
    trans = trans.to(pcd.device)

    if rotate_first:
        return torch.einsum('x y, n y -> n x', rotat, pcd) + trans
    else:
        return torch.einsum('x y, n y -> n x', rotat, pcd + trans)

def _multi_part_assemble(pcds, rotat, trans):
    pcd_t = []
    for pcd, R, t in zip(pcds, rotat, trans):
        pcd_t.append(_transform(pcd.squeeze(0), R.inverse(), t, False))
    return torch.cat(pcd_t, dim=0), pcd_t

def _chamfer_distance(assm1, assm2, scaling=1000):
    chd = chamfer_dist()
    dist1, dist2, idx1, idx2 = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
    cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling
    return cd

def _correspondence_distance(assm1, assm2, scaling=100):
    corr_dist = (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling
    return corr_dist

def _transformation_error(rotat1, rotat2, trans1, trans2, rrmse_scaling=100):
    rrmse, trmse = 0., 0.
    for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
        r1_deg = torch.tensor(Rotation.from_matrix(r1.cpu()).as_euler('xyz', degrees=True))
        r2_deg = torch.tensor(Rotation.from_matrix(r2.cpu()).as_euler('xyz', degrees=True))
        diff1 = (r1_deg - r2_deg).abs()
        diff2 = 360. - (r1_deg - r2_deg).abs()
        diff = torch.minimum(diff1, diff2)
        rrmse += diff.pow(2).mean().pow(0.5)
        trmse += (t1 - t2).pow(2).mean().pow(0.5) * rrmse_scaling
    div = len(rotat1)
    return rrmse / div, trmse / div

def _part_accuracy(assm_pts1, assm_pts2, scaling=100):
    success = 0
    for pred_pts, gt_pts in zip(assm_pts1, assm_pts2):
        part_cd = _chamfer_distance(pred_pts, gt_pts, 1)
        if part_cd < args.distance_threshold: success += 1
    return success / len(assm_pts1)

def _part_accuracy_crd(assm_pts1, assm_pts2, scaling=100):
    success = 0
    for pred_pts, gt_pts in zip(assm_pts1, assm_pts2):
        part_crd = _correspondence_distance(pred_pts, gt_pts, 1)
        if part_crd < 0.1: success += 1
    return success / len(assm_pts1)

def estimate_poses_given_rot(
        factors: gtsam.BetweenFactorPose3s,
        rotations: gtsam.Values,
        uncertainty,
        anchor_idx
):
    """Estimate Poses from measurements, given rotations. From SfmProblem in shonan.
    Arguments:
        factors -- data structure with many BetweenFactorPose3 factors
        rotations {Values} -- Estimated rotations
    Returns:
        Values -- Estimated Poses
    """

    graph = gtsam.GaussianFactorGraph()
    model = gtsam.noiseModel.Unit.Create(3)

    # Add a factor anchoring t_anchor
    graph.add(anchor_idx, np.eye(3), np.zeros((3,)), model)

    # Add a factor saying t_j - t_i = Ri*t_ij for all edges (i,j)
    for idx in range(len(factors)):
        factor = factors[idx]
        keys = factor.keys()
        i, j, Tij = keys[0], keys[1], factor.measured()
        if i == j:
            continue
        model = gtsam.noiseModel.Diagonal.Variances(
            uncertainty[idx] * (1e-2) * np.ones(3)
        )
        measured = rotations.atRot3(j).inverse().rotate(Tij.translation())
        # measured = Tij.translation()
        graph.add(j, np.eye(3), i, -np.eye(3), measured, model)
        # graph.add(j, rotations.atRot3(j).matrix(), i, -rotations.atRot3(i).matrix(), measured, model)

    # Solve linear system
    translations = graph.optimize()
    # Convert to Values.
    result = gtsam.Values()
    for j in range(rotations.size()):
        tj = translations.at(j)
        result.insert(j, gtsam.Pose3(rotations.atRot3(j), tj))

    return result

def test(args):
    torch.cuda.synchronize()
    start_time = time.time()
    # Model initialization
    utils.fix_randseed(0)
    model = EquiAssem(lr=args.lr,
                        backbone=args.backbone,
                        occ_loss=args.occ_loss, 
                        no_ori=args.no_ori,
                        registration=args.registration,
                        match_selection_str=args.initial_match_selection,
                        visualize=args.visualize,
                        debug=args.debug,
                        inlier_threshold=args.distance_threshold,
                        score_threshold=args.score_threshold,
                        auto_score_threshold=args.auto_score_threshold,
                        ori_threshold=args.ori_threshold,
                        weighted_voting=args.weighted_voting,
                        gt_normal_threshold=args.gt_normal_threshold,
                        gt_mating_surface=args.gt_mating_surface,
                        score_comb=args.score_comb,
                        topk=args.topk,
                        optimal_matching_choice=args.optimal_matching_choice)
    # print(model)

    model.to(torch.device('cuda:0'))
    model.eval()
    # Load checkpoint
    if args.load:
        print(f"Loading checkpoint from {args.load}")
        checkpoint = torch.load(args.load, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("No checkpoint specified. Exiting program.")
        sys.exit(1)
    
    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.min_part, args.max_part, args.n_pts, args.scale)
    dataloader_test = GADataset.build_dataloader(args.batch_size, args.n_worker, 'test')
    total = len(dataloader_test)
    crd_list, cd_list, rrmse_list, trsme_list, pa_list, pa_crd_list = [], [], [], [], [], []
    strange_cases = []
    less_than_3 = 0

    for idx, in_dict in enumerate(dataloader_test):
        # 1. Network forward pass: Pairwise matching & assembly
        in_dict = utils.to_cuda(in_dict)

        # if args.visualize:
        #     if len(in_dict["pcd_t"]) > 5: 
        #         result_str = f'{in_dict["eval_idx"][0].item()}/{total} | #-Part: {len(in_dict["pcd_t"])} | CRD: nan | CD: nan | RRMSE: nan | TRMSE: nan | PA(cd): nan | PA(crd): nan'
        #         print(result_str)

        #         # Write results in 'w' mode first to clear previous results
        #         if idx == 0:
        #             with open('mpa_artifact_results.txt', 'w') as f:
        #                 f.write(result_str + '\n')
        #         else: 
        #             with open('mpa_artifact_results.txt', 'a') as f:
        #                 f.write(result_str + '\n')
        #         continue

        pair_indices = list(itertools.permutations([i for i in range(in_dict['n_frac'])], 2))
        out_dict = {}
        for pair_idx in pair_indices:
            # Initialize pairwise input
            pair_idx0, pair_idx1 = pair_idx
            in_dict['pcd_t'][in_dict['anchor_idx']] = in_dict['pcd_t'][in_dict['anchor_idx']].squeeze(0) @ in_dict['gt_rotat'][in_dict['anchor_idx']] + in_dict['gt_trans'][in_dict['anchor_idx']]
            # if (not args.html_vis) or (args.html_vis and args.initial_match_selection == 'unidirectional_nn' and (in_dict['pcd'][pair_idx0].size(1) >= in_dict['pcd'][pair_idx1].size(1))):
            if in_dict['gt_mating_surface'][f'{pair_idx0}-{pair_idx1}'][0].sum() > 3 and in_dict['gt_mating_surface'][f'{pair_idx0}-{pair_idx1}'][1].sum() > 3:
                if in_dict['gt_mating_surface'][f'{pair_idx0}-{pair_idx1}'][0].sum() >= in_dict['gt_mating_surface'][f'{pair_idx0}-{pair_idx1}'][1].sum():
                    in_dict_pair = {
                        'filepath': in_dict['filepath'], 'obj_class': in_dict['obj_class'],
                        'pcd_t': [in_dict['pcd_t'][pair_idx0], in_dict['pcd_t'][pair_idx1]],
                        'pcd': [in_dict['pcd'][pair_idx0], in_dict['pcd'][pair_idx1]],
                        'n_frac': 2, # Pairwise matching
                        'gt_trans': [in_dict['gt_trans'][pair_idx0], in_dict['gt_trans'][pair_idx1]],
                        'gt_rotat': [in_dict['gt_rotat'][pair_idx0], in_dict['gt_rotat'][pair_idx1]],
                        'gt_rotat_inv': [in_dict['gt_rotat_inv'][pair_idx0], in_dict['gt_rotat_inv'][pair_idx1]],
                        'gt_trans_inv': [in_dict['gt_trans_inv'][pair_idx0], in_dict['gt_trans_inv'][pair_idx1]],
                        'relative_trsfm': {f'{pair_idx0}-{pair_idx1}': in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}']},
                        'pair_idx': f'{pair_idx0}-{pair_idx1}',
                        'gt_correspondence': in_dict['gt_correspondence'][f'{pair_idx0}-{pair_idx1}'],
                        'gt_normals': [in_dict['gt_normals'][pair_idx0], in_dict['gt_normals'][pair_idx1]],
                        'gt_mating_surface': in_dict['gt_mating_surface'][f'{pair_idx0}-{pair_idx1}'],
                    }
                    # Forward pass
                    print(f'===== Pair {pair_idx0}-{pair_idx1} =====')
                    out_dict[f'{pair_idx0}-{pair_idx1}'] = model.forward_mpa(in_dict_pair)
                    torch.cuda.empty_cache(); gc.collect()
            else:
                if (not args.html_vis) or (args.html_vis and args.initial_match_selection == 'unidirectional_nn' and (in_dict['pcd'][pair_idx0].size(1) >= in_dict['pcd'][pair_idx1].size(1))):
                    in_dict_pair = {
                        'filepath': in_dict['filepath'], 'obj_class': in_dict['obj_class'],
                        'pcd_t': [in_dict['pcd_t'][pair_idx0], in_dict['pcd_t'][pair_idx1]],
                        'pcd': [in_dict['pcd'][pair_idx0], in_dict['pcd'][pair_idx1]],
                        'n_frac': 2, # Pairwise matching
                        'gt_trans': [in_dict['gt_trans'][pair_idx0], in_dict['gt_trans'][pair_idx1]],
                        'gt_rotat': [in_dict['gt_rotat'][pair_idx0], in_dict['gt_rotat'][pair_idx1]],
                        'gt_rotat_inv': [in_dict['gt_rotat_inv'][pair_idx0], in_dict['gt_rotat_inv'][pair_idx1]],
                        'gt_trans_inv': [in_dict['gt_trans_inv'][pair_idx0], in_dict['gt_trans_inv'][pair_idx1]],
                        'relative_trsfm': {f'{pair_idx0}-{pair_idx1}': in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}']},
                        'pair_idx': f'{pair_idx0}-{pair_idx1}',
                        'gt_correspondence': in_dict['gt_correspondence'][f'{pair_idx0}-{pair_idx1}'],
                        'gt_normals': [in_dict['gt_normals'][pair_idx0], in_dict['gt_normals'][pair_idx1]],
                        'gt_mating_surface': in_dict['gt_mating_surface'][f'{pair_idx0}-{pair_idx1}'],
                    }
                    # Forward pass
                    print(f'===== Pair {pair_idx0}-{pair_idx1} =====')
                    out_dict[f'{pair_idx0}-{pair_idx1}'] = model.forward_mpa(in_dict_pair)
                    torch.cuda.empty_cache(); gc.collect()

                    strange_cases.append([in_dict['filepath'], pair_idx0, pair_idx1, in_dict['gt_mating_surface'][f'{pair_idx0}-{pair_idx1}'][0].sum(), in_dict['gt_mating_surface'][f'{pair_idx0}-{pair_idx1}'][1].sum()])
                    print(strange_cases)
        
        if not args.html_vis:
            if args.registration == 'wsvd':
                # 2. Pose Graph Optimization
                params = gtsam.ShonanAveragingParameters3(gtsam.LevenbergMarquardtParams.CeresDefaults())
                factors = gtsam.BetweenFactorPose3s()

                # # 2-1. Add factors(relative transformations)
                uncertainty = []
                for i in range(in_dict['n_frac'].item()):
                    # search highest score
                    max_score = 0
                    for j in range(in_dict['n_frac'].item()):
                        if i == j: continue
                        value = torch.exp(out_dict[f'{j}-{i}']['matching_scores_drop']).sum()
                        if max_score < value:
                            max_score = value
                            max_idx = j
                    relative_rotat = Rotation.from_matrix(out_dict[f'{i}-{max_idx}']['estimated_rotat'].cpu().numpy()).as_quat()
                    relative_trans = out_dict[f'{i}-{max_idx}']['estimated_trans'].cpu().numpy()
                    max_score = max_score.cpu()

                    # add factor
                    pose = gtsam.Pose3(gtsam.Rot3.Quaternion(relative_rotat[3], relative_rotat[0], relative_rotat[1], relative_rotat[2]), gtsam.Point3(relative_trans))
                    factors.append(gtsam.BetweenFactorPose3(i, max_idx, pose, gtsam.noiseModel.Diagonal.Information((1/max_score) * np.eye(6))))
                    uncertainty.append(1/max_score)
                
                # 2-3. Run shonan averaging
                sa3 = gtsam.ShonanAveraging3(factors, params)
                initial = sa3.initializeRandomly()
                pMax = 20
                shonan_fail = False
                while True:
                    pMax += 20
                    if pMax == 60:
                        shonan_fail = True; print("shonan failed")
                        break
                    try: 
                        abs_rotat, _ = sa3.run(initial, 3, pMax)
                        break
                    except RuntimeError as e:
                        print(f"An error occurred during Shonan::run: with pMax {pMax}")
                        continue

                # Align predicted rotation to anchor fracture
                anchor_idx = in_dict['anchor_idx']
                # if not shonan_fail:
                #     aligned_pred_rotat, aligned_pred_trans = [], []
                #     abs_anchor_R = abs_rotat.atRot3(anchor_idx)
                #     # Align rotations
                #     for j in range(abs_rotat.size()):
                #         aligned_pred_rotat.append(torch.tensor(abs_anchor_R.between(abs_rotat.atRot3(j)).matrix()).to(torch.float32).cuda())
                # else:
                #     aligned_pred_rotat = []
                #     for i in range(0, in_dict['n_frac']):
                #         if i == anchor_idx: aligned_pred_rotat.append(torch.eye(3).to(torch.float32).cuda())
                #         else: aligned_pred_rotat.append(out_dict[f'{anchor_idx}-{i}']['estimated_rotat'].squeeze(0))
                
                # # Align predicted translation to anchor fracture
                # for i in range(0, in_dict['n_frac']):
                #     if i == anchor_idx: aligned_pred_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
                #     else: aligned_pred_trans.append(out_dict[f'{anchor_idx}-{i}']['estimated_trans'])
                if not shonan_fail:
                    aligned_pred_rotat, aligned_pred_trans = [], []

                    aligned_pred_rotat2 = []
                    abs_anchor_R2 = abs_rotat.atRot3(anchor_idx)
                    # Align rotations
                    for j in range(abs_rotat.size()):
                        aligned_pred_rotat2.append(torch.tensor(abs_anchor_R2.between(abs_rotat.atRot3(j)).matrix()).to(torch.float32).cuda())

                    ## Align rotations and translations
                    rel_rotat = gtsam.Values()
                    abs_anchor_R = abs_rotat.atRot3(anchor_idx)
                    for i in range(abs_rotat.size()):
                        if i == anchor_idx:
                            rel_rotat.insert(i, gtsam.Rot3(np.eye(3)))
                        else:
                            rel_rotat.insert(i, abs_anchor_R.inverse().compose(abs_rotat.atRot3(i)))

                    poses = estimate_poses_given_rot(
                        factors, rel_rotat, np.array(uncertainty), anchor_idx
                    )
                    # abs_anchor_R = poses.atPose3(anchor_idx).rotation()
                    abs_anchor_T = poses.atPose3(anchor_idx).translation()
                    
                    for j in range(poses.size()):
                        aligned_pred_rotat.append(torch.tensor(abs_anchor_R.between(abs_rotat.atRot3(j)).matrix()).to(torch.float32).cuda())
                        pred_trans = poses.atPose3(j).rotation().rotate(abs_anchor_T - poses.atPose3(j).translation())
                        # measured = rotations.atRot3(j).inverse().rotate(Tij.translation())
                        aligned_pred_trans.append(torch.tensor(pred_trans).to(torch.float32).cuda())

                else:
                    aligned_pred_rotat, aligned_pred_trans = [], []
                    for i in range(0, in_dict['n_frac']):
                        if i == anchor_idx: 
                            aligned_pred_rotat.append(torch.eye(3).to(torch.float32).cuda())
                            aligned_pred_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
                        else: 
                            aligned_pred_rotat.append(out_dict[f'{anchor_idx}-{i}']['estimated_rotat'].squeeze(0))
                            aligned_pred_trans.append(out_dict[f'{anchor_idx}-{i}']['estimated_trans'])
                
                # Align GT transformation to anchor fracture
                aligned_gt_rotat, aligned_gt_trans = [], []
                for i in range(0, in_dict['n_frac']):
                    if i == anchor_idx:
                        aligned_gt_rotat.append(torch.eye(3).to(torch.float32).cuda())
                        aligned_gt_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
                    else:
                        aligned_gt_rotat.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][0].squeeze(0))
                        aligned_gt_trans.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][1].squeeze(0))

                # Save aligned transformations
                in_dict['aligned_gt_trans'] = aligned_gt_trans
                in_dict['aligned_gt_rotat'] = aligned_gt_rotat
                out_dict['aligned_pred_trans'] = aligned_pred_trans
                out_dict['aligned_pred_rotat'] = aligned_pred_rotat
                assm_pred, pcds_pred = _multi_part_assemble(in_dict['pcd_t'], aligned_pred_rotat, aligned_pred_trans)
                assm_grtr, pcds_grtr = _multi_part_assemble(in_dict['pcd_t'], aligned_gt_rotat, aligned_gt_trans)

                cd = _chamfer_distance(assm_pred, assm_grtr).item()
                crd = _correspondence_distance(assm_pred, assm_grtr).item()
                rrmse, trmse = _transformation_error(aligned_pred_rotat, aligned_gt_rotat, aligned_pred_trans, aligned_gt_trans)
                rrmse, trmse = rrmse.item(), trmse.item()
                pa = _part_accuracy(pcds_pred, pcds_grtr)
                pa_crd = _part_accuracy_crd(pcds_pred, pcds_grtr)

                if args.visualize: save_mesh(in_dict, out_dict, crd, cd, pa_crd, pa)
                
                crd_list.append(crd)
                cd_list.append(cd)
                rrmse_list.append(rrmse)
                trsme_list.append(trmse)
                pa_list.append(pa)
                pa_crd_list.append(pa_crd)

                result_str = f'{idx}/{total} | #-Part: {len(pcds_pred)} | CRD: {round(crd,2)} | CD: {round(cd,2)} | RRMSE: {round(rrmse,2)} | TRMSE: {round(trmse,2)} | PA(cd): {round(pa,2)} | PA(crd): {round(pa_crd,2)}'
                print(result_str)

                # Write results in 'w' mode first to clear previous results
                # if idx == 0:
                #     with open('mpa_artifact_results.txt', 'w') as f:
                #         f.write(result_str + '\n')
                # else: 
                #     with open('mpa_artifact_results.txt', 'a') as f:
                #         f.write(result_str + '\n')
            elif args.registration == 'ransac':
                # Assembly without graph optimization (일단은...)
                anchor_idx = in_dict['anchor_idx']
                aligned_pred_rotat, aligned_pred_trans = [], []
                # breakpoint()
                for i in range(0, in_dict['n_frac']):
                    if i == anchor_idx: 
                        aligned_pred_rotat.append(torch.eye(3).to(torch.float32).cuda())
                        aligned_pred_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
                    else: 
                        if f'{anchor_idx}-{i}' in out_dict.keys():
                            aligned_pred_rotat.append(out_dict[f'{anchor_idx}-{i}']['estimated_rotat'].squeeze(0))
                            aligned_pred_trans.append(out_dict[f'{anchor_idx}-{i}']['estimated_trans'])
                            less_than_3 += out_dict[f'{anchor_idx}-{i}']['less_than_3']
                            if out_dict[f'{anchor_idx}-{i}']['less_than_3']:
                                strange_cases.append(in_dict['filepath'])
                            # if 'strange_case' in out_dict[f'{anchor_idx}-{i}'].keys():
                            #     strange_cases += out_dict[f'{anchor_idx}-{i}']['strange_case']
                        else:
                            aligned_pred_rotat.append(out_dict[f'{i}-{anchor_idx}']['estimated_rotat'].squeeze(0).inverse())
                            aligned_pred_trans.append(-out_dict[f'{i}-{anchor_idx}']['estimated_trans'] @ out_dict[f'{i}-{anchor_idx}']['estimated_rotat'].squeeze(0))
                            less_than_3 += out_dict[f'{i}-{anchor_idx}']['less_than_3']
                            if out_dict[f'{i}-{anchor_idx}']['less_than_3']:
                                strange_cases.append(in_dict['filepath'])
                            # if 'strange_case' in out_dict[f'{i}-{anchor_idx}'].keys():
                            #     strange_cases += out_dict[f'{i}-{anchor_idx}']['strange_case']
                
                # Align GT transformation to anchor fracture
                aligned_gt_rotat, aligned_gt_trans = [], []
                for i in range(0, in_dict['n_frac']):
                    if i == anchor_idx:
                        aligned_gt_rotat.append(torch.eye(3).to(torch.float32).cuda())
                        aligned_gt_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
                    else:
                        aligned_gt_rotat.append(in_dict['gt_rotat'][i].squeeze(0))
                        aligned_gt_trans.append(in_dict['gt_trans'][i].squeeze(0))
                        # if f'{anchor_idx}-{i}' in out_dict.keys():
                        #     aligned_gt_rotat.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][0].squeeze(0))
                        #     aligned_gt_trans.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][1].squeeze(0))
                        # else:
                        #     aligned_gt_rotat.append(in_dict['relative_trsfm'][f'{i}-{anchor_idx}'][0].squeeze(0).inverse())
                        #     aligned_gt_trans.append(-in_dict['relative_trsfm'][f'{i}-{anchor_idx}'][1].squeeze(0) @ in_dict['relative_trsfm'][f'{i}-{anchor_idx}'][0].squeeze(0))

                # Save aligned transformations
                in_dict['aligned_gt_trans'] = aligned_gt_trans
                in_dict['aligned_gt_rotat'] = aligned_gt_rotat
                out_dict['aligned_pred_trans'] = aligned_pred_trans
                out_dict['aligned_pred_rotat'] = aligned_pred_rotat
                assm_pred, pcds_pred = _multi_part_assemble(in_dict['pcd_t'], aligned_pred_rotat, aligned_pred_trans)
                assm_grtr, pcds_grtr = _multi_part_assemble(in_dict['pcd_t'], aligned_gt_rotat, aligned_gt_trans)

                cd = _chamfer_distance(assm_pred, assm_grtr).item()
                crd = _correspondence_distance(assm_pred, assm_grtr).item()
                rrmse, trmse = _transformation_error(aligned_pred_rotat, aligned_gt_rotat, aligned_pred_trans, aligned_gt_trans)
                rrmse, trmse = rrmse.item(), trmse.item()
                pa = _part_accuracy(pcds_pred, pcds_grtr)
                pa_crd = _part_accuracy_crd(pcds_pred, pcds_grtr)

                if args.visualize: save_ransaced_mesh(in_dict, out_dict, crd, cd, pa_crd, pa, args.vis_folder)
                
                crd_list.append(crd)
                cd_list.append(cd)
                rrmse_list.append(rrmse)
                trsme_list.append(trmse)
                pa_list.append(pa)
                pa_crd_list.append(pa_crd)

                # if pa == 0.5:
                #     breakpoint()
                result_str = f'{idx}/{total} | #-Part: {len(pcds_pred)} | CRD: {round(crd,2)} | CD: {round(cd,2)} | RRMSE: {round(rrmse,2)} | TRMSE: {round(trmse,2)} | PA(cd): {round(pa,2)} | PA(crd): {round(pa_crd,2)}'
                print(result_str)
                breakpoint()
                save_folder = 'test_' + args.results_folder.split('.')[0]
                mating_idx = [torch.unique(in_dict['gt_correspondence']['0-1'].squeeze()[:,0]).to(torch.int64), torch.unique(in_dict['gt_correspondence']['0-1'].squeeze()[:,1]).to(torch.int64)]
                if rrmse > 5:
                    # breakpoint()
                    os.makedirs(save_folder, exist_ok=True)
                    save_pc(f'./{save_folder}/test_'+str(rrmse)+in_dict['filepath'].replace('/','_')+'.pcd', pcds_pred, mating_idx)
                os.makedirs(save_folder+'_total', exist_ok=True)
                save_pc(f'./{save_folder}_total/test_'+str(rrmse)+in_dict['filepath'].replace('/','_')+'.pcd', pcds_pred, mating_idx)
                with open(args.results_folder, "a", encoding="utf-8") as f:
                    f.write(result_str + "\n")

        else:
            visualier = MatchVisualizer()
            for pair_idx in pair_indices:
                pair_idx0, pair_idx1 = pair_idx
                if f'{pair_idx0}-{pair_idx1}' in out_dict.keys():
                    print('------------------------------------')
                    idx_initial_matches = out_dict[f'{pair_idx0}-{pair_idx1}']['idx_initial_matches']
                    if args.registration == 'ransac':
                        if args.vis_inlier_only:
                            idx_initial_matches_inliers = idx_initial_matches[out_dict[f'{pair_idx0}-{pair_idx1}']['corr_inliers'][idx_initial_matches[:,0].cpu(), idx_initial_matches[:,1].cpu()]]
                            print('This is inliers of the initial matches')
                        else:
                            idx_initial_matches_inliers = torch.from_numpy(np.argwhere(out_dict[f'{pair_idx0}-{pair_idx1}']['corr_inliers_upper']))
                            print('This is inliers of entire matches')
                        print(f"{pair_idx0}-{pair_idx1} | len(initial_matches_inliers) : {len(idx_initial_matches_inliers)}")
                    
                    src_pts = in_dict['pcd_t'][pair_idx0].squeeze(0)
                    trg_pts = in_dict['pcd_t'][pair_idx1].squeeze(0)
                    trg_pts_trsfm = _transform(trg_pts, out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_rotat'].squeeze(0).inverse(), out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_trans'], False)

                    ## true_match_vis
                    true_match = []
                    if in_dict['cls_gt'][0][pair_idx0][pair_idx1]:
                        ## true match among the inliers
                        src_corr_pts = src_pts[idx_initial_matches_inliers[:,0], :]
                        trg_corr_pts = trg_pts[idx_initial_matches_inliers[:,1], :]
                        trg_corr_pts_trsfm_gt = _transform(trg_corr_pts, in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}'][0].squeeze(0).inverse(), in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}'][1].squeeze(0), False)
                        dists_gt = torch.norm(src_corr_pts - trg_corr_pts_trsfm_gt, dim=1)
                        true_match = dists_gt < args.distance_threshold
                        if args.visual_check:
                            trg_corr_pts_trsfm = _transform(trg_corr_pts, out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_rotat'].squeeze(0).inverse(), out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_trans'], False)
                            dists_estimation = torch.norm(src_corr_pts - trg_corr_pts_trsfm, dim=1)
                            print(f'estimatedRT_corr_dists: {dists_estimation}')
                        ## upper bound value of the number of true matches for entire match
                        src_pts_gt = in_dict['pcd'][pair_idx0].squeeze(0)
                        trg_pts_gt = in_dict['pcd'][pair_idx1].squeeze(0)
                        dists_entire_match_gt = cdist(src_pts_gt.cpu().numpy(), trg_pts_gt.cpu().numpy())
                        true_match_for_entire_match = dists_entire_match_gt < args.distance_threshold
                        print(f"true match upper bound for entire match : {np.count_nonzero(np.sum(true_match_for_entire_match, axis=0))}")
                        ## upper bound value of the number of true matches for initial match
                        src_initial_match_pts_gt = src_pts_gt[idx_initial_matches[:,0], :]
                        trg_initial_match_pts_gt = trg_pts_gt[idx_initial_matches[:,1], :]
                        dists_initial_match_gt = torch.norm(src_initial_match_pts_gt - trg_initial_match_pts_gt, dim=1)
                        true_match_for_initial_match = dists_initial_match_gt < args.distance_threshold
                        print(f"true match upper bound for initial match : {true_match_for_initial_match.sum().item()}")
                        if args.visual_check:
                            src_initial_match_pts_gt_inlier = src_pts_gt[idx_initial_matches_inliers[:,0], :]
                            trg_initial_match_pts_gt_inlier = trg_pts_gt[idx_initial_matches_inliers[:,1], :]
                            dists_initial_match_gt_inlier = torch.norm(src_initial_match_pts_gt_inlier - trg_initial_match_pts_gt_inlier, dim=1)
                            true_match_for_initial_match_inlier = dists_initial_match_gt_inlier < args.distance_threshold
                            print(f'gtRT_corr_dists: {dists_initial_match_gt_inlier}')
                        
                        if args.visual_check:
                            so = out_dict[f'{pair_idx0}-{pair_idx1}']['src_ori']
                            to = out_dict[f'{pair_idx0}-{pair_idx1}']['trg_ori']
                            to_t = torch.matmul(to, in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}'][0].squeeze(0).cpu())
                            for vec in range(3):
                                so_vec = so[:,vec,:]
                                to_t_vec = to_t[:,vec,:]
                                cos_entire = torch.matmul(so_vec, to_t_vec.T)
                                cos_gt = cos_entire[true_match_for_entire_match]
                                # breakpoint()

                        # score = out_dict[f'{pair_idx0}-{pair_idx1}']['matching_scores_drop']
                        # true_score = score[true_match_for_entire_match]
                        # false_score = score[~true_match_for_entire_match]
                        # print(f"true_score(min): {true_score.min().item()}")
                        # print(f"true_score(max): {true_score.max().item()}")
                        # print(f"true_score(size): {true_score.size()}")
                        # print(f"true_score(mean): {true_score.mean().item()}")
                        # print(f"false_score(min): {false_score.min().item()}")
                        # print(f"false_score(max): {false_score.max().item()}")
                        # print(f"false_score(size): {false_score.size()}")
                        # print(f"false_score(mean): {false_score.mean().item()}")
                        # print(f"true_score(quarter): {torch.quantile(true_score, 0.25)}")
                        # print(f"false_score(quarter): {torch.quantile(false_score, 0.25)}")
                        # print(f"total mean: {score.mean().item()}")

                        # gmm = GaussianMixture(n_components=3, random_state=0)
                        # gmm.fit(score.flatten().cpu().numpy().reshape(-1,1))
                        # gmm_means =gmm.means_.flatten()
                        # gmm_means.sort()
                        # print(f"gmm_means: {gmm_means}")
                        # print(f"2/3 point of GMM: {(gmm_means[1] + gmm_means[2])/2}")
                        # st = false_score.flatten().cpu().numpy()
                        # plt.figure(figsize=(6,4))
                        # sns.kdeplot(st, fill=True, color='skyblue', linewidth=2)
                        # plt.title("Score Distribution")
                        # plt.xlabel("Score")
                        # plt.ylabel("Density")
                        # plt.grid(True)
                        # plt.tight_layout()
                        # # 이미지로 저장
                        # plt.savefig(f"False_score_distribution({pair_idx0}-{pair_idx1}).png", dpi=300)  # dpi는 해상도 (선택사항)
                        # plt.close()


                    rrmse, trmse = _transformation_error(
                        out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_rotat'].squeeze(0).inverse().unsqueeze(0), 
                        in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}'][0].squeeze(0).inverse().unsqueeze(0), 
                        out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_trans'].unsqueeze(0), 
                        in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}'][1]
                        )

                    output_dir = 'visualize_correspondence/output/'
                    if args.ori_threshold != -1.0:
                        output_dir += 'orientation/'
                    if args.gt_normal_threshold != -1.0:
                        output_dir += 'gt_normal/'
                    if args.gt_mating_surface:
                        output_dir += 'gt_mating_surface/'
                    if args.weighted_voting:
                        output_dir += f"weighted_voting_design-{out_dict[f'{pair_idx0}-{pair_idx1}']['weighted_voting_design']}/"
                    output_dir += f"{out_dict[f'{pair_idx0}-{pair_idx1}']['which_transform']}/{out_dict[f'{pair_idx0}-{pair_idx1}']['dist_mat_type']}"
                    if args.distance_threshold != 0.01:
                        output_dir = output_dir + f"/distance_threshold={args.distance_threshold}"
                    if args.score_threshold != 0:
                        output_dir = output_dir + f"/score_threshold={args.score_threshold}"
                    if args.ori_threshold != -1.0:
                        output_dir = output_dir + f"/ori_threshold={args.ori_threshold}"
                    if args.gt_normal_threshold != -1.0:
                        output_dir = output_dir + f"/gt_normal_threshold={args.gt_normal_threshold}"
                    if args.auto_score_threshold:
                        output_dir = output_dir + "/feature_thresholding"
                    # print(f"output dir : {output_dir}")
                    # _, _ = visualier.save_fragments_visualization(src_pts.cpu().numpy(), trg_pts_trsfm.cpu().numpy(), idx_initial_matches_inliers, None, None, f'{idx}||{pair_idx0}-{pair_idx1}', true_match, args.registration, args.initial_match_selection, args.vis_inlier_only, rrmse.item(), trmse.item(), output_dir=output_dir)
                    
                    if args.visual_check and in_dict['cls_gt'][0][pair_idx0][pair_idx1]:
                        src_ori = out_dict[f'{pair_idx0}-{pair_idx1}']['src_ori']
                        trg_ori = out_dict[f'{pair_idx0}-{pair_idx1}']['trg_ori']
                        # breakpoint()
                        visualier.save_fragments_visualization(
                            src_corr_pts.cpu().numpy(), 
                            trg_corr_pts_trsfm.cpu().numpy(), 
                            idx_initial_matches_inliers, 
                            src_ori, #torch.matmul(src_ori, out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_rotat'].squeeze(0).cpu()), 
                            torch.matmul(trg_ori, out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_rotat'].squeeze(0).cpu()), 
                            f"visual_check(estimatedRT)||{pair_idx0}-{pair_idx1}", 
                            true_match, 
                            args.registration, 
                            args.initial_match_selection, 
                            args.vis_inlier_only, 
                            rrmse.item(), 
                            trmse.item(), 
                            output_dir=output_dir
                        )
                        visualier.save_fragments_visualization(
                            src_initial_match_pts_gt_inlier.cpu().numpy(), 
                            trg_initial_match_pts_gt_inlier.cpu().numpy(), 
                            idx_initial_matches_inliers, 
                            src_ori, #torch.matmul(src_ori, in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}'][0].squeeze(0).cpu()),
                            torch.matmul(trg_ori, in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}'][0].squeeze(0).cpu()), 
                            f"visual_check(gtRT)||{pair_idx0}-{pair_idx1}", 
                            true_match_for_initial_match_inlier, 
                            args.registration, 
                            args.initial_match_selection, 
                            args.vis_inlier_only, 
                            rrmse.item(), 
                            trmse.item(), 
                            output_dir=output_dir
                        )
                        trg_ori = torch.matmul(trg_ori, in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}'][0].squeeze(0).cpu())
                        for i in range(3):
                            src_normal = src_ori[:,i,:]
                            trg_normal = trg_ori[:,i,:]
                            src_normal = src_normal / torch.norm(src_normal, dim=1, keepdim=True)
                            trg_normal = trg_normal / torch.norm(trg_normal, dim=1, keepdim=True)
                            cos_sim = torch.matmul(src_normal, trg_normal.T)
                            inlier = cos_sim > args.ori_threshold
                            print(inlier.shape)
                            print(cos_sim[inlier])
                            print(f"gtRT_ori_cos_sim: {np.count_nonzero(np.sum(inlier.cpu().numpy(), axis=0))}")
            
            
                        
    if args.html_vis:
        print(f"{in_dict['n_frac']} : {idx}/{total}")
        print(f"cls_gt: {in_dict['cls_gt']}")
    # else:
    print('====MULTI PART ASSEMBLY RESULTS====')
    print('CRD: ', sum(crd_list)/len(crd_list))
    print('CD: ', sum(cd_list)/len(cd_list))
    print('RRMSE: ', sum(rrmse_list)/len(rrmse_list))
    print('TRMSE: ', sum(trsme_list)/len(trsme_list))
    print('PA(CD): ', sum(pa_list)/len(pa_list)*100)
    print('PA(CRD): ', sum(pa_crd_list)/len(pa_crd_list)*100)
    print(f'strange cases: {strange_cases}')
    print(f'less than 3 inlier case: {less_than_3}')
    
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'Inference time: {(end_time - start_time)/60:.4f} minutes')

    with open(args.results_folder, "a", encoding="utf-8") as f:
        if args.html_vis:
            f.write(f"{in_dict['n_frac']} : {idx}/{total}\n")
            f.write(f"cls_gt: {in_dict['cls_gt']}\n")
        # else:
        f.write("====MULTI PART ASSEMBLY RESULTS====\n")
        f.write(f"CRD: {sum(crd_list)/len(crd_list):.6f}\n")
        f.write(f"CD: {sum(cd_list)/len(cd_list):.6f}\n")
        f.write(f"RRMSE: {sum(rrmse_list)/len(rrmse_list):.6f}\n")
        f.write(f"TRMSE: {sum(trsme_list)/len(trsme_list):.6f}\n")
        f.write(f"PA(CD): {sum(pa_list)/len(pa_list)*100:.2f}\n")
        f.write(f"PA(CRD): {sum(pa_crd_list)/len(pa_crd_list)*100:.2f}\n")
        f.write(f"strange cases: {strange_cases}\n")
        f.write(f"less than 3 inlier case: {less_than_3}\n")

        f.write(f"Inference time: {(end_time - start_time)/60:.4f} minutes\n")
        f.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'synthetic'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--min_part', type=int, default=2)
    parser.add_argument('--max_part', type=int, default=20)

    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--load', type=str, default='')

    parser.add_argument('--scale', type=str, default='ransac', choices=['full', 'small', 'overfitting', 'ransac'])

    # Ablation studies
    parser.add_argument('--model', type=str, default='both', choices=['both', 'shape_only', 'occ_only'])
    parser.add_argument('--backbone', type=str, default='vn_unet', choices=['vn_unet', 'vn_dgcnn', 'unet', 'dgcnn'])
    parser.add_argument('--shape_loss', type=str, default='positive', choices=['positive', 'negative'])
    parser.add_argument('--occ_loss', type=str, default='negative', choices=['positive', 'negative'])
    parser.add_argument('--no_ori', action='store_false')
    parser.add_argument('--attention', type=str, default='channel', choices=['channel', 'none'])
    
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # RANSAC
    parser.add_argument('--html_vis', action = 'store_true')
    parser.add_argument('--vis_inlier_only', action = 'store_true')
    parser.add_argument('--registration', type=str, default='ransac', choices=['wsvd', 'ransac'])
    parser.add_argument('--initial_match_selection', type=str, default='unidirectional_nn', choices=['topk', 'unidirectional_nn', 'injective', 'bijective', 'reciprocal', 'mutual', 'soft'])
    parser.add_argument('--visual_check', action = 'store_true')
    parser.add_argument('--distance_threshold', type=float, default=0.01)
    parser.add_argument('--score_threshold', type=int, default=0)
    parser.add_argument('--auto_score_threshold', action='store_true')
    parser.add_argument('--ori_threshold', type=float, default=-1.0)
    parser.add_argument('--weighted_voting', action='store_true')
    parser.add_argument('--gt_normal_threshold', type=float, default=-1.0)
    parser.add_argument('--gt_mating_surface', action='store_true')
    parser.add_argument('--score_comb', type=str, default='intersection', choices=['sum', 'intersection'])
    parser.add_argument('--topk', type=int, default='1', choices=[1, 2, 3, 128, 0])
    parser.add_argument('--optimal_matching_choice', type=str, default='many-to-many', choices=['many-to-many', 'many-to-one', 'one-to-one'])

    parser.add_argument('--vis_folder', type=str, default='gt_mating')
    parser.add_argument('--results_folder', type=str, default='results.txt')
      
    args = parser.parse_args()

    test(args)