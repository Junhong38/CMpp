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

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset import GADataset
from common import utils
import open3d as o3d

from model.equiassem import EquiAssem

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

import itertools
import gtsam
import numpy as np
import random
import open3d as o3d
import trimesh

from chamfer_distance import ChamferDistance as chamfer_dist

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

def save_mesh(in_dict):
    base_path = '../../data/bbad_v2/'
    obj_paths = [os.path.join(base_path+in_dict['filepath'], x) for x in os.listdir(base_path+in_dict['filepath'])]
    mesh = [trimesh.load_mesh(x) for x in obj_paths]
    
    # Create directories if they don't exist
    save_dir = os.path.join('vis', 'everyday_onetime_only_ori_full')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{in_dict["eval_idx"][0].item()}_{len(in_dict["gt_rotat"])}part_'+in_dict['filepath'].replace('/','_'))
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
        if part_cd < 0.01: success += 1
    return success / len(assm_pts1)

def _part_accuracy_crd(assm_pts1, assm_pts2, scaling=100):
    success = 0
    for pred_pts, gt_pts in zip(assm_pts1, assm_pts2):
        part_crd = _correspondence_distance(pred_pts, gt_pts, 1)
        if part_crd < 0.1: success += 1
    return success / len(assm_pts1)

def test(args):
    # Model initialization
    utils.fix_randseed(0)
    model = EquiAssem(lr=args.lr,
                        backbone=args.backbone,
                        shape_loss=args.shape_loss, 
                        occ_loss=args.occ_loss, 
                        no_ori=args.no_ori,
                        attention=args.attention,
                        visualize=args.visualize,
                        debug=args.debug)
    print(model)

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
    for idx, in_dict in enumerate(dataloader_test):
        # 1. Network forward pass: Pairwise matching & assembly
        in_dict = utils.to_cuda(in_dict)
        
        _ = model.forward_pass(in_dict, 'test')
        torch.cuda.empty_cache(); gc.collect()

        # pair_indices = list(itertools.permutations([i for i in range(in_dict['n_frac'])], 2))
        # out_dict, corr_scores = {}, {}
        # for pair_idx in pair_indices:
        #     # Initialize pairwise input
        #     pair_idx0, pair_idx1 = pair_idx
        #     in_dict_pair = {
        #         'filepath': in_dict['filepath'], 'obj_class': in_dict['obj_class'],
        #         'pcd_t': [in_dict['pcd_t'][pair_idx0], in_dict['pcd_t'][pair_idx1]],
        #         'pcd': [in_dict['pcd'][pair_idx0], in_dict['pcd'][pair_idx1]],
        #         'n_frac': 2, # Pairwise matching
        #         'gt_trans': [in_dict['gt_trans'][pair_idx0], in_dict['gt_trans'][pair_idx1]],
        #         'gt_rotat': [in_dict['gt_rotat'][pair_idx0], in_dict['gt_rotat'][pair_idx1]],
        #         'gt_rotat_inv': [in_dict['gt_rotat_inv'][pair_idx0], in_dict['gt_rotat_inv'][pair_idx1]],
        #         'gt_trans_inv': [in_dict['gt_trans_inv'][pair_idx0], in_dict['gt_trans_inv'][pair_idx1]],
        #         'relative_trsfm': {f'{pair_idx0}-{pair_idx1}': in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}']},
        #         'gt_normals': [in_dict['gt_normals'][pair_idx0], in_dict['gt_normals'][pair_idx1]],
        #         'eval_idx': f'{idx}|{pair_idx0}-{pair_idx1}'
        #     }
        #     # Forward pass
        #     _ = model.forward_pass(in_dict_pair, 'test')
        #     # out_dict[f'{pair_idx0}-{pair_idx1}'] = model.forward_pass(in_dict_pair, 'test')
        #     torch.cuda.empty_cache(); gc.collect()
        
    #     # 2. Pose Graph Optimization
    #     params = gtsam.ShonanAveragingParameters3(gtsam.LevenbergMarquardtParams.CeresDefaults())
    #     factors = gtsam.BetweenFactorPose3s()

    #     # 2-1. Add factors(relative transformations)
    #     for pair_idx in pair_indices:
    #         pair_idx0, pair_idx1 = pair_idx
    #         relative_rotat = Rotation.from_matrix(out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_rotat'].cpu().numpy()).as_quat()
    #         relative_trans = -out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_trans'].cpu().numpy()
    #         pose = gtsam.Pose3(gtsam.Rot3.Quaternion(relative_rotat[3], relative_rotat[0], relative_rotat[1], relative_rotat[2]), gtsam.Point3(relative_trans))
    #         # Load matching score
    #         score = (torch.pow(out_dict[f'{pair_idx0}-{pair_idx1}']['corr_scores'], 2)).detach().cpu().mean()
    #         factors.append(gtsam.BetweenFactorPose3(pair_idx0, pair_idx1, pose, gtsam.noiseModel.Diagonal.Information((1/score) * np.eye(6))))

    #     # 2-2. Run shonan averaging
    #     sa3 = gtsam.ShonanAveraging3(factors, params)
    #     initial = sa3.initializeRandomly()
    #     pMax = 20
    #     shonan_fail = False
    #     while True:
    #         pMax += 20
    #         if pMax == 60:
    #             shonan_fail = True; print("shonan failed")
    #             break
    #         try: 
    #             abs_rotat, _ = sa3.run(initial, 3, pMax)
    #             break
    #         except RuntimeError as e:
    #             print(f"An error occurred during Shonan::run: with pMax {pMax}")
    #             continue
            
    #     # Align predicted rotation to anchor fracture
    #     anchor_idx = in_dict['anchor_idx']
    #     if not shonan_fail:
    #         aligned_pred_rotat, aligned_pred_trans = [], []
    #         abs_anchor_R = abs_rotat.atRot3(anchor_idx)
    #         # Align rotations
    #         for j in range(abs_rotat.size()):
    #             aligned_pred_rotat.append(torch.tensor(abs_anchor_R.between(abs_rotat.atRot3(j)).matrix()).to(torch.float32).cuda())
    #     else:
    #         aligned_pred_rotat = []
    #         for i in range(0, in_dict['n_frac']):
    #             if i == anchor_idx: aligned_pred_rotat.append(torch.eye(3).to(torch.float32).cuda())
    #             else: aligned_pred_rotat.append(out_dict[f'{anchor_idx}-{i}']['estimated_rotat'].squeeze(0))
        
    #     # Align predicted rotation to anchor fracture
    #     for i in range(0, in_dict['n_frac']):
    #         if i == anchor_idx: aligned_pred_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
    #         else: aligned_pred_trans.append(out_dict[f'{anchor_idx}-{i}']['estimated_trans'])
    #         # else: aligned_pred_trans.append(torch.tensor((abs_trans.atPoint3(i) - abs_trans.atPoint3(anchor_idx))).to(torch.float32).cuda())

    #     # Align GT transformation to anchor fracture
    #     aligned_gt_rotat, aligned_gt_trans = [], []
    #     for i in range(0, in_dict['n_frac']):
    #         if i == anchor_idx:
    #             aligned_gt_rotat.append(torch.eye(3).to(torch.float32).cuda())
    #             aligned_gt_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
    #         else:
    #             aligned_gt_rotat.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][0].squeeze(0))
    #             aligned_gt_trans.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][1].squeeze(0))

    #     # Save aligned transformations
    #     in_dict['aligned_gt_trans'] = aligned_gt_trans
    #     in_dict['aligned_gt_rotat'] = aligned_gt_rotat
    #     out_dict['aligned_pred_trans'] = aligned_pred_trans
    #     out_dict['aligned_pred_rotat'] = aligned_pred_rotat
    #     assm_pred, pcds_pred = _multi_part_assemble(in_dict['pcd_t'], aligned_pred_rotat, aligned_pred_trans)
    #     assm_grtr, pcds_grtr = _multi_part_assemble(in_dict['pcd_t'], aligned_gt_rotat, aligned_gt_trans)

    #     cd = _chamfer_distance(assm_pred, assm_grtr).item()
    #     crd = _correspondence_distance(assm_pred, assm_grtr).item()
    #     rrmse, trmse = _transformation_error(aligned_pred_rotat, aligned_gt_rotat, aligned_pred_trans, aligned_gt_trans)
    #     rrmse, trmse = rrmse.item(), trmse.item()
    #     pa = _part_accuracy(pcds_pred, pcds_grtr)
    #     pa_crd = _part_accuracy_crd(pcds_pred, pcds_grtr)

        if args.visualize: save_mesh(in_dict)
        
    #     crd_list.append(crd)
    #     cd_list.append(cd)
    #     rrmse_list.append(rrmse)
    #     trsme_list.append(trmse)
    #     pa_list.append(pa)
    #     pa_crd_list.append(pa_crd)

    #     print(f'{idx}/{total} | #-Part: {len(pcds_pred)} | CRD: {round(crd,2)} | CD: {round(cd,2)} | RRMSE: {round(rrmse,2)} | TRMSE: {round(trmse,2)} | PA(cd): {round(pa,2)} | PA(crd): {round(pa_crd,2)}')
    #     # save_pc(f"./vis/mpa_everyday_vis/{len(pcds_pred)}part_crd{round(crd,2)}_rrmse{round(rrmse,1)}_trmse{round(trmse,2)}_cd{round(cd,2)}_pred_{in_dict['filepath'].replace('/','_')}.pcd", pcds_pred)
    #     # save_pc(f"./vis/mpa_everyday_vis/{len(pcds_pred)}part_crd{round(crd,2)}_rrmse{round(rrmse,1)}_trmse{round(trmse,2)}_cd{round(cd,2)}_grtr_{in_dict['filepath'].replace('/','_')}.pcd", pcds_grtr)
    
    # print('====MULTI PART ASSEMBLY RESULTS====')
    # print('CRD: ', sum(crd_list)/len(crd_list))
    # print('CD: ', sum(cd_list)/len(cd_list))
    # print('RRMSE: ', sum(rrmse_list)/len(rrmse_list))
    # print('TRMSE: ', sum(trsme_list)/len(trsme_list))
    # print('PA(CD): ', sum(pa_list)/len(pa_list))
    # print('PA(CRD): ', sum(pa_crd_list)/len(pa_crd_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../data/bbad_v2')
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

    parser.add_argument('--scale', type=str, default='full', choices=['full', 'small', 'overfitting'])

    # Ablation studies
    parser.add_argument('--model', type=str, default='both', choices=['both', 'shape_only', 'occ_only'])
    parser.add_argument('--backbone', type=str, default='vn_unet', choices=['vn_unet', 'vn_dgcnn', 'unet', 'dgcnn'])
    parser.add_argument('--shape_loss', type=str, default='positive', choices=['positive', 'negative'])
    parser.add_argument('--occ_loss', type=str, default='negative', choices=['positive', 'negative'])
    parser.add_argument('--no_ori', action='store_true')
    parser.add_argument('--attention', type=str, default='channel', choices=['channel', 'none'])

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')
      
    args = parser.parse_args()

    test(args)