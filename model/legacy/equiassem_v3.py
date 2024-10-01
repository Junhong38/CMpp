from functools import reduce
from operator import add
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from common.rotation import ortho2rotation
from chamfer_distance import ChamferDistance as chamfer_dist

from model.backbone.vn_dgcnn import EQCNN_equi
from model.backbone.vn_layers import VNLinear, VNLeakyReLU, VNLinearLeakyReLU, VNLinearNoActivation
from model.loss import CircleLoss, PointMatchingLoss, OrientationLoss, OccupancyLossCosineDistance
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from model.local_global_registration import LocalGlobalRegistration, WeightedProcrustes

from einops import rearrange, repeat
import torch.nn.functional as F

import open3d as o3d
import random

import pickle

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

class EquiAssem_v3(pl.LightningModule):
    def __init__(self, lr, backbone='eqcnn', visualize=False):
        super(EquiAssem_v3, self).__init__()

        self.lr = lr

        # Output feature dimension of Feature Extractor
        self.feat_dim = 1024

        # Feature Extractor
        self.backbone = EQCNN_equi(feat_dim=self.feat_dim, pooling="mean")
        
        # Basis Vector
        self.proj = VNLinear(self.feat_dim//3, 2)

        self.inv_mlp = nn.Sequential(nn.InstanceNorm1d(self.feat_dim//3*3),
                                    nn.LeakyReLU(),
                                    nn.Conv1d(self.feat_dim//3*3, self.feat_dim//3*3, kernel_size=1))

        self.global_mlp = nn.Sequential(nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1),
                                nn.InstanceNorm1d(self.feat_dim//2),
                                nn.Tanh())

        self.matching_mlp = nn.Sequential(nn.Conv1d(self.feat_dim//3*3, self.feat_dim//3*3, kernel_size=1),
                                nn.InstanceNorm1d(self.feat_dim//3*3),
                                nn.LeakyReLU())

        # Optimal Transport
        self.optimal_transport = LearnableLogOptimalTransport(num_iterations=100)

        # LGR
        self.fine_matching = LocalGlobalRegistration(
            k=3,
            acceptance_radius=0.1,
            mutual=True,
            confidence_threshold=0.05,
            use_dustbin=False,
            use_global_score=False,
            correspondence_threshold=3,
            correspondence_limit=None,
            num_refinement_steps=5,
        )
        
        self.circle_loss = CircleLoss()
        self.matching_loss = PointMatchingLoss()
        self.orientation_loss = OrientationLoss()
        self.occupancy_loss = OccupancyLossCosineDistance()

        # Weights for losses
        self.c_loss_weight = 1. 
        self.p_loss_weight = 1. 
        self.o_loss_weight = 0.1
        self.occ_loss_weight = 0.1

        # Random rotation for equivariance checking
        rotation_matrix = torch.tensor([[0.26726124, -0.57735027,  0.77151675],
                  [0.53452248, -0.57735027, -0.6172134],
                  [0.80178373,  0.57735027,  0.15430335]], dtype=torch.float64)
        self.R = rotation_matrix.unsqueeze(0)
        
        self.debug = False
        self.visualize = visualize

        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.lr
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.)
        return optimizer

    
    def training_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(
            in_dict, mode='train')
        if loss_dict['loss']==0.: return None
        return loss_dict['loss']
    
    def validation_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(
            in_dict, mode='val')
        self.validation_step_outputs.append(loss_dict)
        return loss_dict

    def on_validation_epoch_end(self):    
        # avg_loss among all data
        losses = {
            f'val/{k}': torch.stack([output[k] for output in self.validation_step_outputs])
            for k in self.validation_step_outputs[0].keys()
        }
        avg_loss = {k: (v).sum() / v.size(0) for k, v in losses.items()}
        self.log_dict(avg_loss, sync_dist=True, batch_size=1)
        self.validation_step_outputs.clear()

    def test_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='test', visualize=self.visualize)
        self.test_step_outputs.append(loss_dict)
        return loss_dict

    def on_test_epoch_end(self):    
        # avg_loss among all data
        losses = {
            f'val/{k}': torch.stack([output[k] for output in self.test_step_outputs])
            for k in self.test_step_outputs[0].keys()
        }
        avg_loss = {k: (v).sum() / v.size(0) for k, v in losses.items()}
        print('; '.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))
        # this is a hack to get results outside `Trainer.test()` function
        self.test_results = avg_loss
        self.test_step_outputs.clear()

    def forward_pass(self, in_dict, mode, visualize=False):

        out_dict, loss = {}, {}
        src_pcd_raw = in_dict['pcd'][0].squeeze(0)
        trg_pcd_raw = in_dict['pcd'][1].squeeze(0)
        src_pcd = in_dict['pcd_t'][0] # (1, N ,3)
        trg_pcd = in_dict['pcd_t'][1] # (1, M ,3)

        # 1. SO(3)-Equivariant Feature Extractor
        src_equi_feats = self.backbone(src_pcd)
        trg_equi_feats = self.backbone(trg_pcd)

        # 2. Basis Vector Projection 
        src_vecs = self.proj(src_equi_feats).permute(0, 3, 1, 2) # (1, C//3, 3, N) -> (1, 2, 3, N) -> (1, N, 2, 3)
        trg_vecs = self.proj(trg_equi_feats).permute(0, 3, 1, 2) # (1, C//3, 3, M) -> (1, 2, 3, M) -> (1, M, 2, 3)

        # 3. Gram Schmidt & Cross-product
        src_ori = ortho2rotation(src_vecs) # (1, N, 2, 3) -> (1, N, 3, 3)
        trg_ori = ortho2rotation(trg_vecs) # (1, M, 2, 3) -> (1, M, 3, 3)

        # 4. Invariant Features
        src_inv_feats = torch.matmul(src_equi_feats.permute(0, 3, 1, 2), src_ori.transpose(-2,-1)) # (1, N, C//3, 3) x (1, N, 3, 3) -> (1, N, C//3, 3)
        trg_inv_feats = torch.matmul(trg_equi_feats.permute(0, 3, 1, 2), trg_ori.transpose(-2,-1)) # (1, M, C//3, 3) x (1, M, 3, 3) -> (1, M, C//3, 3)
        src_inv_feats = rearrange(src_inv_feats, 'b n c r -> b (c r) n') # (1, N, C//3, 3) -> (1, C, N)
        trg_inv_feats = rearrange(trg_inv_feats, 'b n c r -> b (c r) n') # (1, M, C//3, 3) -> (1, C, M)

        # 5. Divide shape and occupancy descriptors
        src_inv_feats = self.inv_mlp(src_inv_feats)
        trg_inv_feats = self.inv_mlp(trg_inv_feats)
        C = src_inv_feats.size(1)//2
        src_shape_feats, src_occ_feats = src_inv_feats[:, :C], src_inv_feats[:, C:]
        trg_shape_feats, trg_occ_feats = trg_inv_feats[:, :C], trg_inv_feats[:, C:]

        src_occ_feats = self.global_mlp(src_occ_feats)
        trg_occ_feats = self.global_mlp(trg_occ_feats)

        # 7. Combine Shape and Occupancy Descriptors
        src_matching_feature = self.matching_mlp(torch.cat([src_shape_feats, src_occ_feats], dim=1))
        trg_matching_feature = self.matching_mlp( torch.cat([trg_shape_feats, trg_occ_feats], dim=1))

        # 8. Optimal Transport
        matching_scores = torch.einsum('b c n , b c m -> b n m', src_matching_feature, trg_matching_feature) # (1, N, M)
        matching_scores = matching_scores / src_matching_feature.shape[1] ** 0.5 # (1, N, M)
        matching_scores = self.optimal_transport(matching_scores) # (1, N, M) -> (1, N+1, M+1)
        matching_scores_drop = matching_scores[:,:-1,:-1] # (1, N+1, M+1) -> (1, N, M)

        # 9. Weighted SVD with top-k correspondence selections
        with torch.no_grad():
            src_corr_pts, trg_corr_pts, corr_scores, estimated_transform = self.fine_matching(
                src_pcd, trg_pcd, matching_scores_drop, k=128)

        out_dict['estimated_rotat'] = estimated_transform[:3, :3].T
        out_dict['estimated_trans'] = -(estimated_transform[:3, :3].inverse() @ -estimated_transform[:3, 3])

        # 10. Calculate Loss
        gt_corr = in_dict['gt_correspondence'].squeeze(0)
        
        # 9-1. circle loss
        loss['c_loss'], loss['FMR'] = self.circle_loss(src_pcd_raw, trg_pcd_raw, src_shape_feats.transpose(-2,-1), trg_shape_feats.transpose(-2,-1), gt_corr)

        # 9-2 point matching loss
        loss['p_loss'] = self.matching_loss(matching_scores, gt_corr, src_pcd_raw, trg_pcd_raw)

        # 9-3. orientation loss
        loss['o_loss'] = self.orientation_loss(src_ori, trg_ori, gt_corr, in_dict['gt_rotat'])
        
        # 9-4. occupancy loss
        loss['occ_loss']= self.occupancy_loss(src_occ_feats, -trg_occ_feats, gt_corr)

        # 9-4. final loss
        loss['loss'] = self.c_loss_weight * loss['c_loss'] + self.p_loss_weight * loss['p_loss'] + self.o_loss_weight * loss['o_loss'] +  self.occ_loss_weight * loss['occ_loss']
        out_dict.update(loss)

        # 10. Evaluation
        eval_dict = self.evaluate_prediction(in_dict, out_dict, visualize=visualize)
        loss.update(eval_dict)

        # in training we log for every step
        if mode == 'train':
            log_dict = {f'{mode}/{k}': v.item() for k, v in loss.items()}
            self.log_dict(log_dict, logger=True, sync_dist=False, rank_zero_only=True, on_step=False, on_epoch=True, batch_size=1)
        
        return out_dict, loss

    @torch.no_grad()
    def evaluate_prediction(self, in_dict, out_dict, visualize=False, multi_part=False):

        # Init return buffer
        eval_result = {}
        
        pred_relative_trsfm = out_dict['estimated_rotat'], out_dict['estimated_trans'] 
        grtr_relative_trsfm = [x.squeeze(0) for x in in_dict['relative_trsfm']['0-1']]
        src_pcd, trg_pcd = [x.squeeze(0) for x in in_dict['pcd_t']]
        is_trg_larger = self._is_trg_larger(src_pcd, trg_pcd)

        # Assemble using prediction, pseudo-gt, and ground-truth
        assm_pred, pcds_pred = self._pairwise_mating(src_pcd, trg_pcd, pred_relative_trsfm[0], pred_relative_trsfm[1], is_trg_larger)
        assm_grtr, pcds_grtr = self._pairwise_mating(src_pcd, trg_pcd, grtr_relative_trsfm[0], grtr_relative_trsfm[1], is_trg_larger)
        
        # (a) Compute CD between prediction & ground-truth
        eval_result['cd'] = self._chamfer_distance(assm_pred, assm_grtr, is_trg_larger)

        # (b) Compute MSE between prediction & ground-truth for rotation (in degree) and translation
        eval_result['rrmse'], eval_result['trmse'] = self._transformation_error(pred_relative_trsfm, grtr_relative_trsfm, multi_part)
        
        # (c) Compute CoRrespondence Distance (CRD) betwween prediction & ground-truth
        eval_result['crd'] = self._correspondence_distance(assm_pred, assm_grtr, is_trg_larger)

        if visualize:
            save_pc(f"./vis/o_loss{round(out_dict['o_loss'].item(),2)}_occ_loss{round(out_dict['occ_loss'].item(),2)}_rrmse{round(eval_result['rrmse'].item(),1)}_crd{round(eval_result['crd'].item(),2)}_pred.pcd", pcds_pred)
            save_pc(f"./vis/o_loss{round(out_dict['o_loss'].item(),2)}_occ_loss{round(out_dict['occ_loss'].item(),2)}_rrmse{round(eval_result['rrmse'].item(),1)}_crd{round(eval_result['crd'].item(),2)}_grtr.pcd", pcds_grtr)

        return eval_result
    
    def _is_trg_larger(self, src_pcd, trg_pcd):
        src_volume = (src_pcd.max(dim=0)[0] - src_pcd.min(dim=0)[0]).prod(dim=0)
        trg_volume = (trg_pcd.max(dim=0)[0] - trg_pcd.min(dim=0)[0]).prod(dim=0)

        return src_volume < trg_volume
    
    def _pairwise_mating(self, src_pcd, trg_pcd, rotat, trans, is_trg_larger):
        pcd_t = []
        if is_trg_larger:
            src_pcd_t = self._transform(src_pcd.squeeze(0), rotat, -trans, True)
            pcd_t = [src_pcd_t, trg_pcd.squeeze(0)]
        else:
            trg_pcd_t = self._transform(trg_pcd.squeeze(0), rotat.T, trans, False)
            pcd_t = [src_pcd.squeeze(0), trg_pcd_t]
        return torch.cat(pcd_t, dim=0), pcd_t
    
    def _transform(self, pcd, rotat=None, trans=None, rotate_first=True):
        if rotat == None: rotat = torch.eye(3, 3)
        if trans == None: trans = torch.zeros(3)

        rotat = rotat.to(pcd.device)
        trans = trans.to(pcd.device)

        if rotate_first:
            return torch.einsum('x y, n y -> n x', rotat, pcd) + trans
        else:
            return torch.einsum('x y, n y -> n x', rotat, pcd + trans)
    
    def _correspondence_distance(self, assm1, assm2, is_trg_larger, scaling=100):
        corr_dist = (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling
        return corr_dist

    def _chamfer_distance(self, assm1, assm2, is_trg_larger, scaling=1000):
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
        cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling
        return cd
    
    def _transformation_error(self, trnsf1, trnsf2, multi_part, rrmse_scaling=100):
        if multi_part:
            rotat1, trans1 = trnsf1
            rotat2, trans2 = trnsf2
        else:
            rotat1, trans1 = [trnsf1[0]], [trnsf1[1]]
            rotat2, trans2 = [trnsf2[0]], [trnsf2[1]]
        rrmse, trmse = 0., 0.
        for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
            r1_deg = torch.tensor(Rotation.from_matrix(r1.cpu()).as_euler('xyz', degrees=True))
            r2_deg = torch.tensor(Rotation.from_matrix(r2.cpu()).as_euler('xyz', degrees=True))
            diff1 = (r1_deg - r2_deg).abs()
            diff2 = 360. - (r1_deg - r2_deg).abs()
            diff = torch.minimum(diff1, diff2)
            rrmse += diff.pow(2).mean().pow(0.5)
            trmse += (t1 - t2).pow(2).mean().pow(0.5) * rrmse_scaling
        div = len(rotat1) if multi_part else 1
        return rrmse / div, trmse / div