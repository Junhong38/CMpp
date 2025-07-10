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


from model.backbone.vn_dgcnn import EQCNN_equi_unet
from model.backbone.vn_dgcnn import EQCNN_equi

from model.backbone.vn_layers import VNLinear, VNLeakyReLU, VNLinearLeakyReLU, VNLinearNoActivation
from model.loss import CircleLoss, PointMatchingLoss, OrientationLoss
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from model.local_global_registration import LocalGlobalRegistration, WeightedProcrustes

from einops import rearrange, repeat
import torch.nn.functional as F

import open3d as o3d
import random

import pickle
from common.utils import save_pc, knn, get_graph_feature


class EquiAssem_shape(pl.LightningModule):
    def __init__(self, lr, backbone='vn_unet', shape_loss='positive', occ_loss='negative', no_ori=False, attention='channel', visualize=False, debug=False):
        super(EquiAssem_shape, self).__init__()

        self.lr = lr
        self.shape_loss = shape_loss
        self.no_ori = no_ori
        self.attention = attention

        # Output feature dimension of Feature Extractor
        self.feat_dim = 1024

        # Feature Extractor
        if backbone == 'vn_unet':
            self.backbone = EQCNN_equi_unet(feat_dim=self.feat_dim, pooling="mean")
        elif backbone == 'vn_dgcnn':
            self.backbone = EQCNN_equi(feat_dim=self.feat_dim, pooling="mean")
        elif backbone == 'unet':
            raise NotImplementedError("DGCNN_Unet backbone not implemented")
        elif backbone == 'dgcnn':
            raise NotImplementedError("DGCNN backbone not implemented")
        
        # Basis Vector
        self.proj = VNLinear(self.feat_dim//3, 2)

        # Shape Descriptor
        self.shape_mlp = nn.Sequential(nn.Conv1d(1023, 1024, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(1024),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(1024),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(1024),
                                nn.LeakyReLU(negative_slope=0.2),
                                )
        
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
        
        # Objectives
        self.circle_loss = CircleLoss()
        self.matching_loss = PointMatchingLoss()
        self.orientation_loss = OrientationLoss()

        # Weights for losses
        self.c_loss_weight = 1.0
        self.p_loss_weight = 1.0
        self.o_loss_weight = 0 if no_ori else 0.1

        self.debug = debug
        self.visualize = visualize

        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.lr
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16919, eta_min=1e-3) # 16919, 6671
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler}

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
        _, loss_dict = self.forward_pass(in_dict, mode='test')
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

    def forward_pass(self, in_dict, mode):

        out_dict, loss = {}, {}
        src_pcd_raw = in_dict['pcd'][0].squeeze(0)
        trg_pcd_raw = in_dict['pcd'][1].squeeze(0)
        src_pcd = in_dict['pcd_t'][0] # (1, N ,3)
        trg_pcd = in_dict['pcd_t'][1] # (1, M ,3)

        # 1. SO(3)-Equivariant Feature Extractor
        src_equi_feats = self.backbone(src_pcd) # (1, 341, 3, N)
        trg_equi_feats = self.backbone(trg_pcd) # (1, 341, 3, M)

        # 2. Basis Vector Projection 
        src_vecs = self.proj(src_equi_feats).permute(0, 3, 1, 2) # (1, 341, 3, N) -> (1, 2, 3, N) -> (1, N, 2, 3)
        trg_vecs = self.proj(trg_equi_feats).permute(0, 3, 1, 2) # (1, 341, 3, M) -> (1, 2, 3, M) -> (1, M, 2, 3)

        # 3. Gram Schmidt & Cross-product
        src_ori = ortho2rotation(src_vecs) # (1, N, 2, 3) -> (1, N, 3, 3)
        trg_ori = ortho2rotation(trg_vecs) # (1, M, 2, 3) -> (1, M, 3, 3)

        # 4. Invariant Features
        src_inv_feats = torch.matmul(src_equi_feats.permute(0, 3, 1, 2), src_ori.transpose(-2,-1)) # (1, N, 341, 3) x (1, N, 3, 3) -> (1, N, 341, 3)
        trg_inv_feats = torch.matmul(trg_equi_feats.permute(0, 3, 1, 2), trg_ori.transpose(-2,-1)) # (1, M, 341, 3) x (1, M, 3, 3) -> (1, M, 341, 3)
        src_inv_feats = rearrange(src_inv_feats, 'b n c r -> b (c r) n') # (1, N, 341, 3) -> (1, 1023, N)
        trg_inv_feats = rearrange(trg_inv_feats, 'b n c r -> b (c r) n') # (1, M, 341, 3) -> (1, 1023, M)
        
        #### 6. SHAPE DESCRIPTOR ####
        src_shape_feats = self.shape_mlp(src_inv_feats) # (1, 1023, M) -> (1, 512, M)
        trg_shape_feats = self.shape_mlp(trg_inv_feats) # (1, 1023, M) -> (1, 512, M)
        #### 6. SHAPE DESCRIPTOR ####

        # 8. Optimal Transport
        shape_matching_scores = torch.einsum('b c n , b c m -> b n m', src_shape_feats, trg_shape_feats) # (1, N, M)
        shape_matching_scores = shape_matching_scores / src_shape_feats.shape[1] ** 0.5

        matching_scores = self.optimal_transport(shape_matching_scores) # (1, N, M) -> (1, N+1, M+1)
        matching_scores_drop = matching_scores[:,:-1,:-1]

        # 9. Weighted SVD with top-k correspondence selections
        if mode in ['val', 'test']:
            with torch.no_grad():
                src_corr_pts, trg_corr_pts, corr_scores, estimated_transform, pred_corr = self.fine_matching(
                    src_pcd, trg_pcd, matching_scores_drop, k=128)

            out_dict['estimated_rotat'] = estimated_transform[:3, :3].T
            out_dict['estimated_trans'] = -(estimated_transform[:3, :3].inverse() @ -estimated_transform[:3, 3])

        if self.debug:
            out_dict['src_shape_feats'] = src_shape_feats.squeeze(0)
            out_dict['trg_shape_feats'] = trg_shape_feats.squeeze(0)
            out_dict['src_ori'] = src_ori.squeeze(0)
            out_dict['trg_ori'] = trg_ori.squeeze(0)
            out_dict['src_pcd'] = src_pcd.squeeze(0)
            out_dict['trg_pcd'] = trg_pcd.squeeze(0)
            out_dict['src_pcd_raw'] = src_pcd_raw.squeeze(0)
            out_dict['trg_pcd_raw'] = trg_pcd_raw.squeeze(0)
            out_dict['src_gt_rot'] = in_dict['gt_rotat'][0].squeeze(0)
            out_dict['trg_gt_rot'] = in_dict['gt_rotat'][1].squeeze(0)
            out_dict['gt_correspondence'] = in_dict['gt_correspondence'].squeeze(0)
            out_dict['pred_corr'] = pred_corr
            with open(f'./pickle/shape/{in_dict["eval_idx"].item()}_debug.pickle', 'wb') as f:
                pickle.dump(out_dict, f)
        
        # 10. Calculate Loss
        gt_corr = in_dict['gt_correspondence'].squeeze(0)
        
        # 9-1. circle loss
        loss['c_loss'] = self.circle_loss(src_pcd_raw, trg_pcd_raw, src_shape_feats.transpose(-2,-1), trg_shape_feats.transpose(-2,-1), gt_corr)

        # 9-2 point matching loss
        loss['p_loss'] = self.matching_loss(matching_scores, gt_corr, src_pcd_raw, trg_pcd_raw)

        # 9-3. orientation loss
        loss['o_loss'] = self.orientation_loss(src_ori, trg_ori, gt_corr, in_dict['gt_rotat'])
        
        # 9-4. final loss
        loss['loss'] = self.c_loss_weight * loss['c_loss'] + self.p_loss_weight * loss['p_loss'] + self.o_loss_weight * loss['o_loss']
        out_dict.update(loss)
        
        # 10. Evaluation
        if mode in ['val', 'test']:
            eval_dict = self.evaluate_prediction(in_dict, out_dict, gt_corr)
            loss.update(eval_dict)
        
        # in training we log for every step
        if mode == 'train':
            log_dict = {f'{mode}/{k}': v.item() for k, v in loss.items()}
            self.log_dict(log_dict, logger=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True, batch_size=1)
        
        return out_dict, loss

    @torch.no_grad()
    def evaluate_prediction(self, in_dict, out_dict, gt_corr, multi_part=False):

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

        if self.visualize:
            pcds_pred.append(pcds_pred[0][gt_corr[:,0]])
            pcds_pred.append(pcds_pred[1][gt_corr[:,1]])
            pcds_grtr.append(pcds_grtr[0][gt_corr[:,0]])
            pcds_grtr.append(pcds_grtr[1][gt_corr[:,1]])
            save_pc(f"./vis_fantastic/o_loss{round(out_dict['o_loss'].item(),2)}_cd{round(eval_result['cd'].item(),2)}_crd{round(eval_result['crd'].item(),2)}_c_loss{round(out_dict['c_loss'].item(),2)}_occ_loss{round(out_dict['occ_loss'].item(),2)}_rrmse{round(eval_result['rrmse'].item(),1)}_pred.pcd", pcds_pred)
            save_pc(f"./vis_fantastic/o_loss{round(out_dict['o_loss'].item(),2)}_cd{round(eval_result['cd'].item(),2)}_crd{round(eval_result['crd'].item(),2)}_c_loss{round(out_dict['c_loss'].item(),2)}_occ_loss{round(out_dict['occ_loss'].item(),2)}_rrmse{round(eval_result['rrmse'].item(),1)}_grtr.pcd", pcds_grtr)
        
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
