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

from scipy.spatial.distance import cdist

from common.rotation import ortho2rotation
from chamfer_distance import ChamferDistance as chamfer_dist


from model.backbone.vn_dgcnn import EQCNN_equi_unet
from model.backbone.vn_dgcnn import EQCNN_equi

from model.backbone.vn_layers import VNLinear, VNLeakyReLU, VNLinearLeakyReLU, VNLinearNoActivation
from model.loss import CircleLoss, PointMatchingLoss, OrientationLoss
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from model.local_global_registration import LocalGlobalRegistration, WeightedProcrustes

from model.ransac import ransac_rigid
from model.match_selection import topk_matching, unidirectional_nn_matching, injective_matching, bijective_matching, mutual_topk_matching, soft_topk_matching
from data.utils import get_correspondences

from einops import rearrange, repeat
import torch.nn.functional as F

import open3d as o3d
import random

import pickle
from common.utils import save_pc, knn, get_graph_feature, save_ori, SVD, save_normal

import os, trimesh
import math

class ChannelAttentionModule(nn.Module):
    """ this function is used to achieve the channel attention module in CBAM paper"""
    def __init__(self, in_dim=1024, out_dim=1024, ratio=4):
        super(ChannelAttentionModule, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=out_dim // ratio, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels= out_dim // ratio, out_channels=out_dim, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        out1 = torch.mean(x, dim=-1, keepdim=True)  # 1, c, 1
        out1 = self.mlp(out1) # 1, c, 1

        out2 = nn.AdaptiveMaxPool1d(1)(x) # 1, c, 1
        out2 = self.mlp(out2) # 1, c, 1
        
        out = F.normalize(out1+out2, p=2, dim=1)
        attention = self.sigmoid(out)
        
        return attention

class EquiAssem(pl.LightningModule):
    def __init__(self, lr, backbone='vn_unet', shape_loss='positive', occ_loss='negative', no_ori=False, attention='channel', visualize=False, debug=False):
            # self, 
            # lr, backbone='vn_unet', shape_loss='positive', occ_loss='negative', 
            # no_ori=False, attention='channel', registration='ransac', match_selection_str='injective', 
            # visualize=False, debug=False, inlier_threshold=0.01, score_threshold=0, 
            # auto_score_threshold=False, ori_threshold=-1, weighted_voting=False,
            # gt_normal_threshold=-1, gt_mating_surface=False,
            # score_comb='intersection'
            # ):
        super(EquiAssem, self).__init__()

        self.lr = lr
        self.shape_loss = shape_loss
        self.occ_loss = occ_loss
        self.no_ori = no_ori
        self.attention = attention
        
        # self.registration = registration
        # self.match_selection = match_selection_str

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
        # self.proj = VNLinear(self.feat_dim//3, 2)
        self.proj = VNLinear(self.feat_dim, 1)

        # # Channel Attention
        # if attention == 'channel':
        #     self.c_attn = ChannelAttentionModule(self.feat_dim, self.feat_dim, ratio=4)
        
        # # Shape Descriptor
        # self.shape_mlp = nn.Sequential(nn.Conv1d(self.feat_dim, self.feat_dim//2, kernel_size=1, bias=False),
        #                         nn.InstanceNorm1d(self.feat_dim//2),
        #                         nn.LeakyReLU(negative_slope=0.2),
        #                         nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
        #                         nn.InstanceNorm1d(self.feat_dim//2),
        #                         nn.LeakyReLU(negative_slope=0.2),
        #                         nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
        #                         nn.InstanceNorm1d(self.feat_dim//2),
        #                         nn.LeakyReLU(negative_slope=0.2),
        #                         )
        
        # # Occupancy Descriptor
        # self.occ_mlp = nn.Sequential(nn.Conv1d(self.feat_dim, self.feat_dim//2, kernel_size=1, bias=False),
        #                         nn.InstanceNorm1d(self.feat_dim//2),
        #                         nn.LeakyReLU(negative_slope=0.2),
        #                         nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
        #                         nn.InstanceNorm1d(self.feat_dim//2),
        #                         nn.LeakyReLU(negative_slope=0.2),
        #                         nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
        #                         nn.InstanceNorm1d(self.feat_dim//2),
        #                         nn.Tanh()
        #                         )
        
        # # Optimal Transport
        # self.optimal_transport = LearnableLogOptimalTransport(num_iterations=100)

        # # LGR
        # self.fine_matching = LocalGlobalRegistration(
        #     k=3,
        #     acceptance_radius=0.1,
        #     mutual=True,
        #     confidence_threshold=0.05,
        #     use_dustbin=False,
        #     use_global_score=False,
        #     correspondence_threshold=3,
        #     correspondence_limit=None,
        #     num_refinement_steps=5,
        # )
        
        # Objectives
        # self.circle_loss = CircleLoss()
        # self.matching_loss = PointMatchingLoss()
        self.orientation_loss = OrientationLoss()
        # self.occupancy_loss = CircleLoss()

        # # Weights for losses
        # self.c_loss_weight = 0.5 
        # self.occ_loss_weight = 0.5
        # self.p_loss_weight = 1.0
        # self.o_loss_weight = 0 if no_ori else 0.1

        self.debug = debug
        self.visualize = visualize

        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.inlier_threshold = inlier_threshold
        # self.score_threshold = score_threshold
        # self.auto_score_threshold = auto_score_threshold
        # self.ori_threshold = ori_threshold
        # self.weighted_voting = weighted_voting
        # self.gt_normal_threshold = gt_normal_threshold
        # self.gt_mating_surface = gt_mating_surface
        # self.score_comb = score_comb
        
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

    # @torch.no_grad()
    def forward_pass(self, in_dict, mode):

        out_dict, loss = {}, {}
        src_pcd_raw = in_dict['pcd'][0].squeeze(0)
        trg_pcd_raw = in_dict['pcd'][1].squeeze(0)
        src_pcd = in_dict['pcd_t'][0] # (1, N ,3)
        trg_pcd = in_dict['pcd_t'][1] # (1, M ,3)
        
        # 1. SO(3)-Equivariant Feature Extractor
        src_equi_feats = self.backbone(src_pcd) # (1, 341, 3, N)
        trg_equi_feats = self.backbone(trg_pcd) # (1, 341, 3, M)

        out_dict['src_equi_feats'] = src_equi_feats
        out_dict['trg_equi_feats'] = trg_equi_feats

        # 2. Basis Vector Projection 
        src_vecs = self.proj(src_equi_feats).permute(0, 3, 1, 2) # (1, 341, 3, N) -> (1, 1, 3, N) -> (1, N, 1, 3)
        trg_vecs = self.proj(trg_equi_feats).permute(0, 3, 1, 2) # (1, 341, 3, M) -> (1, 1, 3, M) -> (1, M, 1, 3)

        # 3. Gram Schmidt & Cross-product
        # src_ori = ortho2rotation(src_vecs) # (1, N, 2, 3) -> (1, N, 3, 3)
        # trg_ori = ortho2rotation(trg_vecs) # (1, M, 2, 3) -> (1, M, 3, 3)

        # 3. Normalization
        eps = 1e-8
        src_ori = src_vecs / (torch.norm(src_vecs, dim=3, keepdim=True) + eps)
        trg_ori = trg_vecs / (torch.norm(trg_vecs, dim=3, keepdim=True) + eps)

        # # 4. Invariant Features
        # src_inv_feats = torch.matmul(src_equi_feats.permute(0, 3, 1, 2), src_ori.transpose(-2,-1)) # (1, N, 341, 3) x (1, N, 3, 1) -> (1, N, 341, 1)
        # trg_inv_feats = torch.matmul(trg_equi_feats.permute(0, 3, 1, 2), trg_ori.transpose(-2,-1)) # (1, M, 341, 3) x (1, M, 3, 1) -> (1, M, 341, 1)
        # src_inv_feats = rearrange(src_inv_feats, 'b n c r -> b (c r) n') # (1, N, 341, 3) -> (1, 1024, N)
        # trg_inv_feats = rearrange(trg_inv_feats, 'b n c r -> b (c r) n') # (1, M, 341, 3) -> (1, 1024, M)
        
        # # 5. Chaneel Attention Map
        # if self.attention == 'channel':
        #     inv_feats = torch.cat([src_inv_feats, trg_inv_feats], dim=-1)  # (1, 1024, N+M)
        #     attention = self.c_attn(inv_feats) # (1, 1024, N+M) -> (1, 1024, N+M)
            
        #     shape_attention, occ_attention = attention[:, :512], attention[:, 512:]
        #     loss['shape_attn_ratio'] = shape_attention.sum() / (shape_attention.sum()+occ_attention.sum())
        #     loss['occ_attn_ratio'] = occ_attention.sum() / (shape_attention.sum()+occ_attention.sum())
        
        # #### 6. SHAPE DESCRIPTOR ####
        # src_shape_feats = self.shape_mlp(src_inv_feats) # (1, 1024, M) -> (1, 512, M)
        # if self.attention == 'channel': src_shape_feats = src_shape_feats * shape_attention
        # trg_shape_feats = self.shape_mlp(trg_inv_feats) # (1, 1024, M) -> (1, 512, M)
        # if self.attention == 'channel': trg_shape_feats = trg_shape_feats * shape_attention
        # #### 6. SHAPE DESCRIPTOR ####

        # #### 7. OCCUPANCY DESCRIPTOR ####
        # src_occ_feats = self.occ_mlp(src_inv_feats) # (1, 1024, N) -> (1, 512, N)
        # if self.attention == 'channel': src_occ_feats = src_occ_feats * occ_attention
        # trg_occ_feats = self.occ_mlp(trg_inv_feats) # (1, 1024, M) -> (1, 512, M)
        # if self.attention == 'channel': trg_occ_feats = trg_occ_feats * occ_attention
        # #### 7. OCCUPANCY DESCRIPTOR ####

        # # 8. Optimal Transport
        # shape_matching_scores = torch.einsum('b c n , b c m -> b n m', src_shape_feats, trg_shape_feats) # (1, N, M)
        # shape_matching_scores = shape_matching_scores / src_shape_feats.shape[1] ** 0.5

        # if self.occ_loss=='positive': 
        #     occ_matching_scores = torch.einsum('b c n , b c m -> b n m', src_occ_feats, trg_occ_feats) # (1, N, M)
        # else: 
        #     occ_matching_scores = -torch.einsum('b c n , b c m -> b n m', src_occ_feats, trg_occ_feats) # (1, N, M)
        # occ_matching_scores = occ_matching_scores / src_occ_feats.shape[1] ** 0.5

        # matching_scores = self.optimal_transport(shape_matching_scores + occ_matching_scores) # (1, N, M) -> (1, N+1, M+1)
        # matching_scores_drop = matching_scores[:,:-1,:-1]

        # # 9. Weighted SVD with top-k correspondence selections
        # if mode in ['val', 'test']:
        #     with torch.no_grad():
        #         src_corr_pts, trg_corr_pts, corr_scores, estimated_transform, pred_corr = self.fine_matching(
        #             src_pcd, trg_pcd, matching_scores_drop, k=128)

        #     out_dict['estimated_rotat'] = estimated_transform[:3, :3].T
        #     out_dict['estimated_trans'] = -(estimated_transform[:3, :3].inverse() @ -estimated_transform[:3, 3])

        # # 10. Calculate Loss
        # gt_corr = in_dict['gt_correspondence'].squeeze(0)
        
        # # 9-1. circle loss
        # loss['c_loss'] = self.circle_loss(src_pcd_raw, trg_pcd_raw, src_shape_feats.transpose(-2,-1), trg_shape_feats.transpose(-2,-1), gt_corr)

        # # 9-2 point matching loss
        # loss['p_loss'] = self.matching_loss(matching_scores, gt_corr, src_pcd_raw, trg_pcd_raw)

        # 9-3. orientation loss
        # loss['o_loss'] = self.orientation_loss(src_ori, trg_ori, gt_corr, in_dict['gt_rotat'])
        loss['o_loss'] = self.orientation_loss(src_ori, trg_ori, in_dict['gt_normals'])
        
        # # 9-4. occupancy loss
        # if self.occ_loss=='positive': 
        #     loss['occ_loss'] = self.occupancy_loss(src_pcd_raw, trg_pcd_raw, src_occ_feats.transpose(-2,-1), trg_occ_feats.transpose(-2,-1), gt_corr)
        # else:
        #     loss['occ_loss'] = self.occupancy_loss(src_pcd_raw, trg_pcd_raw, src_occ_feats.transpose(-2,-1), -trg_occ_feats.transpose(-2,-1), gt_corr)

        # 9-4. final loss
        # loss['loss'] = self.c_loss_weight * loss['c_loss'] + self.p_loss_weight * loss['p_loss'] + self.o_loss_weight * loss['o_loss'] +  self.occ_loss_weight * loss['occ_loss']
        loss['loss'] = loss['o_loss']
        out_dict.update(loss)
        
        # # 10. Evaluation
        # if mode in ['val', 'test']:
        #     eval_dict = self.evaluate_prediction(in_dict, out_dict, gt_corr)
        #     loss.update(eval_dict)

        # breakpoint()

        if self.debug:
            # if min(src_pcd.size(1), trg_pcd.size(1))>1000:
            vis_dict = {}
            # vis_dict['src_shape_feats'] = src_shape_feats.squeeze(0).cpu().detach()
            # vis_dict['src_occ_feats'] = src_occ_feats.squeeze(0).cpu().detach()
            # vis_dict['trg_shape_feats'] = trg_shape_feats.squeeze(0).cpu().detach()
            # vis_dict['trg_occ_feats'] = trg_occ_feats.squeeze(0).cpu().detach()
            vis_dict['src_vec'] = src_vecs.squeeze(0).cpu().detach()
            vis_dict['trg_vec'] = trg_vecs.squeeze(0).cpu().detach()
            vis_dict['src_ori'] = src_ori.squeeze(0).cpu().detach()
            vis_dict['trg_ori'] = trg_ori.squeeze(0).cpu().detach()
            # vis_dict['src_pcd'] = src_pcd.squeeze(0).cpu().detach()
            # vis_dict['trg_pcd'] = trg_pcd.squeeze(0).cpu().detach()
            vis_dict['src_pcd_raw'] = src_pcd_raw.squeeze(0).cpu().detach()
            vis_dict['trg_pcd_raw'] = trg_pcd_raw.squeeze(0).cpu().detach()
            vis_dict['src_gt_rot'] = in_dict['gt_rotat'][0].squeeze(0).cpu().detach()
            vis_dict['trg_gt_rot'] = in_dict['gt_rotat'][1].squeeze(0).cpu().detach()
            # vis_dict['gt_correspondence'] = in_dict['gt_correspondence'].squeeze(0).cpu().detach()
            # vis_dict['pred_corr'] = pred_corr.cpu().detach()

            save_folder = './pickles/only_ori'
            os.makedirs(save_folder, exist_ok=True)
            with open(f'{save_folder}/{in_dict["eval_idx"]}_debug.pickle', 'wb') as f:
                pickle.dump(vis_dict, f)
            print("writing...")

        # in training we log for every step
        if mode == 'train':
            log_dict = {f'{mode}/{k}': v.item() for k, v in loss.items()}
            self.log_dict(log_dict, logger=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True, batch_size=1)
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', lr, prog_bar=True, logger=True)
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
            if min(src_pcd.size(0), trg_pcd.size(0))>1000:
                pcds_pred.append(pcds_pred[0][gt_corr[:,0]])
                pcds_pred.append(pcds_pred[1][gt_corr[:,1]])
                pcds_grtr.append(pcds_grtr[0][gt_corr[:,0]])
                pcds_grtr.append(pcds_grtr[1][gt_corr[:,1]])
                save_pc(f'./junhong/{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["crd"].item(),3)}_pred.pcd', pcds_pred)
                save_pc(f"./junhong/{in_dict['eval_idx'].item()}_{in_dict['obj_class'][0]}_{round(eval_result['crd'].item(),3)}_grtr.pcd", pcds_grtr)
            
            ### MESH VISUALIZATION ###
            # base_path = '../../data/bbad_v2/'
            # obj_paths = [os.path.join(base_path + in_dict['filepath'][0], x) for x in os.listdir(base_path + in_dict['filepath'][0])]

            # mesh = [trimesh.load_mesh(x) for x in obj_paths]
            
            # for idx in range(len(mesh)):
            #     mesh[idx].vertices = in_dict['mesh_t'][idx].squeeze(0).cpu().detach().numpy()

            # assm_pred, mesh_pred = self._pairwise_mating_mesh(mesh[0], mesh[1], pred_relative_trsfm[0], pred_relative_trsfm[1], is_trg_larger)
            # assm_grtr, mesh_grtr = self._pairwise_mating_mesh(mesh[0], mesh[1], grtr_relative_trsfm[0], grtr_relative_trsfm[1], is_trg_larger)
            
            # # Set base name
            # if in_dict['filepath'][0].split('/')[0] == 'everyday':
            #     base_name = f"{in_dict['eval_idx'].item()}_{len(mesh_pred)}_part_crd{round(eval_result['crd'].item(), 2)}_cd{round(eval_result['cd'].item(), 1)}_rrmse{round(eval_result['rrmse'].item(), 2)}_{in_dict['filepath'][0].split('/')[2]}_{in_dict['filepath'][0].split('/')[3]}"
            # elif in_dict['filepath'][0].split('/')[0] == 'artifact':
            #     base_name = f"{in_dict['eval_idx'].item()}_{len(mesh_pred)}_part_crd{round(eval_result['crd'].item(), 2)}_cd{round(eval_result['cd'].item(), 1)}_rrmse{round(eval_result['rrmse'].item(), 2)}_{in_dict['filepath'][0].split('/')[1]}_{in_dict['filepath'][0].split('/')[2]}"
            
            # if min(in_dict['pcd'][0].size(1), in_dict['pcd'][1].size(1))>1000:
            #     if not os.path.exists(os.path.join('./teaser_mesh_pos', base_name)):
            #         os.makedirs(os.path.join('./teaser_mesh_pos', base_name))
            #     for idx, _mesh in enumerate(mesh_pred):
            #         _mesh.export(os.path.join('./teaser_mesh_pos', base_name, f'{idx}_fracture.obj'), file_type='obj')
            #     assm_pred.export(os.path.join('./teaser_mesh_pos', base_name ,f'assemble.obj'), file_type='obj')
            
            # if not os.path.exists(os.path.join('./vis_mesh_grtr', base_name)):
            #     os.makedirs(os.path.join('./vis_mesh_grtr', base_name))
            # for idx, _mesh in enumerate(mesh_grtr):
            #     _mesh.export(os.path.join('./vis_mesh_grtr', base_name, f'{idx}_fracture.obj'), file_type='obj')
            # assm_grtr.export(os.path.join('./vis_mesh_grtr', base_name, f'assemble.obj'), file_type='obj')

            # if min(in_dict['pcd'][0].size(1), in_dict['pcd'][1].size(1))>1000:
            #     for idx, _mesh in enumerate(mesh):
            #         _mesh.export(os.path.join('./vec_ours', f'{in_dict["eval_idx"].item()}_{idx}_fracture.obj'), file_type='obj')

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

    def _pairwise_mating_mesh(self, src_mesh, trg_mesh, rotat, trans, is_trg_larger):
        mesh_t = []
        src_mesh_t, trg_mesh_t = src_mesh.copy(), trg_mesh.copy()
        if is_trg_larger:
            src_mesh_t.vertices = self._transform(torch.tensor(src_mesh_t.vertices).float(), rotat, -trans, True).numpy()
            mesh_t = [src_mesh_t, trg_mesh]
        else:
            trg_mesh_t.vertices = self._transform(torch.tensor(trg_mesh_t.vertices).float(), rotat.T, trans, False).numpy()
            mesh_t = [src_mesh, trg_mesh_t]
        return sum(mesh_t), mesh_t
    
    
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

    def _part_accuracy(self, assm_pts1, assm_pts2, scaling=100):
        success = 0
        for pred_pts, gt_pts in zip(assm_pts1, assm_pts2):
            part_cd = self._chamfer_distance(pred_pts, gt_pts, 1)
            if part_cd < 0.01: success += 1
        return success / len(assm_pts1) * scaling

    def _part_accuracy_crd(self, assm_pts1, assm_pts2, scaling=100):
        success = 0
        for pred_pts, gt_pts in zip(assm_pts1, assm_pts2):
            part_crd = self._correspondence_distance(pred_pts, gt_pts, 1)
            if part_crd < 0.1: success += 1
        return success / len(assm_pts1) * scaling

    @torch.no_grad()
    def forward_mpa(self, in_dict):

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
        src_inv_feats = rearrange(src_inv_feats, 'b n c r -> b (c r) n') # (1, N, 341, 3) -> (1, 1024, N)
        trg_inv_feats = rearrange(trg_inv_feats, 'b n c r -> b (c r) n') # (1, M, 341, 3) -> (1, 1024, M)
        
        # 5. Chaneel Attention Map
        if self.attention == 'channel':
            inv_feats = torch.cat([src_inv_feats, trg_inv_feats], dim=-1)  # (1, 1024, N+M)
            attention = self.c_attn(inv_feats) # (1, 1024, N+M) -> (1, 1024, N+M)
            
            shape_attention, occ_attention = attention[:, :512], attention[:, 512:]
            loss['shape_attn_ratio'] = shape_attention.sum() / (shape_attention.sum()+occ_attention.sum())
            loss['occ_attn_ratio'] = occ_attention.sum() / (shape_attention.sum()+occ_attention.sum())
        
        #### 6. SHAPE DESCRIPTOR ####
        src_shape_feats = self.shape_mlp(src_inv_feats) # (1, 1024, M) -> (1, 512, M)
        if self.attention == 'channel': src_shape_feats = src_shape_feats * shape_attention
        trg_shape_feats = self.shape_mlp(trg_inv_feats) # (1, 1024, M) -> (1, 512, M)
        if self.attention == 'channel': trg_shape_feats = trg_shape_feats * shape_attention
        #### 6. SHAPE DESCRIPTOR ####

        #### 7. OCCUPANCY DESCRIPTOR ####
        src_occ_feats = self.occ_mlp(src_inv_feats) # (1, 1024, N) -> (1, 512, N)
        if self.attention == 'channel': src_occ_feats = src_occ_feats * occ_attention
        trg_occ_feats = self.occ_mlp(trg_inv_feats) # (1, 1024, M) -> (1, 512, M)
        if self.attention == 'channel': trg_occ_feats = trg_occ_feats * occ_attention
        #### 7. OCCUPANCY DESCRIPTOR ####

        # 8. Optimal Transport
        shape_matching_scores = torch.einsum('b c n , b c m -> b n m', src_shape_feats, trg_shape_feats) # (1, N, M)
        # shape_matching_scores = shape_matching_scores / src_shape_feats.shape[1] ** 0.5

        if self.occ_loss=='positive': 
            occ_matching_scores = torch.einsum('b c n , b c m -> b n m', src_occ_feats, trg_occ_feats) # (1, N, M)
        else: 
            occ_matching_scores = -torch.einsum('b c n , b c m -> b n m', src_occ_feats, trg_occ_feats) # (1, N, M)
        # occ_matching_scores = occ_matching_scores / src_occ_feats.shape[1] ** 0.5

        matching_scores = self.optimal_transport(shape_matching_scores + occ_matching_scores) # (1, N, M) -> (1, N+1, M+1)
        matching_scores_drop = matching_scores[:,:-1,:-1]


        if self.debug:
            vis_dict = {}
            # vis_dict['src_shape_feats'] = src_shape_feats.squeeze(0).cpu().detach()
            # vis_dict['src_occ_feats'] = src_occ_feats.squeeze(0).cpu().detach()
            # vis_dict['trg_shape_feats'] = trg_shape_feats.squeeze(0).cpu().detach()
            # vis_dict['trg_occ_feats'] = trg_occ_feats.squeeze(0).cpu().detach()
            vis_dict['src_vec'] = src_vecs.squeeze(0).cpu().detach()
            vis_dict['trg_vec'] = trg_vecs.squeeze(0).cpu().detach()
            src_ori_vec = torch.matmul(src_ori, in_dict['gt_rotat'][0].squeeze(0))
            trg_ori_vec = torch.matmul(trg_ori, in_dict['gt_rotat'][1].squeeze(0))
            src_ori_vec = src_ori_vec.squeeze(0).cpu().detach()
            trg_ori_vec = trg_ori_vec.squeeze(0).cpu().detach()
            vis_dict['src_ori'] = src_ori_vec
            vis_dict['trg_ori'] = trg_ori_vec
            # vis_dict['src_pcd'] = src_pcd.squeeze(0).cpu().detach()
            # vis_dict['trg_pcd'] = trg_pcd.squeeze(0).cpu().detach()
            vis_dict['src_pcd_raw'] = src_pcd_raw.squeeze(0).cpu().detach()
            vis_dict['trg_pcd_raw'] = trg_pcd_raw.squeeze(0).cpu().detach()
            vis_dict['src_gt_rot'] = in_dict['gt_rotat'][0].squeeze(0).cpu().detach()
            vis_dict['trg_gt_rot'] = in_dict['gt_rotat'][1].squeeze(0).cpu().detach()
            gt_correspond = in_dict['gt_correspondence'].squeeze(0).cpu().detach()
            vis_dict['gt_correspondence'] = gt_correspond
            # vis_dict['pred_corr'] = pred_corr.cpu().detach()
            if len(gt_correspond.shape) == 2:
                print(f"ori_loss: {torch.mean(torch.norm((src_ori_vec[gt_correspond[:,0],:,:] - trg_ori_vec[gt_correspond[:,1],:,:]), p='fro', dim=(1,2)))}")

            with open(f'./orientation_visualization_train/{in_dict["pair_idx"]}_debug.pickle', 'wb') as f:
                pickle.dump(vis_dict, f)

        # 9. Registration
        with torch.no_grad():
            if self.registration == 'wsvd':
                src_corr_pts, trg_corr_pts, corr_scores, estimated_transform, pred_corr = self.fine_matching(
                    src_pcd, trg_pcd, matching_scores_drop, k=128)

                out_dict['estimated_rotat'] = estimated_transform[:3, :3].T
                out_dict['estimated_trans'] = -(estimated_transform[:3, :3].inverse() @ -estimated_transform[:3, 3])
                out_dict['corr_scores'] = corr_scores
                out_dict['matching_scores_drop'] = matching_scores_drop.unsqueeze(0)
                
            elif self.registration == 'ransac':
                # 9-0. use the matching scores before Sinkhorn
                matching_scores_before_Sinkhorn = shape_matching_scores + occ_matching_scores

                # 9-1. Select initial matches
                matching_scores_before_Sinkhorn = matching_scores_before_Sinkhorn.squeeze(0)

                # if self.gt_mating_surface and in_dict['gt_mating_surface'][0].sum() and in_dict['gt_mating_surface'][1].sum():
                #     gtms_src = in_dict['gt_mating_surface'][0].squeeze(0)
                #     gtms_trg = in_dict['gt_mating_surface'][1].squeeze(0)
                #     breakpoint()
                #     matching_scores_before_Sinkhorn = matching_scores_before_Sinkhorn[gtms_src][:, gtms_trg]
                
                if self.match_selection == 'topk':
                    initial_matches = topk_matching(matching_scores_before_Sinkhorn, k=128)
                elif self.match_selection == 'unidirectional_nn':
                    initial_matches = unidirectional_nn_matching(matching_scores_before_Sinkhorn)
                elif self.match_selection == 'injective':
                    initial_matches = injective_matching(matching_scores_before_Sinkhorn)
                elif self.match_selection == 'bijective': # ERROR: size of src_pcd and trg_pcd are now different
                    initial_matches = bijective_matching(matching_scores_before_Sinkhorn)
                elif self.match_selection == 'reciprocal':
                    initial_matches = reciprocal_test(matching_scores_before_Sinkhorn)
                
                src_idx, trg_idx = initial_matches[:, 0], initial_matches[:, 1]

                
                ### hyperparameters ###
                inlier_threshold = self.inlier_threshold
                dist_mat_type = 'cdist' # ['l2_norm', 'cdist']
                which_transform = 'whole' # ['corr', 'whole']
                weighted_voting_design = 0


                # score thresholding
                score_threshold = self.score_threshold
                feature_mask = None
                weights = None
                if self.auto_score_threshold or self.weighted_voting:
                    src_shape_feats_norm = F.normalize(src_shape_feats.transpose(-2,-1).squeeze(0), p=2, dim=-1)
                    trg_shape_feats_norm = F.normalize(trg_shape_feats.transpose(-2,-1).squeeze(0), p=2, dim=-1)
                    src_occ_feats_norm = F.normalize(src_occ_feats.transpose(-2,-1).squeeze(0), p=2, dim=-1)
                    trg_occ_feats_norm = F.normalize(-trg_occ_feats.transpose(-2,-1).squeeze(0), p=2, dim=-1)
                    shape_feats_dist = (2.0 - 2.0 * torch.einsum('x d, y d -> x y', src_shape_feats_norm, trg_shape_feats_norm)).pow(0.5)
                    occ_feats_dist = (2.0 - 2.0 * torch.einsum('x d, y d -> x y', src_occ_feats_norm, trg_occ_feats_norm)).pow(0.5)
                    if self.auto_score_threshold:
                        shape_mask = shape_feats_dist < 0.75
                        occ_mask = occ_feats_dist < 0.75
                        feature_mask = shape_mask | occ_mask
                        src_idx, trg_idx = src_idx[feature_mask[src_idx, trg_idx]], trg_idx[feature_mask[src_idx, trg_idx]]
                    elif self.weighted_voting:
                        feats_dist = (shape_feats_dist + occ_feats_dist) / 2
                        # feats_dist = shape_feats_dist
                        # feats_dist = occ_feats_dist
                        if weighted_voting_design == 0:
                            weights = 2 - feats_dist
                        elif weighted_voting_design == 1:
                            weights = 1 / torch.maximum(feats_dist, torch.tensor(1e-3))
                        elif weighted_voting_design == 2:
                            weights = torch.max(matching_scores_before_Sinkhorn, torch.tensor(0))
                        elif weighted_voting_design == 3:
                            x = matching_scores_before_Sinkhorn
                            weights = torch.max(2 * (x - x.min()) / (x.max() - x.min()) - 1, torch.tensor(0))
                        elif weighted_voting_design == 4:
                            weights = torch.max(torch.log1p(matching_scores_before_Sinkhorn), torch.tensor(0)) ## Log Normalization
                # elif self.gt_mating_surface and in_dict['gt_mating_surface'][0].sum() and in_dict['gt_mating_surface'][1].sum():
                #     gtms_src = in_dict['gt_mating_surface'][0].squeeze(0)
                #     gtms_trg = in_dict['gt_mating_surface'][1].squeeze(0)
                #     gtms = gtms_src[src_idx] & gtms_trg[trg_idx]
                #     src_idx = src_idx[gtms]
                #     trg_idx = trg_idx[gtms]
                #     out_dict['strange_case'] = 0
                #     if gtms.sum() == 0:
                #         score_mask = matching_scores_before_Sinkhorn[src_idx[:], trg_idx[:]] >= score_threshold
                #         src_idx, trg_idx = src_idx[score_mask], trg_idx[score_mask]
                #         out_dict['strange_case'] = 1
                else:
                    if self.score_comb == 'sum':
                        score_mask = matching_scores_before_Sinkhorn[src_idx[:], trg_idx[:]] >= score_threshold
                    elif self.score_comb == 'intersection':
                        shape_mask = shape_matching_scores.squeeze(0)[src_idx[:], trg_idx[:]] >= score_threshold
                        occ_mask = occ_matching_scores.squeeze(0)[src_idx[:], trg_idx[:]] >= score_threshold
                        score_mask = shape_mask & occ_mask
                    src_idx, trg_idx = src_idx[score_mask], trg_idx[score_mask]

                src_corr_pts = src_pcd[:,src_idx].squeeze(0) # (C, 3)
                trg_corr_pts = trg_pcd[:,trg_idx].squeeze(0) # (C, 3)

                # 9-2. Run RANSAC


                # vis_src_ori = torch.matmul(src_ori.squeeze(0), in_dict['gt_rotat'][0])
                # vis_trg_ori = torch.matmul(trg_ori.squeeze(0), in_dict['gt_rotat'][1])
                # vis_src_ori = torch.mean(vis_src_ori, dim=2)
                # vis_trg_ori = torch.mean(vis_trg_ori, dim=2)
                # save_ori('test_ori.ply', [src_pcd_raw, trg_pcd_raw], [vis_src_ori, vis_trg_ori])
                # breakpoint()
                print(f"filepath: {in_dict['filepath']}")
                print(f'src_corr_pts: {src_corr_pts.shape} | trg_corr_pts: {trg_corr_pts.shape}')
                
                num_iters = math.ceil(src_corr_pts.shape[0] * 3 / 3 / 10) * 10
                print(f"num_iters: {num_iters}")
                matching_choice = 'many-to-one'
                inl_R, inl_t, inliers, best_CD, min_CD, normals = ransac_rigid(src_corr_pts, trg_corr_pts, 
                                             src_pcd.squeeze(0), trg_pcd.squeeze(0), 
                                             next(iter(in_dict['relative_trsfm'].values())),
                                             in_dict['gt_normals'][0].squeeze(0), in_dict['gt_normals'][1].squeeze(0),
                                             scores = matching_scores_before_Sinkhorn,
                                             shape_scores = shape_matching_scores.squeeze(0),
                                             occ_scores = occ_matching_scores.squeeze(0),
                                             score_threshold = score_threshold,
                                             feature_mask = feature_mask,
                                             feature_thresholding = self.auto_score_threshold,
                                             src_ori=src_ori.squeeze(0), trg_ori=trg_ori.squeeze(0),
                                             dist_mat_type=dist_mat_type,
                                             which_transform=which_transform, 
                                             orientation_threshold=self.ori_threshold,
                                             num_iters=num_iters, 
                                             threshold=inlier_threshold,
                                             weights=weights,
                                             gt_normal_threshold=self.gt_normal_threshold,
                                             matching_choice=matching_choice)
                ## optimal estimation (re-estimation)
                # many-to-one case : 복제하여 one-to-one으로
                if matching_choice == 'many-to-one':
                    optimal_correspondences = np.argwhere(inliers)
                    # breakpoint()
                    src_opt_corr = optimal_correspondences[:, 0]
                    trg_opt_corr = optimal_correspondences[:, 1]
                    src_opt_pcd = src_pcd[:, src_opt_corr].squeeze(0)
                    trg_opt_pcd = trg_pcd[:, trg_opt_corr].squeeze(0)
                    R, t = SVD(src_opt_pcd, trg_opt_pcd)
                
                
                # logging
                if which_transform == 'corr': 
                    if dist_mat_type == 'l2_norm': print(f'inliers: {inliers.sum()}/{trg_corr_pts.shape[0]}')
                    elif dist_mat_type == 'cdist': print(f'inliers: {inliers.sum()}/{src_corr_pts.shape[0]*trg_corr_pts.shape[0]}')
                elif which_transform == 'whole': print(f'inliers: {np.count_nonzero(np.sum(inliers, axis=0))}/{src_pcd.shape[1]*trg_pcd.shape[1]}')

                trg_transformed = (R @ trg_pcd.squeeze(0).T).T + t
                dist_whole_match = cdist(src_pcd.squeeze(0).cpu().numpy(), trg_transformed.cpu().numpy())
                inliers_upperbound = dist_whole_match < inlier_threshold
                inliers_count_upper = np.count_nonzero(np.sum(inliers_upperbound, axis=0))
                min_mask = dist_whole_match == np.min(dist_whole_match, axis=0)
                idx_inliers_upper = inliers_upperbound & min_mask
                print(f"inliers for whole matches : {inliers_count_upper}/{src_pcd.shape[1]*trg_pcd.shape[1]}")

                # breakpoint()

                # src_corr_pts, trg_corr_pts, corr_scores, estimated_transform, pred_corr = self.fine_matching(
                #     src_pcd, trg_pcd, matching_scores_drop, k=128)

                # out_dict['estimated_rotat'] = estimated_transform[:3, :3].T
                # out_dict['estimated_trans'] = -(estimated_transform[:3, :3].inverse() @ -estimated_transform[:3, 3])

                idx_initial_matches = torch.stack([src_idx, trg_idx], dim=1)
                out_dict['estimated_rotat'] = R.T
                out_dict['estimated_trans'] = t@R
                out_dict['idx_initial_matches'] = idx_initial_matches
                out_dict['corr_inliers'] = inliers
                out_dict['corr_inliers_upper'] = idx_inliers_upper
                out_dict['which_transform'] = which_transform
                out_dict['dist_mat_type'] = dist_mat_type
                out_dict['matching_scores_drop'] = matching_scores_before_Sinkhorn
                out_dict['src_ori'] = src_ori.squeeze(0).cpu().detach()
                out_dict['trg_ori'] = trg_ori.squeeze(0).cpu().detach()
                out_dict['weighted_voting_design'] = weighted_voting_design
                out_dict['best_CD'] = best_CD
                out_dict['min_CD'] = min_CD
                out_dict['inlier_normals'] = normals

        
        return out_dict
