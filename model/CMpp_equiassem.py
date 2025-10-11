import os
import pickle
from scipy.spatial.transform import Rotation

import pytorch_lightning as pl

from chamfer_distance import ChamferDistance as chamfer_dist

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange

from model.backbone.vn_dgcnn import EQCNN_equi_unet, EQCNN_equi
from model.backbone.vn_layers import VNLinear, VNLinearLeakyReLU
from model.loss import PointMatchingLoss, OrientationLoss, CircleLoss
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from model.local_global_registration import LocalGlobalRegistration

from common.rotation import ortho2rotation
from common.utils import save_pc



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
    def __init__(
            self, 
            lr, backbone='vn_unet', attention='channel', 
            pos_margin=0.1, neg_margin=1.4, log_scale=24,
            s_loss_weight=1.0, p_loss_weight=1.0, o_loss_weight=1.0,
            visualize=False, debug=False,
            ):
        """Equivariant Assembly Model for 3D Object Assembly

        Args:
            lr (float): Learning rate for optimizer.
            backbone (str, optional): Backbone network architecture. Defaults to 'vn_unet'.
            attention (str, optional): Attention mechanism type ('channel' or 'none'). Defaults to 'channel'.
            pos_margin (float, optional): Margin for positive samples in loss computation. Defaults to 0.1.
            neg_margin (float, optional): Margin for negative samples in loss computation. Defaults to 1.4.
            log_scale (int, optional): Log scaling factor for loss computation. Defaults to 24.
            s_loss_weight (float, optional): Weight for shape loss. Defaults to 1.0.
            p_loss_weight (float, optional): Weight for point loss. Defaults to 1.0.
            o_loss_weight (float, optional): Weight for orientation loss. Defaults to 1.0.
            visualize (bool, optional): Whether to save visualization results. Defaults to False.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
        """
        super(EquiAssem, self).__init__()

        print("------------------------------------------------------")
        print("INITIALIZING EquiAssem(pl.LightningModule)")
        print("------------------------------------------------------")
        print(f"lr: {lr}")
        print(f"backbone: {backbone}")
        print(f"attention: {attention}")
        print(f"pos_margin: {pos_margin}")
        print(f"neg_margin: {neg_margin}")
        print(f"log_scale: {log_scale}")
        print(f"s_loss_weight: {s_loss_weight}")
        print(f"p_loss_weight: {p_loss_weight}")
        print(f"o_loss_weight: {o_loss_weight}")
        print(f"visualize: {visualize}")
        print(f"debug: {debug}")
        print("------------------------------------------------------")

        self.lr = lr
        self.attention = attention
        self.debug = debug
        self.visualize = visualize


        # Output feature dimension of Feature Extractor
        self.feat_dim = 1024

        
        # Objectives
        self.matching_loss = PointMatchingLoss()
        self.shape_loss = CircleLoss(pos_optimal=pos_margin, neg_optimal=neg_margin, log_scale=log_scale)
        self.orientation_loss = OrientationLoss()


        # Weights for losses
        self.s_loss_weight = s_loss_weight
        self.p_loss_weight = p_loss_weight
        self.o_loss_weight = o_loss_weight


        # Logging
        self.validation_step_outputs = []
        self.test_step_outputs = []


        # Declare Modules

        # VN BACKBONE
        if backbone == 'vn_unet':
            self.backbone = EQCNN_equi_unet(feat_dim=self.feat_dim, pooling="mean")
        elif backbone == 'vn_dgcnn':
            self.backbone = EQCNN_equi(feat_dim=self.feat_dim, pooling="mean")
        else:
            raise NotImplementedError("DGCNN backbone not implemented")


        # Layer for Rotation Matrix, it will predict frame vectors
        self.proj = VNLinear(2 * (self.feat_dim//3), 2)

        
        # Layer for Equivariant feature
        self.equi_layer = nn.Sequential(
            VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3),
            VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3),
            VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3),
            VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3),
            VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3),
        )


        # Channel Attention
        if attention == 'channel':
            self.c_attn = ChannelAttentionModule((self.feat_dim//3) * 3, self.feat_dim, ratio=4)
        

        # Module for invariant Shape Descriptor
        self.shape_mlp = nn.Sequential(nn.Conv1d((self.feat_dim//3) * 3, self.feat_dim, kernel_size=1, bias=False),
                                       nn.InstanceNorm1d(self.feat_dim),
                                       nn.LeakyReLU(negative_slope=0.2),
                                       nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
                                       nn.InstanceNorm1d(self.feat_dim),
                                       nn.LeakyReLU(negative_slope=0.2),
                                       nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
                                       nn.InstanceNorm1d(self.feat_dim),
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

    
    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.lr
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16919, eta_min=1e-3) # 16919, 6671
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def training_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='train')
        if torch.isnan(loss_dict['loss']):
            print(loss_dict['loss'])
        if loss_dict['loss']==0.: 
            return None
        print(f"shape loss : {loss_dict['s_loss']}")
        print(f"point matching loss : {loss_dict['p_loss']}")
        return loss_dict['loss']
    

    def validation_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='val')
        if torch.isnan(loss_dict['loss']):
            print(loss_dict['loss'])
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
        """
        Args:
            Assumption: Batch size is 1

            in_dict (dict):
                - eval_idx (torch.Tensor): (1, )
                - filepath (list): (1, ), e.g. ['everyday/BeerBottle/2927d6c8438f6e24fe6460d8d9bd16c6/fractured_37']
                - obj_class (list): (1, ), e.g. ['BeerBottle']
                
                - mesh (list): length is 2, only for two pieces
                    - mesh[0]: (1, N', 3)
                    - mesh[1]: (1, M', 3)
                - mesh_t (list): length is 2, only for two pieces
                    - mesh_t[0]: (1, N', 3)
                    - mesh_t[1]: (1, M', 3)
                
                - pcd_t (list): length is 2, only for two pieces
                    - pcd_t[0]: (1, N, 3)
                    - pcd_t[1]: (1, M, 3)
                - pcd (list): length is 2, only for two pieces
                    - pcd[0]: (1, N, 3)
                    - pcd[1]: (1, M, 3)
                
                - n_frac (torch.Tensor): (1, )
                - anchor_idx (torch.Tensor): (1, )
                
                - gt_trans (list): length is 2, only for two pieces
                    - gt_trans[0]: (1, 3)
                    - gt_trans[1]: (1, 3)
                - gt_rotat (list): length is 2, only for two pieces
                    - gt_rotat[0]: (1, 3, 3)
                    - gt_rotat[1]: (1, 3, 3)
                - gt_trans_inv (list): length is 2, only for two pieces
                    - gt_trans_inv[0]: (1, 3)
                    - gt_trans_inv[1]: (1, 3)
                - gt_rotat_inv (list): length is 2, only for two pieces
                    - gt_rotat_inv[0]: (1, 3, 3)
                    - gt_rotat_inv[1]: (1, 3, 3)
                
                - relative_trsfm (dict):
                    - key: relative_rotat, relative_trans
                        - relative_rotat: (1, 3, 3)
                        - relative_trans: (1, 3)
                
                - gt_normals (list): length is 2, only for two pieces
                    - gt_normals[0]: (1, N, 3)
                    - gt_normals[1]: (1, M, 3)

                - gt_correspondence (torch.Tensor): (1, P, 2)

            mode (string): ['train', 'val', 'test']

        Returns:
            out_dict (dict)
                - src_equi_feats: (1, D, 3, N)
                - trg_equi_feats: (1, D, 3, M)
                - estimated_rotat: (3, 3)
                - estimated_trans: (3)
                - p_loss: (1, )
                - loss: (1, )

            loss (dict)
                - p_loss: (1, )
                - loss: (1, )

                When validation or test,
                - cd: (1, )
                - rrmse: (1, )
                - trmse: (1, )
                - crd: (1, )
        """
        out_dict, loss = {}, {}

        exit("stop")


        # 0. Get Point Clouds and Ground Truth Correspondence
        src_pcd_raw = in_dict['pcd'][0].squeeze(0) # (N, 3)
        trg_pcd_raw = in_dict['pcd'][1].squeeze(0) # (M, 3)
        src_pcd = in_dict['pcd_t'][0] # (1, N ,3)
        trg_pcd = in_dict['pcd_t'][1] # (1, M ,3)
        gt_corr = in_dict['gt_correspondence'].squeeze(0) # (1, P, 2) -> (P, 2)


        # 1. SO(3)-Equivariant Feature Extractor
        src_equi_feats_backbone = self.backbone(src_pcd) # (1, C, 3, N)
        trg_equi_feats_backbone = self.backbone(trg_pcd) # (1, C, 3, M)


        # 2. Start frame prediction
        # 2-1. Merge global information by averaging
        # (1, C, 3, N) -> (1, C, 3, 1) -> (1, C, 3, N)
        src_equi_feats_backbone_mean = src_equi_feats_backbone.mean(dim=-1, keepdim=True).expand(src_equi_feats_backbone.size())
        # (1, C, 3, M) -> (1, C, 3, 1) -> (1, C, 3, M)
        trg_equi_feats_backbone_mean = trg_equi_feats_backbone.mean(dim=-1, keepdim=True).expand(trg_equi_feats_backbone.size())

        # 2-2. Basis Vector Projection, those vectors will be used as frame basis vectors
        # (1, C, 3, N) concat (1, C, 3, N) ->  (1, 2C, 3, N) -> (1, 2C, 3, N, 1) -> (1, 2, 3, N, 1) -> (1, 2, 3, N) -> (1, N, 2, 3)
        src_vecs = self.proj(torch.cat((src_equi_feats_backbone, src_equi_feats_backbone_mean), 1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2) 
        # (1, C, 3, M) concat (1, C, 3, M) ->  (1, 2C, 3, M) -> (1, 2C, 3, M, 1) -> (1, 2, 3, M, 1) -> (1, 2, 3, M) -> (1, M, 2, 3)
        trg_vecs = self.proj(torch.cat((trg_equi_feats_backbone, trg_equi_feats_backbone_mean), 1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2) 


        # 3. Calculate equivariant shape features
        src_equi_feats = self.equi_layer(src_equi_feats_backbone.unsqueeze(-1)).squeeze(-1) # (1, C, 3, N)
        trg_equi_feats = self.equi_layer(trg_equi_feats_backbone.unsqueeze(-1)).squeeze(-1) # (1, C, 3, M)


        
        # [TODO] WE WILL DO IN THE FUTURE
        # 4. Gram Schmidt & Cross-product, this is for making three basis vectors by using two predicted vectors
        # src_ori = ortho2rotation(src_vecs) # (1, N, 2, 3) -> (1, N, 3, 3)
        # trg_ori = ortho2rotation(trg_vecs) # (1, M, 2, 3) -> (1, M, 3, 3)



        # 5. Invariant Features
        src_inv_feats = torch.matmul(src_equi_feats.permute(0, 3, 1, 2).float(), src_ori.transpose(-2,-1).float()) # (1, N, C, 3) x (1, N, 3, 3) -> (1, N, C, 3)
        trg_inv_feats = torch.matmul(trg_equi_feats.permute(0, 3, 1, 2).float(), trg_ori.transpose(-2,-1).float()) # (1, M, C, 3) x (1, M, 3, 3) -> (1, M, C, 3)
        src_inv_feats = rearrange(src_inv_feats, 'b n c r -> b (c r) n') # (1, N, C, 3) -> (1, C*3, N)
        trg_inv_feats = rearrange(trg_inv_feats, 'b n c r -> b (c r) n') # (1, M, C, 3) -> (1, C*3, M)
        

        # OPTIONAL 5. Chaneel Attention Map
        if self.attention == 'channel':
            inv_feats = torch.cat([src_inv_feats, trg_inv_feats], dim=-1)  # (1, C*3, N+M)
            attention = self.c_attn(inv_feats) # (1, C*3, N+M) -> (1, D, N+M)
            shape_attention, occ_attention = attention[:, :512], attention[:, 512:] # [TODO] We should check this part, This can incurr problem
            loss['shape_attn_ratio'] = shape_attention.sum() / (shape_attention.sum()+occ_attention.sum())
            loss['occ_attn_ratio'] = occ_attention.sum() / (shape_attention.sum()+occ_attention.sum())
        

        # 6. SHAPE DESCRIPTOR 
        src_shape_feats = self.shape_mlp(src_inv_feats) # (1, C*3, N) -> (1, D, N)
        if self.attention == 'channel': # (1, D, N) * channel attention
            src_shape_feats = src_shape_feats * shape_attention
        
        trg_shape_feats = self.shape_mlp(trg_inv_feats) # # (1, C*3, M) -> (1, D, N)
        if self.attention == 'channel': # (1, D, M) * channel attention
            trg_shape_feats = trg_shape_feats * shape_attention


        # 7. Optimal Transport
        shape_matching_scores = torch.einsum('b c n , b c m -> b n m', src_shape_feats, trg_shape_feats) # (1, N, M)
        shape_matching_scores = shape_matching_scores / (src_shape_feats.shape[1] ** 0.5 + 1e-8) # 1e-8 is for avoiding division by zero
        
        matching_scores = self.optimal_transport(shape_matching_scores)
        matching_scores = torch.exp(matching_scores) # Optimal Transport is in log space, so before registration, we need to exp it
        matching_scores_drop = matching_scores[:,:-1,:-1]   


        # 8. Calculate Loss
        # Orientation loss
        loss['o_loss'] = self.orientation_loss(src_vecs, trg_vecs, gt_corr, in_dict['gt_normals'])
        
        # Shape loss
        loss['s_loss'], out_dict['pos_margin'], out_dict['neg_margin'] = self.shape_loss(src_pcd_raw, trg_pcd_raw, src_shape_feats.transpose(-2,-1), trg_shape_feats.transpose(-2,-1), gt_corr)

        # Point matching loss
        loss['p_loss'] = 1.0 + self.matching_loss(matching_scores, gt_corr, src_pcd_raw, trg_pcd_raw).float()

        # Final loss
        loss['loss'] = self.o_loss_weight * loss['o_loss'] + self.s_loss_weight * loss['s_loss'] + self.p_loss_weight * loss['p_loss']

        out_dict.update(loss)


        # 9. Evaluation
        if mode in ['val', 'test']:
            # Point cloud registration
            with torch.no_grad():
                src_corr_pts, trg_corr_pts, corr_scores, estimated_transform, pred_corr = self.fine_matching(src_pcd, trg_pcd, matching_scores_drop, k=128) # Param: ref_points, src_points, so it is reversed

            # estimated_transform: source_point = R * target_point + t
            out_dict['estimated_rotat'] = estimated_transform[:3, :3].T # R.T
            out_dict['estimated_trans'] = -(estimated_transform[:3, :3].inverse() @ -estimated_transform[:3, 3]) # R.T @ t

            # Evaluation
            eval_dict = self.evaluate_prediction(in_dict, out_dict, gt_corr)
            loss.update(eval_dict)
        

        if self.debug:
            vis_dict = {}
            vis_dict['src_vec'] = src_vecs.squeeze(0).cpu().detach()
            vis_dict['trg_vec'] = trg_vecs.squeeze(0).cpu().detach()
            vis_dict['src_ori'] = src_ori.squeeze(0).cpu().detach()
            vis_dict['trg_ori'] = trg_ori.squeeze(0).cpu().detach()

            vis_dict['src_pcd_raw'] = src_pcd_raw.squeeze(0).cpu().detach()
            vis_dict['trg_pcd_raw'] = trg_pcd_raw.squeeze(0).cpu().detach()
            vis_dict['src_gt_rot'] = in_dict['gt_rotat'][0].squeeze(0).cpu().detach()
            vis_dict['trg_gt_rot'] = in_dict['gt_rotat'][1].squeeze(0).cpu().detach()

            save_folder = './pickles/expanded_normal_reverse'
            os.makedirs(save_folder, exist_ok=True)
            with open(f'{save_folder}/{in_dict["eval_idx"].item()}_debug.pickle', 'wb') as f:
                pickle.dump(vis_dict, f)


        # in training we log for every step
        if mode == 'train':
            log_dict = {f'{mode}/{k}': v.item() for k, v in loss.items()}
            self.log_dict(log_dict, logger=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True, batch_size=1)
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', lr, prog_bar=True, logger=True)
        else:
            torch.cuda.empty_cache()

        return out_dict, loss


    @torch.no_grad()
    def evaluate_prediction(self, in_dict, out_dict, gt_corr, multi_part=False):
        """
        Args:
            in_dict (dict): it is same as forward_pass
            out_dict (dict): it is same as forward_pass
            gt_corr (torch.Tensor): (P, 2)
            multi_part (bool, optional): _description_. Defaults to False.

        Returns:
            eval_result (dict):
                - cd (float): CD between prediction & ground-truth
                - rrmse (float): MSE between prediction & ground-truth for rotation (in degree)
                - trmse (float): MSE between prediction & ground-truth for translation (in cm)
                - crd (float): CoRrespondence Distance (CRD) betwween prediction & ground-truth
        """

        # Init return buffer
        eval_result = {}
        
        pred_relative_trsfm = out_dict['estimated_rotat'].float(), out_dict['estimated_trans'].float() # (3, 3), (3)
        grtr_relative_trsfm = [x.squeeze(0) for x in in_dict['relative_trsfm']['0-1']] # (1, 3, 3) -> (3, 3), (1, 3) -> (3)
        src_pcd, trg_pcd = [x.squeeze(0) for x in in_dict['pcd_t']] # (1, N, 3) -> (N, 3), (1, M, 3) -> (M, 3)
        is_trg_larger = self._is_trg_larger(src_pcd, trg_pcd)

        # Assemble using prediction, pseudo-gt, and ground-truth
        assm_pred, pcds_pred = self._pairwise_mating(src_pcd, trg_pcd, pred_relative_trsfm[0], pred_relative_trsfm[1], is_trg_larger)
        assm_grtr, pcds_grtr = self._pairwise_mating(src_pcd, trg_pcd, grtr_relative_trsfm[0], grtr_relative_trsfm[1], is_trg_larger)

        assm_pred, assm_grtr = assm_pred.float(), assm_grtr.float()
        
        # (a) Compute CD between prediction & ground-truth
        eval_result['cd'] = self._chamfer_distance(assm_pred, assm_grtr, is_trg_larger)

        # (b) Compute MSE between prediction & ground-truth for rotation (in degree) and translation
        eval_result['rrmse'], eval_result['trmse'] = self._transformation_error(pred_relative_trsfm, grtr_relative_trsfm, multi_part)

        # (c) Compute CoRrespondence Distance (CRD) betwween prediction & ground-truth
        eval_result['crd'] = self._correspondence_distance(assm_pred, assm_grtr, is_trg_larger)

        if self.visualize:
            vis_folder = './vis/expanded_normal_reverse'
            os.makedirs(vis_folder, exist_ok=True)

            pcds_pred.append(pcds_pred[0][gt_corr[:,0]])
            pcds_pred.append(pcds_pred[1][gt_corr[:,1]])
            pcds_grtr.append(pcds_grtr[0][gt_corr[:,0]])
            pcds_grtr.append(pcds_grtr[1][gt_corr[:,1]])
            save_pc(f'{vis_folder}/{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["crd"].item(),3)}_pred.pcd', pcds_pred)
            save_pc(f"{vis_folder}/{in_dict['eval_idx'].item()}_{in_dict['obj_class'][0]}_{round(eval_result['crd'].item(),3)}_grtr.pcd", pcds_grtr)

            # TODO, MESH AND FRAME VISUALIZATION

        return eval_result
    

    def _is_trg_larger(self, src_pcd, trg_pcd):
        """
        Args:
            src_pcd (torch.Tensor): (N, 3)
            trg_pcd (torch.Tensor): (M, 3)

        Returns:
            bool: True if source point cloud is smaller than target point cloud
        """
        # max - min -> volume
        # Calculate max - min for all xyz coordinates, and product for all xyz.
        # Finally, we can calculate bounding box volume
        src_volume = (src_pcd.max(dim=0)[0] - src_pcd.min(dim=0)[0]).prod(dim=0)
        trg_volume = (trg_pcd.max(dim=0)[0] - trg_pcd.min(dim=0)[0]).prod(dim=0)
        return src_volume < trg_volume
    

    def _pairwise_mating(self, src_pcd, trg_pcd, rotat, trans, is_trg_larger):
        """
        Args:
            src_pcd (torch.Tensor): (N, 3)
            trg_pcd (torch.Tensor): (M, 3)
            rotat (torch.Tensor): (3, 3)
            trans (torch.Tensor): (3)
            is_trg_larger (bool): True if source point cloud is smaller than target point cloud

        Returns:
            pcd_t (torch.Tensor): (N+M, 3)
            pcd_t (list): [(N, 3), (M, 3)] if is_trg_larger else [(N, 3), (M, 3)]
        """
        # Remind:
        # estimated_transform: source_point = R * target_point + t
        # estimated_rotat (rotat) = R.T, estimated_trans (trans) = R.T @ t

        pcd_t = []
        if is_trg_larger: # Fix target point, and move source point to target point
            # source_point = R * target_point + t -> src_pcd_t = R.T * src_pcd - (R.T @ t)
            # src_pcd_t = R.T * src_pcd - (R.T @ t)
            src_pcd_t = self._transform(src_pcd.squeeze(0), rotat, -trans, True)
            pcd_t = [src_pcd_t, trg_pcd.squeeze(0)]
        
        else: # Fix source point, and move target point to source point
            # source_point = R * target_point + t
            # However, estimated_rotat (rotat) = R.T, estimated_trans (trans) = R.T @ t
            # trg_pcd_t = R.T.T * (trg_pcd + (R.T @ t)) = R * (trg_pcd + (R.T @ t)) = R * trg_pcd + R * (R.T @ t) = R * trg_pcd + t
            trg_pcd_t = self._transform(trg_pcd.squeeze(0), rotat.T, trans, False)
            pcd_t = [src_pcd.squeeze(0), trg_pcd_t]
        
        return torch.cat(pcd_t, dim=0), pcd_t
    

    def _transform(self, pcd, rotat=None, trans=None, rotate_first=True):
        """
        Args:
            pcd (torch.Tensor): (N, 3)
            rotat (torch.Tensor, optional): (3, 3). Defaults to None.
            trans (torch.Tensor, optional): (3). Defaults to None.
            rotate_first (bool, optional): True if rotate first. Defaults to True.

        Returns:
            pcd_t (torch.Tensor): (N, 3)
        """
        if rotat == None: rotat = torch.eye(3, 3)
        if trans == None: trans = torch.zeros(3)

        rotat = rotat.to(pcd.device)
        trans = trans.to(pcd.device)

        if rotate_first:
            return torch.einsum('x y, n y -> n x', rotat, pcd) + trans
        else:
            return torch.einsum('x y, n y -> n x', rotat, pcd + trans)


    def _correspondence_distance(self, assm1, assm2, is_trg_larger, scaling=100):
        """
        Args:
            assm1 (torch.Tensor): (N, 3)
            assm2 (torch.Tensor): (M, 3)
            is_trg_larger (bool): True if source point cloud is smaller than target point cloud
            scaling (int, optional): Scaling factor for CD. Defaults to 100.

        Returns:
            corr_dist (torch.Tensor): (1)
        """
        corr_dist = (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling
        return corr_dist


    def _chamfer_distance(self, assm1, assm2, is_trg_larger, scaling=1000):
        """
        Args:
            assm1 (torch.Tensor): (N, 3)
            assm2 (torch.Tensor): (M, 3)
            is_trg_larger (bool): True if source point cloud is smaller than target point cloud
            scaling (int, optional): Scaling factor for CD. Defaults to 1000.

        Returns:
            cd (torch.Tensor): (1)
        """
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
        cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling
        return cd
    

    def _transformation_error(self, trnsf1, trnsf2, multi_part, rrmse_scaling=100):
        """
        Args:
            trnsf1 (tuple): (3, 3), (3)
            trnsf2 (tuple): (3, 3), (3)
            multi_part (bool): True if multi-part
            rrmse_scaling (int, optional): Scaling factor for RMSE. Defaults to 100.

        Returns:
            rrmse (torch.Tensor): (1)
            trmse (torch.Tensor): (1)
        """
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
        return (rrmse / div).to(trmse.device), trmse / div


