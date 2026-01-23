import pytorch_lightning as pl
from functools import partial


import torch
import torch.nn as nn
import torch.optim as optim

from model.backbone.vn_dgcnn import EQCNN_equi_unet, EQCNN_equi_unet_v2
from model.backbone.vn_layers import VNLinear, VNLinearLeakyReLU
from model.backbone.simple_mlps import return_simple_mlps
from model.loss import CircleLoss, PointMatchingLoss, OrientationLoss, DiceLoss, binary_cross_entropy_loss
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from model.local_global_registration import LocalGlobalRegistration
from model.eval_utils import run_evaluation
from model.op_utils import *
from model.multi_part_op_utils import *

from RANSAC.ransac import _RANSAC

from common.metric_utils import *
from common.misc import batch_scaling, extract_all_objects_by_offset, batch2offset
from common.utils import instance_wise_results_to_json, divide_parameters_into_ori_and_others, pairwise_mating, save_json
from common.viz import visualize_negative_hard_mask, draw_test_results_histogram, save_pc

import os
import gtsam


class EquiAssem(pl.LightningModule):
    def __init__(
            self, 
            lr, 
            ori_backbone_lr_weight=1.0,
            scheduler_mode='cos',
            backbone='vn_unet', 
            double_bacbone='none',
            seg_head_mode='none',
            
            # Circle loss and point matching loss arguments
            pos_radius=0.018, 
            safe_radius=0.03, 
            
            # Circle loss arguments
            pos_margin=0.1, 
            neg_margin=1.4, 
            pos_offset=0.0,
            neg_offset=0.0,
            log_scale=24, 
            balance_mode='none', 
            hard_negative='none',
            neg_topk=0,
            distance_type='l2',
            anchor_mode='default',
            more_hard_neg=False,
            start_hard_neg_epoch=-1,

            s_loss_weight=1.0, 
            p_loss_weight=1.0, 
            o_loss_weight=1.0,
            seg_loss_weight=1.0,
            seg_loss_mode='bce',

            visualize_mode='none', 
            viz_metric_name='none',
            viz_metric_threshold=0.0,
            viz_train_epoch=0,
            viz_epoch=30, 
            viz_max_arrow_num=0, 
            ckp_dir=None, 
            debug=False,
            success_criterion_in_degree=10,
            only_train_normal=False,
            flip_normal_mode='none',
            consistency_loss_weight=0.0,
            
            n_knn=20,
            m_knn=1.0,
            r_knn=0.0,
            only_one_norm=False,
            n_avn=5,
            mlp_mode='CMpp',
            normal_pred_mode='cross',
            move_smaller=False,

            matching_score_mode='CM',
            matching_norm_mode='sinkhorn',
            learnable_softmax_temperature=False,
            
            # RANSAC arguments
            infer_match_option='topk',
            infer_topk=128,
            infer_score_threshold_ratio=0.0,
            use_RANSAC=False,
            RANSAC_type='default',
            use_predicted_normal=False,
            use_seg_result=False,
            cos_threshold=0.0,
            multi_part_assembly='none',
            ):
        """Equivariant Assembly Model for 3D Object Assembly

        Args:
            lr (float): Learning rate for optimizer.
            ori_backbone_lr_weight (float, optional): Learning rate weight for the original backbone. Defaults to 1.0.
            scheduler_mode (str, optional): Scheduler type ('cos', 'onecycle', 'none). Defaults to 'cos'.
            backbone (str, optional): Backbone network architecture. Defaults to 'vn_unet'.
            double_bacbone (str, optional): 'none' or 'vn_unet'. Defaults to 'none'.
            seg_head_mode (str, optional): 'none' or 'mlp' or 'atten'. Defaults to 'none'.

            # Circle loss and point matching loss arguments
            pos_radius (float, optional): Radius for positive samples in Circle loss computation and point matching loss. Defaults to 0.018.
            safe_radius (float, optional): Radius for safe samples in Circle loss computation. Defaults to 0.03.

            # Circle loss arguments
            pos_margin (float, optional): Margin for positive samples in loss computation. Defaults to 0.1.
            neg_margin (float, optional): Margin for negative samples in loss computation. Defaults to 1.4.
            pos_offset (float, optional): Offset for positive samples in loss computation. Defaults to 0.0.
            neg_offset (float, optional): Offset for negative samples in loss computation. Defaults to 0.0.
            log_scale (int, optional): Log scaling factor for loss computation. Defaults to 24.
            balance_mode (str, optional): 'none' or 'half' or 'all_hard' or 'double'. Defaults to 'none'.
            hard_negative (str, optional): 'none' or 'mix' or 'topk'. Defaults to 'none'.
            negative (str, optional): 'none' or 'topk'. Defaults to 'none'.
            distance_type (str, optional): 'l2' or 'cossim'. Defaults to 'l2'.
            anchor_mode (str, optional): 'default' or 'all_pos'. Defaults to 'default'.
            more_hard_neg (bool, optional): Whether to use more hard negative samples. Defaults to False.
            start_hard_neg_epoch (int, optional): Start hard negative sampling from this epoch. Defaults to -1.

            s_loss_weight (float, optional): Weight for shape loss. Defaults to 1.0.
            p_loss_weight (float, optional): Weight for point matching loss. Defaults to 1.0.
            o_loss_weight (float, optional): Weight for orientation loss. Defaults to 1.0.
            seg_loss_weight (float, optional): Weight for segmentation loss. Defaults to 0.1.
            seg_loss_mode (str, optional): 'dice' or 'bce'. Defaults to 'bce'.
            
            visualize_mode (str, optional): 'none' or 'light' or 'all'. Defaults to 'none'.
            viz_metric_name (str, optional): 'none' or 'crd' or 'cd' or 'rrmse_geo' or 'trmse_geo'. Defaults to 'none'.
            viz_metric_threshold (float, optional): Threshold for visualization. Defaults to 0.0.
            viz_train_epoch (int, optional): Epoch for visualizing the negative hard mask during training. Defaults to 0.
            viz_epoch (int, optional): Epoch for mesh visualization. Defaults to 30.
            viz_max_arrow_num (int, optional): Maximum number of arrows for visualization. Defaults to 0.
            ckp_dir (str, optional): Checkpoint directory. Defaults to None.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            success_criterion_in_degree (int, optional): Success criterion in degree for normal error. Defaults to 10.
            only_train_normal (bool, optional): Whether to only train the normal vector, it will be used for stage 1 training. Defaults to False.
            flip_normal_mode (str, optional): 'none' or 'right' or 'rightv1_2' or 'rightv2' or 'rightv3' or 'rightv4' or 'rightv5' or 'mix'. Defaults to 'none'.
            consistency_loss_weight (float, optional): Weight for consistency loss. Defaults to 0.0.
            one_to_one_consistency (bool, optional): Whether to use one-to-one consistency loss. Defaults to False.

            n_knn (int, optional): Number of nearest neighbors for KNN. Defaults to 20.
            m_knn (float, optional): Multiplicative factor for KNN, which corresponds to the number of points in the point cloud. Defaults to 1.0.
            r_knn (float, optional): Radius for KNN. Defaults to 0.0.
            only_one_norm (bool, optional): Whether to use only one Normalization layer for the equivariant shape feature. Defaults to False.
            n_avn (int, optional): Number of AVN layers for the equivariant shape feature. Defaults to 5.
            mlp_mode (str, optional): 'CMpp' or 'half' or 'deep'. Defaults to 'CMpp'.
            normal_pred_mode (str, optional): 'cross' or 'gram'. Defaults to 'cross'.
            move_smaller (bool, optional): Whether to always move the smaller point cloud to the origin. Defaults to False.

            matching_score_mode (str, optional): 'CM' or 'cossim'. Defaults to 'CM'.
            matching_norm_mode (str, optional): ['sinkhorn', 'softmax', 'none']. Defaults to 'sinkhorn'.
            learnable_softmax_temperature (bool, optional): Whether to use learnable temperature for softmax. Defaults to False.

            # RANSAC arguments
            infer_match_option (str, optional): 'topk' or 'mutual_topk' or 'soft_topk' or 'unidirectional_topk' or 'injective' or 'bijective'. Defaults to 'topk'.
            infer_topk (int, optional): Topk value for matching. Defaults to 128.
            infer_score_threshold_ratio (float, optional): Score threshold ratio for filtering correspondences. Defaults to 0.01.
            use_RANSAC (bool, optional): Whether to use RANSAC for transformation estimation. Defaults to False.
            RANSAC_type (str, optional): 'default' or 'score_dependent'. Defaults to 'default'.
            use_predicted_normal (bool, optional): Whether to use predicted normal for inlier counting. Defaults to False.
            use_seg_result (bool, optional): Whether to use segmentation result for matching. Defaults to False.
            cos_threshold (float, optional): Threshold for cosine similarity. This is used only during multi-part assembly. Defaults to 0.0.
            multi_part_assembly (str, optional): 'none' or 'naive' or 'shonan'. Defaults to 'none'.
        """
        super(EquiAssem, self).__init__()

        print("------------------------------------------------------")
        print("INITIALIZING EquiAssem(pl.LightningModule)")
        print("------------------------------------------------------")
        print(f"lr: {lr}")
        print(f"ori_backbone_lr_weight: {ori_backbone_lr_weight}")
        print(f"scheduler_mode: {scheduler_mode}")
        print(f"backbone: {backbone}")
        print(f"double_bacbone: {double_bacbone}")
        print(f"seg_head_mode: {seg_head_mode}")
        
        # Circle loss parameters will be printed in CircleLoss initialization
        # Point matching loss parameters will be printed in PointMatchingLoss initialization

        print(f"s_loss_weight: {s_loss_weight}")
        print(f"p_loss_weight: {p_loss_weight}")
        print(f"o_loss_weight: {o_loss_weight}")
        print(f"seg_loss_weight: {seg_loss_weight}")
        
        print(f"visualize_mode: {visualize_mode}")
        print(f"viz_metric_name: {viz_metric_name}")
        print(f"viz_metric_threshold: {viz_metric_threshold}")
        print(f"viz_train_epoch: {viz_train_epoch}")
        print(f"viz_epoch: {viz_epoch}")
        print(f"viz_max_arrow_num: {viz_max_arrow_num}")
        print(f"ckp_dir: {ckp_dir}")
        print(f"debug: {debug}")
        print(f"success_criterion_in_degree: {success_criterion_in_degree}")
        print(f"only_train_normal: {only_train_normal}")
        print(f"flip_normal_mode: {flip_normal_mode}")
        print(f"consistency_loss_weight: {consistency_loss_weight}")

        print(f"n_knn: {n_knn}")
        print(f"m_knn: {m_knn}")
        print(f"r_knn: {r_knn}")
        print(f"only_one_norm: {only_one_norm}")
        print(f"n_avn: {n_avn}")
        print(f"mlp_mode: {mlp_mode}")
        print(f"normal_pred_mode: {normal_pred_mode}")
        print(f"move_smaller: {move_smaller}")

        print(f"matching_score_mode: {matching_score_mode}")
        print(f"matching_norm_mode: {matching_norm_mode}")
        print(f"learnable_softmax_temperature: {learnable_softmax_temperature}")

        # RANSAC arguments
        print(f"infer_match_option: {infer_match_option}")
        print(f"infer_topk: {infer_topk}")
        print(f"infer_score_threshold_ratio: {infer_score_threshold_ratio}")
        print(f"use_RANSAC: {use_RANSAC}")
        print(f"RANSAC_type: {RANSAC_type}")
        print(f"use_predicted_normal: {use_predicted_normal}")
        print(f"use_seg_result: {use_seg_result}")
        print(f"cos_threshold: {cos_threshold}")
        print(f"multi_part_assembly: {multi_part_assembly}")
        print("------------------------------------------------------")

        self.lr = lr
        self.ori_backbone_lr_weight = ori_backbone_lr_weight
        self.scheduler_mode = scheduler_mode
        self.visualize_mode = visualize_mode
        self.viz_metric_name = viz_metric_name
        self.viz_metric_threshold = viz_metric_threshold
        self.viz_train_epoch = viz_train_epoch
        self.viz_epoch = viz_epoch
        self.viz_max_arrow_num = viz_max_arrow_num
        self.ckp_dir = ckp_dir
        self.debug = debug
        self.success_criterion_in_degree = success_criterion_in_degree
        self.only_train_normal = only_train_normal
        self.flip_normal_mode = flip_normal_mode
        self.normal_pred_mode = normal_pred_mode
        self.seg_head_mode = seg_head_mode

        self.move_smaller = move_smaller

        self.matching_score_mode = matching_score_mode
        self.matching_norm_mode = matching_norm_mode
        self.learnable_softmax_temperature = learnable_softmax_temperature

        # Inference arguments
        self.infer_match_option = infer_match_option
        self.infer_topk = infer_topk
        self.infer_score_threshold_ratio = infer_score_threshold_ratio
        self.use_RANSAC = use_RANSAC
        self.RANSAC_type = RANSAC_type
        self.use_predicted_normal = use_predicted_normal
        self.use_seg_result = use_seg_result
        self.cos_threshold = cos_threshold
        self.multi_part_assembly = multi_part_assembly
        
        # Output feature dimension of Feature Extractor
        self.feat_dim = 1024
        
        # Objectives
        self.pos_radius = pos_radius
        self.safe_radius = safe_radius
        self.circle_loss = CircleLoss(pos_radius=pos_radius, safe_radius=safe_radius, 
                                      log_scale=log_scale, pos_margin=pos_margin, neg_margin=neg_margin, 
                                      pos_offset=pos_offset, neg_offset=neg_offset,
                                      balance_mode=balance_mode, hard_negative=hard_negative,
                                      neg_topk=neg_topk, more_hard_neg=more_hard_neg, distance_type=distance_type, anchor_mode=anchor_mode,
                                      start_hard_neg_epoch=start_hard_neg_epoch)
        self.orientation_loss = OrientationLoss(consistency_loss_weight=consistency_loss_weight, pos_radius=pos_radius, flip_normal_mode=flip_normal_mode)
        self.matching_loss = PointMatchingLoss(pos_radius=pos_radius, safe_radius=safe_radius)
        

        # Weights for losses
        self.s_loss_weight = s_loss_weight # circle loss weight
        self.p_loss_weight = p_loss_weight # point matching loss weight
        self.o_loss_weight = o_loss_weight # orientation loss weight
        self.seg_loss_weight = seg_loss_weight # segmentation loss weight
        self.seg_loss_mode = seg_loss_mode # segmentation loss mode

        print("------------------------------------------------------")
        print("Weight for losses")
        print(f"s_loss_weight: {self.s_loss_weight}")
        print(f"p_loss_weight: {self.p_loss_weight}")
        print(f"o_loss_weight: {self.o_loss_weight}")
        print(f"seg_loss_weight: {self.seg_loss_weight}")
        print(f"seg_loss_mode: {self.seg_loss_mode}")
        print("------------------------------------------------------")


        # Logging
        self.validation_step_outputs = []
        self.test_step_outputs = []


        # Declare Modules
        # VN BACKBONE
        if backbone == 'vn_unet':
            self.backbone = EQCNN_equi_unet(feat_dim=self.feat_dim, pooling="mean", k=n_knn, m=m_knn, r=r_knn)
        elif backbone == 'vn_unet_v2':
            self.backbone = EQCNN_equi_unet_v2(feat_dim=self.feat_dim, pooling="mean", k=n_knn, m=m_knn, r=r_knn)
        else:
            raise NotImplementedError("DGCNN backbone not implemented")
        
        if double_bacbone == 'vn_unet':
            self.ori_backbone = EQCNN_equi_unet(feat_dim=self.feat_dim, pooling="mean", k=n_knn, m=m_knn, r=r_knn)
        elif double_bacbone == 'vn_unet_v2':
            self.ori_backbone = EQCNN_equi_unet_v2(feat_dim=self.feat_dim, pooling="mean", k=n_knn, m=m_knn, r=r_knn)
        elif double_bacbone == 'none':
            self.ori_backbone = None
        else:
            raise NotImplementedError("DGCNN backbone not implemented")

        # Layer for predicting frame vectors
        if self.normal_pred_mode == 'cross':
            self.proj = VNLinear(2 * (self.feat_dim//3), 2)
        elif self.normal_pred_mode == 'gram':
            self.proj = VNLinear(2 * (self.feat_dim//3), 3)
        else:
            raise ValueError(f"normal_pred_mode must be in ['cross', 'gram'], but got {self.normal_pred_mode}")

        # Layer for Equivariant feature
        if n_avn > 0:
            self.equi_layer = nn.Sequential(*([VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3, no_norm=False)] + [VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3, no_norm=only_one_norm) for _ in range(n_avn-1)]))
        else:
            self.equi_layer = nn.Identity()
        
        # Layer for invariant shape features
        self.shape_mlp, channel_dim_of_shape_feats = return_simple_mlps(mlp_mode, self.feat_dim)

        # Segmentation head
        if seg_head_mode == 'none':
            self.seg_head = None
        elif seg_head_mode == 'mlp': # This is based on GARF
            self.seg_head = nn.Sequential(nn.Conv1d(channel_dim_of_shape_feats, 16, kernel_size=1, bias=True),
                                          nn.ReLU(inplace=False),
                                          nn.Conv1d(16, 1, kernel_size=1, bias=True)
                                          )

        elif seg_head_mode == 'atten': # This is based on GARF
            self.layer_norm_for_self_atten = nn.Sequential(nn.SiLU(), nn.LayerNorm(channel_dim_of_shape_feats, elementwise_affine=False))
            self.layer_norm_for_global_atten = nn.Sequential(nn.SiLU(), nn.LayerNorm(channel_dim_of_shape_feats, elementwise_affine=False))
            self.final_layer_norm = nn.Sequential(nn.SiLU(), nn.LayerNorm(channel_dim_of_shape_feats, elementwise_affine=False))
            self.self_attn_to_qkv = nn.Linear(channel_dim_of_shape_feats, channel_dim_of_shape_feats * 3, bias=False)
            self.global_attn_to_qkv = nn.Linear(channel_dim_of_shape_feats, channel_dim_of_shape_feats * 3, bias=False)
            self.seg_head = nn.Linear(channel_dim_of_shape_feats, 1, bias=True)

        else:
            raise ValueError(f"seg_head_mode must be in ['none', 'mlp', 'atten'], but got {seg_head_mode}")
        
        if seg_head_mode != 'none':
            self.seg_loss_func = binary_cross_entropy_loss if self.seg_loss_mode == 'bce' else DiceLoss
        
        # Optimal Transport
        if self.matching_norm_mode == 'sinkhorn':
            self.optimal_transport = LearnableLogOptimalTransport(num_iterations=100)
        elif self.matching_norm_mode == 'softmax':
            self.register_parameter('slack_variable', torch.nn.Parameter(torch.tensor(1.0)))

            if self.learnable_softmax_temperature:
                self.register_parameter('softmax_temperature', torch.nn.Parameter(torch.tensor(2.0)))
            else:
                self.softmax_temperature = 1.0 # We do not use temperature for softmax
        
        if not self.use_RANSAC: # If not using RANSAC, use LGR for fine matching
            # LGR
            self.fine_matching = LocalGlobalRegistration(
                k=self.infer_topk,
                match_option=self.infer_match_option,
                acceptance_radius=0.1,
                num_refinement_steps=5,
                score_threshold_ratio=self.infer_score_threshold_ratio,
            )
    

    def freeze_ori_backbone(self, freeze_ori_2nd_stage):
        assert freeze_ori_2nd_stage in ['all', 'no_2nd'], "freeze_ori_2nd_stage must be in ['all', 'no_2nd']"

        if freeze_ori_2nd_stage == 'all':
            if self.ori_backbone is not None:
                for param in self.ori_backbone.parameters():
                    param.requires_grad = False
            
            for param in self.proj.parameters():
                param.requires_grad = False
        
        else: # freeze_ori_2nd_stage == 'no_2nd'
            if self.ori_backbone is not None:
                for param in self.ori_backbone.parameters():
                    param.requires_grad = False
            
            for name, param in self.proj.named_parameters():
                param.requires_grad = True
                # Register hook to set the gradient of the 1st row to 0
                def make_hook(row_idx):
                    def hook(grad):
                        if grad is not None:
                            grad = grad.clone()
                            grad[row_idx, :] = 0 
                        return grad
                    return hook
                param.register_hook(make_hook(0))
    

    def freeze_all_except_seg_head(self):
        assert self.seg_head is not None, "Segmentation head is not defined"
        seg_head_module_names = ['seg_head', 'layer_norm_for_self_atten', 'layer_norm_for_global_atten', 'final_layer_norm', 'self_attn_to_qkv', 'global_attn_to_qkv']
        
        for name, param in self.named_parameters():
            if any(name.startswith(module_name) for module_name in seg_head_module_names):
                pass
            
            else:
                param.requires_grad = False
    

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        # Lightning 2.x: Support this funcionality
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        max_epochs = self.trainer.max_epochs

        assert total_steps > 0, "Total steps must be greater than 0"

        
        if self.learnable_softmax_temperature:
            if self.ori_backbone is not None:
                ori_parameters, other_parameters = divide_parameters_into_ori_and_others(self.named_parameters())
                optimizer = torch.optim.AdamW([
                    {'params': other_parameters, 'lr': self.lr},
                    {'params': ori_parameters, 'lr': self.lr * self.ori_backbone_lr_weight},
                    {'params': self.softmax_temperature, 'lr': self.lr * 0.1}
                    ],  lr=self.lr, weight_decay=0.) # We use 10% of the learning rate for softmax temperature
            else:
                optimizer = torch.optim.AdamW([
                    {'params': [p for n, p in self.named_parameters() if 'softmax_temperature' not in n]},
                    {'params': self.softmax_temperature, 'lr': self.lr * 0.1} 
                    ],  lr=self.lr, weight_decay=0.) # We use 10% of the learning rate for softmax temperature
        else:
            if self.ori_backbone is not None:
                ori_parameters, other_parameters = divide_parameters_into_ori_and_others(self.named_parameters())
                optimizer = optim.AdamW([
                    {'params': other_parameters, 'lr': self.lr},
                    {'params': ori_parameters, 'lr': self.lr * self.ori_backbone_lr_weight},
                    ],  lr=self.lr, weight_decay=0.)
            else:
                optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.)
            
        print(f"optimizer: {optimizer}")
        
        if self.scheduler_mode == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-3)
            
        elif self.scheduler_mode == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.lr, epochs=max_epochs, steps_per_epoch=steps_per_epoch,
                                                      pct_start=0.05, anneal_strategy="cos", div_factor=10.0,
                                                      final_div_factor=1000.0)
        
        else: # none
            scheduler = None

        if scheduler is not None:
            return {
                    'optimizer': optimizer, 
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',  # Update scheduler every step
                        'frequency': 1
                        }
                    }
        else:
            return {'optimizer': optimizer}


    def on_train_epoch_start(self):
        self.circle_loss.update_start_hard_neg_epoch(self.current_epoch)
    

    def training_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='train', batch_idx=batch_idx)
        if torch.isnan(loss_dict['loss']):
            assert False, "Loss is NaN, Stop training"
        return loss_dict['loss']
    

    def validation_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='val')
        if torch.isnan(loss_dict['loss']):
            assert False, "Loss is NaN, Stop validation"
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
        """
        if batch_idx == 0:
            _, loss_dict = self.forward_pass(in_dict, mode='test')
            self.test_step_outputs.append(loss_dict)
        else:
            exit("stop")
        """

        if self.multi_part_assembly != 'none':
            assert in_dict['num_parts'][0] >= 2, f"num_parts must be greater than or equal to 2, but got {in_dict['num_parts'][0]}"
            loss_dict = self.forward_pass_for_multiple_parts(in_dict, mode='test', infer_mode=self.multi_part_assembly)
        
        else:
            assert in_dict['num_parts'][0] == 2, f"num_parts must be 2, but got {in_dict['num_parts'][0]}"
            _, loss_dict = self.forward_pass(in_dict, mode='test')
        
        self.test_step_outputs.append(loss_dict)
        return loss_dict


    def on_test_epoch_end(self):
        print(f"self.global_rank: {self.trainer.global_rank} DONE")

        instance_score_dict = dict()
        
        for output in self.test_step_outputs:
            metric_dict = dict()
            
            for k, v in output.items():
                if k == 'filepath':
                    continue
                else:
                    metric_dict[k] = v
            
            instance_score_dict[output['filepath']] = metric_dict
        
        # Multi-GPU support
        if self.trainer.world_size > 1:
            place_holder_list = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(place_holder_list, instance_score_dict)

            total_instance_score_dict = dict()
            for gpu_i_result in place_holder_list:
                total_instance_score_dict.update(gpu_i_result)
        else:
            total_instance_score_dict = instance_score_dict
        
        print(f"self.global_rank: {self.trainer.global_rank} ALL GATHER OBJECT DONE")

        if self.trainer.global_rank == 0:
            result_avg_dict = dict()
            for metric_name in total_instance_score_dict[list(total_instance_score_dict.keys())[0]].keys():
                result_avg_dict[f'val/{metric_name}'] = torch.stack([output[metric_name].cpu() for output in total_instance_score_dict.values()])
            
            avg_result = {k: v.sum() / v.size(0) for k, v in result_avg_dict.items()}
            self.test_results = avg_result
            self.log_dict(avg_result, logger=True, sync_dist=False, batch_size=1,)
            
            # Json dump for instance-wise results
            instance_wise_results_to_json(total_instance_score_dict, self.ckp_dir, 'test_results')

            # Json dump for test results
            save_json(self.test_results, os.path.join(self.ckp_dir, 'total.json'))

            # Make histogram for each metric
            draw_test_results_histogram(total_instance_score_dict, self.ckp_dir, 'test_metrics_histogram')

        
        self.test_step_outputs.clear()

        print(f"self.global_rank: {self.trainer.global_rank} LOG_DICT DONE")
        
        # Wait for all processes to reach this point
        if self.trainer.world_size > 1:
            torch.distributed.barrier()
    
    
    # @torch.no_grad()
    def forward_pass(self, in_dict, mode, batch_idx=0):
        """
        Args:
            Assumption: Batch size is 1

            in_dict (dict):
                - eval_idx (torch.Tensor): (B, )
                - filepath (list): (B, ), e.g. ['everyday/BeerBottle/2927d6c8438f6e24fe6460d8d9bd16c6/fractured_37']
                - obj_class (list): (B, ), e.g. ['BeerBottle']
                - n_frac (torch.Tensor): (B, )
                - anchor_idx (torch.Tensor): (B, )

                - pcd (torch.Tensor): (B, N+M, 3)
                - pcd_t (torch.Tensor): (B, N+M, 3)
                - gt_normals (torch.Tensor): (B, N+M, 3)
                - pcd_batch_info (torch.Tensor): (B, N+M,)

                - gt_rot_from_src_to_trg (torch.Tensor): (B, 3, 3)

                For evaluation
                    - mesh (list): length is 2, only for two pieces
                        - mesh[0]: (N', 3)
                        - mesh[1]: (M', 3)
                    - mesh_t (list): length is 2, only for two pieces
                        - mesh_t[0]: (N', 3)
                        - mesh_t[1]: (M', 3)
                    - mesh_faces (list): length is 2, only for two pieces
                        - mesh_faces[0]: (F_1, 3)
                        - mesh_faces[1]: (F_2, 3)
                    
                    - relative_trsfm (dict):
                        - key: relative_rotat, relative_trans
                            - relative_rotat: (3, 3)
                            - relative_trans: (3)

            mode (string): ['train', 'val', 'test']

            batch_idx (int): Batch index, only for visualization
        """
        assert in_dict['pcd_batch_info'].max() == 1, f"We assume there are two objects in the batch, but got {in_dict['pcd_batch_info'].max()}"

        out_dict, loss = {}, {}

        # 0. Get Point Clouds and Ground Truth Correspondence
        pcd_raw = in_dict['pcd'] # (B, N+M, 3)
        pcd_input = in_dict['pcd_t'] # (B, N+M, 3)
        gt_normals = in_dict['gt_normals'] # (B, N+M, 3)
        pcd_batch_info = in_dict['pcd_batch_info'] # (B, N+M, )
        batch_scaled_pcd_batch_info = batch_scaling(pcd_batch_info) # (B, N+M, )
        gt_rot_from_src_to_trg = in_dict['gt_rot_from_src_to_trg'] # (B, 3, 3)


        # Step 1 - 4
        # Get equivariant features and orientation matrices
        # equivariant features: (B, C, 3, N+M) , orientation matrices: (B, N+M, 3, 3)
        equi_feats, oris = get_feats_and_oris(self.backbone, self.ori_backbone, self.equi_layer, self.proj, self.normal_pred_mode, self.flip_normal_mode, self.only_train_normal, pcd_input, pcd_batch_info, batch_scaled_pcd_batch_info)
        out_dict['oris'] = oris


        # Only train the normal vector
        if self.only_train_normal:
            loss['o_loss'], consistency_loss_dict = self.orientation_loss(oris, gt_rot_from_src_to_trg, gt_normals, pcd_batch_info, None, pcd_raw, self.return_active_mask(pcd_batch_info))
            loss['loss'] = loss['o_loss']

            loss.update(consistency_loss_dict)

            # Compute Normal Error
            with torch.no_grad():
                # (d) Compute Normal Error
                loss['n_error'], _, loss['n_suc_rate'] = normal_error(in_dict, out_dict, success_criterion_in_degree=self.success_criterion_in_degree)
            
            if mode == 'train':
                self.log_for_training(loss=loss, pos_neg_distribution=None, mode=mode)
            
            return out_dict, loss

        
        # 5. Invariant Features
        inv_feats = make_inv_feats(oris, pcd_batch_info, equi_feats, self.flip_normal_mode, flip_mode='src') # (B, C*3, N+M)
        if self.flip_normal_mode != 'none' and mode in ['train', 'val']:
            symmetric_inv_feats = make_inv_feats(oris, pcd_batch_info, equi_feats, self.flip_normal_mode, flip_mode='trg') # (B, C*3, N+M)
        

        # 6. SHAPE DESCRIPTOR 
        shape_feats = self.shape_mlp(inv_feats) # (B, C*3, N+M) -> (B, D, N+M)
        if self.flip_normal_mode != 'none' and mode in ['train', 'val']:
            symmetric_shape_feats = self.shape_mlp(symmetric_inv_feats) # (B, C*3, N+M) -> (B, D, N+M)
        

        # [Optional] Segmentation Head
        mating_surface_seg_results = self.feed_forward_seg_head(shape_feats, batch_scaled_pcd_batch_info) if self.seg_head_mode != 'none' else None # (B, D, N+M) -> (B, N+M)
        

        # 7. Calculate Matching Scores
        active_mask = return_active_mask(pcd_batch_info)
        shape_matching_scores = calculate_matching_score(shape_feats, shape_feats, active_mask, eps=1e-8, mode=self.matching_score_mode)
        if self.flip_normal_mode != 'none' and mode in ['train', 'val']:
            symmetric_shape_matching_scores = calculate_matching_score(symmetric_shape_feats, symmetric_shape_feats, active_mask, eps=1e-8, mode=self.matching_score_mode)
        

        # 8. Optimal Transport
        # Optimal Transport is in log space, so inside registration, there is exp operation
        matching_scores = self.multibatch_optimal_transport(shape_matching_scores, pcd_batch_info, active_mask, mode=self.matching_norm_mode) # (B, N+M+1, N+M+1) if mode is ['sinkhorn', 'softmax'], otherwise (B, N+M, N+M)
        matching_scores_drop = matching_scores[:,:-1,:-1] if self.matching_norm_mode in ['sinkhorn', 'softmax'] else matching_scores # (B, N+M, N+M)
        if self.flip_normal_mode != 'none' and mode in ['train', 'val']:
            symmetric_matching_scores = self.multibatch_optimal_transport(symmetric_shape_matching_scores, pcd_batch_info, active_mask, mode=self.matching_norm_mode) # (B, N+M+1, N+M+1) if mode is ['sinkhorn', 'softmax'], otherwise (B, N+M, N+M)
        

        if mode in ['train', 'val']: # Do not calculate for test
            # 8. Calculate Loss
            if self.flip_normal_mode != 'none':
                src_move_circle_loss, coords_dist, pos_neg_distribution, neg_hard_mask_for_viz = self.circle_loss(pcd_raw, shape_feats, shape_matching_scores, active_mask)
                trg_move_circle_loss, _, _, _ = self.circle_loss(pcd_raw, symmetric_shape_feats, symmetric_shape_matching_scores, active_mask)

                src_move_matching_scores = self.matching_loss(matching_scores, coords_dist, active_mask, matching_norm_mode=self.matching_norm_mode).float() if self.p_loss_weight != 0 else torch.tensor(0.).to(matching_scores.device)
                trg_move_matching_scores = self.matching_loss(symmetric_matching_scores, coords_dist, active_mask, matching_norm_mode=self.matching_norm_mode).float() if self.p_loss_weight != 0 else torch.tensor(0.).to(symmetric_matching_scores.device)

                loss['s_loss'] = (src_move_circle_loss + trg_move_circle_loss) / 2
                loss['p_loss'] = (src_move_matching_scores + trg_move_matching_scores) / 2
            
            else:
                loss['s_loss'], coords_dist, pos_neg_distribution, neg_hard_mask_for_viz = self.circle_loss(pcd_raw, shape_feats, shape_matching_scores, active_mask)
                loss['p_loss'] = self.matching_loss(matching_scores, coords_dist, active_mask, matching_norm_mode=self.matching_norm_mode).float() if self.p_loss_weight != 0 else torch.tensor(0.).to(matching_scores.device)
            

            loss['o_loss'], consistency_loss_dict = self.orientation_loss(oris, gt_rot_from_src_to_trg, gt_normals, pcd_batch_info, coords_dist, pcd_raw, active_mask)
            loss['seg_loss'] = self.seg_loss_func(mating_surface_seg_results, coords_dist, active_mask, pos_radius=self.pos_radius) if self.seg_head_mode != 'none' else torch.tensor(0.).to(matching_scores.device)
            loss['loss'] = self.o_loss_weight * loss['o_loss'] + self.s_loss_weight * loss['s_loss'] + self.p_loss_weight * loss['p_loss'] + self.seg_loss_weight * loss['seg_loss']
            
            out_dict.update(loss)
            loss.update(consistency_loss_dict)

            if mode == 'train':
                with torch.no_grad():
                    # This is for checking the normal error
                    loss['n_error'], _, loss['n_suc_rate'] = normal_error(in_dict, out_dict, success_criterion_in_degree=self.success_criterion_in_degree)

                    if ((not self.trainer.sanity_checking) and self.viz_train_epoch > 0 and batch_idx == 0 and (self.current_epoch % self.viz_train_epoch == 0 or self.current_epoch == self.trainer.max_epochs-1)):
                        visualize_negative_hard_mask(in_dict, neg_hard_mask_for_viz['neg_mask'], neg_hard_mask_for_viz['hard_neg_mask'], active_mask, self.ckp_dir, self.current_epoch, self.trainer.global_rank, self.pos_radius, self.safe_radius)
                        # exit("stop")
        

        # 9. Evaluation
        if mode in ['val', 'test']:
            # Save output for evaluation
            out_dict['shape_matching_scores'] = shape_matching_scores
            out_dict['matching_scores_drop'] = matching_scores_drop
            out_dict['active_mask'] = active_mask
            out_dict['mating_surface_seg_results'] = mating_surface_seg_results
            out_dict, eval_dict = self.evaluate_prediction(in_dict, out_dict, mode)
            loss.update(eval_dict)

            if mode == 'test':
                loss['filepath'] = in_dict['filepath'][0]

        # in training we log for every step
        if mode == 'train':
            self.log_for_training(loss=loss, pos_neg_distribution=pos_neg_distribution, mode=mode)
        else:
            torch.cuda.empty_cache()
        
        return out_dict, loss    
    
    
    def log_for_training(self, loss, pos_neg_distribution, mode):
        log_dict = {f'{mode}/{k}': v.item() for k, v in loss.items()}
        
        if pos_neg_distribution is not None:
            log_pos_neg_distribution = {f'{mode}-dist/{k}': v for k, v in pos_neg_distribution.items()}
            log_dict.update(log_pos_neg_distribution)
        
        if self.matching_norm_mode == 'softmax':
            log_dict[f'{mode}/softmax_temperature'] = self.softmax_temperature.item() if self.learnable_softmax_temperature else self.softmax_temperature

        training_loss = log_dict.pop(f'{mode}/loss')
        another_lr = {}
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            if i == 0:
                current_lr = param_group['lr']
            else:
                another_lr[f'param_group_{i}'] = param_group['lr']
        
        self.log_dict(log_dict, prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=True)
        self.log(f'{mode}/loss', training_loss, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=True)
        self.log('current_lr', current_lr, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False)
        self.log_dict(another_lr, prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False)
    
    

    def feed_forward_seg_head(self, shape_feats, batch_scaled_pcd_batch_info):
        """
        Feed forward the shape features through the segmentation head
        We assume there are two objects in the batch
        When atten, head size is 8

        Args:
            shape_feats (torch.Tensor): (B, D, N+M)
            batch_scaled_pcd_batch_info (torch.Tensor): (B, N+M, ), batch index of the point cloud

        Returns:
            mating_surface_seg_results (torch.Tensor): (B, N+M)
        """
        modules_dict = {
            'seg_head': self.seg_head,
            'layer_norm_for_self_atten': self.layer_norm_for_self_atten if self.seg_head_mode == 'atten' else None,
            'layer_norm_for_global_atten': self.layer_norm_for_global_atten if self.seg_head_mode == 'atten' else None,
            'final_layer_norm': self.final_layer_norm if self.seg_head_mode == 'atten' else None,
            'self_attn_to_qkv': self.self_attn_to_qkv if self.seg_head_mode == 'atten' else None,
            'global_attn_to_qkv': self.global_attn_to_qkv if self.seg_head_mode == 'atten' else None,
        }

        mating_surface_seg_results = do_feed_forward_seg_head(seg_head_mode=self.seg_head_mode, shape_feats=shape_feats, batch_scaled_pcd_batch_info=batch_scaled_pcd_batch_info, modules_dict=modules_dict)
        return mating_surface_seg_results
    

    def multibatch_optimal_transport(self, matching_scores, batch_info, active_mask, mode='sinkhorn'):
        """
        Calculate optimal transport between multiple batches

        Args:
            matching_scores (torch.Tensor): (B, N+M, N+M), inactive parts are already removed
            batch_info (torch.Tensor): (B, N+M, ), batch index of the point cloud
            active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active
            mode (str, optional): 'sinkhorn', 'softmax', 'none'. Defaults to 'sinkhorn'.
        
        Returns:
            result (torch.Tensor): 
            - (B, N+M+1, N+M+1) if mode is ['sinkhorn', 'softmax']
            - (B, N+M, N+M) if mode is 'none'
        """
        auxiliary_info_dict = {
            'optimal_transport': self.optimal_transport if mode == 'sinkhorn' else None,
            'softmax_temperature': self.softmax_temperature if mode == 'softmax' else None,
            'slack_variable': self.slack_variable if mode == 'softmax' else None,
        }

        result = do_multibatch_optimal_transport(matching_scores, batch_info, active_mask, auxiliary_info_dict, mode=mode)
        return result
    

    @torch.no_grad()
    def evaluate_prediction(self, in_dict, out_dict, mode):

        # Prepare settings for evaluation
        settings_for_evaluation = {
            'ckp_dir': self.ckp_dir,
            'pos_radius': self.pos_radius,
            'seg_head_mode': self.seg_head_mode,
            'use_seg_result': self.use_seg_result,
            'success_criterion_in_degree': self.success_criterion_in_degree,
            'move_smaller': self.move_smaller,
            'use_RANSAC': self.use_RANSAC,
            'use_predicted_normal': self.use_predicted_normal,

            'infer_topk': self.infer_topk,
            
            'visualize_mode': self.visualize_mode,
            'viz_metric_name': self.viz_metric_name,
            'viz_metric_threshold': self.viz_metric_threshold,
            'viz_max_arrow_num': self.viz_max_arrow_num,
            'viz_epoch': self.viz_epoch,
            
            'trainer_sanity_checking': self.trainer.sanity_checking,
            'trainer_global_rank': self.trainer.global_rank,
            'current_epoch': self.current_epoch,  
            'trainer.max_epochs': self.trainer.max_epochs,
        }

        # Prepare function for prediction rotation and translation
        if self.use_RANSAC:
            func_for_rot_and_trans = partial(_RANSAC, match_option=self.infer_match_option, RANSAC_type=self.RANSAC_type, topk=self.infer_topk)
        else:
            func_for_rot_and_trans = partial(self.fine_matching, no_exp=(self.matching_norm_mode != 'sinkhorn'))

        out_dict, eval_dict = run_evaluation(in_dict=in_dict, out_dict=out_dict, settings_dict=settings_for_evaluation, func_for_pred=func_for_rot_and_trans, mode=mode)
        return out_dict, eval_dict
    

    def forward_pass_for_multiple_parts(self, in_dict, mode, infer_mode):
        """
        Assume batch size must be 1

        Args:
            in_dict (dict): input dictionary for forward pass, which is same as forward_pass
            mode (str): ['train', 'val', 'test']
            infer_mode (str): 'naive' or 'shonan'
        """
        assert mode in ['test'], f"mode must be in ['test'], but got {mode}"
        
        if infer_mode == 'naive':
            pred_rot_and_trans_dict, list_of_assembled_pcds, list_of_gt_assembled_pcds, step_collector_for_viz = self.assemble_obj_by_obj(in_dict)
        elif infer_mode == 'shonan':
            pred_rot_and_trans_dict, list_of_assembled_pcds, list_of_gt_assembled_pcds, step_collector_for_viz = self.assemble_shonan(in_dict)
        else:
            raise ValueError(f"infer_mode must be in ['naive', 'shonan'], but got {infer_mode}")

        # Compute metrics
        eval_result = compute_metrics(list_of_assembled_pcds, list_of_gt_assembled_pcds, pred_rot_and_trans_dict, in_dict['relative_trsfm'])
        eval_result['filepath'] = in_dict['filepath'][0]

        # Visualize results

        if (mode =='test' and (self.visualize_mode != 'none')):
            # Name of case
            case_name = in_dict["filepath"][0].replace('/', '_')

            vis_folder = os.path.join(self.ckp_dir, 'vis', f'GPU_{self.trainer.global_rank}', mode, case_name) # For mesh visualization
            vis_hist_folder = os.path.join(self.ckp_dir, 'vis_hist', f'GPU_{self.trainer.global_rank}', mode, case_name) # For normal error histogram visualization
            os.makedirs(vis_folder, exist_ok=True)
            os.makedirs(vis_hist_folder, exist_ok=True)

            # PCD light visualization
            save_pc(f"{vis_folder}/assembled_pcds.ply", list_of_assembled_pcds)
            save_pc(f"{vis_folder}/gt_assembled_pcds.ply", list_of_gt_assembled_pcds)

            # Step-by-step assembled PCD visualization
            for ith_step, step_pcds in enumerate(step_collector_for_viz):
                save_pc(f"{vis_folder}/step_{ith_step}.ply", step_pcds)
        
        exit("stop")
        
        return eval_result
    
    
    def assemble_obj_by_obj(self, in_dict):
        """
        Assume batch size must be 1

        Args:
            in_dict (dict): input dictionary for forward pass, which is same as forward_pass
        
        Returns:
            pred_rot_and_trans_dict (dict): dictionary of predicted rotations and translations
            list_of_assembled_pcds (list): list of assembled point clouds
            list_of_gt_assembled_pcds (list): list of GT assembled point clouds
            step_collector_for_viz (list): list of step-by-step assembled point clouds for visualization
        """

        pcd_input = in_dict['pcd_t'] # (B, N+M, 3)
        pcd_batch_info = in_dict['pcd_batch_info'] # (B, N+M, )
        offset = torch.concat([torch.zeros(1, device=pcd_batch_info.device, dtype=torch.int), batch2offset(pcd_batch_info[0])], dim=0) # size of parts
        num_of_parts = in_dict['num_parts'][0] # int
        initial_anchor_idx = in_dict['anchor_idx'][0] # int

        # Extract all point clouds
        list_of_all_pcds = extract_all_objects_by_offset(pcd_input[0], offset) # list of (N, 3)

        # Set initial setting
        # If there are N objs, then let us assume the anchor is the target object, and the rest N-1 objects are the another single huge source object
        setting_for_assembly = make_setting_for_next_iteration(None, list_of_all_pcds[in_dict['anchor_idx'][0]], list_of_all_pcds, in_dict['anchor_idx'][0], -1)


        pred_rot_and_trans = []
        step_collector_for_viz = []
        for i in range(num_of_parts - 1):
            # Calculate matching scores
            # setting_for_assembly['two_part_assumption_batch_info']: We assume there are two objects in the batch, so idx:0 means src, idx:1 means trg
            # Actually, there are N-1 objects in the source part, so we want to make VN-DGCNN run KNN only inside each object
            # Hence, we give setting_for_assembly['batch_scaled_pcd_batch_info'] to the backbone for distinguishing N-1 objects
            # However, for orientation, we want to assume that only source object flip in opposite direction, when it is needed, so we give setting_for_assembly['two_part_assumption_batch_info']
            oris, matching_scores_drop, shape_matching_scores, mating_surface_seg_results = self.return_matching_scores(setting_for_assembly['input_pcds'], setting_for_assembly['two_part_assumption_batch_info'], setting_for_assembly['batch_scaled_batch_info'])
            list_of_oris = extract_all_objects_by_offset(oris[0,:,0,:], setting_for_assembly['offset']) # list of (N, 3), only extract normals
            
            # Select object to be assembled
            list_of_all_src_to_trg_score, selected_obj_idx = select_obj_to_be_assembled(matching_scores_drop, setting_for_assembly['anchor'], setting_for_assembly['offset'], self.infer_topk)
            final_src_pcd = setting_for_assembly['list_of_input_pcds'][selected_obj_idx]
            final_trg_pcd = setting_for_assembly['list_of_input_pcds'][setting_for_assembly['anchor']]
            final_matching_scores_drop = list_of_all_src_to_trg_score[selected_obj_idx]

            # Calculate transformation
            if self.use_RANSAC:
                list_of_shape_matching_scores = make_score_into_list_format(shape_matching_scores, setting_for_assembly['anchor'], setting_for_assembly['offset'])
                final_shape_matching_scores = list_of_shape_matching_scores[selected_obj_idx]

                if self.use_predicted_normal:
                    list_of_frames = extract_all_objects_by_offset(oris[0,:,:,:], setting_for_assembly['offset']) # list of (N, 3, 3)
                    src_frame = list_of_frames[selected_obj_idx] # (N, 3, 3)
                    trg_frame = list_of_frames[setting_for_assembly['anchor']] # (M, 3, 3)
                else:
                    raise ValueError(f"use_predicted_normal must be True, but got {self.use_predicted_normal}")


                estimated_transform, used_corr = _RANSAC(in_dict=in_dict, 
                                                         shape_matching_scores=final_shape_matching_scores, 
                                                         src_pcd=final_src_pcd, 
                                                         trg_pcd=final_trg_pcd, 
                                                         src_predicted_frame=src_frame,
                                                         trg_predicted_frame=trg_frame,
                                                         match_option=self.infer_match_option, 
                                                         RANSAC_type=self.RANSAC_type, 
                                                         topk=self.infer_topk)
            else:
                estimated_transform, used_corr = self.fine_matching(final_src_pcd.unsqueeze(0), final_trg_pcd.unsqueeze(0), final_matching_scores_drop.unsqueeze(0), no_exp=(self.matching_norm_mode != 'sinkhorn'))
            
            estimated_rotat = estimated_transform[:3, :3]
            estimated_trans = estimated_transform[:3, 3]

            # Assemble
            assm_pred, list_of_assm_pred = pairwise_mating(final_src_pcd, final_trg_pcd, estimated_rotat, estimated_trans) # (N+M, 3)
            step_collector_for_viz.append(list_of_assm_pred)

            # Remove inner parts which is not needed for next iteration
            assm_pred_after_removing_inner_parts = remove_inner_parts(assm_pred, list_of_assm_pred, list_of_oris, setting_for_assembly['anchor'], selected_obj_idx, pos_radius=self.pos_radius, cos_threshold=self.cos_threshold)

            # Make setting for next iteration
            setting_for_assembly = make_setting_for_next_iteration(setting_for_assembly, assm_pred_after_removing_inner_parts, setting_for_assembly['list_of_input_pcds'], setting_for_assembly['anchor'], selected_obj_idx)

            # Save prediction
            pred_rot_and_trans.append([estimated_rotat, estimated_trans])
        
        
        assert len(setting_for_assembly['obj_ids'][1]) == 0, f"There are left objects to be assembled, len(setting_for_assembly['obj_ids'][1]): {len(setting_for_assembly['obj_ids'][1])}"


        # Refine predictions
        pred_rot_and_trans_dict = {}
        for i in range(len(pred_rot_and_trans)):
            obj_idx_to_move = setting_for_assembly['obj_ids'][0][1+i]
            pred_rot_and_trans_dict[f"{obj_idx_to_move}-{in_dict['anchor_idx'][0]}"] = (pred_rot_and_trans[i][0], pred_rot_and_trans[i][1])


        # Apply transformation to the point clouds
        list_of_assembled_pcds = apply_transformation_to_point_clouds(list_of_all_pcds, pred_rot_and_trans_dict, initial_anchor_idx, num_of_parts)

        # Apply GT transformation to the point clouds for evaluation
        list_of_gt_assembled_pcds = apply_transformation_to_point_clouds(list_of_all_pcds, in_dict['relative_trsfm'], initial_anchor_idx, num_of_parts)
        
        return pred_rot_and_trans_dict, list_of_assembled_pcds, list_of_gt_assembled_pcds, step_collector_for_viz
    
    
    def return_matching_scores(self, pcd_input, batch_info, batch_scaled_batch_info):
        # Get equivariant features and orientation matrices
        # equivariant features: (B, C, 3, N+M) , orientation matrices: (B, N+M, 3, 3)
        equi_feats, oris = get_feats_and_oris(self.backbone, self.ori_backbone, self.equi_layer, self.proj, self.normal_pred_mode, self.flip_normal_mode, self.only_train_normal, pcd_input, batch_info, batch_scaled_batch_info)

        # Invariant Features
        inv_feats = make_inv_feats(oris, batch_info, equi_feats, self.flip_normal_mode, flip_mode='src') # (B, C*3, N+M)

        # SHAPE DESCRIPTOR 
        shape_feats = self.shape_mlp(inv_feats) # (B, C*3, N+M) -> (B, D, N+M)

        # [Optional] Segmentation Head
        mating_surface_seg_results = self.feed_forward_seg_head(shape_feats, batch_scaled_batch_info) if self.seg_head_mode != 'none' else None # (B, D, N+M) -> (B, N+M)

        # Calculate Matching Scores
        active_mask = return_active_mask(batch_info)
        shape_matching_scores = calculate_matching_score(shape_feats, shape_feats, active_mask, eps=1e-8, mode=self.matching_score_mode)

        # Optimal Transport
        # Optimal Transport is in log space, so inside registration, there is exp operation
        matching_scores = self.multibatch_optimal_transport(shape_matching_scores, batch_info, active_mask, mode=self.matching_norm_mode) # (B, N+M+1, N+M+1) if mode is ['sinkhorn', 'softmax'], otherwise (B, N+M, N+M)
        matching_scores_drop = matching_scores[:,:-1,:-1] if self.matching_norm_mode in ['sinkhorn', 'softmax'] else matching_scores # (B, N+M, N+M)

        return oris, matching_scores_drop, shape_matching_scores, mating_surface_seg_results
    

    def assemble_shonan(self, in_dict):
        """
        Assume batch size must be 1

        Args:
            in_dict (dict): input dictionary for forward pass, which is same as forward_pass
        
        Returns:
            pred_rot_and_trans_dict (dict): dictionary of predicted rotations and translations
            list_of_assembled_pcds (list): list of assembled point clouds
            list_of_gt_assembled_pcds (list): list of GT assembled point clouds
            step_collector_for_viz (list): list of step-by-step assembled point clouds for visualization
        """

        pcd_input = in_dict['pcd_t'] # (B, N+M, 3)
        pcd_batch_info = in_dict['pcd_batch_info'] # (B, N+M, )
        offset = torch.concat([torch.zeros(1, device=pcd_batch_info.device, dtype=torch.int), batch2offset(pcd_batch_info[0])], dim=0) # size of parts
        num_of_parts = in_dict['num_parts'][0] # int
        anchor_idx = in_dict['anchor_idx'][0] # int


        # Extract all point clouds
        list_of_all_pcds = extract_all_objects_by_offset(pcd_input[0], offset) # list of (N, 3)
        list_of_all_gt_normals = extract_all_objects_by_offset(in_dict['gt_normals'][0], offset) # list of (N, 3)


        # Calculate matching scores for all pairs and predict rotation and translation
        pred_dict = {}
        pred_rot_and_trans_dict = {}
        for src_idx in range(num_of_parts):
            for trg_idx in range(num_of_parts):
                if src_idx == trg_idx: 
                    continue
                

                # Prepare input
                input_dict = make_input_dicts_for_shonan(src_idx, trg_idx, list_of_all_pcds, list_of_all_gt_normals)

                # Calculate matching scores
                # (1, N+M, 3, 3), (1, N+M, N+M), (1, N+M, N+M), (1, N+M)
                oris, matching_scores_drop, shape_matching_scores, mating_surface_seg_results = self.return_matching_scores(input_dict['input_pcds'], input_dict['pcd_batch_info'], input_dict['batch_scaled_batch_info'])

                # Calculate transformation
                num_of_src_pcd = input_dict['src_pcd'].shape[0]

                if self.use_RANSAC:
                    final_matching_scores = shape_matching_scores[0,0:num_of_src_pcd,num_of_src_pcd:]
                    src_frame = oris[0,0:num_of_src_pcd,:,:] if self.use_predicted_normal else None # (N, 3, 3)
                    trg_frame = oris[0,num_of_src_pcd:,:,:] if self.use_predicted_normal else None # (M, 3, 3)


                    estimated_transform, used_corr = _RANSAC(in_dict=input_dict, 
                                                             shape_matching_scores=final_matching_scores, 
                                                             src_pcd=input_dict['src_pcd'], 
                                                             trg_pcd=input_dict['trg_pcd'], 
                                                             src_predicted_frame=src_frame,
                                                             trg_predicted_frame=trg_frame,
                                                             match_option=self.infer_match_option, 
                                                             RANSAC_type=self.RANSAC_type, 
                                                             topk=self.infer_topk)
                else:
                    final_matching_scores = matching_scores_drop[0,0:num_of_src_pcd,num_of_src_pcd:]
                    estimated_transform, used_corr = self.fine_matching(input_dict['src_pcd'].unsqueeze(0), input_dict['trg_pcd'].unsqueeze(0), final_matching_scores.unsqueeze(0), no_exp=(self.matching_norm_mode != 'sinkhorn'))
                
                score_for_this_assembly = torch.topk(final_matching_scores.reshape(-1), k=self.infer_topk)[0].mean()
                pred_dict[f"{src_idx}-{trg_idx}"] = (score_for_this_assembly, estimated_transform) # allways move src to trg, src:ith, trg:jth

        # Prepare Graph Optimization
        factors, params, uncertainty_dict = make_shonan_factors(pred_dict, num_of_parts, selection_mode='max')
        
        # Select which edge should be added to the graph
        abs_rotat = run_shonan_averaging(factors, params, max_iter=60)
        list_of_relative_rotations = calculate_relative_rotation(abs_rotat, anchor_idx, num_of_parts)

        # Calculate translation after shonan averaging
        list_of_relative_translations = optimize_translation_after_shonan_averaging(list_of_relative_rotations, factors, anchor_idx, uncertainty_dict)

        # Make relative transformation dictionary
        pred_rot_and_trans_dict = make_relative_transformation_dict(list_of_relative_rotations, list_of_relative_translations, anchor_idx, num_of_parts, device=pcd_input.device)

        # Apply transformation to the point clouds
        list_of_assembled_pcds = apply_transformation_to_point_clouds(list_of_all_pcds, pred_rot_and_trans_dict, anchor_idx, num_of_parts)

        # Apply GT transformation to the point clouds for evaluation
        list_of_gt_assembled_pcds = apply_transformation_to_point_clouds(list_of_all_pcds, in_dict['relative_trsfm'], anchor_idx, num_of_parts)

        return pred_rot_and_trans_dict, list_of_assembled_pcds, list_of_gt_assembled_pcds, []





        
        
   
