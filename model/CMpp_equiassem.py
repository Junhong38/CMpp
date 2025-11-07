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
from model.loss import PointMatchingLoss, OrientationLoss
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from model.local_global_registration import LocalGlobalRegistration

from RANSAC.ransac import _RANSAC

from common.rotation import ortho2rotation
from common.utils import save_pc, check_inf_or_nan
from common.viz import draw_frames, draw_normal_error_histogram

from pytorch3d.ops import iterative_closest_point



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

        # Because of nn.AdaptiveMaxPool1d, deterministic option is not supported
        out2 = nn.AdaptiveMaxPool1d(1)(x) # 1, c, 1
        out2 = self.mlp(out2) # 1, c, 1
        
        out = F.normalize(out1+out2, p=2, dim=1)
        attention = self.sigmoid(out)
        
        return attention



class EquiAssem(pl.LightningModule):
    def __init__(
            self, 
            lr, 
            scheduler_mode='cos',
            training_total_steps=0,
            backbone='vn_unet', attention='channel', 
            pos_margin=0.1, neg_margin=1.4, log_scale=24, detach_mode=False, same_opt=False, only_corr=False, max_points=0, no_balance=False, div_mode='none',
            s_loss_weight=1.0, p_loss_weight=1.0, o_loss_weight=1.0,
            visualize=False, viz_epoch=30, viz_max_arrow_num=0, ckp_dir=None, debug=False,
            success_criterion_in_degree=10,
            delete_Sinkhorn=False,
            use_Sinkhorn_infer=False,
            matching_score_mode='CM',
            svd_no_exp=False,
            flip_normal=False,
            use_consistency_loss=0.0,
            only_train_normal=False,
            freeze_normal_param=False,

            # Developing temporarily used experiments arguments
            additional_VNLinearLeakyReLU=False,
            debugged_circle_loss=False,
            debugged_point_matching_loss=False,
            exp_scale_for_point_matching_loss=False,
            n_knn=20,
            new_orientation_module=False,
            delete_occupancy_loss=False,
            use_opt_gram=False,

            only_one_norm=False,
            n_avn=5,
            move_smaller=False,
            
            # RANSAC arguments
            infer_match_option='topk',
            infer_topk=128,
            infer_score_threshold_ratio=0.0,
            use_RANSAC=False,
            RANSAC_type='default',
            use_predicted_normal=False
            ):
        """Equivariant Assembly Model for 3D Object Assembly

        Args:
            lr (float): Learning rate for optimizer.
            scheduler_mode (str, optional): Scheduler type ('cos' or 'onecycle'). Defaults to 'cos'.
            training_total_steps (int, optional): Total number of training steps. Defaults to 0.
            backbone (str, optional): Backbone network architecture. Defaults to 'vn_unet'.
            attention (str, optional): Attention mechanism type ('channel' or 'none'). Defaults to 'channel'.
            
            # Circle loss arguments
            pos_margin (float, optional): Margin for positive samples in loss computation. Defaults to 0.1.
            neg_margin (float, optional): Margin for negative samples in loss computation. Defaults to 1.4.
            log_scale (int, optional): Log scaling factor for loss computation. Defaults to 24.
            detach_mode (bool, optional): Whether to use the detach mode for circle loss computation. Defaults to False.
            same_opt (bool, optional): Whether to use the same optimal value for positive and negative samples in loss computation. Defaults to False.
            only_corr (bool, optional): Whether to use only correspondence for circle loss computation. Defaults to False.
            max_points (int, optional): Maximum number of points for circle loss computation. Defaults to 0.
            no_balance (bool, optional): Whether to use positive and negative balance for circle loss computation. Defaults to False.
            div_mode (str, optional): Division mode for circle loss computation. Defaults to 'none'.
            
            s_loss_weight (float, optional): Weight for shape loss. Defaults to 1.0.
            p_loss_weight (float, optional): Weight for point loss. Defaults to 1.0.
            o_loss_weight (float, optional): Weight for orientation loss. Defaults to 1.0.
            visualize (bool, optional): Whether to save visualization results. Defaults to False.
            viz_epoch (int, optional): Epoch for mesh visualization. Defaults to 30.
            viz_max_arrow_num (int, optional): Maximum number of arrows for visualization. Defaults to 0.
            ckp_dir (str, optional): Checkpoint directory. Defaults to None.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            success_criterion_in_degree (int, optional): Success criterion in degree for normal error. Defaults to 10.
            delete_Sinkhorn (bool, optional): Whether to delete the optimal transport (Sinkhorn). Defaults to False.
            use_Sinkhorn_infer (bool, optional): Whether to use Sinkhorn for inference, hence just before registration. Defaults to False.
            matching_score_mode (str, optional): 'CM' or 'cos'. Defaults to 'CM'.
            svd_no_exp (bool, optional): Whether to do not use exp for SVD. Defaults to False.
            flip_normal (bool, optional): Whether to flip the normal vector. Defaults to False.
            use_consistency_loss (float, optional): Weight for consistency loss. Defaults to 0.0.
            only_train_normal (bool, optional): Whether to only train the normal vector, it will be used for stage 1 training. Defaults to False.
            freeze_normal_param (bool, optional): Whether to freeze the normal parameter. Defaults to False.

            # Developing temporarily used experiments arguments
            additional_VNLinearLeakyReLU (bool, optional): Whether to use additional VNLinearLeakyReLU layers for the equivariant shape feature. Defaults to False.
            debugged_circle_loss (bool, optional): Whether to use the debugged version of Circle Loss. Defaults to False.
            debugged_point_matching_loss (bool, optional): Whether to use the debugged version of Point Matching Loss. Defaults to False.
            exp_scale_for_point_matching_loss (bool, optional): Whether to make the matching score to exp-scaled value before computing point matching loss. Defaults to False.
            n_knn (int, optional): Number of nearest neighbors for KNN. Defaults to 20.
            new_orientation_module (bool, optional): Whether to use the new module for orientation loss. Defaults to False.
            delete_occupancy_loss (bool, optional): Whether to delete the occupancy loss. Defaults to False.
            use_opt_gram (bool, optional): Whether to use the optimum Gram Schmidt Orthogonalization. Defaults to False.


            only_one_norm (bool, optional): Whether to use only one Normalization layer for the equivariant shape feature. Defaults to False.
            n_avn (int, optional): Number of AVN layers for the equivariant shape feature. Defaults to 5.
            move_smaller (bool, optional): Whether to always move the smaller point cloud to the origin. Defaults to False.

            # RANSAC arguments
            infer_match_option (str, optional): 'topk' or 'mutual_topk' or 'soft_topk' or 'unidirectional_topk' or 'injective' or 'bijective'. Defaults to 'topk'.
            infer_topk (int, optional): Topk value for matching. Defaults to 128.
            infer_score_threshold_ratio (float, optional): Score threshold ratio for filtering correspondences. Defaults to 0.01.
            use_RANSAC (bool, optional): Whether to use RANSAC for transformation estimation. Defaults to False.
            RANSAC_type (str, optional): 'default' or 'score_dependent'. Defaults to 'default'.
            use_predicted_normal (bool, optional): Whether to use predicted normal for inlier counting. Defaults to False.
        """
        super(EquiAssem, self).__init__()

        print("------------------------------------------------------")
        print("INITIALIZING EquiAssem(pl.LightningModule)")
        print("------------------------------------------------------")
        print(f"lr: {lr}")
        print(f"scheduler_mode: {scheduler_mode}")
        print(f"training_total_steps: {training_total_steps}")
        print(f"backbone: {backbone}")
        print(f"attention: {attention}")
        
        print(f"pos_margin: {pos_margin}")
        print(f"neg_margin: {neg_margin}")
        print(f"log_scale: {log_scale}")
        print(f"detach_mode: {detach_mode}")
        print(f"same_opt: {same_opt}")
        print(f"only_corr: {only_corr}")
        print(f"max_points: {max_points}")
        print(f"no_balance: {no_balance}")
        print(f"div_mode: {div_mode}")

        print(f"s_loss_weight: {s_loss_weight}")
        print(f"p_loss_weight: {p_loss_weight}")
        print(f"o_loss_weight: {o_loss_weight}")
        
        print(f"visualize: {visualize}")
        print(f"viz_epoch: {viz_epoch}")
        print(f"viz_max_arrow_num: {viz_max_arrow_num}")
        print(f"ckp_dir: {ckp_dir}")
        print(f"debug: {debug}")
        print(f"success_criterion_in_degree: {success_criterion_in_degree}")
        print(f"delete_Sinkhorn: {delete_Sinkhorn}")
        print(f"use_Sinkhorn_infer: {use_Sinkhorn_infer}")
        print(f"matching_score_mode: {matching_score_mode}")
        print(f"svd_no_exp: {svd_no_exp}")
        print(f"flip_normal: {flip_normal}")
        print(f"use_consistency_loss: {use_consistency_loss}")
        print(f"only_train_normal: {only_train_normal}")
        print(f"freeze_normal_param: {freeze_normal_param}")
        
        print(f"additional_VNLinearLeakyReLU: {additional_VNLinearLeakyReLU}")
        print(f"debugged_circle_loss: {debugged_circle_loss}")
        print(f"debugged_point_matching_loss: {debugged_point_matching_loss}")
        print(f"exp_scale_for_point_matching_loss: {exp_scale_for_point_matching_loss}")
        print(f"n_knn: {n_knn}")
        print(f"new_orientation_module: {new_orientation_module}")
        print(f"delete_occupancy_loss: {delete_occupancy_loss}")
        print(f"use_opt_gram: {use_opt_gram}")

        print(f"only_one_norm: {only_one_norm}")
        print(f"n_avn: {n_avn}")
        print(f"move_smaller: {move_smaller}")


        # RANSAC arguments
        print(f"infer_match_option: {infer_match_option}")
        print(f"infer_topk: {infer_topk}")
        print(f"infer_score_threshold_ratio: {infer_score_threshold_ratio}")
        print(f"use_RANSAC: {use_RANSAC}")
        print(f"RANSAC_type: {RANSAC_type}")
        print(f"use_predicted_normal: {use_predicted_normal}")
        print("------------------------------------------------------")

        self.lr = lr
        self.scheduler_mode = scheduler_mode
        self.training_total_steps = training_total_steps
        self.attention = attention
        self.visualize = visualize
        self.viz_epoch = viz_epoch
        self.viz_max_arrow_num = viz_max_arrow_num
        self.ckp_dir = ckp_dir
        self.debug = debug
        self.success_criterion_in_degree = success_criterion_in_degree
        self.delete_Sinkhorn = delete_Sinkhorn
        self.use_Sinkhorn_infer = use_Sinkhorn_infer
        self.matching_score_mode = matching_score_mode
        self.svd_no_exp = svd_no_exp
        self.flip_normal = flip_normal
        self.only_train_normal = only_train_normal
        self.freeze_normal_param = freeze_normal_param
        
        self.additional_VNLinearLeakyReLU = additional_VNLinearLeakyReLU
        self.debugged_circle_loss = debugged_circle_loss
        self.debugged_point_matching_loss = debugged_point_matching_loss
        self.exp_scale_for_point_matching_loss = exp_scale_for_point_matching_loss
        self.new_orientation_module = new_orientation_module
        self.delete_occupancy_loss = delete_occupancy_loss
        self.use_opt_gram = use_opt_gram

        self.move_smaller = move_smaller

        # Inference arguments
        self.infer_match_option = infer_match_option
        self.infer_topk = infer_topk
        self.infer_score_threshold_ratio = infer_score_threshold_ratio
        self.use_RANSAC = use_RANSAC
        self.RANSAC_type = RANSAC_type
        self.use_predicted_normal = use_predicted_normal
        
        # Output feature dimension of Feature Extractor
        self.feat_dim = 1024

        
        # Objectives
        if debugged_circle_loss:
            print("Using the debugged version of Circle Loss")
            from model.loss import CircleLoss
            self.shape_loss = CircleLoss(log_scale=log_scale, pos_optimal=pos_margin, neg_optimal=neg_margin, detach_mode=detach_mode, 
                                         same_opt=same_opt, only_corr=only_corr, max_points=max_points, no_balance=no_balance, div_mode=div_mode)

        else:
            from model.CM_loss import CircleLoss
            self.shape_loss = CircleLoss(log_scale=log_scale, pos_optimal=pos_margin, neg_optimal=neg_margin)

        if debugged_point_matching_loss:
            print("Using the debugged version of Point Matching Loss")
            from model.loss import PointMatchingLoss
        else:
            from model.CM_loss import PointMatchingLoss

        if new_orientation_module:
            print("Using the new module for orientation loss")
            from model.loss import OrientationLoss
            self.orientation_loss = OrientationLoss(use_consistency_loss=use_consistency_loss)
        else:
            from model.CM_loss import OrientationLoss
            self.orientation_loss = OrientationLoss()
        
        if delete_occupancy_loss:
            print("Deleting the occupancy loss")
        else:
            if debugged_circle_loss:
                self.occupancy_loss = CircleLoss(log_scale=log_scale, pos_optimal=pos_margin, neg_optimal=neg_margin, detach_mode=detach_mode, 
                                                 same_opt=same_opt, only_corr=only_corr, max_points=max_points, no_balance=no_balance, div_mode=div_mode)
            else:
                self.occupancy_loss = CircleLoss(log_scale=log_scale, pos_optimal=pos_margin, neg_optimal=neg_margin)


        self.matching_loss = PointMatchingLoss()
        

        # Weights for losses
        self.s_loss_weight = s_loss_weight
        self.p_loss_weight = p_loss_weight
        self.o_loss_weight = o_loss_weight
        self.occ_loss_weight = s_loss_weight if not delete_occupancy_loss else 0
        
        """
        For reproducibility, we use the following weights
        self.s_loss_weight = 0.5 
        self.p_loss_weight = 1.0
        self.o_loss_weight = 0.1
        self.occ_loss_weight = 0.5

        However, for CMpp_equiassem, we use the following weights
        self.s_loss_weight = 1.0
        self.p_loss_weight = 1.0
        self.o_loss_weight = 1.0
        """

        print("------------------------------------------------------")
        print("Weight for losses")
        print(f"s_loss_weight: {self.s_loss_weight}")
        print(f"p_loss_weight: {self.p_loss_weight}")
        print(f"o_loss_weight: {self.o_loss_weight}")
        print(f"occ_loss_weight: {self.occ_loss_weight}")
        print("------------------------------------------------------")


        # Logging
        self.validation_step_outputs = []
        self.test_step_outputs = []


        # Declare Modules

        # VN BACKBONE
        if backbone == 'vn_unet':
            self.backbone = EQCNN_equi_unet(feat_dim=self.feat_dim, pooling="mean", k=n_knn)
        elif backbone == 'vn_dgcnn':
            self.backbone = EQCNN_equi(feat_dim=self.feat_dim, pooling="mean", k=n_knn)
        else:
            raise NotImplementedError("DGCNN backbone not implemented")

 
        if self.additional_VNLinearLeakyReLU:
            print("Using additional VNLinearLeakyReLU layers for the equivariant shape feature")
            # Layer for predicting frame vectors
            self.proj = VNLinear(2 * (self.feat_dim//3), 2)

            # Layer for Equivariant feature
            if n_avn > 0:
                self.equi_layer = nn.Sequential(*([VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3, no_norm=False)] + [VNLinearLeakyReLU(self.feat_dim//3, self.feat_dim//3, no_norm=only_one_norm) for _ in range(n_avn-1)]))
            else:
                self.equi_layer = nn.Identity()
            
        else:
            # Layer for predicting frame vectors
            self.proj = VNLinear(self.feat_dim//3, 2)
            # Layer for Equivariant feature
            self.equi_layer = nn.Identity()


        # Channel Attention
        if attention == 'channel':
            self.c_attn = ChannelAttentionModule((self.feat_dim//3) * 3, self.feat_dim, ratio=4)
        

        # Module for invariant Shape Descriptor
        if not delete_occupancy_loss:
            # Shape Descriptor
            self.shape_mlp = nn.Sequential(nn.Conv1d(1023, 512, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(512),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(512),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(512),
                                           nn.LeakyReLU(negative_slope=0.2),
                                        )
            # Occupancy Descriptor
            self.occ_mlp = nn.Sequential(nn.Conv1d(1023, 512, kernel_size=1, bias=False),
                                         nn.InstanceNorm1d(512),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                         nn.InstanceNorm1d(512),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                         nn.InstanceNorm1d(512),
                                         nn.Tanh()
                                         )
        else:
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
        if not (self.delete_Sinkhorn and not self.use_Sinkhorn_infer):
            self.optimal_transport = LearnableLogOptimalTransport(num_iterations=100)

        if not self.use_RANSAC: # If not using RANSAC, use LGR for fine matching
            # LGR
            self.fine_matching = LocalGlobalRegistration(
                k=self.infer_topk,
                match_option=self.infer_match_option,
                acceptance_radius=0.1,
                num_refinement_steps=5,
                score_threshold_ratio=self.infer_score_threshold_ratio,
            )

        if self.freeze_normal_param:
            print("Freezing parameters of backbone and proj")
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            for param in self.proj.parameters():
                param.requires_grad = False

    
    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        # Lightning 2.x: Support this funcionality
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        max_epochs = self.trainer.max_epochs

        print(f"total_steps: {total_steps}")
        print(f"steps_per_epoch: {steps_per_epoch}")
        print(f"max_epochs: {max_epochs}")

        assert total_steps > 0, "Total steps must be greater than 0"

        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.)
        
        if self.scheduler_mode in ['cos', 'CMpp']:
            if self.scheduler_mode == 'CMpp':
                assert self.training_total_steps > 0, "Training total steps must be greater than 0"
                total_steps = self.training_total_steps
            # T_max should be the total number of training steps, not a fixed value
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-3)
            
        elif self.scheduler_mode == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.lr, epochs=max_epochs, steps_per_epoch=steps_per_epoch,
                                                      pct_start=0.05, anneal_strategy="cos", div_factor=10.0,
                                                      final_div_factor=1000.0)
        
        elif self.scheduler_mode == 'CM': # Just for debugging purpose
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16919, eta_min=1e-3) # 16919, 6671
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        else:
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


    def training_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='train')
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
        self.log_dict(avg_loss, logger=True, sync_dist=True, batch_size=1,)
        self.test_step_outputs.clear()
    

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass
        # If you want to check the gradient and NaN, uncomment the following line
        # self.check_grad_and_nan()

    
    def check_grad_and_nan(self):
        total_modules = [self.backbone, self.proj, self.equi_layer, self.shape_mlp]

        if self.attention == 'channel':
            total_modules.append(self.c_attn)
        
        if not self.delete_occupancy_loss:
            total_modules.append(self.occ_mlp)
        
        total_grad_abs_sum = 0.0
        total_grad_abs_max = 0.0
        total_grad_count = 0
        nan_param_dict = {}
        
        for module in total_modules:
            for name, param in module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    total_grad_abs_sum += torch.abs(param.grad).sum().item()
                    total_grad_count += param.grad.shape.numel()
                    current_grad_abs_max = torch.abs(param.grad).max().item()
                    if current_grad_abs_max > total_grad_abs_max:
                        total_grad_abs_max = current_grad_abs_max
                    
                    # Check whether gradient is inf or NaN
                    check_inf_or_nan(param.grad, f'{name}.grad')
                
                # Check whether parameter is NaN
                if torch.isnan(param).any():
                    nan_param_dict[name] = param

        if total_grad_count > 0:
            total_grad_abs_mean = total_grad_abs_sum / total_grad_count
            self.log('train-grad/abs_mean', total_grad_abs_mean, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)
            self.log('train-grad/abs_max', total_grad_abs_max, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)
        else:
            assert False, "total_grad_count is 0"

        if len(nan_param_dict) > 0:
            for key, value in nan_param_dict.items():
                print(f"NaN parameter found - key: {key}, value: {value}")
            assert False, "NaN parameters found"
    
    
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

                - mesh_faces (list): length is 2, only for two pieces
                    - mesh_faces[0]: (1, F, 3)
                    - mesh_faces[1]: (1, F, 3)
                
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
                - During training,
                    - o_loss: (1, )
                    - s_loss: (1, )
                    - p_loss: (1, )
                    - loss: (1, )
                    
                    if not delete_occupancy_loss:
                        - occ_loss: (1, )
                
                - During validation or test, the following keys are added
                    - estimated_rotat: (3, 3)
                    - estimated_trans: (3)
                    - src_ori: (1, N, 3, 3)
                    - trg_ori: (1, M, 3, 3)

            loss (dict)
                - During training,
                    - o_loss: (1, )
                    - s_loss: (1, )
                    - p_loss: (1, )
                    - loss: (1, )

                    if not delete_occupancy_loss:
                        - occ_loss: (1, )
                
                - During validation or test, the following keys are added
                    - cd: (1, )
                    - rrmse: (1, )
                    - trmse: (1, )
                    - crd: (1, )
                    - rpf_rmse: (1, )
                    - rpf_tmse: (1, )
        """

        out_dict, loss = {}, {}

        # 0. Get Point Clouds and Ground Truth Correspondence
        src_pcd_raw = in_dict['pcd'][0].squeeze(0) # (N, 3)
        trg_pcd_raw = in_dict['pcd'][1].squeeze(0) # (M, 3)
        src_pcd = in_dict['pcd_t'][0] # (1, N ,3)
        trg_pcd = in_dict['pcd_t'][1] # (1, M ,3)
        gt_corr = in_dict['gt_correspondence'].squeeze(0) # (1, P, 2) -> (P, 2)
        
        # check_inf_or_nan(src_pcd_raw, 'src_pcd_raw')
        # check_inf_or_nan(trg_pcd_raw, 'trg_pcd_raw')
        # check_inf_or_nan(src_pcd, 'src_pcd')
        # check_inf_or_nan(trg_pcd, 'trg_pcd')
        # check_inf_or_nan(gt_corr, 'gt_corr')


        # 1. SO(3)-Equivariant Feature Extractor
        src_equi_feats_backbone = self.backbone(src_pcd) # (1, C, 3, N)
        trg_equi_feats_backbone = self.backbone(trg_pcd) # (1, C, 3, M)


        # check_inf_or_nan(src_equi_feats_backbone, 'src_equi_feats_backbone', log=(self.log if mode=='train' else None))
        # check_inf_or_nan(trg_equi_feats_backbone, 'trg_equi_feats_backbone', log=(self.log if mode=='train' else None))

        
        if self.additional_VNLinearLeakyReLU: # 2. Frame Prediction
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

        else: # 2. Basis Vector Projection 
            src_vecs = self.proj(src_equi_feats_backbone).permute(0, 3, 1, 2) # (1, N, 2, 3)
            trg_vecs = self.proj(trg_equi_feats_backbone).permute(0, 3, 1, 2) # (1, M, 2, 3)
        

        # check_inf_or_nan(src_vecs, 'src_vecs')
        # check_inf_or_nan(trg_vecs, 'trg_vecs')

        
        # 3. Calculate equivariant shape features
        src_equi_feats = self.equi_layer(src_equi_feats_backbone.unsqueeze(-1)).squeeze(-1) # (1, C, 3, N)
        trg_equi_feats = self.equi_layer(trg_equi_feats_backbone.unsqueeze(-1)).squeeze(-1) # (1, C, 3, M)


        # check_inf_or_nan(src_equi_feats, 'src_equi_feats', log=(self.log if mode=='train' else None))
        # check_inf_or_nan(trg_equi_feats, 'trg_equi_feats', log=(self.log if mode=='train' else None))


        # 4. Gram Schmidt & Cross-product, this is for making three basis vectors by using two predicted vectors
        src_ori = ortho2rotation(src_vecs, optimum=self.use_opt_gram) # (1, N, 2, 3) -> (1, N, 3, 3)
        trg_ori = ortho2rotation(trg_vecs, optimum=self.use_opt_gram) # (1, M, 2, 3) -> (1, M, 3, 3)


        # check_inf_or_nan(src_ori, 'src_ori')
        # check_inf_or_nan(trg_ori, 'trg_ori')

        # Save for visualization
        out_dict['src_ori'] = src_ori
        out_dict['trg_ori'] = trg_ori


        if self.only_train_normal:
            # Only train the normal vector
            loss['o_loss'] = self.orientation_loss(src_ori, trg_ori, gt_corr, in_dict['gt_normals'])
            loss['loss'] = loss['o_loss']

            # Compute Normal Error
            with torch.no_grad():
                # (d) Compute Normal Error
                loss['n_error'], _, loss['n_suc_rate'] = self._normal_error(in_dict, out_dict, success_criterion_in_degree=self.success_criterion_in_degree)
            
            if mode == 'train':
                self.log_for_training(loss=loss, pos_neg_distribution=None, mode=mode)
            return out_dict, loss


        # 5. Invariant Features
        if self.flip_normal: # Flip normal of src frame
            # (1, N, 3) stack -> (1, N, 3, 3)
            flipped_src_ori = torch.stack([- src_ori[:, :, 0, :], src_ori[:, :, 1, :], - src_ori[:, :, 2, :]], dim=-2)
            src_inv_feats = torch.matmul(src_equi_feats.permute(0, 3, 1, 2).float(), flipped_src_ori.transpose(-2,-1).float()) # (1, N, C, 3) x (1, N, 3, 3) -> (1, N, C, 3)

        else:
            src_inv_feats = torch.matmul(src_equi_feats.permute(0, 3, 1, 2).float(), src_ori.transpose(-2,-1).float()) # (1, N, C, 3) x (1, N, 3, 3) -> (1, N, C, 3)
        
        trg_inv_feats = torch.matmul(trg_equi_feats.permute(0, 3, 1, 2).float(), trg_ori.transpose(-2,-1).float()) # (1, M, C, 3) x (1, M, 3, 3) -> (1, M, C, 3)

        src_inv_feats = rearrange(src_inv_feats, 'b n c r -> b (c r) n') # (1, N, C, 3) -> (1, C*3, N)
        trg_inv_feats = rearrange(trg_inv_feats, 'b n c r -> b (c r) n') # (1, M, C, 3) -> (1, C*3, M)

        # check_inf_or_nan(src_inv_feats, 'src_inv_feats', log=(self.log if mode=='train' else None))
        # check_inf_or_nan(trg_inv_feats, 'trg_inv_feats', log=(self.log if mode=='train' else None))


        # OPTIONAL 5. Chaneel Attention Map
        if self.attention == 'channel':
            inv_feats = torch.cat([src_inv_feats, trg_inv_feats], dim=-1)  # (1, C*3, N+M)
            attention = self.c_attn(inv_feats) # (1, C*3, N+M) -> (1, D, N+M)
            shape_attention, occ_attention = attention[:, :512], attention[:, 512:] # [TODO] We should check this part, This can incurr problem
        

        # 6. SHAPE DESCRIPTOR 
        src_shape_feats = self.shape_mlp(src_inv_feats) # (1, C*3, N) -> (1, D, N)
        if self.attention == 'channel': # (1, D, N) * channel attention
            src_shape_feats = src_shape_feats * shape_attention
        
        trg_shape_feats = self.shape_mlp(trg_inv_feats) # # (1, C*3, M) -> (1, D, N)
        if self.attention == 'channel': # (1, D, M) * channel attention
            trg_shape_feats = trg_shape_feats * shape_attention
        

        # check_inf_or_nan(src_shape_feats, 'src_shape_feats', log=(self.log if mode=='train' else None))
        # check_inf_or_nan(trg_shape_feats, 'trg_shape_feats', log=(self.log if mode=='train' else None))


        if not self.delete_occupancy_loss:
            # 6-2. OCCUPANCY DESCRIPTOR
            src_occ_feats = self.occ_mlp(src_inv_feats) # (1, 1023, N) -> (1, 512, N)
            if self.attention == 'channel': 
                src_occ_feats = src_occ_feats * occ_attention
            
            trg_occ_feats = self.occ_mlp(trg_inv_feats) # (1, 1023, M) -> (1, 512, M)
            if self.attention == 'channel': 
                trg_occ_feats = trg_occ_feats * occ_attention
        

        # 7. Optimal Transport
        if not self.delete_occupancy_loss: # Only negative occupancy loss is used
            shape_matching_scores = self.calculate_matching_score(src_shape_feats, trg_shape_feats, eps=0.0)
            occ_matching_scores = - self.calculate_matching_score(src_occ_feats, trg_occ_feats, eps=0.0)
            shape_matching_scores = shape_matching_scores + occ_matching_scores # Combine shape and occupancy scores
            
        else:
            shape_matching_scores = self.calculate_matching_score(src_shape_feats, trg_shape_feats, eps=1e-8)
        
        

        if self.delete_Sinkhorn:
            row_slack = -shape_matching_scores.mean(dim=1) # (1, N, M) -> (1, M)
            col_slack = -shape_matching_scores.mean(dim=2) # (1, N, M) -> (1, N)
            corner = torch.tensor([[0.0]], device=shape_matching_scores.device, dtype=shape_matching_scores.dtype) # (1, 1)
            matching_scores = torch.cat([
                torch.cat([shape_matching_scores, col_slack.unsqueeze(2)], dim=2), # (1, N, M) concat (1, N, 1) -> (1, N, M+1)
                torch.cat([row_slack, corner], dim=1).unsqueeze(1) # (1, M) concat (1,1) -> (1, M+1) ->  (1, 1, M+1)
            ], dim=1) # (1, N+1, M+1) each slack is fill with minus mean value of each row/column.
            matching_scores_drop = shape_matching_scores
        else:
            matching_scores = self.optimal_transport(shape_matching_scores) # Optimal Transport is in log space, so inside registration, there is exp operation
            matching_scores_drop = matching_scores[:,:-1,:-1]   

        
        # check_inf_or_nan(matching_scores, 'matching_scores')


        if mode in ['train', 'val']: # Do not calculate for test
            # 8. Calculate Loss
            # Orientation loss
            if self.new_orientation_module:
                loss['o_loss'] = self.orientation_loss(src_ori, trg_ori, gt_corr, in_dict['gt_normals'])
            else:
                loss['o_loss'] = self.orientation_loss(src_ori, trg_ori, gt_corr, in_dict['gt_rotat'])
            

            # Shape loss
            if self.debugged_circle_loss:
                loss['s_loss'], pos_neg_distribution = self.shape_loss(src_pcd_raw, trg_pcd_raw, src_shape_feats.transpose(-2,-1), trg_shape_feats.transpose(-2,-1), gt_corr)
            else:
                loss['s_loss'], pos_neg_distribution = self.shape_loss(src_pcd_raw, trg_pcd_raw, src_shape_feats.transpose(-2,-1), trg_shape_feats.transpose(-2,-1), gt_corr)

            
            # Point matching loss
            if self.exp_scale_for_point_matching_loss:
                loss['p_loss'] = 1.0 + self.matching_loss(torch.exp(matching_scores), gt_corr, src_pcd_raw, trg_pcd_raw).float() # Optimal Transport is in log space, so before registration, we need to exp it
            else:
                loss['p_loss'] = self.matching_loss(matching_scores, gt_corr, src_pcd_raw, trg_pcd_raw).float()


            if not self.delete_occupancy_loss:
                loss['occ_loss'], _ = self.occupancy_loss(src_pcd_raw, trg_pcd_raw, src_occ_feats.transpose(-2,-1), -trg_occ_feats.transpose(-2,-1), gt_corr)
            

            # Final loss
            if not self.delete_occupancy_loss:
                loss['loss'] = self.o_loss_weight * loss['o_loss'] + self.s_loss_weight * loss['s_loss'] + self.p_loss_weight * loss['p_loss'] + self.occ_loss_weight * loss['occ_loss']
            else:
                loss['loss'] = self.o_loss_weight * loss['o_loss'] + self.s_loss_weight * loss['s_loss'] + self.p_loss_weight * loss['p_loss']

            
            # Check for Inf or Nan
            # for loss_name, loss_value in loss.items():
            #     check_inf_or_nan(loss_value, f'{loss_name}')
            

            out_dict.update(loss)


        if mode == 'train':
            with torch.no_grad():
                # This is for checking the normal error
                loss['n_error'], _, loss['n_suc_rate'] = self._normal_error(in_dict, out_dict, success_criterion_in_degree=self.success_criterion_in_degree) 


        # 9. Evaluation
        if mode in ['val', 'test']:
            if self.use_Sinkhorn_infer:
                with torch.no_grad():
                    matching_scores_drop = self.optimal_transport(matching_scores_drop) # Optimal Transport is in log space, so inside registration, there is exp operation
                    matching_scores_drop = matching_scores_drop[:,:-1,:-1]


            # Point cloud registration
            src_predicted_frame = None
            trg_predicted_frame = None
            if self.use_predicted_normal:
                src_predicted_frame = src_ori.squeeze(0) # (1, N, 3, 3) -> (N, 3, 3)
                trg_predicted_frame = trg_ori.squeeze(0) # (1, M, 3, 3) -> (M, 3, 3)
            with torch.no_grad():
                if self.use_RANSAC:
                    estimated_transform = _RANSAC(in_dict=in_dict, 
                                                  shape_matching_scores=shape_matching_scores, 
                                                  src_pcd=src_pcd, 
                                                  trg_pcd=trg_pcd, 
                                                  src_predicted_frame=src_predicted_frame,
                                                  trg_predicted_frame=trg_predicted_frame,
                                                  match_option=self.infer_match_option, 
                                                  RANSAC_type=self.RANSAC_type, 
                                                  topk=self.infer_topk)
                else:
                    # fine_matching predict Rt to move points from src_points to ref_points
                    estimated_transform = self.fine_matching(src_pcd,trg_pcd, matching_scores_drop, no_exp=self.svd_no_exp)

            # estimated_transform: target_point = R * source_point + t
            out_dict['estimated_rotat'] = estimated_transform[:3, :3] # R
            out_dict['estimated_trans'] = estimated_transform[:3, 3] # t

            # Evaluation
            eval_dict = self.evaluate_prediction(in_dict, out_dict, gt_corr, mode)

            ## Matching Recall
            with torch.no_grad():
                eval_dict.update(self._calculate_recall(matching_scores_drop, gt_corr))

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
            self.log_for_training(loss=loss, pos_neg_distribution=pos_neg_distribution, mode=mode)
        else:
            torch.cuda.empty_cache()

        return out_dict, loss

    
    def log_for_training(self, loss, pos_neg_distribution, mode):
        log_dict = {f'{mode}/{k}': v.item() for k, v in loss.items()}
        
        if pos_neg_distribution is not None:
            log_pos_neg_distribution = {f'{mode}-dist/{k}': v for k, v in pos_neg_distribution.items()}
            log_dict.update(log_pos_neg_distribution)

        training_loss = log_dict.pop(f'{mode}/loss')
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log_dict(log_dict, prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=True, batch_size=1)
        self.log(f'{mode}/loss', training_loss, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('current_lr', current_lr, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)

    
    def calculate_matching_score(self, src_feats, trg_feats, eps=1e-8):
        """
        Calculate matching score between src and trg features
        When mode is CM, then calculate score like CM
        When mode is cos, then calculate score like cosine similarity

        Args:
            src_feats (torch.Tensor): (1, C, N)
            trg_feats (torch.Tensor): (1, C, M)

        Returns:
            matching_scores (torch.Tensor): (1, N, M)
        """
        if self.matching_score_mode == 'CM':
            matching_scores = torch.einsum('b c n , b c m -> b n m', src_feats, trg_feats) # (1, N, M)
            matching_scores = matching_scores / (src_feats.shape[1] ** 0.5 + eps) # 1e-8 is for avoiding division by zero

        else:
            # Calculate cosine similarity for all pairs (N, M)
            # Normalize features along channel dimension (dim=1)
            src_feats_norm = F.normalize(src_feats, p=2, dim=1)  # (1, C, N)
            trg_feats_norm = F.normalize(trg_feats, p=2, dim=1)  # (1, C, M)
            # Compute dot product for all pairs
            matching_scores = torch.einsum('b c n , b c m -> b n m', src_feats_norm, trg_feats_norm)  # (1, N, M)

        return matching_scores
    
    
    @torch.no_grad()
    def evaluate_prediction(self, in_dict, out_dict, gt_corr, mode, multi_part=False):
        """
        Args:
            in_dict (dict): it is same as forward_pass
            out_dict (dict): it is same as forward_pass
            gt_corr (torch.Tensor): (P, 2)
            mode (str): 'val' or 'test'
            multi_part (bool, optional): _description_. Defaults to False.

        Returns:
            eval_result (dict):
                - cd (float): CD between prediction & ground-truth
                - rrmse (float): MSE between prediction & ground-truth for rotation (in degree)
                - trmse (float): MSE between prediction & ground-truth for translation (in cm)
                - crd (float): CoRrespondence Distance (CRD) betwween prediction & ground-truth
        """
        assert mode in ['val', 'test'], f"mode must be in ['val', 'test'], but got {mode}"

        # Init return buffer
        eval_result = {}
        
        pred_relative_trsfm = out_dict['estimated_rotat'].float(), out_dict['estimated_trans'].float() # (3, 3), (3)
        grtr_relative_trsfm = [x.squeeze(0) for x in in_dict['relative_trsfm']['0-1']] # (1, 3, 3) -> (3, 3), (1, 3) -> (3)
        src_pcd, trg_pcd = [x.squeeze(0) for x in in_dict['pcd_t']] # (1, N, 3) -> (N, 3), (1, M, 3) -> (M, 3)


        # Move larger point cloud
        if self.move_smaller and not self._is_trg_larger(src_pcd, trg_pcd):
            # if source point cloud is bigger than target point cloud, we want to move trg to src
            # However, our code is designed to move src to trg
            # So, we need to inverse the relative transformation
            # trg = R * src + t -> src = R^T * (trg - t) -> src = R^T * trg - R^T * t
            src_pcd, trg_pcd = trg_pcd, src_pcd
            pred_relative_trsfm = pred_relative_trsfm[0].T, -  pred_relative_trsfm[0].T @ pred_relative_trsfm[1]
            grtr_relative_trsfm = grtr_relative_trsfm[0].T, -  grtr_relative_trsfm[0].T @ grtr_relative_trsfm[1]
            gt_corr = torch.stack([gt_corr[:,1], gt_corr[:,0]], dim=1) # (P,) stack (P,) -> (P,2)
            is_swap_triggered = True
       
        else:
            is_swap_triggered = False


        # Assemble using prediction, pseudo-gt, and ground-truth
        assm_pred, pcds_pred = self._pairwise_mating(src_pcd, trg_pcd, pred_relative_trsfm[0], pred_relative_trsfm[1])
        assm_grtr, pcds_grtr = self._pairwise_mating(src_pcd, trg_pcd, grtr_relative_trsfm[0], grtr_relative_trsfm[1])

        assm_pred, assm_grtr = assm_pred.float(), assm_grtr.float()
        
        # (a) Compute CD between prediction & ground-truth
        eval_result['cd'] = self._chamfer_distance(assm_pred, assm_grtr)

        # (b) Compute MSE between prediction & ground-truth for rotation (in degree) and translation
        eval_result['rrmse_rpf'], eval_result['trmse_rpf'] = self._transformation_error_RPFver(pcds_pred, pcds_grtr, multi_part)
        eval_result['rrmse'], eval_result['trmse'] = self._transformation_error(pred_relative_trsfm, grtr_relative_trsfm, multi_part)

        # (c) Compute CoRrespondence Distance (CRD) betwween prediction & ground-truth
        eval_result['crd'] = self._correspondence_distance(assm_pred, assm_grtr)

        # (d) Compute Normal Error
        eval_result['n_error'], normal_error_hist, eval_result['n_suc_rate'] = self._normal_error(in_dict, out_dict, success_criterion_in_degree=self.success_criterion_in_degree)


        if (mode=='val' and (not self.trainer.sanity_checking) and \
            self.trainer.global_rank == 0 and \
            self.visualize and \
            (self.current_epoch % self.viz_epoch == 0 or self.current_epoch == self.trainer.max_epochs-1) and \
            in_dict['eval_idx'].item() == 0) or \
            (mode=='test' and self.visualize and in_dict['eval_idx'].item() == 0):
            # Do not visualize in sanity checking
            # Only rank 0 should do visualization to avoid file I/O conflicts in DDP
            # Visualize for every self.viz_epoch
            # However, if it is the last epoch, then visualize
            # Also, only visualize first batch

            vis_folder = os.path.join(self.ckp_dir, 'vis', mode) # For mesh visualization
            vis_hist_folder = os.path.join(self.ckp_dir, 'vis_hist', mode) # For normal error histogram visualization
            os.makedirs(vis_folder, exist_ok=True)
            os.makedirs(vis_hist_folder, exist_ok=True)

            # PCD light visualization
            pcds_pred_for_viz = [] + pcds_pred
            pcds_grtr_for_viz = [] + pcds_grtr
            pcds_pred_for_viz.append(pcds_pred[0][gt_corr[:,0]])
            pcds_pred_for_viz.append(pcds_pred[1][gt_corr[:,1]])
            pcds_grtr_for_viz.append(pcds_grtr[0][gt_corr[:,0]])
            pcds_grtr_for_viz.append(pcds_grtr[1][gt_corr[:,1]])
            save_pc(f'{vis_folder}/E{self.current_epoch}_{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["crd"].item(),3)}_pred.ply', pcds_pred_for_viz)
            save_pc(f"{vis_folder}/E{self.current_epoch}_{in_dict['eval_idx'].item()}_{in_dict['obj_class'][0]}_{round(eval_result['crd'].item(),3)}_grtr.ply", pcds_grtr_for_viz)

            # MESH AND FRAME VISUALIZATION
            output_src_ori, output_trg_ori = out_dict['src_ori'][0], out_dict['trg_ori'][0] # (1,N,3,3) -> (N,3,3), (1,M,3,3) -> (M,3,3)
            gt_src_normals, gt_trg_normals = in_dict['gt_normals'][0][0].float(), in_dict['gt_normals'][1][0].float() # (1,N,3) -> (N,3), (1,M,3) -> (M,3)
            src_mesh_verts, trg_mesh_verts = in_dict['mesh_t'][0][0].float(), in_dict['mesh_t'][1][0].float() # (1,N,3) -> (N,3), (1,M,3) -> (M,3)
            src_mesh_faces, trg_mesh_faces = in_dict['mesh_faces'][0][0].float(), in_dict['mesh_faces'][1][0].float() # (1,F,3) -> (F,3), (1,F,3) -> (F,3)
            
            if is_swap_triggered: # To move smaller one, we swap src and trg in the above part
                output_src_ori, output_trg_ori = output_trg_ori, output_src_ori
                gt_src_normals, gt_trg_normals = gt_trg_normals, gt_src_normals
                src_mesh_verts, trg_mesh_verts = trg_mesh_verts, src_mesh_verts
                src_mesh_faces, trg_mesh_faces = trg_mesh_faces, src_mesh_faces
            
            mesh_faces_for_viz = [src_mesh_faces, trg_mesh_faces]

            reshaped_output_src_ori = output_src_ori.reshape(-1,3) # (N,3,3) -> (N*3,3)
            reshaped_output_trg_ori = output_trg_ori.reshape(-1,3) # (M,3,3) -> (M*3,3)

            zero_trans = torch.zeros(3).to(grtr_relative_trsfm[0].device)

            # Rotate by using gt
            _, rot_frame_ori_in_gt = self._pairwise_mating(reshaped_output_src_ori, reshaped_output_trg_ori, grtr_relative_trsfm[0], zero_trans)
            _, rot_gt_normals_in_gt = self._pairwise_mating(gt_src_normals, gt_trg_normals, grtr_relative_trsfm[0], zero_trans)
            _, rot_mesh_verts_in_gt = self._pairwise_mating(src_mesh_verts, trg_mesh_verts, grtr_relative_trsfm[0], grtr_relative_trsfm[1])


            # DRAW FRAME by using gt
            draw_frames(mesh_verts=rot_mesh_verts_in_gt, mesh_faces=mesh_faces_for_viz, 
                        frame_ori=rot_frame_ori_in_gt, gt_normals=rot_gt_normals_in_gt, pcds_list=pcds_grtr, dir_path=vis_folder,
                        filename=f'E{self.current_epoch}_{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["crd"].item(),3)}_in_gt',
                        viz_max_arrow_num=self.viz_max_arrow_num,
                        viz_piece=True, viz_full=True)

            # Rotate by using pred
            _, rot_frame_ori_in_pred = self._pairwise_mating(reshaped_output_src_ori, reshaped_output_trg_ori, pred_relative_trsfm[0], zero_trans)
            _, rot_gt_normals_in_pred = self._pairwise_mating(gt_src_normals, gt_trg_normals, pred_relative_trsfm[0], zero_trans)
            _, rot_mesh_verts_in_pred = self._pairwise_mating(src_mesh_verts, trg_mesh_verts, pred_relative_trsfm[0], pred_relative_trsfm[1])

            # DRAW FRAME by using prediction
            draw_frames(mesh_verts=rot_mesh_verts_in_pred, mesh_faces=mesh_faces_for_viz, 
                        frame_ori=rot_frame_ori_in_pred, gt_normals=rot_gt_normals_in_pred, pcds_list=pcds_pred, dir_path=vis_folder,
                        filename=f'E{self.current_epoch}_{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["crd"].item(),3)}_in_pred',
                        viz_max_arrow_num=self.viz_max_arrow_num,
                        viz_piece=False, viz_full=True)
            

            # DRAW NORMAL ERROR HISTOGRAM
            draw_normal_error_histogram(normal_error_hist=normal_error_hist, dir_path=vis_hist_folder, 
                                        filename=f'E{self.current_epoch}_{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["n_error"].item(),3)}_hist.png')
            

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
    
    
    def _pairwise_mating(self, src_pcd, trg_pcd, rotat, trans):
        """
        move src to trg

        Args:
            src_pcd (torch.Tensor): (N, 3)
            trg_pcd (torch.Tensor): (M, 3)
            rotat (torch.Tensor): (3, 3)
            trans (torch.Tensor): (3)

        Returns:
            pcd_t (torch.Tensor): (N+M, 3)
            pcd_t (list): [(N, 3), (M, 3)] if is_trg_larger else [(N, 3), (M, 3)]
        """
        # Remind:
        # estimated_transform: trg_pcd = R * src_pcd + t

        # When GT
        # GT Rt format already fits to R * src + t

        # When pred
        # target_point = R * source_point + t
        # Hence, pred format already fits to R * src + t format

        pcd_t = []
        # Fix target point, and move source point to target point
        # src_pcd_t = R * src_pcd + t
        src_pcd_t = self._transform(src_pcd, rotat, trans)
        pcd_t = [src_pcd_t, trg_pcd]
        
        return torch.cat(pcd_t, dim=0), pcd_t
    

    def _transform(self, pcd, rotat=None, trans=None):
        """
        rotat * pcd + trans

        Args:
            pcd (torch.Tensor): (N, 3)
            rotat (torch.Tensor, optional): (3, 3). Defaults to None.
            trans (torch.Tensor, optional): (3). Defaults to None.

        Returns:
            pcd_t (torch.Tensor): (N, 3) 
        """
        if rotat == None: rotat = torch.eye(3, 3)
        if trans == None: trans = torch.zeros(3)

        rotat = rotat.to(pcd.device)
        trans = trans.to(pcd.device)

        return torch.einsum('x y, n y -> n x', rotat, pcd) + trans


    def _correspondence_distance(self, assm1, assm2, scaling=100):
        """
        Args:
            assm1 (torch.Tensor): (N, 3)
            assm2 (torch.Tensor): (M, 3)
            scaling (int, optional): Scaling factor for CD. Defaults to 100.

        Returns:
            corr_dist (torch.Tensor): (1)
        """
        corr_dist = (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling
        return corr_dist


    def _chamfer_distance(self, assm1, assm2, scaling=1000):
        """
        Args:
            assm1 (torch.Tensor): (N, 3)
            assm2 (torch.Tensor): (M, 3)
            scaling (int, optional): Scaling factor for CD. Defaults to 1000.

        Returns:
            cd (torch.Tensor): (1)
        """
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
        cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling
        return cd
    

    def _transformation_error(self, trnsf1, trnsf2, multi_part, trmse_scaling=100):
        """
        Args:
            trnsf1 (tuple): (3, 3), (3)
            trnsf2 (tuple): (3, 3), (3)
            multi_part (bool): True if multi-part
            trmse_scaling (int, optional): Scaling factor for TRMSE. Defaults to 100.

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
            trmse += (t1 - t2).pow(2).mean().pow(0.5) * trmse_scaling
        div = len(rotat1) if multi_part else 1
        return (rrmse / div).to(trmse.device), trmse / div


    def _transformation_error_RPFver(self, pcds_pred, pcds_grtr, multi_part, scaling=100):
        """
        Args:
            pcds_pred (list): [(N, 3), (M, 3)]
            pcds_grtr (list): [(N, 3), (M, 3)]
            multi_part (bool): True if multi-part
            scaling (int, optional): Scaling factor for TRMSE. Defaults to 100. 

        Returns:
            rrmse (torch.Tensor): (1)
            trmse (torch.Tensor): (1)
        """
        
        num_parts = len(pcds_grtr)
        rot_errors = torch.zeros(num_parts, device=pcds_grtr[0].device) # (K), rotation error
        trans_errors = torch.zeros(num_parts, device=pcds_grtr[0].device)  # (K), translation error

        for p in range(num_parts):
            pcd_pred = pcds_pred[p].unsqueeze(0) # (N, 3) -> (1, N, 3)
            pcd_grtr = pcds_grtr[p].unsqueeze(0) # (N, 3) -> (1, N, 3)
            assert pcd_pred.shape == pcd_grtr.shape, f"Point clouds should be same size, but got {pcd_pred.shape} and {pcd_grtr.shape}"

            # ICP algorithm
            error = iterative_closest_point(pcd_grtr, pcd_pred).RTs
            
            # tr(R) = 1 + 2cos(θ) -> θ = acos((tr(R) - 1) / 2), torch.acos is in radian, so we need to convert to degree
            rot_errors[p] = torch.rad2deg(torch.acos(torch.clamp(0.5 * (torch.trace(error.R[0]) - 1.0), -1.0, 1.0)))
            trans_errors[p] = torch.norm(error.T[0]) * scaling

        div = len(pcds_grtr)
        return rot_errors.sum() / div, trans_errors.sum() / div 
    

    def _normal_error(self, in_dict, out_dict, success_criterion_in_degree=10):
        """
        Args:
            in_dict (dict): it is same as forward_pass
            out_dict (dict): it is same as forward_pass
            success_criterion_in_degree (int, optional): Success criterion in degree. Defaults to 10.

        Returns:
            normal_error (torch.Tensor): (1)
        """
        output_src_ori, output_trg_ori = out_dict['src_ori'][0], out_dict['trg_ori'][0] # (1,N,3,3) -> (N,3,3), (1,M,3,3) -> (M,3,3)
        gt_src_normals, gt_trg_normals = in_dict['gt_normals'][0][0].float(), in_dict['gt_normals'][1][0].float() # (1,N,3) -> (N,3), (1,M,3) -> (M,3)

        pred_normals = torch.cat([output_src_ori[:,0,:], output_trg_ori[:,0,:]], dim=0) # (N,3) concat (M,3) -> (N+M, 3)
        gt_normals = torch.cat([gt_src_normals, gt_trg_normals], dim=0) # (N,3) concat (M,3) -> (N+M, 3)

        cosine_similarity = torch.clamp(torch.nn.functional.cosine_similarity(pred_normals, gt_normals, dim=-1), min=-1, max=1) # (N+M, )
        theta_deg = torch.rad2deg(torch.acos(cosine_similarity)) # (N+M, )
        
        normal_error = theta_deg.mean()
        normal_error_hist = torch.histogram(theta_deg.cpu(), bins=90, range=(0, 180)) # Total 180 degrees, so we choose 90 bins

        success_mask = theta_deg <= success_criterion_in_degree
        success_count = success_mask.sum()
        total_count = theta_deg.shape[0]
        success_rate = success_count / total_count

        return normal_error, normal_error_hist, success_rate
    

    def _calculate_recall(self, matching_scores_drop, gt_corr, topks=[1,5,10,20]):
        """
        Calculate recall of matching scores

        Args:
            matching_scores_drop (torch.Tensor): (1, N, M)
            gt_corr (torch.Tensor): (P, 2)
            topks (list, optional): Recall@1, Recall@5, Recall@10, Recall@20.

        Returns:
            matching_recall (torch.Tensor): (1)
        """
        _, _N, _M = matching_scores_drop.shape # (1, N, M) -> M

        result_dict = dict()

        correspondence_mask = torch.zeros((_N, _M), device=matching_scores_drop.device)
        correspondence_mask[gt_corr[:,0], gt_corr[:,1]] = True
        correspondence_mask_src = correspondence_mask.sum(dim=-1) > 0 # (N, M) -> N
        correspondence_mask_trg = correspondence_mask.sum(dim=-2) > 0 # (N, M) -> M
        
        for topk in topks:
            ## Recall from src
            _, topk_inds_src = torch.topk(matching_scores_drop[:, correspondence_mask_src], k=topk, dim=-1) # (1, N, M) -> (1, gt_N, M) -> (1, gt_N, topk)
            topk_mask_src = torch.zeros((_N, _M), device=matching_scores_drop.device) # (N, M)
            for i, _ in enumerate(range(topk_inds_src.shape[-1])): # for i in range(topk)
                # [all gt_N, ith topk from gt_src]
                topk_mask_src[torch.nonzero(correspondence_mask_src)[:, 0], topk_inds_src[0, :, i]] = True

            # (N, M) -> N
            is_success_src = (topk_mask_src * correspondence_mask).sum(dim=-1) > 0
            recall_src = is_success_src[correspondence_mask_src].sum() / correspondence_mask_src.sum()

            ## Recall from trg
            _, topk_inds_trg = torch.topk(matching_scores_drop[:, :, correspondence_mask_trg], k=topk, dim=-2) # (1, N, M) -> (1, N, gt_M) -> (1, topk, gt_M)
            topk_mask_trg = torch.zeros((_N, _M), device=matching_scores_drop.device) # (N, M)
            for i, _ in enumerate(range(topk_inds_trg.shape[-2])): # for i in range(topk)
                # [ith topk from gt_trg, all gt_N]
                topk_mask_trg[topk_inds_trg[0, i, :], torch.nonzero(correspondence_mask_trg)[:, 0]] = True

            # (N, M) -> M
            is_success_trg = (topk_mask_trg * correspondence_mask).sum(dim=-2) > 0
            recall_trg = is_success_trg[correspondence_mask_trg].sum() / correspondence_mask_trg.sum()
            
            recall_dot_k = (recall_src + recall_trg) / 2

            ## Logging results
            result_dict[f"recall@{str(topk)}"] = recall_dot_k
        
        return result_dict




