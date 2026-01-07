import os
from scipy.spatial.transform import Rotation

import pytorch_lightning as pl

from chamfer_distance import ChamferDistance as chamfer_dist

import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

from model.backbone.vn_dgcnn import EQCNN_equi_unet, EQCNN_equi_unet_v2
from model.backbone.vn_layers import VNLinear, VNLinearLeakyReLU
from model.loss import CircleLoss, PointMatchingLoss, OrientationLoss
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from model.local_global_registration import LocalGlobalRegistration

from RANSAC.ransac import _RANSAC

from common.rotation import gram_schmidt_with_cross, gram_schmidt, rodrigues_to_rotmat, rotate_by_rotation_matrix, src_reverse_trg_normal_gram_schmidt_with_cross
from common.utils import instance_wise_results_to_json
from common.viz import visualize_negative_hard_mask, save_pcd_for_light_visualization, draw_frames, draw_normal_error_histogram, draw_test_results_histogram
from common.misc import extract_all_objects, batch_scaling

from pytorch3d.ops import iterative_closest_point


class EquiAssem(pl.LightningModule):
    def __init__(
            self, 
            lr, 
            scheduler_mode='cos',
            backbone='vn_unet', 
            double_bacbone='none',
            
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
            only_nearest_consistency=False,
            
            n_knn=20,
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
            use_predicted_normal=False
            ):
        """Equivariant Assembly Model for 3D Object Assembly

        Args:
            lr (float): Learning rate for optimizer.
            scheduler_mode (str, optional): Scheduler type ('cos', 'onecycle', 'none). Defaults to 'cos'.
            backbone (str, optional): Backbone network architecture. Defaults to 'vn_unet'.
            double_bacbone (str, optional): 'none' or 'vn_unet'. Defaults to 'none'.

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
            flip_normal_mode (str, optional): 'none' or 'right' or 'rightv1_2' or 'rightv2' or 'rightv3' or 'rightv4' or 'mix'. Defaults to 'none'.
            consistency_loss_weight (float, optional): Weight for consistency loss. Defaults to 0.0.
            one_to_one_consistency (bool, optional): Whether to use one-to-one consistency loss. Defaults to False.

            n_knn (int, optional): Number of nearest neighbors for KNN. Defaults to 20.
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
        """
        super(EquiAssem, self).__init__()

        print("------------------------------------------------------")
        print("INITIALIZING EquiAssem(pl.LightningModule)")
        print("------------------------------------------------------")
        print(f"lr: {lr}")
        print(f"scheduler_mode: {scheduler_mode}")
        print(f"backbone: {backbone}")
        print(f"double_bacbone: {double_bacbone}")
        
        # Circle loss parameters will be printed in CircleLoss initialization
        # Point matching loss parameters will be printed in PointMatchingLoss initialization

        print(f"s_loss_weight: {s_loss_weight}")
        print(f"p_loss_weight: {p_loss_weight}")
        print(f"o_loss_weight: {o_loss_weight}")
        
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
        print(f"only_nearest_consistency: {only_nearest_consistency}")

        print(f"n_knn: {n_knn}")
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
        print("------------------------------------------------------")

        self.lr = lr
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
        self.orientation_loss = OrientationLoss(consistency_loss_weight=consistency_loss_weight, pos_radius=pos_radius, 
                                                flip_normal_mode=flip_normal_mode, only_nearest_consistency=only_nearest_consistency)
        self.matching_loss = PointMatchingLoss(pos_radius=pos_radius, safe_radius=safe_radius)
        

        # Weights for losses
        self.s_loss_weight = s_loss_weight # circle loss weight
        self.p_loss_weight = p_loss_weight # point matching loss weight
        self.o_loss_weight = o_loss_weight # orientation loss weight

        print("------------------------------------------------------")
        print("Weight for losses")
        print(f"s_loss_weight: {self.s_loss_weight}")
        print(f"p_loss_weight: {self.p_loss_weight}")
        print(f"o_loss_weight: {self.o_loss_weight}")
        print("------------------------------------------------------")


        # Logging
        self.validation_step_outputs = []
        self.test_step_outputs = []


        # Declare Modules
        # VN BACKBONE
        if backbone == 'vn_unet':
            self.backbone = EQCNN_equi_unet(feat_dim=self.feat_dim, pooling="mean", k=n_knn)
        elif backbone == 'vn_unet_v2':
            self.backbone = EQCNN_equi_unet_v2(feat_dim=self.feat_dim, pooling="mean", k=n_knn)
        else:
            raise NotImplementedError("DGCNN backbone not implemented")
        
        if double_bacbone == 'vn_unet':
            self.ori_backbone = EQCNN_equi_unet(feat_dim=self.feat_dim, pooling="mean", k=n_knn)
        elif double_bacbone == 'vn_unet_v2':
            self.ori_backbone = EQCNN_equi_unet_v2(feat_dim=self.feat_dim, pooling="mean", k=n_knn)
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

        if mlp_mode == 'deep':
            self.shape_mlp = nn.Sequential(nn.Conv1d((self.feat_dim//3) * 3, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim//2, self.feat_dim, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           )
        
        elif mlp_mode == 'half':
            self.shape_mlp = nn.Sequential(nn.Conv1d((self.feat_dim//3) * 3, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim//2, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           )
        
        
        elif mlp_mode == 'CMpp':
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
        
        elif mlp_mode == 'CMpp_half':
            self.shape_mlp = nn.Sequential(nn.Conv1d((self.feat_dim//3) * 3, self.feat_dim//2, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim//2),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim//2, self.feat_dim, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
                                           nn.InstanceNorm1d(self.feat_dim),
                                           nn.LeakyReLU(negative_slope=0.2),
                                           )
        
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


    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        # Lightning 2.x: Support this funcionality
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        max_epochs = self.trainer.max_epochs

        assert total_steps > 0, "Total steps must be greater than 0"

        if self.learnable_softmax_temperature:
            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in self.named_parameters() if 'softmax_temperature' not in n]},
                {'params': self.softmax_temperature, 'lr': self.lr * 0.1} 
                ],  lr=self.lr, weight_decay=0.) # We use 10% of the learning rate for softmax temperature
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.)
        
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

        Returns:
            out_dict (dict)
                - During training,
                    - o_loss: (1, )
                    - s_loss: (1, )
                    - p_loss: (1, )
                    - loss: (1, )
                
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
                
                - During validation or test, the following keys are added
                    - cd: (1, )
                    - crd: (1, )
                    - rrmse: (1, )
                    - trmse: (1, )
                    - rpf_rmse: (1, )
                    - rpf_tmse: (1, )
        """
        assert in_dict['pcd_batch_info'].max() == 1, f"We assume there are two objects in the batch, but got {in_dict['pcd_batch_info'].max()}"

        out_dict, loss = {}, {}

        # 0. Get Point Clouds and Ground Truth Correspondence
        pcd_raw = in_dict['pcd'] # (B, N+M, 3)
        pcd_input = in_dict['pcd_t'] # (B, N+M, 3)
        gt_normals = in_dict['gt_normals'] # (B, N+M, 3)
        pcd_batch_info = in_dict['pcd_batch_info'] # (B, N+M, )
        batch_scaled_pcd_batch_info = batch_scaling(pcd_batch_info) # (B, N+M, )


        # 1. SO(3)-Equivariant Feature Extractor
        equi_feats_backbone = self.backbone(pcd_input, batch_scaled_pcd_batch_info) # (B, C, 3, N+M)


        # 2. Calculate equivariant shape features
        equi_feats = self.equi_layer(equi_feats_backbone.unsqueeze(-1)).squeeze(-1) # (B, C, 3, N+M)


        # 3. Frame Prediction
        equi_feats_ori_backbone = self.ori_backbone(pcd_input, batch_scaled_pcd_batch_info) if self.ori_backbone is not None else equi_feats_backbone

        # 3-1. Merge global information by averaging
        # (B, C, 3, N+M) -> (B, C, 3, 1) -> (B, C, 3, N+M)
        equi_feats_ori_backbone_mean = equi_feats_ori_backbone.mean(dim=-1, keepdim=True).expand(equi_feats_ori_backbone.size())

        # 3-2. Basis Vector Projection, those vectors will be used as frame basis vectors
        # (B, C, 3, N+M) concat (B, C, 3, N+M) ->  (B, 2C, 3, N+M) -> (B, 2C, 3, N+M, 1) -> (B, 2, 3, N+M, 1) -> (B, 2, 3, N+M) -> (B, N+M, 2, 3)
        vecs = self.proj(torch.cat((equi_feats_ori_backbone, equi_feats_ori_backbone_mean), dim=1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2) 


        # 4. Gram Schmidt & Cross-product, this is for making three basis vectors by using two predicted vectors
        if self.normal_pred_mode == 'cross':
            if self.flip_normal_mode == 'rightv4':
                oris = src_reverse_trg_normal_gram_schmidt_with_cross(vecs, pcd_batch_info) # (B, N+M, 3, 3)
            else:
                oris = gram_schmidt_with_cross(vecs) # (B, N+M, 2, 3) -> (B, N+M, 3, 3)
        elif self.normal_pred_mode == 'gram':
            oris = gram_schmidt(vecs) # (B, N+M, 3, 3) -> (B, N+M, 3, 3)
        else:
            raise ValueError(f"normal_pred_mode must be in ['cross', 'gram'], but got {self.normal_pred_mode}")
        
        out_dict['oris'] = oris


        # Only train the normal vector
        if self.only_train_normal:
            loss['o_loss'], loss['o_consistency_loss'] = self.orientation_loss(oris, gt_normals, batch_scaled_pcd_batch_info, None, pcd_raw, self.return_active_mask(pcd_batch_info))
            loss['loss'] = loss['o_loss']

            # Compute Normal Error
            with torch.no_grad():
                # (d) Compute Normal Error
                loss['n_error'], _, loss['n_suc_rate'] = self._normal_error(in_dict, out_dict, success_criterion_in_degree=self.success_criterion_in_degree)
            
            if mode == 'train':
                self.log_for_training(loss=loss, pos_neg_distribution=None, mode=mode)
            
            return out_dict, loss

        
        # 5. Invariant Features
        inv_feats = self.make_inv_feats(oris, pcd_batch_info, equi_feats, src_flip=True) # (B, C*3, N+M)
        if self.flip_normal_mode != 'none' and mode in ['train', 'val']:
            symmetric_inv_feats = self.make_inv_feats(oris, pcd_batch_info, equi_feats, src_flip=False) # (B, C*3, N+M)
        

        # 6. SHAPE DESCRIPTOR 
        shape_feats = self.shape_mlp(inv_feats) # (B, C*3, N+M) -> (B, D, N+M)
        if self.flip_normal_mode != 'none' and mode in ['train', 'val']:
            symmetric_shape_feats = self.shape_mlp(symmetric_inv_feats) # (B, C*3, N+M) -> (B, D, N+M)
        

        # 7. Calculate Matching Scores
        active_mask = self.return_active_mask(pcd_batch_info)
        shape_matching_scores = self.calculate_matching_score(shape_feats, active_mask, eps=1e-8, mode=self.matching_score_mode)
        if self.flip_normal_mode != 'none' and mode in ['train', 'val']:
            symmetric_shape_matching_scores = self.calculate_matching_score(symmetric_shape_feats, active_mask, eps=1e-8, mode=self.matching_score_mode)
        

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
            
            # oris, gt_normals, batch_scaled_batch_info, coords_dist, pcd_raw, active_mask
            loss['o_loss'], loss['o_consistency_loss'], loss['o_consistency_loss_2nd'], loss['o_consistency_loss_3rd'] = self.orientation_loss(oris, gt_normals, batch_scaled_pcd_batch_info, coords_dist, pcd_raw, active_mask)
            loss['loss'] = self.o_loss_weight * loss['o_loss'] + self.s_loss_weight * loss['s_loss'] + self.p_loss_weight * loss['p_loss']
            
            out_dict.update(loss)

            if mode == 'train':
                with torch.no_grad():
                    # This is for checking the normal error
                    loss['n_error'], _, loss['n_suc_rate'] = self._normal_error(in_dict, out_dict, success_criterion_in_degree=self.success_criterion_in_degree)

                    if ((not self.trainer.sanity_checking) and self.viz_train_epoch > 0 and batch_idx == 0 and (self.current_epoch % self.viz_train_epoch == 0 or self.current_epoch == self.trainer.max_epochs-1)):
                        visualize_negative_hard_mask(in_dict, neg_hard_mask_for_viz['neg_mask'], neg_hard_mask_for_viz['hard_neg_mask'], active_mask, self.ckp_dir, self.current_epoch, self.trainer.global_rank, self.pos_radius, self.safe_radius)
                        # exit("stop")
        

        # 9. Evaluation
        if mode in ['val', 'test']:
            # Save output for evaluation
            out_dict['shape_matching_scores'] = shape_matching_scores
            out_dict['matching_scores_drop'] = matching_scores_drop
            out_dict['active_mask'] = active_mask
            out_dict, eval_dict = self.progress_evaluation(in_dict, out_dict, mode)
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
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log_dict(log_dict, prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=True)
        self.log(f'{mode}/loss', training_loss, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=True)
        self.log('current_lr', current_lr, prog_bar=True, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False)
    

    def make_inv_feats(self, oris, oris_batch_info, equi_feats, src_flip=True):
        """Make invariant features
        Assume there are two objects in the batch
        
        Args:
            oris (torch.Tensor): (B, N+M, 3, 3)
            oris_batch_info (torch.Tensor): (B, N+M, ), batch index of the point cloud
            equi_feats (torch.Tensor): (B, C, 3, N+M)
            src_flip (bool, optional): Whether to flip the normal vector of src. Defaults to True.

        Returns:
            inv_feats (torch.Tensor): (B, C*3, N)
        """

        if self.flip_normal_mode in ['right', 'rightv1_2', 'rightv1_3', 'rightv2', 'rightv3', 'rightv4', 'mix']:
            # (B, N+M, 3, 3)
            if self.flip_normal_mode == 'right':
                postprocessed_oris = torch.stack([- oris[:, :, 0, :], oris[:, :, 2, :], oris[:, :, 1, :]], dim=-2)
            elif self.flip_normal_mode == 'rightv1_2':
                rotation_matrix = rodrigues_to_rotmat(oris[:, :, 0, :], torch.ones_like(oris[:, :, 0, 0]) * 90.0)
                rotated_oris = rotate_by_rotation_matrix(oris, rotation_matrix)
                postprocessed_oris = torch.stack([- rotated_oris[:, :, 0, :], rotated_oris[:, :, 1, :], - rotated_oris[:, :, 2, :]], dim=-2)
            elif self.flip_normal_mode == 'rightv1_3':
                rotation_axis = nn.functional.normalize(oris[:, :, 1, :] + oris[:, :, 2, :], dim=-1) # (B, N, 3)
                rotation_matrix = rodrigues_to_rotmat(rotation_axis, torch.ones_like(oris[:, :, 0, 0]) *  180)
                postprocessed_oris = rotate_by_rotation_matrix(oris, rotation_matrix)
            elif self.flip_normal_mode == 'rightv2':
                postprocessed_oris = torch.stack([- oris[:, :, 0, :], oris[:, :, 1, :], - oris[:, :, 2, :]], dim=-2)
            elif self.flip_normal_mode == 'rightv3':
                postprocessed_oris = torch.stack([- oris[:, :, 0, :], - oris[:, :, 1, :], oris[:, :, 2, :]], dim=-2)
            elif self.flip_normal_mode in ['rightv4', 'mix']:
                postprocessed_oris = torch.stack([- oris[:, :, 0, :], oris[:, :, 1, :], oris[:, :, 2, :]], dim=-2)
            
            if src_flip: # Flip the normal vector of src
                # We assume there are two objects in the batch
                src_batch_info = oris_batch_info == 0 # (B, N+M, )
                result_oris = postprocessed_oris * src_batch_info[:,:,None,None] + oris * (~ src_batch_info)[:,:,None,None]
            
            else: # Flip the normal vector of trg
                trg_batch_info = oris_batch_info == 1 # (B, N+M, )
                result_oris = postprocessed_oris * trg_batch_info[:,:,None,None] + oris * (~ trg_batch_info)[:,:,None,None]
        
        elif self.flip_normal_mode == 'none':
            result_oris = oris
        
        else:
            raise ValueError(f"flip_normal_mode must be in ['right', 'mix', 'none'], but got {self.flip_normal_mode}")
        
        # (B, C, 3, N) -> (B, N, C, 3) @ (B, N, 3, 3) -> (B, N, 3, 3) => (B, N, C, 3)
        inv_feats = torch.matmul(equi_feats.permute(0, 3, 1, 2).float(), result_oris.transpose(-2,-1).float()) 
        inv_feats = rearrange(inv_feats, 'b n c r -> b (c r) n') # (B, N, C, 3) -> (B, C*3, N)
        return inv_feats
    

    def return_active_mask(self, batch_info):
        """
        Return active mask between the different objects

        Args:
            batch_info (torch.Tensor): (B, N+M, ), batch index of the point cloud

        Returns:
            active_parts (torch.Tensor): (B, N+M, N+M), True if the point is active
        """
        # Leave only the matching scores between the different objects
        # Right-Upper part is only left
        num_of_points = batch_info.size(1)
        repeated_batch_info_row_for_src = batch_info[:,:,None].expand(-1, -1, num_of_points) == 0  # (B, N+M, N+M)
        repeated_batch_info_col_for_trg = batch_info[:,None,:].expand(-1, num_of_points, -1) == 1 # (B, N+M, N+M)
        active_parts = torch.logical_and(repeated_batch_info_row_for_src, repeated_batch_info_col_for_trg) # (B, N+M, N+M))
        return active_parts
    
    
    def calculate_matching_score(self, shape_feats, active_mask, eps=1e-8, mode='CM'):
        """
        Calculate matching score between src and trg features
        Assume there are two objects in the batch

        Args:
            shape_feats (torch.Tensor): (B, D, N+M)
            active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active
            eps (float, optional): Epsilon for avoiding division by zero. Defaults to 1e-8.
            mode (str, optional): 'CM' or 'cossim'. Defaults to 'CM'.
        Returns:
            matching_scores (torch.Tensor): (B, N+M, N+M)
        """

        if mode == 'CM':
            matching_scores = torch.einsum('b c n , b c m -> b n m', shape_feats, shape_feats) # (B, N+M, N+M)
            matching_scores = matching_scores / (shape_feats.shape[1] ** 0.5 + eps) # 1e-8 is for avoiding division by zero
        
        else:
            normalized_shape_feats = nn.functional.normalize(shape_feats, p=2, dim=1) # (B, D, N+M)
            matching_scores = torch.einsum('b c n , b c m -> b n m', normalized_shape_feats, normalized_shape_feats) # (B, N+M, N+M)

        # Remove the matching scores between the same objects
        matching_scores = matching_scores * active_mask
        return matching_scores
    
    
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

        batch_size, row_size, col_size = matching_scores.shape

        if mode == 'sinkhorn':
            result_list = []

            for batch_idx in range(batch_size):
                # Postprocess matching scores to make its shape (N, M)
                pcd_num_info = batch_info[batch_idx].bincount() # (2, )
                assert len(pcd_num_info) == 2, f"There must be two objects in the batch, but got {len(pcd_num_info)}"
                
                num_src_pcd, num_trg_pcd = pcd_num_info
                postprocessed_matching_scores = matching_scores[batch_idx][active_mask[batch_idx]] # (N*M,)
                postprocessed_matching_scores = postprocessed_matching_scores.reshape(1, num_src_pcd, num_trg_pcd) # (1, N, M)

                normalized_matching_scores = self.optimal_transport(postprocessed_matching_scores).squeeze(0) # (1, N+1, M+1) -> (N+1, M+1)

                # Recover shape
                place_holder = torch.zeros(row_size+1, col_size+1, device=matching_scores.device)
                place_holder[:num_src_pcd, (col_size-num_trg_pcd):-1] = normalized_matching_scores[:-1,:-1]
                place_holder[:num_src_pcd,-1] = normalized_matching_scores[:-1,-1]
                place_holder[-1,(col_size-num_trg_pcd):-1] = normalized_matching_scores[-1,:-1]
                place_holder[-1,-1] = normalized_matching_scores[-1,-1]

                result_list.append(place_holder)
            
            result = torch.stack(result_list, dim=0) # (B, N+M+1, N+M+1)
        
        
        elif mode == 'softmax':
            # Calculate active mask
            place_holder_active_mask = torch.zeros(batch_size, row_size+1, col_size+1, device=matching_scores.device, dtype=torch.bool) # (B, N+M+1, N+M+1)
            place_holder_active_mask[:, :-1, :-1] = active_mask # (B, N+M, N+M)
            place_holder_active_mask[:, :-1, -1] = active_mask.any(dim=-1) # (B, N+M)
            place_holder_active_mask[:, -1, :-1] = active_mask.any(dim=-2) # (B, N+M)
            place_holder_active_mask[:, -1, -1] = True

            # Calculate padded matching scores
            place_holder = torch.zeros(batch_size, row_size+1, col_size+1, device=matching_scores.device) # (B, N+M+1, N+M+1)
            place_holder[:, :-1, :-1] = matching_scores # (B, N+M, N+M)
            place_holder[:, :-1, -1] = self.slack_variable.expand(batch_size, row_size)
            place_holder[:, -1, :] = self.slack_variable.expand(batch_size, col_size+1)
            place_holder = place_holder * place_holder_active_mask + -1e12 * (~place_holder_active_mask)

            row_softmax_matching_scores = nn.functional.softmax(place_holder / self.softmax_temperature, dim=-1)
            col_softmax_matching_scores = nn.functional.softmax(place_holder / self.softmax_temperature, dim=-2)
            softmax_matching_scores = (row_softmax_matching_scores + col_softmax_matching_scores) / 2
            softmax_matching_scores[:, :-1, -1] = row_softmax_matching_scores[:, :-1, -1] # Fill the last column with the row softmax matching scores
            softmax_matching_scores[:, -1, :-1] = col_softmax_matching_scores[:, -1, :-1] # Fill the last row with the col softmax matching scores
            softmax_matching_scores = softmax_matching_scores * place_holder_active_mask

            result = softmax_matching_scores

        
        elif mode == 'none':
            result = matching_scores
        
        return result
    
    
    @torch.no_grad()
    def progress_evaluation(self, in_dict, out_dict, mode):
        """
        Evaluate the progress of the model
        Batch size must be 1 for evaluation

        Args:
            in_dict (dict): it is same as forward_pass
            out_dict (dict): it is same as forward_pass
            mode (str): 'val' or 'test'
        """
        assert mode in ['val', 'test'], f"mode must be in ['val', 'test'], but got {mode}"
        assert in_dict['pcd'].shape[0] == 1, f"in_dict['pcd'].shape[0]: {in_dict['pcd'].shape[0]}, must be 1"

        # Postprocess input/output to fit the evaluation function
        # Dataloader will returns (B, N+M, ....) format.
        # However, batch size must be 1 for evaluation
        # So, we will use src/trg individually for evaluation
        src_pcd_raw, trg_pcd_raw = extract_all_objects(in_dict['pcd'][0], in_dict['pcd_batch_info'][0]) # (N, 3), (M, 3)
        src_pcd, trg_pcd = extract_all_objects(in_dict['pcd_t'][0], in_dict['pcd_batch_info'][0]) # (N, 3), (M, 3)
        src_ori, trg_ori = extract_all_objects(out_dict['oris'][0], in_dict['pcd_batch_info'][0]) # (N, 3, 3), (M, 3, 3)
        gt_src_normals, gt_trg_normals = extract_all_objects(in_dict['gt_normals'][0].float(), in_dict['pcd_batch_info'][0]) # (N, 3), (M, 3)
        num_src_pcd, num_trg_pcd = src_pcd.shape[0], trg_pcd.shape[0] # (N), (M)
        out_shape_matching_scores = out_dict['shape_matching_scores'][0] # (N+M, N+M)
        out_matching_scores_drop = out_dict['matching_scores_drop'][0] # (N+M, N+M)
        out_active_mask = out_dict['active_mask'][0] # (N+M, N+M)

        # Postprocess matching scores to make its shape (N, M)
        postprocessed_shape_matching_scores = out_shape_matching_scores[out_active_mask] # (N*M)
        postprocessed_matching_scores_drop = out_matching_scores_drop[out_active_mask] # (N*M)
        postprocessed_shape_matching_scores = postprocessed_shape_matching_scores.reshape(num_src_pcd, num_trg_pcd) # (N, M)
        postprocessed_matching_scores_drop = postprocessed_matching_scores_drop.reshape(num_src_pcd, num_trg_pcd) # (N, M)

        # Calculate ground truth correspondence
        gt_corr = torch.nonzero(torch.cdist(src_pcd_raw, trg_pcd_raw, p=2) < self.pos_radius) # (corr, 2)

        # Save split tensors for evaluating prediction
        split_input_dict = {
            'src_pcd_raw': src_pcd_raw, # (N, 3)
            'trg_pcd_raw': trg_pcd_raw, # (M, 3)
            'src_pcd': src_pcd, # (N, 3)
            'trg_pcd': trg_pcd, # (M, 3)
            'src_ori': src_ori, # (N, 3, 3)
            'trg_ori': trg_ori, # (M, 3, 3)
            'gt_src_normals': gt_src_normals, # (N, 3)
            'gt_trg_normals': gt_trg_normals, # (M, 3)
            'gt_corr': gt_corr, # (corr, 2)
        }

        # Point cloud registration
        src_predicted_frame = src_ori if self.use_predicted_normal else None # (N, 3, 3)
        trg_predicted_frame = trg_ori if self.use_predicted_normal else None # (M, 3, 3)

        if self.use_RANSAC:
            estimated_transform, used_corr = _RANSAC(in_dict=in_dict, 
                                                     shape_matching_scores=postprocessed_shape_matching_scores, 
                                                     src_pcd=src_pcd, 
                                                     trg_pcd=trg_pcd, 
                                                     src_predicted_frame=src_predicted_frame,
                                                     trg_predicted_frame=trg_predicted_frame,
                                                     match_option=self.infer_match_option, 
                                                     RANSAC_type=self.RANSAC_type, 
                                                     topk=self.infer_topk)

        else:
            # fine_matching predict Rt to move points from src_points to ref_points
            estimated_transform, used_corr = self.fine_matching(src_pcd.unsqueeze(0), trg_pcd.unsqueeze(0), postprocessed_matching_scores_drop.unsqueeze(0), no_exp=(self.matching_norm_mode != 'sinkhorn'))

        # estimated_transform: target_point = R * source_point + t
        out_dict['estimated_rotat'] = estimated_transform[:3, :3] # R, (3,3)
        out_dict['estimated_trans'] = estimated_transform[:3, 3] # t, (3)
        out_dict['used_corr'] = used_corr # (K, 2)

        # Evaluation
        eval_dict = self.evaluate_prediction(in_dict, split_input_dict, out_dict, mode)

        # Matching Recall
        eval_dict.update(self._calculate_recall(postprocessed_matching_scores_drop, gt_corr))

        # Calculate ratio of GT among topk scores
        eval_dict['gt_among_topk'] = self.calculate_ratio_of_gt_among_topk_scores(src_pcd_raw, trg_pcd_raw, postprocessed_matching_scores_drop, topk=self.infer_topk, pos_radius=self.pos_radius)

        # log size of gt_corr
        eval_dict['gt_corr_size'] = torch.tensor(gt_corr.shape[0]).to(src_pcd_raw.device)

        return out_dict, eval_dict
    

    @torch.no_grad()
    def evaluate_prediction(self, in_dict, split_input_dict, out_dict, mode,):
        """
        Args:
            in_dict (dict): it is same as forward_pass
            split_input_dict (dict): split input dictionary for evaluation
            out_dict (dict): it is same as forward_pass
            mode (str): 'val' or 'test'

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
        grtr_relative_trsfm = [x for x in in_dict['relative_trsfm']['0-1']] # (3, 3), (3)
        src_pcd, trg_pcd = split_input_dict['src_pcd'], split_input_dict['trg_pcd'] # (N, 3), (M, 3)
        gt_corr = split_input_dict['gt_corr'] # (corr, 2)
        used_corr = out_dict['used_corr'] # (K, 2)


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
            used_corr = torch.stack([used_corr[:,1], used_corr[:,0]], dim=1) # (K, 2)
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
        eval_result['rrmse_rpf'], eval_result['trmse_rpf'] = self._transformation_error_RPFver(pcds_pred, pcds_grtr)
        eval_result['rrmse'], eval_result['trmse'] = self._transformation_error(pred_relative_trsfm, grtr_relative_trsfm)
        eval_result['rrmse_geo'], eval_result['trmse_geo'] = self._transformation_error_geodesic(pred_relative_trsfm, grtr_relative_trsfm)

        # (c) Compute CoRrespondence Distance (CRD) betwween prediction & ground-truth
        eval_result['crd'] = self._correspondence_distance(assm_pred, assm_grtr)

        # (d) Compute Normal Error
        eval_result['n_error'], normal_error_hist, eval_result['n_suc_rate'] = self._normal_error(in_dict, out_dict, success_criterion_in_degree=self.success_criterion_in_degree)

        if mode == 'test':
            if self.viz_metric_name == 'none':
                # Visualization is only depend on self.visualize
                metric_based_visualization = True
            else:
                # Only visualize if the metric is greater than the threshold
                metric_based_visualization = eval_result[self.viz_metric_name] >= self.viz_metric_threshold
                # metric_based_visualization = in_dict['filepath'][0] == 'everyday/Bottle/d851cbc873de1c4d3b6eb309177a6753/mode_1'

        if (mode =='val' and (not self.trainer.sanity_checking) and \
            self.trainer.global_rank == 0 and \
            (self.visualize_mode != 'none') and \
            (self.current_epoch % self.viz_epoch == 0 or self.current_epoch == self.trainer.max_epochs-1) and \
            in_dict['eval_idx'][0].item() == 0) or \
            (mode =='test' and (self.visualize_mode != 'none') and metric_based_visualization):
            # Do not visualize in sanity checking
            # Only rank 0 should do visualization to avoid file I/O conflicts in DDP
            # Visualize for every self.viz_epoch
            # However, if it is the last epoch, then visualize
            # Also, only visualize first batch

            # Name of case
            case_name = in_dict["filepath"][0].replace('/', '_')

            vis_folder = os.path.join(self.ckp_dir, 'vis', f'GPU_{self.trainer.global_rank}', mode, case_name) # For mesh visualization
            vis_hist_folder = os.path.join(self.ckp_dir, 'vis_hist', f'GPU_{self.trainer.global_rank}', mode, case_name) # For normal error histogram visualization
            os.makedirs(vis_folder, exist_ok=True)
            os.makedirs(vis_hist_folder, exist_ok=True)

            # PCD light visualization
            save_pcd_for_light_visualization(pcds_pred, gt_corr, used_corr, f'{vis_folder}/E{self.current_epoch}_{in_dict['eval_idx'][0].item()}_{in_dict['obj_class'][0]}_{round(eval_result['crd'].item(),3)}_pred_top{self.infer_topk}')
            save_pcd_for_light_visualization(pcds_grtr, gt_corr, used_corr, f'{vis_folder}/E{self.current_epoch}_{in_dict['eval_idx'][0].item()}_{in_dict['obj_class'][0]}_{round(eval_result['crd'].item(),3)}_grtr_top{self.infer_topk}')


            # MESH AND FRAME VISUALIZATION
            if self.visualize_mode == 'all':
                output_src_ori, output_trg_ori = split_input_dict['src_ori'], split_input_dict['trg_ori'] # (N, 3, 3), (M, 3, 3)
                gt_src_normals, gt_trg_normals = split_input_dict['gt_src_normals'], split_input_dict['gt_trg_normals'] # (N, 3), (M, 3)
                src_mesh_verts, trg_mesh_verts = in_dict['mesh_t'][0].float(), in_dict['mesh_t'][1].float() # (N,3), (M,3)
                src_mesh_faces, trg_mesh_faces = in_dict['mesh_faces'][0].float(), in_dict['mesh_faces'][1].float() # (F,3), (F',3)
                
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
            

            # exit("stop")
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
    

    def _transformation_error(self, trnsf1, trnsf2, trmse_scaling=100):
        """
        Args:
            trnsf1 (tuple): (3, 3), (3)
            trnsf2 (tuple): (3, 3), (3)
            trmse_scaling (int, optional): Scaling factor for TRMSE. Defaults to 100.

        Returns:
            rrmse (torch.Tensor): (1)
            trmse (torch.Tensor): (1)
        """
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
        
        # div = len(rotat1) if multi_part else 1
        div = 1
        return (rrmse / div).to(trmse.device), trmse / div


    def _transformation_error_geodesic(self, trnsf1, trnsf2, trmse_scaling=100):
        """
        Args:
            trnsf1 (tuple): (3, 3), (3)
            trnsf2 (tuple): (3, 3), (3)
            trmse_scaling (int, optional): Scaling factor for TRMSE. Defaults to 100.

        Returns:
            rrmse (torch.Tensor): (1)
            trmse (torch.Tensor): (1)
        """
        rotat1, trans1 = [trnsf1[0]], [trnsf1[1]]
        rotat2, trans2 = [trnsf2[0]], [trnsf2[1]]
        
        rrmse_geo, trmse_geo = 0., 0.
        for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
            # pred_rotat^T @ gt_rotat
            relative_rotat = r1 @ r2.T

            # tr(R) = 1 + 2cos(θ) -> θ = acos((tr(R) - 1) / 2), torch.acos is in radian, so we need to convert to degree
            rrmse_geo += torch.rad2deg(torch.acos(torch.clamp(0.5 * (torch.trace(relative_rotat) - 1.0), -1.0, 1.0)))
            trmse_geo += torch.norm(t1 - t2) * trmse_scaling
        
        div = 1
        return (rrmse_geo / div).to(trmse_geo.device), trmse_geo / div


    def _transformation_error_RPFver(self, pcds_pred, pcds_grtr, scaling=100):
        """
        Args:
            pcds_pred (list): [(N, 3), (M, 3)]
            pcds_grtr (list): [(N, 3), (M, 3)]
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
            in_dict (dict): it is same as forward_pass. From in_dict, only need gt_normals, which is torch.Tensor: (B, N+M, 3)
            out_dict (dict): it is same as forward_pass. From out_dict, only need src_ori and trg_ori, which are torch.Tensor: (B, N+M, 3, 3) and torch.Tensor: (B, N+M, 3, 3)
            success_criterion_in_degree (int, optional): Success criterion in degree. Defaults to 10.

        Returns:
            normal_error (torch.Tensor): (1)
        """
        pred_oris = out_dict['oris'] #  (B, N+M, 3, 3)
        gt_normals = in_dict['gt_normals'] # (B, N+M, 3)

        pred_normals = pred_oris[:,:,0,:] # (B, N+M, 3)

        cosine_similarity = torch.clamp(torch.nn.functional.cosine_similarity(pred_normals, gt_normals, dim=-1), min=-1, max=1) # (B, N+M, )
        theta_deg = torch.rad2deg(torch.acos(cosine_similarity)).reshape(-1) # (B, N+M, ) -> (B*(N+M), )
        
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
            matching_scores_drop (torch.Tensor): (N, M)
            gt_corr (torch.Tensor): (P, 2)
            topks (list, optional): Recall@1, Recall@5, Recall@10, Recall@20.

        Returns:
            matching_recall (torch.Tensor): (1)
        """
        _N, _M = matching_scores_drop.shape # (N, M)

        result_dict = dict()

        if len(gt_corr) == 0:
            for topk in topks:
                result_dict[f"recall@{str(topk)}"] = 0.0
            return result_dict

        correspondence_mask = torch.zeros((_N, _M), device=matching_scores_drop.device)
        correspondence_mask[gt_corr[:,0], gt_corr[:,1]] = True
        correspondence_mask_src = correspondence_mask.sum(dim=-1) > 0 # (N, M) -> N
        correspondence_mask_trg = correspondence_mask.sum(dim=-2) > 0 # (N, M) -> M
        
        for topk in topks:
            ## Recall from src
            _, topk_inds_src = torch.topk(matching_scores_drop[correspondence_mask_src, :], k=topk, dim=-1) # (N, M) -> (gt_N, M) -> (gt_N, topk)
            topk_mask_src = torch.zeros((_N, _M), device=matching_scores_drop.device) # (N, M)
            for i, _ in enumerate(range(topk_inds_src.shape[-1])): # for i in range(topk)
                # [all gt_N, ith topk from gt_src]
                topk_mask_src[torch.nonzero(correspondence_mask_src)[:, 0], topk_inds_src[:, i]] = True

            # (N, M) -> N
            is_success_src = (topk_mask_src * correspondence_mask).sum(dim=-1) > 0
            recall_src = is_success_src[correspondence_mask_src].sum() / correspondence_mask_src.sum()

            ## Recall from trg
            _, topk_inds_trg = torch.topk(matching_scores_drop[:, correspondence_mask_trg], k=topk, dim=-2) # (N, M) -> (N, gt_M) -> (topk, gt_M)
            topk_mask_trg = torch.zeros((_N, _M), device=matching_scores_drop.device) # (N, M)
            for i, _ in enumerate(range(topk_inds_trg.shape[-2])): # for i in range(topk)
                # [ith topk from gt_trg, all gt_N]
                topk_mask_trg[topk_inds_trg[i, :], torch.nonzero(correspondence_mask_trg)[:, 0]] = True

            # (N, M) -> M
            is_success_trg = (topk_mask_trg * correspondence_mask).sum(dim=-2) > 0
            recall_trg = is_success_trg[correspondence_mask_trg].sum() / correspondence_mask_trg.sum()
            
            recall_dot_k = (recall_src + recall_trg) / 2

            ## Logging results
            result_dict[f"recall@{str(topk)}"] = recall_dot_k
        
        return result_dict
    

    def calculate_ratio_of_gt_among_topk_scores(self, src_pcd_raw, trg_pcd_raw, matching_scores, topk=128, pos_radius=0.018):
        """Calculate ratio of GT among topk scores

        Args:
            src_pcd (torch.Tensor): (N, 3)
            trg_pcd (torch.Tensor): (M, 3)
            matching_scores (torch.Tensor): (N, M)
            topk (int, optional): Topk value for matching. Defaults to 128.

        Returns:
            ratio_of_gt_among_topk_scores (torch.Tensor): (1)
        """
        # Calculate distance between source and target points, and check if it is within the positive radius
        corr_dist = torch.cdist(src_pcd_raw, trg_pcd_raw, p=2) # (N, M)
        pos_mask = corr_dist < pos_radius # (N, M)

        # Find pairs that have topk scores
        topk_scores = torch.topk(matching_scores.reshape(-1), k=topk, dim=-1)[0] # (N*M) -> (topk)
        kth_biggest_score = topk_scores[-1] # (topk) -> (1, )
        topk_mask = matching_scores >= kth_biggest_score # (N, M)

        # Calculate ratio of GT among topk scores
        ratio_of_gt_among_topk_scores = torch.logical_and(topk_mask, pos_mask).sum() / topk_mask.sum() # (N, M) -> (1, )
        return ratio_of_gt_among_topk_scores






