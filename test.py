import os
import argparse
from typing import Optional

import torch

from data.dataset import GADataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger, CSVLogger


@torch.no_grad()
def test(args):
    seed_everything(42, workers=True)


    # Create checkpoint directory
    cfg_name = args.logpath
    ckp_dir = os.path.join('checkpoint', cfg_name, 'models')
    os.makedirs(ckp_dir, exist_ok=True)

    # Print checkpoint directory
    print(f"checkpoint directory (ckp_dir): {ckp_dir}")


    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.min_part, args.max_part, args.n_pts, args.scale, args.multiplicity, CMorigin_mode=(args.model == 'CM_equiassem'))
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val')


    # Model initialization
    if args.model == 'CM_equiassem':
        from model.CM_equiassem import EquiAssem
        model = EquiAssem(lr=0, # We don't need to use learning rate for testing
                          backbone=args.backbone,
                          shape_loss=args.shape_loss, 
                          occ_loss=args.occ_loss, 
                          no_ori=args.no_ori,
                          attention=args.attention,
                          visualize=args.visualize,
                          debug=args.debug)
        
    elif args.model == 'CMpp_equiassem': # Import developing mode model
        from model.CMpp_equiassem import EquiAssem
        model = EquiAssem(lr=0, # We don't need to use learning rate for testing
                          
                          scheduler_mode=None,
                          training_total_steps=0, # We don't need to use training total steps for testing

                          backbone=args.backbone,
                          attention=args.attention,

                          # Circle loss arguments
                          pos_margin=0, # We don't need to use circle loss for testing  
                          neg_margin=0, # We don't need to use circle loss for testing
                          log_scale=1, # We don't need to use circle loss for testing
                          detach_mode=False, # We don't need to use circle loss for testing
                          same_opt=False, # We don't need to use circle loss for testing
                          only_corr=False, # We don't need to use circle loss for testing
                          max_points=0, # We don't need to use circle loss for testing
                          no_balance=False, # We don't need to use circle loss for testing
                          div_mode='none', # We don't need to use circle loss for testing

                          s_loss_weight=0.0, # We don't need to use circle loss for testing
                          p_loss_weight=0.0, # We don't need to use circle loss for testing
                          o_loss_weight=0.0, # We don't need to use circle loss for testing

                          visualize=args.visualize,
                          viz_epoch=args.viz_epoch,
                          viz_max_arrow_num=args.viz_max_arrow_num,
                          ckp_dir=ckp_dir,
                          debug=args.debug, 
                          success_criterion_in_degree=args.success_criterion_in_degree,
                          delete_Sinkhorn=args.delete_Sinkhorn,
                          use_Sinkhorn_infer=args.use_Sinkhorn_infer,
                          matching_score_mode=args.matching_score_mode,
                          svd_no_exp=args.svd_no_exp,
                          flip_normal=args.flip_normal,
                          use_consistency_loss=False, # We don't need to use consistency loss for testing
                          only_train_normal=False, # We don't need to use only train normal for testing
                          freeze_normal_param=False, # We don't need to use freeze normal param for testing

                          additional_VNLinearLeakyReLU=args.additional_VNLinearLeakyReLU,
                          debugged_circle_loss=args.debugged_circle_loss,
                          debugged_point_matching_loss=args.debugged_point_matching_loss,
                          exp_scale_for_point_matching_loss=args.exp_scale_for_point_matching_loss,
                          n_knn=args.n_knn,
                          new_orientation_module=args.new_orientation_module,
                          delete_occupancy_loss=args.delete_occupancy_loss,
                          use_opt_gram=args.use_opt_gram,
                          
                          only_one_norm=args.only_one_norm,
                          n_avn=args.n_avn,
                          move_smaller=args.move_smaller,
                          
                          infer_match_option=args.infer_match_option,
                          infer_topk=args.infer_topk,
                          infer_score_threshold_ratio=args.infer_score_threshold_ratio,
                          use_RANSAC=args.use_RANSAC,
                          RANSAC_type=args.RANSAC_type,
                          use_predicted_normal=args.use_predicted_normal
                          )
    else:
        raise NotImplementedError("Model not implemented")


    
    # Wandb logger
    if args.wandb:
        logger = WandbLogger(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.logpath, # same as logpath
            id=None, 
            save_dir=ckp_dir,
            tags=[args.scale],
        )
    else:
        # CSV logger for saving all metrics in a text file
        logger = CSVLogger(
            save_dir=ckp_dir,
            name="csv_logs",
            version=None,
        )
    
    
    all_gpus = list(args.gpus)
    print(f"all_gpus: {all_gpus}")

    ckpt_path = args.load
    assert ckpt_path is not None, "Checkpoint path is not set"


    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=all_gpus,
        precision=32,
        deterministic=args.deterministic,
        strategy=args.parallel_strategy,
    )


    trainer.test(model, dataloader_val, ckpt_path=ckpt_path)
    results = model.test_results
    results = {k[5:]: v.detach().cpu().numpy() for k, v in results.items()}
    print('--------------------------------')
    print(results)
    print('--------------------------------')
    print('Done testing...')



if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')

    # Dataset arguments
    parser.add_argument('--datapath', type=str, default='/home/kimsangki/breaking_bad/volume_constrained') 
    #'../../../../hdd/junhong/data/bbad_v2' and /mnt/nvme2n1p1/kimsangki_datasets/breaking_bad/volume_constrained , /home/kimsangki/breaking_bad/volume_constrained
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'synthetic'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--min_part', type=int, default=2)
    parser.add_argument('--max_part', type=int, default=2)
    parser.add_argument('--multiplicity', type=int, default=1, help='Multiplicity of the dataset')


    # Training arguments
    parser.add_argument('--logpath', type=str, default='default_logpath', help='Log path and name for project')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. DO NOT CHANGE THIS VALUE. WE ASSUME THAT BATCH SIZE IS 1')
    parser.add_argument('--n_worker', type=int, default=4, help='Number of workers. If you use multi-GPU training, the number of workers is multiplied by the number of GPUs.')
    parser.add_argument('--load', type=str, default='')


    # Debugging arguments
    parser.add_argument('--model', type=str, default='CMpp_equiassem', choices=['CM_equiassem', 'CMpp_equiassem'])
    parser.add_argument('--scale', type=str, default='overfitting', choices=['overfitting', 'tiny', 'small', 'full'])


    # This arguments are used only for CM_equiassem
    parser.add_argument('--backbone', type=str, default='vn_unet', choices=['vn_unet', 'vn_dgcnn', 'unet', 'dgcnn'])
    parser.add_argument('--shape_loss', type=str, default='positive', choices=['positive', 'negative'])
    parser.add_argument('--occ_loss', type=str, default='negative', choices=['positive', 'negative'])
    parser.add_argument('--no_ori', action='store_true')
    parser.add_argument('--attention', type=str, default='channel', choices=['channel', 'none'])

    
    # Developing temporarily used experiments arguments
    parser.add_argument('--additional_VNLinearLeakyReLU', action='store_true', help='If True, use VNLinearLeakyReLU layers for the equivariant shape feature')
    parser.add_argument('--debugged_circle_loss', action='store_true', help='If True, use Debugged version of Circle Loss')
    parser.add_argument('--debugged_point_matching_loss', action='store_true', help='If True, use Debugged version of Point Matching Loss')
    parser.add_argument('--exp_scale_for_point_matching_loss', action='store_true', help='If True, make the matching score to exp-scaled value before computing point matching loss')
    parser.add_argument('--n_knn', type=int, default=20, help='Number of nearest neighbors for KNN')
    parser.add_argument('--new_orientation_module', action='store_true', help='If True, use New module for orientation')
    parser.add_argument('--delete_occupancy_loss', action='store_true', help='If True, delete the Occupancy Loss')
    parser.add_argument('--use_opt_gram', action='store_true', help='If True, use Optimum Gram Schmidt Orthogonalization')

    parser.add_argument('--only_one_norm', action='store_true', help='If True, use only one Normalization layer for the equivariant shape feature')
    parser.add_argument('--n_avn', type=int, default=5, help='Number of AVN layers for the equivariant shape feature')
    parser.add_argument('--move_smaller', action='store_true', help='If True, always move the smaller point cloud to the origin')


    # Inference arguments
    parser.add_argument('--infer_match_option', type=str, default='topk', choices=['topk', 'mutual_topk', 'soft_topk', 'unidirectional_nn_matching', 'injective_matching', 'bijective_matching'])
    parser.add_argument('--infer_topk', type=int, default=128)
    parser.add_argument('--infer_score_threshold_ratio', type=float, default=0.0)
    parser.add_argument('--use_RANSAC', action='store_true', help='If True, use RANSAC for transformation estimation')
    parser.add_argument('--RANSAC_type', type=str, default='default', choices=['default', 'score_dependent'])
    parser.add_argument('--use_predicted_normal', action='store_true', help='If True, use predicted normal for inlier counting')


    # Additional experiments
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--viz_epoch', type=int, default=30, help='Epoch for visualization. This only works when visualize is True')
    parser.add_argument('--viz_max_arrow_num', type=int, default=0, help='Maximum number of arrows for visualization. This only works when visualize is True')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--success_criterion_in_degree', type=int, default=10, help='Success criterion in degree for normal error')
    parser.add_argument('--delete_Sinkhorn', action='store_true', help='If True, delte the optimal transport (Sinkhorn).')
    parser.add_argument('--use_Sinkhorn_infer', action='store_true', help='Use Sinkhorn for inference, hence just before registration. So automatically set delete_Sinkhorn to True.')
    parser.add_argument('--matching_score_mode', type=str, default='CM', choices=['CM', 'cos'])
    parser.add_argument('--svd_no_exp', action='store_true', help='If True, do not use exp for SVD')
    parser.add_argument('--flip_normal', action='store_true', help='If True, flip predicted normal')


    # DDP argument
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)


    # Wandb argument
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='default_wandb_project')


    # Deterministic argument
    parser.add_argument('--deterministic', action='store_true')


    args = parser.parse_args()


    # Set number of workers automatically
    if len(args.gpus) > 1: # Multi-GPU training
        from pytorch_lightning.strategies import DDPStrategy
        args.parallel_strategy = DDPStrategy(find_unused_parameters=False)
    
    else: # Single-GPU training
        args.parallel_strategy = 'auto'


    # Setting developing experiments arguments automatically
    if args.model == 'CMpp_equiassem': # If the model is CMpp_equiassem
        arg_order = [
            "additional_VNLinearLeakyReLU",
            "debugged_circle_loss",
            "debugged_point_matching_loss",
            "new_orientation_module", # Use delete_occupancy_loss for automatically setting this to True
            "delete_occupancy_loss",
            "use_opt_gram",
        ]

        for i, name in enumerate(arg_order):
            if getattr(args, name):
                for prev_name in arg_order[:i]:
                    setattr(args, prev_name, True)
        
        if args.delete_occupancy_loss:
            # Now, we will test the model with normal vector method
            args.attention = 'none'
    

    # Assertions
    assert args.batch_size == 1, "Batch size must be 1"

    if args.use_Sinkhorn_infer:
        # We will remove Sinkhorn from training.
        # Only use Sinkhorn for inference.
        args.delete_Sinkhorn = True

    
    print("================================================")
    print(f"args: {args}")
    print("================================================")


    test(args)
