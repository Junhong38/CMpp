import os
import argparse

import torch

from data.dataset import GADataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from model.CMpp_equiassem import EquiAssem

@torch.no_grad()
def test(args):
    if not args.not_seed_fix:
        seed_everything(42, workers=True)


    # Create checkpoint directory
    cfg_name = args.logpath
    ckp_dir = os.path.join('checkpoint', cfg_name, 'models')
    os.makedirs(ckp_dir, exist_ok=True)

    # Print checkpoint directory
    print(f"checkpoint directory (ckp_dir): {ckp_dir}")


    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.scale, 1, args.min_part, args.max_part, args.min_n_pts, args.n_pts, args.overlap_radius, args.sampling_mode)
    dataloader_val = GADataset.build_dataloader(1, args.n_worker, 'val')


    # Model initialization
    model = EquiAssem(lr=0, # We don't need to use learning rate for testing
                      scheduler_mode=None, # We don't need to use this parameter for testing
                      backbone=args.backbone,
                      double_bacbone=args.double_bacbone,

                      # Circle loss and point matching loss arguments
                      pos_radius=args.pos_radius,
                      safe_radius=0.0,

                      # Circle loss arguments
                      pos_margin=0, # We don't need to use this parameter for testing  
                      neg_margin=0, # We don't need to use this parameter for testing  
                      pos_offset=0, # We don't need to use this parameter for testing
                      neg_offset=0, # We don't need to use this parameter for testing
                      log_scale=1, # We don't need to use this parameter for testing
                      balance_mode='none', # We don't need to use this parameter for testing
                      hard_negative='none', # We don't need to use this parameter for testing
                      neg_topk=0, # We don't need to use this parameter for testing
                      distance_type='cossim', # We don't need to use this parameter for testing
                      anchor_mode='default', # We don't need to use this parameter for testing
                      more_hard_neg=False, # We don't need to use this parameter for testing

                      s_loss_weight=0.0, # We don't need to use this parameter for testing
                      p_loss_weight=0.0, # We don't need to use this parameter for testing
                      o_loss_weight=0.0, # We don't need to use this parameter for testing

                      visualize_mode=args.visualize_mode,
                      viz_metric_name=args.viz_metric_name,
                      viz_metric_threshold=args.viz_metric_threshold,
                      viz_train_epoch=0, # We don't need to use this parameter for testing
                      viz_epoch=args.viz_epoch,
                      viz_max_arrow_num=args.viz_max_arrow_num,
                      ckp_dir=ckp_dir,
                      debug=args.debug, 
                      success_criterion_in_degree=args.success_criterion_in_degree,
                      only_train_normal=False, # We don't need to use this parameter for testing  
                      flip_normal_mode=args.flip_normal_mode,
                      consistency_loss=False, # We don't need to use this parameter for testing
                        
                      n_knn=args.n_knn,
                      only_one_norm=args.only_one_norm,
                      n_avn=args.n_avn,
                      mlp_mode=args.mlp_mode,
                      normal_pred_mode=args.normal_pred_mode,
                      move_smaller=args.move_smaller,

                      matching_score_mode=args.matching_score_mode,
                      matching_norm_mode=args.matching_norm_mode,
                      learnable_softmax_temperature=args.learnable_softmax_temperature,
                        
                      infer_match_option=args.infer_match_option,
                      infer_topk=args.infer_topk,
                      infer_score_threshold_ratio=args.infer_score_threshold_ratio,
                      use_RANSAC=args.use_RANSAC,
                      RANSAC_type=args.RANSAC_type,
                      use_predicted_normal=args.use_predicted_normal
                      )
    
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

    if model.trainer.global_rank == 0:
        results = model.test_results
        results = {k[4:]: v.detach().cpu().numpy() for k, v in results.items()}
        print('--------------------------------')
        print(results)
        print('--------------------------------')
        print('Done testing...')



if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')

    # Dataset arguments
    parser.add_argument('--datapath', type=str, default='/home/kimsangki/breaking_bad/volume_constrained/') 
    #'../../../../hdd/junhong/data/bbad_v2' and /mnt/nvme2n1p1/kimsangki_datasets/breaking_bad/volume_constrained , /home/kimsangki/breaking_bad/volume_constrained 
    # ../data/temp_breaking_bad/breaking_bad/volume_constrained , ../../../../../hdd/junhong/temp_data/breaking_bad/volume_constrained
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--scale', type=str, default='full', choices=['overfitting', 'tiny', 'small', 'full'])
    parser.add_argument('--min_part', type=int, default=2)
    parser.add_argument('--max_part', type=int, default=2)
    parser.add_argument('--min_n_pts', type=int, default=256)
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--overlap_radius', type=float, default=0.018)
    parser.add_argument('--sampling_mode', type=str, default='random', choices=['random', 'mesh', 'same_prev', 'same'], help='Sampling mode for point cloud sampling, mesh mode is only for testing')


    # Training arguments
    parser.add_argument('--logpath', type=str, default='default_logpath', help='Log path and name for project')
    parser.add_argument('--n_worker', type=int, default=4, help='Number of workers. If you use multi-GPU training, the number of workers is multiplied by the number of GPUs.')
    parser.add_argument('--load', type=str, default='')


    # Model arguments
    parser.add_argument('--model', type=str, default='CMpp_equiassem', choices=['CMpp_equiassem'])
    parser.add_argument('--backbone', type=str, default='vn_unet', choices=['vn_unet', 'vn_unet_v2'])
    parser.add_argument('--double_bacbone', type=str, default='none', choices=['none', 'vn_unet', 'vn_unet_v2'])
    parser.add_argument('--n_knn', type=int, default=20, help='Number of nearest neighbors for KNN')
    parser.add_argument('--only_one_norm', action='store_true', help='If True, use only one Normalization layer for the equivariant shape feature')
    parser.add_argument('--n_avn', type=int, default=5, help='Number of AVN layers for the equivariant shape feature')
    parser.add_argument('--mlp_mode', type=str, default='CMpp', choices=['CMpp', 'CMpp_half', 'half', 'deep'])
    parser.add_argument('--normal_pred_mode', type=str, default='cross', choices=['cross', 'gram'])


    # Only for evaluation metrics
    parser.add_argument('--pos_radius', type=float, default=0.018, help='Radius for positive samples in Circle loss computation and point matching loss')


    # Additional experiments
    parser.add_argument('--success_criterion_in_degree', type=int, default=10, help='Success criterion in degree for normal error')
    parser.add_argument('--flip_normal_mode', type=str, default='none', choices=['none', 'right', 'rightv2', 'mix'])
    parser.add_argument('--move_smaller', action='store_true', help='If True, always move the smaller point cloud to the origin')


    # Sinkhorn/Matching experments
    parser.add_argument('--matching_score_mode', type=str, default='CM', choices=['CM', 'cossim'])
    parser.add_argument('--matching_norm_mode', type=str, default='sinkhorn', choices=['sinkhorn', 'softmax', 'none'])
    parser.add_argument('--learnable_softmax_temperature', action='store_true', help='If True, use learnable temperature for softmax')

    
    # Inference arguments
    parser.add_argument('--infer_match_option', type=str, default='topk', choices=['topk', 'mutual_topk', 'soft_topk', 'unidirectional_nn_matching', 'injective_matching', 'bijective_matching'])
    parser.add_argument('--infer_topk', type=int, default=128)
    parser.add_argument('--infer_score_threshold_ratio', type=float, default=0.0)
    parser.add_argument('--use_RANSAC', action='store_true', help='If True, use RANSAC for transformation estimation')
    parser.add_argument('--RANSAC_type', type=str, default='default', choices=['default', 'score_dependent'])
    parser.add_argument('--use_predicted_normal', action='store_true', help='If True, use predicted normal for inlier counting')


    # Visualization arguments
    parser.add_argument('--visualize_mode', type=str, default='none', choices=['none', 'light', 'all'])
    parser.add_argument('--viz_metric_name', type=str, default='crd', choices=['none', 'crd', 'cd', 'rrmse_geo', 'trmse_geo'])
    parser.add_argument('--viz_metric_threshold', type=float, default=0.0)
    parser.add_argument('--viz_epoch', type=int, default=30, help='Epoch for visualization. This only works when visualize is True')
    parser.add_argument('--viz_max_arrow_num', type=int, default=0, help='Maximum number of arrows for visualization. This only works when visualize is True')
    parser.add_argument('--debug', action='store_true')

    # DDP argument
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)


    # Wandb argument
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='default_wandb_project')


    # Deterministic argument
    parser.add_argument('--not_seed_fix', action='store_true')
    parser.add_argument('--deterministic', action='store_true')


    args = parser.parse_args()


    # Set number of workers automatically
    if len(args.gpus) > 1: # Multi-GPU training
        from pytorch_lightning.strategies import DDPStrategy
        args.parallel_strategy = DDPStrategy(find_unused_parameters=False)
    
    else: # Single-GPU training
        args.parallel_strategy = 'auto'
    
    print("================================================")
    print(f"args: {args}")
    print("================================================")

    if args.learnable_softmax_temperature:
        assert args.matching_norm_mode == 'softmax', f"learnable_softmax_temperature is only allowed when matching_norm_mode is softmax, but got {args.matching_norm_mode}"


    test(args)
