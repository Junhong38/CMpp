import os
import pwd
import argparse
import torch

# Set matplotlib backend environment variable to avoid X server issues
# This ensures all processes (including worker processes) use the correct backend
os.environ['MPLBACKEND'] = 'Agg'

from data.dataset import GADataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything

from model.CMpp_equiassem import EquiAssem


def main(args):
    if not args.not_seed_fix:
        seed_everything(42, workers=True)

    # Create checkpoint directory
    cfg_name = args.logpath
    ckp_dir = os.path.join('checkpoint', cfg_name, 'models')
    os.makedirs(ckp_dir, exist_ok=True)

    # Print checkpoint directory
    print(f"checkpoint directory (ckp_dir): {ckp_dir}")

    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.scale, args.multiplicity, args.min_part, args.max_part, args.min_n_pts, args.n_pts, args.sampling_mode)
    dataloader_trn = GADataset.build_dataloader(args.batch_size, args.n_worker, 'train')
    dataloader_val = GADataset.build_dataloader(1, args.n_worker, 'val')

    # Model initialization        
    model = EquiAssem(lr=args.lr,
                      ori_backbone_lr_weight=args.ori_backbone_lr_weight,
                      scheduler_mode=args.scheduler_mode,
                      backbone=args.backbone,
                      double_bacbone=args.double_bacbone,
                      seg_head_mode=args.seg_head_mode,

                      # Circle loss and point matching loss arguments
                      pos_radius=args.pos_radius,
                      safe_radius=args.safe_radius,

                      # Circle loss arguments
                      pos_margin=args.pos_margin,
                      neg_margin=args.neg_margin,
                      pos_offset=args.pos_offset,
                      neg_offset=args.neg_offset,
                      log_scale=args.log_scale,
                      balance_mode=args.balance_mode,
                      hard_negative=args.hard_negative,
                      neg_topk=args.neg_topk,
                      distance_type=args.distance_type,
                      anchor_mode=args.anchor_mode,
                      more_hard_neg=args.more_hard_neg,
                      start_hard_neg_epoch=args.start_hard_neg_epoch,

                      s_loss_weight=args.s_loss_weight,
                      p_loss_weight=args.p_loss_weight,
                      o_loss_weight=args.o_loss_weight,
                      seg_loss_weight=args.seg_loss_weight,
                      seg_loss_mode=args.seg_loss_mode,

                      visualize_mode=args.visualize_mode,
                      viz_metric_name='none', # This is not used for training
                      viz_metric_threshold=0.0, # This is not used for training
                      viz_train_epoch=args.viz_train_epoch,
                      viz_epoch=args.viz_epoch,
                      viz_max_arrow_num=args.viz_max_arrow_num,
                      ckp_dir=ckp_dir,
                      debug=args.debug,
                      success_criterion_in_degree=args.success_criterion_in_degree,
                      only_train_normal=args.only_train_normal,
                      flip_normal_mode=args.flip_normal_mode,
                      consistency_loss_weight=args.consistency_loss_weight,

                      n_knn=args.n_knn,
                      only_one_norm=args.only_one_norm,
                      n_avn=args.n_avn,
                      mlp_mode=args.mlp_mode,
                      normal_pred_mode=args.normal_pred_mode,
                      move_smaller=args.move_smaller,

                      matching_score_mode=args.matching_score_mode,
                      matching_norm_mode=args.matching_norm_mode,
                      learnable_softmax_temperature=args.learnable_softmax_temperature,
                        
                      infer_match_option='topk', # Fix match option value during training
                      infer_topk=128, # Fix topk value during training
                      infer_score_threshold_ratio=0.0, # Block filtering correspondences during training
                      use_RANSAC=False, # RANSAC is not used for training
                      RANSAC_type='default', # RANSAC is not used for training
                      use_predicted_normal=False, # RANSAC is not used for training
                      use_seg_result=False # During training, we do not use the segmentation result
                      )

    
    
    # This code is for running on clusters
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    print(f"SLURM_JOB_ID: {SLURM_JOB_ID} | if None, it is not running on cluster")

    # On clusters, quota under user dir is usually limited
    # soft link to save the weights in temp space for checkpointing
    # TODO: modify this if you are not running on clusters
    CHECKPOINT_DIR = '/checkpoint/'
    if SLURM_JOB_ID and CHECKPOINT_DIR and os.path.isdir(CHECKPOINT_DIR):
        if not os.path.exists(ckp_dir):
            # on my cluster, the temp dir is /checkpoint/$USER/$SLURM_JOB_ID
            # TODO: modify this if your cluster is different
            usr = pwd.getpwuid(os.getuid())[0]
            os.system(r'ln -s /checkpoint/{}/{}/ {}'.format(usr, SLURM_JOB_ID, ckp_dir))
    

    # it's not good to hard-code the wandb id
    # but on preemption clusters, we want the job to resume the same wandb
    # process after resuming training (i.e. drawing the same graph)
    # so we have to keep the same wandb id
    # TODO: modify this if you are not running on preemption clusters
    preemption = True  # False
    if SLURM_JOB_ID and preemption:
        logger_id = logger_name = f'{cfg_name}-{SLURM_JOB_ID}'
    else:
        logger_name = cfg_name
        logger_id = None
    

    # configure callbacks
    checkpoint_callback_crd = ModelCheckpoint(dirpath=ckp_dir, filename='model-crd-{epoch:03d}', monitor='val/crd', save_top_k=1, mode='min')
    checkpoint_callback_cd = ModelCheckpoint(dirpath=ckp_dir, filename='model-cd-{epoch:03d}', monitor='val/cd', save_top_k=1, mode='min')
    checkpoint_callback_rrmse = ModelCheckpoint(dirpath=ckp_dir, filename='model-rrmse-{epoch:03d}', monitor='val/rrmse', save_top_k=1, mode='min')
    checkpoint_callback_trmse = ModelCheckpoint(dirpath=ckp_dir, filename='model-trmse-{epoch:03d}', monitor='val/trmse', save_top_k=1, mode='min')
    checkpoint_callback_Oloss = ModelCheckpoint(dirpath=ckp_dir, filename='model-Oloss-{epoch:03d}', monitor='val/o_loss', save_top_k=1, mode='min')
    latest_checkpoint_callback = ModelCheckpoint(dirpath=ckp_dir, filename='model-latest', save_last=True)
    
    
    if not args.only_train_normal: # Normal training
        callbacks = [
            LearningRateMonitor('epoch'),
            checkpoint_callback_crd,
            checkpoint_callback_cd,
            checkpoint_callback_rrmse,
            checkpoint_callback_trmse,
            checkpoint_callback_Oloss,
            latest_checkpoint_callback,
        ]
    else: # Only train the normal vector
        callbacks = [
            LearningRateMonitor('epoch'),
            checkpoint_callback_Oloss,
            latest_checkpoint_callback,
        ]


    # Wandb logger
    if args.wandb:
        logger = WandbLogger(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=logger_name, # same as logpath
            id=logger_id, # same as SLURM_JOB_ID
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


    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=all_gpus,
        precision=32,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm='value',
        deterministic=args.deterministic,
        strategy=args.parallel_strategy,
        max_epochs=args.epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        # detect_anomaly=True, # If you want to NaN check, uncomment this
        # profiler='simple',
        fast_dev_run=False,
    )

    # automatically detect existing checkpoints in case of preemption
    ckp_files = os.listdir(ckp_dir)
    ckp_files = [ckp for ckp in ckp_files if 'model-' in ckp]
    if ckp_files:  # note that this will overwrite `args.weight`
        ckp_files = sorted(ckp_files, key=lambda x: os.path.getmtime(os.path.join(ckp_dir, x))) # sort by modified time
        last_ckp = ckp_files[-1]
        print(f'INFO: automatically detect checkpoint {last_ckp}')
        ckp_path = os.path.join(ckp_dir, last_ckp)
    
    elif args.load != '': # Load checkpoint
        ckp = torch.load(args.load, map_location='cpu')
        print(f"Loading checkpoint from {ckp.keys()}")
        ckp_path = None
        model.load_state_dict(ckp['state_dict'])
    
    elif args.load_ori != '': # Load orientation backbone network
        ckp = torch.load(args.load_ori, map_location='cpu')
        print(f"Loading orientation backbone network from {ckp.keys()}")
        ckp_path = None
        
        loaded_weights = {}
        for key in ckp['state_dict'].keys():
            if key.startswith('ori_backbone.') or key.startswith('proj.'):
                loaded_weights[key] = ckp['state_dict'][key]
        
        load_result = model.load_state_dict(loaded_weights, strict=False)
        print(f"Missing keys: {load_result[0]}")
        print(f"Unexpected keys: {load_result[1]}")

        if args.freeze_ori_2nd_stage:
            print(f"Freezing orientation backbone network")
            model.freeze_ori_backbone()
        else:
            print(f"Not freezing orientation backbone network")
    
    elif args.resume != '': # Resume training from the checkpoint
        ckp_path = args.resume
    
    else: # No checkpoint
        ckp_path = None
    

    print(f"ckp_path: {ckp_path}")
    trainer.fit(model, dataloader_trn, dataloader_val, ckpt_path=ckp_path)
    print('Done training...')


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
    parser.add_argument('--multiplicity', type=int, default=1, help='Multiplicity of the dataset')
    parser.add_argument('--min_part', type=int, default=2)
    parser.add_argument('--max_part', type=int, default=2)
    parser.add_argument('--min_n_pts', type=int, default=256)
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--sampling_mode', type=str, default='random', choices=['random'], help='Sampling mode for point cloud sampling')
    

    # Training arguments
    parser.add_argument('--logpath', type=str, default='default_logpath', help='Log path and name for project')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate. If you use multi-GPU training, the learning rate is multiplied by the number of GPUs.')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs. If 0, it is automatically set to 90 for everyday dataset and 300 for other datasets.')
    parser.add_argument('--n_worker', type=int, default=4, help='Number of workers. If you use multi-GPU training, the number of workers is multiplied by the number of GPUs.')
    parser.add_argument('--load', type=str, default='', help='Load checkpoint for training')
    parser.add_argument('--load_ori', type=str, default='', help='Only load the orientation backbone network, this is only allowed when double_backbone, and stage 2')
    parser.add_argument('--freeze_ori_2nd_stage', action='store_true')
    parser.add_argument('--resume', type=str, default='', help='Resume training from the checkpoint')
    parser.add_argument('--scheduler_mode', type=str, default='cos', choices=['none', 'cos', 'onecycle'])
    parser.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clip value')
    parser.add_argument('--ori_backbone_lr_weight', type=float, default=1.0, help='Gradient clip value')


    # Model arguments
    parser.add_argument('--backbone', type=str, default='vn_unet', choices=['vn_unet', 'vn_unet_v2'])
    parser.add_argument('--double_bacbone', type=str, default='vn_unet', choices=['none', 'vn_unet', 'vn_unet_v2'])
    parser.add_argument('--n_knn', type=int, default=20, help='Number of nearest neighbors for KNN')
    parser.add_argument('--only_one_norm', action='store_true', help='If True, use only one Normalization layer for the equivariant shape feature')
    parser.add_argument('--n_avn', type=int, default=0, help='Number of AVN layers for the equivariant shape feature')
    parser.add_argument('--mlp_mode', type=str, default='half', choices=['CMpp', 'CMpp_half', 'half', 'deep'])
    parser.add_argument('--normal_pred_mode', type=str, default='cross', choices=['cross', 'gram'])
    parser.add_argument('--seg_head_mode', type=str, default='none', choices=['none', 'mlp', 'atten'])
    

    # Weights for losses
    parser.add_argument('--s_loss_weight', type=float, default=1.0, help='Weight for shape loss')
    parser.add_argument('--p_loss_weight', type=float, default=1.0, help='Weight for point loss')
    parser.add_argument('--o_loss_weight', type=float, default=1.0, help='Weight for orientation loss')
    parser.add_argument('--seg_loss_weight', type=float, default=0.1, help='Weight for segmentation loss')
    parser.add_argument('--seg_loss_mode', type=str, default='bce', choices=['dice', 'bce'])


    # Margin arguments which are used in circle loss
    parser.add_argument('--pos_radius', type=float, default=0.018, help='Radius for positive samples in Circle loss computation and point matching loss')
    parser.add_argument('--safe_radius', type=float, default=0.03, help='Radius for safe samples in Circle loss computation')
    parser.add_argument('--pos_margin', type=float, default=0.075, help='Margin for positive samples in Circle loss computation')
    parser.add_argument('--neg_margin', type=float, default=1.45, help='Margin for negative samples in Circle loss computation')
    parser.add_argument('--pos_offset', type=float, default=0.0, help='Offset for positive samples in Circle loss computation')
    parser.add_argument('--neg_offset', type=float, default=0.0, help='Offset for negative samples in Circle loss computation')
    parser.add_argument('--log_scale', type=float, default=24, help='Log scale for Circle loss computation')
    parser.add_argument('--balance_mode', type=str, default='double', choices=['none', 'half', 'all_hard', 'double'])
    parser.add_argument('--hard_negative', type=str, default='mix', choices=['none', 'mix'])
    parser.add_argument('--neg_topk', type=int, default=0, help='')
    parser.add_argument('--distance_type', type=str, default='cossim', choices=['l2', 'cossim'])
    parser.add_argument('--anchor_mode', type=str, default='default', choices=['default', 'all_pos', 'all'])
    parser.add_argument('--more_hard_neg', action='store_true', help='')
    parser.add_argument('--start_hard_neg_epoch', type=int, default=-1, help='Start hard negative sampling from this epoch')


    # Additional experiments
    parser.add_argument('--success_criterion_in_degree', type=int, default=10, help='Success criterion in degree for normal error')
    parser.add_argument('--only_train_normal', action='store_true', help='Only train the normal vector, it will be used for stage 1 training')
    parser.add_argument('--flip_normal_mode', type=str, default='none', choices=['none', 'right', 'rightv1_2', 'rightv1_3', 'rightv2', 'rightv3', 'rightv4', 'rightv5', 'mix'])
    parser.add_argument('--consistency_loss_weight', type=float, default=0.0, help='Weight for consistency loss')
    parser.add_argument('--move_smaller', action='store_true', help='If True, always move the smaller point cloud to the origin')


    # Sinkhorn/Matching experments
    parser.add_argument('--matching_score_mode', type=str, default='CM', choices=['CM', 'cossim'])
    parser.add_argument('--matching_norm_mode', type=str, default='softmax', choices=['sinkhorn', 'softmax', 'none'])
    parser.add_argument('--learnable_softmax_temperature', action='store_true', help='If True, use learnable temperature for softmax')


    # Visualization arguments
    parser.add_argument('--visualize_mode', type=str, default='none', choices=['none', 'light', 'all'])
    parser.add_argument('--viz_train_epoch', type=int, default=0, help='This is for visualizing the negative hard mask during training')
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

    if args.epochs <= 0: # If epochs is not set, set number of epochs automatically
        args.epochs = 90 if args.data_category == 'everyday' else 120
        args.epochs = 300 if args.max_part > 2 else args.epochs
    else:
        args.epochs = args.epochs # If epochs is set, use the set value

    # Set number of workers automatically
    if len(args.gpus) > 1: # Multi-GPU training
        from pytorch_lightning.strategies import DDPStrategy
        args.parallel_strategy = DDPStrategy(find_unused_parameters=args.only_train_normal)
        args.lr = len(args.gpus) * args.lr # Learning rate is multiplied by the number of GPUs
    
    else: # Single-GPU training
        args.parallel_strategy = 'auto'

    # If gradient clip value is 0.0, set it to None
    if args.gradient_clip_val <= 0.0:
        args.gradient_clip_val = None


    
    print("================================================")
    print(f"args: {args}")
    print("================================================")


    assert not ((args.load != '') and (args.resume != '')), "Load and resume cannot be used together"
    assert not ((args.load_ori != '') and (args.load != '')), "load_ori and load cannot be used together"
    
    if args.load_ori != '':
        assert args.double_bacbone != 'none', "load_ori is only allowed when double_bacbone is not none"
    
    if args.learnable_softmax_temperature:
        assert args.matching_norm_mode == 'softmax', f"learnable_softmax_temperature is only allowed when matching_norm_mode is softmax, but got {args.matching_norm_mode}"
    
    
    main(args)