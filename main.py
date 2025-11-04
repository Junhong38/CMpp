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



def main(args):
    seed_everything(42, workers=True)


    # Create checkpoint directory
    cfg_name = args.logpath
    ckp_dir = os.path.join('checkpoint', cfg_name, 'models')
    os.makedirs(ckp_dir, exist_ok=True)

    # Print checkpoint directory
    print(f"checkpoint directory (ckp_dir): {ckp_dir}")


    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.min_part, args.max_part, args.n_pts, args.scale, args.multiplicity, CMorigin_mode=(args.model == 'CM_equiassem'))
    dataloader_trn = GADataset.build_dataloader(args.batch_size, args.n_worker, 'train')
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val')

    # Training total steps
    training_total_steps = len(dataloader_trn) * args.epochs
    print(f"training_total_steps: {training_total_steps}")


    # Model initialization
    # [TODO] MODEL IS CHANGED
    if args.model == 'CM_equiassem':
        from model.CM_equiassem import EquiAssem
        model = EquiAssem(lr=args.lr,
                          backbone=args.backbone,
                          shape_loss=args.shape_loss, 
                          occ_loss=args.occ_loss, 
                          no_ori=args.no_ori,
                          attention=args.attention,
                          visualize=args.visualize,
                          debug=args.debug)
        
    elif args.model == 'CMpp_equiassem': # Import developing mode model
        from model.CMpp_equiassem import EquiAssem
        model = EquiAssem(lr=args.lr,
                          
                          scheduler_mode=args.scheduler_mode,
                          training_total_steps=training_total_steps,

                          backbone=args.backbone,
                          attention=args.attention,

                          # Circle loss arguments
                          pos_margin=args.pos_margin,
                          neg_margin=args.neg_margin,
                          log_scale=args.log_scale,
                          detach_mode=args.detach_mode,
                          same_opt=args.same_opt,
                          only_corr=args.only_corr,
                          max_points=args.max_points,
                          no_balance=args.no_balance,
                          div_mode=args.div_mode,

                          s_loss_weight=args.s_loss_weight,
                          p_loss_weight=args.p_loss_weight,
                          o_loss_weight=args.o_loss_weight,

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
                          use_consistency_loss=args.use_consistency_loss,
                          only_train_normal=args.only_train_normal,
                          
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
                          
                          infer_match_option='topk', # Fix match option value during training
                          infer_topk=128, # Fix topk value during training
                          infer_score_threshold_ratio=0.0, # Block filtering correspondences during training
                          use_RANSAC=False, # RANSAC is not used for training
                          RANSAC_type='default', # RANSAC is not used for training
                          use_predicted_normal=False # RANSAC is not used for training
                          )
    else:
        raise NotImplementedError("Model not implemented")


    
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
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate. If you use multi-GPU training, the learning rate is multiplied by the number of GPUs.')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs. If 0, it is automatically set to 200 for everyday dataset and 300 for other datasets.')
    parser.add_argument('--n_worker', type=int, default=4, help='Number of workers. If you use multi-GPU training, the number of workers is multiplied by the number of GPUs.')
    parser.add_argument('--load', type=str, default='', help='Load checkpoint for training')
    parser.add_argument('--resume', type=str, default='', help='Resume training from the checkpoint')
    parser.add_argument('--scheduler_mode', type=str, default='cos', choices=['none', 'cos', 'onecycle', 'CM', 'CMpp'])
    parser.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clip value')


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

    # Weights for losses
    parser.add_argument('--s_loss_weight', type=float, default=0.5, help='Weight for shape loss, in the future, we will change this into 1.0')
    parser.add_argument('--p_loss_weight', type=float, default=1.0, help='Weight for point loss, in the future, we will change this into 1.0')
    parser.add_argument('--o_loss_weight', type=float, default=0.1, help='Weight for orientation loss, in the future, we will change this into 1.0')


    # Margin arguments which are used in circle loss
    parser.add_argument('--pos_margin', type=float, default=0.1, help='Margin for positive samples in Circle loss computation')
    parser.add_argument('--neg_margin', type=float, default=1.4, help='Margin for negative samples in Circle loss computation')
    parser.add_argument('--log_scale', type=float, default=24, help='Log scale for Circle loss computation')
    parser.add_argument('--detach_mode', action='store_true', help='Detach gradient from value for deciding strength of circle loss')
    parser.add_argument('--same_opt', action='store_true', help='Make margin value be same with optimal value')
    parser.add_argument('--only_corr', action='store_true', help='Use only correspondence for Circle loss computation')
    parser.add_argument('--max_points', type=int, default=0, help='Maximum number of points for Circle loss computation')
    parser.add_argument('--no_balance', action='store_true', help='Use positive and negative balance for Circle loss computation')
    parser.add_argument('--div_mode', type=str, default='none', choices=['none', 'dynamic', 'static'])


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
    parser.add_argument('--use_consistency_loss', action='store_true', help='Use consistency loss related to frame for training')
    parser.add_argument('--only_train_normal', action='store_true', help='Only train the normal vector, it will be used for stage 1 training')

        

    # DDP argument
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)


    # Wandb argument
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='default_wandb_project')


    # Deterministic argument
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
            args.s_loss_weight = 1.0
            args.p_loss_weight = 1.0
            args.o_loss_weight = 1.0
    
    
    # If gradient clip value is 0.0, set it to None
    if args.gradient_clip_val <= 0.0:
        args.gradient_clip_val = None
    

    if args.use_Sinkhorn_infer:
        # We will remove Sinkhorn from training.
        # Only use Sinkhorn for inference.
        args.delete_Sinkhorn = True


    # Assertions
    assert args.batch_size == 1, "Batch size must be 1"

    
    print("================================================")
    print(f"args: {args}")
    print("================================================")


    assert not ((args.load != '') and (args.resume != '')), "Load and resume cannot be used together"


    main(args)



"""
DEFAULT
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --wandb --wandb_project CMpp
rm -rf checkpoint/DTEST/ && CUDA_VISIBLE_DEVICES=2 python test.py --model CMpp_equiassem --logpath DTEST --scale small --multiplicity 1 --visualize --only_one_norm --use_opt_gram --use_RANSAC --load checkpoint/T3_S_AVNOON_NCD_NPM_NODOCC_OG/models/last.ckpt
"""
