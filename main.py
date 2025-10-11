import os
import sys
import pwd
import argparse
import importlib
import time
import gc
from distutils.dir_util import copy_tree

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.transform import Rotation

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import seed_everything

from data.dataset import GADataset
from common import utils
import open3d as o3d

from model.equiassem import EquiAssem


def main(args):
    seed_everything(42, workers=True)
    
    # Model initialization
    # [TODO] MODEL IS CHANGED
    model = EquiAssem(lr=args.lr,
                      backbone=args.backbone,
                      shape_loss=args.shape_loss, 
                      occ_loss=args.occ_loss, 
                      no_ori=args.no_ori,
                      attention=args.attention,
                      visualize=args.visualize,
                      debug=args.debug,
                      pos_margin=args.pos_margin, # [TODO] FOR REPRODUCIBILITY, WE SHOULD SET PROPER DEFUL VALUES
                      neg_margin=args.neg_margin, # [TODO] FOR REPRODUCIBILITY, WE SHOULD SET PROPER DEFAULT VALUES
                      log_scale=args.log_scale) # [TODO] FOR REPRODUCIBILITY, WE SHOULD SET PROPER DEFAULT VALUES


    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.min_part, args.max_part, args.n_pts, args.scale)
    dataloader_trn = GADataset.build_dataloader(args.batch_size, args.n_worker, 'train')
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val')


    # Create checkpoint directory
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    print(f"SLURM_JOB_ID: {SLURM_JOB_ID} | if None, it is not running on cluster")
    cfg_name = args.logpath
    ckp_dir = os.path.join('checkpoint/', cfg_name, 'models')
    os.makedirs(os.path.dirname(ckp_dir), exist_ok=True)


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
    else:
        os.makedirs(ckp_dir, exist_ok=True)
    
    
    # Print checkpoint directory
    print(f"checkpoint directory (ckp_dir): {ckp_dir}")
    

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
    
    callbacks = [
        LearningRateMonitor('epoch'),
        checkpoint_callback_crd,
        checkpoint_callback_cd,
        checkpoint_callback_rrmse,
        checkpoint_callback_trmse,
        checkpoint_callback_Oloss,
        latest_checkpoint_callback,
    ]


    # Wandb logger
    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=logger_name, # same as logpath
            id=logger_id, # same as SLURM_JOB_ID
            save_dir=ckp_dir,
            tags=[args.scale],
        )
    else: # [TODO] IF THERE IS NOT LOGGER, THEN ERROR WILL OCCUR
        logger = None


    all_gpus = list(args.gpus)
    print(f"all_gpus: {all_gpus}")


    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=all_gpus,
        precision=32,
        gradient_clip_val=None,
        strategy=args.parallel_strategy,
        max_epochs=args.epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
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
        # check if it has trainint states, or just a model weight
        ckp = torch.load(args.load, map_location='cpu')
        # if it has, then it's a checkpoint compatible with pl
        if 'state_dict' in ckp.keys():
            ckp_path = args.load
        # if it's just a weight, then manually load it to the model
        else:
            ckp_path = None
            model.load_state_dict(ckp)
    
    else: # No checkpoint
        ckp_path = None

    print(f"ckp_path: {ckp_path}")
    trainer.fit(model, dataloader_trn, dataloader_val, ckpt_path=ckp_path)
    print('Done training...')


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')

    # Dataset arguments
    parser.add_argument('--datapath', type=str, default='/mnt/nvme2n1p1/kimsangki_datasets/breaking_bad/volume_constrained')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'synthetic'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--min_part', type=int, default=2)
    parser.add_argument('--max_part', type=int, default=2)


    # Training arguments
    parser.add_argument('--logpath', type=str, default='default_logpath', help='Log path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. DO NOT CHANGE THIS VALUE. WE ASSUME THAT BATCH SIZE IS 1')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate. If you use multi-GPU training, the learning rate is multiplied by the number of GPUs.')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs. This is automatically set to 200 for everyday dataset and 300 for other datasets.')
    parser.add_argument('--n_worker', type=int, default=4, help='Number of workers. If you use multi-GPU training, the number of workers is multiplied by the number of GPUs.')
    parser.add_argument('--load', type=str, default='')


    # Debugging arguments
    parser.add_argument('--scale', type=str, default='full', choices=['full', 'small', 'overfitting', 'tiny'])


    # Ablation studies
    parser.add_argument('--backbone', type=str, default='vn_unet', choices=['vn_unet', 'vn_dgcnn', 'unet', 'dgcnn'])
    parser.add_argument('--shape_loss', type=str, default='positive', choices=['positive', 'negative'])
    parser.add_argument('--occ_loss', type=str, default='negative', choices=['positive', 'negative'])
    parser.add_argument('--no_ori', action='store_true')
    parser.add_argument('--attention', type=str, default='channel', choices=['channel', 'none'])


    # Margin arguments which are used in circle loss
    parser.add_argument('--pos_margin', type=float, default=0.1)
    parser.add_argument('--neg_margin', type=float, default=1.4)
    parser.add_argument('--log_scale', type=float, default=24)
    

    # Additional experiments
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')
        

    # DDP argument
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)


    # Wandb argument
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='default_wandb_project')


    args = parser.parse_args()


    # Set number of epochs automatically
    args.epochs = 90 if args.data_category == 'everyday' else 120
    args.epochs = 300 if args.max_part > 2 else args.epochs


    # Set number of workers automatically
    if len(args.gpus) > 1: # Multi-GPU training
        from pytorch_lightning.strategies import DDPStrategy
        args.parallel_strategy = DDPStrategy(find_unused_parameters=False)
        args.lr = len(args.gpus) * args.lr # Learning rate is multiplied by the number of GPUs
        args.n_worker = min(len(args.gpus) * 8, 48) # Number of workers is multiplied by the number of GPUs
    
    else: # Single-GPU training
        args.parallel_strategy = None


    # Assertions
    assert args.batch_size == 1, "Batch size must be 1"

    
    print("================================================")
    print(f"args: {args}")
    print("================================================")


    main(args)


