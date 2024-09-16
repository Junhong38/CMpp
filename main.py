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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset import GADataset
from common import utils
import open3d as o3d

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

torch.set_float32_matmul_precision('medium')

def main(args):
    # Model initialization
    if args.model == 'v1': 
        from model.equiassem_v1 import EquiAssem_v1
        model = EquiAssem_v1(lr=args.lr)
    elif args.model == 'v2': 
        from model.equiassem_v2 import EquiAssem_v2
        model = EquiAssem_v2(lr=args.lr)
    elif args.model == 'v3': 
        from model.equiassem_v3 import EquiAssem_v3
        model = EquiAssem_v3(lr=args.lr)
    elif args.model == 'v4': 
        from model.equiassem_v4 import EquiAssem_v4
        model = EquiAssem_v4(lr=args.lr)
    elif args.model == 'v5': 
        from model.equiassem_v5 import EquiAssem_v5
        model = EquiAssem_v5(lr=args.lr)
    elif args.model == 'v6': 
        from model.equiassem_v6 import EquiAssem_v6
        model = EquiAssem_v6(lr=args.lr)
    elif args.model == 'v7': 
        from model.equiassem_v7 import EquiAssem_v7
        model = EquiAssem_v7(lr=args.lr)
    elif args.model == 'v8': 
        from model.equiassem_v8 import EquiAssem_v8
        model = EquiAssem_v8(lr=args.lr)
    print(model)

    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.n_pts, args.scale)
    dataloader_trn = GADataset.build_dataloader(args.batch_size, args.n_worker, 'train')
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val')

    # Create checkpoint directory
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    cfg_name = args.logpath
    ckp_dir = os.path.join('checkpoint/', cfg_name, 'models')
    os.makedirs(os.path.dirname(ckp_dir), exist_ok=True)

    # on clusters, quota under user dir is usually limited
    # soft link to save the weights in temp space for checkpointing
    # TODO: modify this if you are not running on clusters
    CHECKPOINT_DIR = '/checkpoint/'  # ''
    if SLURM_JOB_ID and CHECKPOINT_DIR and os.path.isdir(CHECKPOINT_DIR):
        if not os.path.exists(ckp_dir):
            # on my cluster, the temp dir is /checkpoint/$USER/$SLURM_JOB_ID
            # TODO: modify this if your cluster is different
            usr = pwd.getpwuid(os.getuid())[0]
            os.system(r'ln -s /checkpoint/{}/{}/ {}'.format(
                usr, SLURM_JOB_ID, ckp_dir))
    else:
        os.makedirs(ckp_dir, exist_ok=True)
    
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
    checkpoint_callback_crd = ModelCheckpoint(
        dirpath=ckp_dir,
        filename='model-crd-{epoch:03d}',
        monitor='val/crd',
        save_top_k=1,
        mode='min',
    )
    checkpoint_callback_cd = ModelCheckpoint(
        dirpath=ckp_dir,
        filename='model-cd-{epoch:03d}',
        monitor='val/cd',
        save_top_k=1,
        mode='min',
    )
    checkpoint_callback_rrmse = ModelCheckpoint(
        dirpath=ckp_dir,
        filename='model-rrmse-{epoch:03d}',
        monitor='val/rrmse',
        save_top_k=1,
        mode='min',
    )
    checkpoint_callback_trmse = ModelCheckpoint(
        dirpath=ckp_dir,
        filename='model-trmse-{epoch:03d}',
        monitor='val/trmse',
        save_top_k=1,
        mode='min',
    )

    callbacks = [
        LearningRateMonitor('epoch'),
        checkpoint_callback_crd,
        checkpoint_callback_cd,
        checkpoint_callback_rrmse,
        checkpoint_callback_trmse,
    ]

    logger = WandbLogger(
        project='equiassem_new',
        name=logger_name,
        id=logger_id,
        save_dir=ckp_dir,
    )

    all_gpus = list(args.gpus)

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
        profiler='simple',
        fast_dev_run=False,
    )

    # automatically detect existing checkpoints in case of preemption
    ckp_files = os.listdir(ckp_dir)
    ckp_files = [ckp for ckp in ckp_files if 'model-' in ckp]
    if ckp_files:  # note that this will overwrite `args.weight`
        ckp_files = sorted(
            ckp_files,
            key=lambda x: os.path.getmtime(os.path.join(ckp_dir, x)))
        last_ckp = ckp_files[-1]
        print(f'INFO: automatically detect checkpoint {last_ckp}')
        ckp_path = os.path.join(ckp_dir, last_ckp)
    elif args.load != '':
        # check if it has trainint states, or just a model weight
        ckp = torch.load(args.load, map_location='cpu')
        # if it has, then it's a checkpoint compatible with pl
        if 'state_dict' in ckp.keys():
            ckp_path = args.load
        # if it's just a weight, then manually load it to the model
        else:
            ckp_path = None
            model.load_state_dict(ckp)
    else:
        ckp_path = None

    trainer.fit(model, dataloader_trn, dataloader_val, ckpt_path=ckp_path)
    print('Done training...')


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--backbone', type=str, default='eqcnn', choices=['eqcnn', 'dgcnn'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--scale', type=str, default='small')

    # DDP argument
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)

    args = parser.parse_args()

    if len(args.gpus) > 1: 
        from pytorch_lightning.strategies import DDPStrategy
        args.parallel_strategy = DDPStrategy(find_unused_parameters=False)
        args.lr = len(args.gpus) * args.lr
        args.n_worker = len(args.gpus) * 4
    else: args.parallel_strategy = 'auto'

    main(args)