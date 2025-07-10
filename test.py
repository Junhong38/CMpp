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

from data.dataset_pa import GADataset
from common import utils
import open3d as o3d

from model.equiassem import EquiAssem
from model.equiassem_shape import EquiAssem_shape
from model.equiassem_occ import EquiAssem_occ

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

@torch.no_grad()
def test(args):
    
    # Model initialization
    if args.model == 'both':
        model = EquiAssem(lr=args.lr,
                        backbone=args.backbone,
                        shape_loss=args.shape_loss, 
                        occ_loss=args.occ_loss, 
                        no_ori=args.no_ori,
                        attention=args.attention,
                        visualize=args.visualize,
                        debug=args.debug)
    elif args.model == 'shape_only':
        model = EquiAssem_shape(lr=args.lr,
                        backbone=args.backbone,
                        shape_loss=args.shape_loss, 
                        no_ori=args.no_ori,
                        visualize=args.visualize,
                        debug=args.debug)
    elif args.model == 'occ_only':
        model = EquiAssem_occ(lr=args.lr,
                        backbone=args.backbone,
                        occ_loss=args.occ_loss, 
                        no_ori=args.no_ori,
                        visualize=args.visualize,
                        debug=args.debug)

    model.to(torch.device('cuda:0'))
    model.eval()
    
    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.min_part, args.max_part, args.n_pts, args.scale)
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val')

    trainer = pl.Trainer(accelerator='gpu',
                        devices=[0])
    trainer.test(model, dataloader_val, ckpt_path=args.load)
    results = model.test_results
    results = {k[5:]: v.detach().cpu().numpy() for k, v in results.items()}
    print('Done testing...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'synthetic', 'fantastic'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--min_part', type=int, default=2)
    parser.add_argument('--max_part', type=int, default=2)

    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--load', type=str, default='')

    parser.add_argument('--scale', type=str, default='full', choices=['full', 'small', 'overfitting'])

    # Ablation studies
    parser.add_argument('--model', type=str, default='both', choices=['both', 'shape_only', 'occ_only'])
    parser.add_argument('--backbone', type=str, default='vn_unet', choices=['vn_unet', 'vn_dgcnn', 'unet', 'dgcnn'])
    parser.add_argument('--shape_loss', type=str, default='positive', choices=['positive', 'negative'])
    parser.add_argument('--occ_loss', type=str, default='negative', choices=['positive', 'negative'])
    parser.add_argument('--no_ori', action='store_false')
    parser.add_argument('--attention', type=str, default='channel', choices=['channel', 'none'])

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()

    test(args)