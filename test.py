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

@torch.no_grad()
def test(args):
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
        model = EquiAssem_v4(lr=args.lr, visualize=args.visualize)
    print(model)
    model.to(torch.device('cuda:0'))
    model.eval()
    
    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.n_pts, args.scale)
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val')

    trainer = pl.Trainer(accelerator='gpu',
                        devices=[0])
    trainer.test(model, dataloader_val, ckpt_path=args.load)
    results = model.test_results
    results = {k[5:]: v.detach().cpu().numpy() for k, v in results.items()}
    print('Done testing...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--backbone', type=str, default='eqcnn', choices=['eqcnn', 'dgcnn'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--scale', type=str, default='small')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()
    test(args)