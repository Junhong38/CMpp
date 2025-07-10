import os
import torch
import torch.nn as nn
import torch.optim as optim
from common import utils
from model.equiassem import EquiAssem
import wandb

import argparse

from model.backbone.vn_layers import VNLinearLeakyReLU, VNLinear, VNInstanceNorm

from data.dataset import GADataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# # === 모델 정의 ===
class NormalEstimator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        utils.fix_randseed(0)
        self.pre_model = EquiAssem(lr=1e-2,
                            backbone="vn_unet",
                            occ_loss='negative', 
                            no_ori=False,
                            registration='wsvd',
                            match_selection_str='topk',
                            visualize=False,
                            debug=False,
                            inlier_threshold=0.01,
                            score_threshold=0,
                            auto_score_threshold=False,
                            ori_threshold=-1.0,
                            weighted_voting=False,
                            gt_normal_threshold=-1.0,
                            gt_mating_surface=False)

        self.pre_model.to(torch.device('cuda:0'))
        self.pre_model.eval()
        # Load checkpoint
        checkpoint = torch.load("./checkpoint/CM-v2-mpa-every-model-trmse-epoch=325.ckpt", map_location="cuda:0")
        self.pre_model.load_state_dict(checkpoint['state_dict'])

        self.model = nn.Sequential(
            VNLinearLeakyReLU(1023, 512, dim=3, negative_slope=0.2),
            VNInstanceNorm(512, dim=3),
            VNLinearLeakyReLU(512, 512, dim=3, negative_slope=0.2),
            VNInstanceNorm(512, dim=3),
            VNLinearLeakyReLU(512, 512, dim=3, negative_slope=0.2),
            VNInstanceNorm(512, dim=3),
            VNLinear(512, 3)  # Output: (B, 3, N, 3)
        )

        self.criterion = nn.MSELoss()

    def forward(self, A):
        return self.model(A)

    def training_step(self, in_dict, batch_idx):
        breakpoint()
        out_dict, _ = self.pre_model.forward_pass(
            in_dict, mode='test'
        )
        
        src_equi_feats = out_dict['src_equi_feats'] # (1, 341, 3, N)
        trg_equi_feats = out_dict['trg_equi_feats'] # (1, 341, 3, N)

        src_gt_normal = in_dict['gt_normals'][0]
        trg_gt_normal = in_dict['gt_normals']

        # A, gt = batch  # A: (B, 1023, N, 3), gt: (B, 3, N, 3)
        # pred = self(A)
        # loss = self.criterion(pred, gt)
        # self.log("train_loss", loss)
        # return loss

    def validation_step(self, batch, batch_idx):
        pass
        # A, gt = batch
        # pred = self(A)
        # loss = self.criterion(pred, gt)
        # self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    
def main(args):
    model = NormalEstimator()
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.min_part, args.max_part, args.n_pts, args.scale)
    dataloader_trn = GADataset.build_dataloader(args.batch_size, args.n_worker, 'train')
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val')

    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    cfg_name = args.logpath
    ckp_dir = os.path.join('normal_checkpoint/', cfg_name, 'models')
    os.makedirs(os.path.dirname(ckp_dir), exist_ok=True)

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

    preemption = True  # False
    if SLURM_JOB_ID and preemption:
        logger_id = logger_name = f'{cfg_name}-{SLURM_JOB_ID}'
    else:
        logger_name = cfg_name
        logger_id = None
    
    latest_checkpoint_callback = ModelCheckpoint(dirpath=ckp_dir, filename='model-latest', save_last=True)

    callbacks = [
        LearningRateMonitor('epoch'),
        latest_checkpoint_callback,
    ]

    logger = WandbLogger(
        project='Normal-Estimation',
        name=logger_name,
        id=logger_id,
        save_dir=ckp_dir,
        tags=[args.scale],
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

    trainer.fit(model, dataloader_trn, dataloader_val, ckpt_path=None)
    print('Done training...')

    breakpoint()

if __name__ == '__main__':
    wandb.init(mode="disabled")
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'synthetic'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--min_part', type=int, default=2)
    parser.add_argument('--max_part', type=int, default=2)

    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_worker', type=int, default=4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--scale', type=str, default='full', choices=['full', 'small', 'overfitting'])

    # Ablation studies
    parser.add_argument('--model', type=str, default='both', choices=['both', 'shape_only', 'occ_only'])
    parser.add_argument('--backbone', type=str, default='vn_unet', choices=['vn_unet', 'vn_dgcnn', 'unet', 'dgcnn'])
    parser.add_argument('--shape_loss', type=str, default='positive', choices=['positive', 'negative'])
    parser.add_argument('--occ_loss', type=str, default='negative', choices=['positive', 'negative'])
    parser.add_argument('--no_ori', action='store_true')
    parser.add_argument('--attention', type=str, default='channel', choices=['channel', 'none'])
    
    # Additional experiments
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')
        
    # DDP argument
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)

    args = parser.parse_args()

    args.epochs = 90 if args.data_category == 'everyday' else 120
    args.epochs = 300 if args.max_part > 2 else args.epochs
    
    if len(args.gpus) > 1: 
        from pytorch_lightning.strategies import DDPStrategy
        args.parallel_strategy = DDPStrategy(find_unused_parameters=False)
        args.lr = len(args.gpus) * args.lr
        args.n_worker = len(args.gpus) * 4
    else: args.parallel_strategy = None

    main(args)