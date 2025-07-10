import torch
import torch.nn as nn
import torchmetrics
import torch_scatter
import torch.nn.functional as F
import lightning as pl
from functools import partial

from .loss import dice_loss


class FracSeg2Pcs(pl.LightningModule):
    """
    LightningModule for mating surface segmentation on two-part assemblies.
    """

    def __init__(
        self,
        pc_feat_dim: int,
        encoder: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        seg_warmup_epochs: int = 10,
        grid_size: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.pc_feat_dim = pc_feat_dim
        self.encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.seg_warmup_epochs = seg_warmup_epochs
        self.grid_size = grid_size

        # BatchNorm and segmentation MLP head
        self.batch_norm = nn.BatchNorm1d(self.pc_feat_dim)
        self.coarse_segmenter = nn.Sequential(
            nn.Linear(self.pc_feat_dim, 16, 1),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1, 1),
            nn.Flatten(0, 1),
        )

    def criteria(self, input_dict, output_dict):
        # Get batch size and points per part information
        B = len(input_dict["points_per_part"])
        pts1 = input_dict["points_per_part"][:,
                                             0].tolist()  # list of n1 per sample
        pts2 = input_dict["points_per_part"][:,
                                             1].tolist()  # list of n2 per sample

        # Original dice loss on all points
        coarse_seg_loss = dice_loss(
            output_dict["coarse_seg_pred"],
            output_dict["coarse_seg_gt"].float(),
        )

        # Calculate part-specific metrics
        part1_preds = []
        part2_preds = []
        part1_gts = []
        part2_gts = []

        # Separate predictions and ground truths for part1 and part2
        start_idx = 0
        for i in range(B):
            end_idx1 = start_idx + pts1[i]
            end_idx2 = end_idx1 + pts2[i]

            part1_preds.append(
                output_dict["coarse_seg_pred"][start_idx:end_idx1])
            part1_gts.append(output_dict["coarse_seg_gt"][start_idx:end_idx1])

            part2_preds.append(
                output_dict["coarse_seg_pred"][end_idx1:end_idx2])
            part2_gts.append(output_dict["coarse_seg_gt"][end_idx1:end_idx2])

            start_idx = end_idx2

        # Concatenate all part1 and part2 points
        part1_pred_all = torch.cat(part1_preds)
        part1_gt_all = torch.cat(part1_gts)
        part2_pred_all = torch.cat(part2_preds)
        part2_gt_all = torch.cat(part2_gts)

        # Binary predictions
        part1_pred_binary = part1_pred_all > 0.5
        part2_pred_binary = part2_pred_all > 0.5

        # Calculate part-specific metrics
        part1_acc = torchmetrics.functional.accuracy(
            part1_pred_binary, part1_gt_all, task="binary"
        )
        part2_acc = torchmetrics.functional.accuracy(
            part2_pred_binary, part2_gt_all, task="binary"
        )

        # Calculate part-specific F1 scores
        part1_f1 = torchmetrics.functional.f1_score(
            part1_pred_binary, part1_gt_all, task="binary"
        )
        part2_f1 = torchmetrics.functional.f1_score(
            part2_pred_binary, part2_gt_all, task="binary"
        )

        # Compute overall metrics (same as before)
        coarse_seg_acc = torchmetrics.functional.accuracy(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_recall = torchmetrics.functional.recall(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_precision = torchmetrics.functional.precision(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_f1 = torchmetrics.functional.f1_score(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )

        # Calculate part accuracy (average of part1 and part2)
        part_acc = (part1_acc + part2_acc) / 2

        # Use original loss as total loss
        loss = coarse_seg_loss

        return loss, {
            "coarse_seg_loss": coarse_seg_loss,
            "coarse_seg_acc": coarse_seg_acc,
            "coarse_seg_recall": coarse_seg_recall,
            "coarse_seg_precision": coarse_seg_precision,
            "coarse_seg_f1": coarse_seg_f1,
            "part1_acc": part1_acc,
            "part2_acc": part2_acc,
            "part1_f1": part1_f1,
            "part2_f1": part2_f1,
            # This is what the model checkpoint is looking for (after prefix)
            "part_acc": part_acc,
        }

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, metrics = self.criteria(batch, out)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, metrics = self.criteria(batch, out)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Log the eval/part_acc metric that the checkpoint is monitoring
        self.log(
            "eval/part_acc",
            metrics["part_acc"],
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, metrics = self.criteria(batch, out)
        self.log(
            "test/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Log the eval/part_acc metric for testing too
        self.log(
            "eval/part_acc",
            metrics["part_acc"],
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def forward(self, batch):
        # Flatten part data across the batch to process in parallel
        pointclouds = batch["pointclouds"]               # [B, N_total, 3]
        normals = batch["pointclouds_normals"]           # [B, N_total, 3]
        points_per_part = batch["points_per_part"]       # [B, 2]
        fracture_surface = batch["fracture_surface_gt"]  # [B, N_total]
        B, N_total, _ = pointclouds.shape

        # get counts for each fragment across batch
        pts1 = points_per_part[:, 0].tolist()  # list of n1 per sample
        pts2 = points_per_part[:, 1].tolist()  # list of n2 per sample

        # flatten all part1 and part2 coords, normals, and ground-truth masks
        coords1 = torch.cat([pointclouds[i, :pts1[i]]
                            for i in range(B)], dim=0)
        coords2 = torch.cat(
            [pointclouds[i, pts1[i]: pts1[i] + pts2[i]] for i in range(B)], dim=0)
        nm1 = torch.cat([normals[i, :pts1[i]] for i in range(B)], dim=0)
        nm2 = torch.cat([normals[i, pts1[i]: pts1[i] + pts2[i]]
                        for i in range(B)], dim=0)
        gt1 = torch.cat([fracture_surface[i, :pts1[i]].view(-1)
                        for i in range(B)], dim=0)
        gt2 = torch.cat([fracture_surface[i, pts1[i]: pts1[i] +
                        pts2[i]].view(-1) for i in range(B)], dim=0)
        gt_all = torch.cat([gt1, gt2], dim=0)

        # build cumulative offsets so Point.batch maps points to sample indices
        device = coords1.device
        offsets1 = torch.cumsum(torch.tensor(
            pts1, dtype=torch.long, device=device), dim=0)
        offsets2 = torch.cumsum(torch.tensor(
            pts2, dtype=torch.long, device=device), dim=0)

        data1 = {
            "coord": coords1,
            "feat": torch.cat([coords1, nm1], dim=-1),
            "offset": offsets1,
            "grid_size": torch.tensor(self.grid_size, device=device),
        }
        data2 = {
            "coord": coords2,
            "feat": torch.cat([coords2, nm2], dim=-1),
            "offset": offsets2,
            "grid_size": torch.tensor(self.grid_size, device=device),
        }

        # encode all in parallel
        enc1, enc2 = self.encoder(data1, data2)
        feat1, feat2 = enc1.feat, enc2.feat

        # segmentation head on concatenated features
        pt_feats = torch.cat([feat1, feat2], dim=0)
        pt_feats = self.batch_norm(pt_feats)
        coarse_pred = torch.sigmoid(self.coarse_segmenter(pt_feats))
        coarse_pred_binary = coarse_pred > 0.5

        return {
            "coarse_seg_pred": coarse_pred,
            "coarse_seg_pred_binary": coarse_pred_binary,
            "coarse_seg_gt": gt_all,
        }

    def configure_optimizers(self):
        optim = self.optimizer(self.parameters())
        if self.lr_scheduler is None:
            return {"optimizer": optim}
        sched = self.lr_scheduler(optim)
        return {"optimizer": optim, "lr_scheduler": sched}
