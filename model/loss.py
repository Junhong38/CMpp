import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class CircleLoss(nn.Module):

    def __init__(self, log_scale=24, pos_optimal=0.1, neg_optimal=1.4):
        super(CircleLoss,self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = pos_optimal - 0.05
        self.neg_margin = neg_optimal + 0.05
        
        self.pos_radius = 0.018
        self.safe_radius = 0.03

        # self.max_points = 128

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        Trivially modified from GeoTransformer Implementation

        Args:
            coords_dist (torch.Tensor): (N, M)
            feats_dist (torch.Tensor): (N, M)

        Returns:
            torch.Tensor: (1, ), circle loss
            dict: (1, ), pos_neg_distribution
        """
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        # Calculate Positive/Negative feats_dist distribution
        with torch.no_grad():
            pos_dists = feats_dist[pos_mask]
            neg_dists = feats_dist[neg_mask]

            pos_neg_distribution = {
                'pos_mean': pos_dists.mean().item(),
                'pos_std': pos_dists.std().item(),
                'pos_min': pos_dists.min().item(),
                'pos_max': pos_dists.max().item(),
                'neg_mean': neg_dists.mean().item(),
                'neg_std': neg_dists.std().item(),
                'neg_min': neg_dists.min().item(),
                'neg_max': neg_dists.max().item(),
            }
        
        # sample the neg_mask to match proportions
        neg_indices = neg_mask.nonzero(as_tuple=False)
        neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[pos_mask.sum():]]
        neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1]] = False

        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)) # (N,M) -> (N, )
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)) # (N,M) -> (M, )

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight) # (N,M)

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight) # (N,M)

        # log(Σ exp(γ * (d - m_pos) * w_pos))
        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-1) # (N, )
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-2) # (M, )

        # log(Σ exp(γ * (m_neg - d) * w_neg))
        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-1) # (N, )
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-2) # (M, )

        # Softplus = log(1+exp(x))
        # So, log(1+exp(x)) / log_scale -> log(1 + Σ exp(γ * (d - m_pos) * w_pos) + Σ exp(γ * (m_neg - d) * w_neg)) / log_scale
        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale # (N, )
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale # (M, )

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss, pos_neg_distribution


    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence):
        """
        Args:
            src_pcd (torch.Tensor): (N, 3)
            tgt_pcd (torch.Tensor): (M, 3)
            src_feats (torch.Tensor): (1, N, D)
            tgt_feats (torch.Tensor): (1, M, D)
            correspondence (torch.Tensor): (P, 2)

        Returns:
            torch.Tensor: (1, ), circle loss
            dict: (1, ), pos_neg_distribution
        """

        if len(correspondence) == 0:
            print('[circle loss] No correspondence!')
            return torch.tensor(0.).to(src_feats.device), None

        # Get coordinate distance
        coords_dist = torch.sqrt(torch.clamp(torch.sum((src_pcd[:, None, :] - tgt_pcd[None, :, :]) ** 2, dim=-1), min=0.0))


        # Get feature distance (from GeoTransformer Implementation)
        src_feats = F.normalize(src_feats.squeeze(0), p=2, dim=-1) # (1, N, D) -> (N, D)
        tgt_feats = F.normalize(tgt_feats.squeeze(0), p=2, dim=-1) # (1, M, D) -> (M, D)


        # Handle NaN
        if torch.isnan(src_feats).any() or torch.isnan(tgt_feats).any():
            print("NaN detected in features!")
            src_feats = torch.nan_to_num(src_feats)
            tgt_feats = torch.nan_to_num(tgt_feats)
        

        # Get feature distance
        dot = torch.einsum('x d, y d -> x y', src_feats, tgt_feats)
        dot = torch.clamp(dot, min=-1.0, max=1.0)
        value = 2.0 - 2.0 * dot
        assert (value >= 0).all(), f"Negative value detected in sqrt input: min={value.min()}"
        # (|x| - |y|)^2 = |x|^2 - 2<x, y> + |y|^2 where <x, y> = |x||y|cos(theta)
        # Also, we already normalized the features, so |x| = |y| = 1
        # so, |x|^2 - 2<x, y> + |y|^2 = 2 - 2<x, y> = 2 - 2cos(theta)
        # By, triangle formulat, 2 - 2 cos(theta) = 4 * sin(theta/2)^2
        # Hence, feats_dist = 2 * sin(theta/2)
        feats_dist = torch.sqrt(torch.clamp(value, min=0.0))
        

        # Calculate circle loss and feature matching recall (FMR)
        circle_loss, pos_neg_distribution = self.get_circle_loss(coords_dist, feats_dist)

        if torch.isnan(circle_loss):
            print('[circle loss] NaN detected! :', circle_loss)
            circle_loss = torch.tensor(0.).to(src_feats.device)
        
        return circle_loss, pos_neg_distribution


class PointMatchingLoss(nn.Module):
    def __init__(self):
        super(PointMatchingLoss, self).__init__()
        self.positive_radius = 0.018

    def forward(self, matching_scores, correlations, src_pcd, trg_pcd):
        """
        Args:
            matching_scores (torch.Tensor): (1, N, M)
            correlations (torch.Tensor): (P, 2)
            src_pcd (torch.Tensor): (N, 3)
            trg_pcd (torch.Tensor): (M, 3)

        Returns:
            torch.Tensor: (1, ), point matching loss
        """

        coords_dist = torch.sqrt(torch.sum((src_pcd[:, None, :] - trg_pcd[None, :, :]) ** 2, dim=-1))
        gt_corr_map = coords_dist < self.positive_radius

        # Initialize labels for the loss calculation
        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        
        # Handle slack rows and columns
        slack_row_labels = torch.sum(gt_corr_map, dim=1) == 0
        slack_col_labels = torch.sum(gt_corr_map, dim=0) == 0

        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels
        
        # Calculate the loss
        loss = - matching_scores[labels].mean()

        return loss


class OrientationLoss(nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='mean')

    
    def forward(self, src_ori, trg_ori, correspondence, gt_normals):
        """
        Args:
            src_ori (torch.Tensor): (1, N, 3, 3), first basis should be aligned with gt_normals[0]
            trg_ori (torch.Tensor): (1, M, 3, 3), first basis should be aligned with gt_normals[1]
            correspondence (torch.Tensor): (P, 2)
            gt_normals (list): length is 2, only for two pieces
                - gt_normals[0]: (1, N, 3)
                - gt_normals[1]: (1, M, 3)

        Returns:
            torch.Tensor: (1, ), orientation loss
        """
        if len(correspondence) == 0:
            return torch.tensor(0.).to(src_ori.device)

        src_normal_basis = src_ori[:, :, 0, :] # (1, N, 3)
        trg_normal_basis = trg_ori[:, :, 0, :] # (1, M, 3)

        src_normal_basis_loss = self.loss_fn(src_normal_basis, gt_normals[0])
        trg_normal_basis_loss = self.loss_fn(trg_normal_basis, gt_normals[1])

        final_loss = (src_normal_basis_loss + trg_normal_basis_loss) / 2

        return final_loss

        



