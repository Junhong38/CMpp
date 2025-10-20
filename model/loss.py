import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class CircleLoss(nn.Module):

    def __init__(self, log_scale=24, pos_optimal=0.1, neg_optimal=1.4, 
                 detach_mode=False, same_opt=False, only_corr=False, max_points=0,
                 no_balance=False, div_mode='none'):


        super(CircleLoss,self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.detach_mode = detach_mode
        self.same_opt = same_opt
        self.only_corr = only_corr
        self.max_points = max_points
        self.no_balance = no_balance
        self.div_mode = div_mode

        if same_opt:
            self.pos_margin = pos_optimal
            self.neg_margin = neg_optimal
        else:
            self.pos_margin = pos_optimal - 0.05
            self.neg_margin = neg_optimal + 0.05

        self.pos_radius = 0.018
        self.safe_radius = 0.03

        print("------------------------------------------------------")
        print("INITIALIZING CircleLoss")
        print("------------------------------------------------------")
        print(f"log_scale: {log_scale}")
        print(f"pos_optimal: {pos_optimal}, pos_margin: {self.pos_margin}")
        print(f"neg_optimal: {neg_optimal}, neg_margin: {self.neg_margin}")
        print(f"detach_mode: {detach_mode}")
        print(f"same_opt: {same_opt}")
        print(f"only_corr: {only_corr}, max_points: {max_points}")
        print(f"no_balance: {no_balance}")
        print(f"div_mode: {div_mode}")
        print("------------------------------------------------------")

    
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

            does_pos_mask_exist = pos_mask.sum() > 0
            does_neg_mask_exist = neg_mask.sum() > 0

            pos_neg_distribution = {
                'pos_mean': pos_dists.mean().item() if does_pos_mask_exist else 0,
                'pos_std': pos_dists.std().item() if does_pos_mask_exist else 0,
                'pos_min': pos_dists.min().item() if does_pos_mask_exist else 0,
                'pos_max': pos_dists.max().item() if does_pos_mask_exist else 0,
                'neg_mean': neg_dists.mean().item() if does_neg_mask_exist else 0,
                'neg_std': neg_dists.std().item() if does_neg_mask_exist else 0,
                'neg_min': neg_dists.min().item() if does_neg_mask_exist else 0,
                'neg_max': neg_dists.max().item() if does_neg_mask_exist else 0,
            }
            
        if not self.no_balance:
            # sample the neg_mask to match proportions
            neg_indices = neg_mask.nonzero(as_tuple=False)
            neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[pos_mask.sum():]]
            neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1]] = False


        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach() # (N,M) -> (N, )
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach() # (N,M) -> (M, )

        if self.detach_mode:
            # print("Using detach mode")
            row_sel = row_sel.detach()
            col_sel = col_sel.detach()


        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight) # (N,M)

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight) # (N,M)

        if self.detach_mode:
            # print("Using detach mode")
            pos_weight = pos_weight.detach()
            neg_weight = neg_weight.detach()


        # log(Σ exp(γ * (d - m_pos) * w_pos))
        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-1) # (N, )
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-2) # (M, )

        # log(Σ exp(γ * (m_neg - d) * w_neg))
        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-1) # (N, )
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-2) # (M, )


        # Softplus = log(1+exp(x))
        # So, log(1+exp(x)) / log_scale -> log(1 + Σ exp(γ * (d - m_pos) * w_pos) + Σ exp(γ * (m_neg - d) * w_neg)) / log_scale
        loss_row = F.softplus(lse_pos_row + lse_neg_row) # (N, )
        loss_col = F.softplus(lse_pos_col + lse_neg_col) # (M, )

        if self.div_mode == 'dynamic':
            non_zero_pos_weight = (pos_weight > 0)
            non_zero_neg_weight = (neg_weight > 0)
            non_zero_total_weight = torch.logical_or(non_zero_pos_weight, non_zero_neg_weight) # (N, M)

            non_zero_total_row = non_zero_total_weight.sum(dim=-1) # N
            non_zero_total_col = non_zero_total_weight.sum(dim=-2) # M

            loss_row = loss_row / non_zero_total_row # N
            loss_col = loss_col / non_zero_total_col # N


        elif self.div_mode == 'static':
            loss_row = loss_row / loss_col.shape[0] # divide by M
            loss_col = loss_col / loss_row.shape[0] # divide by N
        
        else:
            loss_row = loss_row / self.log_scale
            loss_col = loss_col / self.log_scale

        # Prevent NaN
        anchor_loss_row = loss_row[row_sel].mean() if row_sel.sum() > 0 else torch.tensor(0.).to(loss_row.device)
        anchor_loss_col = loss_col[col_sel].mean() if col_sel.sum() > 0 else torch.tensor(0.).to(loss_col.device)

        
        if self.div_mode in ['dynamic', 'static']:
            circle_loss = (anchor_loss_row + anchor_loss_col)
        else:
            circle_loss = (anchor_loss_row + anchor_loss_col) / 2

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

        # Check NaN
        if torch.isnan(src_pcd).any() or torch.isnan(tgt_pcd).any() or torch.isnan(src_feats).any() or torch.isnan(tgt_feats).any():
            assert False, "[Circle Loss] Input features are nan\n src_pcd: {}\n tgt_pcd: {}\n src_feats: {}\n tgt_feats: {}".format(src_pcd, tgt_pcd, src_feats, tgt_feats)


        if len(correspondence) == 0:
            print('[circle loss] No correspondence!')
            return torch.tensor(0.).to(src_feats.device), None

        
        if self.only_corr:
            if self.max_points > 0:
                correspondence_selected = correspondence[torch.randperm(correspondence.size(0))[:self.max_points]]
            else:
                correspondence_selected = correspondence

            correspondence_mask = torch.zeros((src_pcd.size(0), tgt_pcd.size(0)), device=src_feats.device)
            correspondence_mask[correspondence_selected[:,0], correspondence_selected[:,1]] = True
            correspondence_mask_src = correspondence_mask.sum(dim=-1) > 0 # N
            correspondence_mask_tgt = correspondence_mask.sum(dim=-2) > 0 # M

            src_pcd_selected = src_pcd[correspondence_mask_src, :]
            tgt_pcd_selected = tgt_pcd[correspondence_mask_tgt, :]

            src_feats_selected = src_feats[:, correspondence_mask_src, :]
            tgt_feats_selected = tgt_feats[:, correspondence_mask_tgt, :]
        
        else:
            src_pcd_selected = src_pcd
            tgt_pcd_selected = tgt_pcd
            src_feats_selected = src_feats
            tgt_feats_selected = tgt_feats


        
        # Get coordinate distance
        coords_dist = torch.sqrt(torch.clamp(torch.sum((src_pcd_selected[:, None, :] - tgt_pcd_selected[None, :, :]) ** 2, dim=-1), min=0.0))


        # Get feature distance (from GeoTransformer Implementation)
        normalized_src_feats = F.normalize(src_feats_selected.squeeze(0), p=2, dim=-1) # (1, N, D) -> (N, D)
        normalized_tgt_feats = F.normalize(tgt_feats_selected.squeeze(0), p=2, dim=-1) # (1, M, D) -> (M, D)


        # Check NaN
        if torch.isnan(normalized_src_feats).any() or torch.isnan(normalized_tgt_feats).any():
            assert False, "[Circle Loss] Normalized features are nan\n src_feats: {}\n tgt_feats: {}".format(normalized_src_feats, normalized_tgt_feats)
        

        # Get feature distance
        dot = torch.einsum('x d, y d -> x y', normalized_src_feats, normalized_tgt_feats)
        dot = torch.clamp(dot, min=-1.0, max=1.0)
        value = 2.0 - 2.0 * dot
        assert (value >= 0).all(), f"Negative value detected in sqrt input: min={value.min()}"
        # (|x| - |y|)^2 = |x|^2 - 2<x, y> + |y|^2 where <x, y> = |x||y|cos(theta)
        # Also, we already normalized the features, so |x| = |y| = 1
        # so, |x|^2 - 2<x, y> + |y|^2 = 2 - 2<x, y> = 2 - 2cos(theta)
        # By, triangle formulat, 2 - 2 cos(theta) = 4 * sin(theta/2)^2
        # Hence, feats_dist = 2 * sin(theta/2)
        feats_dist = torch.sqrt(torch.clamp(value, min=1e-8))

        # Calculate circle loss and feature matching recall (FMR)
        circle_loss, pos_neg_distribution = self.get_circle_loss(coords_dist, feats_dist)

        if torch.isnan(circle_loss):
            assert False, "Circle loss is nan"
        
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

        



