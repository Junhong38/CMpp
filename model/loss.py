import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class CircleLoss(nn.Module):

    def __init__(self, pos_radius=0.018, safe_radius=0.03, log_scale=24, pos_optimal=0.1, neg_optimal=1.4, same_opt=False, no_balance=False, hard_negative=False):


        super(CircleLoss,self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.no_balance = no_balance
        self.hard_negative = hard_negative


        if same_opt:
            self.pos_margin = pos_optimal
            self.neg_margin = neg_optimal
        else:
            self.pos_margin = pos_optimal - 0.05
            self.neg_margin = neg_optimal + 0.05

        self.pos_radius = pos_radius
        self.safe_radius = safe_radius

        
        print("------------------------------------------------------")
        print("INITIALIZING CircleLoss")
        print("------------------------------------------------------")
        print(f"pos_radius: {self.pos_radius}, safe_radius: {self.safe_radius}")
        print(f"log_scale: {self.log_scale}")
        print(f"pos_optimal: {self.pos_optimal}, pos_margin: {self.pos_margin}")
        print(f"neg_optimal: {self.neg_optimal}, neg_margin: {self.neg_margin}")
        print(f"same_opt: {same_opt}, no_balance: {self.no_balance}")
        print(f"hard_negative: {self.hard_negative}")
        print("------------------------------------------------------")


    def negative_sampling(self, matching_scores, pos_mask, neg_mask):
        """
        Args:
            matching_scores (torch.Tensor): (1, N, M)
            pos_mask (torch.Tensor): (N, M)
            neg_mask (torch.Tensor): (N, M)
        """

        if self.hard_negative: # Hard negative sampling
            smallest_pos_score = matching_scores[0][pos_mask].min()
            bigger_than_smallest_pos_score = matching_scores[0] >= smallest_pos_score

            # We want to divide pos and neg completely.
            # So, if neg sample has bigger score than smallest pos sample, it is a hard negative.
            hard_neg_mask = torch.logical_and(neg_mask, bigger_than_smallest_pos_score)

        else:
            hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        

        if not self.no_balance:
            num_of_pos = pos_mask.sum()
            num_of_hard_negs = hard_neg_mask.sum()

            if num_of_hard_negs < num_of_pos // 2: 
                # If hard negatives are less than half of positive samples, we should sample more negative samples.
                num_of_sampled_negs = num_of_pos - num_of_hard_negs
                num_of_sampled_hards = num_of_hard_negs
            else:
                # If hard negatives are greater than half of positive samples, we should sample equal ratio from negative and hard negative samples.
                num_of_sampled_negs = num_of_pos - num_of_pos // 2
                num_of_sampled_hards = num_of_pos // 2

            # Sample the hard negatives
            hard_neg_indices = hard_neg_mask.nonzero(as_tuple=False)
            hard_neg_nonsampled = hard_neg_indices[torch.randperm(hard_neg_indices.size(0))[num_of_sampled_hards:]]
            hard_neg_mask[hard_neg_nonsampled[:,0], hard_neg_nonsampled[:,1]] = False

            # Sample the neg_mask to match proportions, and do not overlap with hard negatives
            neg_indices = torch.logical_and(neg_mask, ~hard_neg_mask).nonzero(as_tuple=False)
            neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[num_of_sampled_negs:]]
            neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1]] = False


        neg_mask = torch.logical_or(neg_mask, hard_neg_mask) 
        return neg_mask, hard_neg_mask.sum()
    
    
    def get_circle_loss(self, coords_dist, feats_dist, matching_scores):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        Trivially modified from GeoTransformer Implementation

        Args:
            coords_dist (torch.Tensor): (N, M)
            feats_dist (torch.Tensor): (N, M)
            matching_scores (torch.Tensor): (1, N, M)

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
        
        neg_mask, pos_neg_distribution['num_of_hard_neg'] = self.negative_sampling(matching_scores, pos_mask, neg_mask)
            
        
        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach() # (N,M) -> (N, )
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach() # (N,M) -> (M, )

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() # (N,M)

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach() # (N,M)

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

        loss_row = loss_row / self.log_scale
        loss_col = loss_col / self.log_scale
        
        # Prevent NaN
        anchor_loss_row = loss_row[row_sel].mean() if row_sel.sum() > 0 else torch.tensor(0.).to(loss_row.device)
        anchor_loss_col = loss_col[col_sel].mean() if col_sel.sum() > 0 else torch.tensor(0.).to(loss_col.device)
        

        circle_loss = (anchor_loss_row + anchor_loss_col) / 2
        
        return circle_loss, pos_neg_distribution


    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, matching_scores):
        """
        Args:
            src_pcd (torch.Tensor): (N, 3)
            tgt_pcd (torch.Tensor): (M, 3)
            src_feats (torch.Tensor): (1, N, D)
            tgt_feats (torch.Tensor): (1, M, D)
            correspondence (torch.Tensor): (P, 2)
            matching_scores (torch.Tensor): (1, N, M)

        Returns:
            torch.Tensor: (1, ), circle loss
            dict: (1, ), pos_neg_distribution
        """

        print(f"src_pcd.shape: {src_pcd.shape}")
        print(f"tgt_pcd.shape: {tgt_pcd.shape}")
        print(f"src_feats.shape: {src_feats.shape}")
        print(f"tgt_feats.shape: {tgt_feats.shape}")
        print(f"correspondence.shape: {correspondence.shape}")
        print(f"matching_scores.shape: {matching_scores.shape}")

        # Check NaN
        if torch.isnan(src_pcd).any() or torch.isnan(tgt_pcd).any() or torch.isnan(src_feats).any() or torch.isnan(tgt_feats).any():
            assert False, "[Circle Loss] Input features are nan\n src_pcd: {}\n tgt_pcd: {}\n src_feats: {}\n tgt_feats: {}".format(src_pcd, tgt_pcd, src_feats, tgt_feats)


        if len(correspondence) == 0:
            print('[circle loss] No correspondence!')
            zero_pos_neg_distribution = {
                'pos_mean': 0,'pos_std': 0, 'pos_min': 0, 'pos_max': 0,
                'neg_mean': 0,'neg_std': 0, 'neg_min': 0, 'neg_max': 0,
            }
            return torch.tensor(0.).to(src_feats.device), zero_pos_neg_distribution

        
        # Get coordinate distance
        coords_dist = torch.sqrt(torch.clamp(torch.sum((src_pcd[:, None, :] - tgt_pcd[None, :, :]) ** 2, dim=-1), min=0.0))


        # Get feature distance (from GeoTransformer Implementation)
        normalized_src_feats = F.normalize(src_feats.squeeze(0), p=2, dim=-1) # (1, N, D) -> (N, D)
        normalized_tgt_feats = F.normalize(tgt_feats.squeeze(0), p=2, dim=-1) # (1, M, D) -> (M, D)


        # Check NaN
        if torch.isnan(normalized_src_feats).any() or torch.isnan(normalized_tgt_feats).any():
            assert False, "[Circle Loss] Normalized features are nan\n src_feats: {}\n tgt_feats: {}".format(normalized_src_feats, normalized_tgt_feats)
        

        # Get feature distance
        dot = torch.einsum('x d, y d -> x y', normalized_src_feats, normalized_tgt_feats)
        dot = torch.clamp(dot, min=-1.0, max=1.0)
        value = 2.0 - 2.0 * dot
        assert (value >= 0).all(), f"Negative value detected in sqrt input: min={value.min()}"
        # (x - y)^2 = |x|^2 - 2<x, y> + |y|^2 where <x, y> = |x||y|cos(theta)
        # Also, we already normalized the features, so |x| = |y| = 1
        # so, |x|^2 - 2<x, y> + |y|^2 = 2 - 2<x, y> = 2 - 2cos(theta)
        # By, triangle formula, 2 - 2 cos(theta) = 4 * sin(theta/2)^2
        # Hence, feats_dist = 2 * sin(theta/2)
        # Finally, to prevent NaN during backward, use minimum value 1e-8
        feats_dist = torch.sqrt(torch.clamp(value, min=1e-8))

        # Calculate circle loss and feature matching recall (FMR)
        circle_loss, pos_neg_distribution = self.get_circle_loss(coords_dist, feats_dist, matching_scores)

        if torch.isnan(circle_loss):
            assert False, "Circle loss is nan"
        
        return circle_loss, pos_neg_distribution


class PointMatchingLoss(nn.Module):
    def __init__(self, pos_radius=0.018):
        super(PointMatchingLoss, self).__init__()
        self.positive_radius = pos_radius

    def forward(self, matching_scores, src_pcd, trg_pcd):
        """
        Args:
            matching_scores (torch.Tensor): (1, N, M)
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
    def __init__(self, consistency_loss=False):
        super(OrientationLoss, self).__init__()
        self.consistency_loss = consistency_loss
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
        
        src_normal_basis = src_ori[:, :, 0, :] # (1, N, 3)
        trg_normal_basis = trg_ori[:, :, 0, :] # (1, M, 3)

        src_normal_basis_loss = self.loss_fn(src_normal_basis, gt_normals[0])
        trg_normal_basis_loss = self.loss_fn(trg_normal_basis, gt_normals[1])

        final_loss = (src_normal_basis_loss + trg_normal_basis_loss) / 2

        if self.consistency_loss and (len(correspondence) > 0): # Make frame from src and trg be consistent with each other
            src_from_mating_surface = src_ori[:, correspondence[:,0], :, :] # (1, P, 3, 3)
            trg_from_mating_surface = trg_ori[:, correspondence[:,1], :, :] # (1, P, 3, 3)

            consistency_loss_2nd = self.loss_fn(src_from_mating_surface[:, :, 1, :], trg_from_mating_surface[:, :, 2, :]) # 2nd <-> 3rd
            consistency_loss_3rd = self.loss_fn(src_from_mating_surface[:, :, 2, :], trg_from_mating_surface[:, :, 1, :]) # 3rd <-> 2nd
            consistency_loss = (consistency_loss_2nd + consistency_loss_3rd) / 2
            final_loss = final_loss + consistency_loss

        return final_loss

        



