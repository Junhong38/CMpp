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
            matching_scores (torch.Tensor): (B, N+M, N+M)
            pos_mask (torch.Tensor): (B, N+M, N+M)
            neg_mask (torch.Tensor): (B, N+M, N+M)
        
        Returns:
            neg_mask (torch.Tensor): (B, N+M, N+M)
            hard_neg_mask (torch.Tensor): (B, N+M, N+M)
        """
        batch_size, num_row, num_col = matching_scores.shape
        torch.set_printoptions(threshold=torch.inf, linewidth=1000, precision=5)


        if self.hard_negative: # Hard negative sampling
            # To find smallest pos score from each batch, we need to fill redundant scores with maximum score.
            postprocessed_for_pos = matching_scores * pos_mask + matching_scores.max() * (~pos_mask)
            smallest_pos_score = postprocessed_for_pos.reshape(batch_size, -1).min(dim=-1)[0] # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, )

            # Check if the score is bigger than the smallest pos score.
            bigger_than_smallest_pos_score = matching_scores >= smallest_pos_score[:, None, None]

            # We want to divide pos and neg completely.
            # So, if neg sample has bigger score than smallest pos sample, it is a hard negative.
            hard_neg_mask = torch.logical_and(neg_mask, bigger_than_smallest_pos_score)

        else:
            hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        

        if not self.no_balance:
            num_of_pos = pos_mask.reshape(batch_size, -1).sum(dim=-1) # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, )
            num_of_hard_negs = hard_neg_mask.reshape(batch_size, -1).sum(dim=-1) # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, )

            if num_of_hard_negs < num_of_pos // 2: 
                # If hard negatives are less than half of positive samples, we should sample more negative samples.
                num_of_sampled_negs = num_of_pos - num_of_hard_negs
                num_of_sampled_hards = num_of_hard_negs
            
            else:
                # If hard negatives are greater than half of positive samples, we should sample equal ratio from negative and hard negative samples.
                num_of_sampled_negs = num_of_pos - num_of_pos // 2
                num_of_sampled_hards = num_of_pos // 2

            # Sample the hard negatives
            hard_neg_indices = hard_neg_mask.nonzero(as_tuple=False) # (B, N+M, N+M) -> (num_of_true_parts, 3), where 3 is (batch_index, row_index, col_index)
            hard_neg_nonsampled = hard_neg_indices[torch.randperm(hard_neg_indices.size(0))[num_of_sampled_hards:]]
            hard_neg_mask[hard_neg_nonsampled[:,0], hard_neg_nonsampled[:,1], hard_neg_nonsampled[:,2]] = False

            # Sample the neg_mask to match proportions, and do not overlap with hard negatives
            neg_indices = torch.logical_and(neg_mask, ~hard_neg_mask).nonzero(as_tuple=False) # (B, N+M, N+M) -> (num_of_true_parts, 3), where 3 is (batch_index, row_index, col_index)
            neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[num_of_sampled_negs:]]
            neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1], neg_nonsampled[:,2]] = False


        neg_mask = torch.logical_or(neg_mask, hard_neg_mask) 
        avg_num_of_hard_negs = hard_neg_mask.reshape(batch_size, -1).sum(dim=-1).float().mean().item()
        return neg_mask, avg_num_of_hard_negs
    
    
    def get_circle_loss(self, coords_dist, feats_dist, matching_scores, active_mask):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        Trivially modified from GeoTransformer Implementation

        Args:
            coords_dist (torch.Tensor): (B, N+M, N+M)
            feats_dist (torch.Tensor): (B, N+M, N+M)
            matching_scores (torch.Tensor): (B, N+M, N+M)
            active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active

        Returns:
            torch.Tensor: (1, ), circle loss
            dict: (1, ), pos_neg_distribution
        """
        pos_mask = (coords_dist < self.pos_radius) * active_mask
        neg_mask = (coords_dist > self.safe_radius) * active_mask
        

        # Calculate Positive/Negative feats_dist distribution
        with torch.no_grad():
            pos_dists = feats_dist[pos_mask] # (B, N+M, N+M) -> (num_of_true_parts, )
            neg_dists = feats_dist[neg_mask] # (B, N+M, N+M) -> (num_of_true_parts, )

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

        # Masking the inactive points
        maks_for_row = active_mask.any(dim=-1) # (B, N+M, N+M) -> (B, N+M)
        maks_for_col = active_mask.any(dim=-2) # (B, N+M, N+M) -> (B, N+M)
            
        
        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach() # (B, N+M, N+M) -> (B, N+M)
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach() # (B, N+M, N+M) -> (B, N+M)

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() # (B, N+M, N+M)

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach() # (B, N+M, N+M)

        # log(Σ exp(γ * (d - m_pos) * w_pos))
        # If the point is inactive, set the loss to -1e12 to prevent it from affecting the loss
        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight * active_mask + -1e12 * (~active_mask), dim=-1) # (B, N+M, N+M) -> (B, N+M)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight * active_mask + -1e12 * (~active_mask), dim=-2) # (B, N+M, N+M) -> (B, N+M)

        # log(Σ exp(γ * (m_neg - d) * w_neg))
        # If the point is inactive, set the loss to -1e12 to prevent it from affecting the loss
        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight * active_mask + -1e12 * (~active_mask), dim=-1) # (B, N+M, N+M) -> (B, N+M)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight * active_mask + -1e12 * (~active_mask), dim=-2) # (B, N+M, N+M) -> (B, N+M)

        # Softplus = log(1+exp(x))
        # So, log(1+exp(x)) / log_scale -> log(1 + Σ exp(γ * (d - m_pos) * w_pos) + Σ exp(γ * (m_neg - d) * w_neg)) / log_scale
        loss_row = F.softplus(lse_pos_row + lse_neg_row) # (B, N+M)
        loss_col = F.softplus(lse_pos_col + lse_neg_col) # (B, N+M)

        loss_row = loss_row / self.log_scale
        loss_col = loss_col / self.log_scale
        
        # Prevent NaN
        anchor_loss_row = loss_row[row_sel].mean() if row_sel.sum() > 0 else torch.tensor(0.).to(loss_row.device)
        anchor_loss_col = loss_col[col_sel].mean() if col_sel.sum() > 0 else torch.tensor(0.).to(loss_col.device)
        

        circle_loss = (anchor_loss_row + anchor_loss_col) / 2
        
        return circle_loss, pos_neg_distribution


    def forward(self, pcd_raw, feats, gt_corr, gt_corr_offset_info, matching_scores, active_mask):
        """
        Args:
            pcd_raw (torch.Tensor): (B, N+M, 3)
            feats (torch.Tensor): (B, N+M, D )
            gt_corr (torch.Tensor): (total_Corr, 2) where total_Corr := Corr_1 + Corr_2 + ... + Corr_B
            gt_corr_offset_info (torch.Tensor): (B, ) where gt_corr_offset_info[i] shows size of Corr_i
            matching_scores (torch.Tensor): (B, N+M, N+M)
            active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active

        Returns:
            torch.Tensor: (1, ), circle loss
            dict: (1, ), pos_neg_distribution
        """
        # Check NaN
        if torch.isnan(pcd_raw).any() or torch.isnan(feats).any():
            assert False, "[Circle Loss] Input features are nan\n pcd_raw: {}\n feats: {}".format(pcd_raw, feats)
        
        # Get coordinate distance
        # (B, N+M, 1, 3) - (B, 1 , N+M, 3) -> (B, N+M, N+M, 3) -> (B, N+M, N+M)
        coords_dist = torch.sqrt(torch.clamp(torch.sum((pcd_raw[:, :, None, :] - pcd_raw[:, None, :, :]) ** 2, dim=-1), min=0.0)) # (B, N+M, N+M)
        coords_dist = coords_dist * active_mask # Remove inactive points
        
        # Get feature distance (from GeoTransformer Implementation)
        normalized_feats = F.normalize(feats, p=2, dim=-1) # (B, N+M, D)

        # Check NaN
        if torch.isnan(normalized_feats).any():
            assert False, "[Circle Loss] Normalized features are nan\n feats: {}".format(normalized_feats)
        
        # Get feature distance
        dot = torch.einsum('b x d, b y d -> b x y', normalized_feats, normalized_feats)
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
        feats_dist = feats_dist * active_mask # Remove inactive points

        # Calculate circle loss and feature matching recall (FMR)
        circle_loss, pos_neg_distribution = self.get_circle_loss(coords_dist, feats_dist, matching_scores, active_mask)

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

        



