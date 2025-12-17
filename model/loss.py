import torch.nn as nn
import torch
import torch.nn.functional as F

class CircleLoss(nn.Module):

    def __init__(self, pos_radius=0.018, safe_radius=0.03, log_scale=24, pos_margin=0.1, neg_margin=1.4, pos_offset=0.0, neg_offset=0.0,
                 balance_mode='none', hard_negative='none', neg_topk=0, distance_type='l2', anchor_mode='default'):


        super(CircleLoss,self).__init__()
        self.log_scale = log_scale
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_offset = pos_offset
        self.neg_offset = neg_offset
        self.balance_mode = balance_mode
        self.hard_negative = hard_negative
        self.neg_topk = neg_topk
        self.distance_type = distance_type
        self.anchor_mode = anchor_mode

        self.pos_optimal = pos_margin - pos_offset
        self.neg_optimal = neg_margin + neg_offset

        self.pos_radius = pos_radius
        self.safe_radius = safe_radius

        
        print("------------------------------------------------------")
        print("INITIALIZING CircleLoss")
        print("------------------------------------------------------")
        print(f"pos_radius: {self.pos_radius}, safe_radius: {self.safe_radius}")
        print(f"log_scale: {self.log_scale}")
        print(f"pos_optimal: {self.pos_optimal}, pos_margin: {self.pos_margin}")
        print(f"neg_optimal: {self.neg_optimal}, neg_margin: {self.neg_margin}")
        print(f"pos_offset: {self.pos_offset}, neg_offset: {self.neg_offset}")
        print(f"balance_mode: {self.balance_mode}")
        print(f"hard_negative: {self.hard_negative}, neg_topk: {self.neg_topk}")
        print(f"distance_type: {self.distance_type}, anchor_mode: {self.anchor_mode}")
        print("------------------------------------------------------")


    def negative_sampling(self, matching_scores, pos_mask, neg_mask):
        """
        Args:
            matching_scores (torch.Tensor): (B, N+M, N+M), This already removed inactive points
            pos_mask (torch.Tensor): (B, N+M, N+M)
            neg_mask (torch.Tensor): (B, N+M, N+M)
        
        Returns:
            neg_mask (torch.Tensor): (B, N+M, N+M)
            hard_neg_mask (torch.Tensor): (B, N+M, N+M)
        """
        batch_size, num_row, num_col = matching_scores.shape

        if self.hard_negative == 'mix': # Hard negative sampling
            # To find smallest pos score from each batch, we need to fill redundant scores with maximum score.
            postprocessed_for_pos = matching_scores * pos_mask + (matching_scores.max() + 1) * (~pos_mask)
            smallest_pos_score = postprocessed_for_pos.reshape(batch_size, -1).min(dim=-1)[0] # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, )

            # Check if the score is bigger than the smallest pos score.
            bigger_than_smallest_pos_score = matching_scores >= smallest_pos_score[:, None, None]

            # We want to divide pos and neg completely.
            # So, if neg sample has bigger score than smallest pos sample, it is a hard negative.
            hard_neg_mask = torch.logical_and(neg_mask, bigger_than_smallest_pos_score)
        
        else: # 'none'
            hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        
        
        if self.neg_topk > 0:
            # Do not overlap with hard negatives
            pure_neg_mask = torch.logical_and(neg_mask, ~ hard_neg_mask)
            
            # Only sample topk neg samples. Value K will be same as the number of positive samples.
            # If default topk value is bigger than, choose default value.
            num_of_pos = pos_mask.reshape(batch_size, -1).sum(dim=-1) # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, )
            topk = num_of_pos.max()
            topk = max(topk, self.neg_topk)

            # To find topk neg score from each batch, we need to fill redundant scores with minimum score.
            postprocessed_for_neg = matching_scores * pure_neg_mask + matching_scores.min() * (~pure_neg_mask)
            topk_neg_score = postprocessed_for_neg.reshape(batch_size, -1).topk(k=topk, dim=-1)[0] # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, topk)
            kth_biggest_neg_score = topk_neg_score[:, -1] # (B, topk) -> (B, )
            bigger_than_kth_neg_score = matching_scores >= kth_biggest_neg_score[:, None, None]
            neg_mask = torch.logical_and(pure_neg_mask, bigger_than_kth_neg_score)
        

        if (self.balance_mode in ['half', 'only_hard']):
            # Do not overlap with hard negatives
            neg_mask = torch.logical_and(neg_mask, ~hard_neg_mask)

            num_of_pos = pos_mask.reshape(batch_size, -1).sum(dim=-1) # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, )
            num_of_negs = neg_mask.reshape(batch_size, -1).sum(dim=-1) # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, )
            num_of_hard_negs = hard_neg_mask.reshape(batch_size, -1).sum(dim=-1) # (B, N+M, N+M) -> (B, (N+M)*(N+M)) -> (B, )

            if self.balance_mode == 'half':
                # If hard negatives are less than half of positive samples, we should sample more negative samples.
                # If hard negatives are greater than half of positive samples, we should sample equal ratio from negative and hard negative samples.
                not_enough_hard_negs_part = num_of_hard_negs < num_of_pos // 2
                num_of_sampled_negs = (num_of_pos - num_of_hard_negs) * not_enough_hard_negs_part + (num_of_pos - num_of_pos // 2) * (~not_enough_hard_negs_part) # (B, )
                num_of_sampled_hards = num_of_hard_negs * not_enough_hard_negs_part + (num_of_pos // 2) * (~not_enough_hard_negs_part) # (B, )

                # If there is no positive samples, we should not sample any negative samples or hard negatives
                zero_num_of_pos = num_of_pos == 0
                num_of_sampled_negs = num_of_sampled_negs * (~zero_num_of_pos) + 0 * zero_num_of_pos
                num_of_sampled_hards = num_of_sampled_hards * (~zero_num_of_pos) + 0 * zero_num_of_pos

            elif self.balance_mode == 'only_hard':
                # Use all hard negatives, but make balance between negative and positive samples.
                num_of_sampled_negs = num_of_pos # (B, )
                num_of_sampled_hards = num_of_hard_negs # (B, )
            
            else:
                raise NotImplementedError(f"Balance mode {self.balance_mode} not implemented")

            # Sample the hard negatives
            hard_neg_indices = hard_neg_mask.nonzero(as_tuple=False) # (B, N+M, N+M) -> (num_of_true_parts, 3), where 3 is (batch_index, row_index, col_index)
            criteria_for_hard_negs = num_of_sampled_hards.repeat_interleave(num_of_hard_negs) # (num_of_true_parts, )
            randperm_for_hard_negs = torch.cat([torch.randperm(num_of_hard_negs[i], device=hard_neg_indices.device) for i in range(len(num_of_hard_negs))], dim=0) # (num_of_true_parts, )
            non_sampled_part_for_hard_negs = randperm_for_hard_negs >= criteria_for_hard_negs # (num_of_true_parts, )
            hard_neg_nonsampled = hard_neg_indices[non_sampled_part_for_hard_negs] # (num_of_non_sampled_parts, 3)
            hard_neg_mask[hard_neg_nonsampled[:,0], hard_neg_nonsampled[:,1], hard_neg_nonsampled[:,2]] = False

            # Sample the neg_mask to match proportions, and do not overlap with hard negatives
            neg_indices = neg_mask.nonzero(as_tuple=False) # (B, N+M, N+M) -> (num_of_true_parts, 3), where 3 is (batch_index, row_index, col_index)
            criteria_for_negs = num_of_sampled_negs.repeat_interleave(num_of_negs) # (num_of_true_parts, )
            randperm_for_negs = torch.cat([torch.randperm(num_of_negs[i], device=neg_indices.device) for i in range(len(num_of_negs))], dim=0) # (num_of_true_parts, )
            non_sampled_part_for_negs = randperm_for_negs >= criteria_for_negs # (num_of_true_parts, )
            neg_nonsampled = neg_indices[non_sampled_part_for_negs] # (num_of_non_sampled_parts, 3)
            neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1], neg_nonsampled[:,2]] = False

        neg_mask = torch.logical_or(neg_mask, hard_neg_mask) 
        avg_num_of_hard_negs = hard_neg_mask.reshape(batch_size, -1).sum(dim=-1).float().mean().item()
        avg_num_of_negs = neg_mask.reshape(batch_size, -1).sum(dim=-1).float().mean().item()
        
        return neg_mask, avg_num_of_hard_negs, avg_num_of_negs
    
    
    def get_circle_loss(self, coords_dist, feats_dist, matching_scores, active_mask):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        Trivially modified from GeoTransformer Implementation

        Args:
            coords_dist (torch.Tensor): (B, N+M, N+M)
            feats_dist (torch.Tensor): (B, N+M, N+M)
            matching_scores (torch.Tensor): (B, N+M, N+M), This already removed inactive points
            active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active

        Returns:
            torch.Tensor: (1, ), circle loss
            dict: (1, ), pos_neg_distribution
        """

        # Masking the inactive points
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
        
        neg_mask, pos_neg_distribution['num_of_hard_neg'], pos_neg_distribution['num_of_neg'] = self.negative_sampling(matching_scores, pos_mask, neg_mask)

        if self.anchor_mode == 'default':
            # get anchors that have both positive and negative pairs
            row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach() # (B, N+M, N+M) -> (B, N+M)
            col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach() # (B, N+M, N+M) -> (B, N+M)
        elif self.anchor_mode == 'all_pos':
            # Use all positive pairs as anchors
            row_sel = (pos_mask.sum(-1)>0).detach()
            col_sel = (pos_mask.sum(-2)>0).detach()
        elif self.anchor_mode == 'all':
            # Use all pairs as anchors
            row_sel = torch.ones_like(pos_mask[:,:,0], dtype=torch.bool).detach() # (B, N+M)
            col_sel = torch.ones_like(pos_mask[:,:,0], dtype=torch.bool).detach() # (B, N+M)
        else:
            raise NotImplementedError(f"Anchor mode {self.anchor_mode} not implemented")

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() # (B, N+M, N+M)

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach() # (B, N+M, N+M)

        # log(Σ exp(γ * (d - m_pos) * w_pos))
        # If the point is inactive, set the loss to -1e12 to prevent it from affecting the loss
        lse_pos_part = self.log_scale * (feats_dist - self.pos_margin) * pos_weight * active_mask + -1e12 * (~active_mask)
        lse_pos_row = torch.logsumexp(lse_pos_part, dim=-1) # (B, N+M, N+M) -> (B, N+M)
        lse_pos_col = torch.logsumexp(lse_pos_part, dim=-2) # (B, N+M, N+M) -> (B, N+M)

        # log(Σ exp(γ * (m_neg - d) * w_neg))
        # If the point is inactive, set the loss to -1e12 to prevent it from affecting the loss
        lse_neg_part = self.log_scale * (self.neg_margin - feats_dist) * neg_weight * active_mask + -1e12 * (~active_mask)
        lse_neg_row = torch.logsumexp(lse_neg_part, dim=-1) # (B, N+M, N+M) -> (B, N+M)
        lse_neg_col = torch.logsumexp(lse_neg_part, dim=-2) # (B, N+M, N+M) -> (B, N+M)

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


    def forward(self, pcd_raw, feats, matching_scores, active_mask):
        """
        Args:
            pcd_raw (torch.Tensor): (B, N+M, 3)
            feats (torch.Tensor): (B, D, N+M)
            matching_scores (torch.Tensor): (B, N+M, N+M), This already removed inactive points
            active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active

        Returns:
            torch.Tensor: (1, ), circle loss
            dict: (1, ), pos_neg_distribution
        """
        # Check NaN
        if torch.isnan(pcd_raw).any() or torch.isnan(feats).any():
            assert False, "[Circle Loss] Input features are nan\n pcd_raw: {}\n feats: {}".format(pcd_raw, feats)
        
        # Get coordinate distance
        coords_dist = torch.cdist(pcd_raw, pcd_raw, p=2) # (B, N+M, N+M)
        coords_dist = coords_dist * active_mask # Remove inactive points
        
        # Get feature distance (from GeoTransformer Implementation)
        normalized_feats = F.normalize(feats, p=2, dim=-2) # (B, D, N+M)

        # Check NaN
        if torch.isnan(normalized_feats).any():
            assert False, "[Circle Loss] Normalized features are nan\n feats: {}".format(normalized_feats)
        
        # Get feature distance
        dot = torch.einsum('b d x, b d y -> b x y', normalized_feats, normalized_feats)
        dot = torch.clamp(dot, min=-1.0, max=1.0)

        if self.distance_type == 'l2':
            value = 2.0 - 2.0 * dot
            # (x - y)^2 = |x|^2 - 2<x, y> + |y|^2 where <x, y> = |x||y|cos(theta)
            # Also, we already normalized the features, so |x| = |y| = 1
            # so, |x|^2 - 2<x, y> + |y|^2 = 2 - 2<x, y> = 2 - 2cos(theta)
            # By, triangle formula, 2 - 2 cos(theta) = 4 * sin(theta/2)^2
            # Hence, feats_dist = 2 * sin(theta/2)
            # Finally, to prevent NaN during backward, use minimum value 1e-8
            
        else: # Cosine similarity
            value = 1 - dot # (B, N+M, N+M)
        
        assert (value >= 0).all(), f"Negative value detected in sqrt input: min={value.min()}"
        feats_dist = torch.sqrt(torch.clamp(value, min=1e-8)) if self.distance_type == 'l2' else value
        feats_dist = feats_dist * active_mask # Remove inactive points

        # Calculate circle loss and feature matching recall (FMR)
        circle_loss, pos_neg_distribution = self.get_circle_loss(coords_dist, feats_dist, matching_scores, active_mask)

        if torch.isnan(circle_loss):
            assert False, "Circle loss is nan"
        
        return circle_loss, coords_dist, pos_neg_distribution


class PointMatchingLoss(nn.Module):
    def __init__(self, pos_radius=0.018, safe_radius=0.03):
        super(PointMatchingLoss, self).__init__()
        self.pos_radius = pos_radius
        self.safe_radius = safe_radius

        print("------------------------------------------------------")
        print("INITIALIZING PointMatchingLoss")
        print("------------------------------------------------------")
        print(f"pos_radius: {self.pos_radius}, safe_radius: {self.safe_radius}")
        print("------------------------------------------------------")


    def forward(self, matching_scores, coords_dist, active_mask, matching_norm_mode):
        """
        Args:
            matching_scores (torch.Tensor): (B, N+M+1, N+M+1) if mode is ['sinkhorn', 'softmax'], otherwise (B, N+M, N+M) (This already removed inactive points)
            coords_dist (torch.Tensor): (B, N+M, N+M)
            active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active
            matching_norm_mode (str): 'sinkhorn', 'softmax', 'none'

        Returns:
            torch.Tensor: (1, ), point matching loss
        """
        gt_corr_map = torch.logical_and(coords_dist < self.pos_radius, active_mask) # (B, N+M, N+M)

        if matching_norm_mode == 'none': # log-likelihood loss
            # To prevent INF value, use minimum value 1e-8
            matching_loss_scores = torch.log(matching_scores + 1e-8)
            neg_mask = torch.logical_and(coords_dist > self.safe_radius, active_mask) # (B, N+M, N+M)
            pos_part_loss = - matching_loss_scores[gt_corr_map].mean()
            neg_part_loss = matching_loss_scores[neg_mask].mean()

            # Make positive samples' score to be larger, also make negative samples' score to be smaller
            loss = pos_part_loss + neg_part_loss
        
        else: # Use slack variables, ['sinkhorn', 'softmax'], negative log-likelihood loss
            # To prevent INF value, use minimum value 1e-8
            matching_loss_scores = matching_scores if matching_norm_mode == 'sinkhorn' else torch.log(matching_scores + 1e-8)

            # Initialize labels for the loss calculation
            labels = torch.zeros_like(matching_scores, dtype=torch.bool) # (B, N+M+1, N+M+1)

            # Handle slack rows and columns
            # torch.sum(gt_corr_map, dim=-1) == 0 -> True if there is no matching parts
            # active_mask.any(dim=-1) -> True if the row is active
            slack_row_labels = torch.logical_and(torch.sum(gt_corr_map, dim=-1) == 0, active_mask.any(dim=-1)) # (B, N+M)
            slack_col_labels = torch.logical_and(torch.sum(gt_corr_map, dim=-2) == 0, active_mask.any(dim=-2)) # (B, N+M)

            labels[:, :-1, :-1] = gt_corr_map
            labels[:, :-1, -1] = slack_row_labels
            labels[:, -1, :-1] = slack_col_labels

            # Calculate the loss
            loss = - matching_loss_scores[labels].mean()

        return loss


class OrientationLoss(nn.Module):
    def __init__(self, consistency_loss=False):
        super(OrientationLoss, self).__init__()
        self.consistency_loss = consistency_loss
        self.loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='mean')
    
    def forward(self, oris, gt_normals, batch_scaled_batch_info, gt_corr, gt_corr_bincount_info):
        """
        Assume there are two objects in the batch

        Args:
            oris (torch.Tensor): (B, N+M, 3, 3), first basis should be aligned with gt_normals[0]
            gt_normals (torch.Tensor): (B, N+M, 3)
            batch_scaled_batch_info (torch.Tensor): (B, N+M), batch index of the point cloud
            gt_corr (torch.Tensor): (total_Corr, 2) where total_Corr := Corr_1 + Corr_2 + ... + Corr_B
            gt_corr_bincount_info (torch.Tensor): (B, ) where gt_corr_bincount_info[i] shows size of Corr_i

        Returns:
            torch.Tensor: (1, ), orientation loss
        """
        pred_normal = oris[:, :, 0, :] # (B, N+M, 3)
        normal_loss = self.loss_fn(pred_normal, gt_normals)

        if self.consistency_loss and (not torch.all(gt_corr_bincount_info == 0)): # Make frame from src and trg be consistent with each other
            batch_size, num_points = oris.shape[:2]

            # We assume there are two objects in the batch
            obj_bincounts = batch_scaled_batch_info.reshape(-1).bincount().reshape(batch_size, 2) # (B*num_of_objs, ) -> (B, 2), num_of_objs = 2

            # Distinguish between src and trg
            obj_idx_base = obj_bincounts[:,0] # (B, )
            obj_idx_base = obj_idx_base.repeat_interleave(gt_corr_bincount_info) # (B,) -> (total_Corr,)
            obj_idx_base = torch.stack([torch.zeros_like(obj_idx_base), obj_idx_base], dim=-1) # (total_Corr, 2)

            # Distinguish between batch index
            idx_base = torch.arange(0, batch_size, device=oris.device) * num_points # (B, )
            idx_base = idx_base.repeat_interleave(gt_corr_bincount_info) # (total_Corr, )

            # Final batch scaled gt_corr
            batch_scaled_gt_corr = gt_corr + obj_idx_base + idx_base[:, None] # (total_Corr, 2)

            reshaped_oris = oris.reshape(-1, 3, 3) # (B, N+M, 3, 3) -> (B*(N+M), 3, 3)

            src_from_mating_surface = reshaped_oris[batch_scaled_gt_corr[:,0], :, :] # (total_Corr, 3, 3)
            trg_from_mating_surface = reshaped_oris[batch_scaled_gt_corr[:,1], :, :] # (total_Corr, 3, 3)

            consistency_loss_2nd = self.loss_fn(src_from_mating_surface[:, 1, :], trg_from_mating_surface[:, 2, :])
            consistency_loss_3rd = self.loss_fn(src_from_mating_surface[:, 2, :], trg_from_mating_surface[:, 1, :])
            consistency_loss = (consistency_loss_2nd + consistency_loss_3rd) / 2 
        
        else:
            consistency_loss = torch.tensor(0.).to(pred_normal.device)
        
        final_loss = normal_loss + consistency_loss

        return final_loss

        



