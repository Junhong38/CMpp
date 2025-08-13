import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class CircleLoss(nn.Module):

    def __init__(self, log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(CircleLoss,self).__init__()
        self.log_scale = 24
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = 0.1
        self.neg_margin = 1.4
        
        self.pos_radius = 0.018
        self.safe_radius = 0.03

        self.max_points = 128

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """

        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() 

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    # def get_recall(self, coords_dist, feats_dist):
    #     """
    #     Get feature match recall, divided by number of true inliers
    #     """
    #     pos_mask = coords_dist < self.pos_radius
    #     n_gt_pos = (pos_mask.sum(-1)>0).float().sum()+1e-12
    #     try:
    #         _, sel_idx = torch.min(feats_dist, -1)
    #     except:
    #         return torch.tensor(0.).to(feats_dist.device)
    #     sel_dist = torch.gather(coords_dist,dim=-1,index=sel_idx[:,None])[pos_mask.sum(-1)>0]
    #     n_pred_pos = (sel_dist < self.pos_radius).float().sum()
    #     recall = n_pred_pos / n_gt_pos
    #     return recall

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence):
        if len(correspondence) == 0:
            print('[circle loss] No correspondence!')
            return torch.tensor(0.).to(src_feats.device)

        c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        c_select = c_dist < self.pos_radius - 0.001
        correspondence = correspondence[c_select]
        
        if correspondence.size(0) > self.max_points:
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]

        # Use only correspondence points
        src_idx = correspondence[:,0]
        tgt_idx = correspondence[:,1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[:, src_idx, :], tgt_feats[:, tgt_idx, :]

        # Get coordinate distance
        coords_dist = torch.sqrt(torch.sum((src_pcd[:, None, :] - tgt_pcd[None, :, :]) ** 2, dim=-1))

        # Get feature distance (from GeoTransformer Implementation)
        src_feats = F.normalize(src_feats.squeeze(0), p=2, dim=-1)
        tgt_feats = F.normalize(tgt_feats.squeeze(0), p=2, dim=-1)
        feats_dist = (2.0 - 2.0 * torch.einsum('x d, y d -> x y', src_feats, tgt_feats)).pow(0.5)
        
        # Calculate circle loss and feature matching recall (FMR)
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)
        
        if circle_loss != circle_loss:
            # print('[circle loss] NaN detected!')
            circle_loss = torch.tensor(0.).to(src_feats.device)
            
        return circle_loss

class PointMatchingLoss(nn.Module):
    def __init__(self):
        super(PointMatchingLoss, self).__init__()
        self.positive_radius = 0.018

    def forward(self, matching_scores, correlations, src_pcd, trg_pcd):
        coords_dist = torch.sqrt(torch.sum((src_pcd[:, None, :] - trg_pcd[None, :, :]) ** 2, dim=-1))
        gt_corr_map = coords_dist < self.positive_radius

        # Initialize labels for the loss calculation
        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        
        # Handle slack rows and columns
        slack_row_labels = torch.sum(gt_corr_map[:, :-1], dim=1) == 0
        slack_col_labels = torch.sum(gt_corr_map[:-1, :], dim=0) == 0

        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels
        
        # Calculate the loss
        loss = -matching_scores[labels].mean()

        return loss

# class OrientationLoss(nn.Module):
#     def __init__(self):
#         super(OrientationLoss, self).__init__()
#         self.eps = 1e-7

#     def inter_loss(self, src_ori, trg_ori, correspondence, src_gt_rot, trg_gt_rot):
#         src_ori = src_ori[:, correspondence[:,0]] 
#         trg_ori = trg_ori[:, correspondence[:,1]]

#         src_ori = torch.matmul(src_ori, src_gt_rot)
#         trg_ori = torch.matmul(trg_ori, trg_gt_rot)

#         diff = src_ori - trg_ori
#         f_norm = torch.norm(diff, p='fro', dim=(2, 3))
#         inter_loss = torch.mean(f_norm)
        
#         return inter_loss

#     def forward(self, src_ori, trg_ori, correspondence, gt_rot):
#         if len(correspondence) == 0:
#             return torch.tensor(0.).to(src_ori.device)

#         src_gt_rot = gt_rot[0]
#         trg_gt_rot = gt_rot[1]
#         ori_loss = self.inter_loss(src_ori, trg_ori, correspondence, src_gt_rot, trg_gt_rot)

#         return ori_loss

class OrientationLoss(nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()
        self.eps = 1e-7

    def inter_loss(self, orientation, normal):
        # Smooth L1 loss
        loss_fn = nn.SmoothL1Loss(beta=0.1, reduction='mean')
        inter_loss = loss_fn(orientation, normal)
        return inter_loss

    def forward(self, src_ori, trg_ori, gt_normals):
        src_ori = src_ori.squeeze() # (1, N, 1, 3) --> (N, 3)
        trg_ori = trg_ori.squeeze() # (1, M, 1, 3) --> (M, 3)

        src_normals = gt_normals[0].squeeze() # (1, N, 3) --> (N, 3)
        trg_normals = gt_normals[1].squeeze() # (1, M, 3) --> (M, 3)

        src_ori_loss = self.inter_loss(src_ori, src_normals)
        trg_ori_loss = self.inter_loss(trg_ori, trg_normals)

        return src_ori_loss + trg_ori_loss

class OrientationLossGeodesic(nn.Module):
    def __init__(self):
        super(OrientationLossGeodesic, self).__init__()
        self.eps = 1e-7

    def inter_loss(self, src_ori, trg_ori, correspondence, src_gt_rot, trg_gt_rot):
        src_ori = src_ori[:, correspondence[:,0]] 
        trg_ori = trg_ori[:, correspondence[:,1]]

        src_ori = torch.matmul(src_ori, src_gt_rot)
        trg_ori = torch.matmul(trg_ori, trg_gt_rot)
        
        R_diff = torch.matmul(src_ori.transpose(2,3), trg_ori)
        trace_R_diff = torch.einsum('bnii->bn', R_diff)
        theta = torch.acos(torch.clamp((trace_R_diff-1)/2, -1.0+self.eps, 1.0-self.eps))
        inter_loss = torch.mean(theta ** 2)
        return inter_loss

    def forward(self, src_ori, trg_ori, correspondence, gt_rot):
        if len(correspondence) == 0:
            return torch.tensor(0.).to(src_ori.device)

        src_gt_rot = gt_rot[0]
        trg_gt_rot = gt_rot[1]
        ori_loss = self.inter_loss(src_ori, trg_ori, correspondence, src_gt_rot, trg_gt_rot)

        return ori_loss