import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class CircleLoss(nn.Module):

    def __init__(self, log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(CircleLoss,self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        # self.pos_margin = 0.1
        # self.neg_margin = 1.4
        
        self.pos_radius = 0.018
        self.safe_radius = 0.03

        # self.max_points = 128

    # def get_circle_loss(self, coords_dist, feats_dist):
    #     """
    #     Modified from: https://github.com/XuyangBai/D3Feat.pytorch
    #     """

    #     pos_mask = coords_dist < self.pos_radius
    #     neg_mask = coords_dist > self.safe_radius

    #     if pos_mask.sum() > neg_mask.sum():
    #         breakpoint()
        
    #     # sample the neg_mask to match proportions
    #     neg_indices = neg_mask.nonzero(as_tuple=False)
    #     neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[pos_mask.sum():]]
    #     neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1]] = False

    #     # get anchors that have both positive and negative pairs
    #     row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0))
    #     col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0))

    #     # get alpha for both positive and negative pairs
    #     pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
    #     pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
    #     pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)

    #     neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
    #     neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
    #     neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight)

    #     # lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
    #     # lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

    #     # lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
    #     # lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

    #     # P_distribution = (feats_dist * pos_mask)[row_sel, col_sel]
    #     # N_distribution = (feats_dist * neg_mask)[row_sel, col_sel]

    #     lse_pos_row = torch.logsumexp(self.log_scale * pos_weight, dim=-1)
    #     lse_pos_col = torch.logsumexp(self.log_scale * pos_weight, dim=-2)

    #     lse_neg_row = torch.logsumexp(self.log_scale * neg_weight, dim=-1)
    #     lse_neg_col = torch.logsumexp(self.log_scale * neg_weight, dim=-2)

    #     loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
    #     loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

    #     circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

    #     P_margin = (lse_pos_row[row_sel].mean().detach().cpu() + lse_pos_col[col_sel].mean().detach().cpu()) / 2
    #     N_margin = (lse_neg_row[row_sel].mean().detach().cpu() + lse_neg_col[col_sel].mean().detach().cpu()) / 2

    #     return circle_loss, P_margin.mean().detach().cpu(), N_margin.mean().detach().cpu()

    # def get_circle_loss(self, coords_dist, feats_dist):
        # """
        # Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        # """
 
        # pos_mask = coords_dist < self.pos_radius
        # neg_mask = coords_dist > self.safe_radius
 
        # # sample the neg_mask to match proportions
        # neg_indices = neg_mask.nonzero(as_tuple=False)
        # neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[pos_mask.sum():]]
        # neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1]] = False
 
        # # get alpha for both positive and negative pairs
        # pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        # pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        # pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight) # 205, 205
 
        # neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        # neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        # neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight) # 205, 205
 
        # lse_pos = torch.logsumexp(pos_weight[pos_mask], dim=0) #
        # lse_neg = torch.logsumexp(neg_weight[neg_mask], dim=0) #
 
        # circle_loss = (lse_pos + lse_neg) / 2
 
        # P, N = None, None
        # return circle_loss, P, N

######
    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
 
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius
 
        # sample the neg_mask to match proportions
        neg_indices = neg_mask.nonzero(as_tuple=False)
        neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[pos_mask.sum():]]
        neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1]] = False
 
        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight) # 205, 205
 
        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight) # 205, 205
        
        lse_pos = torch.logsumexp(self.log_scale * pos_weight[pos_mask], dim=0) #
        lse_neg = torch.logsumexp(self.log_scale * neg_weight[neg_mask], dim=0) #
 
        lse_pos = F.softplus(lse_pos)/self.log_scale
        lse_neg = F.softplus(lse_neg)/self.log_scale
 
        # circle_loss = 5*lse_pos + lse_neg
        circle_loss = lse_pos
        # breakpoint()

        circle_loss = pos_weight[pos_mask].sum()

        P_margin, N_margin = None, None

        breakpoint()
 
        return circle_loss, P_margin, N_margin

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

        # c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        # c_select = c_dist < self.pos_radius - 0.001
        # correspondence = correspondence[c_select]
        
        # if correspondence.size(0) > self.max_points:
        #     choice = np.random.permutation(correspondence.size(0))[:self.max_points]
        #     correspondence = correspondence[choice]

        # Use only correspondence points
        # src_idx = correspondence[:,0]
        # tgt_idx = correspondence[:,1]
        # src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        # src_feats, tgt_feats = src_feats[:, src_idx, :], tgt_feats[:, tgt_idx, :]

        # 중복제거 ver.
        src_idx = torch.unique(correspondence[:,0])
        tgt_idx = torch.unique(correspondence[:,1])
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

class ContrastiveLoss(nn.Module):

    def __init__(self, log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(ContrastiveLoss, self).__init__()
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        
        self.pos_radius = 0.018
        self.safe_radius = 0.03

        self.max_points = 128

    def get_loss(self, coords_dist, feats_sim):
        """
        InfoNCE-style loss (NT-Xent)
        """

        temperature = 0.1
        logits = feats_sim / temperature

        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        # sample the neg_mask to match proportions
        neg_indices = neg_mask.nonzero(as_tuple=False)
        neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[pos_mask.sum():]]
        neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1]] = False

        ####
        # pos_logits = logits.masked_fill(~pos_mask, -1e9)
        # neg_logits = logits.masked_fill(~neg_mask, -1e9)

        # exp_pos = torch.exp(pos_logits)
        # exp_neg = torch.exp(neg_logits)

        # pos_sum = exp_pos.sum(dim=-1) # (N, )
        # denom = pos_sum + exp_neg.sum(dim=-1) # (N, )
        
        # loss_per_sample = -torch.log((pos_sum + 1e-9) / (denom + 1e-9))
        # loss = loss_per_sample.mean()
        ####

        P_margin = torch.max((feats_sim[pos_mask] - self.pos_optimal), torch.zeros_like(feats_sim[pos_mask])) 
        N_margin = torch.max((self.neg_optimal - feats_sim[neg_mask]), torch.zeros_like(feats_sim[neg_mask]))

        loss = (P_margin.mean() + N_margin.mean()) / 2

        return loss

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence):
        if len(correspondence) == 0:
            print('[circle loss] No correspondence!')
            return torch.tensor(0.).to(src_feats.device)

        # c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        # c_select = c_dist < self.pos_radius - 0.001
        # correspondence = correspondence[c_select]
        
        # if correspondence.size(0) > self.max_points:
        #     choice = np.random.permutation(correspondence.size(0))[:self.max_points]
        #     correspondence = correspondence[choice]

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
        # feats_sim = torch.einsum('x d, y d -> x y', src_feats, tgt_feats)  # (N, M), cosine similarity
        
        # Calculate circle loss and feature matching recall (FMR)
        loss = self.get_loss(coords_dist, feats_dist)
        
        if loss != loss:
            # print('[circle loss] NaN detected!')
            loss = torch.tensor(0.).to(src_feats.device)
            
        return loss

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
        # slack_row_labels = torch.sum(gt_corr_map[:, :-1], dim=1) == 0
        # slack_col_labels = torch.sum(gt_corr_map[:-1, :], dim=0) == 0
        slack_row_labels = torch.sum(gt_corr_map, dim=1) == 0
        slack_col_labels = torch.sum(gt_corr_map, dim=0) == 0

        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels
        
        # Calculate the loss
        loss = -matching_scores[labels].mean()
        # breakpoint()

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
        self.reg_weight = 1.0 #0.1

    def angle_loss(self, src_angle, trg_angle, correspondence, src_gt_normal, trg_gt_normal):
        src_angle = src_angle[correspondence[:,0]] 
        trg_angle = trg_angle[correspondence[:,1]]

        loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='mean')
        angle_loss = loss_fn(src_angle, trg_angle)
        
        return angle_loss

    def axis_loss(self, axis, normal):
        # Smooth L1 loss
        loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='mean')
        axis_loss = loss_fn(axis, normal)
        return axis_loss

    def angle_regularization(self, angle):
        return torch.mean(torch.relu(angle - 0.9*torch.pi) ** 2)

    def forward(self, src_ori, trg_ori, correspondence, gt_normals):
        if len(correspondence) == 0:
            return torch.tensor(0.).to(src_ori.device)

        src_ori = src_ori.squeeze() # (1, N, 1, 3) --> (N, 3)
        trg_ori = trg_ori.squeeze() # (1, M, 1, 3) --> (M, 3)

        src_angle = torch.linalg.norm(src_ori, dim=-1, keepdim=True)
        trg_angle = torch.linalg.norm(trg_ori, dim=-1, keepdim=True)
        print(f"src_MIN : {src_angle.min()} | src_MAX : {src_angle.max()}")

        src_axis = src_ori / (src_angle + self.eps)
        trg_axis = trg_ori / (trg_angle + self.eps)

        src_normals = gt_normals[0].squeeze() # (1, N, 3) --> (N, 3)
        trg_normals = gt_normals[1].squeeze() # (1, M, 3) --> (M, 3)

        src_axis_loss = self.axis_loss(src_axis, src_normals)
        trg_axis_loss = self.axis_loss(trg_axis, trg_normals)

        angle_loss = self.angle_loss(src_angle, trg_angle, correspondence, src_normals, trg_normals)

        reg_loss = self.angle_regularization(src_angle) + self.angle_regularization(trg_angle)

        return (src_axis_loss + trg_axis_loss) / 2 + angle_loss + self.reg_weight * reg_loss
        # return src_axis_loss + trg_axis_loss + reg_loss

class OrientationLoss_old(nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()
        self.eps = 1e-7

    def inter_loss(self, orientation, normal):
        # Smooth L1 loss
        loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='mean')
        inter_loss = loss_fn(orientation, normal)
        return inter_loss

    def forward(self, src_ori, trg_ori, src_gt_rot, trg_gt_rot):
        
        src_ori_loss = self.inter_loss(src_ori, src_gt_rot)
        trg_ori_loss = self.inter_loss(trg_ori, trg_gt_rot)

        return src_ori_loss + trg_ori_loss

class OrientationLoss_old2(nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()
        self.eps = 1e-7

    def inter_loss(self, orientation, normal):
        # Smooth L1 loss
        loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='mean')
        inter_loss = loss_fn(orientation, normal)
        return inter_loss

    def forward(self, src_vec, trg_vec, gt_normals):
        src_vec = src_vec.squeeze() # (1, N, 3) --> (N, 3)
        trg_vec = trg_vec.squeeze() # (1, M, 3) --> (M, 3)

        src_normals = gt_normals[0].squeeze() # (1, N, 3) --> (N, 3)
        trg_normals = gt_normals[1].squeeze() # (1, M, 3) --> (M, 3)
        
        src_vec_loss = self.inter_loss(src_vec, src_normals)
        trg_vec_loss = self.inter_loss(trg_vec, trg_normals)

        return src_vec_loss + trg_vec_loss

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