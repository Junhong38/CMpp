import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # 추가

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

        self.pos_means = []
        self.pos_stds = []
        self.neg_means = []
        self.neg_stds = []

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """

        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius
        
        # Positive/Negative feats_dist 분포 출력
        pos_dists = feats_dist[pos_mask]
        neg_dists = feats_dist[neg_mask]
        if pos_dists.numel() > 0 and neg_dists.numel() > 0:
            print(f"[CircleLoss] pos_dists: mean={pos_dists.mean():.4f}, std={pos_dists.std():.4f}, min={pos_dists.min():.4f}, max={pos_dists.max():.4f}")
            print(f"[CircleLoss] neg_dists: mean={neg_dists.mean():.4f}, std={neg_dists.std():.4f}, min={neg_dists.min():.4f}, max={neg_dists.max():.4f}")
            # 값 저장
            self.pos_means.append(pos_dists.mean().item())
            self.pos_stds.append(pos_dists.std().item())
            self.neg_means.append(neg_dists.mean().item())
            self.neg_stds.append(neg_dists.std().item())
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

    def save_stats_figure(self, save_path='circle_loss_stats.png'):
        """학습 종료 후 호출해서 figure로 저장"""
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Positive Dists')
        plt.plot(self.pos_means, label='mean')
        plt.plot(self.pos_stds, label='std')
        plt.legend()
        plt.subplot(1,2,2)
        plt.title('Negative Dists')
        plt.plot(self.neg_means, label='mean')
        plt.plot(self.neg_stds, label='std')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class CircleLoss_changed(nn.Module):

    def __init__(self, log_scale=24, pos_optimal=0.1, neg_optimal=1.4):
        super(CircleLoss_changed,self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = pos_optimal - 0.05
        self.neg_margin = neg_optimal + 0.05
        
        self.pos_radius = 0.018
        self.safe_radius = 0.03

        # self.max_points = 128

        self.pos_means = []
        self.pos_stds = []
        self.neg_means = []
        self.neg_stds = []

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """

        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        # Positive/Negative feats_dist 분포 출력
        pos_dists = feats_dist[pos_mask]
        neg_dists = feats_dist[neg_mask]
        if pos_dists.numel() > 0 and neg_dists.numel() > 0:
            print(f"[CircleLoss] pos_dists: mean={pos_dists.mean():.4f}, std={pos_dists.std():.4f}, min={pos_dists.min():.4f}, max={pos_dists.max():.4f}")
            print(f"[CircleLoss] neg_dists: mean={neg_dists.mean():.4f}, std={neg_dists.std():.4f}, min={neg_dists.min():.4f}, max={neg_dists.max():.4f}")
            # 값 저장
            self.pos_means.append(pos_dists.mean().item())
            self.pos_stds.append(pos_dists.std().item())
            self.neg_means.append(neg_dists.mean().item())
            self.neg_stds.append(neg_dists.std().item())

        if pos_mask.sum() > neg_mask.sum():
            breakpoint()
        
        # sample the neg_mask to match proportions
        neg_indices = neg_mask.nonzero(as_tuple=False)
        neg_nonsampled = neg_indices[torch.randperm(neg_indices.size(0))[pos_mask.sum():]]
        neg_mask[neg_nonsampled[:,0], neg_nonsampled[:,1]] = False

        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0))
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0))
        # row_sel = (pos_mask.sum(-1)>0)
        # col_sel = (pos_mask.sum(-2)>0)

        # get alpha for both positive and negative pairs
        # print(f"pos_mask : {pos_mask.sum()}")
        # print(f"feats_dist : {feats_dist.max()}")
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight)

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        # P_distribution = (feats_dist * pos_mask)[row_sel, col_sel]
        # N_distribution = (feats_dist * neg_mask)[row_sel, col_sel]

        # print(f"pos_weight : {pos_weight.max()}")
        # lse_pos_row = torch.logsumexp(self.log_scale * pos_weight, dim=-1)
        # lse_pos_col = torch.logsumexp(self.log_scale * pos_weight, dim=-2)

        # lse_neg_row = torch.logsumexp(self.log_scale * neg_weight, dim=-1)
        # lse_neg_col = torch.logsumexp(self.log_scale * neg_weight, dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale
        # loss_row = F.softplus(lse_pos_row)/self.log_scale
        # loss_col = F.softplus(lse_pos_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        P_margin = (lse_pos_row[row_sel].mean().detach().cpu() + lse_pos_col[col_sel].mean().detach().cpu()) / 2
        # N_margin = (lse_neg_row[row_sel].mean().detach().cpu() + lse_neg_col[col_sel].mean().detach().cpu()) / 2

        return circle_loss

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence):
        if len(correspondence) == 0:
            print('[circle loss] No correspondence!')
            return torch.tensor(0.).to(src_feats.device)

        # Get coordinate distance
        coords_dist = torch.sqrt(torch.clamp(torch.sum((src_pcd[:, None, :] - tgt_pcd[None, :, :]) ** 2, dim=-1), min=0.0))
        # breakpoint()

        # Get feature distance (from GeoTransformer Implementation)
        src_feats = F.normalize(src_feats.squeeze(0), p=2, dim=-1)
        tgt_feats = F.normalize(tgt_feats.squeeze(0), p=2, dim=-1)
        if torch.isnan(src_feats).any() or torch.isnan(tgt_feats).any():
            print("NaN detected in features!")
            src_feats = torch.nan_to_num(src_feats)
            tgt_feats = torch.nan_to_num(tgt_feats)
        dot = torch.einsum('x d, y d -> x y', src_feats, tgt_feats)
        dot = torch.clamp(dot, min=-1.0, max=1.0)
        value = 2.0 - 2.0 * dot
        assert (value >= 0).all(), f"Negative value detected in sqrt input: min={value.min()}"
        feats_dist = torch.sqrt(torch.clamp(value, min=0.0))
        
        # Calculate circle loss and feature matching recall (FMR)
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)
        # if torch.isnan(circle_loss[0]):
        #     print('[circle loss] NaN detected!')
        #     circle_loss = (torch.tensor(0.).to(src_feats.device), torch.tensor(0.).to(src_feats.device), None)
        
        if circle_loss != circle_loss:
            # print('[circle loss] NaN detected!')
            circle_loss = torch.tensor(0.).to(src_feats.device)
            
        return circle_loss
    
    def save_stats_figure(self, save_path='circle_loss_stats.png'):
        """학습 종료 후 호출해서 figure로 저장"""
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Positive Dists')
        plt.plot(self.pos_means, label='mean')
        plt.plot(self.pos_stds, label='std')
        plt.legend()
        plt.subplot(1,2,2)
        plt.title('Negative Dists')
        plt.plot(self.neg_means, label='mean')
        plt.plot(self.neg_stds, label='std')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

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
    
class PointMatchingLoss_changed(nn.Module):
    def __init__(self):
        super(PointMatchingLoss_changed, self).__init__()
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

class OrientationLoss(nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()
        self.eps = 1e-7

    def inter_loss(self, src_ori, trg_ori, correspondence, src_gt_rot, trg_gt_rot):
        src_ori = src_ori[:, correspondence[:,0]] 
        trg_ori = trg_ori[:, correspondence[:,1]]

        src_ori = torch.matmul(src_ori, src_gt_rot)
        trg_ori = torch.matmul(trg_ori, trg_gt_rot)

        diff = src_ori - trg_ori
        f_norm = torch.norm(diff, p='fro', dim=(2, 3))
        inter_loss = torch.mean(f_norm)
        
        return inter_loss

    def forward(self, src_ori, trg_ori, correspondence, gt_rot):
        if len(correspondence) == 0:
            return torch.tensor(0.).to(src_ori.device)

        src_gt_rot = gt_rot[0]
        trg_gt_rot = gt_rot[1]
        ori_loss = self.inter_loss(src_ori, trg_ori, correspondence, src_gt_rot, trg_gt_rot)

        return ori_loss

class OrientationLoss_changed(nn.Module):
    def __init__(self):
        super(OrientationLoss_changed, self).__init__()
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
        return torch.max(torch.relu(angle - 0.9*2*torch.pi) ** 2)

    def forward(self, src_ori, trg_ori, correspondence, gt_normals):
        if len(correspondence) == 0:
            return torch.tensor(0.).to(src_ori.device)

        src_ori = src_ori.squeeze() # (1, N, 1, 3) --> (N, 3)
        trg_ori = trg_ori.squeeze() # (1, M, 1, 3) --> (M, 3)

        src_angle = torch.linalg.norm(src_ori, dim=-1, keepdim=True)
        trg_angle = torch.linalg.norm(trg_ori, dim=-1, keepdim=True)
        # trg_angle = 2*torch.pi - torch.linalg.norm(trg_ori, dim=-1, keepdim=True)
        # print(f"src_MIN : {src_angle.min()} | src_MAX : {src_angle.max()}")
        # print(f"trg_MIN : {trg_angle.min()} | trg_MAX : {trg_angle.max()}")

        src_axis = src_ori / (src_angle + self.eps)
        trg_axis = trg_ori / (trg_angle + self.eps)

        src_normals = gt_normals[0].squeeze() # (1, N, 3) --> (N, 3)
        trg_normals = gt_normals[1].squeeze() # (1, M, 3) --> (M, 3)

        src_axis_loss = self.axis_loss(src_axis, src_normals)
        trg_axis_loss = self.axis_loss(trg_axis, trg_normals)

        # angle_loss = self.angle_loss(src_angle, trg_angle, correspondence, src_normals, trg_normals)

        reg_loss = self.angle_regularization(src_angle) + self.angle_regularization(trg_angle)
        # reg_loss = self.angle_regularization(src_angle) + self.angle_regularization(2*torch.pi - trg_angle)
        print(f"reg_loss : {reg_loss}")

        # return (src_axis_loss + trg_axis_loss) / 2 + angle_loss + self.reg_weight * reg_loss
        return src_axis_loss + trg_axis_loss + reg_loss

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