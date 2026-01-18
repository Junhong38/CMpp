import torch
from scipy.spatial.transform import Rotation
from chamfer_distance import ChamferDistance as chamfer_dist
from pytorch3d.ops import iterative_closest_point


def correspondence_distance(assm1, assm2, scaling=100):
    """
    Args:
        assm1 (torch.Tensor): (N, 3)
        assm2 (torch.Tensor): (M, 3)
        scaling (int, optional): Scaling factor for CD. Defaults to 100.

    Returns:
        corr_dist (torch.Tensor): (1)
    """
    corr_dist = (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling
    return corr_dist


def chamfer_distance(assm1, assm2, scaling=1000):
    """
    Args:
        assm1 (torch.Tensor): (N, 3)
        assm2 (torch.Tensor): (M, 3)
        scaling (int, optional): Scaling factor for CD. Defaults to 1000.

    Returns:
        cd (torch.Tensor): (1)
    """
    chd = chamfer_dist()
    dist1, dist2, idx1, idx2 = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
    cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling
    return cd


def transformation_error(trnsf1, trnsf2, trmse_scaling=100):
    """
    Args:
        trnsf1 (tuple): (3, 3), (3)
        trnsf2 (tuple): (3, 3), (3)
        trmse_scaling (int, optional): Scaling factor for TRMSE. Defaults to 100.

    Returns:
        rrmse (torch.Tensor): (1)
        trmse (torch.Tensor): (1)
    """
    rotat1, trans1 = [trnsf1[0]], [trnsf1[1]]
    rotat2, trans2 = [trnsf2[0]], [trnsf2[1]]
    
    rrmse, trmse = 0., 0.
    for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
        r1_deg = torch.tensor(Rotation.from_matrix(r1.cpu()).as_euler('xyz', degrees=True))
        r2_deg = torch.tensor(Rotation.from_matrix(r2.cpu()).as_euler('xyz', degrees=True))
        diff1 = (r1_deg - r2_deg).abs()
        diff2 = 360. - (r1_deg - r2_deg).abs()
        diff = torch.minimum(diff1, diff2)
        rrmse += diff.pow(2).mean().pow(0.5)
        trmse += (t1 - t2).pow(2).mean().pow(0.5) * trmse_scaling
    
    # div = len(rotat1) if multi_part else 1
    div = 1
    return (rrmse / div).to(trmse.device), trmse / div


def transformation_error_geodesic(trnsf1, trnsf2, trmse_scaling=100):
    """
    Args:
        trnsf1 (tuple): (3, 3), (3)
        trnsf2 (tuple): (3, 3), (3)
        trmse_scaling (int, optional): Scaling factor for TRMSE. Defaults to 100.

    Returns:
        rrmse (torch.Tensor): (1)
        trmse (torch.Tensor): (1)
    """
    rotat1, trans1 = [trnsf1[0]], [trnsf1[1]]
    rotat2, trans2 = [trnsf2[0]], [trnsf2[1]]
    
    rrmse_geo, trmse_geo = 0., 0.
    for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
        # pred_rotat^T @ gt_rotat
        relative_rotat = r1 @ r2.T

        # tr(R) = 1 + 2cos(θ) -> θ = acos((tr(R) - 1) / 2), torch.acos is in radian, so we need to convert to degree
        rrmse_geo += torch.rad2deg(torch.acos(torch.clamp(0.5 * (torch.trace(relative_rotat) - 1.0), -1.0, 1.0)))
        trmse_geo += torch.norm(t1 - t2) * trmse_scaling
    
    div = 1
    return (rrmse_geo / div).to(trmse_geo.device), trmse_geo / div


def transformation_error_RPFver(pcds_pred, pcds_grtr, scaling=100):
    """
    Args:
        pcds_pred (list): [(N, 3), (M, 3)]
        pcds_grtr (list): [(N, 3), (M, 3)]
        scaling (int, optional): Scaling factor for TRMSE. Defaults to 100. 

    Returns:
        rrmse (torch.Tensor): (1)
        trmse (torch.Tensor): (1)
    """
    
    num_parts = len(pcds_grtr)
    rot_errors = torch.zeros(num_parts, device=pcds_grtr[0].device) # (K), rotation error
    trans_errors = torch.zeros(num_parts, device=pcds_grtr[0].device)  # (K), translation error

    for p in range(num_parts):
        pcd_pred = pcds_pred[p].unsqueeze(0) # (N, 3) -> (1, N, 3)
        pcd_grtr = pcds_grtr[p].unsqueeze(0) # (N, 3) -> (1, N, 3)
        assert pcd_pred.shape == pcd_grtr.shape, f"Point clouds should be same size, but got {pcd_pred.shape} and {pcd_grtr.shape}"

        # ICP algorithm
        error = iterative_closest_point(pcd_grtr, pcd_pred).RTs

        
        # tr(R) = 1 + 2cos(θ) -> θ = acos((tr(R) - 1) / 2), torch.acos is in radian, so we need to convert to degree
        rot_errors[p] = torch.rad2deg(torch.acos(torch.clamp(0.5 * (torch.trace(error.R[0]) - 1.0), -1.0, 1.0)))
        trans_errors[p] = torch.norm(error.T[0]) * scaling

    div = len(pcds_grtr)
    return rot_errors.sum() / div, trans_errors.sum() / div 


def normal_error(in_dict, out_dict, success_criterion_in_degree=10):
    """
    Args:
        in_dict (dict): it is same as forward_pass. From in_dict, only need gt_normals, which is torch.Tensor: (B, N+M, 3)
        out_dict (dict): it is same as forward_pass. From out_dict, only need src_ori and trg_ori, which are torch.Tensor: (B, N+M, 3, 3) and torch.Tensor: (B, N+M, 3, 3)
        success_criterion_in_degree (int, optional): Success criterion in degree. Defaults to 10.

    Returns:
        normal_error (torch.Tensor): (1)
    """
    pred_oris = out_dict['oris'] #  (B, N+M, 3, 3)
    gt_normals = in_dict['gt_normals'] # (B, N+M, 3)

    pred_normals = pred_oris[:,:,0,:] # (B, N+M, 3)

    cosine_similarity = torch.clamp(torch.nn.functional.cosine_similarity(pred_normals, gt_normals, dim=-1), min=-1, max=1) # (B, N+M, )
    theta_deg = torch.rad2deg(torch.acos(cosine_similarity)).reshape(-1) # (B, N+M, ) -> (B*(N+M), )
    
    normal_error = theta_deg.mean()
    normal_error_hist = torch.histogram(theta_deg.cpu(), bins=90, range=(0, 180)) # Total 180 degrees, so we choose 90 bins

    success_mask = theta_deg <= success_criterion_in_degree
    success_count = success_mask.sum()
    total_count = theta_deg.shape[0]
    success_rate = success_count / total_count

    return normal_error, normal_error_hist, success_rate


def calculate_recall(matching_scores_drop, gt_corr, topks=[1,5,10,20]):
    """
    Calculate recall of matching scores

    Args:
        matching_scores_drop (torch.Tensor): (N, M)
        gt_corr (torch.Tensor): (P, 2)
        topks (list, optional): Recall@1, Recall@5, Recall@10, Recall@20.

    Returns:
        matching_recall (torch.Tensor): (1)
    """
    _N, _M = matching_scores_drop.shape # (N, M)

    result_dict = dict()

    if len(gt_corr) == 0:
        for topk in topks:
            result_dict[f"recall@{str(topk)}"] = 0.0
        return result_dict

    correspondence_mask = torch.zeros((_N, _M), device=matching_scores_drop.device)
    correspondence_mask[gt_corr[:,0], gt_corr[:,1]] = True
    correspondence_mask_src = correspondence_mask.sum(dim=-1) > 0 # (N, M) -> N
    correspondence_mask_trg = correspondence_mask.sum(dim=-2) > 0 # (N, M) -> M
    
    for topk in topks:
        ## Recall from src
        _, topk_inds_src = torch.topk(matching_scores_drop[correspondence_mask_src, :], k=topk, dim=-1) # (N, M) -> (gt_N, M) -> (gt_N, topk)
        topk_mask_src = torch.zeros((_N, _M), device=matching_scores_drop.device) # (N, M)
        for i, _ in enumerate(range(topk_inds_src.shape[-1])): # for i in range(topk)
            # [all gt_N, ith topk from gt_src]
            topk_mask_src[torch.nonzero(correspondence_mask_src)[:, 0], topk_inds_src[:, i]] = True

        # (N, M) -> N
        is_success_src = (topk_mask_src * correspondence_mask).sum(dim=-1) > 0
        recall_src = is_success_src[correspondence_mask_src].sum() / correspondence_mask_src.sum()

        ## Recall from trg
        _, topk_inds_trg = torch.topk(matching_scores_drop[:, correspondence_mask_trg], k=topk, dim=-2) # (N, M) -> (N, gt_M) -> (topk, gt_M)
        topk_mask_trg = torch.zeros((_N, _M), device=matching_scores_drop.device) # (N, M)
        for i, _ in enumerate(range(topk_inds_trg.shape[-2])): # for i in range(topk)
            # [ith topk from gt_trg, all gt_N]
            topk_mask_trg[topk_inds_trg[i, :], torch.nonzero(correspondence_mask_trg)[:, 0]] = True

        # (N, M) -> M
        is_success_trg = (topk_mask_trg * correspondence_mask).sum(dim=-2) > 0
        recall_trg = is_success_trg[correspondence_mask_trg].sum() / correspondence_mask_trg.sum()
        
        recall_dot_k = (recall_src + recall_trg) / 2

        ## Logging results
        result_dict[f"recall@{str(topk)}"] = recall_dot_k
    
    return result_dict


def calculate_ratio_of_gt_among_topk_scores(src_pcd_raw, trg_pcd_raw, matching_scores, topk=128, pos_radius=0.018):
    """Calculate ratio of GT among topk scores

    Args:
        src_pcd (torch.Tensor): (N, 3)
        trg_pcd (torch.Tensor): (M, 3)
        matching_scores (torch.Tensor): (N, M)
        topk (int, optional): Topk value for matching. Defaults to 128.

    Returns:
        ratio_of_gt_among_topk_scores (torch.Tensor): (1)
    """
    # Calculate distance between source and target points, and check if it is within the positive radius
    corr_dist = torch.cdist(src_pcd_raw, trg_pcd_raw, p=2) # (N, M)
    pos_mask = corr_dist < pos_radius # (N, M)

    # Find pairs that have topk scores
    topk_scores = torch.topk(matching_scores.reshape(-1), k=topk, dim=-1)[0] # (N*M) -> (topk)
    kth_biggest_score = topk_scores[-1] # (topk) -> (1, )
    topk_mask = matching_scores >= kth_biggest_score # (N, M)

    # Calculate ratio of GT among topk scores
    ratio_of_gt_among_topk_scores = torch.logical_and(topk_mask, pos_mask).sum() / topk_mask.sum() # (N, M) -> (1, )
    return ratio_of_gt_among_topk_scores


def calculate_accuracy_of_seg_results(seg_results, positive_mask):
    """
    Args:
        seg_results (torch.Tensor): (N+M)
        target (torch.Tensor): (N, M)
    Returns:
        accuracy (float): accuracy of segmentation results
    """
    seg_pred = seg_results > 0.5

    src_part_gt = positive_mask.any(dim=-1) # (N, )
    trg_part_gt = positive_mask.any(dim=-2) # (M, )
    total_gt = torch.concat([src_part_gt, trg_part_gt], dim=0) # (N+M, )

    intersection = torch.logical_and(seg_pred, total_gt) # (N+M, )

    sum_of_total_gt = total_gt.sum()
    sum_of_seg_pred = seg_pred.sum()
    sum_of_intersection = intersection.sum()
    seg_recall = sum_of_intersection / sum_of_total_gt if sum_of_total_gt > 0 else torch.tensor(0.0, device=seg_results.device) # Among all gt points, how many points are covered by the predicted points
    seg_precision = sum_of_intersection / sum_of_seg_pred if sum_of_seg_pred > 0 else torch.tensor(0.0, device=seg_results.device) # Among all predicted points, how many points are correctly predicted
    return seg_recall, seg_precision

