import torch
from typing import Set, Tuple

from RANSAC.utils import _squeeze_leading_dim, _transform_points, _select_correspondences, estimate_rigid_transform
from RANSAC.weighted_procrustes import weighted_procrustes
from RANSAC.best_topk import TopKBest

@torch.no_grad()
def ransac_rigid(
        src_corr_pcd: torch.Tensor,
        trg_corr_pcd: torch.Tensor,
        src_pcd: torch.Tensor,
        trg_pcd: torch.Tensor,
        src_normal: torch.Tensor,
        trg_normal: torch.Tensor,
        scores: torch.Tensor,
        score_threshold: float,
        num_iters: int = 100,
        threshold: float = 0.01,
        normal_threshold: float = 0.0,
        # matching_choice: str = 'one-to-one', # if sampling is the "same"
        matching_choice: str = 'many-to-many', # if sampling is just uniform
        strong_normal_threshold = 0.0,
        use_penetration = False,

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robustly estimate a rigid transform that aligns ``src_pcd`` to ``trg_pcd``.

    Args:
        src_corr_pcd: Source correspondence points of shape (N, 3) or (1, N, 3).
        trg_corr_pcd: Target correspondence points of shape (N, 3) or (1, N, 3).
        src_pcd: Full source point cloud of shape (M, 3) or (1, M, 3).
        trg_pcd: Full target point cloud of shape (K, 3) or (1, K, 3).
        src_gt_normal: Source normals aligned with ``src_pcd``.
        trg_gt_normal: Target normals aligned with ``trg_pcd``.
        scores: Similarity scores with the same spatial shape as the distance matrix.
        score_threshold: Minimum score required for an inlier.
        num_iters: Number of RANSAC iterations.
        threshold: Distance threshold used during the RANSAC stage.
        gt_normal_threshold: Cosine similarity threshold for normal filtering.
        matching_choice: Strategy to turn the inlier mask into correspondences.

    Returns:
        rotation, translation, final inlier mask.
    """
    src_corr_pcd = _squeeze_leading_dim(src_corr_pcd)
    trg_corr_pcd = _squeeze_leading_dim(trg_corr_pcd)
    src_pcd = _squeeze_leading_dim(src_pcd)
    trg_pcd = _squeeze_leading_dim(trg_pcd)
    src_normal = _squeeze_leading_dim(src_normal)
    trg_normal = _squeeze_leading_dim(trg_normal)
    scores = _squeeze_leading_dim(scores)


    if src_corr_pcd.shape[0] < 3:
        raise ValueError("At least three correspondences are required for RANSAC.")

    device = src_corr_pcd.device

    N = src_corr_pcd.shape[0]
    max_total_score = - torch.inf
    best_score = None
    best_rotation = None
    best_translation = None

    best_inliers = None
    score_mask = (scores > score_threshold).to(device)

    unique_src_pcd = torch.unique(src_corr_pcd, dim=0)
    unique_trg_pcd = torch.unique(trg_corr_pcd, dim=0)

    top5 = TopKBest(k=int(num_iters*5/100))


    # RANSAC Iterations
    for _ in range(num_iters):
        for _ in range(10000):
            indices = torch.randperm(N, device=device)[:3]
            src_sample = src_corr_pcd.index_select(0, indices)
            trg_sample = trg_corr_pcd.index_select(0, indices)
            if matching_choice == 'many-to-one':
                break
            
            if len(unique_src_pcd) < 3:
                if torch.unique(src_sample, dim=0).size(0) == len(unique_src_pcd):
                    break
            if len(unique_trg_pcd) < 3:
                if torch.unique(trg_sample, dim=0).size(0) == len(unique_trg_pcd):
                    break
            else:
                if (torch.unique(src_sample, dim=0).size(0) == src_sample.size(0)) and (torch.unique(trg_sample, dim=0).size(0) == trg_sample.size(0)):
                    break

        try:
            rotation, translation = estimate_rigid_transform(src_sample, trg_sample)
        except RuntimeError:
            continue
        
        temp_scores = scores.clone()
        temp_scores.masked_fill_(temp_scores.abs() < 1e-9, -scores.max())

        transformed_src = _transform_points(src_pcd, rotation, translation)
        rotated_normals = torch.matmul(src_normal, rotation.T.to(src_normal.dtype))

        if use_penetration:
            if penetration_checking(transformed_src, trg_pcd, rotated_normals, trg_normal, threshold):
                continue

        dist_mat = torch.cdist(transformed_src, trg_pcd)
        distance_mask = dist_mat < threshold

        if temp_scores.shape != distance_mask.shape:
            raise ValueError("Distance mask shape does not match shape of the inlier score.")
        temp_scores *= distance_mask
        inliers = distance_mask

        cos_sim = torch.matmul(rotated_normals, trg_normal.T)
        angle = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
        normal_mask = angle > normal_threshold
        if temp_scores.shape != normal_mask.shape:
            raise ValueError("Normal mask shape does not match shape of the inlier score.")
        temp_scores *= normal_mask
        inliers &= normal_mask

        if inliers.shape != score_mask.shape:
            raise ValueError("Score mask shape does not match inliers shape.")
        inliers &= score_mask

        total_survived_score = temp_scores.sum().item()
        top5.try_add(total_survived_score, rotation, translation, inliers)

    # Optimal Estimation
    strong_distance_threshold = 0.008
    num_iters_for_optimal_estimation = 100
    best_list = top5.get_sorted()[0:1]

    top_rotation = None
    top_translation = None
    top_score = None
    
    for cand in best_list:
        best_score = cand.score
        best_rotation = cand.rotation
        best_translation = cand.translation
        best_inliers = cand.inliers
        for _ in range(num_iters_for_optimal_estimation):
            refined_score = scores.clone()
            refined_score.masked_fill_(refined_score.abs() < 1e-9, -scores.max())

            transformed_src = _transform_points(src_pcd, best_rotation, best_translation)
            dist_mat = torch.cdist(transformed_src, trg_pcd)
            distance_mask = dist_mat < strong_distance_threshold
            if distance_mask.shape != refined_score.shape:
                raise ValueError("Distance mask shape does not match shape of the refined score.")
            refined_score *= distance_mask
            refined_inliers = distance_mask

            rotated_normals = torch.matmul(src_normal, best_rotation.T.to(src_normal.dtype))
            cos_sim = torch.matmul(rotated_normals, trg_normal.T)
            angle = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
            normal_mask = angle > strong_normal_threshold
            if normal_mask.shape != refined_score.shape:
                raise ValueError("Normal mask shape does not match shape of the refined score.")
            refined_score *= normal_mask
            refined_inliers &= normal_mask

            if refined_inliers.shape != score_mask.shape:
                raise ValueError("Score mask shape does not match shape of the refined inlier.")
            refined_inliers &= score_mask

            if use_penetration:
                if penetration_checking(transformed_src, trg_pcd, rotated_normals, trg_normal, strong_distance_threshold):
                    break
            if best_score == refined_score.sum():
                break

            correspondences = _select_correspondences(refined_score, dist_mat, matching_choice)
            if correspondences.size(0) < 3:
                best_score = refined_score.sum().item()
                break

            src_indices = correspondences[:, 0]
            trg_indices = correspondences[:, 1]
            src_points = src_pcd.index_select(0, src_indices)
            trg_points = trg_pcd.index_select(0, trg_indices)

            try:
                best_rotation, best_translation = weighted_procrustes(src_points, trg_points, refined_score[src_indices, trg_indices], return_transform=False)
            except RuntimeError:
                break

            best_score = refined_score.sum().item()
        
        if best_score > max_total_score:
            top_rotation = best_rotation
            top_translation = best_translation
            top_score = best_score

    return top_rotation, top_translation, top_score



def penetration_checking(
    src_pcd: torch.Tensor, # (M, 3)
    trg_pcd: torch.Tensor, # (N, 3)
    src_normals: torch.Tensor, # (M, 3)
    trg_normals: torch.Tensor, # (N, 3)
    dist_th: float = 0.018,
    normal_buffer: int = 80,
    penetration_buffer: int = 0
) -> bool:
    import math

    diff = src_pcd[:, None, :] - trg_pcd[None, :, :] # (M,N,3) = p - q
    v2 = diff.square().sum(dim=-1)
    dist_ok = v2 < (dist_th * dist_th)

    # normals within 90deg: n_p · n_q > 0
    normal_cos_thres = math.cos(math.radians(90.0 - float(normal_buffer)))
    np_dot_nq = (src_normals[:, None, :] * trg_normals[None, :, :]).sum(dim=-1) > normal_cos_thres

    # penetration angle buffer: angle(?, normal) > 90 + buf
    s = math.sin(math.radians(float(penetration_buffer)))
    s2 = s * s

    # (q->p) vs n_q : angle > 90+buf  <=>  (p-q)·n_q / ||p-q|| < -sin(buf)
    dot_q = (diff * trg_normals[None, :, :]).sum(dim=-1)  # (M,N)
    cond_q = (dot_q < 0) & (dot_q.square() > s2 * v2)

    # (p->q) vs n_p : angle > 90+buf  <=>  (q-p)·n_p / ||q-p|| < -sin(buf)
    # q-p = -diff  =>  (-diff)·n_p < -sin*||diff||  <=>  diff·n_p > sin*||diff||
    dot_p = (diff * src_normals[:, None, :]).sum(dim=-1)  # (M,N)
    cond_p = (dot_p > 0) & (dot_p.square() > s2 * v2)

    dir_ok = cond_q | cond_p

    ok = dist_ok & np_dot_nq & dir_ok
    return bool(ok.any())