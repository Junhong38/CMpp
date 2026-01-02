import torch
from typing import Set, Tuple

from RANSAC.utils import _squeeze_leading_dim, _transform_points, _select_correspondences, estimate_rigid_transform


def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=True,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)
    
    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    from torch_batch_svd import svd
    try: U, _, V = svd(H)
    except: 
        print('use torch svd!')
        U, _, V = torch.svd(H.cpu())
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    # eye[:, -1, -1] = torch.sign(torch.det((V @ Ut).to(torch.float32)))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t

def ransac_rigid(
        src_corr_pcd: torch.Tensor,
        trg_corr_pcd: torch.Tensor,
        src_pcd: torch.Tensor,
        trg_pcd: torch.Tensor,
        src_gt_normal: torch.Tensor,
        trg_gt_normal: torch.Tensor,
        scores: torch.Tensor,
        score_threshold: float,
        num_iters: int = 100,
        threshold: float = 0.01,
        gt_normal_threshold: float = -0.7,
        matching_choice: str = 'many-to-many' #'one-to-one'
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
    src_gt_normal = _squeeze_leading_dim(src_gt_normal)
    trg_gt_normal = _squeeze_leading_dim(trg_gt_normal)
    scores = _squeeze_leading_dim(scores)

    if src_corr_pcd.shape[0] < 3:
        raise ValueError("At least three correspondences are required for RANSAC.")

    device = src_corr_pcd.device

    N = src_corr_pcd.shape[0]
    max_total_score = 0.0
    best_score = None
    best_rotation = None
    best_translation = None

    unique_src_pcd = torch.unique(src_corr_pcd, dim=0)


    # RANSAC Iterations
    for _ in range(num_iters):
        while True:
            indices = torch.randperm(N, device=device)[:3]
            src_sample = src_corr_pcd.index_select(0, indices)
            trg_sample = trg_corr_pcd.index_select(0, indices)
            if matching_choice == 'many-to-one':
                break
            
            if len(unique_src_pcd) < 3:
                if torch.unique(src_sample, dim=0).size(0) == len(unique_src_pcd):
                    break
            else:
                if torch.unique(src_sample, dim=0).size(0) == src_sample.size(0):
                    break

        try:
            rotation, translation = estimate_rigid_transform(src_sample, trg_sample)
        except RuntimeError:
            continue
        
        temp_scores = scores.clone()

        transformed_src = _transform_points(src_pcd, rotation, translation)
        dist_mat = torch.cdist(transformed_src, trg_pcd)
        distance_mask = dist_mat < threshold

        if temp_scores.shape != distance_mask.shape:
            raise ValueError("Score mask shape does not match distance matrix.")
        temp_scores *= distance_mask

        rotated_normals = torch.matmul(src_gt_normal, rotation.T.to(src_gt_normal.dtype))
        cos_sim = torch.matmul(rotated_normals, trg_gt_normal.T)
        normal_mask = cos_sim < gt_normal_threshold
        if temp_scores.shape != normal_mask.shape:
            raise ValueError("Normal mask shape does not match inlier mask.")
        temp_scores *= normal_mask

        total_survived_score = temp_scores.sum().item()
        if total_survived_score > max_total_score:
            max_total_score = total_survived_score
            best_score = temp_scores
            best_rotation = rotation
            best_translation = translation

    if best_score is None:
        raise RuntimeError("Failed to estimate a valid transform via RANSAC.")

    # print(f"max_total_score: {max_total_score}")
    # print(f"best_rotation: {best_rotation}")
    # print(f"best_translation: {best_translation}")

    # Optimal Estimation
    strong_distance_threshold = 0.008
    strong_normal_threshold = -0.9
    num_iters_for_optimal_estimation = 100
    
    for _ in range(num_iters_for_optimal_estimation):
        refined_score = scores.clone()

        transformed_src = _transform_points(src_pcd, best_rotation, best_translation)
        dist_mat = torch.cdist(transformed_src, trg_pcd)
        distance_mask = dist_mat < strong_distance_threshold
        refined_score *= distance_mask

        rotated_normals = torch.matmul(src_gt_normal, best_rotation.T.to(src_gt_normal.dtype))
        cos_sim = torch.matmul(rotated_normals, trg_gt_normal.T)
        normal_mask = cos_sim < strong_normal_threshold
        if normal_mask.shape != refined_score.shape:
            raise ValueError("Normal mask shape does not match refined inlier mask.")
        refined_score *= normal_mask

        if torch.equal(best_score, refined_score):
            break

        correspondences = _select_correspondences(refined_score, dist_mat, matching_choice)
        if correspondences.size(0) < 3:
            best_score = refined_score
            break

        src_indices = correspondences[:, 0]
        trg_indices = correspondences[:, 1]
        src_points = src_pcd.index_select(0, src_indices)
        trg_points = trg_pcd.index_select(0, trg_indices)

        try:
            best_rotation, best_translation = weighted_procrustes(src_points, trg_points, refined_score[src_indices, trg_indices], return_transform=False)
        except RuntimeError:
            break

        best_score = refined_score

    return best_rotation, best_translation, best_score
