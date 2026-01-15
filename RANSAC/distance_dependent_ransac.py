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
        src_normal: torch.Tensor,
        trg_normal: torch.Tensor,
        scores: torch.Tensor,
        score_threshold: float,
        num_iters: int = 100,
        threshold: float = 0.01,
        normal_threshold: float = 0.0,
        matching_choice: str = 'one-to-one', # if sampling is the "same"
        # matching_choice: str = 'many-to-many' # if sampling is just uniform
        strong_normal_threshold = 0.0
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
    score_mask = (scores >= score_threshold).to(device)

    N = src_corr_pcd.shape[0]
    max_RANSAC_score = -1
    best_inliers = None
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
        
        transformed_src = _transform_points(src_pcd, rotation, translation)
        dist_mat = torch.cdist(transformed_src, trg_pcd)
        inliers = dist_mat < threshold

        if inliers.shape != score_mask.shape:
            raise ValueError("Score mask shape does not match distance matrix.")
        inliers &= score_mask

        rotated_normals = torch.matmul(src_normal, rotation.T.to(src_normal.dtype))
        cos_sim = torch.matmul(rotated_normals, trg_normal.T)
        angle = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
        normal_mask = angle > normal_threshold
        if inliers.shape != normal_mask.shape:
            raise ValueError("Normal mask shape does not match inlier mask.")
        inliers &= normal_mask

        inlier_count = inliers.sum()
        N_ = src_corr_pcd.shape[0]
        lam = 34.7
        mean_inlier_dist = dist_mat[inliers].mean()
        RANSAC_score = (inlier_count/N_ * torch.exp(-lam * mean_inlier_dist)).item()

        total_survived_dist = dist_mat.sum().item()
        if RANSAC_score > max_RANSAC_score:
            max_RANSAC_score = RANSAC_score
            best_inliers = inliers
            best_rotation = rotation
            best_translation = translation

    if best_inliers is None:
        raise RuntimeError("Failed to estimate a valid transform via RANSAC.")

    # print(f"max_total_score: {max_total_score}")
    # print(f"best_rotation: {best_rotation}")
    # print(f"best_translation: {best_translation}")

    # Optimal Estimation
    strong_distance_threshold = 0.008
    num_iters_for_optimal_estimation = 100
    
    for _ in range(num_iters_for_optimal_estimation):
        transformed_src = _transform_points(src_pcd, best_rotation, best_translation)
        dist_mat = torch.cdist(transformed_src, trg_pcd)
        refined_inliers = dist_mat < strong_distance_threshold
        
        refined_inliers &= score_mask

        rotated_normals = torch.matmul(src_normal, best_rotation.T.to(src_normal.dtype))
        cos_sim = torch.matmul(rotated_normals, trg_normal.T)
        angle = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
        normal_mask = angle > strong_normal_threshold
        if normal_mask.shape != refined_inliers.shape:
            raise ValueError("Normal mask shape does not match refined inlier mask.")
        refined_inliers &= normal_mask

        if torch.equal(best_inliers, refined_inliers):
            break

        correspondences = _select_correspondences(refined_inliers, dist_mat, matching_choice)
        if correspondences.size(0) < 3:
            best_inliers = refined_inliers
            break

        src_indices = correspondences[:, 0]
        trg_indices = correspondences[:, 1]
        src_points = src_pcd.index_select(0, src_indices)
        trg_points = trg_pcd.index_select(0, trg_indices)

        try:
            best_rotation, best_translation = weighted_procrustes(src_points, trg_points, scores[src_indices, trg_indices], return_transform=False)
        except RuntimeError:
            break

        best_inliers = refined_inliers
    return best_rotation, best_translation, best_inliers
