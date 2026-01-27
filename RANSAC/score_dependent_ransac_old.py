import torch
from typing import Set, Tuple

from RANSAC.utils import _squeeze_leading_dim, _transform_points, _select_correspondences, estimate_rigid_transform
from RANSAC.weighted_procrustes import weighted_procrustes

@torch.no_grad()
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
        normal_threshold: float = 90,
        strong_normal_threshold: float = 90,
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
    max_total_score = - torch.inf
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
        angle = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
        normal_mask = angle > normal_threshold
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
    num_iters_for_optimal_estimation = 100
    
    for _ in range(num_iters_for_optimal_estimation):
        refined_score = scores.clone()

        transformed_src = _transform_points(src_pcd, best_rotation, best_translation)
        dist_mat = torch.cdist(transformed_src, trg_pcd)
        distance_mask = dist_mat < strong_distance_threshold
        refined_score *= distance_mask

        rotated_normals = torch.matmul(src_gt_normal, best_rotation.T.to(src_gt_normal.dtype))
        cos_sim = torch.matmul(rotated_normals, trg_gt_normal.T)
        angle = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
        normal_mask = angle > strong_normal_threshold
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

    return best_rotation, best_translation, best_score, 0, 0