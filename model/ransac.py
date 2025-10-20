from typing import Set, Tuple

import torch


def _squeeze_leading_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Remove a leading singleton batch dimension if present."""
    if tensor is None:
        return None
    if tensor.dim() >= 2 and tensor.size(0) == 1:
        return tensor.squeeze(0)
    return tensor


def _transform_points(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Apply the rigid transform defined by rotation and translation."""
    return points @ rotation.T + translation


def _select_correspondences(inlier_mask: torch.Tensor, dist_mat: torch.Tensor, matching_choice: str) -> torch.Tensor:
    """Select correspondence indices according to the matching strategy."""
    rows, cols = torch.nonzero(inlier_mask, as_tuple=True)
    if rows.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=inlier_mask.device)

    if matching_choice == 'many-to-many':
        return torch.stack((rows, cols), dim=1)

    dists = dist_mat[rows, cols]

    if matching_choice == 'many-to-one':
        pairs = []
        for col in cols.unique(sorted=True).tolist():
            mask = cols == col
            col_rows = rows[mask]
            col_dists = dists[mask]
            best_idx = torch.argmin(col_dists)
            pairs.append((col_rows[best_idx].item(), col))
        return torch.tensor(pairs, dtype=torch.long, device=inlier_mask.device)

    if matching_choice == 'one-to-one':
        order = torch.argsort(dists)
        used_src: Set[int] = set()
        used_trg: Set[int] = set()
        pairs = []
        for idx in order.tolist():
            src_idx = rows[idx].item()
            trg_idx = cols[idx].item()
            if src_idx in used_src or trg_idx in used_trg:
                continue
            used_src.add(src_idx)
            used_trg.add(trg_idx)
            pairs.append((src_idx, trg_idx))
        if not pairs:
            return torch.zeros((0, 2), dtype=torch.long, device=inlier_mask.device)
        return torch.tensor(pairs, dtype=torch.long, device=inlier_mask.device)

    raise ValueError(f"Unknown matching choice: {matching_choice}")


def estimate_rigid_transform(source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the rigid transform that brings ``target`` onto ``source``.

    Args:
        source: Tensor of shape (N, 3), reference points.
        target: Tensor of shape (N, 3), points to be transformed.

    Returns:
        rotation: Tensor of shape (3, 3).
        translation: Tensor of shape (3,).
    """
    if source.shape != target.shape:
        raise ValueError(f"Point sets must share a shape. Got {source.shape} and {target.shape}.")
    if source.numel() == 0:
        raise ValueError("At least one correspondence is required.")

    centroid_source = source.mean(dim=0)
    centroid_target = target.mean(dim=0)

    source_centered = source - centroid_source
    target_centered = target - centroid_target

    h_matrix = target_centered.T @ source_centered
    u, _, v = torch.linalg.svd(h_matrix)
    rotation = v.T @ u.T

    if torch.det(rotation) < 0:
        v[-1, :] *= -1
        rotation = v.T @ u.T

    translation = centroid_source - rotation @ centroid_target
    return rotation, translation


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
        matching_choice: str = 'one-to-one',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robustly estimate a rigid transform that aligns ``trg_pcd`` to ``src_pcd``.

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
    score_mask = (scores >= score_threshold).to(device)

    N = src_corr_pcd.shape[0]
    max_inliers = -1
    best_inliers = None
    best_rotation = None
    best_translation = None

    for _ in range(num_iters):
        while True:
            indices = torch.randperm(N, device=device)[:3]
            src_sample = src_corr_pcd.index_select(0, indices)
            trg_sample = trg_corr_pcd.index_select(0, indices)
            if matching_choice == 'many-to-one':
                break
            if torch.unique(src_sample, dim=0).size(0) == src_sample.size(0):
                break

        try:
            rotation, translation = estimate_rigid_transform(src_sample, trg_sample)
        except RuntimeError:
            continue

        transformed_trg = _transform_points(trg_pcd, rotation, translation)
        dist_mat = torch.cdist(src_pcd, transformed_trg)
        inliers = dist_mat < threshold

        if inliers.shape != score_mask.shape:
            raise ValueError("Score mask shape does not match distance matrix.")
        inliers &= score_mask

        rotated_normals = torch.matmul(trg_gt_normal, rotation.T.to(trg_gt_normal.dtype))
        cos_sim = torch.matmul(src_gt_normal, rotated_normals.T)
        normal_mask = cos_sim < gt_normal_threshold
        if normal_mask.shape != inliers.shape:
            raise ValueError("Normal mask shape does not match inlier mask.")
        inliers &= normal_mask

        num_inliers = inliers.any(dim=0).sum().item()
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_rotation = rotation
            best_translation = translation

    if best_inliers is None:
        raise RuntimeError("Failed to estimate a valid transform via RANSAC.")

    strong_distance_threshold = 0.01
    strong_normal_threshold = -0.7

    for _ in range(num_iters):
        transformed_trg = _transform_points(trg_pcd, best_rotation, best_translation)
        dist_mat = torch.cdist(src_pcd, transformed_trg)
        refined_inliers = dist_mat < strong_distance_threshold
        refined_inliers &= score_mask

        rotated_normals = torch.matmul(trg_gt_normal, best_rotation.T.to(trg_gt_normal.dtype))
        cos_sim = torch.matmul(src_gt_normal, rotated_normals.T)
        normal_mask = cos_sim < strong_normal_threshold
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
            best_rotation, best_translation = estimate_rigid_transform(src_points, trg_points)
        except RuntimeError:
            break

        best_inliers = refined_inliers

    return best_rotation, best_translation, best_inliers
