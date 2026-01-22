import torch
from typing import Set, Tuple

from RANSAC.utils import _squeeze_leading_dim, _transform_points, _select_correspondences, estimate_rigid_transform
from RANSAC.weighted_procrustes import weighted_procrustes

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
        gt_corr = None,
        file_path = None,
        gtRT = None,

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

    # For inlier checking
    num_gt_corr_inliers = []
    num_non_gt_corr_inliers = []
    gt_corr_inliers = []
    non_gt_corr_inliers = []
    gt_corr_R = []
    gt_corr_t = []
    non_gt_corr_R = []
    non_gt_corr_t = []

    if src_corr_pcd.shape[0] < 3:
        raise ValueError("At least three correspondences are required for RANSAC.")

    device = src_corr_pcd.device

    N = src_corr_pcd.shape[0]
    max_total_score = - torch.inf
    best_score = None
    best_rotation = None
    best_translation = None

    best_inliers = None
    max_inliers = -1
    score_mask = (scores >= score_threshold).to(device)

    best_RT = torch.inf
    best_src_sam = None
    best_trg_sam = None
    best_re = None
    best_te = None

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
        inliers = distance_mask

        rotated_normals = torch.matmul(src_normal, rotation.T.to(src_normal.dtype))
        cos_sim = torch.matmul(rotated_normals, trg_normal.T)
        angle = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
        normal_mask = angle > normal_threshold
        if temp_scores.shape != normal_mask.shape:
            raise ValueError("Normal mask shape does not match inlier mask.")
        temp_scores *= normal_mask
        inliers &= normal_mask

        if inliers.shape != score_mask.shape:
            raise ValueError("Score mask shape does not match distance matrix.")
        inliers &= score_mask

        # total_survived_score = temp_scores.sum().item()
        # if total_survived_score > max_total_score:
        #     max_total_score = total_survived_score
        #     best_score = temp_scores
        #     best_rotation = rotation
        #     best_translation = translation

        num_inliers = inliers.sum().item()
        # if num_inliers > max_inliers:
        #     max_inliers = num_inliers
        #     best_inliers = inliers
        #     best_score = temp_scores
        #     best_rotation = rotation
        #     best_translation = translation
        
        re, te = _transformation_error_geodesic(gtRT, [rotation, translation])
        # if (re+te).item() < best_RT:
        if (re+te).item() < best_RT and (src_sample[:, None, :] == src_pcd[gt_corr[:, 0]][None, :, :]).all(dim=-1).any(dim=1).all().item() and (trg_sample[:, None, :] == trg_pcd[gt_corr[:, 1]][None, :, :]).all(dim=-1).any(dim=1).all().item():
            best_RT = (re+te).item()
            # print(f'RE: {re}')
            # print(f'TE: {te}')
            best_inliers = inliers
            best_score = temp_scores
            best_rotation = rotation
            best_translation = translation
            best_num_inliers = num_inliers
            best_src_sample = src_sample
            best_trg_sample = trg_sample
            best_re = re
            best_te = te
        
        # For inlier checking
        if (src_sample[:, None, :] == src_pcd[gt_corr[:, 0]][None, :, :]).all(dim=-1).any(dim=1).all().item() and (trg_sample[:, None, :] == trg_pcd[gt_corr[:, 1]][None, :, :]).all(dim=-1).any(dim=1).all().item():
            num_gt_corr_inliers.append(num_inliers)
            gt_corr_inliers.append(inliers)
            gt_corr_R.append(rotation)
            gt_corr_t.append(translation)
        else:
            num_non_gt_corr_inliers.append(num_inliers)
            non_gt_corr_inliers.append(inliers)
            non_gt_corr_R.append(rotation)
            non_gt_corr_t.append(translation)

    if best_score is None:
        # raise RuntimeError("Failed to estimate a valid transform via RANSAC.")
        print(f"{file_path[0]}")
        best_rotation, best_translation = estimate_rigid_transform(src_corr_pcd, trg_corr_pcd)
        best_score = 0
        best_re, best_te = _transformation_error_geodesic(gtRT, [best_rotation, best_translation])
        return best_rotation, best_translation, best_score, best_re, best_te

    # print(f"max_total_score: {max_total_score}")
    # print(f"best_rotation: {best_rotation}")
    # print(f"best_translation: {best_translation}")

    # Optimal Estimation
    strong_distance_threshold = 0.008
    # strong_distance_threshold = threshold
    num_iters_for_optimal_estimation = 100
    
    for _ in range(num_iters_for_optimal_estimation):
        refined_score = scores.clone()

        transformed_src = _transform_points(src_pcd, best_rotation, best_translation)
        dist_mat = torch.cdist(transformed_src, trg_pcd)
        distance_mask = dist_mat < strong_distance_threshold
        refined_score *= distance_mask

        rotated_normals = torch.matmul(src_normal, best_rotation.T.to(src_normal.dtype))
        cos_sim = torch.matmul(rotated_normals, trg_normal.T)
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

    ### for inlier checking
    # visualization as the histogram
    import numpy as np
    import matplotlib.pyplot as plt

    if len(num_gt_corr_inliers) > 0 and len(num_non_gt_corr_inliers) > 0:
        gt = np.array(num_gt_corr_inliers)
        non_gt = np.array(num_non_gt_corr_inliers)

        bins = np.linspace(min(gt.min(), non_gt.min()),
                        max(gt.max(), non_gt.max()), 35)

        plt.figure(figsize=(10, 6))
        plt.hist(gt, bins=bins, alpha=0.6, label="GT Corr Inliers")
        plt.hist(non_gt, bins=bins, alpha=0.6, label="Non-GT Corr Inliers")

        plt.xlabel("Number of Inliers")
        plt.ylabel("Number of such cases")
        plt.title(f"filepath: {file_path[0].replace('/', '_')}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        import os
        vis_dir = f"inlier_vis/{file_path[0].replace('/','_')}"
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(f"{vis_dir}/gt_vs_non_gt_inliers_hist.png", dpi=200, bbox_inches="tight")
        plt.close()

        # visualization each inliers
        # num_gt_corr_inliers 기준으로 오름차순 정렬
        idx_gt = sorted(range(len(num_gt_corr_inliers)), key=lambda i: num_gt_corr_inliers[i], reverse=True)
        num_gt_sorted = [num_gt_corr_inliers[i] for i in idx_gt]
        gt_sorted  = [gt_corr_inliers[i] for i in idx_gt]
        gt_R_sorted = [gt_corr_R[i] for i in idx_gt]
        gt_t_sorted = [gt_corr_t[i] for i in idx_gt]


        idx_non_gt = sorted(range(len(num_non_gt_corr_inliers)), key=lambda i: num_non_gt_corr_inliers[i], reverse=True)
        num_non_gt_sorted = [num_non_gt_corr_inliers[i] for i in idx_non_gt]
        non_gt_sorted  = [non_gt_corr_inliers[i] for i in idx_non_gt]
        non_gt_R_sorted = [non_gt_corr_R[i] for i in idx_non_gt]
        non_gt_t_sorted = [non_gt_corr_t[i] for i in idx_non_gt]

        K = 1
        for i in range(K):
            save_src_trg_with_inliers_ply(src_pcd, trg_pcd, gt_R_sorted[i], gt_t_sorted[i], gt_sorted[i], f'./{vis_dir}/gt_corr_top{i}_num{num_gt_sorted[i]}.ply')
            save_src_trg_with_inliers_ply(src_pcd, trg_pcd, non_gt_R_sorted[i], non_gt_t_sorted[i], non_gt_sorted[i], f'./{vis_dir}/non_gt_corr_top{i}_num{num_non_gt_sorted[i]}.ply')
    else:
        import os
        vis_dir = f"inlier_vis/{file_path[0].replace('/','_')}"
        os.makedirs(vis_dir, exist_ok=True)
    # print(best_src_sample, best_trg_sample)
    save_src_trg_with_inliers_ply(src_pcd, trg_pcd, best_rotation, best_translation, best_inliers, f'./{vis_dir}/best_gt_corr_num{best_inliers.sum().item()}_re{best_re}_te{best_te}.ply', best_src_sample, best_trg_sample)
    ###

    return best_rotation, best_translation, best_score, best_re, best_te

def save_src_trg_with_inliers_ply(
    src_pcd,
    trg_pcd,
    rotation,
    translation,
    inliers,
    out_path: str,
    src_sam=None,
    trg_sam=None,
    *,
    src_color=(0.1, 0.6, 1.0),        # 색1
    src_inlier_color=(0.9, 0.2, 0.2), # 색1'
    trg_color=(0.6, 0.6, 0.6),        # 색2
    trg_inlier_color=(0.2, 0.9, 0.2), # 색2'
):
    """
    Save transformed source and target point clouds as a single colored PLY.
    Points that participate in any inlier correspondence are colored differently.

    Args:
        src_pcd: (N,3) numpy array or torch tensor
        trg_pcd: (M,3) numpy array or torch tensor
        rotation: (3,3) numpy array or torch tensor
        translation: (3,) or (3,1) numpy array or torch tensor
        inliers: (N,M) bool numpy array or torch tensor
        out_path: output .ply path

    Notes:
        - Requires an existing function: _transform_points(points, R, t) -> (N,3)
        - Colors are RGB floats in [0,1].
    """

    import numpy as np
    import open3d as o3d

    # --- helper: torch/numpy -> numpy float64 ---
    def to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        return np.asarray(x)

    # --- transform source using your existing function ---
    src_tf = _transform_points(src_pcd, rotation, translation)  # (N,3)
    
    if src_sam is not None:
        src_sam_tf = _transform_points(src_sam, rotation, translation)
        src_sam_tf = to_numpy(src_sam_tf).astype(np.float64)
        trg_sam = to_numpy(trg_sam).astype(np.float64)

    src = to_numpy(src_pcd).astype(np.float64)
    src_tf = to_numpy(src_tf).astype(np.float64)
    trg = to_numpy(trg_pcd).astype(np.float64)
    R = to_numpy(rotation).astype(np.float64)
    t = to_numpy(translation).astype(np.float64)
    inl = to_numpy(inliers)

    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src_pcd must be (N,3), got {src.shape}")
    if trg.ndim != 2 or trg.shape[1] != 3:
        raise ValueError(f"trg_pcd must be (M,3), got {trg.shape}")
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be (3,3), got {R.shape}")

    # translation shape normalize to (3,)
    t = t.reshape(-1)
    if t.shape[0] != 3:
        raise ValueError(f"translation must have 3 elements, got {t.shape}")

    if inl.shape != (src.shape[0], trg.shape[0]):
        raise ValueError(f"inliers must be (N,M) = ({src.shape[0]},{trg.shape[0]}), got {inl.shape}")

    # ensure boolean
    inl = inl.astype(bool)

    # --- inlier membership per point ---
    # src point i is highlighted if any inlier in row i
    src_has_inlier = inl.any(axis=1)  # (N,)
    # trg point j is highlighted if any inlier in col j
    trg_has_inlier = inl.any(axis=0)  # (M,)

    # --- build colors ---
    src_colors = np.tile(np.array(src_color, dtype=np.float64), (src_tf.shape[0], 1))
    trg_colors = np.tile(np.array(trg_color, dtype=np.float64), (trg.shape[0], 1))

    src_colors[src_has_inlier] = np.array(src_inlier_color, dtype=np.float64)
    trg_colors[trg_has_inlier] = np.array(trg_inlier_color, dtype=np.float64)

    # --- merge and save ---
    points = np.vstack([src_tf, trg])
    colors = np.vstack([src_colors, trg_colors])

    if src_sam is not None:
        # 원하는 색 (기존 색들과 겹치지 않게)
        src_sam_color = np.array([1.0, 1.0, 0.0], dtype=np.float64)  # yellow
        trg_sam_color = np.array([1.0, 0.0, 1.0], dtype=np.float64)  # magenta

        # src_sam_tf / trg_sam: already numpy float64 above
        sam_points = np.vstack([src_sam_tf, trg_sam])

        sam_colors = np.vstack([
            np.tile(src_sam_color, (src_sam_tf.shape[0], 1)),
            np.tile(trg_sam_color, (trg_sam.shape[0], 1)),
        ])

        # append into existing points/colors
        points = np.vstack([points, sam_points])
        colors = np.vstack([colors, sam_colors])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ok = o3d.io.write_point_cloud(out_path, pcd, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write point cloud to: {out_path}")

def _transformation_error_geodesic(trnsf1, trnsf2, trmse_scaling=100):
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