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
        gt_corr = None,
        file_path = None,
        gtRT = None,
        normal_buffer: int = 0,
        penetration_buffer: int = 0,
        use_penetration = False

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
    score_mask = (scores > score_threshold).to(device)

    best_RT = torch.inf
    best_src_sam = None
    best_trg_sam = None
    best_re = None
    best_te = None

    unique_src_pcd = torch.unique(src_corr_pcd, dim=0)
    unique_trg_pcd = torch.unique(trg_corr_pcd, dim=0)

    top5 = TopKBest(k=int(num_iters*5/100))
    # top5 = TopKBest(k=5)


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
            if penetration_checking(transformed_src, trg_pcd, rotated_normals, trg_normal, threshold, normal_buffer, penetration_buffer):
                continue

        # penetration_mask = penetration_checking(transformed_src, trg_pcd, rotated_normals, trg_normal, threshold, normal_buffer, penetration_buffer)
        # panalty_score = (torch.abs(temp_scores) * penetration_mask).sum().item()

        dist_mat = torch.cdist(transformed_src, trg_pcd)
        distance_mask = dist_mat < threshold

        if temp_scores.shape != distance_mask.shape:
            raise ValueError("Distance mask shape does not match shape of the inlier score.")
        temp_scores *= distance_mask
        inliers = distance_mask

        # penetration_mask = penetration_checking(transformed_src, trg_pcd, rotated_normals, trg_normal, threshold, normal_buffer, penetration_buffer)
        # temp_scores *= ~penetration_mask
        # inliers &= ~penetration_mask

        cos_sim = torch.matmul(rotated_normals, trg_normal.T)
        angle = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))
        normal_mask = angle > normal_threshold
        # normal_mask = (angle > normal_threshold) | (angle < 20)
        if temp_scores.shape != normal_mask.shape:
            raise ValueError("Normal mask shape does not match shape of the inlier score.")
        temp_scores *= normal_mask
        inliers &= normal_mask

        if inliers.shape != score_mask.shape:
            raise ValueError("Score mask shape does not match inliers shape.")
        inliers &= score_mask

        # if penetration_checking_inliers(transformed_src, trg_pcd, inliers, rotated_normals, trg_normal, threshold, normal_buffer, penetration_buffer):
        #     continue

        total_survived_score = temp_scores.sum().item() #- panalty_score
        # if total_survived_score > max_total_score:
        #     max_total_score = total_survived_score
        #     best_score = temp_scores
        #     best_rotation = rotation
        #     best_translation = translation
        
        top5.try_add(total_survived_score, rotation, translation, inliers)

        # num_inliers = inliers.sum().item()
        # if num_inliers > max_inliers:
        #     max_inliers = num_inliers
        #     best_inliers = inliers
        #     best_score = temp_scores
        #     best_rotation = rotation
        #     best_translation = translation
        
        re, te = _transformation_error_geodesic(gtRT, [rotation, translation])
        # if (re+te).item() < best_RT:
        if re.item() < best_RT:
        # if (re+te).item() < best_RT and (src_sample[:, None, :] == src_pcd[gt_corr[:, 0]][None, :, :]).all(dim=-1).any(dim=1).all().item() and (trg_sample[:, None, :] == trg_pcd[gt_corr[:, 1]][None, :, :]).all(dim=-1).any(dim=1).all().item():
            # best_RT = (re+te).item()
            best_RT = re.item()
            # print(f'RE: {re}')
            # print(f'TE: {te}')
            # best_inliers = inliers
            # best_score = temp_scores
            # best_rotation = rotation
            # best_translation = translation
            # best_num_inliers = num_inliers
            best_src_sample = src_sample
            best_trg_sample = trg_sample
            best_re = re
            best_te = te
            best_RT_rotation = rotation
            best_RT_translation = translation
            best_RT_inliers = inliers
            best_RT_score = temp_scores.sum().item()
        
        # For inlier checking
        if (src_sample[:, None, :] == src_pcd[gt_corr[:, 0]][None, :, :]).all(dim=-1).any(dim=1).all().item() and (trg_sample[:, None, :] == trg_pcd[gt_corr[:, 1]][None, :, :]).all(dim=-1).any(dim=1).all().item():
            num_gt_corr_inliers.append(total_survived_score)
            gt_corr_inliers.append(inliers)
            gt_corr_R.append(rotation)
            gt_corr_t.append(translation)
        else:
            num_non_gt_corr_inliers.append(total_survived_score)
            non_gt_corr_inliers.append(inliers)
            non_gt_corr_R.append(rotation)
            non_gt_corr_t.append(translation)

    # if best_score is None:
    #     # raise RuntimeError("Failed to estimate a valid transform via RANSAC.")
    #     print(f"{file_path[0]}")
    #     best_rotation, best_translation = estimate_rigid_transform(src_corr_pcd, trg_corr_pcd)
    #     best_score = 0
    #     best_re, best_te = _transformation_error_geodesic(gtRT, [best_rotation, best_translation])
    #     return best_rotation, best_translation, best_score, best_re, best_te

    # print(f"max_total_score: {max_total_score}")
    # print(f"best_rotation: {best_rotation}")
    # print(f"best_translation: {best_translation}")

    # Optimal Estimation
    strong_distance_threshold = 0.008
    # strong_distance_threshold = threshold
    num_iters_for_optimal_estimation = 100
    best_list = top5.get_sorted()
    if len(best_list)==0:
        # raise RuntimeError("Failed to estimate a valid transform via RANSAC.")
        print(f"{file_path[0]}")
        best_rotation, best_translation = estimate_rigid_transform(src_corr_pcd, trg_corr_pcd)
        best_score = 0
        best_re, best_te = _transformation_error_geodesic(gtRT, [best_rotation, best_translation])
        return best_rotation, best_translation, best_score, best_re, best_te

    top_rotation = None
    top_translation = None
    top_inliers = None
    top_score = None
    
    for cand in best_list:
        best_score = cand.score
        best_rotation = cand.rotation
        best_translation = cand.translation
        best_inliers = cand.inliers
        for _ in range(num_iters_for_optimal_estimation):
            refined_score = scores.clone()
            refined_score.masked_fill_(refined_score.abs() < 1e-9, -scores.max())

            # penetration_mask = penetration_checking(transformed_src, trg_pcd, rotated_normals, trg_normal, strong_distance_threshold, normal_buffer, penetration_buffer)
            # panalty_score = (torch.abs(refined_score) * penetration_mask).sum()

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
            # normal_mask = (angle > normal_threshold) | (angle < 20)
            if normal_mask.shape != refined_score.shape:
                raise ValueError("Normal mask shape does not match shape of the refined score.")
            refined_score *= normal_mask
            refined_inliers &= normal_mask

            if refined_inliers.shape != score_mask.shape:
                raise ValueError("Score mask shape does not match shape of the refined inlier.")
            refined_inliers &= score_mask

            # if penetration_checking_inliers(transformed_src, trg_pcd, refined_inliers, rotated_normals, trg_normal, strong_distance_threshold, normal_buffer, penetration_buffer):
            if use_penetration:
                if penetration_checking(transformed_src, trg_pcd, rotated_normals, trg_normal, strong_distance_threshold, normal_buffer, penetration_buffer):
                    break
            # penetration_mask = penetration_checking(transformed_src, trg_pcd, rotated_normals, trg_normal, strong_distance_threshold, normal_buffer, penetration_buffer)
            # refined_score *= ~penetration_mask
            # refined_inliers &= ~penetration_mask

            # if torch.equal(best_score, refined_score):
            if best_score == refined_score.sum() #- panalty_score:
                break
            # elif best_score.sum() > refined_score.sum():
            #     continue

            correspondences = _select_correspondences(refined_score, dist_mat, matching_choice)
            if correspondences.size(0) < 3:
                best_score = refined_score.sum().item() #- panalty_score
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
            top_inliers = best_inliers
            top_score = best_score

    # ### for inlier checking
    # # visualization as the histogram
    # import numpy as np
    # import matplotlib.pyplot as plt

    # if len(num_gt_corr_inliers) > 0 and len(num_non_gt_corr_inliers) > 0:
    #     gt = np.array(num_gt_corr_inliers)
    #     non_gt = np.array(num_non_gt_corr_inliers)

    #     bins = np.linspace(min(gt.min(), non_gt.min()),
    #                     max(gt.max(), non_gt.max()), 35)

    #     plt.figure(figsize=(10, 6))
    #     plt.hist(gt, bins=bins, alpha=0.6, label="GT Corr Inliers")
    #     plt.hist(non_gt, bins=bins, alpha=0.6, label="Non-GT Corr Inliers")

    #     plt.xlabel("Number of Inliers")
    #     plt.ylabel("Number of such cases")
    #     plt.title(f"filepath: {file_path[0].replace('/', '_')}")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)

    #     import os
    #     vis_dir = f"inlier_vis/{file_path[0].replace('/','_')}"
    #     os.makedirs(vis_dir, exist_ok=True)
    #     plt.savefig(f"{vis_dir}/gt_vs_non_gt_inliers_hist.png", dpi=200, bbox_inches="tight")
    #     plt.close()

    #     # visualization each inliers
    #     # num_gt_corr_inliers 기준으로 오름차순 정렬
    #     idx_gt = sorted(range(len(num_gt_corr_inliers)), key=lambda i: num_gt_corr_inliers[i], reverse=True)
    #     num_gt_sorted = [num_gt_corr_inliers[i] for i in idx_gt]
    #     gt_sorted  = [gt_corr_inliers[i] for i in idx_gt]
    #     gt_R_sorted = [gt_corr_R[i] for i in idx_gt]
    #     gt_t_sorted = [gt_corr_t[i] for i in idx_gt]


    #     idx_non_gt = sorted(range(len(num_non_gt_corr_inliers)), key=lambda i: num_non_gt_corr_inliers[i], reverse=True)
    #     num_non_gt_sorted = [num_non_gt_corr_inliers[i] for i in idx_non_gt]
    #     non_gt_sorted  = [non_gt_corr_inliers[i] for i in idx_non_gt]
    #     non_gt_R_sorted = [non_gt_corr_R[i] for i in idx_non_gt]
    #     non_gt_t_sorted = [non_gt_corr_t[i] for i in idx_non_gt]

    #     K = 1
    #     for i in range(K):
    #         save_src_trg_with_inliers_ply(src_pcd, trg_pcd, gt_R_sorted[i], gt_t_sorted[i], gt_sorted[i], f'./{vis_dir}/gt_corr_top{i}_score{num_gt_sorted[i]}.ply')
    #         save_src_trg_with_inliers_ply(src_pcd, trg_pcd, non_gt_R_sorted[i], non_gt_t_sorted[i], non_gt_sorted[i], f'./{vis_dir}/non_gt_corr_top{i}_score{num_non_gt_sorted[i]}.ply')
    # else:
    #     import os
    #     vis_dir = f"inlier_vis/{file_path[0].replace('/','_')}"
    #     os.makedirs(vis_dir, exist_ok=True)
    # # print(best_src_sample, best_trg_sample)
    # save_src_trg_with_inliers_ply(src_pcd, trg_pcd, best_RT_rotation, best_RT_translation, best_RT_inliers, f'./{vis_dir}/best_gt_corr_score{best_RT_score}_re{best_re}_te{best_te}.ply', best_src_sample, best_trg_sample)
    ###

    return top_rotation, top_translation, top_score, best_re, best_te

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

    # =========================
    # Build & save LineSet for inlier correspondences
    # =========================
    base_points = np.vstack([src_tf, trg])  # indices: [0..N-1]=src_tf, [N..N+M-1]=trg
    N = src_tf.shape[0]

    corr = np.argwhere(inl)  # (K,2) pairs (i,j)

    # (옵션) 라인이 너무 많으면 시각화/저장 부담 -> 제한
    max_lines = 20000
    if corr.shape[0] > max_lines:
        sel = np.random.choice(corr.shape[0], size=max_lines, replace=False)
        corr = corr[sel]

    lines = np.column_stack([corr[:, 0], corr[:, 1] + N]).astype(np.int32)  # (K,2)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(base_points)
    lineset.lines = o3d.utility.Vector2iVector(lines)

    # (옵션) 라인 색 지정 (빨강)
    line_color = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    lineset.colors = o3d.utility.Vector3dVector(np.tile(line_color, (lines.shape[0], 1)))

    # save lineset to separate ply
    line_path = out_path.replace(".ply", "_inlier_lines.ply")
    ok_line = o3d.io.write_line_set(line_path, lineset)
    if not ok_line:
        raise RuntimeError(f"Failed to write line set to: {line_path}")


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

def penetration_checking_inliers(
    src_pcd: torch.Tensor,      # (M, 3)
    trg_pcd: torch.Tensor,      # (N, 3)
    inliers: torch.Tensor,      # (M, N) bool
    src_normals: torch.Tensor,  # (M, 3)
    trg_normals: torch.Tensor,  # (N, 3)
    dist_th: float = 0.018,
    normal_buffer: float = 0.0,
    penetration_buffer: float = 0.0,
) -> bool:
    if inliers.numel() == 0 or not bool(inliers.any()):
        return False

    import math

    src_anchor = inliers.any(dim=1)  # (M,)
    trg_anchor = inliers.any(dim=0)  # (N,)

    dist2_th = dist_th * dist_th

    # norms for normalization (broadcast-friendly)
    n_norm_src = src_normals.norm(dim=-1).clamp_min(1e-12)[:, None]  # (M,1)
    n_norm_trg = trg_normals.norm(dim=-1).clamp_min(1e-12)[None, :]  # (1,N)

    normal_cos_thres = math.cos(math.radians(90.0 - float(normal_buffer)))
    penetration_cos_thres = math.cos(math.radians(90.0 + float(penetration_buffer)))

    # ---------- src-anchor p vs all trg q' ----------
    diff_pq = src_pcd[:, None, :] - trg_pcd[None, :, :]              # (M,N,3)
    diff_pq_norm = diff_pq.norm(dim=-1).clamp_min(1e-12)             # (M,N)
    dist_ok = diff_pq.square().sum(dim=-1) < dist2_th                # (M,N)

    # cos(theta) where theta = angle between (p-q') and n_{q'}
    cos_q_to_p_vs_qn = (diff_pq * trg_normals[None, :, :]).sum(dim=-1) / (diff_pq_norm * n_norm_trg)  # (M,N)
    q_to_p_vs_qn = cos_q_to_p_vs_qn < penetration_cos_thres           # (M,N)

    cos_np_nq = (src_normals[:, None, :] * trg_normals[None, :, :]).sum(dim=-1) #/ (n_norm_src * n_norm_trg)  # (M,N)
    np_dot_nq = cos_np_nq > normal_cos_thres                          # (M,N)

    ok_src = src_anchor[:, None] & dist_ok & q_to_p_vs_qn & np_dot_nq

    # ---------- trg-anchor q vs all src p' ----------
    diff_qp = trg_pcd[:, None, :] - src_pcd[None, :, :]              # (N,M,3)
    diff_qp_norm = diff_qp.norm(dim=-1).clamp_min(1e-12)             # (N,M)
    dist_ok2 = diff_qp.square().sum(dim=-1) < dist2_th               # (N,M)

    # For broadcasting, we want src norms as (1,M), trg norms as (N,1)
    n_norm_src_T = n_norm_src.squeeze(1)[None, :]                    # (1,M)
    n_norm_trg_T = n_norm_trg.squeeze(0)[:, None]                    # (N,1)

    cos_p_to_q_vs_pn = (diff_qp * src_normals[None, :, :]).sum(dim=-1) / (diff_qp_norm * n_norm_src_T)  # (N,M)
    p_to_q_vs_pn = cos_p_to_q_vs_pn < penetration_cos_thres

    cos_nq_np = (trg_normals[:, None, :] * src_normals[None, :, :]).sum(dim=-1) #/ (n_norm_trg_T * n_norm_src_T)  # (N,M)
    nq_dot_np = cos_nq_np > normal_cos_thres

    ok_trg = trg_anchor[:, None] & dist_ok2 & p_to_q_vs_pn & nq_dot_np

    return bool(ok_src.any() or ok_trg.any())


def penetration_checking(
    src_pcd: torch.Tensor, # (M, 3)
    trg_pcd: torch.Tensor, # (N, 3)
    src_normals: torch.Tensor, # (M, 3)
    trg_normals: torch.Tensor, # (N, 3)
    dist_th: float = 0.018,
    normal_buffer: int = 0,
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
    # return ok

