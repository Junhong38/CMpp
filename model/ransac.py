import torch
from scipy.spatial.distance import cdist
import numpy as np
 
def estimate_rigid_transform(A, B):
    """
    Estimate R, t using SVD from A (source) to B (target)
    A, B: [N, 3]
    Returns:
        R: [3, 3]
        t: [3]
    """
    centroid_A = A.mean(dim=0)
    centroid_B = B.mean(dim=0)
 
    A_centered = A - centroid_A
    B_centered = B - centroid_B
 
    H = B_centered.T @ A_centered
    U, S, V = torch.linalg.svd(H)
    R = V.T @ U.T
    # Reflection correction
    if torch.det(R) < 0:
        V[2, :] *= -1
        R = V.T @ U.T
 
    t = centroid_A - R @ centroid_B
    return R, t
 
def ransac_rigid(
        src_corr_pcd, trg_corr_pcd, 
        src_pcd, trg_pcd, 
        src_gt_normal, trg_gt_normal,
        scores,
        score_threshold,
        num_iters=100, 
        threshold=0.01,
        gt_normal_threshold=-0.7,
        matching_choice='one-to-one',
        ):
    """
    Run RANSAC to robustly estimate rigid transform from A to B
    A, B: [N, 3]
    Returns:
        best_R, best_t, best_inliers
    """
    N = src_corr_pcd.shape[0]
    max_inliers = -1
    best_inliers, best_R, best_t = None, None, None
 
    # RANSAC iterations
    for _ in range(num_iters):
        while True:
            idx = torch.randperm(N)[:3]  # minimum 3 pts
            src_sample = src_corr_pcd[idx]
            trg_sample = trg_corr_pcd[idx]
            if matching_choice == 'many-to-one':
                break
            if torch.unique(src_sample, dim=0).size(0) == src_sample.size(0):
                break
 
        try:
            R, t = estimate_rigid_transform(src_sample, trg_sample)
        except:
            print("fail-fail-fail-fail-fail-fail-fail-fail")
            continue
        
        # Calculate distance matrix
        dist_mat = cdist(src_pcd.cpu().numpy(), ((R @ trg_pcd.T).T + t).cpu().numpy())

        # Inlier selection
        inliers = torch.from_numpy(dist_mat < threshold).to('cuda')
        
        # Score thresholding to filter inliers
        score_mask = scores >= score_threshold
        if inliers.shape == score_mask.shape:
            inliers = inliers & score_mask
        else:
            raise ValueError("Something wrong in score thresholding~")
            
        
        # Using gt_normals to further filter inliers
        R_ = R.to(dtype=trg_gt_normal.dtype)
        trg_gt_normal_rotat = torch.matmul(trg_gt_normal, R_.T)
        cos_sim = torch.matmul(src_gt_normal, trg_gt_normal_rotat.T)
        normal_mask = cos_sim < gt_normal_threshold
        # print(f"inliers shape: {inliers.shape} | normal_mask shape: {normal_mask.shape}")
        if inliers.shape == normal_mask.shape:
            inliers = inliers & normal_mask
        else:
            raise ValueError("Something wrong in gt normal thresholding~")
        

        # Inlier Voting (inlier counting)
        num_inliers = torch.count_nonzero(inliers.sum(dim=0))
 
        # Update best model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_R = R
            best_t = t

    # Optimal estimation (re-estimation)
    optimal_inliers, optimal_R, optimal_t = None, None, None
    strong_distance_threshold = 0.01 #0.008
    strong_normal_threshold = -0.7 # -0.9
    for _ in range(num_iters):
        # inlier counting 1 : distance thresholding
        trg_transformed = (best_R @ trg_pcd.T).T + best_t
        optimal_dist_mat = cdist(src_pcd.cpu().numpy(), trg_transformed.cpu().numpy())
        optimal_inliers = torch.from_numpy(optimal_dist_mat < strong_distance_threshold).to(dtype=best_inliers.dtype, device='cuda')

        # inlier counting 2 : score thresholding
        optimal_score_mask = scores >= score_threshold
        if optimal_inliers.shape == optimal_score_mask.shape:
            optimal_inliers = optimal_inliers & optimal_score_mask
        else:
            raise ValueError("[optimal estimation] Somthing wrong in score thresholding")
        
        # inlier counting 3 : normal thresholding
        best_R_ = best_R.to(dtype=trg_gt_normal.dtype)
        trg_gt_normal_transformed = torch.matmul(trg_gt_normal, best_R_.T)
        opt_cos_sim = torch.matmul(src_gt_normal, trg_gt_normal_transformed.T)
        temp_optimal_normal_mask = opt_cos_sim < strong_normal_threshold
        optimal_normal_mask = temp_optimal_normal_mask
        
        if optimal_inliers.shape == optimal_normal_mask.shape:
            optimal_inliers = optimal_inliers & optimal_normal_mask
        else:
            raise ValueError("[optimal estimation] Somthing wrong in GT normal thresholding")

        # many-to-one case : 복제하여 one-to-one으로
        if matching_choice == 'many-to-many':
            optimal_correspondences = torch.nonzero(optimal_inliers, as_tuple=False)
        elif matching_choice == 'many-to-one':
            trg_transformed = (best_R @ trg_pcd.squeeze(0).T).T + best_t
            dist_mat = torch.from_numpy(cdist(src_pcd.squeeze(0).cpu().numpy(), trg_transformed.cpu().numpy())).to(dtype=src_pcd.dtype, device='cuda')
            candidates = torch.stack([torch.nonzero(optimal_inliers, as_tuple=False)[:,0], torch.nonzero(optimal_inliers, as_tuple=False)[:,1], dist_mat[optimal_inliers]], dim=1)
            final_selected = []
            for col in torch.unique(candidates[:, 1]):
                col_group = candidates[candidates[:, 1] == col]
                # 같은 col에 대해 가장 작은 cost만 남김
                best = col_group[torch.argmin(col_group[:, 2])][:2].to(dtype=torch.long)
                final_selected.append(best)
            optimal_correspondences = torch.stack(final_selected, dim=0)  # shape: (K, 3)
        elif matching_choice == 'one-to-one':
            trg_transformed = (best_R @ trg_pcd.squeeze(0).T).T + best_t
            dist_mat = torch.from_numpy(cdist(src_pcd.squeeze(0).cpu().numpy(), trg_transformed.cpu().numpy())).to(dtype=src_pcd.dtype, device='cuda')
            candidates = torch.stack([torch.nonzero(optimal_inliers, as_tuple=False)[:,0], torch.nonzero(optimal_inliers, as_tuple=False)[:,1], dist_mat[optimal_inliers]], dim=1)
            
            sorted_candidates = candidates[torch.argsort(candidates[:,2])]
            used_src = set()
            used_trg = set()
            final_selected = []
            for row in sorted_candidates:
                src_, trg_, _ = row.tolist()
                if  src_ not in used_src and trg_ not in used_trg:
                    final_selected.append([src_, trg_])
                    used_src.add(src_)
                    used_trg.add(trg_)
            optimal_correspondences = torch.tensor(final_selected, dtype=torch.long)
        else:
            raise ValueError("Unknown matching choice!")
        
        try:
            src_opt_corr = optimal_correspondences[:, 0]
        except IndexError as e:
            print("⚠️ IndexError 발생:", e)
            breakpoint()  # 디버깅 모드 진입
        trg_opt_corr = optimal_correspondences[:, 1]
        src_opt_pcd = src_pcd[src_opt_corr]
        trg_opt_pcd = trg_pcd[trg_opt_corr]

        # print(f"src_opt_pcd : {src_opt_pcd.shape} | trg_opt_pcd : {trg_opt_pcd.shape}")

        if src_opt_pcd.shape[0] < 3 or src_opt_pcd.dim() == 1:
            optimal_R = best_R
            optimal_t = best_t
        else:
            optimal_R, optimal_t = estimate_rigid_transform(src_opt_pcd, trg_opt_pcd)

        if torch.equal(best_inliers, optimal_inliers):
            print("[OPTIMAL ESTIMATION] inlisers and optimal_inliers are the same, so break optimal estimation loop")
            break

        best_inliers = optimal_inliers
        best_R, best_t = optimal_R, optimal_t
        print("HI")

    return optimal_R, optimal_t, optimal_inliers