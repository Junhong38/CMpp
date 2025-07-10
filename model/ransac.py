import torch
from scipy.spatial.distance import cdist
from chamfer_distance import ChamferDistance as chamfer_dist
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

# def estimate_rigid_transform(A, B):
#     """
#     Estimate R, t using SVD from A (source) to B (target)
#     A, B: [N, 3]
#     Returns:
#         R: [3, 3]
#         t: [3]
#     """
#     centroid_A = A.mean(dim=0)
#     centroid_B = B.mean(dim=0)
 
#     A_centered = A - centroid_A
#     B_centered = B - centroid_B
 
#     H = B_centered.T @ A_centered
#     U, S, V = torch.linalg.svd(H)
#     R = V.T @ U.T
#     # Reflection correction
#     if torch.det(R) < 0:
#         V[2, :] *= -1
#         R = V.T @ U.T
 
#     t = centroid_A - R @ centroid_B
#     return R, t
 
def ransac_rigid(
        src_corr_pcd, trg_corr_pcd, 
        src_pcd, trg_pcd, 
        gt_trsfm,
        src_gt_normal, trg_gt_normal,
        scores, 
        shape_scores,
        occ_scores,
        score_threshold, 
        feature_mask, 
        feature_thresholding, 
        src_ori, trg_ori, 
        dist_mat_type='l2_norm', 
        which_transform='corr', 
        orientation_threshold=-1.0, 
        num_iters=1000, 
        threshold=0.01, 
        weights=None,
        gt_normal_threshold=-1.0,
        matching_choice='many-to-one',
        score_comb='intersection'
        ):
    """
    Run RANSAC to robustly estimate rigid transform from A to B
    A, B: [N, 3]
    Returns:
        best_R, best_t, best_inliers
    """
    N = src_corr_pcd.shape[0]
    best_inliers = None
    max_inliers = -99999999
    best_R = None
    best_t = None
    best_CD = None
    min_CD = 10000
    best_normals = None
 
    for _ in range(num_iters):
        while True:
            idx = torch.randperm(N)[:3]  # minimum 3 pts
            # idx = torch.randperm(N)[:10]  # minimum 10 pts
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
        if which_transform == 'corr':
            trg_transformed = (R @ trg_corr_pcd.T).T + t
            if dist_mat_type == 'l2_norm':
                dist_mat = torch.norm(src_corr_pcd - trg_transformed, dim=1)
            elif dist_mat_type == 'cdist':
                dist_mat = cdist(src_corr_pcd.cpu().numpy(), trg_corr_pcd.cpu().numpy())
        elif which_transform == 'whole':
            trg_transformed = (R @ trg_pcd.T).T + t
            if dist_mat_type == 'l2_norm':
                raise ValueError("'l2_norm' distance matrix type is not supported for 'whole' transform type")
            elif dist_mat_type == 'cdist':
                # if src_pcd.cpu().numpy().ndim == 1:
                #     breakpoint()
                dist_mat = cdist(src_pcd.cpu().numpy(), trg_transformed.cpu().numpy())


        # chd = chamfer_dist()
        # trg_gt_transformed = (gt_trsfm[0].squeeze(0).T @ trg_pcd.T).T + gt_trsfm[1].squeeze(0) @ gt_trsfm[0].squeeze(0)
        # dist1, dist2, idx1, idx2 = chd(torch.cat([src_pcd, trg_gt_transformed], dim=0).unsqueeze(0), torch.cat([src_pcd, trg_transformed], dim=0).unsqueeze(0))
        # cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * 1000
        # cd = round(cd.item(),2)


        # Inlier selection
        inliers = torch.from_numpy(dist_mat < threshold).to('cuda')


        

        # Orientation masking
        if orientation_threshold != -1:
            # src_ori = torch.matmul(src_ori, R)
            trg_ori = torch.matmul(trg_ori, R.T)
            
            # src_normal = torch.mean(src_ori, dim=2)
            # trg_normal = torch.mean(trg_ori, dim=2)
            for i in range(3):
                src_normal = src_ori[:,i,:]
                trg_normal = trg_ori[:,i,:]
                src_normal = src_normal / torch.norm(src_normal, dim=1, keepdim=True)
                trg_normal = trg_normal / torch.norm(trg_normal, dim=1, keepdim=True)
                
                cos_sim = torch.matmul(src_normal, trg_normal.T)
                ori_mask = cos_sim > orientation_threshold
                # breakpoint()
                if which_transform == 'whole':
                    if inliers.shape == ori_mask.shape:
                        inliers = inliers & ori_mask.cpu().numpy()
                    else:
                        raise ValueError("Something wrong in orientation thresholding~")
                else:
                    raise ValueError("The code has not yet been implemented;;")
        
        # Score thresholding
        if which_transform == 'whole':
            if feature_thresholding:
                if inliers.shape == feature_mask.shape:
                    inliers = inliers & feature_mask.cpu().numpy()
                else:
                    raise ValueError("Something wrong in feature thresholding~")
            else:
                if score_comb == 'sum':
                    score_mask = scores >= score_threshold
                elif score_comb == 'intersection':
                    shape_mask = shape_scores >= score_threshold
                    occ_mask = occ_scores >= score_threshold
                    score_mask = shape_mask & occ_mask
                if inliers.shape == score_mask.shape:
                    inliers = inliers & score_mask
                else:
                    raise ValueError("Something wrong in score thresholding~")
            
        
        # Using gt_normals
        if gt_normal_threshold != -1:
            R_ = R.to(dtype=trg_gt_normal.dtype)
            trg_gt_normal_rotat = torch.matmul(trg_gt_normal, R_.T)
            cos_sim = torch.matmul(src_gt_normal, trg_gt_normal_rotat.T)
            normal_mask = cos_sim < gt_normal_threshold
            # print(f"inliers shape: {inliers.shape} | normal_mask shape: {normal_mask.shape}")
            if which_transform == 'whole':
                if inliers.shape == normal_mask.shape:
                    # not_normals = inliers & ~normal_mask
                    inliers = inliers & normal_mask
                    # if not_normals.sum() == 0:
                    #     print(f"not_normals_among_inleirs: {not_normals.sum()}")
                    # print(np.array_equal(inliers, temp_inliers))
                else:
                    breakpoint()
                    raise ValueError("Something wrong in gt normal thresholding~")
            else:
                raise ValueError("The code has not yet been implemented;;")
        

        # Weighted Voting
        if which_transform == 'corr':
            num_inliers = inliers.sum().item()
        elif which_transform == 'whole':
            if weights == None:
                num_inliers = torch.count_nonzero(inliers.sum(dim=0))
                num_inliers = inliers.sum()
                # num_inliers = inliers.sum()
                # num_not_normals = torch.count_zero(inliers.sum(dim=0))
                # num_not_normals = torch.count_nonzero(not_normals.sum(dim=0))
                # print(f"[In RANSAC] num_inliers: {num_inliers} | num_not_normals: {num_not_normals}")
                # num_inliers = num_inliers - num_not_normals
                # print(f"num_inliers - num_not_normals: {num_inliers}-{num_not_normals}")
                # if num_not_normals > 0:
                #     num_inliers = 0
            else:
                inliers = inliers * weights.cpu().numpy()
                num_inliers = np.sum(np.mean(inliers, axis=0))
                # num_inliers = np.sum(np.min(inliers, axis=0))
                # num_inliers = np.sum(np.max(inliers, axis=0))
 
        # print(num_inliers)
        # Update best model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_R = R
            best_t = t
            # best_CD = cd
            best_normals = cos_sim[inliers]
        
        # if min_CD > cd:
        #     min_CD = cd
        #     # print(min_CD)

    return best_R, best_t, best_inliers, best_normals