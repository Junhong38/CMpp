import torch
import math
from RANSAC.match_selection import topk_matching, mutual_topk_matching, soft_topk_matching, unidirectional_nn_matching, injective_matching, bijective_matching 


from RANSAC.default_ransac import ransac_rigid as ransac_rigid_original
from RANSAC.score_dependent_ransac import ransac_rigid as score_dependent_ransac_rigid
from RANSAC.distance_dependent_ransac import ransac_rigid as distance_dependent_ransac_rigid

from common.misc import extract_all_objects


def _RANSAC(
        in_dict, 
        shape_matching_scores, 
        src_pcd, trg_pcd, 
        src_predicted_frame=None, trg_predicted_frame=None, 
        match_option='topk', RANSAC_type='default', topk=128,
        normal_threshold=0, strong_normal_threshold=0
        ):
    """
    RANSAC for point cloud registration

    Args:
        in_dict (dict): Input dictionary. Please refer CMpp_equiassem.py for more details.
        shape_matching_scores (torch.Tensor): (N, M) shape matching scores
        src_pcd (torch.Tensor): (M, 3) source point cloud
        trg_pcd (torch.Tensor): (N, 3) target point cloud
        src_predicted_frame (torch.Tensor, optional): (N, 3, 3) source predicted frame. Defaults to None.
        trg_predicted_frame (torch.Tensor, optional): (M, 3, 3) target predicted frame. Defaults to None.
        match_option (str, optional): 'topk' or 'mutual_topk' or 'soft_topk'. Defaults to 'topk'.
        RANSAC_type (str, optional): 'default' or 'score_dependent'. Defaults to 'default'.
        topk (int, optional): Topk value for matching. Defaults to 128.
    """
    matching_scores_before_Sinkhorn = shape_matching_scores # (N, M)
                    
    # Initial matches for RANSAC
    if match_option == 'topk':
        if topk < 0:
            topk = int((matching_scores_before_Sinkhorn.shape[0] * matching_scores_before_Sinkhorn.shape[1]) * (-topk) / 100)
        initial_matches = topk_matching(matching_scores_before_Sinkhorn, k=int(topk)) # (K, 2)
    elif match_option == 'mutual_topk':
        initial_matches = mutual_topk_matching(matching_scores_before_Sinkhorn, topk=int(topk)) # (K, 2)
    elif match_option == 'soft_topk':
        initial_matches = soft_topk_matching(matching_scores_before_Sinkhorn, topk=int(topk)) # (K, 2)
    elif match_option == 'unidirectional_nn_matching':
        initial_matches = unidirectional_nn_matching(matching_scores_before_Sinkhorn, topk=int(topk)) # (K, 2)
    elif match_option == 'injective_matching':
        initial_matches = injective_matching(matching_scores_before_Sinkhorn) # (K, 2)
    elif match_option == 'bijective_matching':
        initial_matches = bijective_matching(matching_scores_before_Sinkhorn) # (K, 2)
    else:
        raise ValueError(f"Invalid match option: {match_option}")
    

    # Get initial matches
    src_idx, trg_idx = initial_matches[:, 0], initial_matches[:, 1] # (K, ), (K, )

    # Score thresholding for initial matches
    # Score is consine similarity between shape features from src and trg
    # Hence, if the score is less than 0.0, then the correspondence is not good
    score_threshold = 0.0
    score_mask = matching_scores_before_Sinkhorn[src_idx, trg_idx] >= score_threshold # (K, )
    src_idx, trg_idx = src_idx[score_mask], trg_idx[score_mask] # (K_filtered, ), (K_filtered, )

    # Real used correspondences
    used_corr = torch.stack([src_idx, trg_idx], dim=1) # (K_filtered, 2)


    # Prepare to run RANSAC
    src_corr_pts = src_pcd[src_idx, :] # (K_filtered, 3)
    trg_corr_pts = trg_pcd[trg_idx, :] # (K_filtered, 3)


    # RANSAC
    # [TODO]
    # We assume that many of correspondences are good
    # Hence, those are inliers in high probability
    # So, we use another formula to decide the number of iterations for RANSAC

    # Sampling pool is not same as pool for counting inliers
    # So, we want to minmum number of iteration to cover all the points
    # If a point is not chosen for RANSAC, this probability is (N-1)_C_k / N_C_k
    # If we fail to choose that specific point during t iterations, this probability is ((N-1)_C_k / N_C_k)^t
    # There is N points, so total probability is N * ((N-1)_C_k / N_C_k)^t
    # This final probability should be less than delta which is 1-p
    # Hence N * ((N-1)_C_k / N_C_k)^t <= delta
    # Finally, we have t = log(N / delta) / log((N-1)_C_k / N_C_k)
    N = initial_matches.shape[0]
    k = 3  # minimum number of points to estimate the model
    delta = 0.05  # probability of choosing at least one outlier-free subset
    num_iters = max(math.ceil((N / k) * math.log(N / delta)), 100)

    if RANSAC_type == 'score_dependent':
        ransac_function = score_dependent_ransac_rigid
    elif RANSAC_type == 'distance_dependent':
        ransac_function = distance_dependent_ransac_rigid
    else:
        ransac_function = ransac_rigid_original

    if src_predicted_frame == None:
        src_normal, trg_normal = extract_all_objects(in_dict['gt_normals'][0], in_dict['pcd_batch_info'][0]) # (N, 3), (M, 3)
        
    else:
        src_normal = src_predicted_frame[:,0,:] # (N, 3, 3) -> (N, 3), select only predicted normal
        trg_normal = trg_predicted_frame[:,0,:] # (M, 3, 3) -> (M, 3), 
    
    print(src_corr_pts.shape[0])
    if src_corr_pts.shape[0] < 3:
        # Not enough correspondences for RANSAC
        from RANSAC.weighted_procrustes import weighted_procrustes
        inl_R, inl_t = weighted_procrustes(src_corr_pts, trg_corr_pts, 
                                           weights=matching_scores_before_Sinkhorn[src_idx, trg_idx],
                                           return_transform=False)
    else:
        inl_R, inl_t, inliers = ransac_function(src_corr_pts, trg_corr_pts, 
                                                src_pcd.squeeze(0), trg_pcd.squeeze(0),
                                                src_normal, trg_normal,
                                                scores = matching_scores_before_Sinkhorn,
                                                score_threshold=score_threshold,
                                                num_iters = num_iters,
                                                normal_threshold=normal_threshold,
                                                strong_normal_threshold=strong_normal_threshold)


    estimated_transform = torch.eye(4, device=inl_R.device, dtype=inl_R.dtype)
    estimated_transform[:3, :3] = inl_R
    estimated_transform[:3, 3] = inl_t


    return estimated_transform, used_corr