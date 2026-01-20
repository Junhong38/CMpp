import torch
from common.metric_utils import *
from common.misc import batch_scaling, extract_all_objects_by_offset, bincount2offset, offset2batch
from common.utils import pairwise_mating



def make_setting_for_next_iteration(previous_setting, removed_assm_pred, list_of_all_pcds, anchor_obj_idx, selected_obj_idx):
    """
    Make setting for the next iteration
    Args:
        previous_setting (dict): dictionary of the previous setting
        removed_assm_pred (torch.Tensor): (N', 3)
        list_of_all_pcds (list): list of (N, 3), len == num_of_parts
        anchor_obj_idx (int): index of the anchor object
        selected_obj_idx (int): index of the selected object
    Returns:
        new_setting (dict): dictionary of the new setting
    """
    if previous_setting is None:
        first_obj_ids = list(range(len(list_of_all_pcds)))
        first_left_obj_ids = first_obj_ids[:anchor_obj_idx] + first_obj_ids[anchor_obj_idx+1:]
        obj_ids = ([anchor_obj_idx.item()], first_left_obj_ids)
    else:
        obj_ids = (previous_setting['obj_ids'][0] + [previous_setting['obj_ids'][1][selected_obj_idx]], previous_setting['obj_ids'][1][:selected_obj_idx] + previous_setting['obj_ids'][1][selected_obj_idx+1:])


    # Refine left pcds
    left_pcds = []
    for i in range(len(list_of_all_pcds)):
        if i == anchor_obj_idx or i == selected_obj_idx:
            continue
        left_pcds.append(list_of_all_pcds[i])
    
    # Refine offset to fit to new pcds
    new_bincount = torch.tensor([a_pcd.shape[0] for a_pcd in left_pcds] + [removed_assm_pred.shape[0]], device=removed_assm_pred.device) # (num_of_parts,)
    new_offset = bincount2offset(new_bincount) # (num_of_parts,)
    new_batch_info = offset2batch(new_offset).unsqueeze(0) # (1, N+M)
    new_batch_scaled_batch_info = batch_scaling(new_batch_info) # (1, N+M)

    # Change format to fit to required input format for forward pass
    new_list_of_input_pcds = left_pcds + [removed_assm_pred]
    new_anchor = len(new_list_of_input_pcds) - 1 # Our code tries to move src, so the anchor should have idx as 1 for being the target object

    two_part_assumption_batch_info = (new_batch_info == new_anchor).int() # idx:0 means src, idx:1 means trg

    new_setting = {
        'anchor': new_anchor, # int
        'bincount': new_bincount, # (num_of_parts,)
        'offset': torch.cat([torch.zeros(1, device=removed_assm_pred.device), new_offset], dim=0).int(), # (num_of_parts+1,)
        'batch_info': new_batch_info, # (1, N+M)
        'two_part_assumption_batch_info': two_part_assumption_batch_info, # (1, N+M)
        'batch_scaled_batch_info': new_batch_scaled_batch_info, # (1, N+M)
        'list_of_input_pcds': new_list_of_input_pcds, # list of (N, 3)
        'input_pcds': torch.cat(new_list_of_input_pcds, dim=0).unsqueeze(0), # (1, N+M, 3)
        'obj_ids': obj_ids, # (2, ), first: original ids for the ordered objects and this can be used for assembly sequence, second: original ids for the left objects
    }
    
    return new_setting



def make_score_into_list_format(score_matrix, anchor_idx, offset):
    """
    Args:
        score_matrix (torch.Tensor): (B, N+M, N+M)
        anchor_idx (int): index of the anchor object
        offset (torch.Tensor): (num_of_parts+1,)
    Returns:
        list_of_scores (list): list of (N, M)
    """
    anchor_idx_start_idx = offset[anchor_idx]
    anchor_idx_end_idx = offset[anchor_idx+1]

    all_src_to_trg_score = score_matrix[0, :, anchor_idx_start_idx:anchor_idx_end_idx] # (N+M, M)
    list_of_all_src_to_trg_score = extract_all_objects_by_offset(all_src_to_trg_score, offset) # list of (N, M)
    return list_of_all_src_to_trg_score


def select_obj_to_be_assembled(matching_scores_drop, anchor_idx, offset, infer_topk):
    anchor_idx_start_idx = offset[anchor_idx]
    anchor_idx_end_idx = offset[anchor_idx+1]
    
    all_src_to_trg_score = matching_scores_drop[0, :, anchor_idx_start_idx:anchor_idx_end_idx] # (N+M, M)
    list_of_all_src_to_trg_score = extract_all_objects_by_offset(all_src_to_trg_score, offset) # list of (N, M)
    
    all_correspondence_scores = []
    for i in range(len(list_of_all_src_to_trg_score)):
        if i == anchor_idx:
            all_correspondence_scores.append(- torch.inf)
        else:
            src_to_trg_score = list_of_all_src_to_trg_score[i]
            topk_correspondences, _ = torch.topk(src_to_trg_score.reshape(-1), k=infer_topk)
            average_of_topk_correspondences = topk_correspondences.mean()
            all_correspondence_scores.append(average_of_topk_correspondences)
    
    all_correspondence_scores = torch.tensor(all_correspondence_scores)
    selected_obj_idx = torch.argmax(all_correspondence_scores)
    
    assert selected_obj_idx != anchor_idx, f"selected_obj_idx: {selected_obj_idx}, anchor_idx: {anchor_idx}, all_correspondence_scores: {all_correspondence_scores}"

    return list_of_all_src_to_trg_score, selected_obj_idx



def remove_inner_parts(assm_pred, list_of_assm_pred, list_of_oris, anchor_obj_idx, selected_obj_idx, pos_radius=0.018, cos_threshold=0.0):
    """
    Remove inner parts from the assembled point cloud

    Args:
        assm_pred (torch.Tensor): (N+M, 3)
        list_of_assembled_pcds (list): list of (N, 3), len == 2
        list_of_oris (list): list of (N, 3), which is already normalized
        anchor_obj_idx (int): index of the anchor object
        selected_obj_idx (int): index of the selected object
        pos_radius (float): radius of the small enough part
        cos_threshold (float): threshold of the cosine similarity
    Returns:
        assem_pred_after_removing_inner_parts (torch.Tensor): (N+M, 3)
    """
    assert len(list_of_assm_pred) == 2, f"len(list_of_assembled_pcds): {len(list_of_assm_pred)}"

    # Distance-based thresholding
    distance_in_obj = torch.cdist(list_of_assm_pred[0], list_of_assm_pred[1], p=2) # (N, M)
    small_enough_part = distance_in_obj < pos_radius # (N, M)

    # Normal-based thresholding
    pred_normal_from_src = list_of_oris[selected_obj_idx] # (N, 3)
    pred_normal_from_trg = list_of_oris[anchor_obj_idx] # (M, 3)
    cosine_similarity_by_normals = torch.einsum('n x, m x -> n m', pred_normal_from_src, pred_normal_from_trg) # (N, M)
    opposite_enough_part = cosine_similarity_by_normals < cos_threshold # (N, M)
    
    # Combine the two masks
    mask_for_removing_inner_parts = torch.logical_and(small_enough_part, opposite_enough_part) # (N, M)
    mask_for_src = torch.any(mask_for_removing_inner_parts, dim=1) # (N, )
    mask_for_trg = torch.any(mask_for_removing_inner_parts, dim=0) # (M, )
    total_mask = torch.concat([mask_for_src, mask_for_trg], dim=0) # (N+M, )
    final_mask_for_removing_inner_parts = ~ total_mask # (N+M, )
    
    # Return the assembled point cloud after removing inner parts
    assem_pred_after_removing_inner_parts = assm_pred[final_mask_for_removing_inner_parts]

    return assem_pred_after_removing_inner_parts




def apply_transformation_to_point_clouds(list_of_pcds, rot_and_trans_dict, anchor_idx, total_num_of_parts):
    """
    Apply transformation to the point clouds
    Args:
        list_of_pcds (list): list of (N, 3), len == total_num_of_parts
        rot_and_trans_dict (dict): dictionary of the transformation
        anchor_idx (int): index of the anchor object
        total_num_of_parts (int): total number of parts
    Returns:
        list_of_pcds_after_transformation (list): list of (N, 3), len == total_num_of_parts
    """
    all_obj_idx = list(range(total_num_of_parts))
    all_obj_idx.remove(anchor_idx)

    assembled_pcds = [list_of_pcds[anchor_idx]]
    for ith_obj in all_obj_idx:
        origin_src_pcd = list_of_pcds[ith_obj]
        rot_and_trans = rot_and_trans_dict[f"{ith_obj}-{anchor_idx}"]
        assm_pred, list_of_assm_pred = pairwise_mating(origin_src_pcd, list_of_pcds[anchor_idx], rot_and_trans[0], rot_and_trans[1]) # (N+M, 3)
        assembled_pcds.append(list_of_assm_pred[0])
    
    return assembled_pcds





def compute_metrics(list_of_assembled_pcds, list_of_gt_assembled_pcds, pred_rot_and_trans_dict, GT_rot_and_trans_dict):
    """
    Compute metrics
    Args:
        list_of_assembled_pcds (list): list of (N, 3), len == total_num_of_parts
        list_of_gt_assembled_pcds (list): list of (N, 3), len == total_num_of_parts
        pred_rot_and_trans_dict (dict): dictionary of the transformation
        GT_rot_and_trans_dict (dict): dictionary of the transformation
    Returns:
        eval_result (dict): dictionary of the evaluation results
    """
    eval_result = {}
    assembled_pcds = torch.cat(list_of_assembled_pcds, dim=0) # (N1+N2+...+Nk, 3)
    gt_assembled_pcds = torch.cat(list_of_gt_assembled_pcds, dim=0) # (N1+N2+...+Nk, 3)

    # Compute metrics
    # CD/CRD
    eval_result['cd'] = chamfer_distance(assembled_pcds, gt_assembled_pcds)
    eval_result['crd'] = correspondence_distance(assembled_pcds, gt_assembled_pcds)

    # RRMSE_GEO/TRMSE_GEO
    list_of_pred_rot_and_trans = []
    list_of_gt_rot_and_trans = []
    for key, value in pred_rot_and_trans_dict.items():
        list_of_pred_rot_and_trans.append(value)
        list_of_gt_rot_and_trans.append(GT_rot_and_trans_dict[key])
    
    eval_result['rrmse_geo'], eval_result['trmse_geo'] = transformation_error_geodesic(list_of_pred_rot_and_trans, list_of_gt_rot_and_trans, multi_part=True)
    eval_result['rrmse'], eval_result['trmse'] = transformation_error(list_of_pred_rot_and_trans, list_of_gt_rot_and_trans, multi_part=True)
    
    
    # Part Accuracy
    eval_result['part_acc_cd'] = part_accuracy_based_on_cd(list_of_assembled_pcds, list_of_gt_assembled_pcds)
    eval_result['part_acc_crd'] = part_accuracy_based_on_crd(list_of_assembled_pcds, list_of_gt_assembled_pcds)

    return eval_result