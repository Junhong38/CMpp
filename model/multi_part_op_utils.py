import torch
from common.metric_utils import *
from common.misc import batch_scaling, extract_all_objects_by_offset, bincount2offset, offset2batch
from common.utils import pairwise_mating
from scipy.spatial.transform import Rotation as scipy_rot
import numpy as np
import gtsam

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


def make_input_dicts_for_shonan(src_idx, trg_idx, list_of_all_pcds, list_of_all_gt_normals):
    """
    Args:
        src_idx (int): index of the source object
        trg_idx (int): index of the target object
        list_of_all_pcds (list): list of (N, 3), len == total_num_of_parts
        list_of_all_gt_normals (list): list of (N, 3), len == total_num_of_parts
    Returns:
        input_dict (dict): dictionary of the input
            src_pcd (torch.Tensor): (N, 3)
            trg_pcd (torch.Tensor): (M, 3)
            input_pcds (torch.Tensor): (1, N+M, 3)
            batch_info (torch.Tensor): (1, N+M, )
            batch_scaled_batch_info (torch.Tensor): (1, N+M, )
        
    """
    src_pcd = list_of_all_pcds[src_idx] # (N, 3)
    trg_pcd = list_of_all_pcds[trg_idx] # (M, 3)
    src_gt_normals = list_of_all_gt_normals[src_idx] # (N, 3)
    trg_gt_normals = list_of_all_gt_normals[trg_idx] # (M, 3)
    bincount = torch.tensor([a_pcd.shape[0] for a_pcd in [src_pcd, trg_pcd]], device=src_pcd.device) # (2,)
    offset = bincount2offset(bincount) # (3,)
    batch_info = offset2batch(offset).unsqueeze(0) # (1, N+M)
    batch_scaled_batch_info = batch_scaling(batch_info) # (1, N+M)
    input_pcds = torch.cat([src_pcd, trg_pcd], dim=0).unsqueeze(0) # (1, N+M, 3)
    gt_normals = torch.cat([src_gt_normals, trg_gt_normals], dim=0).unsqueeze(0) # (1, N+M, 3)

    input_dict = {
        'src_pcd': src_pcd, # (N, 3)
        'trg_pcd': trg_pcd, # (M, 3)
        'input_pcds': input_pcds, # (1, N+M, 3)
        'pcd_batch_info': batch_info, # (1, N+M, )
        'batch_scaled_batch_info': batch_scaled_batch_info, # (1, N+M, )
        'gt_normals': gt_normals, # (1, N+M, 3)
    }
    
    return input_dict


def make_shonan_factors(pred_dict, num_of_parts, selection_mode='max'):
    """
    Args:
        pred_dict (dict): dictionary of the predicted transformation
            - key: (src_idx-trg_idx), value: (score, estimated_transform)
        num_of_parts (int): number of parts
        selection_mode (str): 'max'
    Returns:
        factors (gtsam.BetweenFactorPose3s): factors of the problem
        params (gtsam.ShonanAveragingParameters3): parameters of the shonan averaging
    """
    params = gtsam.ShonanAveragingParameters3(gtsam.LevenbergMarquardtParams.CeresDefaults())
    factors = gtsam.BetweenFactorPose3s()

    uncertainty_dict = {}
    for src_idx in range(num_of_parts):
        max_score = - torch.inf
        max_idx = -1
        for trg_idx in range(num_of_parts):
            if src_idx == trg_idx:
                continue
            key = f"{src_idx}-{trg_idx}"
            score = pred_dict[key][0]

            if max_score < score:
                max_score = score
                max_idx = trg_idx
        
        assert max_idx != -1, f"max_idx: {max_idx}, src_idx: {src_idx}, num_of_parts: {num_of_parts}"

        rot_and_trans = pred_dict[f"{src_idx}-{max_idx}"][1] # move src_idx to max_idx
        rot_matrix = rot_and_trans[:3,:3].cpu().numpy()
        translation = rot_and_trans[:3,3].cpu().numpy()
        rot_quat = scipy_rot.from_matrix(rot_matrix).as_quat()
        max_score = max_score.cpu().numpy()

        # add factor
        pose = gtsam.Pose3(gtsam.Rot3.Quaternion(rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]), gtsam.Point3(translation))
        factors.append(gtsam.BetweenFactorPose3(src_idx, max_idx, pose, gtsam.noiseModel.Diagonal.Information(max_score * np.eye(6))))
        uncertainty_dict[f"{src_idx}-{max_idx}"] = 1/max_score

    return factors, params, uncertainty_dict




def run_shonan_averaging(factors, params, max_iter=60):
    """
    Run shonan averaging
    Args:
        factors (gtsam.BetweenFactorPose3s): factors of the problem
        params (gtsam.ShonanAveragingParameters3): parameters of the shonan averaging
        max_iter (int): maximum number of iterations
    Returns:
        abs_rotat (gtsam.Values): absolute rotations
    """

    # Run shonan averaging
    sa3 = gtsam.ShonanAveraging3(factors, params)
    initial = sa3.initializeRandomly()
    pMax = 20
    while True:
        pMax += 20
        try: 
            abs_rotat, _ = sa3.run(initial, 3, pMax)
            break
        except RuntimeError as e:
            print(f"An error occurred during Shonan::run: with pMax {pMax}")
        
        if pMax >= max_iter:
            raise RuntimeError(f"Shonan averaging failed after {max_iter} iterations")
    
    return abs_rotat



def calculate_relative_rotation(abs_rotat, anchor_idx, num_of_parts):
    """
    Calculate relative rotation
    Args:
        abs_rotat (gtsam.Values): absolute rotations
        anchor_idx (int): index of the anchor object
        num_of_parts (int): number of parts
        device (torch.device): device
    Returns:
        relative_rotation (gtsam.Values): relative rotations
    """
    list_of_relative_rotations = []
    anchor_rot = np.array(abs_rotat.atRot3(anchor_idx).matrix())

    for ith_obj in range(num_of_parts):
        obj_rot = np.array(abs_rotat.atRot3(ith_obj).matrix())

        # Align obj in anchor coordinate system, then rotate object
        # Result from shonan averaging is rotation from local to world coordinate system
        # Hence, for real obj rotation, we need to invert the rotation
        relative_rotation = np.linalg.inv(obj_rot) @ anchor_rot
        
        list_of_relative_rotations.append(relative_rotation)
    
    return list_of_relative_rotations
    


def optimize_translation_after_shonan_averaging(list_of_relative_rotations, factors, anchor_idx, uncertainty_dict, scale=1e-2):
    """
    Optimize translation after shonan averaging
    Because shonan averaging only gives relative rotation, we need to optimize translation to make the assembled point cloud

    Args:
        list_of_relative_rotations (list): list of the relative rotation
        factors (gtsam.BetweenFactorPose3s): factors of the problem
        anchor_idx (int): index of the anchor object
        uncertainty_dict (dict): dictionary of the uncertainty
    Returns:
        abs_trans (gtsam.Values): absolute translations
    """

    graph = gtsam.GaussianFactorGraph()

    # We assume that the coordinate system of the anchor object is the world coordinate system
    graph.add(anchor_idx, np.eye(3), np.zeros((3,)), gtsam.noiseModel.Unit.Create(3))

    # Add a factor saying t_j - t_i = Ri * t_ij for all edges (i,j)
    for idx in range(len(factors)):
        factor = factors[idx]
        keys = factor.keys()
        src_i, trg_j, Tij = keys[0], keys[1], factor.measured()
        assert src_i != trg_j, f"src_i: {src_i}, trg_j: {trg_j}"

        relative_rot = list_of_relative_rotations[trg_j]
        measured = relative_rot @ Tij.translation()
        
        # Relative translation must be kept
        # In anchor coordinate system, translation is src_i - trg_j
        # This will be kept even after rotation, which is relative_rot @ Tij.translation()
        graph.add(src_i, np.eye(3), trg_j, -np.eye(3), measured, gtsam.noiseModel.Diagonal.Variances(uncertainty_dict[f"{src_i}-{trg_j}"] * scale * np.ones(3)))


    # Solve linear system
    result_translations = []
    translations = graph.optimize()
    for i in range(translations.size()):
        result_translations.append(translations.at(i))


    return result_translations


def make_relative_transformation_dict(list_of_relative_rotations, list_of_relative_translations, anchor_idx, num_of_parts, device):
    """
    Make relative transformation dictionary
    Args:
        list_of_relative_rotations (list): list of the relative rotation
        list_of_relative_translations (list): list of the relative translation
        anchor_idx (int): index of the anchor object
        num_of_parts (int): number of parts
        device (torch.device): device
    Returns:
        relative_transformation_dict (dict): dictionary of the relative transformation
    """
    anchor_trans = list_of_relative_translations[anchor_idx]

    relative_transformation_dict = {}
    for ith_obj in range(num_of_parts):
        if ith_obj == anchor_idx:
            continue
        
        rot_matrix = torch.tensor(list_of_relative_rotations[ith_obj], device=device).float()
        trans_vector = torch.tensor(list_of_relative_translations[ith_obj] - anchor_trans, device=device).float()
        relative_transformation_dict[f"{ith_obj}-{anchor_idx}"] = (rot_matrix, trans_vector)

    return relative_transformation_dict