import os
import torch
from common.metric_utils import *
from common.misc import extract_all_objects
from common.utils import is_trg_larger, pairwise_mating
from common.viz import save_pcd_for_light_visualization, draw_frames, draw_normal_error_histogram


def run_evaluation(in_dict, out_dict, settings_dict, func_for_pred, mode):
    """
    Evaluate the progress of the model
    Batch size must be 1 for evaluation

    Args:
        in_dict (dict): it is same as forward_pass
        out_dict (dict): it is same as forward_pass
        settings_dict (dict): settings for evaluation
            - ckp_dir: (str)
            - pos_radius: (float)
            - seg_head_mode: (str)
            - use_seg_result: (bool)
            - success_criterion_in_degree: (int)
            - move_smaller: (bool)
            - use_RANSAC: (bool)
            - use_predicted_normal: (bool)
            - infer_topk: (int)
            - visualize_mode: (str)
            - viz_metric_name: (str)
            - viz_metric_threshold: (float)
            - viz_max_arrow_num: (int)
            - viz_epoch: (int)
            - trainer_sanity_checking: (bool)
            - trainer_global_rank: (int)
            - current_epoch: (int)
            - trainer_max_epochs: (int)
        mode (str): 'val' or 'test'
    """

    assert mode in ['val', 'test'], f"mode must be in ['val', 'test'], but got {mode}"
    assert in_dict['pcd'].shape[0] == 1, f"in_dict['pcd'].shape[0]: {in_dict['pcd'].shape[0]}, must be 1"

    # Postprocess input/output to fit the evaluation function
    # Dataloader will returns (B, N+M, ....) format.
    # However, batch size must be 1 for evaluation
    # So, we will use src/trg individually for evaluation
    src_pcd_raw, trg_pcd_raw = extract_all_objects(in_dict['pcd'][0], in_dict['pcd_batch_info'][0]) # (N, 3), (M, 3)
    src_pcd, trg_pcd = extract_all_objects(in_dict['pcd_t'][0], in_dict['pcd_batch_info'][0]) # (N, 3), (M, 3)
    src_ori, trg_ori = extract_all_objects(out_dict['oris'][0], in_dict['pcd_batch_info'][0]) # (N, 3, 3), (M, 3, 3)
    gt_src_normals, gt_trg_normals = extract_all_objects(in_dict['gt_normals'][0].float(), in_dict['pcd_batch_info'][0]) # (N, 3), (M, 3)
    num_src_pcd, num_trg_pcd = src_pcd.shape[0], trg_pcd.shape[0] # (N), (M)
    out_shape_matching_scores = out_dict['shape_matching_scores'][0] # (N+M, N+M)
    out_matching_scores_drop = out_dict['matching_scores_drop'][0] # (N+M, N+M)
    out_active_mask = out_dict['active_mask'][0] # (N+M, N+M)
    out_mating_surface_seg_results = out_dict['mating_surface_seg_results'][0] if out_dict['mating_surface_seg_results'] is not None else None # (N+M) or None

    # Postprocess matching scores to make its shape (N, M)
    postprocessed_shape_matching_scores = out_shape_matching_scores[out_active_mask] # (N*M)
    postprocessed_matching_scores_drop = out_matching_scores_drop[out_active_mask] # (N*M)
    postprocessed_shape_matching_scores = postprocessed_shape_matching_scores.reshape(num_src_pcd, num_trg_pcd) # (N, M)
    postprocessed_matching_scores_drop = postprocessed_matching_scores_drop.reshape(num_src_pcd, num_trg_pcd) # (N, M)

    if settings_dict['use_seg_result']:
        pred_mating_surface = out_mating_surface_seg_results > 0.5 # (N+M,)
        src_seg_result = pred_mating_surface[:num_src_pcd] # (N,)
        trg_seg_result = pred_mating_surface[num_src_pcd:] # (M,)
        src_trg_seg_result = torch.logical_and(src_seg_result[:,None], trg_seg_result[None,:]) # (N,1) and (1, M) -> (N, M)
        postprocessed_matching_scores_drop = (postprocessed_matching_scores_drop + src_trg_seg_result.float()) / 2

    # Calculate ground truth correspondence
    coord_dist = torch.cdist(src_pcd_raw, trg_pcd_raw, p=2) # (N, M)
    positive_mask = coord_dist < settings_dict['pos_radius'] # (N, M)
    gt_corr = torch.nonzero(positive_mask) # (corr, 2)

    # Save split tensors for evaluating prediction
    split_input_dict = {
        'src_pcd_raw': src_pcd_raw, # (N, 3)
        'trg_pcd_raw': trg_pcd_raw, # (M, 3)
        'src_pcd': src_pcd, # (N, 3)
        'trg_pcd': trg_pcd, # (M, 3)
        'src_ori': src_ori, # (N, 3, 3)
        'trg_ori': trg_ori, # (M, 3, 3)
        'gt_src_normals': gt_src_normals, # (N, 3)
        'gt_trg_normals': gt_trg_normals, # (M, 3)
        'gt_corr': gt_corr, # (corr, 2)
    }

    # Point cloud registration
    src_predicted_frame = src_ori if settings_dict['use_predicted_normal'] else None # (N, 3, 3)
    trg_predicted_frame = trg_ori if settings_dict['use_predicted_normal'] else None # (M, 3, 3)

    if settings_dict['use_RANSAC']:
        estimated_transform, used_corr = func_for_pred(in_dict=in_dict, 
                                                       shape_matching_scores=postprocessed_shape_matching_scores, 
                                                       src_pcd=src_pcd, 
                                                       trg_pcd=trg_pcd, 
                                                       src_predicted_frame=src_predicted_frame,
                                                       trg_predicted_frame=trg_predicted_frame)
    else:
        estimated_transform, used_corr = func_for_pred(src_pcd.unsqueeze(0), trg_pcd.unsqueeze(0), postprocessed_matching_scores_drop.unsqueeze(0))

    # estimated_transform: target_point = R * source_point + t
    out_dict['estimated_rotat'] = estimated_transform[:3, :3] # R, (3,3)
    out_dict['estimated_trans'] = estimated_transform[:3, 3] # t, (3)
    out_dict['used_corr'] = used_corr # (K, 2)

    # Evaluation
    eval_dict = evaluate_prediction(in_dict, split_input_dict, out_dict, settings_dict, mode)

    # Matching Recall
    eval_dict.update(calculate_recall(postprocessed_matching_scores_drop, gt_corr))

    # Calculate ratio of GT among topk scores
    eval_dict['gt_among_topk'] = calculate_ratio_of_gt_among_topk_scores(src_pcd_raw, trg_pcd_raw, postprocessed_matching_scores_drop, topk=settings_dict['infer_topk'], pos_radius=settings_dict['pos_radius'])

    # log size of gt_corr
    eval_dict['gt_corr_size'] = torch.tensor(gt_corr.shape[0]).to(src_pcd_raw.device)

    # Calculate accuracy of segmentation results
    if settings_dict['seg_head_mode'] != 'none':
        eval_dict['seg_recall'], eval_dict['seg_precision'] = calculate_accuracy_of_seg_results(out_mating_surface_seg_results, positive_mask)

        if (eval_dict['seg_recall'] + eval_dict['seg_precision']) > 0:
            eval_dict['seg_F1_score'] = 2 * (eval_dict['seg_recall'] * eval_dict['seg_precision']) / (eval_dict['seg_recall'] + eval_dict['seg_precision'])
        else:
            eval_dict['seg_F1_score'] = torch.tensor(0.0).to(src_pcd_raw.device)

    return out_dict, eval_dict



def evaluate_prediction(in_dict, split_input_dict, out_dict, settings_dict, mode,):
    """
    Args:
        in_dict (dict): it is same as forward_pass
        split_input_dict (dict): split input dictionary for evaluation
        out_dict (dict): it is same as forward_pass
        settings_dict (dict): settings for evaluation
        mode (str): 'val' or 'test'

    Returns:
        eval_result (dict):
            - cd (float): CD between prediction & ground-truth
            - rrmse (float): MSE between prediction & ground-truth for rotation (in degree)
            - trmse (float): MSE between prediction & ground-truth for translation (in cm)
            - crd (float): CoRrespondence Distance (CRD) betwween prediction & ground-truth
    """
    assert mode in ['val', 'test'], f"mode must be in ['val', 'test'], but got {mode}"

    # Init return buffer
    eval_result = {}
    
    pred_relative_trsfm = out_dict['estimated_rotat'].float(), out_dict['estimated_trans'].float() # (3, 3), (3)
    grtr_relative_trsfm = [x for x in in_dict['relative_trsfm']['0-1']] # (3, 3), (3)
    src_pcd, trg_pcd = split_input_dict['src_pcd'], split_input_dict['trg_pcd'] # (N, 3), (M, 3)
    gt_corr = split_input_dict['gt_corr'] # (corr, 2)
    used_corr = out_dict['used_corr'] # (K, 2)


    # Move larger point cloud
    if settings_dict['move_smaller'] and not is_trg_larger(src_pcd, trg_pcd):
        # if source point cloud is bigger than target point cloud, we want to move trg to src
        # However, our code is designed to move src to trg
        # So, we need to inverse the relative transformation
        # trg = R * src + t -> src = R^T * (trg - t) -> src = R^T * trg - R^T * t
        src_pcd, trg_pcd = trg_pcd, src_pcd
        pred_relative_trsfm = pred_relative_trsfm[0].T, -  pred_relative_trsfm[0].T @ pred_relative_trsfm[1]
        grtr_relative_trsfm = grtr_relative_trsfm[0].T, -  grtr_relative_trsfm[0].T @ grtr_relative_trsfm[1]
        gt_corr = torch.stack([gt_corr[:,1], gt_corr[:,0]], dim=1) # (P,) stack (P,) -> (P,2)
        used_corr = torch.stack([used_corr[:,1], used_corr[:,0]], dim=1) # (K, 2)
        is_swap_triggered = True
    
    else:
        is_swap_triggered = False


    # Assemble using prediction, pseudo-gt, and ground-truth
    assm_pred, pcds_pred = pairwise_mating(src_pcd, trg_pcd, pred_relative_trsfm[0], pred_relative_trsfm[1])
    assm_grtr, pcds_grtr = pairwise_mating(src_pcd, trg_pcd, grtr_relative_trsfm[0], grtr_relative_trsfm[1])

    assm_pred, assm_grtr = assm_pred.float(), assm_grtr.float()
    
    # (a) Compute CD between prediction & ground-truth
    eval_result['cd'] = chamfer_distance(assm_pred, assm_grtr)

    # (b) Compute MSE between prediction & ground-truth for rotation (in degree) and translation
    eval_result['rrmse_rpf'], eval_result['trmse_rpf'] = transformation_error_RPFver(pcds_pred, pcds_grtr)
    eval_result['rrmse'], eval_result['trmse'] = transformation_error(pred_relative_trsfm, grtr_relative_trsfm)
    eval_result['rrmse_geo'], eval_result['trmse_geo'] = transformation_error_geodesic(pred_relative_trsfm, grtr_relative_trsfm)

    # (c) Compute CoRrespondence Distance (CRD) betwween prediction & ground-truth
    eval_result['crd'] = correspondence_distance(assm_pred, assm_grtr)

    # (d) Compute Normal Error
    eval_result['n_error'], normal_error_hist, eval_result['n_suc_rate'] = normal_error(in_dict, out_dict, success_criterion_in_degree=settings_dict['success_criterion_in_degree'])

    if mode == 'test':
        if settings_dict['viz_metric_name'] == 'none':
            # Visualization is only depend on self.visualize
            metric_based_visualization = True
        else:
            # Only visualize if the metric is greater than the threshold
            metric_based_visualization = eval_result[settings_dict['viz_metric_name']] >= settings_dict['viz_metric_threshold']
            # metric_based_visualization = in_dict['filepath'][0] == 'everyday/Bottle/d851cbc873de1c4d3b6eb309177a6753/mode_1'

    if (mode =='val' and (not settings_dict['trainer_sanity_checking']) and \
        settings_dict['trainer_global_rank'] == 0 and \
        (settings_dict['visualize_mode'] != 'none') and \
        (settings_dict['current_epoch'] % settings_dict['viz_epoch'] == 0 or settings_dict['current_epoch'] == settings_dict['trainer_max_epochs']-1) and \
        in_dict['eval_idx'][0].item() == 0) or \
        (mode =='test' and (settings_dict['visualize_mode'] != 'none') and metric_based_visualization):
        # Do not visualize in sanity checking
        # Only rank 0 should do visualization to avoid file I/O conflicts in DDP
        # Visualize for every self.viz_epoch
        # However, if it is the last epoch, then visualize
        # Also, only visualize first batch

        # Name of case
        case_name = in_dict["filepath"][0].replace('/', '_')

        vis_folder = os.path.join(settings_dict['ckp_dir'], 'vis', f'GPU_{settings_dict['trainer_global_rank']}', mode, case_name) # For mesh visualization
        vis_hist_folder = os.path.join(settings_dict['ckp_dir'], 'vis_hist', f'GPU_{settings_dict['trainer_global_rank']}', mode, case_name) # For normal error histogram visualization
        os.makedirs(vis_folder, exist_ok=True)
        os.makedirs(vis_hist_folder, exist_ok=True)

        # PCD light visualization
        save_pcd_for_light_visualization(pcds_pred, gt_corr, used_corr, f'{vis_folder}/E{settings_dict['current_epoch']}_{in_dict['eval_idx'][0].item()}_{in_dict['obj_class'][0]}_{round(eval_result['crd'].item(),3)}_pred_top{settings_dict['infer_topk']}')
        save_pcd_for_light_visualization(pcds_grtr, gt_corr, used_corr, f'{vis_folder}/E{settings_dict['current_epoch']}_{in_dict['eval_idx'][0].item()}_{in_dict['obj_class'][0]}_{round(eval_result['crd'].item(),3)}_grtr_top{settings_dict['infer_topk']}')


        # MESH AND FRAME VISUALIZATION
        if settings_dict['visualize_mode'] == 'all':
            output_src_ori, output_trg_ori = split_input_dict['src_ori'], split_input_dict['trg_ori'] # (N, 3, 3), (M, 3, 3)
            gt_src_normals, gt_trg_normals = split_input_dict['gt_src_normals'], split_input_dict['gt_trg_normals'] # (N, 3), (M, 3)
            src_mesh_verts, trg_mesh_verts = in_dict['mesh_t'][0].float(), in_dict['mesh_t'][1].float() # (N,3), (M,3)
            src_mesh_faces, trg_mesh_faces = in_dict['mesh_faces'][0].float(), in_dict['mesh_faces'][1].float() # (F,3), (F',3)
            
            if is_swap_triggered: # To move smaller one, we swap src and trg in the above part
                output_src_ori, output_trg_ori = output_trg_ori, output_src_ori
                gt_src_normals, gt_trg_normals = gt_trg_normals, gt_src_normals
                src_mesh_verts, trg_mesh_verts = trg_mesh_verts, src_mesh_verts
                src_mesh_faces, trg_mesh_faces = trg_mesh_faces, src_mesh_faces
            
            mesh_faces_for_viz = [src_mesh_faces, trg_mesh_faces]

            reshaped_output_src_ori = output_src_ori.reshape(-1,3) # (N,3,3) -> (N*3,3)
            reshaped_output_trg_ori = output_trg_ori.reshape(-1,3) # (M,3,3) -> (M*3,3)

            zero_trans = torch.zeros(3).to(grtr_relative_trsfm[0].device)

            # Rotate by using gt
            _, rot_frame_ori_in_gt = pairwise_mating(reshaped_output_src_ori, reshaped_output_trg_ori, grtr_relative_trsfm[0], zero_trans)
            _, rot_gt_normals_in_gt = pairwise_mating(gt_src_normals, gt_trg_normals, grtr_relative_trsfm[0], zero_trans)
            _, rot_mesh_verts_in_gt = pairwise_mating(src_mesh_verts, trg_mesh_verts, grtr_relative_trsfm[0], grtr_relative_trsfm[1])


            # DRAW FRAME by using gt
            draw_frames(mesh_verts=rot_mesh_verts_in_gt, mesh_faces=mesh_faces_for_viz, 
                        frame_ori=rot_frame_ori_in_gt, gt_normals=rot_gt_normals_in_gt, pcds_list=pcds_grtr, dir_path=vis_folder,
                        filename=f'E{settings_dict['current_epoch']}_{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["crd"].item(),3)}_in_gt',
                        viz_max_arrow_num=settings_dict['viz_max_arrow_num'],
                        viz_piece=True, viz_full=True)

            # Rotate by using pred
            _, rot_frame_ori_in_pred = pairwise_mating(reshaped_output_src_ori, reshaped_output_trg_ori, pred_relative_trsfm[0], zero_trans)
            _, rot_gt_normals_in_pred = pairwise_mating(gt_src_normals, gt_trg_normals, pred_relative_trsfm[0], zero_trans)
            _, rot_mesh_verts_in_pred = pairwise_mating(src_mesh_verts, trg_mesh_verts, pred_relative_trsfm[0], pred_relative_trsfm[1])

            # DRAW FRAME by using prediction
            draw_frames(mesh_verts=rot_mesh_verts_in_pred, mesh_faces=mesh_faces_for_viz, 
                        frame_ori=rot_frame_ori_in_pred, gt_normals=rot_gt_normals_in_pred, pcds_list=pcds_pred, dir_path=vis_folder,
                        filename=f'E{settings_dict['current_epoch']}_{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["crd"].item(),3)}_in_pred',
                        viz_max_arrow_num=settings_dict['viz_max_arrow_num'],
                        viz_piece=False, viz_full=True)
            

            # DRAW NORMAL ERROR HISTOGRAM
            draw_normal_error_histogram(normal_error_hist=normal_error_hist, dir_path=vis_hist_folder, 
                                        filename=f'E{settings_dict['current_epoch']}_{in_dict["eval_idx"].item()}_{in_dict["obj_class"][0]}_{round(eval_result["n_error"].item(),3)}_hist.png')
        

        # exit("stop")
    return eval_result
