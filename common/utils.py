r""" Helper functions """
import torch
import json
import os



def is_trg_larger(src_pcd, trg_pcd):
    """
    Args:
        src_pcd (torch.Tensor): (N, 3)
        trg_pcd (torch.Tensor): (M, 3)

    Returns:
        bool: True if source point cloud is smaller than target point cloud
    """
    # max - min -> volume
    # Calculate max - min for all xyz coordinates, and product for all xyz.
    # Finally, we can calculate bounding box volume
    src_volume = (src_pcd.max(dim=0)[0] - src_pcd.min(dim=0)[0]).prod(dim=0)
    trg_volume = (trg_pcd.max(dim=0)[0] - trg_pcd.min(dim=0)[0]).prod(dim=0)
    return src_volume < trg_volume


def pairwise_mating(src_pcd, trg_pcd, rotat, trans):
    """
    move src to trg

    Args:
        src_pcd (torch.Tensor): (N, 3)
        trg_pcd (torch.Tensor): (M, 3)
        rotat (torch.Tensor): (3, 3)
        trans (torch.Tensor): (3)

    Returns:
        pcd_t (torch.Tensor): (N+M, 3)
        pcd_t (list): [(N, 3), (M, 3)] if is_trg_larger else [(N, 3), (M, 3)]
    """
    # Remind:
    # estimated_transform: trg_pcd = R * src_pcd + t

    # When GT
    # GT Rt format already fits to R * src + t

    # When pred
    # target_point = R * source_point + t
    # Hence, pred format already fits to R * src + t format

    pcd_t = []
    # Fix target point, and move source point to target point
    # src_pcd_t = R * src_pcd + t
    src_pcd_t = _transform(src_pcd, rotat, trans)
    pcd_t = [src_pcd_t, trg_pcd]
    
    return torch.cat(pcd_t, dim=0), pcd_t


def _transform(pcd, rotat=None, trans=None):
    """
    rotat * pcd + trans

    Args:
        pcd (torch.Tensor): (N, 3)
        rotat (torch.Tensor, optional): (3, 3). Defaults to None.
        trans (torch.Tensor, optional): (3). Defaults to None.

    Returns:
        pcd_t (torch.Tensor): (N, 3) 
    """
    if rotat == None: rotat = torch.eye(3, 3)
    if trans == None: trans = torch.zeros(3)

    rotat = rotat.to(pcd.device)
    trans = trans.to(pcd.device)

    return torch.einsum('x y, n y -> n x', rotat, pcd) + trans



def check_inf_or_nan(tensor, message: str, log=None):
    if torch.isinf(tensor).any():
        assert False, f"Inf found from {message}\n{tensor}"
    if torch.isnan(tensor).any():
        assert False, f"Nan found from {message}\n{tensor}"
    
    if log is not None:
        log(f'DEBUG/{str(message)}-mean', tensor.mean().item(), prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)
        log(f'DEBUG/{str(message)}-max', tensor.max().item(), prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)
        log(f'DEBUG/{str(message)}-min', tensor.min().item(), prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)


def instance_wise_results_to_json(instance_wise_results, dir_path, filename):
    """
    Save instance-wise results to json file.

    Args:
        instance_wise_results (dict): instance-wise results
        dir_path (str): directory path to save
        filename (str): filename to save
    """
    json_results = dict()

    for k, dict_v in instance_wise_results.items():
        placeholder_dict = dict()
        for k_, v_ in dict_v.items():
            placeholder_dict[k_] = v_.cpu().item()
        json_results[k] = placeholder_dict


    with open(os.path.join(dir_path, f"{filename}.json"), 'w') as f:
        json.dump(json_results, f, indent=4)


def divide_parameters_into_ori_and_others(named_parameters):
    ori_parameters = []
    other_parameters = []
    
    for name, param in named_parameters:
        if name.startswith('ori_backbone.') or name.startswith('proj.'):
            ori_parameters.append(param)
        else:
            other_parameters.append(param)
    
    return ori_parameters, other_parameters


