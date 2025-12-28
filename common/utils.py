r""" Helper functions """
import random
import numpy as np
import open3d as o3d
import torch
import json
import os

from common.viz import global_colors_for_objs


def check_inf_or_nan(tensor, message: str, log=None):
    if torch.isinf(tensor).any():
        assert False, f"Inf found from {message}\n{tensor}"
    if torch.isnan(tensor).any():
        assert False, f"Nan found from {message}\n{tensor}"
    
    if log is not None:
        log(f'DEBUG/{str(message)}-mean', tensor.mean().item(), prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)
        log(f'DEBUG/{str(message)}-max', tensor.max().item(), prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)
        log(f'DEBUG/{str(message)}-min', tensor.min().item(), prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False, batch_size=1)





def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, dict):
            # continue
            for k, v in value.items():
                if isinstance(v[0], torch.Tensor):
                    value[k] = [v_.cuda() for v_ in v]
        elif isinstance(value[0], torch.Tensor):
            batch[key] = [v.cuda() for v in value]
    batch['filepath'] = batch['filepath'][0]
    batch['obj_class'] = batch['obj_class'][0]
    batch['gt_correspondence'] = batch['gt_correspondence'][0]

    if batch.get('n_frac') is not None: batch['n_frac'] = batch['n_frac'][0]
    if batch.get('order') is not None: batch['order'] = batch['order'][0]
    if batch.get('anchor_idx') is not None: batch['anchor_idx'] = batch['anchor_idx'][0]

    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()



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
            placeholder_dict[k_] = v_.detach().cpu().item()
        json_results[k] = placeholder_dict


    with open(os.path.join(dir_path, f"{filename}.json"), 'w') as f:
        json.dump(json_results, f, indent=4)
