"""
General Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch


def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


def bincount2batch(bincount):
    return torch.arange(len(bincount), device=bincount.device, dtype=torch.long).repeat_interleave(bincount)


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def batch_scaling(batch):
    """
    Args:
        batch (torch.Tensor): (B, N+M), batch index of the point cloud
    Returns:
        batch_scaled_batch (torch.Tensor): (B, N+M), batch index of the point cloud
    """
    max_val_batch = batch.max(dim=1)[0] # (batch_size,)

    next_start_index = torch.cumsum(max_val_batch + 1, dim=0) # (batch_size,)
    next_start_index = torch.cat([torch.tensor([0], device=next_start_index.device), next_start_index[:-1]], dim=0) # (batch_size)

    # Originally, batch info include [0,0, ... 1,,] for each batch
    # We want to flatten it and distinguish the batch index
    # Hence, final flattened batch info should be [0,0, ... 1,1, ... 2,2, ...3,3,...]
    batch_scaled_batch = next_start_index[:, None] + batch # (batch_size, num_points)
    return batch_scaled_batch


def extract_by_batch_index(tensor_value, batch_info, target_batch_idx):
    """
    Args:
        tensor_value (torch.Tensor): (N, ...)
        batch_info (torch.Tensor): (N, )
        target_batch_idx (int): target batch index
    Returns:
        tensor_value (torch.Tensor): (N, ...)
    """
    assert len(tensor_value) == len(batch_info), f"len(tensor_value): {len(tensor_value)}, len(batch_info): {len(batch_info)}"
    assert target_batch_idx <= batch_info.max(), f"target_batch_idx: {target_batch_idx}, batch_info.max(): {batch_info.max()}"
    selected_part = tensor_value[batch_info == target_batch_idx]
    return selected_part


def extract_all_objects(tensor, batch_info):
    """
    Args:
        tensor (torch.Tensor): (N, ...)
        batch_info (torch.Tensor): (N, )
        target_batch_idx (int): target batch index
    Returns:
        tensor (torch.Tensor): (N, ...)
    """
    assert len(tensor) == len(batch_info), f"len(tensor): {len(tensor)}, len(batch_info): {len(batch_info)}"

    max_num_obj = batch_info.max()

    result_objs = []
    for obj_idx in range(max_num_obj + 1):
        selected_part = extract_by_batch_index(tensor, batch_info, obj_idx)
        result_objs.append(selected_part)
    
    return result_objs


def extract_all_objects_by_offset(tensor, offset):
    """
    Args:
        tensor (torch.Tensor): (N, ....)
        offset (torch.Tensor): (B, )
        target_batch_idx (int): target batch index
    Returns:
        tensor (torch.Tensor): (N, ...)
    """
    all_objs = []
    for obj_idx in range(offset.shape[0]-1):
        obj = tensor[offset[obj_idx]:offset[obj_idx+1]]
        all_objs.append(obj)
    return all_objs
    