"""
General Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)



@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


@torch.no_grad()
def batch_scaling(batch):
    """
    Args:
        batch (torch.Tensor): (batch_size, num_points), batch index of the point cloud
    Returns:
        batch_scaled_batch (torch.Tensor): (batch_size, num_points), batch index of the point cloud
    """
    max_val_batch = batch.max(dim=1)[0] # (batch_size,)

    next_start_index = torch.cumsum(max_val_batch + 1, dim=0) # (batch_size,)
    next_start_index = torch.cat([torch.tensor([0], device=next_start_index.device), next_start_index[:-1]], dim=0) # (batch_size)

    # Originally, batch info include [0,0, ... 1,,] for each batch
    # We want to flatten it and distinguish the batch index
    # Hence, final flattened batch info should be [0,0, ... 1,1, ... 2,2, ...3,3,...]
    batch_scaled_batch = next_start_index[:, None] + batch # (batch_size, num_points)
    return batch_scaled_batch


@torch.no_grad()
def extract_by_offset_info(tensor, offset_info, target_idx):
    """
    Args:
        tensor (torch.Tensor): (N, ...)
        offset_info (torch.Tensor): (B, )
        target_idx (int): target_idx
    Returns:
        tensor (torch.Tensor): (target offset size, ...)
    """
    print(f"[extract_by_offset_info] tensor: {tensor.shape}, offset_info: {offset_info.shape}, target_idx: {target_idx}")

    # offset info is like [0, 1, 3, 6, 10, ...]
    size_cumsum = torch.cat([torch.tensor([0]).to(tensor.device), torch.cumsum(offset_info, dim=0)], dim=0) # (B + 1, )
    print(f"[extract_by_offset_info] size_cumsum: {size_cumsum.shape}, size_cumsum: {size_cumsum}")
    start_idx = size_cumsum[target_idx]
    end_idx = size_cumsum[target_idx + 1]
    selected_part = tensor[start_idx:end_idx]
    print(f"[extract_by_offset_info] selected_part: {selected_part.shape}")
    return tensor[start_idx:end_idx]


@torch.no_grad()
def extract_by_batch_index(tensor, batch_info, target_batch_idx):
    """
    Args:
        tensor (torch.Tensor): (N, ...)
        batch_info (torch.Tensor): (N, )
        target_batch_idx (int): target batch index
    Returns:
        tensor (torch.Tensor): (N, ...)
    """
    assert len(tensor) == len(batch_info), f"len(tensor): {len(tensor)}, len(batch_info): {len(batch_info)}"
    assert target_batch_idx <= batch_info.max(), f"target_batch_idx: {target_batch_idx}, batch_info.max(): {batch_info.max()}"

    print(f"[extract_by_batch_index] tensor: {tensor.shape}, batch_info: {batch_info.shape}, target_batch_idx: {target_batch_idx}")

    selected_part = tensor[batch_info == target_batch_idx]
    print(f"[extract_by_batch_index] selected_part: {selected_part.shape}")
    return selected_part


@torch.no_grad()
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