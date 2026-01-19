import torch
import torch.nn as nn
from einops import rearrange
from common.rotation import rodrigues_to_rotmat, rotate_by_rotation_matrix
from common.rotation import gram_schmidt, gram_schmidt_with_cross, src_reverse_trg_normal_gram_schmidt_with_cross
from common.misc import batch2offset, offset2bincount

import flash_attn


def get_feats_and_oris(backbone, ori_backbone, equi_layer, proj, normal_pred_mode, flip_normal_mode, only_train_normal, pcd_input, pcd_batch_info, batch_scaled_pcd_batch_info):
    """Get equivariant features and orientation matrices
    
    Args:
        backbone (nn.Module): SO(3)-Equivariant Feature Extractor
        ori_backbone (nn.Module): SO(3)-Equivariant Feature Extractor for orientation matrices
        equi_layer (nn.Module): Layer for Equivariant feature
        proj (nn.Module): Layer for Basis Vector Projection
        normal_pred_mode (str): 'cross' or 'gram'
        flip_normal_mode (str): 'right', 'rightv1_2', 'rightv1_3', 'rightv2', 'rightv3', 'rightv4', 'rightv5', 'mix', 'none'
        only_train_normal (bool): Whether to only train the normal vector
        pcd_input (torch.Tensor): (B, N+M, 3)
        pcd_batch_info (torch.Tensor): (B, N+M, )
        batch_scaled_pcd_batch_info (torch.Tensor): (B, N+M, )
    
    returns:
        equi_feats (torch.Tensor): (B, C, 3, N+M)
        oris (torch.Tensor): (B, N+M, 3, 3)
    """

    if not (only_train_normal and (ori_backbone is not None)):
        # 1. SO(3)-Equivariant Feature Extractor
        equi_feats_backbone = backbone(pcd_input, batch_scaled_pcd_batch_info) # (B, C, 3, N+M)


    if not only_train_normal:
        # 2. Calculate equivariant shape features
        equi_feats = equi_layer(equi_feats_backbone.unsqueeze(-1)).squeeze(-1) # (B, C, 3, N+M)
    
    # 3. Frame Prediction
    equi_feats_ori_backbone = ori_backbone(pcd_input, batch_scaled_pcd_batch_info) if ori_backbone is not None else equi_feats_backbone

    # 3-1. Merge global information by averaging
    # (B, C, 3, N+M) -> (B, C, 3, 1) -> (B, C, 3, N+M)
    equi_feats_ori_backbone_mean = equi_feats_ori_backbone.mean(dim=-1, keepdim=True).expand(equi_feats_ori_backbone.size())

    # 3-2. Basis Vector Projection, those vectors will be used as frame basis vectors
    # (B, C, 3, N+M) concat (B, C, 3, N+M) ->  (B, 2C, 3, N+M) -> (B, 2C, 3, N+M, 1) -> (B, 2, 3, N+M, 1) -> (B, 2, 3, N+M) -> (B, N+M, 2, 3)
    vecs = proj(torch.cat((equi_feats_ori_backbone, equi_feats_ori_backbone_mean), dim=1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2) 


    # 4. Gram Schmidt & Cross-product, this is for making three basis vectors by using two predicted vectors
    if normal_pred_mode == 'cross':
        if flip_normal_mode in ['rightv4', 'rightv5']:
            oris = src_reverse_trg_normal_gram_schmidt_with_cross(vecs, pcd_batch_info) # (B, N+M, 3, 3)
        else:
            oris = gram_schmidt_with_cross(vecs) # (B, N+M, 2, 3) -> (B, N+M, 3, 3)
    elif normal_pred_mode == 'gram':
        oris = gram_schmidt(vecs) # (B, N+M, 3, 3) -> (B, N+M, 3, 3)
    else:
        raise ValueError(f"normal_pred_mode must be in ['cross', 'gram'], but got {normal_pred_mode}")

    return equi_feats, oris



def make_inv_feats(oris, oris_batch_info, equi_feats, flip_normal_mode, flip_mode='src'):
    """Make invariant features
    Assume there are two objects in the batch
    
    Args:
        oris (torch.Tensor): (B, N+M, 3, 3)
        oris_batch_info (torch.Tensor): (B, N+M, ), batch index of the point cloud
        equi_feats (torch.Tensor): (B, C, 3, N+M)
        flip_normal_mode (str, optional): 'right', 'rightv1_2', 'rightv1_3', 'rightv2', 'rightv3', 'rightv4', 'rightv5', 'mix', 'none'. Defaults to 'none'.
        flip_mode (str, optional): 'src' or 'trg' or 'all'. Defaults to 'src'.
    Returns:
        inv_feats (torch.Tensor): (B, C*3, N)
    """

    if flip_normal_mode in ['right', 'rightv1_2', 'rightv1_3', 'rightv2', 'rightv3', 'rightv4', 'rightv5', 'mix']:
        # (B, N+M, 3, 3)
        if flip_normal_mode == 'right':
            postprocessed_oris = torch.stack([- oris[:, :, 0, :], oris[:, :, 2, :], oris[:, :, 1, :]], dim=-2)
        elif flip_normal_mode == 'rightv1_2':
            rotation_matrix = rodrigues_to_rotmat(oris[:, :, 0, :], torch.ones_like(oris[:, :, 0, 0]) * 90.0)
            rotated_oris = rotate_by_rotation_matrix(oris, rotation_matrix)
            postprocessed_oris = torch.stack([- rotated_oris[:, :, 0, :], rotated_oris[:, :, 1, :], - rotated_oris[:, :, 2, :]], dim=-2)
        elif flip_normal_mode == 'rightv1_3':
            rotation_axis = nn.functional.normalize(oris[:, :, 1, :] + oris[:, :, 2, :], dim=-1) # (B, N, 3)
            rotation_matrix = rodrigues_to_rotmat(rotation_axis, torch.ones_like(oris[:, :, 0, 0]) *  180)
            postprocessed_oris = rotate_by_rotation_matrix(oris, rotation_matrix)
        elif flip_normal_mode == 'rightv2':
            postprocessed_oris = torch.stack([- oris[:, :, 0, :], oris[:, :, 1, :], - oris[:, :, 2, :]], dim=-2)
        elif flip_normal_mode == 'rightv3':
            postprocessed_oris = torch.stack([- oris[:, :, 0, :], - oris[:, :, 1, :], oris[:, :, 2, :]], dim=-2)
        elif flip_normal_mode in ['rightv4', 'mix']:
            postprocessed_oris = torch.stack([- oris[:, :, 0, :], oris[:, :, 1, :], oris[:, :, 2, :]], dim=-2)
        elif flip_normal_mode == 'rightv5':
            postprocessed_oris = - oris
        
        if flip_mode == 'src': # Flip the normal vector of src
            # We assume there are two objects in the batch
            src_batch_info = oris_batch_info == 0 # (B, N+M, )
            result_oris = postprocessed_oris * src_batch_info[:,:,None,None] + oris * (~ src_batch_info)[:,:,None,None]
        
        elif flip_mode == 'trg': # Flip the normal vector of trg
            trg_batch_info = oris_batch_info == 1 # (B, N+M, )
            result_oris = postprocessed_oris * trg_batch_info[:,:,None,None] + oris * (~ trg_batch_info)[:,:,None,None]
        
        elif flip_mode == 'all': # Flip the normal vector of src and trg
            result_oris = postprocessed_oris
        
        else:
            raise ValueError(f"flip_mode must be in ['src', 'trg', 'all'], but got {flip_mode}")
    
    elif flip_normal_mode == 'none':
        result_oris = oris
    
    else:
        raise ValueError(f"flip_normal_mode must be in ['right', 'rightv1_2', 'rightv1_3', 'rightv2', 'rightv3', 'rightv4', 'rightv5', 'mix', 'none'], but got {flip_normal_mode}")
    
    # (B, C, 3, N) -> (B, N, C, 3) @ (B, N, 3, 3) -> (B, N, 3, 3) => (B, N, C, 3)
    inv_feats = torch.matmul(equi_feats.permute(0, 3, 1, 2).float(), result_oris.transpose(-2,-1).float()) 
    inv_feats = rearrange(inv_feats, 'b n c r -> b (c r) n') # (B, N, C, 3) -> (B, C*3, N)
    return inv_feats


def do_feed_forward_seg_head(seg_head_mode, shape_feats, batch_scaled_pcd_batch_info, modules_dict):
    """
    Feed forward the shape features through the segmentation head
    We assume there are two objects in the batch
    When atten, head size is 8

    Args:
        seg_head_mode (str): 'mlp' or 'atten'
        shape_feats (torch.Tensor): (B, D, N+M)
        batch_scaled_pcd_batch_info (torch.Tensor): (B, N+M, ), batch index of the point cloud
        modules_dict (dict): Dictionary containing modules for the segmentation head
            - 'seg_head': Segmentation head
            - 'layer_norm_for_self_atten': Layer normalization for self attention
            - 'layer_norm_for_global_atten': Layer normalization for global attention
            - 'final_layer_norm': Layer normalization for final output
            - 'self_attn_to_qkv': Linear layer for self attention
            - 'global_attn_to_qkv': Linear layer for global attention

    Returns:
        mating_surface_seg_results (torch.Tensor): (B, N+M)
    """
    batch_size, channel_dim, num_of_points = shape_feats.shape

    if seg_head_mode == 'mlp': 
        mating_surface_seg_results = modules_dict['seg_head'](shape_feats) # (B, D, N+M) -> (B, 1, N+M)
        mating_surface_seg_results = nn.functional.sigmoid(mating_surface_seg_results).squeeze(1) # (B, 1, N+M) -> (B, N+M)
    
    elif seg_head_mode == 'atten':
        assert channel_dim % 8 ==0, f"channel_dim must be divisible by 8, but got {channel_dim}"

        # Prepare for Flash Attention
        batch_scaled_offset = batch2offset(batch_scaled_pcd_batch_info.reshape(-1)) # (B*2, )
        cumulative_batch_scaled_offset = torch.concat([torch.zeros(1, device=batch_scaled_offset.device), batch_scaled_offset], dim=0).int() # (B*2+1, )
        size_of_each_object = offset2bincount(batch_scaled_offset) # (B*2)
        local_max_seqlen = size_of_each_object.max()

        # Save the original dtype of shape_feats, because Flash Attention requires fp16 or bf16
        original_dtype = shape_feats.dtype

        transposed_shape_feats = shape_feats.transpose(1, 2) # (B, D, N+M) -> (B, N+M, D)
        layernormed_shape_feats = modules_dict['layer_norm_for_self_atten'](transposed_shape_feats) # (B, N+M, D)

        # Self Attention
        self_atten_qkv = modules_dict['self_attn_to_qkv'](layernormed_shape_feats) # (B, N+M, D) -> (B, N+M, D*3)
        self_atten_qkv = self_atten_qkv.reshape(batch_size*num_of_points, 3, 8, channel_dim//8) # (B, N+M, D*3) -> (B*(N+M), 3, 8, D//8)
        self_atten_out =  flash_attn.flash_attn_varlen_qkvpacked_func(self_atten_qkv.to(torch.float16), cu_seqlens=cumulative_batch_scaled_offset, max_seqlen=local_max_seqlen, dropout_p=0.0) #  (B*(N+M), 8, D//8)
        self_atten_out = self_atten_out.to(original_dtype)
        self_atten_out = self_atten_out.reshape(batch_size, num_of_points, -1) # (B*(N+M), 8, D//8) -> (B, N+M, D)
        self_atten_out = layernormed_shape_feats + self_atten_out # (B, N+M, D)
        layernormed_self_atten_out = modules_dict['layer_norm_for_global_atten'](self_atten_out) # (B, N+M, D)

        # Global Attention
        global_atten_qkv = modules_dict['global_attn_to_qkv'](layernormed_self_atten_out) # (B, N+M, D) -> (B, N+M, D*3)
        global_atten_qkv = global_atten_qkv.reshape(batch_size, num_of_points, 3, 8, channel_dim//8) # (B, N+M, D*3) -> (B, (N+M), 3, 8, D//8)
        global_atten_qkv = global_atten_qkv.to(torch.float16)
        global_atten_out = flash_attn.flash_attn_qkvpacked_func(global_atten_qkv, dropout_p=0.0) # (B, (N+M), 8, D//8)
        global_atten_out = global_atten_out.to(original_dtype)
        global_atten_out = global_atten_out.reshape(batch_size, num_of_points, -1) # (B, (N+M), 8, D//8) -> (B, N+M, D)
        global_atten_out = layernormed_self_atten_out + global_atten_out
        global_atten_out = modules_dict['final_layer_norm'](global_atten_out) # (B, N+M, D)

        # Segmentation Head
        mating_surface_seg_results = modules_dict['seg_head'](global_atten_out).squeeze(dim=-1) # (B, N+M, D) -> (B, N+M)
        mating_surface_seg_results = nn.functional.sigmoid(mating_surface_seg_results) # (B, 1, N+M) -> (B, N+M)

    else:
        raise ValueError(f"seg_head_mode must be in ['mlp', 'atten'], but got {seg_head_mode}")

    
    return mating_surface_seg_results


def return_active_mask(batch_info):
    """
    Return active mask between the different objects

    Args:
        batch_info (torch.Tensor): (B, N+M, ), batch index of the point cloud

    Returns:
        active_parts (torch.Tensor): (B, N+M, N+M), True if the point is active
    """
    # Leave only the matching scores between the different objects
    # Right-Upper part is only left
    num_of_points = batch_info.size(1)
    repeated_batch_info_row_for_src = batch_info[:,:,None].expand(-1, -1, num_of_points) == 0  # (B, N+M, N+M)
    repeated_batch_info_col_for_trg = batch_info[:,None,:].expand(-1, num_of_points, -1) == 1 # (B, N+M, N+M)
    active_parts = torch.logical_and(repeated_batch_info_row_for_src, repeated_batch_info_col_for_trg) # (B, N+M, N+M))
    return active_parts


def calculate_matching_score(src_shape_feats, trg_shape_feats, active_mask, eps=1e-8, mode='CM'):
    """
    Calculate matching score between src and trg features
    Assume there are two objects in the batch

    Args:
        src_shape_feats (torch.Tensor): (B, D, N+M)
        trg_shape_feats (torch.Tensor): (B, D, N+M)
        active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active
        eps (float, optional): Epsilon for avoiding division by zero. Defaults to 1e-8.
        mode (str, optional): 'CM' or 'cossim'. Defaults to 'CM'.
    Returns:
        matching_scores (torch.Tensor): (B, N+M, N+M)
    """
    assert src_shape_feats.shape[1] == trg_shape_feats.shape[1], f"src_shape_feats.shape: {src_shape_feats.shape}, trg_shape_feats.shape: {trg_shape_feats.shape}"
    assert src_shape_feats.shape[2] == active_mask.shape[1], f"src_shape_feats.shape: {src_shape_feats.shape}, trg_shape_feats.shape: {active_mask.shape}"
    assert trg_shape_feats.shape[2] == active_mask.shape[2], f"trg_shape_feats.shape: {trg_shape_feats.shape}, active_mask.shape: {active_mask.shape}"

    if mode == 'CM':
        matching_scores = torch.einsum('b c n , b c m -> b n m', src_shape_feats, trg_shape_feats) # (B, N+M, N+M)
        matching_scores = matching_scores / (src_shape_feats.shape[1] ** 0.5 + eps) # 1e-8 is for avoiding division by zero
    
    else:
        normalized_src_shape_feats = nn.functional.normalize(src_shape_feats, p=2, dim=1) # (B, D, N+M)
        normalized_trg_shape_feats = nn.functional.normalize(trg_shape_feats, p=2, dim=1) # (B, D, N+M)
        matching_scores = torch.einsum('b c n , b c m -> b n m', normalized_src_shape_feats, normalized_trg_shape_feats) # (B, N+M, N+M)

    # Remove the matching scores between the same objects
    matching_scores = matching_scores * active_mask
    return matching_scores


def do_multibatch_optimal_transport(matching_scores, batch_info, active_mask, auxiliary_info_dict, mode='sinkhorn'):
    """
    Calculate optimal transport between multiple batches

    Args:
        matching_scores (torch.Tensor): (B, N+M, N+M), inactive parts are already removed
        batch_info (torch.Tensor): (B, N+M, ), batch index of the point cloud
        active_mask (torch.Tensor): (B, N+M, N+M), True if the point is active
        auxiliary_info_dict (dict): Dictionary containing auxiliary information for optimal transport
            - 'optimal_transport': Optimal transport module
            - 'softmax_temperature': Softmax temperature
            - 'slack_variable': Slack variable
        mode (str, optional): 'sinkhorn', 'softmax', 'none'. Defaults to 'sinkhorn'.
    
    Returns:
        result (torch.Tensor): 
        - (B, N+M+1, N+M+1) if mode is ['sinkhorn', 'softmax']
        - (B, N+M, N+M) if mode is 'none'
    """

    batch_size, row_size, col_size = matching_scores.shape

    if mode == 'sinkhorn':
        result_list = []

        for batch_idx in range(batch_size):
            # Postprocess matching scores to make its shape (N, M)
            pcd_num_info = batch_info[batch_idx].bincount() # (2, )
            assert len(pcd_num_info) == 2, f"There must be two objects in the batch, but got {len(pcd_num_info)}"
            
            num_src_pcd, num_trg_pcd = pcd_num_info
            postprocessed_matching_scores = matching_scores[batch_idx][active_mask[batch_idx]] # (N*M,)
            postprocessed_matching_scores = postprocessed_matching_scores.reshape(1, num_src_pcd, num_trg_pcd) # (1, N, M)

            normalized_matching_scores = auxiliary_info_dict['optimal_transport'](postprocessed_matching_scores).squeeze(0) # (1, N+1, M+1) -> (N+1, M+1)

            # Recover shape
            place_holder = torch.zeros(row_size+1, col_size+1, device=matching_scores.device)
            place_holder[:num_src_pcd, (col_size-num_trg_pcd):-1] = normalized_matching_scores[:-1,:-1]
            place_holder[:num_src_pcd,-1] = normalized_matching_scores[:-1,-1]
            place_holder[-1,(col_size-num_trg_pcd):-1] = normalized_matching_scores[-1,:-1]
            place_holder[-1,-1] = normalized_matching_scores[-1,-1]

            result_list.append(place_holder)
        
        result = torch.stack(result_list, dim=0) # (B, N+M+1, N+M+1)
    
    
    elif mode == 'softmax':
        # Calculate active mask
        place_holder_active_mask = torch.zeros(batch_size, row_size+1, col_size+1, device=matching_scores.device, dtype=torch.bool) # (B, N+M+1, N+M+1)
        place_holder_active_mask[:, :-1, :-1] = active_mask # (B, N+M, N+M)
        place_holder_active_mask[:, :-1, -1] = active_mask.any(dim=-1) # (B, N+M)
        place_holder_active_mask[:, -1, :-1] = active_mask.any(dim=-2) # (B, N+M)
        place_holder_active_mask[:, -1, -1] = True

        # Calculate padded matching scores
        place_holder = torch.zeros(batch_size, row_size+1, col_size+1, device=matching_scores.device) # (B, N+M+1, N+M+1)
        place_holder[:, :-1, :-1] = matching_scores # (B, N+M, N+M)
        place_holder[:, :-1, -1] = auxiliary_info_dict['slack_variable'].expand(batch_size, row_size)
        place_holder[:, -1, :] = auxiliary_info_dict['slack_variable'].expand(batch_size, col_size+1)
        place_holder = place_holder * place_holder_active_mask + -1e12 * (~place_holder_active_mask)

        row_softmax_matching_scores = nn.functional.softmax(place_holder / auxiliary_info_dict['softmax_temperature'], dim=-1)
        col_softmax_matching_scores = nn.functional.softmax(place_holder / auxiliary_info_dict['softmax_temperature'], dim=-2)
        softmax_matching_scores = (row_softmax_matching_scores + col_softmax_matching_scores) / 2
        softmax_matching_scores[:, :-1, -1] = row_softmax_matching_scores[:, :-1, -1] # Fill the last column with the row softmax matching scores
        softmax_matching_scores[:, -1, :-1] = col_softmax_matching_scores[:, -1, :-1] # Fill the last row with the col softmax matching scores
        softmax_matching_scores = softmax_matching_scores * place_holder_active_mask

        result = softmax_matching_scores

    
    elif mode == 'none':
        result = matching_scores
    
    return result

