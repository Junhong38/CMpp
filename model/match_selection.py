import torch
from scipy.optimize import linear_sum_assignment

def topk_matching(corr_matrix, k=128):
    N, M = corr_matrix.shape
    corr_matrix_drop_1d = corr_matrix.reshape(-1)
    topk_scores, topk_indices = torch.topk(corr_matrix_drop_1d, k=k)

    src_idx = topk_indices // M
    trg_idx = topk_indices % M
    
    return torch.stack([src_idx, trg_idx], dim=1)

def unidirectional_nn_matching(corr_matrix, topk=1):
    # """
    # For each target, choose the top-1 source (argmax over dim=1)
    # Returns: (M, 2) index pairs
    # """
    # N, M = corr_matrix.shape
    # assert M <= N, "Target must be less than or equal to source"
    
    # src_idx = torch.argmax(corr_matrix, dim=0).to(corr_matrix.device)
    # tgt_idx = torch.arange(corr_matrix.size(1)).to(corr_matrix.device)

    # return torch.stack([src_idx, tgt_idx], dim=1)
    values, indices = torch.topk(corr_matrix, k=topk, dim=0)  # shape: (topk, M)

    src_idx = indices                          # (topk, M)
    tgt_idx = torch.arange(corr_matrix.size(1)).repeat(topk, 1).to(corr_matrix.device)

    matches = torch.stack([src_idx, tgt_idx], dim=2)  # (topk, M, 2)
    matches = matches.permute(1, 0, 2).reshape(-1, 2)  # (topk * M, 2)
    return matches

def injective_matching(corr_matrix):
    """
    Greedy injective matching: for each source, choose best target
    but do not allow target duplication.
    If N > M, transpose the matrix and flip the result.
    Returns: (≤ min(N, M), 2)
    """
    N, M = corr_matrix.shape
    
    # If N > M, transpose the matrix
    if N > M:
        corr_matrix = corr_matrix.T
        N, M = M, N
        need_flip = True
    else:
        need_flip = False
    
    matched = torch.zeros(M, dtype=torch.bool, device=corr_matrix.device)
    matches = []

    for i in range(N):
        scores = corr_matrix[i]
        sorted_tgt = torch.argsort(scores, descending=True)
        for j in sorted_tgt:
            if not matched[j]:
                matches.append((i, j.item()))
                matched[j] = True
                break
    
    matches = torch.tensor(matches, device=corr_matrix.device)
    
    # If we transposed the matrix, flip the result
    if need_flip:
        matches = matches.flip(dims=[1])
    
    return matches

def bijective_matching(corr_matrix):
    """
    Bijective matching using Hungarian algorithm (scipy)
    Requires N == M
    Returns: (N, 2) index pairs
    """
    N, M = corr_matrix.shape
    assert N == M, "Source and target must have the same number of points"

    corr_np = corr_matrix.detach().cpu().numpy()
    cost = -corr_np  # maximize similarity → minimize negative
    row_ind, col_ind = linear_sum_assignment(cost)
    return torch.tensor(list(zip(row_ind, col_ind)), device=corr_matrix.device)

def mutual_topk_matching(corr_matrix, topk=1):
    """
    Reciprocal test: for each source, choose the top-1 target (argmax over dim=1)
    Only keep (i, j) where:
    j == argmax(corr_matrix[i]) and
    i == argmax(corr_matrix[:, j])
    Returns: (≤ N, 2) index pairs
    """
    # src_top1 = torch.argmax(corr_matrix, dim=1) # (N,)
    # tgt_top1 = torch.argmax(corr_matrix, dim=0) # (M,)
    trg_top_values, trg_indices = torch.topk(corr_matrix, k=topk, dim=1) # (N, topk)
    src_top_values, src_indices = torch.topk(corr_matrix, k=topk, dim=0) # (topk, M)
    src_indices = src_indices.T

    matches = []
    for i, top_j in enumerate(trg_indices):
        for j in top_j:
            if i in src_indices[j]:
                matches.append((i, j.item()))
    return torch.tensor(matches, device=corr_matrix.device)

def soft_topk_matching(corr_matrix, topk=1):
    trg_top_values, trg_indices = torch.topk(corr_matrix, k=topk, dim=1) # (N, topk)
    trg_indices = trg_indices.T # (topk, N)
    src_idx = torch.arange(corr_matrix.size(0)).repeat(topk, 1).to(corr_matrix.device) # (topk, N)
    matches_t2s = torch.stack([src_idx, trg_indices], dim=2) # (topk, N, 2)
    matches_t2s = matches_t2s.permute(1, 0, 2).reshape(-1, 2) # (topk*N, 2)

    src_top_values, src_indices = torch.topk(corr_matrix, k=topk, dim=0) # (topk, M)
    trg_idx = torch.arange(corr_matrix.size(1)).repeat(topk, 1).to(corr_matrix.device)
    matches_s2t = torch.stack([src_indices, trg_idx], dim=2) # (topk, M, 2)
    matches_s2t = matches_s2t.permute(1, 0, 2).reshape(-1, 2) # (topk * M, 2) 

    matches = torch.cat([matches_t2s, matches_s2t], dim=0) # (topk * (M+N), 2)
    matches = torch.unique(matches, dim=0)

    return matches