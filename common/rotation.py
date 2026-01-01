import torch
import torch.nn.functional as F


def normalize_vector(v):
    r"""
    v: B x N x 3
    """
    v = F.normalize(v, dim=-1)
    return v


def cross_product(u, v):
    r"""
    u: B x N x 3
    v: B x N x 3
    """
    result = torch.cross(u, v, dim=-1)
    return result


def proj_a2u(a,u):
    r"""
    u: B x N x 3
    a: B x N x 3
    proj_u(a) = (a·u)/(u·u) * u
    projection vector a onto u
    """
    # B x N x 3 inner B x N x 3 -> B x N -> B x N x 1, (a·u)
    inner_prod = torch.einsum('...i,...i->...', u, a).unsqueeze(-1)
    # B x N x 3 inner B x N x 3 -> B x N -> B x N x 1, (u·u)
    norm2 = torch.einsum('...i,...i->...', u, u).unsqueeze(-1)
    # (a·u)/(u·u)
    factor = inner_prod / (norm2 + 1e-10)
    return factor * u


def gram_schmidt_with_cross(poses):
    r"""
    poses: B x N x 2 x 3
    optimum: bool, if True, use optimum projection, otherwise use not optimum projection
    """
    first_vec = poses[:, :, 0, :] # (B, N, 3)
    second_vec = poses[:, :, 1, :] # (B, N, 3)

    normalized_first_vec = normalize_vector(first_vec) # (B, N, 3)
    normalized_second_vec = normalize_vector(second_vec - proj_a2u(second_vec, normalized_first_vec)) # (B, N, 3)
    normalized_third_vec = normalize_vector(cross_product(normalized_first_vec, normalized_second_vec)) # (B, N, 3)

    normalized_first_vec = normalized_first_vec[:, :, None, :] # (B, N, 1, 3)
    normalized_second_vec = normalized_second_vec[:, :, None, :] # (B, N, 1, 3)
    normalized_third_vec = normalized_third_vec[:, :, None, :] # (B, N, 1, 3)

    # (B, N, 1, 3) concat (B, N, 1, 3) concat (B, N, 1, 3) -> (B, N, 3, 3)
    result = torch.cat((normalized_first_vec, normalized_second_vec, normalized_third_vec), -2) 
    return result
    

def gram_schmidt(poses):
    r"""
    poses: B x N x 3 x 3
    optimum: bool, if True, use optimum projection, otherwise use not optimum projection
    """
    first_vec = poses[:, :, 0, :] # (B, N, 3)
    second_vec = poses[:, :, 1, :] # (B, N, 3)
    third_vec = poses[:, :, 2, :] # (B, N, 3)

    normalized_first_vec = normalize_vector(first_vec) # (B, N, 3)
    normalized_second_vec = normalize_vector(second_vec - proj_a2u(second_vec, normalized_first_vec)) # (B, N, 3)
    normalized_third_vec = normalize_vector(third_vec - proj_a2u(third_vec, normalized_first_vec) - proj_a2u(third_vec, normalized_second_vec)) # (B, N, 3)

    normalized_first_vec = normalized_first_vec[:, :, None, :] # (B, N, 1, 3)
    normalized_second_vec = normalized_second_vec[:, :, None, :] # (B, N, 1, 3)
    normalized_third_vec = normalized_third_vec[:, :, None, :] # (B, N, 1, 3)

    # (B, N, 1, 3) concat (B, N, 1, 3) concat (B, N, 1, 3) -> (B, N, 3, 3)
    result = torch.cat((normalized_first_vec, normalized_second_vec, normalized_third_vec), -2) 
    return result
