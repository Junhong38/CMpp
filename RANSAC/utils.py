import torch
from typing import Set, Tuple


def _squeeze_leading_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Remove a leading singleton batch dimension if present."""
    if tensor is None:
        return None
    if tensor.dim() >= 2 and tensor.size(0) == 1:
        return tensor.squeeze(0)
    return tensor


def _transform_points(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Apply the rigid transform defined by rotation and translation."""
    return torch.einsum('x y, n y -> n x', rotation, points) + translation


def _select_correspondences(inlier_mask: torch.Tensor, dist_mat: torch.Tensor, matching_choice: str) -> torch.Tensor:
    """Select correspondence indices according to the matching strategy."""
    rows, cols = torch.nonzero(inlier_mask, as_tuple=True)
    if rows.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=inlier_mask.device)

    if matching_choice == 'many-to-many':
        return torch.stack((rows, cols), dim=1)

    dists = dist_mat[rows, cols]

    if matching_choice == 'many-to-one':
        pairs = []
        for col in cols.unique(sorted=True).tolist():
            mask = cols == col
            col_rows = rows[mask]
            col_dists = dists[mask]
            best_idx = torch.argmin(col_dists)
            pairs.append((col_rows[best_idx].item(), col))
        return torch.tensor(pairs, dtype=torch.long, device=inlier_mask.device)

    if matching_choice == 'one-to-one':
        order = torch.argsort(dists)
        used_src: Set[int] = set()
        used_trg: Set[int] = set()
        pairs = []
        for idx in order.tolist():
            src_idx = rows[idx].item()
            trg_idx = cols[idx].item()
            if src_idx in used_src or trg_idx in used_trg:
                continue
            used_src.add(src_idx)
            used_trg.add(trg_idx)
            pairs.append((src_idx, trg_idx))
        if not pairs:
            return torch.zeros((0, 2), dtype=torch.long, device=inlier_mask.device)
        return torch.tensor(pairs, dtype=torch.long, device=inlier_mask.device)

    raise ValueError(f"Unknown matching choice: {matching_choice}")


def estimate_rigid_transform(source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the rigid transform that brings ``source`` onto ``target``.

    Args:
        source: Tensor of shape (N, 3), points to be transformed.
        target: Tensor of shape (N, 3), reference points.

    Returns:
        rotation: Tensor of shape (3, 3)
        translation: Tensor of shape (3,)
    """
    if source.shape != target.shape:
        raise ValueError(f"Point sets must share a shape. Got {source.shape} and {target.shape}.")
    if source.numel() == 0:
        raise ValueError("At least one correspondence is required.")

    centroid_source = source.mean(dim=0)
    centroid_target = target.mean(dim=0)

    source_centered = source - centroid_source
    target_centered = target - centroid_target

    h_matrix = source_centered.T @ target_centered

    u, _, v = torch.linalg.svd(h_matrix)
    rotation = v.T @ u.T

    if torch.det(rotation) < 0:
        v[-1, :] *= -1
        rotation = v.T @ u.T

    translation = centroid_target - rotation @ centroid_source

    return rotation, translation