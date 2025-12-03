from typing import Optional

import torch
import torch.nn as nn
from torch_batch_svd import svd

from RANSAC.match_selection import topk_matching, mutual_topk_matching, soft_topk_matching, unidirectional_nn_matching, injective_matching, bijective_matching 



def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)
    
    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    try: U, _, V = svd(H)
    except: 
        print('use torch svd!')
        U, _, V = torch.svd(H.cpu())
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    # eye[:, -1, -1] = torch.sign(torch.det((V @ Ut).to(torch.float32)))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.0, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights=None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
        )


def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points


class LocalGlobalRegistration(nn.Module):
    def __init__(
        self,
        k: int = 128,
        match_option: str = 'topk',
        acceptance_radius: float = 0.1,
        num_refinement_steps: int = 5,
        score_threshold_ratio: float = 0.01,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            match_option (str): 'topk' or 'mutual_topk' or 'soft_topk' or 'unidirectional_nn_matching' or 'injective_matching' or 'bijective_matching'.
            acceptance_radius (float): acceptance radius for LGR.
            num_refinement_steps (int=5): number of refinement steps.
            score_threshold_ratio (float=0.01): score threshold ratio for filtering correspondences. If 0.0, no filtering is performed.
        """
        super(LocalGlobalRegistration, self).__init__()
        self.k = k
        self.match_option = match_option
        self.acceptance_radius = acceptance_radius
        self.num_refinement_steps = num_refinement_steps
        self.score_threshold_ratio = score_threshold_ratio
        self.procrustes = WeightedProcrustes(return_transform=True)

        if match_option != 'topk':
            assert self.k > 0, f"When match_option is not 'topk', k must be greater than 0, but got {self.k}"
        else: # match_option == 'topk'
            assert self.k != 0, f"When match_option is 'topk', k must be not 0, but got {self.k}"

        assert 0.0 <= self.score_threshold_ratio <= 1.0, f"score_threshold_ratio must be between 0.0 and 1.0, but got {self.score_threshold_ratio}"

        print("-----------------------[LocalGlobalRegistration]-------------------------------")
        print(f"k: {k}")
        print(f"match_option: {match_option}")
        print(f"acceptance_radius: {acceptance_radius}")
        print(f"num_refinement_steps: {num_refinement_steps}")
        print(f"score_threshold_ratio: {score_threshold_ratio}")
        print("-------------------------------------------------------------------------------")


    def sample_correspondences(self, score_mat):
        """Sample correspondences from score matrix
        B == 1

        Args:
            score_mat (torch.Tensor): (B, N, M)

        Returns:
            pred_corr (torch.Tensor): (K, 2)
        """ 
        squeezed_score_mat = score_mat.squeeze(0)

        # Initial matches for RANSAC
        if self.match_option == 'topk':
            topk = int((squeezed_score_mat.shape[0] + squeezed_score_mat.shape[1]) / (- self.k)) if self.k < 0 else self.k
            sampled_correspondences = topk_matching(squeezed_score_mat, k=topk) # (K, 2)
        
        elif self.match_option == 'mutual_topk':
            sampled_correspondences = mutual_topk_matching(squeezed_score_mat, topk=self.k) # (K, 2)
        
        elif self.match_option == 'soft_topk':
            sampled_correspondences = soft_topk_matching(squeezed_score_mat, topk=self.k) # (K, 2)
        
        elif self.match_option == 'unidirectional_nn_matching':
            sampled_correspondences = unidirectional_nn_matching(squeezed_score_mat, topk=self.k) # (K, 2)
        
        elif self.match_option == 'injective_matching':
            sampled_correspondences = injective_matching(squeezed_score_mat) # (K, 2)
        
        elif self.match_option == 'bijective_matching':
            sampled_correspondences = bijective_matching(squeezed_score_mat) # (K, 2)
        
        else:
            raise ValueError(f"Invalid match option: {self.match_option}")
        
        return sampled_correspondences

    
    def filter_correspondences(self, score_mat, corr_scores):
        """Filter correspondences based on score matrix
        If score_threshold_ratio is 0.0, all correspondences are filtered.

        Args:
            score_mat (torch.Tensor): (B, N, M)
            corr_scores (torch.Tensor): (K, )

        Returns:
            filtered_pred_corr (torch.Tensor): (B, K, 2)
        """
        flattend_score_mat = score_mat.reshape(-1) # (N*M, )
        num_elements = flattend_score_mat.shape[0] # N*M
        threshold_index = int(num_elements * self.score_threshold_ratio) # N*M * score_threshold_ratio

        if threshold_index > 0:
            threshold_scores, _ = torch.topk(flattend_score_mat, k=threshold_index, sorted=True)
            score_threshold = threshold_scores[-1]
            filtering_mask = corr_scores >= score_threshold
        
        else: # threshold_index == 0, no filtering is performed
            filtering_mask = torch.ones_like(corr_scores, dtype=torch.bool)
        return filtering_mask

    
    def recompute_correspondence_scores(self, src_corr_points, ref_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores


    def local_to_global_registration(self, src_points, ref_points, pred_corr, score_mat):
        """Local-to-Global Registration

        Args:
            src_points (torch.Tensor): (B, N, 3)
            ref_points (torch.Tensor): (B, M, 3)
            pred_corr (torch.Tensor): (K, 2)
            score_mat (torch.Tensor): (B, N, M)

        Returns:
            estimated_transform (torch.Tensor): (B, 4, 4)
        """
        src_corr_points = src_points.squeeze(0)[pred_corr[:,0]] # (K, 3)
        ref_corr_points = ref_points.squeeze(0)[pred_corr[:,1]] # (K, 3)
        corr_scores = score_mat[:, pred_corr[:,0], pred_corr[:,1]].squeeze(0) # (K, )

        # Filter correspondences based on score matrix
        filtering_mask = self.filter_correspondences(score_mat, corr_scores)

        # Filter correspondences
        src_corr_points = src_corr_points[filtering_mask] # (K', 3)
        ref_corr_points = ref_corr_points[filtering_mask] # (K', 3)
        corr_scores = corr_scores[filtering_mask] # (K', )

        # degenerate: initialize transformation with all correspondences
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
        cur_corr_scores = self.recompute_correspondence_scores(src_corr_points, ref_corr_points, corr_scores, estimated_transform)

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(src_corr_points, ref_corr_points, corr_scores, estimated_transform)
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        
        return estimated_transform

    
    def forward(self, src_points, ref_points, score_mat, no_exp=False):
        r"""Point Matching Module forward propagation with Local-to-Global registration.
        Only assume that batch size is 1.

        Args:
            src_points (Tensor): (B, N, 3)
            ref_points (Tensor): (B, M, 3)
            score_mat (Tensor): (B, N, M), log likelihood

        Returns:
            estimated_transform: torch.Tensor (4, 4)
        """

        score_mat = torch.exp(score_mat) if not no_exp else score_mat
        pred_corr = self.sample_correspondences(score_mat)

        """ THIS IS FOR DEBUGGING
        pred_corr_t = self.sample_correspondences(score_mat.transpose(-2,-1))
        pred_corr_t = torch.stack([pred_corr_t[:,1], pred_corr_t[:,0]], dim=1)
        check_is_same = torch.all(pred_corr == pred_corr_t)
        assert check_is_same, f"pred_corr and pred_corr_t are not the same\n{pred_corr}\n{pred_corr_t}"
        """

        estimated_transform = self.local_to_global_registration(src_points, ref_points, pred_corr, score_mat)
        return estimated_transform