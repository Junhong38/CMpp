import numpy as np

import torch
import torch.nn.functional as F

from pytorch3d.transforms import quaternion_multiply
from pytorch3d.transforms import rotation_6d_to_matrix as rot6d_to_matrix

EPS = 1e-6


@torch.no_grad()
def _is_normalized(mat, dim=-1):
    """
    Check if one dim of a matrix is normalized.
    """
    norm = torch.norm(mat, p=2, dim=dim)
    return (norm - 1.).abs().max() < EPS


@torch.no_grad()
def _is_orthogonal(mat):
    """
    Check if a matrix (..., 3, 3) is orthogonal.
    """
    mat = mat.view(-1, 3, 3)
    iden = torch.eye(3).unsqueeze(0).repeat(mat.shape[0], 1, 1).type_as(mat)
    mat = torch.bmm(mat, mat.transpose(1, 2))
    return (mat - iden).abs().max() < EPS


def qeuler(q, order, epsilon=0, to_degree=False):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    euler = torch.stack((x, y, z), dim=1).view(original_shape)
    if to_degree:
        euler = euler * 180. / np.pi
    return euler

def ortho2rotation(poses):
    r"""
    poses: B x N x 2 x 3
    optimum: bool, if True, use optimum projection, otherwise use not optimum projection
    """

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
    

def rotmat_to_rodrigues(R: torch.Tensor):
    # 1. Compute theta
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + EPS, 1 - EPS)
    theta = torch.acos(cos_theta)

    # 2. Initialize Rodrigues vector
    rodrigues = torch.zeros(R.shape[:-2] + (3,), device=R.device, dtype=R.dtype)

    # 3. Case theta ~ 0 (no rotation)
    mask0 = theta < EPS
    # rodrigues[mask0] already zero

    # 4. Case theta ~ pi (180 deg rotation)
    mask_pi = (torch.abs(theta - torch.pi) < EPS)
    if mask_pi.any():
        R_pi = R[mask_pi]
        # Compute axis: find eigenvector of (R + I)/2
        # since R = I + 2*[u]_x^2 => (R + I)/2 = uu^T
        M = (R_pi + torch.eye(3, device=R.device)) / 2
        # choose largest diagonal component to avoid sign ambiguity
        u = torch.zeros_like(M[..., 0])
        # pick axis components from diagonal entries
        u[..., 0] = torch.sqrt(torch.clamp(M[..., 0, 0], min=0))
        u[..., 1] = torch.sqrt(torch.clamp(M[..., 1, 1], min=0))
        u[..., 2] = torch.sqrt(torch.clamp(M[..., 2, 2], min=0))
        # pick correct signs using off-diagonal entries
        u[..., 1] = torch.sign(M[..., 0,1]) * u[..., 1]
        u[..., 2] = torch.sign(M[..., 0,2]) * u[..., 2]
        # Normalize axis
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(EPS)
        rodrigues[mask_pi] = theta[mask_pi][..., None] * u

    # 5. General case
    mask = ~(mask0 | mask_pi)
    if mask.any():
        R_g = R[mask]
        theta_g = theta[mask]
        r = torch.stack([
            R_g[..., 2,1] - R_g[..., 1,2],
            R_g[..., 0,2] - R_g[..., 2,0],
            R_g[..., 1,0] - R_g[..., 0,1]
        ], dim=-1) / (2 * torch.sin(theta_g)[..., None])
        rodrigues[mask] = theta_g[..., None] * r

    return rodrigues

# [TODO] Before using this function, check if the rotation is correct
def rodrigues_to_rotmat(r):
    """
    r: Rodrigues vector, shape (..., 3)
    returns: rotation matrix, shape (..., 3, 3)
    """
    theta = torch.norm(r, dim=-1, keepdim=True)  # 회전각
    k = r / (theta + 1e-8)  # 회전축 (0으로 나누는 경우 방지)
    
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    
    # skew-symmetric matrix [k]_x
    zeros = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zeros, -kz, ky], dim=-1),
        torch.stack([kz, zeros, -kx], dim=-1),
        torch.stack([-ky, kx, zeros], dim=-1)
    ], dim=-2)  # shape (..., 3, 3)
    
    I = torch.eye(3, device=r.device, dtype=r.dtype).expand(K.shape[:-2] + (3, 3))
    
    theta = theta[..., 0]  # broadcast용
    sin_theta = torch.sin(theta)[..., None, None]
    cos_theta = torch.cos(theta)[..., None, None]
    
    R = I + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)
    
    return R

def R_to_Rprime(R: torch.Tensor):
    """
    R: (..., 3, 3) rotation matrix
    Returns:
        R_prime: (..., 3, 3) rotation matrix rotated by (theta + pi) along original Rodrigues axis
    """
    # 1. Rotation matrix -> Rodrigues vector
    r = rotmat_to_rodrigues(R)  # (..., 3)
    
    # 2. Rodrigues vector -> rotation axis & angle
    theta = torch.norm(r, dim=-1, keepdim=True)  # (...,1)
    u = r / (theta + 1e-8)                        # unit axis, (...,3)
    
    # 3. Add pi to rotation angle
    r_prime = u * (theta + torch.pi)
    
    # 4. Rodrigues -> rotation matrix
    R_prime = rodrigues_to_rotmat(r_prime)
    
    return R_prime


def rotation_matrix_from_vectors(v_from, v_to):
    """
    Compute batched rotation matrices that align v_from to v_to.
    v_from, v_to: (B, N, 3)
    return: (B, N, 3, 3)
    """
    v_from = v_from / (v_from.norm(dim=-1, keepdim=True) + EPS)
    v_to   = v_to   / (v_to.norm(dim=-1, keepdim=True) + EPS)

    # cross product (B, N, 3)
    axis = torch.cross(v_from, v_to, dim=-1)
    axis_norm = axis.norm(dim=-1, keepdim=True)

    # dot product (B, N, 1)
    dot = (v_from * v_to).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)  # (B, N, 1)

    # skew-symmetric matrix K (B, N, 3, 3)
    kx, ky, kz = axis[..., 0], axis[..., 1], axis[..., 2]
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=v_from.device, dtype=v_from.dtype)
    K[..., 0, 1] = -kz
    K[..., 0, 2] =  ky
    K[..., 1, 0] =  kz
    K[..., 1, 2] = -kx
    K[..., 2, 0] = -ky
    K[..., 2, 1] =  kx

    axis_norm = axis_norm + EPS
    K = K / axis_norm.unsqueeze(-1)  # (B, N, 3, 3)

    I = torch.eye(3, device=v_from.device, dtype=v_from.dtype)
    I = I.view(1, 1, 3, 3).expand_as(K)

    theta = theta.unsqueeze(-1)  # (B, N, 1, 1)

    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)  # (B, N, 3, 3)
    return R
    

class Rotation3D:
    """Class for different 3D rotation representations, all util functions,
    model input-output will adopt this as the general interface.

    Supports common properties of torch.Tensor, e.g. shape, device, dtype.
    Also support indexing and slicing which are done to torch.Tensor.

    Currently, we support three rotation representations:
    - quaternion: (..., 4), real part first, unit quaternion
    - rotation matrix: (..., 3, 3), the input `rot` can be either rmat or 6D
        For 6D, it could be either (..., 6) or (..., 2, 3)
        Note that, we will convert it to 3x3 matrix in the constructor
    - axis-angle: (..., 3)
    - euler angles: this is NOT supported as a representation, but we can
        convert from supported representations to euler angles
    """

    ROT_TYPE = ['quat', 'rmat', 'axis']
    ROT_NAME = {
        'quat': 'quaternion',
        'rmat': 'matrix',
        'axis': 'axis_angle',
    }

    def __init__(self, rot, rot_type='rmat'):
        self._rot = rot
        self._rot_type = rot_type

        self._check_valid()

    def _process_zero_quat(self):
        """Convert zero-norm quat to (1, 0, 0, 0)."""
        with torch.no_grad():
            norms = torch.norm(self._rot, p=2, dim=-1, keepdim=True)
            new_rot = torch.zeros_like(self._rot)
            new_rot[..., 0] = 1.  # zero quat
            valid_mask = (norms.abs() > 0.5).repeat_interleave(4, dim=-1)
        self._rot = torch.where(valid_mask, self._rot, new_rot)

    def _normalize_quat(self):
        """Normalize quaternion."""
        self._rot = F.normalize(self._rot, p=2, dim=-1)

    def _check_valid(self):
        """Check the shape of rotation."""
        assert self._rot_type in self.ROT_TYPE, \
            f'rotation {self._rot_type} is not supported'
        assert isinstance(self._rot, torch.Tensor), 'rotation must be a tensor'
        # let's always make rotation in float32
        # otherwise quat won't be unit, and rmat won't be orthogonal
        self._rot = self._rot.float()
        if self._rot_type == 'quat':
            assert self._rot.shape[-1] == 4, 'wrong quaternion shape'
            # quat with norm == 0 are padded, make them (1, 0, 0, 0)
            # because (0, 0, 0, 0) convert to rmat will cause PyTorch bugs
            self._process_zero_quat()
            # too expensive to check
            # self._normalize_quat()
            # assert _is_normalized(self._rot, dim=-1), 'quaternion is not unit'
        elif self._rot_type == 'rmat':
            if self._rot.shape[-1] == 3:
                if self._rot.shape[-2] == 3:  # 3x3 matrix
                    # assert _is_orthogonal(self._rot)
                    pass  # too expensive to check
                elif self._rot.shape[-2] == 2:  # 6D representation
                    # (2, 3): (b1, b2), rot6d_to_matrix will calculate b3
                    # and stack them vertically
                    self._rot = rot6d_to_matrix(self._rot.flatten(-2, -1))
                else:
                    raise ValueError('wrong rotation matrix shape')
            elif self._rot.shape[-1] == 6:  # 6D representation
                # this indeed doing `rmat = torch.stack((b1, b2, b3), dim=-2)`
                self._rot = rot6d_to_matrix(self._rot)
            else:
                raise NotImplementedError('wrong rotation matrix shape')
        else:  # axis-angle
            assert self._rot.shape[-1] == 3

    def apply_rotation(self, rot):
        """Apply `rot` to the current rotation, left multiply."""
        assert rot.rot_type in ['quat', 'rmat']
        rot = rot.convert(self._rot_type)
        if self._rot_type == 'quat':
            new_rot = quaternion_multiply(rot.rot, self._rot)
        else:
            new_rot = rot.rot @ self._rot
        return self.__class__(new_rot, self._rot_type)

    def convert(self, rot_type):
        """Convert to a different rotation type."""
        assert rot_type in self.ROT_TYPE, f'unknown target rotation {rot_type}'
        src_type = self.ROT_NAME[self._rot_type]
        dst_type = self.ROT_NAME[rot_type]
        if src_type == dst_type:
            return self.clone()
        new_rot = eval(f'{src_type}_to_{dst_type}')(self._rot)
        return self.__class__(new_rot, rot_type)

    def to_quat(self):
        """Convert to quaternion and return the tensor."""
        return self.convert('quat').rot

    def to_rmat(self):
        """Convert to rotation matrix and return the tensor."""
        return self.convert('rmat').rot

    def to_axis_angle(self):
        """Convert to axis angle and return the tensor."""
        return self.convert('axis').rot

    def to_euler(self, order='zyx', to_degree=True):
        """Compute to euler angles and return the tensor."""
        quat = self.convert('quat')
        return qeuler(quat._rot, order=order, to_degree=to_degree)

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, rot):
        self._rot = rot
        self._check_valid()

    @property
    def rot_type(self):
        return self._rot_type

    @rot_type.setter
    def rot_type(self, rot_type):
        raise NotImplementedError(
            'please use convert() for rotation type conversion')

    @property
    def shape(self):
        return self._rot.shape

    def reshape(self, *shape):
        return self.__class__(self._rot.reshape(*shape), self._rot_type)

    def view(self, *shape):
        return self.__class__(self._rot.view(*shape), self._rot_type)

    def squeeze(self, dim=None):
        return self.__class__(self._rot.squeeze(dim), self._rot_type)

    def unsqueeze(self, dim=None):
        return self.__class__(self._rot.unsqueeze(dim), self._rot_type)

    def flatten(self, *args, **kwargs):
        return self.__class__(
            self._rot.flatten(*args, **kwargs), self._rot_type)

    def unflatten(self, *args, **kwargs):
        return self.__class__(
            self._rot.unflatten(*args, **kwargs), self._rot_type)

    def transpose(self, *args, **kwargs):
        return self.__class__(
            self._rot.transpose(*args, **kwargs), self._rot_type)

    def permute(self, *args, **kwargs):
        return self.__class__(
            self._rot.permute(*args, **kwargs), self._rot_type)

    def contiguous(self):
        return self.__class__(self._rot.contiguous(), self._rot_type)

    @staticmethod
    def cat(rot_lst, dim=0):
        """Concat a list a Rotation3D object."""
        assert isinstance(rot_lst, (list, tuple))
        assert all([isinstance(rot, Rotation3D) for rot in rot_lst])
        rot_type = rot_lst[0].rot_type
        assert all([rot.rot_type == rot_type for rot in rot_lst])
        rot_lst = [rot.rot for rot in rot_lst]
        return Rotation3D(torch.cat(rot_lst, dim=dim), rot_type)

    @staticmethod
    def stack(rot_lst, dim=0):
        """Stack a list of Rotation3D object."""
        assert isinstance(rot_lst, (list, tuple))
        assert all([isinstance(rot, Rotation3D) for rot in rot_lst])
        rot_type = rot_lst[0].rot_type
        assert all([rot.rot_type == rot_type for rot in rot_lst])
        rot_lst = [rot.rot for rot in rot_lst]
        return Rotation3D(torch.stack(rot_lst, dim=dim), rot_type)

    def __getitem__(self, key):
        return self.__class__(self._rot[key], self._rot_type)

    def __len__(self):
        return self._rot.shape[0]

    @property
    def device(self):
        return self._rot.device

    def to(self, device):
        return self.__class__(self._rot.to(device), self._rot_type)

    def cuda(self, device=None):
        return self.__class__(self._rot.cuda(device), self._rot_type)

    @property
    def dtype(self):
        return self._rot.dtype

    def type(self, dtype):
        return self.__class__(self._rot.type(dtype), self._rot_type)

    def type_as(self, other):
        return self.__class__(self._rot.type_as(other), self._rot_type)

    def detach(self):
        return self.__class__(self._rot.detach(), self._rot_type)

    def clone(self):
        return self.__class__(self._rot.clone(), self._rot_type)
