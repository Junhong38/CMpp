r""" Helper functions """
import random

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from functools import reduce

def save_pc(filename: str, pcd_tensors: list):
    colors = [
        [1, 0.996, 0.804],
        [0.804, 0.98, 1],
        [1, 0.376, 0],
        [0, 0.055, 1]
    ]

    pcds = []
    for i, tensor_ in enumerate(pcd_tensors):
        if tensor_.size()[0] == 1:
            tensor_ = tensor_.squeeze(0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tensor_.cpu().numpy())
        pcd.paint_uniform_color(colors[i % len(colors)])  # Assign color based on index
        pcds.append(pcd)
    
    combined_cloud = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_cloud += pcd
    
    o3d.io.write_point_cloud(filename, combined_cloud)

import plotly.graph_objects as go

def save_normal(
    filename: str,
    point_tensors: list,
    normal_tensors: list,
    normal_length: float = 0.007
):
    """
    Visualize source/target points and normals using Plotly, save as HTML.

    Args:
        filename (str): Output filename (should end with .html).
        point_tensors (list of torch.Tensor): [src_point, trg_point]
        normal_tensors (list of torch.Tensor): [src_normal, trg_normal]
        normal_length (float): Length of normal arrows.
    
    Returns:
        fig (plotly.graph_objects.Figure): The figure object.
    """

    assert len(point_tensors) == len(normal_tensors) == 2, "Expected two point/normal sets (src and trg)"
    
    point_colors = ['lightblue', 'orange']
    normal_colors = ['blue', 'red']
    names = ['source', 'target']

    fig = go.Figure()

    for i in range(2):
        pts = point_tensors[i].squeeze(0).cpu().numpy()
        nml = normal_tensors[i].squeeze(0).cpu().numpy()

        # Add point scatter
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=point_colors[i]),
            name=f'{names[i]} points'
        ))

        # Add normal arrows as lines
        for p, n in zip(pts, nml):
            end = p + n * normal_length
            fig.add_trace(go.Scatter3d(
                x=[p[0], end[0]],
                y=[p[1], end[1]],
                z=[p[2], end[2]],
                mode='lines',
                line=dict(color=normal_colors[i], width=2),
                showlegend=False
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        title='Points and Normals (Plotly)',
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.write_html(filename)  # Save as interactive HTML
    return


# def save_ori(filename: str, pcd_tensors: list, ori_tensors: list):
#     colors = [
#         [1, 0.996, 0.804],
#         [0.804, 0.98, 1],
#         [1, 0.376, 0],
#         [0, 0.055, 1]
#     ]

#     orientations = []
#     pcd_meshs = []
#     for i, (pcd_, ori_) in enumerate(zip(pcd_tensors, ori_tensors)):
#         if pcd_.size(0) == 1:
#             pcd_.squeeze_(0)
#         if ori_.size(0) == 1:
#             ori_.squeeze_(0)
#         for ori, pos in zip(ori_, pcd_):
#             # 기본 화살표 생성 (z축 방향)
#             arrows=[]
#             length = 0.5
#             arrows.append(
#                 o3d.geometry.TriangleMesh.create_arrow(
#                     cylinder_radius=0.0001, cone_radius=0.0002,
#                     cylinder_height=length*0.008, cone_height=length*0.002
#                 )
#             )
#             arrows.append(
#                 o3d.geometry.TriangleMesh.create_arrow(
#                     cylinder_radius=0.0001, cone_radius=0.0002,
#                     cylinder_height=length*0.008, cone_height=length*0.002
#                 )
#             )
#             arrows.append(
#                 o3d.geometry.TriangleMesh.create_arrow(
#                     cylinder_radius=0.0001, cone_radius=0.0002,
#                     cylinder_height=length*0.008, cone_height=length*0.002
#                 )
#             )

#             z = np.array([0,0,1])
#             for j in range(3):
#                 # give the color to arrows
#                 arrows[j].paint_uniform_color(colors[i % len(colors)])
#                 arrows[j].compute_vertex_normals()
#                 # rotate each arrow to each vector direction from z-axis
#                 v = ori[j].cpu().numpy()
#                 cross = np.cross(z, v)
#                 dot = np.dot(z, v)
#                 if np.linalg.norm(cross) != 0:
#                     axis = cross / np.linalg.norm(cross)
#                     angle = np.arccos(dot)
#                     R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
#                     arrows[j].rotate(R, center=(0, 0, 0))
                
#                 # translate each arrow to each position from the origin
#                 arrows[j].translate(pos.cpu().numpy())
#                 orientations.append(arrows[j])
#         ## pcd도 함께 visualization
#         mesh = o3d.geometry.TriangleMesh()
#         mesh.vertices = o3d.utility.Vector3dVector(pcd_.cpu().numpy())
#         mesh.paint_uniform_color(colors[i % len(colors)])
#         pcd_meshs.append(mesh)
    
#     combined_pcd_meshs = reduce(lambda a, b: a + b, pcd_meshs)
#     combined_orientations = reduce(lambda a, b: a + b, orientations)
#     o3d.io.write_triangle_mesh(filename, combined_pcd_meshs + combined_orientations)

import open3d as o3d
import torch
import numpy as np
from functools import reduce

def save_ori(filename: str, pcd_tensors: list, ori_tensors: list):
    colors = [
        [1, 0.996, 0.804],
        [0.804, 0.98, 1],
        [1, 0.376, 0],
        [0, 0.055, 1]
    ]

    orientations = []
    point_spheres = []

    for i, (pcd_, ori_) in enumerate(zip(pcd_tensors, ori_tensors)):
        if pcd_.size(0) == 1:
            pcd_ = pcd_.squeeze(0)
        if ori_.size(0) == 1:
            ori_ = ori_.squeeze(0)

        color = colors[i % len(colors)]

        for ori_vecs, pos in zip(ori_, pcd_):
            z = np.array([0, 0, 1])
            pos_np = pos.cpu().numpy()

            for j in range(3):
                # 화살표 생성
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.0002, cone_radius=0.0004,
                    cylinder_height=0.004, cone_height=0.001
                )
                arrow.paint_uniform_color(color)
                arrow.compute_vertex_normals()

                # 방향 회전
                v = ori_vecs[j].cpu().numpy()
                cross = np.cross(z, v)
                dot = np.dot(z, v)
                if np.linalg.norm(cross) != 0:
                    axis = cross / np.linalg.norm(cross)
                    angle = np.arccos(np.clip(dot, -1.0, 1.0))  # 안정성 확보
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    arrow.rotate(R, center=(0, 0, 0))

                # 위치 이동
                arrow.translate(pos_np)
                orientations.append(arrow)

            # 포인트를 작은 구(mesh)로 변환
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.00005)
            sphere.translate(pos_np)
            sphere.paint_uniform_color(color)
            point_spheres.append(sphere)

    # 전체 메쉬 결합
    all_arrows = reduce(lambda a, b: a + b, orientations)
    all_spheres = reduce(lambda a, b: a + b, point_spheres)
    combined = all_arrows + all_spheres

    # 저장
    o3d.io.write_triangle_mesh(filename, combined, write_vertex_colors=True)



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

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

def SVD(A, B):

    centroid_A = A.mean(dim=0)
    centroid_B = B.mean(dim=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = B_centered.T @ A_centered
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    # Reflection correction
    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_A - R @ centroid_B
    return R, t
