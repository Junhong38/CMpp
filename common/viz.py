import os
import torch
import random
import numpy as np
import open3d as o3d


global_colors_for_objs = {
    "red": [1.0, 0.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "cyan": [0.0, 1.0, 1.0],
    "orange": [1.0, 0.5, 0.0],
    "green": [0.0, 1.0, 0.0],
    "purple": [0.5, 0.0, 1.0]
}


global_colors_for_arrows = {
    "red": [1.0, 0.0, 0.0],

    "orange": [1.0, 0.5, 0.0],
    "green": [0.0, 1.0, 0.0],
    "purple": [0.5, 0.0, 1.0]
}



def draw_frames(frame_ori, gt_normals, pcds_list, dir_path, filename, sphere_radius=0.001, cylinder_radius=0.001, cone_radius=0.002, arrow_scale=0.01, max_points=100):
    """
    Draw frames and GT normals, and save as HTML.

    Args:
        frame_ori (list of torch.Tensor): each element is (N*3, 3), where three means three basis vectors
        gt_normals (list of torch.Tensor): each element is (N, 3)
        pcds_list (list of torch.Tensor): each element is (N, 3)
        dir_path (str): directory path to save
        filename (str): filename to save
        sphere_radius (float): radius of sphere
        cylinder_radius (float): radius of cylinder
        cone_radius (float): radius of cone
        arrow_scale (float): scale of arrow
        max_points (int): maximum number of points
    """
    assert len(frame_ori) == len(gt_normals) == len(pcds_list), f"must have same length, frame_ori: {len(frame_ori)}, gt_normals: {len(gt_normals)}, pcds_list: {len(pcds_list)}"

    for frame_ori_i, gt_normals_i, pcds_i in zip(frame_ori, gt_normals, pcds_list):
        assert gt_normals_i.shape == pcds_i.shape, f"must have same shape, gt_normals_i: {gt_normals_i.shape}, pcds_i: {pcds_i.shape}"
        assert frame_ori_i.shape[0] == gt_normals_i.shape[0] * 3, f"frame_ori_i should have 3 times more points than gt_normals_i, frame_ori_i: {frame_ori_i.shape}, gt_normals_i: {gt_normals_i.shape}"

    # pcd -> sphere meshes
    sphere_meshes = make_spheres_from_pcd_tensors(pcds=pcds_list, sphere_radius=sphere_radius)
    
    # vector -> arrow meshes
    arrow_meshes_gt_normals = make_arrows_from_vector_tensors(pcds=pcds_list, vectors=gt_normals, colors=['red'], cylinder_radius=cylinder_radius, cone_radius=cone_radius, arrow_scale=arrow_scale, max_points=max_points, reshape=False)
    arrow_meshes_pred_frame_ori = make_arrows_from_vector_tensors(pcds=pcds_list, vectors=frame_ori, colors=['orange', 'green', 'purple'], cylinder_radius=cylinder_radius, cone_radius=cone_radius, arrow_scale=arrow_scale, max_points=max_points, reshape=True)
    arrows = arrow_meshes_gt_normals + arrow_meshes_pred_frame_ori

    # save meshes
    save_meshes_as_ply(meshes=(sphere_meshes + arrows), dir_path=dir_path, filename=filename)



def make_spheres_from_pcd_tensors(pcds, sphere_radius=0.005):
    """
    Args:
        pcds (list of torch.Tensor): each element is (N, 3)
    
    Returns:
        list of o3d.geometry.TriangleMesh()
    """
    all_colors = list(global_colors_for_objs.keys())

    sphere_meshes = []
    for i, tensor_ in enumerate(pcds):
        points = tensor_.cpu().numpy()
        selected_color = all_colors[i % len(all_colors)]

        combined_mesh = o3d.geometry.TriangleMesh()

        # Add point cloud as small spheres
        for point in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(point)
            sphere.paint_uniform_color(global_colors_for_objs[selected_color])
            combined_mesh += sphere
        
        sphere_meshes.append(combined_mesh)
    
    return sphere_meshes



def make_arrows_from_vector_tensors(pcds, vectors, colors, cylinder_radius=0.002, cone_radius=0.005, arrow_scale=0.1, max_points=1000, reshape=False):
    """
    Args:
        pcds (list of torch.Tensor): each element is (N, 3)
        vectors (list of torch.Tensor): each element is (N, 3) or (N, 3, 3)
        colors (list of str): e.g. ['red']
        cylinder_radius (float): radius of cylinder
        cone_radius (float): radius of cone
        arrow_scale (float): scale of arrow
        max_points (int): maximum number of points
        reshape (bool): if True, input vectors will be (N*3, 3) -> (N, 3, 3)
    
    Returns:
        list of o3d.geometry.PointCloud: each element is (N, 3)
    """
    assert len(pcds) == len(vectors), f"pcds and vectors must have same length: {len(pcds)} vs {len(vectors)}"

    arrow_geometries = []

    for pcd_tensor, vector_tensor in zip(pcds, vectors):
        point_numpy = pcd_tensor.cpu().numpy()

        if reshape: # (N*3, 3) -> (N, 3, 3)
            vec_numpy = vector_tensor.reshape(-1,3,3).cpu().numpy()
        else: # (N, 3) -> (N, 1, 3)
            vec_numpy = vector_tensor[:,None,:].cpu().numpy() 
        
        assert point_numpy.shape[0] == vec_numpy.shape[0], f"number of points must be same: {pcd_tensor.shape} vs {vector_tensor.shape}"

        # Limit number of points for performance
        if len(point_numpy) > max_points:
            indices = np.linspace(0, len(point_numpy)-1, max_points, dtype=int)
            point_numpy = point_numpy[indices]
            vec_numpy = vec_numpy[indices]
        
        
        for ith, a_vec in enumerate(vec_numpy):
            # a_vec: (1,3) or (3,3)
            for idx in range(a_vec.shape[0]):
                # point_numpy[ith,:]: (N,3) -> (3,) / a_vec[idx,:]: (1,3) -> (3,) or (3,3) -> (3,)
                arrow = make_arrow_from_vector(point_numpy[ith,:], a_vec[idx,:], cylinder_radius=cylinder_radius, cone_radius=cone_radius, arrow_scale=arrow_scale)

                # Color the arrow
                selected_color = colors[idx % len(colors)]
                arrow.paint_uniform_color(global_colors_for_arrows[selected_color])

                arrow_geometries.append(arrow)

    return arrow_geometries



def make_arrow_from_vector(point, vector, cylinder_radius=0.002, cone_radius=0.005, arrow_scale=0.1):
    """
    Args:
        pcd (numpy.ndarray): (3,)
        vector (numpy.ndarray): (3,)
        cylinder_radius (float): radius of cylinder
        cone_radius (float): radius of cone
        arrow_scale (float): scale of arrow
    
    Returns:
        o3d.geometry.TriangleMesh
    """
    # Create arrow geometry
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=arrow_scale * 0.7,
        cone_height=arrow_scale * 0.3
    )

    # Position arrow at point
    arrow.translate(point)
    
    # Orient arrow along normal direction
    # Calculate rotation to align arrow with normal
    z_axis = np.array([0, 0, 1])  # Default arrow direction
    vector = vector / np.linalg.norm(vector)
    
    # Calculate rotation matrix
    v = np.cross(z_axis, vector)
    # cross product -> ||u × v|| = ||u|| ||v|| sin(θ) = sin(θ) where u and v are unit vectors
    # so, s is sin(θ) 
    s = np.linalg.norm(v) # sin(θ)
    c = np.dot(z_axis, vector) # cos(θ)
    vx = np.array([[0, -v[2], v[1]], 
                    [v[2], 0, -v[0]], 
                    [-v[1], v[0], 0]])

    # originally, rotation_matrix = I + sin(θ) * vx + (1 - cos(θ)) * vx^2
    rotation_matrix = np.eye(3) + s * vx + (1 - c) * np.dot(vx, vx)
    
    arrow.rotate(rotation_matrix, center=point)

    return arrow


def save_meshes_as_ply(meshes, dir_path, filename):
    """
    Save meshes as PLY files.
    
    Args:
        meshes (list of o3d.geometry.TriangleMesh): each element is (N, 3)
        dir_path (str): directory path to save
        filename (str): filename to save
    """
    # Combine point cloud with arrows
    combined_mesh = o3d.geometry.TriangleMesh()
    
    # Add meshes to combined mesh
    for i, mesh in enumerate(meshes):
        combined_mesh += mesh
    
    # Save as PLY format
    ply_filename = os.path.join(dir_path, f"{filename}.ply")
    o3d.io.write_triangle_mesh(ply_filename, combined_mesh)
