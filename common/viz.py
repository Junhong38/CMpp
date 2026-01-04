import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch

from common.misc import extract_all_objects

# Set matplotlib backend before any other matplotlib imports
# This must be done in every process (including worker processes)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X server issues
import matplotlib.pyplot as plt



global_colors_for_objs = {
    "red": [1.0, 0.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "cyan": [0.0, 1.0, 1.0],
    "orange": [1.0, 0.5, 0.0],
    "green": [0.0, 1.0, 0.0],
    "purple": [0.5, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
}


global_colors_for_arrows = {
    "red": [1.0, 0.0, 0.0],

    "orange": [1.0, 0.5, 0.0],
    "green": [0.0, 1.0, 0.0],
    "purple": [0.5, 0.0, 1.0]
}



def draw_frames(mesh_verts, mesh_faces, 
                frame_ori, gt_normals, pcds_list, dir_path, filename, 
                sphere_radius=0.001, cylinder_radius=0.001, cone_radius=0.002, arrow_scale=0.01, viz_max_arrow_num=5000,
                viz_piece=False, viz_full=False):
    """
    Draw frames and GT normals, and save as HTML.

    Args:
        mesh_verts (list of torch.Tensor): each element is (N', 3)
        mesh_faces (list of torch.Tensor): each element is (F, 3)
        frame_ori (list of torch.Tensor): each element is (N*3, 3), where three means three basis vectors
        gt_normals (list of torch.Tensor): each element is (N, 3)
        pcds_list (list of torch.Tensor): each element is (N, 3)
        dir_path (str): directory path to save
        filename (str): filename to save
        sphere_radius (float): radius of sphere
        cylinder_radius (float): radius of cylinder
        cone_radius (float): radius of cone
        arrow_scale (float): scale of arrow
        viz_max_arrow_num (int): maximum number of points
        viz_piece (bool): if True, visualize each piece of mesh
        viz_full (bool): if True, visualize full mesh
    """
    assert viz_piece or viz_full, f"viz_piece or viz_full must be True, but got {viz_piece} and {viz_full}"
    assert len(mesh_verts) == len(mesh_faces) == len(frame_ori) == len(gt_normals) == len(pcds_list), f"must have same length, mesh_verts: {len(mesh_verts)}, mesh_faces: {len(mesh_faces)}, frame_ori: {len(frame_ori)}, gt_normals: {len(gt_normals)}, pcds_list: {len(pcds_list)}"

    for frame_ori_i, gt_normals_i, pcds_i in zip(frame_ori, gt_normals, pcds_list):
        assert gt_normals_i.shape == pcds_i.shape, f"must have same shape, gt_normals_i: {gt_normals_i.shape}, pcds_i: {pcds_i.shape}"
        assert frame_ori_i.shape[0] == gt_normals_i.shape[0] * 3, f"frame_ori_i should have 3 times more points than gt_normals_i, frame_ori_i: {frame_ori_i.shape}, gt_normals_i: {gt_normals_i.shape}"

    # pcd -> sphere meshes
    # sphere_meshes = make_spheres_from_pcd_tensors(pcds=pcds_list, sphere_radius=sphere_radius)
    recovered_meshes = make_mesh_from_pcd_tensors(pcds=mesh_verts, mesh_faces=mesh_faces)
    
    # vector -> arrow meshes
    arrow_meshes_gt_normals = make_arrows_from_vector_tensors(pcds=pcds_list, vectors=gt_normals, colors=['red'], cylinder_radius=cylinder_radius, cone_radius=cone_radius, arrow_scale=arrow_scale, viz_max_arrow_num=viz_max_arrow_num, reshape=False)
    arrow_meshes_pred_frame_ori = make_arrows_from_vector_tensors(pcds=pcds_list, vectors=frame_ori, colors=['orange', 'green', 'purple'], cylinder_radius=cylinder_radius, cone_radius=cone_radius, arrow_scale=arrow_scale, viz_max_arrow_num=viz_max_arrow_num, reshape=True)

    assert len(recovered_meshes) == len(arrow_meshes_gt_normals) == len(arrow_meshes_pred_frame_ori), \
    f"must have same length, recovered_meshes: {len(recovered_meshes)}, arrow_meshes_gt_normals: {len(arrow_meshes_gt_normals)}, arrow_meshes_pred_frame_ori: {len(arrow_meshes_pred_frame_ori)}"

    if viz_piece:
        # save each piece of mesh
        for ith, (a_mesh, a_arrow_gt_normals, a_arrow_pred_frame_ori) in enumerate(zip(recovered_meshes, arrow_meshes_gt_normals, arrow_meshes_pred_frame_ori)):
            save_meshes_as_ply(meshes=([a_mesh, a_arrow_gt_normals, a_arrow_pred_frame_ori]), dir_path=dir_path, filename=f"{filename}_piece_{ith}")
    
    if viz_full:
        # save all meshes
        save_meshes_as_ply(meshes=(recovered_meshes + arrow_meshes_gt_normals + arrow_meshes_pred_frame_ori), dir_path=dir_path, filename=filename)


def make_pcds_from_pcd_tensors(pcds):
    """
    Args:
        pcds (list of torch.Tensor): each element is (N, 3)
    
    Returns:
        list of o3d.utility.Vector3dVector: each element is (N, 3)
    """
    all_colors = list(global_colors_for_objs.keys())
    all_pcds = []
    for i, pcd_tensor in enumerate(pcds):
        points = pcd_tensor.cpu().numpy()
        selected_color = all_colors[i % len(all_colors)]

        pcd_open3d = o3d.geometry.PointCloud()
        pcd_open3d.points = o3d.utility.Vector3dVector(points)
        pcd_open3d.paint_uniform_color(global_colors_for_objs[selected_color])
        all_pcds.append(pcd_open3d)
    return all_pcds



def make_mesh_from_pcd_tensors(pcds, mesh_faces):
    """
    Convert trimesh mesh to Open3D mesh
    
    Args:
        pcds (list of torch.Tensor): each element is (N', 3)
        mesh_faces (list of torch.Tensor): each element is (F, 3)
        
    Returns:
        list of o3d.geometry.TriangleMesh: Open3D mesh object
    """
    all_colors = list(global_colors_for_objs.keys())
    
    all_meshes = []
    for i, (pcd_tensor, mesh_face) in enumerate(zip(pcds, mesh_faces)):
        selected_color = all_colors[i % len(all_colors)]
        faces = mesh_face.cpu().numpy()
        points = pcd_tensor.cpu().numpy()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(global_colors_for_objs[selected_color])
        all_meshes.append(mesh)
    return all_meshes


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



def make_arrows_from_vector_tensors(pcds, vectors, colors, cylinder_radius=0.002, cone_radius=0.005, arrow_scale=0.1, viz_max_arrow_num=1000, reshape=False):
    """
    Args:
        pcds (list of torch.Tensor): each element is (N, 3)
        vectors (list of torch.Tensor): each element is (N, 3) or (N, 3, 3)
        colors (list of str): e.g. ['red']
        cylinder_radius (float): radius of cylinder
        cone_radius (float): radius of cone
        arrow_scale (float): scale of arrow
        viz_max_arrow_num (int): maximum number of points
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

        target_arrow_num = viz_max_arrow_num if viz_max_arrow_num > 0 else len(point_numpy)

        # Limit number of points for performance
        if len(point_numpy) > target_arrow_num:
            indices = np.linspace(0, len(point_numpy)-1, target_arrow_num, dtype=int)
            point_numpy = point_numpy[indices]
            vec_numpy = vec_numpy[indices]
        
        combined_mesh = o3d.geometry.TriangleMesh()
        
        for ith, a_vec in enumerate(vec_numpy):
            # a_vec: (1,3) or (3,3)
            for idx in range(a_vec.shape[0]):
                # point_numpy[ith,:]: (N,3) -> (3,), a_vec[idx,:]: (1,3) -> (3,) or (3,3) -> (3,)
                arrow = make_arrow_from_vector(point_numpy[ith,:], a_vec[idx,:], cylinder_radius=cylinder_radius, cone_radius=cone_radius, arrow_scale=arrow_scale)

                # Color the arrow
                selected_color = colors[idx % len(colors)]
                arrow.paint_uniform_color(global_colors_for_arrows[selected_color])

                combined_mesh += arrow
        
        arrow_geometries.append(combined_mesh)

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
    

    # Use scipy's rotation for more reliable calculation
    # Calculate rotation matrix using scipy
    # Find rotation that aligns z_axis with vector
    rotation = R.align_vectors([vector], [z_axis])[0]
    rotation_matrix = rotation.as_matrix()

    arrow.rotate(rotation_matrix, center=point)


    # Check if the rotation is correct
    assert np.all(np.abs(vector - (rotation_matrix @ z_axis)) < 1e-6), f"vector: {vector}, arrow: {rotation_matrix @ z_axis}, difference: {vector - (rotation_matrix @ z_axis)}"

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



def draw_normal_error_histogram(normal_error_hist, dir_path, filename):
    """
    Draw normal error histogram.

    Args:
        normal_error_hist (numpy.ndarray): histogram of normal error
             - [0] (torch.Tensor): number of points in each bin
             - [1] (torch.Tensor): bin edges
        dir_path (str): directory path to save
        filename (str): filename to save
    """
    counts = normal_error_hist[0]
    bins = normal_error_hist[1]

    plt.hist(bins[:-1], bins=bins, weights=counts)
    plt.xlabel('Normal Error (degrees)')
    plt.ylabel('Count')
    plt.title('Normal Error Histogram')
    plt.savefig(os.path.join(dir_path, f"{filename}.png"))
    plt.close()



def draw_test_results_histogram(test_results, dir_path, filename):
    """
    Draw test results histogram.

    Args:
        test_results (dict): test results
            - key: case_name / val: dict of (metric_name: metric_value)
        dir_path (str): directory path to save
        filename (str): filename to save
    """
    metric_names = list(test_results[list(test_results.keys())[0]].keys())

    for metric_name in metric_names:
        metric_values = [metric_dict[metric_name].cpu().item() for metric_dict in test_results.values()]
        plt.hist(metric_values, bins=100)
        plt.xlabel(metric_name)
        plt.ylabel('Count')
        plt.title(f'{metric_name} Histogram')
        plt.savefig(os.path.join(dir_path, f"{filename}_{metric_name}.png"))
        plt.close()


def save_pcd_for_light_visualization(pcd_tensors: list, gt_corr: torch.Tensor, used_corr: torch.Tensor, filename: str):
    """
    Args:
        pcd_tensors (list of torch.Tensor): each element is (N, 3)
        gt_corr (torch.Tensor): (K, 2)
    
    Returns:
        list of torch.Tensor: each element is (N, 3)
    """
    src_pcd, trg_pcd = pcd_tensors
    num_src_pcd, num_trg_pcd = src_pcd.shape[0], trg_pcd.shape[0]

    placeholder_gt_corr = torch.zeros(num_src_pcd, num_trg_pcd, dtype=torch.bool)
    placeholder_gt_corr[gt_corr[:,0], gt_corr[:,1]] = True

    placeholder_used_corr = torch.zeros(num_src_pcd, num_trg_pcd, dtype=torch.bool)
    placeholder_used_corr[used_corr[:,0], used_corr[:,1]] = True

    intersection_mask = torch.logical_and(placeholder_gt_corr, placeholder_used_corr)

    gt_corr_src_mask = placeholder_gt_corr.sum(dim=-1) > 0
    gt_corr_trg_mask = placeholder_gt_corr.sum(dim=-2) > 0
    used_corr_src_mask = placeholder_used_corr.sum(dim=-1) > 0
    used_corr_trg_mask = placeholder_used_corr.sum(dim=-2) > 0
    intersection_mask_src_mask = intersection_mask.sum(dim=-1) > 0
    intersection_mask_trg_mask = intersection_mask.sum(dim=-2) > 0

    left_src_mask = (~gt_corr_src_mask) & (~used_corr_src_mask) & (~intersection_mask_src_mask)
    left_trg_mask = (~gt_corr_trg_mask) & (~used_corr_trg_mask) & (~intersection_mask_trg_mask)
    pure_gt_corr_src_mask = gt_corr_src_mask & (~used_corr_src_mask) & (~intersection_mask_src_mask)
    pure_gt_corr_trg_mask = gt_corr_trg_mask & (~used_corr_trg_mask) & (~intersection_mask_trg_mask)
    pure_used_corr_src_mask = used_corr_src_mask &(~intersection_mask_src_mask)
    pure_used_corr_trg_mask = used_corr_trg_mask & (~intersection_mask_trg_mask)
    pure_intersection_mask_src_mask = intersection_mask_src_mask
    pure_intersection_mask_trg_mask = intersection_mask_trg_mask

    pcds_for_viz = [src_pcd[left_src_mask], trg_pcd[left_trg_mask]] # Red, Blue
    pcds_for_viz.append(src_pcd[pure_gt_corr_src_mask]) # Magenta
    pcds_for_viz.append(trg_pcd[pure_gt_corr_trg_mask]) # Cyan
    pcds_for_viz.append(src_pcd[pure_used_corr_src_mask]) # Orange
    pcds_for_viz.append(trg_pcd[pure_used_corr_trg_mask]) # Green
    pcds_for_viz.append(src_pcd[pure_intersection_mask_src_mask]) # Purple
    pcds_for_viz.append(trg_pcd[pure_intersection_mask_trg_mask]) # Yellow

    len_of_gt = len(pure_gt_corr_src_mask)
    save_pc(f"{filename}_GTCorrlen{len_of_gt}.ply", pcds_for_viz)



def visualize_negative_hard_mask(in_dict, neg_mask, hard_neg_mask, active_mask, dir_path, current_epoch, global_rank, pos_radius=0.018, safe_radius=0.03):
    """
    Args:
        in_dict (dict): input dictionary
        neg_mask (torch.Tensor): (B, N+M, N+M)
        hard_neg_mask (torch.Tensor): (B, N+M, N+M)
        active_mask (torch.Tensor): (B, N+M, N+M)
        dir_path (str): directory path to save
        global_rank (int): global rank
        pos_radius (float): radius for positive samples
        safe_radius (float): radius for safe samples
    """
    # Only take care of the first instance in the batch
    src_pcd_raw, trg_pcd_raw = extract_all_objects(in_dict['pcd'][0], in_dict['pcd_batch_info'][0]) # (N, 3), (M, 3)
    num_src_pcd, num_trg_pcd = src_pcd_raw.shape[0], trg_pcd_raw.shape[0]

    corrd_dist = torch.cdist(src_pcd_raw, trg_pcd_raw, p=2) # (N, M)

    pos_mask = corrd_dist < pos_radius
    processed_neg_mask = neg_mask[0][active_mask[0]] # (N*M)
    processed_hard_neg_mask = hard_neg_mask[0][active_mask[0]] # (N*M)
    processed_neg_mask = processed_neg_mask.reshape(num_src_pcd, num_trg_pcd) # (N, M)
    processed_hard_neg_mask = processed_hard_neg_mask.reshape(num_src_pcd, num_trg_pcd) # (N, M)

    intersection_mask_for_neg = torch.logical_and(pos_mask, processed_neg_mask)
    intersection_mask_for_hard = torch.logical_and(pos_mask, processed_hard_neg_mask)
    assert intersection_mask_for_neg.sum() == 0, f"intersection_mask_for_neg: {intersection_mask_for_neg.sum()}"
    assert intersection_mask_for_hard.sum() == 0, f"intersection_mask_for_hard: {intersection_mask_for_hard.sum()}"

    gt_corr = torch.nonzero(pos_mask) # (N*M, 2)
    hard_neg_corr = torch.nonzero(processed_hard_neg_mask) # (N*M, 2)
    neg_corr = torch.nonzero(processed_neg_mask) # (N*M, 2)

    gt_corr_src_mask = pos_mask.sum(dim=-1) > 0
    gt_corr_trg_mask = pos_mask.sum(dim=-2) > 0
    hard_neg_corr_src_mask = processed_hard_neg_mask.sum(dim=-1) > 0
    hard_neg_corr_trg_mask = processed_hard_neg_mask.sum(dim=-2) > 0
    neg_corr_src_mask = processed_neg_mask.sum(dim=-1) > 0
    neg_corr_trg_mask = processed_neg_mask.sum(dim=-2) > 0

    left_src_mask = (~gt_corr_src_mask) & (~hard_neg_corr_src_mask) & (~neg_corr_src_mask)
    left_trg_mask = (~gt_corr_trg_mask) & (~hard_neg_corr_trg_mask) & (~neg_corr_trg_mask)
    pure_gt_corr_src_mask = gt_corr_src_mask & (~hard_neg_corr_src_mask) & (~neg_corr_src_mask)
    pure_gt_corr_trg_mask = gt_corr_trg_mask & (~hard_neg_corr_trg_mask) & (~neg_corr_trg_mask)
    pure_neg_corr_src_mask = neg_corr_src_mask &(~hard_neg_corr_src_mask)
    pure_neg_corr_trg_mask = neg_corr_trg_mask & (~hard_neg_corr_trg_mask)
    pure_hard_neg_corr_src_mask = hard_neg_corr_src_mask
    pure_hard_neg_corr_trg_mask = hard_neg_corr_trg_mask

    # Visualize negative and hard negative samples
    pcd_list_for_viz = [src_pcd_raw[left_src_mask], trg_pcd_raw[left_trg_mask]] # Red, Blue
    pcd_list_for_viz.append(src_pcd_raw[pure_gt_corr_src_mask]) # Magenta
    pcd_list_for_viz.append(trg_pcd_raw[pure_gt_corr_trg_mask]) # Cyan
    pcd_list_for_viz.append(src_pcd_raw[pure_neg_corr_src_mask]) # Orange 
    pcd_list_for_viz.append(trg_pcd_raw[pure_neg_corr_trg_mask]) # Green 
    pcd_list_for_viz.append(src_pcd_raw[pure_hard_neg_corr_src_mask]) # Purple
    pcd_list_for_viz.append(trg_pcd_raw[pure_hard_neg_corr_trg_mask]) # Yellow

    # Visualize negative and hard negative samples as lineset
    total_points_for_lineset = torch.cat([src_pcd_raw, trg_pcd_raw], dim=0) # (N+M, 3)
    total_lines_for_lineset = [torch.stack([gt_corr[:,0], gt_corr[:,1] + len(src_pcd_raw)], dim=-1)] # (N*M, 2), Red
    total_lines_for_lineset.append(torch.stack([neg_corr[:,0], neg_corr[:,1] + len(src_pcd_raw)], dim=-1)) # (N*M, 2), Blue
    total_lines_for_lineset.append(torch.stack([hard_neg_corr[:,0], hard_neg_corr[:,1] + len(src_pcd_raw)], dim=-1)) # (N*M, 2), Magenta

    # Make folder
    vis_folder = os.path.join(dir_path, 'vis', f'GPU_{global_rank}', 'train', f'E{current_epoch}')
    os.makedirs(vis_folder, exist_ok=True)
    save_pc(os.path.join(vis_folder, f"E{current_epoch}_neg_hard_mask.ply"), pcd_list_for_viz)
    save_lineset(os.path.join(vis_folder, f"E{current_epoch}_neg_hard_mask_lineset.ply"), total_points_for_lineset, total_lines_for_lineset)

    # Simple logging for distance
    max_dist_in_pos = corrd_dist.reshape(-1)[pos_mask.reshape(-1)].max().item() if pos_mask.sum() > 0 else 0
    min_dist_in_hard_negs = corrd_dist.reshape(-1)[processed_hard_neg_mask.reshape(-1)].min().item() if processed_hard_neg_mask.sum() > 0 else 0
    with open(os.path.join(vis_folder, f"E{current_epoch}_neg_hard_mask.txt"), "w") as f:
        f.write(f"max_dist_in_pos: {max_dist_in_pos}, min_dist_in_hard_negs: {min_dist_in_hard_negs}")



def save_pc(filename: str, pcd_tensors: list):
    colors = list(global_colors_for_objs.values())

    pcds = []
    for i, tensor_ in enumerate(pcd_tensors):
        if len(tensor_.shape) == 0:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tensor_.cpu().numpy())
        pcd.paint_uniform_color(colors[i % len(colors)])  # Assign color based on index
        pcds.append(pcd)
    
    combined_cloud = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_cloud += pcd
    
    o3d.io.write_point_cloud(filename, combined_cloud)


def save_lineset(filename: str, points: torch.Tensor, lines: list):
    """
    Save lineset as PLY files.

    Args:
        filename (str): filename to save
        points (torch.Tensor): (N, 3)
        lines (list): each element is (corr, 2)
    """
    colors = list(global_colors_for_objs.values())

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points.cpu().numpy())

    all_lines = []
    all_colors = []

    for i, tensor_ in enumerate(lines):
        if len(tensor_.shape) == 0:
            continue

        curr_lines = tensor_.cpu().numpy()
        all_lines.append(curr_lines)
        
        expanded_colors = [colors[i % len(colors)]] * len(tensor_)
        expanded_colors = np.array(expanded_colors)
        all_colors.append(expanded_colors)


    if len(all_lines) > 0:
        all_lines = np.vstack(all_lines)
        all_colors = np.vstack(all_colors)
    
    lineset.lines = o3d.utility.Vector2iVector(all_lines)
    lineset.colors = o3d.utility.Vector3dVector(all_colors)
    
    o3d.io.write_line_set(filename, lineset)

