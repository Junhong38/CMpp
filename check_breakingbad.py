import os
import pwd
import argparse
import torch
import numpy as np

import glob
import trimesh
import open3d as o3d
from tqdm import tqdm
from common.viz import make_arrows_from_vector_tensors


def write_list_to_txt(list, path):
    """
    Write list to txt file

    Args:
        list (list): list to write
        path (str): path to the txt file
    """
    with open(path, 'w') as f:
        for item in list:
            f.write(f"{item}\n")
    return


def read_data_instances(dataset_path, data_category):
    """
    Read all data instances from the dataset

    Args:
        dataset_path (str): path to the dataset
        data_category (str): data category

    Returns:
        list: all data instances
    """
    all_instances = []

    # Read data split txt files
    path_for_txt = [os.path.join(dataset_path, '../' 'data_split', f'{data_category}.train.txt'), 
                    os.path.join(dataset_path, '../' 'data_split', f'{data_category}.val.txt')]

    
    for txt_file in path_for_txt:
        with open(txt_file, 'r') as f:
            data_dirs = [x.strip() for x in f.readlines() if x.strip()]
        
        for data_dir in data_dirs:
            # Get all cases in the data directory
            total_cases = glob.glob(os.path.join(dataset_path, data_dir, '*'))
            all_instances += total_cases

    all_instances = sorted(all_instances)
    return all_instances


def read_data_list(dataset_path, path_for_txt):
    """
    Read data list from the txt file

    Args:
        dataset_path (str): path to the dataset
        path_for_txt (str): path to the txt file, which comes from CM
    """
    with open(path_for_txt, 'r') as f:
        results = [x.strip() for x in f.readlines() if x.strip()]
    
    data_instances = {}
    for result in results:
        n_frac, filepath = result.split()
        data_instances[os.path.join(dataset_path, filepath)] = n_frac

    return data_instances


def sample_points_from_mesh(mesh, n_pts=5000):
    """
    Sample points from the mesh
    """
    sampled_pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_pts) # (N, 3), (N, )
    sampled_pts = torch.tensor(sampled_pts).float() # (N, 3)
    return sampled_pts, face_idx


def trimesh_to_open3d(trimesh_mesh):
    """
    Convert trimesh mesh to Open3D mesh
    
    Args:
        trimesh_mesh: trimesh mesh object
        
    Returns:
        o3d.geometry.TriangleMesh: Open3D mesh object
    """
    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    
    # Set vertices and faces
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices.copy())
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces.copy())
    return o3d_mesh


def combine_mesh_with_arrows(trimesh_mesh, arrows):
    """
    Combine trimesh mesh with Open3D arrows into a single Open3D mesh
    
    Args:
        trimesh_mesh: trimesh mesh object
        arrows: list of Open3D arrow meshes
        
    Returns:
        o3d.geometry.TriangleMesh: Combined mesh
    """
    # Convert trimesh to Open3D
    o3d_mesh = trimesh_to_open3d(trimesh_mesh)
    
    # Create combined mesh starting with the original mesh
    combined_mesh = o3d.geometry.TriangleMesh()

    # Add original mesh to the combined mesh
    combined_mesh += o3d_mesh
    
    # Add all arrows to the combined mesh
    for arrow in arrows:
        combined_mesh += arrow
    
    return combined_mesh


def save_arrows_with_mesh(arrows, mesh, save_dir, mesh_name):
    """
    Save arrows with mesh as separate files and combined mesh
    """
    # Save original mesh separately
    mesh.export(os.path.join(save_dir, f"{mesh_name}-ori.ply"))
    
    # Create and save combined mesh
    combined_mesh = combine_mesh_with_arrows(mesh, arrows)
    o3d.io.write_triangle_mesh(os.path.join(save_dir, f"{mesh_name}_wn.ply"), combined_mesh)
    

def check_outward_normals(pts, normals, mesh):
    """
    Check if normals are pointing outward from the mesh
    
    Args:
        pts: torch.Tensor or numpy.ndarray of shape (N, 3) - points where normals are computed
        normals: torch.Tensor or numpy.ndarray of shape (N, 3) - normal vectors
        mesh: trimesh mesh object
        
    Returns:
        dict: Dictionary containing:
            - 'is_outward': bool array indicating if each normal is outward
            - 'outward_ratio': float - ratio of outward normals
            - 'outward_count': int - number of outward normals
            - 'total_count': int - total number of normals
    """
    normals = normals.cpu().numpy()
    pts = pts.cpu().numpy()
    
    # Ensure normals are unit vectors
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Get mesh center (centroid)
    mesh_center = mesh.centroid
    
    # Compute vectors from mesh center to each point
    center_to_points = pts - mesh_center
    center_to_points = center_to_points / np.linalg.norm(center_to_points, axis=1, keepdims=True)
    
    # Check if normal and center_to_point vectors are pointing in the same direction
    # (dot product > 0 means they point in similar directions)
    dot_products = np.sum(normals * center_to_points, axis=1)
    
    is_outward = dot_products > 0

    return np.all(is_outward)


def read_and_check_mesh(mesh_path, save_dir):
    """
    Read mesh and check if the number of vertices is correct
    
    Args:
        mesh_path (str): path to the mesh
        save_dir (str): path to the save directory
    """
    # ../class/id/frac or mode/*.obj
    # So, name will be class-id-fac-#.ply
    split_mesh_path = mesh_path.split('/')
    split_mesh_name = split_mesh_path[-1].split('.')[0]
    mesh_name = f"{split_mesh_path[-4]}-{split_mesh_path[-3]}-{split_mesh_path[-2]}-{split_mesh_name}"

    # if mesh_name != 'Mirror-4b3e576378e5571aa9a81fd803d87d3e-fractured_73-piece_1':
    #     return None

    # Load mesh
    mesh = trimesh.load_mesh(mesh_path)

    # Watertight check
    if not mesh.is_watertight:
        # Sample points from the mesh
        sampled_pts, face_idx = sample_points_from_mesh(mesh)

        # Fix normals, and extract normals
        mesh.fix_normals()
        normals = mesh.face_normals[face_idx]
        normals = torch.tensor(normals).float()

        # Check if normals are pointing outward
        outward_result = check_outward_normals(sampled_pts, normals, mesh)

        if outward_result: # If all normals are pointing outward, then return None
            print(f"this is not watertight, but it is outward normal mesh: {mesh_path}")
            return None

        # Visualize the mesh and corresponding normals
        arrows = make_arrows_from_vector_tensors(pcds=[sampled_pts], vectors=[normals], colors=['red'], cylinder_radius=0.001, cone_radius=0.002, arrow_scale=0.01)

        # Save arrows with mesh
        save_arrows_with_mesh(arrows, mesh, save_dir, mesh_name)

        return str(mesh_path)
    
    else:
        # This is watertight mesh
        return None


def check_data_instances(all_instances, CM_data_dict, save_dir):
    """
    Check if all data instances are the same as the data list from CM

    Args:
        all_instances (list): all data instances
        CM_data_dict (dict): data list from CM
        save_dir (str): path to the save directory
    """
    sorted_CM_data_list = sorted(list(CM_data_dict.keys()))

    watertight_objs = []
    not_watertight_objs = []

    for ith, gt_path in tqdm(enumerate(all_instances), total=len(all_instances)):
        cm_path = sorted_CM_data_list[ith]
        cm_n_frac = int(CM_data_dict[cm_path])

        # Check if the path is the same
        assert gt_path == cm_path, f"gt_path and cm_path are not the same, {gt_path} != {cm_path}"

        all_objs = glob.glob(os.path.join(gt_path, '*.obj'))

        # Check if the number of fragments is correct
        gt_n_frac = len(all_objs)
        assert gt_n_frac == cm_n_frac, f"gt_n_frac and cm_n_frac are not the same,\nFrom [gt_path: {gt_path}] gt_n_frac: {gt_n_frac} != cm_n_frac: {cm_n_frac}"

        if gt_n_frac != 2:
            continue


        # Read mesh and check if the number of vertices is correct
        for ith_obj in all_objs:
            check_result = read_and_check_mesh(ith_obj, save_dir)

            if check_result is not None:
                print(f"not watertight obj: {check_result}")
                not_watertight_objs.append(check_result)
            else:
                watertight_objs.append(check_result)

    return not_watertight_objs, watertight_objs



def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    all_instances = read_data_instances(args.datapath, args.data_category)
    print(f"all_instances: {len(all_instances)}")

    CM_data_dict = read_data_list(args.datapath, f'./data/data_list/{args.data_category}_train.txt')
    CM_data_dict.update(read_data_list(args.datapath, f'./data/data_list/{args.data_category}_val.txt'))
    CM_data_list = sorted(list(CM_data_dict.keys()))
    print(f"CM_data_list: {len(CM_data_list)}")

    # If this is not same, then there is a problem in CM data list
    assert len(all_instances) == len(CM_data_list), "all_instances and CM_data_list are not the same"

    # Check data instances
    not_watertight_objs, watertight_objs = check_data_instances(all_instances, CM_data_dict, args.save_dir)

    print(f"not_watertight_objs: {len(not_watertight_objs)}")
    print(f"watertight_objs: {len(watertight_objs)}")
    print(f"total: {len(not_watertight_objs) + len(watertight_objs)}")

    # Write not watertight objects to txt file
    write_list_to_txt(not_watertight_objs, os.path.join(args.save_dir, 'not_watertight_objs.txt'))
    write_list_to_txt(watertight_objs, os.path.join(args.save_dir, 'watertight_objs.txt'))


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')

    # Dataset arguments
    parser.add_argument('--datapath', type=str, default='/mnt/nvme2n1p1/kimsangki_datasets/breaking_bad/volume_constrained') 
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'synthetic'])

    # Save directory arguments
    parser.add_argument('--save_dir', type=str, default='check_results')


    args = parser.parse_args()

    print(f"args: {args}")

    main(args)



"""
rm -rf check_results_everyday/ && python check_breakingbad.py --save_dir check_results_everyday
rm -rf check_results_artifact/ && python check_breakingbad.py --data_category artifact --save_dir check_results_artifact
"""