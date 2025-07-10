import numpy as np
import open3d as o3d
import torch
import trimesh

def to_o3d_pcd(pts):
    '''
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def get_correspondences(src_pcd, tgt_pcd, search_voxel_size, K=None):
    '''
    Give source & target point clouds as well as the relative transformation between them, calculate correspondences according to give threshold
    :param src_pcd: source point cloud
    :param tgt_pcd: target point cloud
    :param search_voxel_size: given threshold
    :param K: if k is not none, select top k nearest neighbors from candidate set after radius search
    :return: (m, 2) torch tensor, consisting of m correspondences
    '''

    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def check_mesh_intersection(mesh1, mesh2):
    try:
        bbox1_min, bbox1_max = mesh1.bounding_box.bounds
        bbox2_min, bbox2_max = mesh2.bounding_box.bounds

        # AABB 겹치는지 확인 (각 축에서 최소/최대 값 비교)
        if np.any(bbox1_max < bbox2_min) or np.any(bbox2_max < bbox1_min):
            return False
        scene = trimesh.collision.CollisionManager()
        scene.add_object("mesh1", mesh1)
        scene.add_object("mesh2", mesh2)

        return scene.in_collision_internal()
    except Exception as e:
        print(f"Collision check failed: {e}")
        return False

def are_meshes_connected(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    decimals: int = 5,
):
    """
    It is from GARF code (GARF/scripts/process_breakingbad.py)
    Check if two meshes are connected.

    Args:
        mesh_a (trimesh.Trimesh): The first mesh.
        mesh_b (trimesh.Trimesh): The second mesh.
        decimals (int, optional): The number of decimal places to round the vertices to. Defaults to 5.

    Returns:
        bool: True if the meshes are connected, False otherwise.
    """
    vertices_a = mesh_a.vertices
    vertices_b = mesh_b.vertices
    faces_a = mesh_a.faces
    faces_b = mesh_b.faces

    shared_faces_a = np.zeros(len(faces_a), dtype=bool)
    shared_faces_b = np.zeros(len(faces_b), dtype=bool)

    vertices_a = vertices_a.round(decimals=decimals)
    vertices_b = vertices_b.round(decimals=decimals)

    common_vertices = set(map(tuple, vertices_a)).intersection(map(tuple, vertices_b))

    # calculate common faces
    if len(common_vertices) > 0:
        for i, face_a in enumerate(faces_a):
            if all([tuple(vertices_a[vertex]) in common_vertices for vertex in face_a]):
                shared_faces_a[i] = True
        for i, face_b in enumerate(faces_b):
            if all([tuple(vertices_b[vertex]) in common_vertices for vertex in face_b]):
                shared_faces_b[i] = True

    return common_vertices, shared_faces_a, shared_faces_b