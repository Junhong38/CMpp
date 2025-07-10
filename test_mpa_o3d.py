import os
import sys
import pwd
import argparse
import importlib
import time
import gc
from distutils.dir_util import copy_tree

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.transform import Rotation

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset import GADataset
from common import utils
import open3d as o3d

from model.equiassem import EquiAssem

from data.utils import to_o3d_pcd
import itertools

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

def transf(pcd_tensor, transform):
    points_np = pcd_tensor.squeeze(0).cpu().numpy()  # Convert to NumPy array if on GPU
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    
    # Convert the transform tensor to a numpy array
    transform_np = transform.cpu().numpy()
    
    # Apply the transformation
    point_cloud.transform(transform_np)
    
    # Convert back to numpy array and then to torch.Tensor
    transformed_points_np = np.asarray(point_cloud.points)
    transformed_points_tensor = torch.from_numpy(transformed_points_np).float().unsqueeze(0)
    
    # Ensure the device of the output tensor matches the input tensor
    return transformed_points_tensor.to(pcd_tensor.device)

@torch.no_grad()
def multi_part_assemble(args):
    model = EquiAssem(lr=args.lr,
                        backbone=args.backbone,
                        shape=args.shape, 
                        occ=args.occ, 
                        shape_loss=args.shape_loss, 
                        occ_loss=args.occ_loss, 
                        no_ori=args.no_ori,
                        visualize=args.visualize,
                        debug=args.debug)
    print(model)

    model.to(torch.device('cuda:0'))
    model.eval()
    # Load checkpoint
    if args.load:
        print(f"Loading checkpoint from {args.load}")
        checkpoint = torch.load(args.load, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("No checkpoint specified. Exiting program.")
        sys.exit(1)
    
    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.min_part, args.max_part, args.n_pts, args.scale)
    dataloader_test = GADataset.build_dataloader(args.batch_size, args.n_worker, 'test')

    for idx, in_dict in enumerate(dataloader_test):
        # 1. Compute relative transformations
        in_dict = utils.to_cuda(in_dict)
        pred_relative_trsfm, corr_scores = {}, {}
        pair_indices = list(itertools.permutations([i for i in range(in_dict['n_frac'])], 2))
        for pair_idx0, pair_idx1 in pair_indices:
            in_dict_pair = {
                'pcd_t': [in_dict['pcd_t'][pair_idx0], in_dict['pcd_t'][pair_idx1]],
                'pcd': [in_dict['pcd'][pair_idx0], in_dict['pcd'][pair_idx1]],
                'gt_trans': [in_dict['gt_trans'][pair_idx0], in_dict['gt_trans'][pair_idx1]],
                'gt_rotat': [in_dict['gt_rotat'][pair_idx0], in_dict['gt_rotat'][pair_idx1]],
                'gt_rotat_inv': [in_dict['gt_rotat_inv'][pair_idx0], in_dict['gt_rotat_inv'][pair_idx1]],
                'gt_trans_inv': [in_dict['gt_trans_inv'][pair_idx0], in_dict['gt_trans_inv'][pair_idx1]],
                'relative_trsfm': {f'{pair_idx0}-{pair_idx1}': in_dict['relative_trsfm']},
            }
            pred_relative_trsfm[f'{pair_idx0}-{pair_idx1}'], corr_scores[f'{pair_idx0}-{pair_idx1}'] = model.forward_mpa(in_dict_pair)
        
        # 2. Pose Graph Optimization
        pose_graph = o3d.pipelines.registration.PoseGraph()

        # Add nodes (parts) to the pose graph
        for i in range(in_dict['n_frac']):
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

        # Add all edges to the pose graph
        for src in range(in_dict['n_frac']):
            for dst in range(in_dict['n_frac']):
                if src != dst:
                    transform = pred_relative_trsfm[f'{src}-{dst}']
                    R = transform[0].squeeze(0).cpu().numpy()
                    t = transform[1].squeeze(0).cpu().numpy()
                    
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = R
                    transform_matrix[:3, 3] = -t
                    mean_corr_score = corr_scores[f"{src}-{dst}"].mean().item()
                    information_corr = np.eye(6) * mean_corr_score**2
                    information_corr = (information_corr + information_corr.T) / 2
                    
                    information_pcd = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                                    to_o3d_pcd(in_dict['pcd_t'][src].squeeze(0).cpu()), 
                                    to_o3d_pcd(in_dict['pcd_t'][dst].squeeze(0).cpu()), 
                                    0.018, transform_matrix)
                    
                    # Combine both information matrices
                    information = information_corr
                    
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(src, dst, transform_matrix, information, uncertain=False)
                    )
        print(f"[{in_dict['eval_idx'][0].item()}] Pose graph has {len(pose_graph.nodes)} nodes and {len(pose_graph.edges)} edges.")
        
        # Optimize the pose graph
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.018,
            edge_prune_threshold=0.25,
            reference_node=in_dict['anchor_idx'][0]
        )
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()

        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria,
            option
        )

        # Extract optimized poses
        optimized_poses = [node.pose.copy() for node in pose_graph.nodes]
        optimized_poses = [torch.from_numpy(pose).float().to(in_dict['pcd_t'][0].device) for pose in optimized_poses]
        

        transformed_pcds = []
        for i, pcd in enumerate(in_dict['pcd_t']):
            # Extract the rotation and translation from the 4x4 transformation matrix
            transformed_pcd = transf(pcd, optimized_poses[i])
            transformed_pcds.append(transformed_pcd)
        utils.save_pc(f'./vis_mpa/{idx}_pred.pcd', transformed_pcds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivariant Assembly Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'synthetic'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--min_part', type=int, default=2)
    parser.add_argument('--max_part', type=int, default=20)

    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--load', type=str, default='')

    parser.add_argument('--scale', type=str, default='full', choices=['full', 'small', 'overfitting'])

    # Ablation studies
    parser.add_argument('--backbone', type=str, default='unet', choices=['dgcnn', 'unet'])
    parser.add_argument('--shape', type=str, default='local', choices=['local', 'global'])
    parser.add_argument('--occ', type=str, default='local', choices=['local', 'global'])
    parser.add_argument('--shape_loss', type=str, default='positive', choices=['positive', 'negative'])
    parser.add_argument('--occ_loss', type=str, default='negative', choices=['positive', 'negative'])
    parser.add_argument('--no_ori', action='store_false')

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')
      
    args = parser.parse_args()

    multi_part_assemble(args)
