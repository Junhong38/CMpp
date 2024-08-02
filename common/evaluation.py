r""" Evaluate assembly result """
import torch

from chamfer_distance import ChamferDistance as chamfer_dist
from common.rotation import Rotation3D
from scipy.spatial.transform import Rotation
import open3d as o3d
import random
import numpy as np

def save_pc(filename:str, pcd_tensors:list):
    pcds = []
    for tensor_ in pcd_tensors:
        if tensor_.size()[0] == 1:
            tensor_ = tensor_.squeeze(0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tensor_.cpu().numpy())
        pcd.paint_uniform_color([random.uniform(0, 1) for _ in range(3)])
        pcds.append(pcd)
    combined_cloud = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_cloud += pcd
    o3d.io.write_point_cloud(filename, combined_cloud)

class Evaluator:
    r""" Computes assembly results """
    @classmethod
    def initialize(cls):
        cls.mesh_buffer = {}

    @classmethod
    @torch.no_grad()
    def evaluate_prediction(cls, in_dict, out_dict, visualize=False):

        # Init return buffer
        eval_result = {}

        pred_relative_trsfm = out_dict['estimated_rotat'], out_dict['estimated_trans'] 
        grtr_relative_trsfm = [x.squeeze(0) for x in in_dict['relative_trsfm']['0-1']]
        src_pcd, trg_pcd = [x.squeeze(0) for x in in_dict['pcd_t']]

        is_trg_larger = cls._is_trg_larger(src_pcd, trg_pcd)
        # Assemble using prediction, pseudo-gt, and ground-truth
        assm_pred, pcds_pred = cls._pairwise_mating(src_pcd, trg_pcd, pred_relative_trsfm[0], pred_relative_trsfm[1], is_trg_larger)
        assm_grtr, pcds_grtr = cls._pairwise_mating(src_pcd, trg_pcd, grtr_relative_trsfm[0], grtr_relative_trsfm[1], is_trg_larger)
        
        # (a) Compute CD between prediction & ground-truth
        eval_result['cd'] = cls._chamfer_distance(assm_pred, assm_grtr, is_trg_larger)

        # (b) Compute MSE between prediction & ground-truth for rotation (in degree) and translation
        eval_result['rrmse'], eval_result['trmse'] = cls._transformation_error(pred_relative_trsfm, grtr_relative_trsfm, multi_part=False)
        
        # (c) Compute CoRrespondence Distance (CRD) betwween prediction & ground-truth
        eval_result['crd'] = cls._correspondence_distance(assm_pred, assm_grtr, is_trg_larger)

        if visualize:
            in_dict['eval_idx'][0] = torch.tensor(99999)
            save_pc(f"vis/{in_dict['eval_idx'][0].item()}_src_pcd.pcd", [pcds_pred[0]])
            save_pc(f"vis/{in_dict['eval_idx'][0].item()}_trg_pcd.pcd", [pcds_pred[1]])
            # np.save(f"vis/{in_dict['eval_idx'][0].item()}_src_ori", out_dict['src_ori'].squeeze(0).cpu().detach().numpy())
            # np.save(f"vis/{in_dict['eval_idx'][0].item()}_trg_ori", out_dict['trg_ori'].squeeze(0).cpu().detach().numpy())
            np.save(f"vis/{in_dict['eval_idx'][0].item()}_src_in_ori", out_dict['src_in_ori'].squeeze(0).cpu().detach().numpy())
            np.save(f"vis/{in_dict['eval_idx'][0].item()}_trg_in_ori", out_dict['trg_in_ori'].squeeze(0).cpu().detach().numpy())
            np.save(f"vis/{in_dict['eval_idx'][0].item()}_src_ex_ori", out_dict['src_ex_ori'].squeeze(0).cpu().detach().numpy())
            np.save(f"vis/{in_dict['eval_idx'][0].item()}_trg_ex_ori", out_dict['trg_ex_ori'].squeeze(0).cpu().detach().numpy())
            
        return eval_result

    @classmethod
    def _transform_mesh(cls, mesh, rotat, trans, rotate_first=True):
        pcd = torch.tensor(mesh.vertices).float()
        mesh_t = mesh.copy()
        mesh_t.vertices = cls._transform(pcd, rotat, trans)
        return mesh_t

    @classmethod
    def _correspondence_distance(cls, assm1, assm2, is_trg_larger, scaling=100):
        corr_dist = (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling

        # Larger src in CRD evaluation
        if is_trg_larger: corr_dist = corr_dist.flip(dims=[0])

        return corr_dist

    @classmethod
    def _chamfer_distance(cls, assm1, assm2, is_trg_larger, scaling=1000):
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
        cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling

        # Larger src in CD evaluation
        if is_trg_larger: cd = cd.flip(dims=[0])

        return cd

    @classmethod
    def _transformation_error(cls, trnsf1, trnsf2, multi_part, rrmse_scaling=100):
        if multi_part:
            rotat1, trans1 = trnsf1
            rotat2, trans2 = trnsf2
        else:
            rotat1, trans1 = [trnsf1[0]], [trnsf1[1]]
            rotat2, trans2 = [trnsf2[0]], [trnsf2[1]]
        rrmse, trmse = 0., 0.
        for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
            r1_deg = torch.tensor(Rotation.from_matrix(r1.cpu()).as_euler('xyz', degrees=True))
            r2_deg = torch.tensor(Rotation.from_matrix(r2.cpu()).as_euler('xyz', degrees=True))
            diff1 = (r1_deg - r2_deg).abs()
            diff2 = 360. - (r1_deg - r2_deg).abs()
            diff = torch.minimum(diff1, diff2)
            rrmse += diff.pow(2).mean().pow(0.5)
            trmse += (t1 - t2).pow(2).mean().pow(0.5) * rrmse_scaling
        div = len(rotat1) if multi_part else 1
        return rrmse / div, trmse / div

    @classmethod
    def _is_trg_larger(cls, src_pcd, trg_pcd):
        src_volume = (src_pcd.max(dim=0)[0] - src_pcd.min(dim=0)[0]).prod(dim=0)
        trg_volume = (trg_pcd.max(dim=0)[0] - trg_pcd.min(dim=0)[0]).prod(dim=0)

        return src_volume < trg_volume

    @classmethod
    def _multi_part_assemble(cls, pcds, rotat, trans):
        pcd_t = []
        for pcd, R, t in zip(pcds, rotat, trans):
            pcd_t.append(cls._transform(pcd.squeeze(0), R.inverse(), t, False))
        return torch.cat(pcd_t, dim=0), pcd_t

    @classmethod
    def _pairwise_mating(cls, src_pcd, trg_pcd, rotat, trans, is_trg_larger):
        pcd_t = []
        if is_trg_larger:
            src_pcd_t = cls._transform(src_pcd.squeeze(0), rotat, -trans, True)
            pcd_t = [src_pcd_t, trg_pcd.squeeze(0)]
        else:
            trg_pcd_t = cls._transform(trg_pcd.squeeze(0), rotat.inverse(), trans, False)
            pcd_t = [src_pcd.squeeze(0), trg_pcd_t]
        return torch.cat(pcd_t, dim=0), pcd_t

    @classmethod
    def _transform(cls, pcd, rotat=None, trans=None, rotate_first=True):
        if rotat == None: rotat = torch.eye(3, 3)
        if trans == None: trans = torch.zeros(3)

        rotat = rotat.to(pcd.device)
        trans = trans.to(pcd.device)

        if rotate_first:
            return torch.einsum('x y, n y -> n x', rotat, pcd) + trans
        else:
            return torch.einsum('x y, n y -> n x', rotat, pcd + trans)