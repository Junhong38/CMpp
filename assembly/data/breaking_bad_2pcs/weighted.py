import trimesh
import numpy as np
from .base import BreakingBad2PcsBase
from assembly.data.transform import recenter_pc, rotate_pc, shuffle_pc


class BreakingBad2PcsWeighted(BreakingBad2PcsBase):
    """
    Two-piece dataset sampling based on mesh area.
    """

    def sample_points(self, meshes, shared_faces):
        # Construct trimesh objects
        mesh_objs = [trimesh.Trimesh(
            vertices=m['vertices'], faces=m['faces']) for m in meshes]
        areas = [m.area for m in mesh_objs]
        total_area = sum(areas)

        # Calculate points per part using the original strategy
        # At least 5% of total points
        min_points = max(100, int(self.num_points_to_sample * 0.05))
        points_per_part = [
            min_points + int(
                (self.num_points_to_sample - min_points * len(meshes))
                * area
                / total_area
            )
            for area in areas
        ]

        # Ensure we have exactly num_points_to_sample points
        current_total = sum(points_per_part)
        if current_total != self.num_points_to_sample:
            diff = self.num_points_to_sample - current_total
            # Add/remove points from the larger part
            larger_idx = np.argmax(points_per_part)
            points_per_part[larger_idx] += diff

        # Debug print for extreme cases
        if min(points_per_part) < min_points:
            print(f"[Warning] Extreme point distribution detected:")
            print(f"  Areas: {areas}")
            print(f"  Points per part: {points_per_part}")
            print(f"  Min points threshold: {min_points}")

        # Sample points using the calculated distribution
        sampled = []
        for i, mesh in enumerate(mesh_objs):
            count = points_per_part[i]
            if self.mesh_sample_strategy == 'poisson':
                pcd, face_idx = trimesh.sample.sample_surface_even(
                    mesh, count=count)
                if len(pcd) < count:
                    extra, extra_idx = trimesh.sample.sample_surface(
                        mesh, count=count - len(pcd))
                    pcd = np.concatenate([pcd, extra], axis=0)
                    face_idx = np.concatenate([face_idx, extra_idx], axis=0)
            else:
                pcd, face_idx = trimesh.sample.sample_surface(
                    mesh, count=count)
            sampled.append((pcd, face_idx))

        # Split into point clouds, normals, and fracture masks
        pcds = [pts for pts, idx in sampled]
        normals = [mesh_objs[i].face_normals[idx]
                   for i, (_, idx) in enumerate(sampled)]
        masks = []
        for i, (_, idx) in enumerate(sampled):
            sf = shared_faces[i]
            if sf.size > 0:
                mask = (sf[idx] != -1)
            else:
                mask = np.zeros(len(idx), dtype=bool)
            masks.append(mask)

        return pcds, normals, masks

    def transform(self, data):
        # Sample points and get face indices
        meshes = data['meshes']
        shared = data['shared_faces']
        # Debug extreme area ratios to catch geometry issues
        mesh_objs = [trimesh.Trimesh(
            vertices=m['vertices'], faces=m['faces']) for m in meshes]
        areas = [mo.area for mo in mesh_objs]
        small, large = min(areas), max(areas)
        ratio = small / large if large > 0 else 0
        # if ratio < 1e-2:
        # print(
        #     f"[DebugArea] sample={data['name']} areas={areas}, small/large={ratio:.6f}")
        # end debug
        pcds, normals_gt, masks = self.sample_points(meshes, shared)
        # Points per part and offsets
        points_per_part = np.array([len(pc) for pc in pcds], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(points_per_part)])

        # Concatenate ground-truth pointclouds, normals, and fracture masks
        pointclouds_gt = np.concatenate(pcds, axis=0)
        normals_gt = np.concatenate(normals_gt, axis=0)
        fracture_surface_gt = np.concatenate(masks, axis=0).astype(np.int8)

        # Apply a random global rotation
        pointclouds_gt, normals_gt, init_rot = rotate_pc(
            pointclouds_gt, normals_gt)

        # Prepare lists for transformed data
        transformed_pcs = []
        transformed_normals = []
        quaternions = []
        translations = []
        scales = []

        # Process each part independently
        for i in range(2):
            start, end = offsets[i], offsets[i+1]
            pc = pointclouds_gt[start:end]

            nm = normals_gt[start:end]
            mask = fracture_surface_gt[start:end]

            # Recenter each fragment
            pc, trans = recenter_pc(pc)

            # Rotate fragment
            pc, nm, quat = rotate_pc(pc, nm)

            # Shuffle fragment
            pc, nm, order = shuffle_pc(pc, nm)

            # Apply same shuffle to mask and GT arrays
            mask = mask[order]
            pointclouds_gt[start:end] = pointclouds_gt[start:end][order]
            normals_gt[start:end] = normals_gt[start:end][order]
            fracture_surface_gt[start:end] = mask

            # Scale fragment to unit max
            scale_val = np.max(np.abs(pc))
            scales.append(scale_val)
            pc = pc / scale_val

            transformed_pcs.append(pc)
            transformed_normals.append(nm)
            quaternions.append(quat)
            translations.append(trans)

        # Concatenate transformed fragments
        pointclouds = np.concatenate(
            transformed_pcs, axis=0).astype(np.float32)
        pointclouds_normals = np.concatenate(
            transformed_normals, axis=0).astype(np.float32)
        quaternions = np.stack(quaternions, axis=0).astype(np.float32)
        translations = np.stack(translations, axis=0).astype(np.float32)
        scales = np.array(scales, dtype=np.float32)

        # Determine reference part (the one with most points)
        ref_part = np.zeros((2,), dtype=bool)
        ref_idx = int(np.argmax(points_per_part))
        ref_part[ref_idx] = True

        # Build adjacency graph for two connected parts
        graph = np.array([[False, True], [True, False]], dtype=bool)
        return {
            'index': data['index'],
            'name': data['name'],
            'pointclouds': pointclouds,
            'pointclouds_gt': pointclouds_gt.astype(np.float32),
            'pointclouds_normals': pointclouds_normals,
            'pointclouds_normals_gt': normals_gt,
            'fracture_surface_gt': fracture_surface_gt,
            'points_per_part': points_per_part,
            'quaternions': quaternions,
            'translations': translations,
            'init_rot': init_rot,
            'scale': scales[:, None],
            'ref_part': ref_part,
            'graph': graph,
            'pieces': data.get('pieces', None),
        }
