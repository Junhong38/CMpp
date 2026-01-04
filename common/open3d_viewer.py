import argparse
import open3d as o3d

def main(args):
    point_set = o3d.io.read_point_cloud(args.pcd)
    line_set = o3d.io.read_line_set(args.line)
    o3d.visualization.draw_geometries([point_set, line_set])




if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Open3D Viewer')

    # Dataset arguments
    parser.add_argument('--pcd', type=str, default='E0_neg_hard_mask_pcd.ply')
    parser.add_argument('--line', type=str, default='E0_neg_hard_mask_line.ply')
    args = parser.parse_args()
    print(f"args: {args}")
    main(args)


"""
python viewer.py --pcd test --line ok
"""