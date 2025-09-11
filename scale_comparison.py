import torch
import numpy as np
from common.utils import save_pc


# load npy
for i in range(10):
    everyday_pcd = np.load(f'./vis_everyday/{i}.npy')
    fantastic_pcd = np.load(f'./vis_fantastic/{i}.npy')

    # npy to torch
    everyday_pcd = torch.from_numpy(everyday_pcd)
    fantastic_pcd = torch.from_numpy(fantastic_pcd)

    # save pcd
    save_pc(f'./vis_combined/{i}.pcd', [everyday_pcd, fantastic_pcd])
