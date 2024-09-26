## SE(3)-Equivariant Geometric Assembly for ICLR'25

Pytorch-lightning Implementation of SE(3)-Equivariant Geometric Assembly for ICLR'25

## Requirements
```
conda create -n equiassem python=3.8 -y
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.9.5
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install einops
pip install tensorboardX
pip install --ignore-installed PyYAML

pip install trimesh
pip install rtree
pip install pytorch3d
pip install tensorflow

pip install wandb
pip install setuptools==59.5.0
pip install open3d
pip install thop

pip install git+https://github.com/KinglittleQ/torch-batch-svd
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+'https://github.com/otaheri/chamfer_distance'
```

### Requirements (New)
```
mamba create -n equiassem_new python=3.8.19 -y
mamba activate equiassem_new
# pytorch: 2.4.1
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y 
# pytorch-lightning: 2.0.8
pip install pytorch-lightning==2.0.8
pip install einops trimesh wandb open3d
pip install git+https://github.com/KinglittleQ/torch-batch-svd
pip install git+'https://github.com/otaheri/chamfer_distance'
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

### Requirements (New - Unet)
```
mamba create -n equiassem python=3.8 -y
mamba activate equiassem
# pytorch 1.10.1 (<= 1.11), use pip
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install pytorch-lightning==1.9
pip install einops trimesh wandb open3d
pip install git+https://github.com/KinglittleQ/torch-batch-svd
pip install git+'https://github.com/otaheri/chamfer_distance'
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

```

#
killall -9 /home/nahyuklee/miniforge3/envs/equiassem/bin/python