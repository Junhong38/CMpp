## SE(3)-Equivariant Geometric Assembly for CVPR'25

Pytorch-lightning Implementation of SE(3)-Equivariant Geometric Assembly for CVPR'25


## Requirements
```
conda create -n CMpp python=3.12 -y
conda activate CMpp

# pytorch 2.4.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 11.8 Runtime
(base 말고 해당 env 활성화)
mamba install -c pytorch -c nvidia pytorch-cuda=11.8

# pytorch3D
pip install iopath
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py312_cu118_pyt241.tar.bz2
conda install pytorch3d-0.7.8-py312_cu118_pyt241.tar.bz2

pip install pytorch-lightning==2.5.5

pip install einops trimesh wandb open3d
pip install git+https://github.com/KinglittleQ/torch-batch-svd
pip install git+'https://github.com/otaheri/chamfer_distance'

# compile pointops
cd pointcept_libs/pointops2/
python setup.py install
```

### Useful commands
```
killall -9 /home/nahyuklee/miniforge3/envs/equiassem/bin/python
```