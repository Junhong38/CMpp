## SE(3)-Equivariant Geometric Assembly for CVPR'25

Pytorch-lightning Implementation of SE(3)-Equivariant Geometric Assembly for CVPR'25


## checkpoints
https://drive.google.com/drive/folders/12IX73rSdzD3jcaH78Tfmo18g3ZLmFxsx?usp=share_link

## Requirements
```
mamba create -n equiassem python=3.8 -y # or conda
mamba activate equiassem # or conda
# pytorch 1.10.1 (<= 1.11), use pip
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install pytorch-lightning==1.9
pip install einops trimesh wandb open3d
pip install git+https://github.com/KinglittleQ/torch-batch-svd
pip install git+'https://github.com/otaheri/chamfer_distance'
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

cd lib/pointops
python setup.py install && cd -
```

### Useful commands
```
killall -9 /home/nahyuklee/miniforge3/envs/equiassem/bin/python
```
