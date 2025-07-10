import torch
from functools import partial
 
 
def move_to_device(dloader, device):
    """Move to device - IDENTICAL to original"""
    device_dloader = {}
    for key, value in dloader.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], torch.Tensor):
                device_dloader[key] = [v.to(device) if isinstance(
                    v, torch.Tensor) else v for v in value]
            else:
                device_dloader[key] = value
        elif isinstance(value, torch.Tensor):
            device_dloader[key] = value.to(device)
        else:
            device_dloader[key] = value
    return device_dloader
 
 
def load_segmentation_model(model_path, device, grid_size=0.02):
    """Load segmentation model - IDENTICAL to original"""
    import torch
    from assembly.models.pretraining import FracSeg
    from assembly.backbones.pointtransformerv3 import PointTransformerV3
 
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"]
 
    encoder = PointTransformerV3(
        in_channels=6,
        stride=[2, 2, 2],
        enc_depths=[2, 2, 6, 2],
        enc_channels=[32, 64, 128, 256],
        enc_num_head=[2, 4, 8, 16],
        enc_patch_size=[1024, 1024, 1024, 1024],
        dec_depths=[2, 2, 2],
        dec_channels=[64, 128, 128],
        dec_num_head=[4, 8, 16],
        dec_patch_size=[1024, 1024, 1024]
    )
 
    optimizer = partial(torch.optim.AdamW, lr=1e-4, weight_decay=1e-5)
    model = FracSeg(
        pc_feat_dim=64,
        encoder=encoder,
        optimizer=optimizer,
        grid_size=grid_size
    )
 
    ## weight permutation
    for key in state_dict:
        if key in [
            'encoder.embedding.stem.conv.weight',
            'encoder.enc.enc0.block0.cpe.0.weight',
            'encoder.enc.enc0.block1.cpe.0.weight',
            'encoder.enc.enc1.block0.cpe.0.weight',
            'encoder.enc.enc1.block1.cpe.0.weight',
            'encoder.enc.enc2.block0.cpe.0.weight',
            'encoder.enc.enc2.block1.cpe.0.weight',
            'encoder.enc.enc2.block2.cpe.0.weight',
            'encoder.enc.enc2.block3.cpe.0.weight',
            'encoder.enc.enc2.block4.cpe.0.weight',
            'encoder.enc.enc2.block5.cpe.0.weight',
            'encoder.enc.enc3.block0.cpe.0.weight',
            'encoder.enc.enc3.block1.cpe.0.weight',
            'encoder.dec.dec2.block0.cpe.0.weight',
            'encoder.dec.dec2.block1.cpe.0.weight',
            'encoder.dec.dec1.block0.cpe.0.weight',
            'encoder.dec.dec1.block1.cpe.0.weight',
            'encoder.dec.dec0.block0.cpe.0.weight',
            'encoder.dec.dec0.block1.cpe.0.weight'
            ]:
            weight = state_dict[key]
            state_dict[key] = weight.permute(1,2,3,0,4)


    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
 
 
def segment_parts(model, dloader, device):
    """Segment parts - IDENTICAL to original"""
    model.eval()
    device_dloader = move_to_device(dloader, device)
 
    with torch.no_grad():
        seg_out = model(device_dloader)['coarse_seg_pred_binary']
 
    points_per_part = dloader['points_per_part'][0][0]
    seg_mask1 = seg_out[:points_per_part]
    seg_mask2 = seg_out[points_per_part:]
    return seg_mask1, seg_mask2
 
 
if __name__ == "__main__":
    model_path = "./checkpoint/GARF_mini.ckpt"
    device = "cuda"
    model = load_segmentation_model(model_path, device)
    dloader = {
        "pointclouds": torch.randn(1000, 3).unsqueeze(0).repeat(8,1,1),
        "pointclouds_normals": torch.randn(1000, 3).unsqueeze(0).repeat(8,1,1),
        "points_per_part": torch.tensor([500, 500]).unsqueeze(0).repeat(8,1),
        "graph": torch.tensor([500, 500]).unsqueeze(0).repeat(8,1)
    }
    seg_mask1, seg_mask2 = segment_parts(model, dloader, device)
    print("Done")