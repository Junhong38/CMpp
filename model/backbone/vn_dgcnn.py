import torch
import torch.nn as nn
from model.backbone.vn_layers import get_graph_feature, mean_pool
from model.backbone.vn_layers import VNLinearLeakyReLU, VNMaxPool

# from lib.pointops.functions import pointops
from pointcept_libs.pointops2.functions import pointops2 as pointops

from common.misc import batch_scaling, batch2offset

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=4):
        super().__init__()
        assert stride > 1, f"stride must be greater than 1, but got {stride}"

        self.stride = stride
        self.nsample = nsample
        self.mlp = VNLinearLeakyReLU(in_planes, out_planes)

        
    def forward(self, p, x, b, o):
        """TransitionDown
        Args:
            p (torch.Tensor): (batch_size * num_points, 3), which is resposible for point coordinates
            x (torch.Tensor): (batch_size, channel, 3, num_points), which is resposible for point features
            b (torch.Tensor): (batch_size, num_points), which is resposible for point batch index
            o (torch.Tensor): (batch_size*num_parts, ), which is resposible for point offset

        Returns:
            n_p (torch.Tensor): (batch_size*sampled_points, 3)
            x (torch.Tensor): (batch_size, channel', 3, sampled_points)
            n_b (torch.Tensor): (batch_size, num_of_sampled_points), which is resposible for sampled point batch index
            n_o (torch.Tensor): (batch_size*num_parts, ), which is resposible for sampled point offset
        """
        batch_size, _, _, num_points = x.shape

        n_b = b[:, ::self.stride] # (batch_size, num_of_sampled_points)
        num_of_sampled_points = n_b.shape[1]

        # new offset, which is the number of points after stride
        n_o = batch2offset(n_b.reshape(-1)).int() # (batch_size*num_parts, )

        # FPS
        idx = pointops.furthestsampling(p, o, n_o)  # (batch_size*sampled_points, )
        n_p = p[idx.long(), :]  # (batch_size*sampled_points, 3)

        # kNN-MLP
        reshaped_x = x.permute(0,3,1,2).reshape(batch_size*num_points, -1).contiguous() # (batch_size, channel, 3, num_points) -> (batch_size, num_points, channel, 3) -> (batch_size*num_points, channel*3)
        x = pointops.queryandgroup(self.nsample, p, n_p, reshaped_x, None, o, n_o, use_xyz=False) # (batch_size*sampled_points, nsample, channel*3)
        x = x.reshape(batch_size, num_of_sampled_points, self.nsample, -1, 3) # (batch_size*sampled_points, nsample, channel*3) -> (batch_size, sampled_points, nsample, channel, 3)

        # (batch_size, sampled_points, nsample, channel, 3) -> (batch_size, channel, 3, sampled_points, nsample) -> (batch_size, channel', 3, sampled_points, nsample)
        x = self.mlp(x.permute(0,3,4,1,2))

        # Mean Pooling
        x = x.mean(dim=-1)  # (batch_size, channel', 3, sampled_points)

        return n_p, x, n_b, n_o


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.mlp1 = VNLinearLeakyReLU(out_planes, out_planes, dim=4)
        self.mlp2 = VNLinearLeakyReLU(in_planes, out_planes, dim=4)
        
    def forward(self, pxo1, pxo2):
        """TransitionUp
        Args:
            pxo1 (tuple): (p1, x1, o1) where p1 is (b*n, 3), x1 is (b,c,3,n), o1 is (b*p, )
            pxo2 (tuple, optional): (p2, x2, o2) where p2 is (b*m, 3), x2 is (b,c',3,m), o2 is (b*p, )

            where n >= m

        Returns:
            x (torch.Tensor): (n, c)
        """
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2

        batch_size1, _, _, num_points1 = x1.shape # (b,c,3,n)
        batch_size2, _, _, num_points2 = x2.shape # (b,c',3,m)
        assert batch_size1 == batch_size2, "Error: Batch size of p1 and p2 must be the same"

        # self.mlp2: (b,c',3,m) -> (b,o,3,m)
        # channel_aligned_x2: (b,o,3,m) -> (b,m,o,3) -> (b*m, o*3)
        channel_aligned_x2 = self.mlp2(x2).permute(0,3,1,2).reshape(batch_size2*num_points2, -1).contiguous()

        # p2: (b*m, 3), p1: (b*n, 3), channel_aligned_x2: (b*m, o*3), o2: (b*p), o1: (b*p)
        # pointops.interpolation: Interpolate features to enlarge p2 to p1
        # Locations of new points will be p1, and features will be interpolated from p2.
        # From new locations, find the closest points from p2, and interpolate features from them.
        # Also, we need to align channel size, so use self.mlp2 to align channel size.     
        # pointops.interpolation: (b*m, o*3) -> (b*n, o*3)
        interpolated_x2 = pointops.interpolation(p2, p1, channel_aligned_x2, o2, o1)

        # (b*n, o*3) -> (b,n,o,3) -> (b,o,3,n)
        interpolated_x2 = interpolated_x2.reshape(batch_size1, num_points1, -1, 3).permute(0,2,3,1)

        # self.mlp1: (b,c,3,n) -> (b,o,3,n)
        aligned_x1 = self.mlp1(x1)

        x = aligned_x1 + interpolated_x2

        return x


class EQCNN_equi_unet(nn.Module): 

    def __init__(self, feat_dim, pooling='mean', k=20):
        super(EQCNN_equi_unet, self).__init__()
        self.k = k

        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool
            self.pool5 = mean_pool
            self.pool6 = mean_pool
            self.pool7 = mean_pool
            self.pool8 = mean_pool
        
        # Encoder
        self.conv1 = VNLinearLeakyReLU(2, 64//3)

        self.downsample1 = TransitionDown(64//3, 64//3, stride=2, nsample=16)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3)

        self.downsample2 = TransitionDown(128//3, 128//3, stride=2, nsample=16)
        self.conv3 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.downsample3 = TransitionDown(256//3, 256//3, stride=2, nsample=16)
        self.conv4 = VNLinearLeakyReLU(256//3*2, 512//3)

        # Mid
        self.conv5 = VNLinearLeakyReLU(512//3*2, 512//3)
    
        # Decoder
        self.upsample1 = TransitionUp(512//3, 256//3)
        self.conv6 = VNLinearLeakyReLU(256//3*2, 256//3)

        self.upsample2 = TransitionUp(256//3, 128//3)
        self.conv7 = VNLinearLeakyReLU(128//3*2, 128//3)

        self.upsample3 = TransitionUp(128//3, 64//3)
        self.conv8 = VNLinearLeakyReLU(64//3*2, 64//3)

        # Proj
        self.conv9 = VNLinearLeakyReLU(64//3, feat_dim//3, dim=4, share_nonlinearity=True)
    
    def forward(self, x, batch_info):
        """EQCNN_equi_unet

        Args:
            x (torch.Tensor): (B, N+M, 3)
            batch_info (torch.Tensor): (B, num_of_objs)

        Returns:
            equi_feat (torch.Tensor): (B, feat_dim//3, 3, N+M)
        """

        batch_scaled_batch = batch_scaling(batch_info) # (batch_size, num_points)

        p1 = x.reshape(-1, 3) # (B*(N+M), 3)
        x1 = x.transpose(2, 1).unsqueeze(1) # (B, 1, 3, N+M)
        b1 = batch_scaled_batch # (B, N+M)
        o1 = batch2offset(batch_scaled_batch.reshape(-1)).int() # (B*num_of_objs, ) 
        
        ### ENCODER 1
        x1 = get_graph_feature(x1, batch_info=b1, k=self.k) # (B, 2, 3, N+M, k) 
        x1 = self.conv1(x1) # (B, 2, 3, N+M, k)  -> (B, C', 3, N+M, k)
        x1 = self.pool1(x1) # (B, C', 3, N+M, k) -> (B, C', 3, N+M)

        ### ENCODER 2
        p2, x2, b2, o2 = self.downsample1(p1, x1, b1, o1) # (B*sampled_points, 3), (B, C', 3, sampled_points), (B*num_parts, )
        x2 = get_graph_feature(x2, batch_info=b2, k=self.k)
        x2 = self.conv2(x2)
        x2 = self.pool2(x2) # (B, C, 3, (N+M)/2)

        ### ENCODER 3
        p3, x3, b3, o3 = self.downsample2(p2, x2, b2, o2)
        x3 = get_graph_feature(x3, batch_info=b3, k=self.k)
        x3 = self.conv3(x3)
        x3 = self.pool3(x3) # (B, C, 3, (N+M)/4)

        ### ENCODER 4
        p4, x4, b4, o4 = self.downsample3(p3, x3, b3, o3)
        x4 = get_graph_feature(x4, batch_info=b4, k=self.k)
        x4 = self.conv4(x4)
        x4 = self.pool4(x4) # (B, C, 3, (N+M)/8)

        ### MID
        x4 = get_graph_feature(x4, batch_info=b4, k=self.k)
        x4 = self.conv5(x4)
        x4 = self.pool5(x4) # (B, C, 3, (N+M)/8)

        ### DECODER 1
        x5 = self.upsample1((p3, x3, o3), (p4, x4, o4)) 
        x5 = get_graph_feature(x5.contiguous(), batch_info=b3, k=self.k)
        x5 = self.conv6(x5)
        x5 = self.pool6(x5) # (B, C, 3, (N+M)/4)

        ### DECODER 2
        x6 = self.upsample2((p2, x2, o2), (p3, x5, o3))
        x6 = get_graph_feature(x6.contiguous(), batch_info=b2, k=self.k)
        x6 = self.conv7(x6)
        x6 = self.pool7(x6) # (B, C, 3, (N+M)/2)

        ### DECODER 3
        x7 = self.upsample3((p1, x1, o1), (p2, x6, o2))
        x7 = get_graph_feature(x7.contiguous(), batch_info=b1, k=self.k)
        x7 = self.conv8(x7)
        x7 = self.pool8(x7) # (B, C, 3, N+M)

        equi_feat = self.conv9(x7)

        return equi_feat





class EQCNN_equi_unet_deep(nn.Module): 

    def __init__(self, feat_dim, pooling='mean', k=20):
        super(EQCNN_equi_unet_deep, self).__init__()
        self.k = k

        assert pooling == 'mean', "Only mean pooling is supported for EQCNN_equi_unet_deep"
        self.pool1 = mean_pool
        
        self.pool2 = mean_pool
        self.pool2_2 = mean_pool
        self.pool3 = mean_pool
        self.pool3_2 = mean_pool
        self.pool4 = mean_pool
        self.pool4_2 = mean_pool
        
        self.pool5 = mean_pool

        self.pool6_2 = mean_pool
        self.pool6 = mean_pool
        self.pool7_2 = mean_pool
        self.pool7 = mean_pool
        self.pool8_2 = mean_pool
        self.pool8 = mean_pool
        


        # Encoder
        self.conv1 = VNLinearLeakyReLU(2, 64//3)

        # Encoder 1-1, sampling
        self.downsample1 = TransitionDown(64//3, 64//3, stride=2, nsample=16)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3)

        # Encoder 1-2, no_sampling
        self.conv2_2 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Encoder 2-1, sampling
        self.downsample2 = TransitionDown(128//3, 128//3, stride=2, nsample=16)
        self.conv3 = VNLinearLeakyReLU(128//3*2, 256//3)

        # Encoder 2-2, no_sampling
        self.conv3_2 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Encoder 3-1, sampling
        self.downsample3 = TransitionDown(256//3, 256//3, stride=2, nsample=16)
        self.conv4 = VNLinearLeakyReLU(256//3*2, 512//3)

        # Encoder 3-2, no_sampling
        self.conv4_2 = VNLinearLeakyReLU(512//3*2, 512//3)

        # Mid
        self.conv5 = VNLinearLeakyReLU(512//3*2, 1024//3)
    
        # Decoder 1-1, no_upsampling
        self.conv6_2 = VNLinearLeakyReLU(1024//3*2, 512//3)

        # Decoder 1-2, upsampling
        self.upsample1 = TransitionUp(512//3, 256//3)
        self.conv6 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Decoder 2-1, no_upsampling
        self.conv7_2 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Decoder 2-2, upsampling
        self.upsample2 = TransitionUp(256//3, 128//3)
        self.conv7 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Decoder 3-1, no_upsampling
        self.conv8_2 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Decoder 3-2, upsampling
        self.upsample3 = TransitionUp(128//3, 64//3)
        self.conv8 = VNLinearLeakyReLU(64//3*2, 64//3)

        # Proj
        self.conv9 = VNLinearLeakyReLU(64//3, feat_dim//3, dim=4, share_nonlinearity=True)


    
    def forward(self, x):
        """EQCNN_equi_unet

        Args:
            x (torch.Tensor): (batch_size, num_points, 3)

        Returns:
            equi_feat (torch.Tensor): (batch_size, feat_dim//3, 3, num_points)
        """

        x = x.transpose(2, 1) # (batch_size, 3, num_points)


        p1 = x.transpose(1,2).squeeze(0) # (num_points, 3)
        x1 = x.unsqueeze(1) # (batch_size, 1, 3, num_points)
        o1 = torch.Tensor([p1.size(0)]).to(torch.int32).cuda() # (1,) which shows the number of points

        
        ### ENCODER
        x1 = get_graph_feature(x1, k=self.k) # (b, 2c, 3, n, k) 
        x1 = self.conv1(x1) # (b, 2c, 3, n, k)  -> (b, c', 3, n, k)
        x1 = self.pool1(x1) # (b, c', 3, n, k) -> (b, c', 3, n)


        ### ENCODER 1
        # Encoder 1-1, sampling
        p2, x2, o2 = self.downsample1(p1, x1.squeeze(0)) # (sampled_points, 3), (batch, channel, 3, sampled_points), (1,)
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2(x2)
        x2 = self.pool2(x2) # (1, 128//3, 3, N/2)
        
        # Encoder 1-2, no_sampling
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2_2(x2)
        x2 = self.pool2_2(x2) # (1, 128//3, 3, N/2)

        ### ENCODER 2
        # Encoder 2-1, sampling
        p3, x3, o3 = self.downsample2(p2, x2.squeeze(0))
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3(x3)
        x3 = self.pool3(x3) # (1, 256//3, 3, N/4)
        
        # Encoder 2-2, no_sampling
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3_2(x3)
        x3 = self.pool3_2(x3) # (1, 256//3, 3, N/4)

        ### ENCODER 3
        # Encoder 3-1, sampling
        p4, x4, o4 = self.downsample3(p3, x3.squeeze(0))
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4(x4)
        x4 = self.pool4(x4) # (1, 512//3, 3, N/8)
        
        # Encoder 3-2, no_sampling
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4_2(x4)
        x4 = self.pool4_2(x4) # (1, 512//3, 3, N/8)


        ### MID
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv5(x4)
        x4 = self.pool5(x4) # (1, 512//3, 3, N/8)


        ### DECODER 1
        # Decoder 1-1, no_upsampling
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv6_2(x4)
        x4 = self.pool6_2(x4) # (1, 512//3, 3, N/8)
        
        # Decoder 1-2, upsampling
        x5 = self.upsample1((p3, x3, o3), (p4, x4, o4)) 
        x5 = get_graph_feature(x5.contiguous(), k=self.k)
        x5 = self.conv6(x5)
        x5 = self.pool6(x5) # (1, 256//3, 3, N/4)

        ### DECODER 2
        # Decoder 2-1, no_upsampling
        x5 = get_graph_feature(x5, k=self.k)
        x5 = self.conv7_2(x5)
        x5 = self.pool7_2(x5) # (1, 256//3, 3, N/4)
        
        # Decoder 2-2, upsampling
        x6 = self.upsample2((p2, x2, o2), (p3, x5, o3))
        x6 = get_graph_feature(x6.contiguous(), k=self.k)
        x6 = self.conv7(x6)
        x6 = self.pool7(x6) # (1, 128//3, 3, N/2)

        ### DECODER 3
        # Decoder 3-1, no_upsampling
        x6 = get_graph_feature(x6, k=self.k)
        x6 = self.conv8_2(x6)
        x6 = self.pool8_2(x6) # (1, 128//3, 3, N/2)
        
        # Decoder 3-2, upsampling
        x7 = self.upsample3((p1, x1, o1), (p2, x6, o2))
        x7 = get_graph_feature(x7.contiguous(), k=self.k)
        x7 = self.conv8(x7)
        x7 = self.pool8(x7) # (1, 64//3, 3, N)


        equi_feat = self.conv9(x7)

        return equi_feat





class EQCNN_equi_unet_deep_v3(nn.Module): 

    def __init__(self, feat_dim, pooling='mean', k=20):
        super(EQCNN_equi_unet_deep_v3, self).__init__()
        self.k = k

        assert pooling == 'mean', "Only mean pooling is supported for EQCNN_equi_unet_deep"
        self.pool1 = mean_pool
        
        self.pool2 = mean_pool
        self.pool2_2 = mean_pool
        self.pool3 = mean_pool
        self.pool3_2 = mean_pool
        self.pool4 = mean_pool
        self.pool4_2 = mean_pool
        
        self.pool5 = mean_pool

        self.pool6_2 = mean_pool
        self.pool6 = mean_pool
        self.pool7_2 = mean_pool
        self.pool7 = mean_pool
        self.pool8_2 = mean_pool
        self.pool8 = mean_pool
        


        # Encoder
        self.conv1 = VNLinearLeakyReLU(2, 64//3)

        # Encoder 1-1, sampling
        self.downsample1 = TransitionDown(64//3, 64//3, stride=2, nsample=16)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3)

        # Encoder 1-2, no_sampling
        self.conv2_2 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Encoder 2-1, sampling
        self.downsample2 = TransitionDown(128//3, 128//3, stride=2, nsample=16)
        self.conv3 = VNLinearLeakyReLU(128//3*2, 256//3)

        # Encoder 2-2, no_sampling
        self.conv3_2 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Encoder 3-1, sampling
        self.downsample3 = TransitionDown(256//3, 256//3, stride=2, nsample=16)
        self.conv4 = VNLinearLeakyReLU(256//3*2, 512//3)

        # Encoder 3-2, no_sampling
        self.conv4_2 = VNLinearLeakyReLU(512//3*2, 1024//3)

        # Mid
        self.conv5 = VNLinearLeakyReLU(1024//3*2, 1024//3)
    
        # Decoder 1-1, no_upsampling
        self.conv6_2 = VNLinearLeakyReLU(1024//3*2, 512//3)

        # Decoder 1-2, upsampling
        self.upsample1 = TransitionUp(512//3, 256//3)
        self.conv6 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Decoder 2-1, no_upsampling
        self.conv7_2 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Decoder 2-2, upsampling
        self.upsample2 = TransitionUp(256//3, 128//3)
        self.conv7 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Decoder 3-1, no_upsampling
        self.conv8_2 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Decoder 3-2, upsampling
        self.upsample3 = TransitionUp(128//3, 64//3)
        self.conv8 = VNLinearLeakyReLU(64//3*2, 64//3)

        # Proj
        self.conv9 = VNLinearLeakyReLU(64//3, feat_dim//3, dim=4, share_nonlinearity=True)


    
    def forward(self, x):
        """EQCNN_equi_unet

        Args:
            x (torch.Tensor): (batch_size, num_points, 3)

        Returns:
            equi_feat (torch.Tensor): (batch_size, feat_dim//3, 3, num_points)
        """

        x = x.transpose(2, 1) # (batch_size, 3, num_points)


        p1 = x.transpose(1,2).squeeze(0) # (num_points, 3)
        x1 = x.unsqueeze(1) # (batch_size, 1, 3, num_points)
        o1 = torch.Tensor([p1.size(0)]).to(torch.int32).cuda() # (1,) which shows the number of points

        
        ### ENCODER
        x1 = get_graph_feature(x1, k=self.k) # (b, 2c, 3, n, k) 
        x1 = self.conv1(x1) # (b, 2c, 3, n, k)  -> (b, c', 3, n, k)
        x1 = self.pool1(x1) # (b, c', 3, n, k) -> (b, c', 3, n)


        ### ENCODER 1
        # Encoder 1-1, sampling
        p2, x2, o2 = self.downsample1(p1, x1.squeeze(0)) # (sampled_points, 3), (batch, channel, 3, sampled_points), (1,)
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2(x2)
        x2 = self.pool2(x2) # (1, 128//3, 3, N/2)
        
        # Encoder 1-2, no_sampling
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2_2(x2)
        x2 = self.pool2_2(x2) # (1, 128//3, 3, N/2)

        ### ENCODER 2
        # Encoder 2-1, sampling
        p3, x3, o3 = self.downsample2(p2, x2.squeeze(0))
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3(x3)
        x3 = self.pool3(x3) # (1, 256//3, 3, N/4)
        
        # Encoder 2-2, no_sampling
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3_2(x3)
        x3 = self.pool3_2(x3) # (1, 256//3, 3, N/4)

        ### ENCODER 3
        # Encoder 3-1, sampling
        p4, x4, o4 = self.downsample3(p3, x3.squeeze(0))
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4(x4)
        x4 = self.pool4(x4) # (1, 512//3, 3, N/8)
        
        # Encoder 3-2, no_sampling
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4_2(x4)
        x4 = self.pool4_2(x4) # (1, 512//3, 3, N/8)


        ### MID
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv5(x4)
        x4 = self.pool5(x4) # (1, 512//3, 3, N/8)


        ### DECODER 1
        # Decoder 1-1, no_upsampling
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv6_2(x4)
        x4 = self.pool6_2(x4) # (1, 512//3, 3, N/8)
        
        # Decoder 1-2, upsampling
        x5 = self.upsample1((p3, x3, o3), (p4, x4, o4)) 
        x5 = get_graph_feature(x5.contiguous(), k=self.k)
        x5 = self.conv6(x5)
        x5 = self.pool6(x5) # (1, 256//3, 3, N/4)

        ### DECODER 2
        # Decoder 2-1, no_upsampling
        x5 = get_graph_feature(x5, k=self.k)
        x5 = self.conv7_2(x5)
        x5 = self.pool7_2(x5) # (1, 256//3, 3, N/4)
        
        # Decoder 2-2, upsampling
        x6 = self.upsample2((p2, x2, o2), (p3, x5, o3))
        x6 = get_graph_feature(x6.contiguous(), k=self.k)
        x6 = self.conv7(x6)
        x6 = self.pool7(x6) # (1, 128//3, 3, N/2)

        ### DECODER 3
        # Decoder 3-1, no_upsampling
        x6 = get_graph_feature(x6, k=self.k)
        x6 = self.conv8_2(x6)
        x6 = self.pool8_2(x6) # (1, 128//3, 3, N/2)
        
        # Decoder 3-2, upsampling
        x7 = self.upsample3((p1, x1, o1), (p2, x6, o2))
        x7 = get_graph_feature(x7.contiguous(), k=self.k)
        x7 = self.conv8(x7)
        x7 = self.pool8(x7) # (1, 64//3, 3, N)


        equi_feat = self.conv9(x7)

        return equi_feat




class EQCNN_equi_unet_deep_v4(nn.Module): 

    def __init__(self, feat_dim, pooling='mean', k=20):
        super(EQCNN_equi_unet_deep_v4, self).__init__()
        self.k = k

        assert pooling == 'mean', "Only mean pooling is supported for EQCNN_equi_unet_deep"
        self.pool1 = mean_pool
        
        self.pool2 = mean_pool
        self.pool2_2 = mean_pool
        self.pool3 = mean_pool
        self.pool3_2 = mean_pool
        self.pool4 = mean_pool
        self.pool4_2 = mean_pool
        
        self.pool5 = mean_pool

        self.pool6_2 = mean_pool
        self.pool6 = mean_pool
        self.pool7_2 = mean_pool
        self.pool7 = mean_pool
        self.pool8_2 = mean_pool
        self.pool8 = mean_pool
        


        # Encoder
        self.conv1 = VNLinearLeakyReLU(2, 64//3)

        # Encoder 1-1, sampling
        self.downsample1 = TransitionDown(64//3, 64//3, stride=2, nsample=16)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3)

        # Encoder 1-2, no_sampling
        self.conv2_2 = VNLinearLeakyReLU(128//3*2, 256//3)

        # Encoder 2-1, sampling
        self.downsample2 = TransitionDown(256//3, 256//3, stride=2, nsample=16)
        self.conv3 = VNLinearLeakyReLU(256//3*2, 512//3)

        # Encoder 2-2, no_sampling
        self.conv3_2 = VNLinearLeakyReLU(512//3*2, 512//3)

        # Encoder 3-1, sampling
        self.downsample3 = TransitionDown(512//3, 512//3, stride=2, nsample=16)
        self.conv4 = VNLinearLeakyReLU(512//3*2, 512//3)

        # Encoder 3-2, no_sampling
        self.conv4_2 = VNLinearLeakyReLU(512//3*2, 1024//3)

        # Mid
        self.conv5 = VNLinearLeakyReLU(1024//3*2, 1024//3)
    
        # Decoder 1-1, no_upsampling
        self.conv6_2 = VNLinearLeakyReLU(1024//3*2, 512//3)

        # Decoder 1-2, upsampling
        self.upsample1 = TransitionUp(512//3, 512//3)
        self.conv6 = VNLinearLeakyReLU(512//3*2, 512//3)

        # Decoder 2-1, no_upsampling
        self.conv7_2 = VNLinearLeakyReLU(512//3*2, 512//3)

        # Decoder 2-2, upsampling
        self.upsample2 = TransitionUp(512//3, 256//3)
        self.conv7 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Decoder 3-1, no_upsampling
        self.conv8_2 = VNLinearLeakyReLU(256//3*2, 128//3)

        # Decoder 3-2, upsampling
        self.upsample3 = TransitionUp(128//3, 64//3)
        self.conv8 = VNLinearLeakyReLU(64//3*2, 64//3)

        # Proj
        self.conv9 = VNLinearLeakyReLU(64//3, feat_dim//3, dim=4, share_nonlinearity=True)


    
    def forward(self, x):
        """EQCNN_equi_unet

        Args:
            x (torch.Tensor): (batch_size, num_points, 3)

        Returns:
            equi_feat (torch.Tensor): (batch_size, feat_dim//3, 3, num_points)
        """

        x = x.transpose(2, 1) # (batch_size, 3, num_points)


        p1 = x.transpose(1,2).squeeze(0) # (num_points, 3)
        x1 = x.unsqueeze(1) # (batch_size, 1, 3, num_points)
        o1 = torch.Tensor([p1.size(0)]).to(torch.int32).cuda() # (1,) which shows the number of points

        
        ### ENCODER
        x1 = get_graph_feature(x1, k=self.k) # (b, 2c, 3, n, k) 
        x1 = self.conv1(x1) # (b, 2c, 3, n, k)  -> (b, c', 3, n, k)
        x1 = self.pool1(x1) # (b, c', 3, n, k) -> (b, c', 3, n)


        ### ENCODER 1
        # Encoder 1-1, sampling
        p2, x2, o2 = self.downsample1(p1, x1.squeeze(0)) # (sampled_points, 3), (batch, channel, 3, sampled_points), (1,)
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2(x2)
        x2 = self.pool2(x2) # (1, 128//3, 3, N/2)
        
        # Encoder 1-2, no_sampling
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2_2(x2)
        x2 = self.pool2_2(x2) # (1, 256//3, 3, N/2)

        ### ENCODER 2
        # Encoder 2-1, sampling
        p3, x3, o3 = self.downsample2(p2, x2.squeeze(0))
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3(x3)
        x3 = self.pool3(x3) # (1, 512//3, 3, N/4)
        
        # Encoder 2-2, no_sampling
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3_2(x3)
        x3 = self.pool3_2(x3) # (1, 512//3, 3, N/4)

        ### ENCODER 3
        # Encoder 3-1, sampling
        p4, x4, o4 = self.downsample3(p3, x3.squeeze(0))
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4(x4)
        x4 = self.pool4(x4) # (1, 512//3, 3, N/8)
        
        # Encoder 3-2, no_sampling
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4_2(x4)
        x4 = self.pool4_2(x4) # (1, 1024//3, 3, N/8)


        ### MID
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv5(x4)
        x4 = self.pool5(x4) # (1, 1024//3, 3, N/8)


        ### DECODER 1
        # Decoder 1-1, no_upsampling
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv6_2(x4)
        x4 = self.pool6_2(x4) # (1, 512//3, 3, N/8)
        
        # Decoder 1-2, upsampling
        x5 = self.upsample1((p3, x3, o3), (p4, x4, o4)) 
        x5 = get_graph_feature(x5.contiguous(), k=self.k)
        x5 = self.conv6(x5)
        x5 = self.pool6(x5) # (1, 512//3, 3, N/4)

        ### DECODER 2
        # Decoder 2-1, no_upsampling
        x5 = get_graph_feature(x5, k=self.k)
        x5 = self.conv7_2(x5)
        x5 = self.pool7_2(x5) # (1, 512//3, 3, N/4)
        
        # Decoder 2-2, upsampling
        x6 = self.upsample2((p2, x2, o2), (p3, x5, o3))
        x6 = get_graph_feature(x6.contiguous(), k=self.k)
        x6 = self.conv7(x6)
        x6 = self.pool7(x6) # (1, 256//3, 3, N/2)

        ### DECODER 3
        # Decoder 3-1, no_upsampling
        x6 = get_graph_feature(x6, k=self.k)
        x6 = self.conv8_2(x6)
        x6 = self.pool8_2(x6) # (1, 128//3, 3, N/2)
        
        # Decoder 3-2, upsampling
        x7 = self.upsample3((p1, x1, o1), (p2, x6, o2))
        x7 = get_graph_feature(x7.contiguous(), k=self.k)
        x7 = self.conv8(x7)
        x7 = self.pool8(x7) # (1, 64//3, 3, N)


        equi_feat = self.conv9(x7)

        return equi_feat



class EQCNN_equi_unet_deep_v2(nn.Module): 

    def __init__(self, feat_dim, pooling='mean', k=20):
        super(EQCNN_equi_unet_deep_v2, self).__init__()
        self.k = k

        assert pooling == 'mean', "Only mean pooling is supported for EQCNN_equi_unet_deep"
        self.pool1 = mean_pool
        
        self.pool2 = mean_pool
        self.pool2_2 = mean_pool
        self.pool2_3 = mean_pool
        self.pool3 = mean_pool
        self.pool3_2 = mean_pool
        self.pool3_3 = mean_pool
        self.pool4 = mean_pool
        self.pool4_2 = mean_pool
        self.pool4_3 = mean_pool
        
        self.pool5 = mean_pool

        self.pool6 = mean_pool
        self.pool6_2 = mean_pool
        self.pool6_3 = mean_pool
        self.pool7 = mean_pool
        self.pool7_2 = mean_pool
        self.pool7_3 = mean_pool
        self.pool8 = mean_pool
        self.pool8_2 = mean_pool
        self.pool8_3 = mean_pool

        # Encoder
        self.conv1 = VNLinearLeakyReLU(2, 64//3)

        # Encoder 1-1, sampling
        self.downsample1 = TransitionDown(64//3, 64//3, stride=2, nsample=16)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3)

        # Encoder 1-2 + 1-3, no_sampling
        self.conv2_2 = VNLinearLeakyReLU(128//3*2, 128//3)
        self.conv2_3 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Encoder 2-1, sampling
        self.downsample2 = TransitionDown(128//3, 128//3, stride=2, nsample=16)
        self.conv3 = VNLinearLeakyReLU(128//3*2, 256//3)

        # Encoder 2-2 + 2-3, no_sampling
        self.conv3_2 = VNLinearLeakyReLU(256//3*2, 256//3)
        self.conv3_3 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Encoder 3-1, sampling
        self.downsample3 = TransitionDown(256//3, 256//3, stride=2, nsample=16)
        self.conv4 = VNLinearLeakyReLU(256//3*2, 512//3)

        # Encoder 3-2 + 3-3, no_sampling
        self.conv4_2 = VNLinearLeakyReLU(512//3*2, 512//3)
        self.conv4_3 = VNLinearLeakyReLU(512//3*2, 512//3)

        # Mid
        self.conv5 = VNLinearLeakyReLU(512//3*2, 512//3)
    
        # Decoder 1-1, upsampling
        self.upsample1 = TransitionUp(512//3, 256//3)
        self.conv6 = VNLinearLeakyReLU(256//3*2, 256//3)
        
        # Decoder 1-2 + 1-3, no_upsampling
        self.conv6_2 = VNLinearLeakyReLU(256//3*2, 256//3)
        self.conv6_3 = VNLinearLeakyReLU(256//3*2, 256//3)

        # Decoder 2-1, upsampling
        self.upsample2 = TransitionUp(256//3, 128//3)
        self.conv7 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Decoder 2-2 + 2-3, no_upsampling
        self.conv7_2 = VNLinearLeakyReLU(128//3*2, 128//3)
        self.conv7_3 = VNLinearLeakyReLU(128//3*2, 128//3)

        # Decoder 3-1, upsampling
        self.upsample3 = TransitionUp(128//3, 64//3)
        self.conv8 = VNLinearLeakyReLU(64//3*2, 64//3)

        # Decoder 3-2 + 3-3, no_upsampling
        self.conv8_2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv8_3 = VNLinearLeakyReLU(64//3*2, 64//3)

        # Proj
        self.conv9 = VNLinearLeakyReLU(64//3, feat_dim//3, dim=4, share_nonlinearity=True)


    
    def forward(self, x):
        """EQCNN_equi_unet

        Args:
            x (torch.Tensor): (batch_size, num_points, 3)

        Returns:
            equi_feat (torch.Tensor): (batch_size, feat_dim//3, 3, num_points)
        """

        x = x.transpose(2, 1) # (batch_size, 3, num_points)


        p1 = x.transpose(1,2).squeeze(0) # (num_points, 3)
        x1 = x.unsqueeze(1) # (batch_size, 1, 3, num_points)
        o1 = torch.Tensor([p1.size(0)]).to(torch.int32).cuda() # (1,) which shows the number of points

        
        ### ENCODER
        x1 = get_graph_feature(x1, k=self.k) # (b, 2c, 3, n, k) 
        x1 = self.conv1(x1) # (b, 2c, 3, n, k)  -> (b, c', 3, n, k)
        x1 = self.pool1(x1) # (b, c', 3, n, k) -> (b, c', 3, n)


        ### ENCODER 1
        # Encoder 1-1, sampling
        p2, x2, o2 = self.downsample1(p1, x1.squeeze(0)) # (sampled_points, 3), (batch, channel, 3, sampled_points), (1,)
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2(x2)
        x2 = self.pool2(x2) # (1, 128//3, 3, N/2)
        
        # Encoder 1-2, no_sampling
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2_2(x2)
        x2 = self.pool2_2(x2) # (1, 128//3, 3, N/2)

        # Encoder 1-3, no_sampling
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2_3(x2)
        x2 = self.pool2_3(x2) # (1, 128//3, 3, N/2)

        ### ENCODER 2
        # Encoder 2-1, sampling
        p3, x3, o3 = self.downsample2(p2, x2.squeeze(0))
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3(x3)
        x3 = self.pool3(x3) # (1, 256//3, 3, N/4)
        
        # Encoder 2-2, no_sampling
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3_2(x3)
        x3 = self.pool3_2(x3) # (1, 256//3, 3, N/4)

        # Encoder 2-3, no_sampling
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3_3(x3)
        x3 = self.pool3_3(x3) # (1, 256//3, 3, N/4)

        ### ENCODER 3
        # Encoder 3-1, sampling
        p4, x4, o4 = self.downsample3(p3, x3.squeeze(0))
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4(x4)
        x4 = self.pool4(x4) # (1, 512//3, 3, N/8)
        
        # Encoder 3-2, no_sampling
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4_2(x4)
        x4 = self.pool4_2(x4) # (1, 512//3, 3, N/8)

        # Encoder 3-3, no_sampling
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4_3(x4)
        x4 = self.pool4_3(x4) # (1, 512//3, 3, N/8)

        ### MID
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv5(x4)
        x4 = self.pool5(x4) # (1, 512//3, 3, N/8)

        ### DECODER 1
        # Decoder 1-1, upsampling
        x5 = self.upsample1((p3, x3, o3), (p4, x4, o4)) 
        x5 = get_graph_feature(x5.contiguous(), k=self.k)
        x5 = self.conv6(x5)
        x5 = self.pool6(x5) # (1, 256//3, 3, N/4)

        # Decoder 1-2, no_upsampling
        x5 = get_graph_feature(x5, k=self.k)
        x5 = self.conv6_2(x5)
        x5 = self.pool6_2(x5) # (1, 256//3, 3, N/4)

        # Decoder 1-3, no_upsampling
        x5 = get_graph_feature(x5, k=self.k)
        x5 = self.conv6_3(x5)
        x5 = self.pool6_3(x5) # (1, 256//3, 3, N/4)


        ### DECODER 2
        # Decoder 2-1, upsampling
        x6 = self.upsample2((p2, x2, o2), (p3, x5, o3))
        x6 = get_graph_feature(x6.contiguous(), k=self.k)
        x6 = self.conv7(x6)
        x6 = self.pool7(x6) # (1, 128//3, 3, N/2)

        # Decoder 2-2, no_upsampling
        x6 = get_graph_feature(x6, k=self.k)
        x6 = self.conv7_2(x6)
        x6 = self.pool7_2(x6) # (1, 256//3, 3, N/4)

        # Decoder 2-3, no_upsampling
        x6 = get_graph_feature(x6, k=self.k)
        x6 = self.conv7_3(x6)
        x6 = self.pool7_3(x6) # (1, 256//3, 3, N/4)

        ### DECODER 3
        # Decoder 3-1, upsampling
        x7 = self.upsample3((p1, x1, o1), (p2, x6, o2))
        x7 = get_graph_feature(x7.contiguous(), k=self.k)
        x7 = self.conv8(x7)
        x7 = self.pool8(x7) # (1, 64//3, 3, N)

        # Decoder 3-2, no_upsampling
        x7 = get_graph_feature(x7, k=self.k)
        x7 = self.conv8_2(x7)
        x7 = self.pool8_2(x7) # (1, 64//3, 3, N)

        # Decoder 3-3, no_upsampling
        x7 = get_graph_feature(x7, k=self.k)
        x7 = self.conv8_3(x7)
        x7 = self.pool8_3(x7) # (1, 64//3, 3, N)

        equi_feat = self.conv9(x7)

        return equi_feat


class EQCNN_equi(nn.Module):

    def __init__(self, feat_dim, pooling='mean', k=20):
        super(EQCNN_equi, self).__init__()
        self.k = k

        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, feat_dim//3, dim=4, share_nonlinearity=True)

    def forward(self, x):
        x = x.transpose(2, 1) # (batch_size, 3, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (batch_size, 1, 3, num_points)

        x = get_graph_feature(x, k=self.k) # (1, 2, 3, num_points, k)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        equi_feat = self.conv5(x) # (batch_size, feat_dim//3, num_points)
        
        return equi_feat