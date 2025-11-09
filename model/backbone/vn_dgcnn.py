import torch
import torch.nn as nn
from model.backbone.vn_layers import knn, get_graph_feature, mean_pool
from model.backbone.vn_layers import VNLinearLeakyReLU, VNMaxPool

# from lib.pointops.functions import pointops
from pointcept_libs.pointops2.functions import pointops2 as pointops


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=4):
        super().__init__()
        assert stride > 1, f"stride must be greater than 1, but got {stride}"

        self.stride = stride
        self.nsample = nsample
        self.mlp = VNLinearLeakyReLU(in_planes, out_planes)

        
    def forward(self, p, x):
        """TransitionDown
        This module assume batch size is 1.

        Args:
            p (torch.Tensor): (num_points, 3), which is resposible for point coordinates
            x (torch.Tensor): (channel, 3, num_points), which is resposible for point features

        Returns:
            n_p (torch.Tensor): (num_points, 3)
            x (torch.Tensor): (batch, channel, 3, num_points) where batch is 1
            n_o (torch.Tensor): (1, )
        """
        # original offset
        o = torch.Tensor([p.size(0)]).to(torch.int32).cuda()

        # new offset, which is the number of points after stride
        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        n_o = torch.cuda.IntTensor(n_o)

        # FPS
        idx = pointops.furthestsampling(p, o, n_o)  # (m)
        n_p = p[idx.long(), :]  # (m, 3)

        # kNN-MLP
        reshaped_x = x.reshape(-1, x.shape[-1]).transpose(1, 0).contiguous() # (channel, 3, points) -> (channel*3, points) -> (points, channel*3)
        x = pointops.queryandgroup(self.nsample, p, n_p, reshaped_x, None, o, n_o, use_xyz=False) # (sampled_points, nsample, channel*3)
        x = x.reshape(x.shape[0], x.shape[1], -1, 3) # (sampled_points, nsample, channel*3) -> (sampled_points, nsample, channel, 3)

        # (sampled_points, nsample, channel, 3) -> (channel, 3, sampled_points, nsample) -> (1, channel, 3, sampled_points, nsample) -> (1, c', 3, sampled_points, nsample)
        x = self.mlp(x.permute(2,3,0,1).unsqueeze(0))

        # Mean Pooling
        x = x.mean(dim=-1)  # (1, c, 3, m)

        return n_p, x, n_o


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.mlp1 = VNLinearLeakyReLU(out_planes, out_planes, dim=4)
        self.mlp2 = VNLinearLeakyReLU(in_planes, out_planes, dim=4)
        
    def forward(self, pxo1, pxo2):
        """TransitionUp
        This module assume batch size is 1.

        Args:
            pxo1 (tuple): (p1, x1, o1) where p1 is (n, 3), x1 is (b,c,3,n), o1 is (1,)
            pxo2 (tuple, optional): (p2, x2, o2) where p2 is (m, 3), x2 is (b,c',3,m), o2 is (1,)

        Returns:
            x (torch.Tensor): (n, c)
        """
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2

        # self.mlp2: (b,c',3,m) -> (b,o,3,m)
        # channel_aligned_x2: (1,o,3,m) -> (o,3,m) -> (o*3,m) -> (m, o*3)
        channel_aligned_x2 = self.mlp2(x2).squeeze(0).reshape(-1, x2.shape[-1]).transpose(1, 0).contiguous()
        
        # p2: (m, 3), p1: (n, 3), channel_aligned_x2: (m, o*3), o2: (1,), o1: (1,)
        # pointops.interpolation: Interpolate features to enlarge p2 to p1
        # Locations of new points will be p1, and features will be interpolated from p2.
        # From new locations, find the closest points from p2, and interpolate features from them.
        # Also, we need to align channel size, so use self.mlp2 to align channel size.     
        # pointops.interpolation: (m, o*3) -> (n, o*3)
        interpolated_x2 = pointops.interpolation(p2, p1, channel_aligned_x2, o2, o1)

        # (n, o*3) -> (o*3, n) -> (o,3,n) -> (1,o,3,n) 
        interpolated_x2 = interpolated_x2.transpose(0,1).reshape(-1, 3, interpolated_x2.shape[0]).unsqueeze(0)

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
        self.conv7 = VNLinearLeakyReLU(127//3*2, 128//3)

        self.upsample3 = TransitionUp(128//3, 64//3)
        self.conv8 = VNLinearLeakyReLU(64//3*2, 64//3)

        # Proj
        self.conv9 = VNLinearLeakyReLU(64//3, feat_dim//3, dim=4, share_nonlinearity=True)

        # Random rotation for equivariance checking
        # rotation_matrix = torch.tensor([[0.26726124, -0.57735027,  0.77151675],
        #           [0.53452248, -0.57735027, -0.6172134],
        #           [0.80178373,  0.57735027,  0.15430335]], dtype=torch.float64)
        # self.R = rotation_matrix.unsqueeze(0)
    
    def forward(self, x):
        """EQCNN_equi_unet

        Args:
            x (torch.Tensor): (batch_size, num_points, 3)

        Returns:
            equi_feat (torch.Tensor): (batch_size, feat_dim//3, 3, num_points)
        """

        x = x.transpose(2, 1) # (batch_size, 3, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        p1 = x.transpose(1,2).squeeze(0) # (num_points, 3)
        x1 = x.unsqueeze(1) # (batch_size, 1, 3, num_points)
        o1 = torch.Tensor([p1.size(0)]).to(torch.int32).cuda() # (1,) which shows the number of points

        
        ### CHECK EQUIV INIT ### 
        # R = self.R.cuda()
        # p1_R = torch.matmul(p1, R.squeeze(0))
        # x1_R = p1_R.unsqueeze(0).transpose(2,1).unsqueeze(1)
        # o1_R = torch.Tensor([p1_R.size(0)]).to(torch.int32).cuda()
        ### CHECK EQUIV INIT ### 
        
        ### ENCODER 1
        x1 = get_graph_feature(x1, k=self.k) # (b, 2c, 3, n, k) 
        x1 = self.conv1(x1) # (b, 2c, 3, n, k)  -> (b, c', 3, n, k)
        x1 = self.pool1(x1) # (b, c', 3, n, k) -> (b, c', 3, n)
        # print(p1.size(), x1.size())
        ### ENCODER 1

        ### CHECK EQUIVARIANCE ###
        # x1_R = get_graph_feature(x1_R, k=self.k) # torch.Size([1, 2, 3, 2556, 20])
        # x1_R = self.conv1(x1_R)
        # x1_R = self.pool1(x1_R) # (1, 21, 3, N)
        # print(torch.allclose(x1_R.permute(0,3,1,2) @ R.transpose(1,2), x1.permute(0,3,1,2), atol=1e-4))
        ### CHECK EQUIVARIANCE ###

        ### ENCODER 2
        p2, x2, o2 = self.downsample1(p1, x1.squeeze(0)) # (sampled_points, 3), (batch, channel, 3, sampled_points), (1,)
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2(x2)
        x2 = self.pool2(x2) # (1, 42, 3, N/2))

        ### CHECK EQUIVARIANCE ###
        # p2_R = torch.matmul(p2, R.squeeze(0))
        # p2_R, x2_R, o2_R = self.downsample1(p1_R, p2_R, x1_R.squeeze(0))
        # x2_R = get_graph_feature(x2_R, k=self.k)
        # x2_R = self.conv2(x2_R)
        # x2_R = self.pool2(x2_R) # (1, 21, 3, N/2)
        # print(torch.allclose(x2_R.permute(0,3,1,2) @ R.transpose(1,2), x2.permute(0,3,1,2), atol=1e-4))
        ### CHECK EQUIVARIANCE ###

        ### ENCODER 3
        p3, x3, o3 = self.downsample2(p2, x2.squeeze(0))
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3(x3)
        x3 = self.pool3(x3) # (1, 85, 3, N/4)
        ### ENCODER 3
        
        ### CHECK EQUIVARIANCE ###
        # p3_R = torch.matmul(p3, R.squeeze(0))
        # p3_R, x3_R, o3_R = self.downsample2(p2_R, p3_R, x2_R.squeeze(0))
        # x3_R = get_graph_feature(x3_R, k=self.k)
        # x3_R = self.conv3(x3_R)
        # x3_R = self.pool3(x3_R) # (1, 21, 3, N/2)
        # print(torch.allclose(x3_R.permute(0,3,1,2) @ R.transpose(1,2), x3.permute(0,3,1,2), atol=1e-3))
        ### CHECK EQUIVARIANCE ###

        ### ENCODER 4
        p4, x4, o4 = self.downsample3(p3, x3.squeeze(0))
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4(x4)
        x4 = self.pool4(x4) # (1, 170, 3, N/8)
        # print(p4.size(), x4.size())

        ### CHECK EQUIVARIANCE ###
        # p4_R = torch.matmul(p4, R.squeeze(0))
        # p4_R, x4_R, o4_R = self.downsample3(p3_R, p4_R, x3_R.squeeze(0))
        # x4_R = get_graph_feature(x4_R, k=self.k)
        # x4_R = self.conv4(x4_R)
        # x4_R = self.pool4(x4_R) # (1, 21, 3, N/2)
        # print(torch.allclose(x4_R.permute(0,3,1,2) @ R.transpose(1,2), x4.permute(0,3,1,2), atol=1e-3))
        ### CHECK EQUIVARIANCE ###

        ### MID
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv5(x4)
        x4 = self.pool5(x4) # (1, 170, 3, N/8)
        # print(p4.size(), x4.size())

        ### CHECK EQUIVARIANCE ###
        # x4_R = get_graph_feature(x4_R, k=self.k)
        # x4_R = self.conv5(x4_R)
        # x4_R = self.pool5(x4_R) # (1, 21, 3, N/2)
        # print(torch.allclose(x4_R.permute(0,3,1,2) @ R.transpose(1,2), x4.permute(0,3,1,2), atol=1e-3))
        ### CHECK EQUIVARIANCE ###

        ### DECODER 1
        x5 = self.upsample1((p3, x3, o3), (p4, x4, o4)) 
        x5 = get_graph_feature(x5.contiguous(), k=self.k)
        x5 = self.conv6(x5)
        x5 = self.pool6(x5) # (1, 85, 3, N/4)
        # print(x5.size())

        ### CHECK EQUIVARIANCE ###
        # x5_R = self.upsample1((p3_R, x3_R, o3_R), (p4_R, x4_R, o4_R)) 
        # x5_R = get_graph_feature(x5_R.contiguous(), k=self.k)
        # x5_R = self.conv6(x5_R)
        # x5_R = self.pool6(x5_R) # (1, 42, 3, N/4)
        # print(torch.allclose(x5_R.permute(0,3,1,2) @ R.transpose(1,2), x5.permute(0,3,1,2), atol=1e-3))
        ### CHECK EQUIVARIANCE ###

        ### DECODER 2
        x6 = self.upsample2((p2, x2, o2), (p3, x5, o3))
        x6 = get_graph_feature(x6.contiguous(), k=self.k)
        x6 = self.conv7(x6)
        x6 = self.pool7(x6) # (1, 42, 3, N/2)
        # print(x6.size())

        ### CHECK EQUIVARIANCE ###
        # x6_R = self.upsample2((p2_R, x2_R, o2_R), (p3_R, x5_R, o3_R))
        # x6_R = get_graph_feature(x6_R.contiguous(), k=self.k)
        # x6_R = self.conv7(x6_R)
        # x6_R = self.pool7(x6_R) # (1, 42, 3, N/4)
        # print(torch.allclose(x6_R.permute(0,3,1,2) @ R.transpose(1,2), x6.permute(0,3,1,2), atol=1e-3))
        ### CHECK EQUIVARIANCE ###

        ### DECODER 3
        x7 = self.upsample3((p1, x1, o1), (p2, x6, o2))
        x7 = get_graph_feature(x7.contiguous(), k=self.k)
        x7 = self.conv8(x7)
        x7 = self.pool8(x7) # (1, 21, 3, N)
        # print(x7.size())

        ### CHECK EQUIVARIANCE ###
        # x7_R = self.upsample3((p1_R, x1_R, o1_R), (p2_R, x6_R, o2_R))
        # x7_R = get_graph_feature(x7_R.contiguous(), k=self.k)
        # x7_R = self.conv8(x7_R)
        # x7_R = self.pool8(x7_R) # (1, 42, 3, N/4)
        # print(torch.allclose(x7_R.permute(0,3,1,2) @ R.transpose(1,2), x7.permute(0,3,1,2), atol=1e-3))
        ### CHECK EQUIVARIANCE ###

        equi_feat = self.conv9(x7)

        ### CHECK EQUIVARIANCE ###
        # equi_feat_R = self.conv9(x7_R)
        # print(torch.allclose(equi_feat_R.permute(0,3,1,2) @ R.transpose(1,2), equi_feat.permute(0,3,1,2), atol=1e-2))
        ### CHECK EQUIVARIANCE ###

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