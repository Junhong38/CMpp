import torch
import torch.nn as nn
from model.backbone.vn_layers import knn, get_graph_feature, mean_pool
from model.backbone.vn_layers import VNLinearLeakyReLU, VNStdFeature, VNMaxPool

class EQCNN_inv(nn.Module):

    def __init__(self, feat_dim, pooling='mean'):
        super(EQCNN_inv, self).__init__()
        self.k = 20

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

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, feat_dim//3//2, dim=4, share_nonlinearity=True)

        self.std_feature = VNStdFeature(feat_dim//3//2*2, dim=4, normalize_frame=False)

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
        x = self.conv5(x) # (batch_size, feat_dim//3//2, num_points)
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1) # (batch_size, feat_dim//3, 3, num_points)
        x, z0 = self.std_feature(x) # (batch_size, feat_dim//3, 3, num_points)
        inv_feat = x.view(batch_size, -1, num_points) # (batch_size, feat_dim, num_points)

        return inv_feat

class EQCNN_equi(nn.Module):

    def __init__(self, feat_dim, pooling='mean'):
        super(EQCNN_equi, self).__init__()
        self.k = 20

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