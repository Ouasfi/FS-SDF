import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial

class ShapeNetPoints_sdf_encoder(nn.Module):

    def __init__(self, device = torch.device('cuda')):
        super(ShapeNetPoints_sdf_encoder, self).__init__()
        # 128**3 res input

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='zeros')
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')

        
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).to(device)
        #self.reshape = Reshape()
        #self.sampler = grid_sampler_.apply #partial(F.grid_sample, padding_mode='border', align_corners=True )#grid_sampler_.apply 
        self.sampler = partial(F.grid_sample, padding_mode='border', align_corners=True )#grid_sampler_.apply 
        #self.sampler =custom_grid_sample
        #self.sampler =sample_features
    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = self.sampler(x, p)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = self.sampler(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = self.sampler(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = self.sampler(net, p) # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = self.sampler(net, p)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = self.sampler(net, p)
        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        return features
class ShapeNetPoints_sdf_maml(nn.Module):
    def __init__(self, encoder , decoder):
        super().__init__()
        self.encoder =  encoder
        self.decoder = decoder
    def forward(self, p, x ):
        return self.decoder(self.encoder(p, x).unsqueeze(0)).squeeze(2)
    def forward_with_params(self, p, x, params):
        context_x = self.encoder(p, x )
        return self.decoder.forward_with_params(context_x, params)