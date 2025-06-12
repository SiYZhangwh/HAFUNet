# import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import timm

import logging

from scipy import ndimage

from model.decoders import HADer
from model.maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from model.maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from model.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from model.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out
from timm.models.layers import trunc_normal_
import math



logger = logging.getLogger(__name__)



def load_pretrained_weights(img_size, model_scale, pre_load):
    if(model_scale=='tiny'):
        if img_size==224:
            backbone = maxvit_tiny_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './model/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
            state_dict = torch.load('./model/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
        elif(img_size==256):
            backbone = maxvit_rmlp_tiny_rw_256_4out()
            print('Loading:', './model/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
            state_dict = torch.load('./model/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")

    elif(model_scale=='small'):
        if img_size==224:
            backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './model/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
            state_dict = torch.load('./model/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', './model/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('./model/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    else:
        sys.exit(model_scale+" is not a valid model scale! Currently supported model scales are 'tiny' and 'small'.")
    if pre_load == True:    
        backbone.load_state_dict(state_dict, strict=False)
        print('Pretrain weights loaded.')
    
    return backbone





class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights



class SpatialWeights(nn.Module):
    def __init__(self, dim=48,ffn_expansion_factor=0.25, bias=False):
        super(SpatialWeights, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(2*dim, hidden_features*3, kernel_size=(1,1,1), bias=bias)

        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, 2*dim, kernel_size=(1,1,1), bias=bias)



    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1,x2,x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        x = F.gelu(x1)*(x2+x3)
        x = x.unsqueeze(2)
        x = self.project_out(x)
        x = x.squeeze(2).chunk(2, dim=1)
        return x
    
#Cross-modal Feature Interaction Module
class CFIM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(CFIM, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim,ffn_expansion_factor=0.5, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2  + self.lambda_s * spatial_weights[1] 
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1  + self.lambda_s * spatial_weights[0] 
        return out_x1, out_x2


class HAFUNet(nn.Module):
    def __init__(self, img_size_s1=(224,224), img_size_s2=(224,224), model_scale='tiny', pre_load = True):
        super(HAFUNet, self).__init__()
        

        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale
        
        # conv block to convert channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale, pre_load= pre_load)
        self.backbone2 = load_pretrained_weights(self.img_size_s2[0], self.model_scale, pre_load= pre_load)
        
        if(self.model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(self.model_scale=='small'):
            self.channels = [768, 384, 192, 96]
     
        # decoder initialization
        self.decoder = HADer(channels=self.channels)

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], 1, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], 1, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], 1, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], 1, 1)


        self.CFIMs = nn.ModuleList([
                    CFIM(dim=self.channels[3], reduction=1),
                    CFIM(dim=self.channels[2], reduction=1),
                    CFIM(dim=self.channels[1], reduction=1),
                    CFIM(dim=self.channels[0], reduction=1)])


    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = self.conv_2(x2)
  
        # transformer backbone as encoder
        if(x1.shape[2]%14!=0):
            f1 = self.backbone1(F.interpolate(x1, size=self.img_size_s1, mode='bilinear'))
        else:
            f1 = self.backbone2(F.interpolate(x1, size=self.img_size_s1, mode='bilinear'))
               
        if (x1.shape[2] % 14 != 0):
            f2 = self.backbone2(F.interpolate(x2, size=self.img_size_s2, mode='bilinear'))
        else:
            f2 = self.backbone1(F.interpolate(x2, size=self.img_size_s2, mode='bilinear'))


        f1_0, f2_0 = self.CFIMs[0](f1[0], f2[0])
        f1_1, f2_1 = self.CFIMs[1](f1[1], f2[1])
        f1_2, f2_2 = self.CFIMs[2](f1[2], f2[2])
        f1_3, f2_3 = self.CFIMs[3](f1[3], f2[3])

        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1_3, [f1_2, f1_1, f1_0])

        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)


        p11 = F.interpolate(p11, scale_factor=32, mode='bilinear')
        p12 = F.interpolate(p12, scale_factor=16, mode='bilinear')
        p13 = F.interpolate(p13, scale_factor=8, mode='bilinear')
        p14 = F.interpolate(p14, scale_factor=4, mode='bilinear')

        x21_o, x22_o, x23_o, x24_o = self.decoder(f2_3, [f2_2, f2_1, f2_0])

        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)

        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode='bilinear')
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode='bilinear')
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode='bilinear')
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode='bilinear')

        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        P_ALL = p1 + p2 + p3 + p4
        
        return P_ALL

                        

if __name__ == '__main__':
    model = HAFUNet(img_size_s1=(224,224), img_size_s2=(224,224), model_scale='tiny', pre_load =False).cuda()
    rgb = torch.randn(1, 1, 224, 224).cuda()
    event = torch.randn(1, 5, 224, 224).cuda()

    p = model(rgb,  event)
    print(p.shape)

