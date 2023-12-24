# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torchvision.utils
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import cv2

class ChannelAttention(nn.Module):
    def __init__(self,channle = None):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ratio = 16
        self.relu = nn.GELU()
        self.fc1 = nn.Conv2d(3*channle, 3*channle // self.ratio, 1)
        self.fc2 = nn.Conv2d(3*channle // self.ratio, 3*channle, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        out =  max_out + avg_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self,channle = None):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(channle, channle//2, 1, 1), nn.BatchNorm2d(channle//2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channle//2, 1, 1, 1), nn.BatchNorm2d(1),nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(channle, channle//2, 3, 1, 1), nn.BatchNorm2d(channle//2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(channle//2, 1, 3, 1, 1), nn.BatchNorm2d(1),nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(channle, channle//2, 5, 1, 2), nn.BatchNorm2d(channle//2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(channle//2, 1, 5, 1, 2), nn.BatchNorm2d(1),nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(nn.Conv2d(channle, channle//2, 7, 1, 3), nn.BatchNorm2d(channle//2), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv2d(channle//2, 1, 7, 1, 3), nn.BatchNorm2d(1),nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(nn.Conv2d(3, 1, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x):
        if x.shape[2] == 7:
            y1 = self.conv2(self.conv1(x))
            y2 = self.conv4(self.conv3(x))
            y3 = self.conv6(self.conv5(x))
        else:
            y1 = self.conv4(self.conv3(x))
            y2 = self.conv6(self.conv5(x))
            y3 = self.conv8(self.conv7(x))
        y = torch.cat([y1, y2, y3], dim=1)
        z = self.conv9(y)

        return z

#COAM
class AACM(nn.Module):
    def __init__(self,channle =None):
        super(AACM , self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channle, channle, 3, 1, 1), nn.BatchNorm2d(channle), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channle, channle, 3, 1, 1), nn.BatchNorm2d(channle), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(3*channle, channle, 3, 1, 1), nn.BatchNorm2d(channle), nn.ReLU(inplace=True))
        self.att_s1  = SpatialAttention(channle)
        self.att_s2  = SpatialAttention(channle)
        self.att_c1  = ChannelAttention(channle)
    def forward(self, main, minor):
        main_1 = self.conv1(main)
        minor_1 = self.conv2(minor)
        mix = torch.cat([main_1, torch.mul(main_1, minor_1), minor_1], dim=1)
        ca_mix = mix.mul(self.att_c1(mix))
        sa_main = main_1.mul(self.att_s1(main_1))
        sa_minor = minor_1.mul(self.att_s2(minor_1))
        out = self.conv3(ca_mix) + sa_main + sa_minor + main + minor
        return out


class FDCM_1(nn.Module):
    def __init__(self):
        super(FDCM_1, self).__init__()
        # 上采样
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2,)
        self.conv1 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64,64, kernel_size=3, padding=1)
        # 最大池化，缩小两倍大小
        self.max_pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        # 做通道注意力
        self.glob_maxpool = nn.AdaptiveMaxPool2d(1)
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(16)
        self.relu = nn.GELU()
        self.ratio = 16
        self.fc1 = nn.Conv2d(128, 128 // self.ratio, 1)
        self.fc2 = nn.Conv2d(128 // self.ratio, 128, 1)
    def forward(self, rgb, t):

        f_r_t = rgb - t
        f_t_r = t - rgb
        w_r_t = self.sigmoid(self.conv3(self.max_pool(self.conv1(self.upsample(f_r_t)))))
        w_t_r = self.sigmoid(self.conv4(self.max_pool(self.conv2(self.upsample(f_t_r)))))

        f_r = torch.mul(w_r_t,rgb)
        f_t = torch.mul(w_t_r,t)

        fusion_t = f_r + t
        fusion_r = f_t + rgb

        fusion_final = torch.cat([fusion_t, fusion_r], dim=1)

        ca = self.sigmoid(self.fc2(self.relu(self.fc1((self.glob_maxpool(fusion_final))))))

        fusion_final = torch.mul(ca,fusion_final)

        return fusion_final

class FDCM_2(nn.Module):
    def __init__(self):
        super(FDCM_2, self).__init__()
        self.frs = nn.Sequential(nn.Conv2d(256, 3,kernel_size=3,padding=1), nn.BatchNorm2d(3), nn.Sigmoid())
        self.conv = nn.Conv2d(256,1,kernel_size=1)
    def forward(self, rgb, t, fusion):
        cat_map = torch.cat([rgb, t, fusion], dim=1)
        w = self.frs(cat_map)
        w1 = w[:, 0:1, :, :,]
        w2 = w[:, 1:2, :, :,]
        w3 = w[:, 2:3, :, :, ]
        f1 = torch.mul(rgb, w1)
        f2 = torch.mul(t, w2)
        f3 = torch.mul(fusion, w3)
        temp = torch.cat([f1, f2, f3], dim=1)
        fusion_final = self.conv(temp)

        return fusion_final

class FDCM(nn.Module):
    def __init__(self):
        super(FDCM, self).__init__()
        self.a = FDCM_1()
        self.b = FDCM_2()
    def forward(self, rgb,  t):
        fusion = self.a(rgb , t)
        res = self.b(rgb, t, fusion)

        return res

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1, ), nn.BatchNorm2d(512), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, ), nn.BatchNorm2d(512), nn.GELU())
        self.skip1 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.up1   = nn.UpsamplingBilinear2d(scale_factor=2, )

        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1, ), nn.BatchNorm2d(256), nn.GELU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, ), nn.BatchNorm2d(256), nn.GELU())
        self.skip2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.up2   = nn.UpsamplingBilinear2d(scale_factor=2, )

        self.conv5 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU())
        self.conv6 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU())
        self.skip3 = nn.Conv2d(256, 128, 1, 1, 0)
        self.up3   = nn.UpsamplingBilinear2d(scale_factor=2, )

        self.conv7 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU())
        self.conv8 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU())
        self.skip4 = nn.Conv2d(128, 64, 1, 1, 0)
        self.up4   = nn.UpsamplingBilinear2d(scale_factor=2, )

    def forward(self, feature, a):
        dec_list = []

        x1 = feature[3] + a[3]
        x2 = self.conv2(self.conv1(x1)) + self.skip1(x1)

        dec1 = self.up1(x2)

        x3 = dec1 + feature[2] + a[2]
        x4 = self.conv4(self.conv3(x3)) + self.skip2(x3)

        dec2 = self.up2(x4)

        x5 = dec2 + feature[1] + a[1]
        x6 = self.conv6(self.conv5(x5)) + self.skip3(x5)

        dec3 = self.up3(x6)

        x7 = dec3 + feature[0] + a[0]
        x8 = self.conv8(self.conv7(x7)) + self.skip4(x7)

        dec4 = self.up4(x8)

        dec_list.extend([dec1, dec2, dec3, dec4])

        return dec_list

'''
convn
'''
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths

        self.attn = nn.ModuleList([
            AACM(128),
            AACM(256),
            AACM(512),
            AACM(1024)
        ])


        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x,y=None):
        feature = []
        if y is not None:
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                x = self.attn[i](x, y[i])
                feature.append(x)
        else:
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                feature.append(x)
        return feature

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

class Generator(nn.Module):
    def __init__(self, norm_layer = nn.LayerNorm):
        super(Generator, self).__init__()
        # 主干
        self.cnext_rgb = convnextv2_base()
        self.cnext_thermal = convnextv2_base()

        # 解码
        self.decoderr = Decoder()
        self.decodert = Decoder()

        self.feature_fusion = FDCM()

        self.upsample1 = nn.Sequential(nn.Conv2d(64, 1, 3, 1, 1, ), nn.BatchNorm2d(1), nn.GELU(),)
        self.upsample2 = nn.Sequential(nn.Conv2d(64, 1, 3, 1, 1, ), nn.BatchNorm2d(1), nn.GELU(),)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2, )
        self.upsample4 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU(),
                                       nn.Conv2d(128, 1, 3, 1, 1, ), nn.BatchNorm2d(1), nn.GELU())
        self.upsample5 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU(),
                                       nn.Conv2d(128, 1, 3, 1, 1, ), nn.BatchNorm2d(1), nn.GELU())
        self.upsample6 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                       nn.Conv2d(64, 1, 3, 1, 1, ), nn.BatchNorm2d(1), nn.GELU())
        self.upsample7 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                       nn.Conv2d(64, 1, 3, 1, 1, ), nn.BatchNorm2d(1), nn.GELU())
        self.upsample8 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1, ), nn.BatchNorm2d(32), nn.GELU(),
                                       nn.Conv2d(32, 1, 3, 1, 1, ), nn.BatchNorm2d(1), nn.GELU())
        self.upsample9 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1, ), nn.BatchNorm2d(32), nn.GELU(),
                                       nn.Conv2d(32, 1, 3, 1, 1, ), nn.BatchNorm2d(1), nn.GELU())

    def forward(self, image, thermal):

        rgb_feature_list1 = self.cnext_rgb(image)
        thermal_feature_list1 = self.cnext_thermal(thermal,rgb_feature_list1)

        thermal_feature_list2 = self.cnext_thermal(thermal)
        rgb_feature_list2 = self.cnext_rgb(image,thermal_feature_list2)


        dec_rgb_list = self.decodert(rgb_feature_list2, thermal_feature_list2)
        dec_t_list   = self.decoderr(thermal_feature_list1, rgb_feature_list1)

        feature_final = self.feature_fusion(dec_t_list[3],dec_rgb_list[3])

        rgb_feature = []
        rgb_feature.append(self.upsample5(dec_rgb_list[0]))
        rgb_feature.append(self.upsample7(dec_rgb_list[1]))
        rgb_feature.append(self.upsample9(dec_rgb_list[2]))
        rgb_feature.append(self.upsample1(dec_rgb_list[3]))
        thermal_feature = []
        thermal_feature.append(self.upsample4(dec_t_list[0]))
        thermal_feature.append(self.upsample6(dec_t_list[1]))
        thermal_feature.append(self.upsample8(dec_t_list[2]))
        thermal_feature.append(self.upsample2(dec_t_list[3]))
        feature_final   = self.upsample3(feature_final)

        return rgb_feature, thermal_feature, feature_final

    def load_pre(self, pre_model):
        self.cnext_rgb.load_state_dict(torch.load(pre_model)['model'],strict=False)
        self.cnext_thermal.load_state_dict(torch.load(pre_model)['model'], strict=False)