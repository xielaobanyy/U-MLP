
import os

from torchsummary import summary
from data import *
import math
from data import calc_uncertainty
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from torchsummary import summary
__all__ = [
    'resmlp_12', 'resmlp_24', 'resmlp_36', 'resmlpB_24'
]

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):

        B, N, C = x.shape
        print(B,N,C,H,W)
        print(x.shape)
        x = x.transpose(1, 2).contiguous().view(B,int(math.sqrt(H)),int(math.sqrt(H)), W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x




class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size = 256, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # print("dimdimdimdimdimdimdimdimdim")
        # print(dim)

    def forward(self, x):
        # _, _, d = x.size()
        # if d == 192:
        #     print(1)
        # if d == 384:
        #     print(2)
        # if d == 768:
        #     print(3)
        # if d == 1536:
        #     print(4)
        # print("x大小x大小x大小x大小x大小x大小")
        # print(x.shape)
        # print("a大小a大小a大小a大小a大小a大小")
        # a = (self.alpha * x)
        # print(a.shape)
        # print("b大小b大小b大小b大小b大小b大小")
        # b = (self.beta)
        # print(b.shape)
        return self.alpha * x + self.beta


class sMLPBlock(nn.Module):
    def __init__(self, W, H, channels=3):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)
        self.proj_h = nn.Conv2d(H, H, (1, 1))
        self.proh_w = nn.Conv2d(W, W, (1, 1))
        self.fuse = nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)

    def forward(self, x):
        x = self.activation(self.BN(x))
        x_h = self.proj_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.fuse(torch.cat([x, x_h, x_w], dim=1))
        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, hidden_features=None, qk_bias=False, qk_scale=None, drop=0., drop_path=0., act_layer=nn.GELU,
                 init_values=1e-4, num_patches=192, downsample=None,
                 norm_layer=nn.LayerNorm, Hw=56):
        super().__init__()
        self.sliding = Centersliding(dim,Hw,
                                     qk_bias=qk_bias, qk_scale=qk_scale,
                                    attn_drop=drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = nn.Linear(num_patches, num_patches)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(4.0 * dim), act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.proj_h = nn.Conv2d(Hw, Hw, (1, 1))
        self.proh_w = nn.Conv2d(Hw, Hw, (1, 1))

        # self.dwconv = DWConv(hidden_features)
        self.LN = norm_layer(dim)
        self.fuse = nn.Conv2d(dim * 3, dim, (1, 1), (1, 1), bias=False)
        self.activation = nn.GELU()

        self.fuse = nn.Conv2d(int(dim * 3), dim, (1, 1), (1, 1), bias=False)
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # self.pool = nn.Conv2d(dim, dim/2, 1, bias=False)
        # self.pool = nn.MaxPool2d(kernel_size=2)


    def forward(self, x, H, W):

        # r = self.drop_path(self.gamma_1 * self.attn(self.norm1(x).transpose(1, 2)).transpose(1, 2))

        # r1 = self.norm1(x).transpose(1, 2)
        # print("r1r1r1r1r1r1r1r1r1r1r1r1r1r1r1r1r1r1r1r1r1r1")
        # print(r1.shape)
        #
        # r2 = self.attn(r1)
        # print("r2r2r2r2r2r2r2r2r2r2r2r2r2r2r2r2r2r2r2r2r2r2")
        # print(r2.shape)
        #
        # r3 = r2.transpose(1, 2)
        # print("r3r3r3r3r3r3r3r3r3r3r3r3r3r3r3r3r3r3r3r3r3r3")
        # print(r3.shape)
        #
        # r4 = self.gamma_1
        # print(r4.shape)
        #
        B,L,C = x.size()
        # # print(x)
        # #
        # print(x.shape)

        x = x + self.drop_path(self.sliding(self.norm1(x), H, W))
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        identity = x
        # print(x)
        # print(x.shape)
        out1 = self.mlp(self.norm2(x))
        # print("out1")
        # print(out1.shape)
        x = self.activation(self.LN(out1))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(x.shape)
        # print("0000000000000000000000")
        # print(x.shape)
        # x_h1 = self.dwconv(x,H,W)
        x_h = self.proj_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        # print(x_h1.shape)
        #
        # x_w1 = self.dwconv(x,H,W)
        # print(x_w1.shape)
        out1 = self.fuse(torch.cat([x, x_h, x_w], dim=1))

        out1 = out1.reshape(B,L,C)
        # print("1111")
        # print(out1.shape)
        # print(H,W)
        # print("2222")
        # print((self.gamma_2).shape)
        x = identity + self.drop_path(self.gamma_2 * out1)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # print(x.shape)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2


        # x = self.pool(x)
        # print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
        # print(x.shape)
        return x, H, W

class Centersliding(nn.Module):
    """
    """
    # self.attn = CenterAttention(
    #     dim=dim,
    #     num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
    #     attn_drop=attn_drop, proj_drop=drop)
    def __init__(self,
                 dim,
                 Hw,
                 num_heads=16,
                 qk_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 stride=1,
                 padding=True,
                 kernel_size=3,
                 ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.k_size = kernel_size  # kernel size
        self.stride = stride  # stride
        # self.pat_size = patch_size  # patch size

        self.in_channels = dim  # origin channel is 3, patch channel is in_channel
        self.num_heads = num_heads
        self.head_channel = dim // num_heads
        # self.dim = dim # patch embedding dim
        # it seems that padding must be true to make unfolded dim matchs query dim h*w*ks*ks
        self.pad_size = kernel_size // 2 if padding is True else 0  # padding size
        self.pad = nn.ZeroPad2d(self.pad_size)  # padding around the input
        self.scale = dim ** -0.5
        self.unfold = nn.Unfold(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)

        self.qkv_bias = qk_bias
        self.q_proj = nn.Linear(Hw, Hw, bias=qk_bias)
        self.kv_proj = nn.Linear(Hw, Hw * 2, bias=qk_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gelu = nn.GELU()


        self.f_qr = nn.Parameter(torch.tensor(1.0),  requires_grad=True)
        self.f_kr = nn.Parameter(torch.tensor(1.0),  requires_grad=True)
        self.f_vr = nn.Parameter(torch.tensor(1.0),  requires_grad=True)


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


    def forward(self, x, H, W):
        B, N, C = x.shape
        # print(B, N, C)
        # print(H, W)

        xn = x.reshape(B, C, H, W)
        # print(xn.shape)
        assert C == self.in_channels

        self.pat_size_h = (H + 2 * self.pad_size - self.k_size) // self.stride + 1
        self.pat_size_w = (W + 2 * self.pad_size - self.k_size) // self.stride + 1
        self.num_patch = self.pat_size_h * self.pat_size_w
        # print(xn.shape)
        q = self.q_proj(xn).reshape(B, H, W, self.num_heads, self.head_channel).permute(0, 3, 1, 2, 4)
        # print(q.shape)

        q = q.unsqueeze(dim=4)

        q = q * self.scale

        kv = self.kv_proj(xn).reshape(B, H, W, 2, self.num_heads, self.head_channel).permute(3, 0, 4, 5, 1, 2)

        kv = self.pad(kv)
        kv = kv.permute(0, 1, 2, 4, 5, 3)
        H, W = H + self.pad_size * 2, W + self.pad_size * 2
        kv = kv.permute(0, 1, 2, 5, 3, 4).reshape(2 * B, -1, H, W)
        kv = self.unfold(kv)

        kv = kv.reshape(2, B, self.num_heads, self.head_channel, self.k_size ** 2,
                        self.num_patch)  # (2, B, NumH, HC, ks*ks, NumPatch)
        kv = kv.permute(0, 1, 2, 5, 4, 3)  # (2, B, NumH, NumPatch, ks*ks, HC)
        k, v = kv[0], kv[1]

        k = torch.mul(k, self.f_kr)
        v = torch.mul(v, self.f_vr)
        q = q.reshape(B, self.num_heads, self.num_patch, 1, self.head_channel)
        # print("q:")
        # print(q.shape)
        q = torch.mul(q, self.f_qr)
        attn = (q @ k.transpose(-2, -1))  # (B, NumH, NumPatch, ks*ks, ks*ks)
        attn = self.softmax(attn)  # softmax last dim
        attn = self.attn_drop(attn)
        out = (attn @ v).squeeze(3)  # (B, NumH, NumPatch, HeadC)
        out = out.permute(0, 2, 1, 3).reshape(B, self.pat_size_h, self.pat_size_w, C)  # (B, Ph, Pw, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.reshape(B, -1, C)

        # print("sliding_out:")
        # print(out.shape)
        return out



class resmlp_models(nn.Module):

    def __init__(self, img_size=512, patch_size=16, in_chans=3,
                 num_classes=1, embed_dim=768, depth=12, drop_rate=0.,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU,norm_layer=nn.LayerNorm,
                 drop_path_rate=0.0, init_scale=1e-4, patch_norm=True):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = depth
        self.num_features = self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # self.patch_embed = Patch_layer(
        #     img_size=img_size, patch_size=patch_size, in_chans=int(in_chans), embed_dim=embed_dim)
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for i in range(depth)]

        self.layer1 = nn.ModuleList([BasicLayer(
                dim=int(embed_dim * 2 ** 0),
                drop=drop_rate,
                drop_path=dpr[0],
                act_layer=act_layer,
                init_values=init_scale,
                Hw=64,
                num_patches=int(num_patches / 4 ** 0),
                downsample=PatchMerging if (0 < self.num_layers - 1) else None,)])
        self.layer2 = nn.ModuleList([BasicLayer(
                dim=int(embed_dim * 2 ** 1),
                drop=drop_rate,
                drop_path=dpr[1],
                act_layer=act_layer,
                init_values=init_scale,
                Hw=32,
                num_patches=int(num_patches / 4 ** 1),
                downsample=PatchMerging if (1 < self.num_layers - 1) else None,)
        ])
        self.layer3 = nn.ModuleList([BasicLayer(
                dim=int(embed_dim * 2 ** 2),
                drop=drop_rate,
                drop_path=dpr[2],
                act_layer=act_layer,
                init_values=init_scale,
                Hw=16,
                num_patches=int(num_patches / 4 ** 2),
                downsample=PatchMerging if (2 < self.num_layers - 1) else None,)
        ])
        self.layer4 = nn.ModuleList([BasicLayer(
                dim=int(embed_dim * 2 ** 3),
                drop=drop_rate,
                drop_path=dpr[3],
                act_layer=act_layer,
                init_values=init_scale,
                Hw=8,
                num_patches=int(num_patches / 4 ** 3),
                downsample=PatchMerging if (3 < self.num_layers - 1) else None,)
        ])


        #self.norm1 = norm_layer(1536)
        #self.norm2 = nn.Linear(1536,3072)
        #self.norm3 = nn.Linear(768,1536)
        #self.norm4 = nn.Linear(384,768)
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm1 = norm_layer(768)
        self.norm2 = nn.Linear(3072,6144)
        self.norm3 = nn.Linear(1536,3072)
        self.norm4 = nn.Linear(768,1536)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        # decoder
        self.decoder1 = nn.Conv2d(3072, 768, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(768, 384, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(384, 192, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(192, 96, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        #self.dbn1 = nn.BatchNorm2d(384)
        #self.dbn2 = nn.BatchNorm2d(192)
        #self.dbn3 = nn.BatchNorm2d(96)
        #self.dbn4 = nn.BatchNorm2d(48)
        self.dbn1 = nn.BatchNorm2d(768)
        self.dbn2 = nn.BatchNorm2d(384)
        self.dbn3 = nn.BatchNorm2d(192)
        self.dbn4 = nn.BatchNorm2d(96)

        # gain low_feats
        # self.conv_7 = nn.Conv2d(384, 64, kernel_size=7, stride=1, padding=3, bias=False)
        # self.dwconv = DWConv(32)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv_7 = nn.Conv2d(768, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.dwconv = DWConv(32)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        #
        self.sec = nn.Conv2d(24, 96, 3, stride=1, padding=1)
        self.final = nn.Conv2d(24, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)
        # refine
        self.local_avg = nn.AvgPool2d(7, stride=1, padding=3)
        self.local_max = nn.MaxPool2d(7, stride=1, padding=3)
        self.local_convFM = nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1)
        self.local_ResMM = ResBlock(1, 1)
        self.local_pred2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
        self.proj = nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1)
        self.local_pred3 = nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=1)
        self.local_pred4 = nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=1)

        self.refine = nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def Gain_Low_feature(self, feats):
        # f = (feats - self.mean) / self.std
        b, h, w = feats.size()
        # print(b,h,w)
        feats = feats.reshape(b,w,32,32)
        # x = self.dwconv(feats, h, w)
        x = self.conv_7(feats)
        # print(x.shape)
        x = self.bn1(x)
        low_feats = self.relu(x)

        return low_feats

    def Local_refinement(self,score,low_feats):

        bs, obj_n, h, w = score.size()
        # print(f'hou:{bs, obj_n, h, w}')
        rough_seg = F.softmax(score, dim=1)
        # print(rough_seg.shape)
        rough_seg = rough_seg.view(bs, obj_n, h, w)
        rough_seg = F.softmax(rough_seg, dim=1)  # object-level normalization
        # print(f"l1{low_feats.shape}")
        # Local refinement
        uncer = calc_uncertainty(rough_seg)  # 计算不确定映射 U
        # print(uncer.shape)
        uncer = uncer.expand(-1, obj_n, -1, -1).reshape(bs * obj_n, 1, h, w)
        # print(uncer.shape)
        rough_seg = rough_seg.view(bs * obj_n, 1, h, w)  # bs*obj_n, 1, h, w

        low_feats = low_feats.view(bs * obj_n, 1, h, w)  # bs*obj_n, 1, h, w
        # print(f"rough_seg{rough_seg.shape}")
        # print(f"low_feats{low_feats.shape}")
        r1_weighted = low_feats * rough_seg  # M * r
        r1_local = self.local_avg(r1_weighted)  # bs*obj_n, 64, h, w
        # print(f'r1_local1:{r1_local.shape}')
        r1_local = r1_local / (self.local_avg(rough_seg) + 1e-8)  # neighborhood reference
        # print(f'r1_local2:{r1_local.shape}')
        r1_conf = self.local_max(rough_seg)  # bs*obj_n, 1, h, w
        # print(f'r1_conf:{r1_conf.shape}')

        local_match = torch.cat([low_feats, r1_local], dim=1)
        # print(f'local_match:{local_match.shape}')
        q = self.local_ResMM(self.local_convFM(local_match))
        # print(f'q:{q.shape}')
        q = r1_conf * self.local_pred2(F.relu(q))  # q 是根据残差网络模块为每个像素分配的局部细化掩码++++对应文中的 e
        # e = c * fl(r, y)
        # print(f'q:{q.shape}')
        # print(f'sc:{score.shape}')
        # score = self.local_pred3(score).view(bs * obj_n, 2, h, w)
        unlocal = uncer * q
        # print(f'unlocal:{unlocal.shape}')
        unlocal = unlocal.view(bs, obj_n, h, w)
        # print(f'unlocal:{unlocal.shape}')
        p = score + unlocal  # 最终的分割结果 S = M + U * e
        # p = p.view(bs, obj_n, h, w)
        # p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)
        # p = F.softmax(p, dim=1)[:, 1]  # no, h, w
        # print(f"p{p.shape}")
        # return p
        # print(uncertainty.shape)
        return p


    def forward(self, x):
        B = x.shape[0]
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # enconder1
        for i, blk in enumerate(self.layer1):
            x1, H1, W1 = blk(x, H, W)
            #print(B, H1, W1)
        low_feats = self.Gain_Low_feature(x1)

        # enconder2
        for i, blk in enumerate(self.layer2):
            x2, H2, W2 = blk(x1, H1, W1)


        # enconder3
        for i, blk in enumerate(self.layer3):
            x3, H3, W3 = blk(x2, H2, W2)

        # enconder4
        for i, blk in enumerate(self.layer4):
            x4, H4, W4 = blk(x3, H3, W3)

        t4 = x4
        #print("tttttttttttt4444444444444444")
        #print(t4.shape)
        t4 = t4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        # t4 = t4.reshape(B, int(H4 * 2), int(W4 * 2), -1).permute(0, 3, 1, 2).contiguous()
        # print(t4.shape)
        ### Stage 4
        out = F.gelu(F.interpolate(self.dbn1(self.decoder1(t4)), scale_factor=(2, 2), mode='bilinear'))
        # print(out.shape)
        # print(out.size())
        t4 = t4.reshape(out.size())
        out = torch.add(out,t4)
        # print(out.shape)

        ### Stage 3
        #print("3333333333333333333333333333333")
        #print(x3.shape)

        t3 = self.norm2(x3)
        t3 = t3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        # print(t3.shape)
        out = F.gelu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        # print(out.shape)
        # print(out.size())
        t3 = t3.reshape(out.size())
        out = torch.add(out,t3)
        # print(out.shape)

        ### Stage 2
        # print("222222222222222222222222222222222")
        # print(x2.shape)

        t2 = self.norm3(x2)
        t2 = t2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        # print(t2.shape)
        out = F.gelu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        # print(out.shape)
        # print(out.size())
        t2 = t2.reshape(out.size())
        out = torch.add(out,t2)
        # print(out.shape)

        ### Stage 1
        # print("11111111111111111111111111111")
        # print(x1.shape)

        t1 = self.norm4(x1)
        t1 = t1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        # print(t1.shape)
        out = F.gelu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        # print(out.shape)
        # print(out.size())
        t1 = t1.reshape(out.size())
        out1 = torch.add(out,t1)

        #print(out1.shape)



        # _,_,H,W = out.shape
        # out = out.flatten(2).transpose(1,2)
        # for i, blk in enumerate(self.dblock1):
        #     out = blk(out, H, W)



        # print(out1.shape)
        #out1 = self.sec(out1)
        b,c,h,w = out1.size()
        # print(out1.shape)

        out1 = out1.resize(b,int(c/4),int(h*2),int(w*2))
        # print(out1.shape)
        final = self.final(out1)
        # print(final.shape)
        score = final
        # print("score")
        #print(score.shape)
        # bs, obj_n, ho, wo = score.size()
        # print(f'qian:{bs, obj_n, ho, wo}')
        # uncertainty = calc_uncertainty(F.softmax(final, dim=1))  # 计算不确定映射 U
        # print(uncertainty.shape)
        # uncertainty = uncertainty.view(bs, -1).norm(p=2, dim=1) / math.sqrt(ho * wo)  # [B,1,H,W]  求 2-范数：Lconf
        # print(uncertainty.shape)
        # uncertainty = uncertainty.mean()

        final = self.Local_refinement(score, low_feats)

        # print(final.shape)
        # print(final.shape)
        # print(uncertainty.shape)
        # print(final.shape)
        return final

@register_model
def resmlp_12(pretrained=False, dist=False, **kwargs):
    model = resmlp_models(
        patch_size=4, embed_dim=384, depth=4,
        Patch_layer=PatchEmbed,
        init_scale=0.1, **kwargs)

    model.default_cfg = _cfg()
    # if pretrained:
    #     #     if dist:
    #     #         url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth"
    #     #     else:
    #     #         url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth"
    #     #     checkpoint = torch.hub.load_state_dict_from_url(
    #     #         url=url_path,
    #     #         map_location="cpu", check_hash=True
    #     #     )
    #     #
    #     #     model.load_state_dict(checkpoint,False)
    return model

@register_model
def resmlp_24(pretrained=False, dist=False, dino=False, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-5, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
            url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth"
        elif dino:
            url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth"
        else:
            url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )

        model.load_state_dict(checkpoint)
    return model


@register_model
def resmlp_36(pretrained=False, dist=False, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=36,
        Patch_layer=PatchEmbed,
        init_scale=1e-6, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
            url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth"
        else:
            url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )

        model.load_state_dict(checkpoint)
    return model


@register_model
def resmlpB_24(pretrained=False, dist=False, in_22k=False, **kwargs):
    model = resmlp_models(
        patch_size=8, embed_dim=768, depth=4,
        Patch_layer=PatchEmbed,
        init_scale=1e-6, **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     if dist:
    #         url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth"
    #     elif in_22k:
    #         url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth"
    #     else:
    #         url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth"
    #
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url=url_path,
    #         map_location="cpu", check_hash=True
    #     )
    #
    #     model.load_state_dict(checkpoint)

    return model




def test():
    from thop import profile
    import time
    # model = Local_refinement()
    # model = Unet(backbone="vgg")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = resmlp_12().to(device)
    net = net.cuda()
    summary(net, input_size=(3, 256, 256))
    start_time = time.time()
    input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(net, inputs=(input,))
    end_time = time.time()
    sum_time = end_time - start_time
    print("infer_time:{:.3f}ms".format(sum_time*1000))
    print("flops:{:.3f}G".format(flops/1e9))
    print("params:{:.3f}M".format(params/1e6))
if __name__ == "__main__":
    test()
