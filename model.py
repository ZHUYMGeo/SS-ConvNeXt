"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, Mlp, PatchEmbed as TimmPatchEmbed
from torch import Tensor
from typing import Optional, Tuple, Union, Dict
import os
from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SS_ConvNeXt(nn.Module):
    def __init__(self, input_shape, num_classes: int = 16, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0.5, layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):

        super(SS_ConvNeXt, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.dims = dims
        self.depths = depths
        self.downsample_layers1 = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.downsample_layers2 = nn.ModuleList()
        self.layers = nn.ModuleList()

        layer1 = nn.Sequential(nn.Conv2d(input_shape[1], dims[0], kernel_size=1, stride=1, padding=0, bias=False),
                               LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                               nn.GELU())
        self.layers.append(layer1)
        layer2 = nn.Sequential(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                                       nn.Conv2d(dims[0], dims[1], kernel_size=1, stride=1, padding=0))
        self.layers.append(layer2)
        layer3 = nn.Sequential(nn.Conv2d(dims[1], dims[2], kernel_size=1, stride=1, padding=0, bias=False),
                               LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                               nn.GELU())
        self.layers.append(layer3)
        layer4 = nn.Sequential(LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                                       nn.Conv2d(dims[2], dims[3], kernel_size=1, stride=1, padding=0))
        self.layers.append(layer4)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = depths[0] + depths[1]
        cur1 = sum(depths) - depths[3]

        self.spatial_conv = nn.ModuleList()

        self.spectral_conv = nn.ModuleList()
        for i in range(2):
            spatial_stage = nn.Sequential(*[spatial_ConvBlock(dim=dims[i], drop_rate=dp_rates[j] if i == 0 else dp_rates[j+depths[0]], layer_scale_init_value=layer_scale_init_value)
                                       for j in range(depths[i])])

            self.spatial_conv.append(spatial_stage)
            spectral_stage = nn.Sequential(*[spectral_ConvBlock(dim=dims[i+2], drop_rate=dp_rates[cur+j] if i+2 == 2 else dp_rates[cur1+j], layer_scale_init_value=layer_scale_init_value)
                                       for j in range(depths[i+2])])

            self.spectral_conv.append(spectral_stage)

        self.ln1 = LayerNorm(dims[2], eps=1e-6, data_format="channels_first")
        self.ln2 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        self.ln3 = LayerNorm(dims[3], eps=1e-6, data_format="channels_first")

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(dims[-1], num_classes)
        self.activate = nn.GELU()   # GELU

        self.apply(self.initialize_weights)
        self.fc.weight.data.mul_(head_init_scale)
        self.fc.bias.data.mul_(head_init_scale)

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
            # nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _forward_spa_cvNet(self,x):
        for i in range(2):
            x = self.layers[i](x)
            x = self.spatial_conv[i](x)

        return x

    def _forward_spe_cvNet(self,x):
        for i in range(2):
            x = self.layers[i+2](x)
            x = self.spectral_conv[i](x)

        return x

    def forward(self, x):
        x = self._forward_spa_cvNet(x)
        x = self._forward_spe_cvNet(x)
        x = self.activate(self.ln3(x))
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class spatial_ConvBlock(nn.Module):

    def __init__(self, dim, drop_rate=0.5, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.ln1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class spectral_ConvBlock(nn.Module):

    def __init__(self, dim, drop_rate=0.5, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.ln = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x
