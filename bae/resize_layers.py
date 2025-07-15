# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DCDownBlock2d(nn.Module):
    def __init__(self, dim: int, downsample: bool = False, shortcut: bool = True) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = dim * self.factor**2 // dim
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio
        else:
            out_channels = dim

        self.conv = nn.Conv2d(
            dim,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0,3,1,2)
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        hidden_states = hidden_states.flatten(2).permute(0,2,1)
        return hidden_states


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        dim: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()
        out_channels = dim
        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // dim

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0,3,1,2)
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x
        # hidden_states = hidden_states.permute(0,2,3,1)
        hidden_states = hidden_states.flatten(2).permute(0,2,1)
        return hidden_states

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
 
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
 
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(dim)
        self.initialize_weights()
    
    def initialize_weights(self):
        with torch.no_grad():
            # Create weights for average interpolation
            weight = torch.zeros(self.reduction.out_features, self.reduction.in_features)
            for i in range(self.reduction.out_features):
                start = i * self.reduction.in_features // self.reduction.out_features
                end = (i + 1) * self.reduction.in_features // self.reduction.out_features
                weight[i, start:end] = 1.0 / (end - start)
            
            self.reduction.weight.copy_(weight)
            if self.reduction.bias is not None:
                self.reduction.bias.zero_()
 
    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape

        res_x = F.interpolate(x.permute(0, 3, 1, 2), scale_factor=0.5, mode='area').permute(0, 2, 3, 1)
        res_x = res_x.view(B, -1, C).contiguous()
        x = x.view(B, H//2, 2, W//2, 2, C).permute(0,1,3,5,2,4).reshape(B, -1, 4 * C)
 
        x = self.reduction(x)
        x = self.norm(x)
        x = 0.5 * x + 0.5 * res_x
        
        return x


class PatchExpand(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expansion = nn.Linear(dim, dim * 4, bias=False)
        self.norm = norm_layer(dim * 4)

    def forward(self, x):
        """
        x: B, L, C
        """

        B, H, W, C = x.shape
        # H = W = int(math.sqrt(L))
        
        # x = x.view(B, H, W, C)
        res_x = F.interpolate(x.permute(0, 3, 1, 2), scale_factor=2, mode='area').permute(0, 2, 3, 1)
        res_x = res_x.view(B, -1, C).contiguous()
        
        x = self.expansion(x)
        x = self.norm(x)
        
        x = x.view(B, H, W , 2, 2, C).permute(0, 1, 3, 5, 2, 4).reshape(B, 2 * H, 2 * W, C)
        x = x.view(B, -1, C).contiguous()
        
        x = 0.5 * x + 0.5 * res_x

        return x