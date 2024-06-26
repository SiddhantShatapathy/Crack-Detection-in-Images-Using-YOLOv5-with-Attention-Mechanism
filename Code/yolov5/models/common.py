# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Common modules."""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

# Import 'ultralytics' package or install if if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os

    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

######################################
########Transformer in yolo##################

#check version of pytorch > 1.9.0 for transformer encoder layer
def check_torch_version():
    return torch.__version__

#Transformer encode layer for patch wise attention

class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=128, num_heads=8, dropout=0.0, act='gelu', normalize_before=False,embed_dim=128):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        #check torch version for transformer layer
        if check_torch_version() < "1.9.0":
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).")
        #print(c1)
        self.embed_dim = embed_dim
        self.ma = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout, batch_first=True)
        #implementation of Feedforward model
        self.fc1 = nn.Linear(self.embed_dim, cm)
        self.fc2 = nn.Linear(cm, self.embed_dim)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Activation '{act}' not supported. Choose 'gelu' or 'silu'.")
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        #print("query shape:",q.shape)
        #print("key shape:",k.shape)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        #print("features after MHA:",src2.shape)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout1(self.act(self.fc1(src))))
        #print("features after FC layer,activation and dropout:",src2.shape)
        src = src + self.dropout2(src2)
        #print("features after adding the original input:",src.shape)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

#Efficient AIFI encoder block using conv2d to get patch embed's
class AIFI_conv(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=128, num_heads=8, dropout=0, act='gelu', normalize_before=False, embed_dim=128,patch_size=[2, 2], stride=[2, 2]):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before,embed_dim)
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Activation '{act}' not supported. Choose 'gelu' or 'silu'.")
        self.patch_embedding = nn.Conv2d(c1, self.embed_dim, kernel_size=patch_size, stride=stride, bias=False)
        self.act_conv = self.act
        self.batch_norm_conv = nn.BatchNorm2d(self.embed_dim)
        self.patch_embedding_back = nn.ConvTranspose2d(self.embed_dim, c1, kernel_size=patch_size, stride=stride, bias=False)
        #self.norm_conv_inv = nn.LayerNorm(c1)
        self.act_conv_inv = self.act
        self.batch_norm_conv_inv = nn.BatchNorm2d(c1)  


    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        B, C, H, W = x.shape
        #print("input shape:",x.shape)
        #print(x)

        # Project patches into embeddings
        x = self.patch_embedding(x)
        # Batch normalization and activation
        x = self.act_conv(x)
        x = self.batch_norm_conv(x)
        
        #print("patch_embedding shape:",x.shape)
        #print(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        num_patches_h = x.shape[1]
        num_patches_w = x.shape[2]
        x = x.view(B,num_patches_h*num_patches_w,self.embed_dim)
        #print("patch_embedding after reshape:",x.shape)
        #print(x)
        

        pos_embed = self.build_2d_sincos_position_embedding(num_patches_h,num_patches_w,embed_dim=self.embed_dim,temperature=10000.0)
        #print("pos_embed shape:",pos_embed.shape)
        #print(pos_embed)
        # Pass through MHA
        x = super().forward(x, pos=pos_embed.to(device=x.device, dtype=x.dtype))
        #x = super().forward(x)
        #print("after mha shape:",x.shape)

        # Reverse the reshaping for reconstruction
        out_H = (H - self.patch_size[0]) // self.stride[0] + 1
        out_W = (W - self.patch_size[1]) // self.stride[1] + 1
        x = x.permute(0,2,1).contiguous()
        x = x.view(B, self.embed_dim, out_H, out_W)
        #print("prefinal reshape:",x.shape)
        x = self.patch_embedding_back(x, output_size=(H, W))
        #print("patch embedding back output shape:",x.shape)
        # Layer normalization and activation
        #x = self.norm_conv_inv(x)
        x = self.act_conv_inv(x)
        x = self.batch_norm_conv_inv(x)
        #print("final shape:",x.shape)
        return x

    #def num_parameters(self):
    #    """Calculate the number of parameters in the model."""
    #    return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def build_2d_sincos_position_embedding(num_patches_h,num_patches_w,embed_dim=128,temperature=10000.0):
        """Builds 2D sine-cosine position embedding for patches."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(num_patches_h, dtype=torch.float32)
        grid_h = torch.arange(num_patches_w, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        position_embedding = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]
        #position_embedding = position_embedding.unsqueeze(-1)
        #position_embedding = torch.repeat_interleave(position_embedding, patch_size[0]*patch_size[1], dim=-1)

        return position_embedding

#Efficient AIFI encoder block FC layer for patch embed's
class AIFI_new(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=128, num_heads=8, dropout=0, act='gelu', normalize_before=False,embed_dim=128,patch_size=[2, 2],stride = [2, 2]):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1 , cm, num_heads, dropout, act, normalize_before,embed_dim)
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Activation '{act}' not supported. Choose 'gelu' or 'silu'.")
        # added linear layer to reduce dimension
        # added linear layer to reduce dimension
        self.fc_reduce_dim = nn.Linear(c1 * patch_size[0] * patch_size[1], self.embed_dim)
        self.norm_fc = nn.LayerNorm(self.embed_dim)
        self.act_fc = self.act  # or any other activation function

        self.fc_org_dim = nn.Linear(self.embed_dim,c1 * patch_size[0] * patch_size[1])
        self.norm_fc_inv = nn.LayerNorm(c1 * patch_size[0] * patch_size[1])
        self.act_fc_inv = self.act  # or any other activation function

        # BatchNorm for both directions
        #self.batch_norm_fc = nn.BatchNorm2d(self.embed_dim)
        #self.batch_norm_fc_inv = nn.BatchNorm1d(c1 * patch_size[0] * patch_size[1])

        


    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        B, C, H, W = x.shape
        #print("org x shape:",x.shape)
        # calculate the number of patches along each dimension
        num_patches_h = (H - self.patch_size[0] ) // self.stride[0] + 1
        num_patches_w = (W - self.patch_size[1] ) // self.stride[1] + 1
        num_patches = num_patches_h * num_patches_w
        #unfold to extract non overlapping patches
        unfolded_patches = x.unfold(2, self.patch_size[0], self.stride[0]).unfold(3, self.patch_size[1], self.stride[1])
        #print(unfolded_patches.shape)
        #reshape the unfolded patches to (B, num_patches_h * num_patches_w, C * patch_size[0] * patch_size[1])
        #x = unfolded_patches.permute(0, 1, 4, 5, 2, 3).contiguous()
        x = unfolded_patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        #print("x",x.shape)
        x = x.view(B, num_patches_h * num_patches_w, C* self.patch_size[0] * self.patch_size[1])
        #x = x.view(B, C * self.patch_size[0] * self.patch_size[1], num_patches_h, num_patches_w)
        #add linear layer
        # linear transformation to reduce the dimension to embed_dim
        x = self.fc_reduce_dim(x)
        x = self.norm_fc(x)
        x = self.act_fc(x)
        
        # Add 1x1 convolutional layers to reduce the dimension to embed_dim
        #x = self.conv_reduce_dim(x)
        #print("new linear x",x.shape)
        #x = x.permute(0, 2, 3, 1).contiguous()
        #x = x.view(B,num_patches_h*num_patches_w,self.embed_dim)
        pos_embed = self.build_2d_sincos_position_embedding(num_patches_h,num_patches_w,self.embed_dim,10000.0)
        #print("pos embed:", pos_embed.shape)

        #print("x shape before MHA:",x.shape)

        #multi-head attention
        x = super().forward(x, pos=pos_embed.to(device=x.device, dtype=x.dtype))
        #print('after mha x shape',x.shape)

        x = self.fc_org_dim(x)
        x = self.norm_fc_inv(x)
        x = self.act_fc_inv(x)
        
        #print('after mha get org embed shape',x.shape)
        #reverse to back the unfolded patches
        x = x.view(B, num_patches_h, num_patches_w, C, self.patch_size[0], self.patch_size[1])
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        #print('after mha get unfolded patches',x.shape)

        #merge the patches to reconstruct the original tensor shape
        x = x.view(B, C, H, W)
        #print('after mha get org shape',x.shape)

        return x

    #def num_parameters(self):
    #    """Calculate the number of parameters in the model."""
    #    return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def build_2d_sincos_position_embedding(n_h,n_w,embed_dim=128,temperature=10000.0):
        """Builds 2D sine-cosine position embedding for patches."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(n_w, dtype=torch.float32)
        grid_h = torch.arange(n_h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        position_embedding = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]
        #position_embedding = position_embedding.unsqueeze(-1)
        #position_embedding = torch.repeat_interleave(position_embedding, patch_size[0]*patch_size[1], dim=-1)

        return position_embedding

######### old method ################################
class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=512, num_heads=8, dropout=0, act='gelu', normalize_before=False, patch_size=[2, 2],stride = [2, 2]):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1 * (patch_size[0] * patch_size[1]), cm, num_heads, dropout, act, normalize_before)
        self.patch_size = patch_size
        self.stride = stride
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Activation '{act}' not supported. Choose 'gelu' or 'silu'.")


    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        B, C, H, W = x.shape
        # calculate the number of patches along each dimension
        num_patches_h = (H - self.patch_size[0]) // self.stride[0] + 1
        num_patches_w = (W - self.patch_size[1]) // self.stride[1] + 1
        num_patches = num_patches_h * num_patches_w
        #unfold to extract non overlapping patches
        unfolded_patches = x.unfold(2, self.patch_size[0], self.stride[0]).unfold(3, self.patch_size[1], self.stride[1])
        #print(unfolded_patches.shape)
        #reshape the unfolded patches to (B, num_patches_h * num_patches_w, C * patch_size[0] * patch_size[1])
        x = unfolded_patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        #print(x.shape)
        x = x.view(B, num_patches_h * num_patches_w, C * self.patch_size[0] * self.patch_size[1])

        #get shapes for pos embed
        #b,c,n_h,n_w,h,w = unfolded_patches.shape
        #print(c)
        pos_embed = self.build_2d_sincos_position_embedding(num_patches_h,num_patches_w,H,W,self.patch_size,C,10000.0)
        #print("pos embed:", pos_embed.shape)

        #print("x shape before MHA:",x.shape)

        #multi-head attention
        x = super().forward(x, pos=pos_embed.to(device=x.device, dtype=x.dtype))
        #print('after mha x shape',x.shape)

        
        #reverse to back the unfolded patches
        x = x.view(B, num_patches_h, num_patches_w, C, self.patch_size[0], self.patch_size[1])
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        #print('after mha get unfolded patches',x.shape)

        #merge the patches to reconstruct the original tensor shape
        x = x.view(B, C, H, W)
        #print('after mha get org shape',x.shape)

        return x

    @staticmethod
    def build_2d_sincos_position_embedding(n_h,n_w,h,w,patch_size,embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding for patches."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(n_w, dtype=torch.float32)
        grid_h = torch.arange(n_h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        position_embedding = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]
        position_embedding = position_embedding.repeat_interleave(patch_size[0]*patch_size[1], dim=2)

        return position_embedding
##############################################

####    C3_cbam     ##########
class C3_cbam(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.se = CBAM(2 * c_)  # CBAM Attention block

    def forward(self, x):
        features = torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)
        features = self.se(features)  # Apply CBAM Attention
        return self.cv3(features)

#############
#### C3_senet ######

class C3_senet(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.se = SEBlock(2 * c_, reduction=16)  # SENet Attention block

    def forward(self, x):
        features = torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)
        features = self.se(features)  # Apply SENet Attention
        return self.cv3(features)


####################

### Coordinate attention block ######

import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CABlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CABlock, self).__init__()

        # Point-wise convolutional layer to reduce the number of channels
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0)
        
        # Batch normalization layer for the reduced channels
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        
        # Activation function using h-swish
        self.act = h_swish()

        # Point-wise convolutional layers for channel-wise attention
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Store the identity for later use
        identity = x

        # Get the batch size, number of channels, height, and width of the input tensor
        n, c, h, w = x.size()
        #print("shape:\n",n, c, h, w)

        # Spatial pooling along height and width separately
        x_h = F.avg_pool2d(x, kernel_size=(1, w))  # Average pooling along height
        #print("x_h before",x_h.shape)
        x_w = F.avg_pool2d(x, kernel_size=(h, 1))
        #print("x_h before",x_h.shape)
        #print("x_w before",x_w.shape)
        x_w = x_w.permute(0, 1, 3, 2)  # Average pooling along width
        #print("x_w after permute",x_w.shape)
        # Concatenate the pooled features along the channel dimension
        y = torch.cat([x_h, x_w], dim=2)
        #print("y before",y.shape)

        # Apply 1x1 convolution followed by batch normalization and activation function
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        #print("y after",y.shape)

        # Split the concatenated features back into separate height and width features
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # Transpose the width features to match the original shape
        #print("x_h after split",x_h.shape)
        #print("x_w after split",x_w.shape)
        # Apply channel-wise attention using 1x1 convolutions and sigmoid activation
        a_h = self.conv_h(x_h).sigmoid()  # Attention weights for height
        a_w = self.conv_w(x_w).sigmoid()  # Attention weights for width
        #print("a_h weights",a_h.shape)
        #print("a_w weights",a_w.shape)
        # Apply the attention weights to the original input tensor
        weights = a_w * a_h
        #print("weights shape",weights.shape)
        #print("initial shape",identity.shape)
        out = identity * weights
        
        #print("final shape",out.shape)

        return out
 
#####################################

#### CBAM Attention module ##########

class CBAM(nn.Module):
    def __init__(self, gate_channels,kernel_size=7, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.kernel_size = kernel_size
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(kernel_size=self.kernel_size)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

#class CBAM_inception(nn.Module):
#    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
#        super(CBAM, self).__init__()
#        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#        self.no_spatial = no_spatial
#        if not no_spatial:
#            self.SpatialGate = SpatialGate()
            #self.SpatialGate1 = SpatialGate(kernel_size=7)
            #self.SpatialGate2 = SpatialGate(kernel_size=9)

#            # Calculate the number of input channels for BasicConv
#            num_input_channels = 1  # Concatenating two SpatialGate outputs
#            self.conv_block = BasicConv(num_input_channels * gate_channels, gate_channels, kernel_size=1, stride=1, padding=0, relu=True)


#    def forward(self, x):
#        x_out = self.ChannelGate(x)
#        b,c,h,w = x_out.shape
        #print(x_out.shape)
#        if not self.no_spatial:
            #x_out_final = self.SpatialGate(x_out)
            #x_out_7 = self.SpatialGate1(x_out)
            #x_out_9 = self.SpatialGate2(x_out)
            #x_out_concat = torch.cat((x_out_7, x_out_9), dim=1)
            #x_out_final = self.conv_block(x_out_concat)

#        return x_out_final

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialGate, self).__init__()
        self.kernel_size = kernel_size
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, self.kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

class InceptionSpatialGate(nn.Module):
    def __init__(self):
        super(InceptionSpatialGate, self).__init__()
        self.compress = ChannelPool()
        #self.conv1x1 = BasicConv(2, 1, 1, stride=1, padding=0, relu=True)
        #self.conv3x3 = BasicConv(2, 1, 3, stride=1, padding=1, relu=True)
        #self.conv5x5 = BasicConv(2, 1, 5, stride=1, padding=2, relu=True)
        self.conv7x7 = BasicConv(2, 1, 7, stride=1, padding=3, relu=True)
        self.conv9x9 = BasicConv(2, 1, 9, stride=1, padding=4, relu=True)
        
        self.final_conv = BasicConv(2, 1, 1, stride=1,padding=0,relu=True)  # 4 input channels, 1 output channel

    def forward(self, x):
        x_compress = self.compress(x)
        #x1 = self.conv1x1(x_compress)
        #x3 = self.conv3x3(x_compress)
        #x5 = self.conv5x5(x_compress)
        x7 = self.conv7x7(x_compress)
        x9 = self.conv9x9(x_compress)

        # Concatenate features from different convolutions
        x_out = torch.cat((x7, x9), dim=1)

        # Apply one more convolution to get back the original dimensions
        x_out = self.final_conv(x_out)

        scale = F.sigmoid(x_out)  # No need for sum, as final_conv already reduces channels to 1
        return x * scale


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


#######################################
##### Adaptive ECANet ################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = self.get_kernel_size(channels, gamma, b)
        padding = (kernel_size - 1) // 2  # Adjust padding based on kernel size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.b = b

    def get_kernel_size(self, channels, gamma, b):
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        return k

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(self.b * y)

        return x * y.expand_as(x)

######################################

### ECABlock channel attention ##########
import torch
from torch import nn, true_divide
from torch.nn.parameter import Parameter

class ECABlock(nn.Module):
    """Constructs an ECA module."""

    def __init__(self, in_channels, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor on the global spatial information
        y = self.avg_pool(x)
        #y = self.max_pool(x)
        #print(y.shape)

        # Adaptive feature recalibration
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        #print(y.shape)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)




#########################################
### SEBlock channel attention #############
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#####################        #########################

############ test 2
import torch
import torch.nn as nn
import torch.nn.functional as F

class Edgedetection(nn.Module):
    def __init__(self, blur_kernel_size, blur_sigma):
        super(Edgedetection, self).__init__()
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.register_buffer('gaussian_kernel', self.create_gaussian_kernel())
        self.register_buffer('scharr_x', torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('scharr_y', torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        edges = self.edge_detection(x)
        return edges

    def edge_detection(self, images):
        # Apply Gaussian blur and Scharr filters to the input images
        blurred = self.gaussian_blur(images)
        scharr_output = self.scharr_filters(blurred)

        return scharr_output

    def create_gaussian_kernel(self):
      sigma = self.blur_sigma
      if sigma == 0:
        sigma = 0.3 * ((self.blur_kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = torch.ones(1, 1, self.blur_kernel_size, self.blur_kernel_size)
        center = self.blur_kernel_size // 2
        for i in range(self.blur_kernel_size):
            for j in range(self.blur_kernel_size):
                x, y = i - center, j - center
                x, y = torch.tensor(x), torch.tensor(y)
                kernel[0, 0, i, j] = torch.exp(-(x**2 + y**2) / (2.0 *sigma**2))
        kernel /= kernel.sum()
        return kernel

    def gaussian_blur(self, x):
        num_channels = x.size(1)
        blurred_channels = []
        for i in range(num_channels):
            channel = x[:, i:i+1, :, :]
            blurred_channel = F.conv2d(channel, self.gaussian_kernel, padding=self.blur_kernel_size//2)
            blurred_channels.append(blurred_channel)
        blurred_output = torch.cat(blurred_channels, dim=1)
        return blurred_output

    def scharr_filters(self, x):
        num_channels = x.size(1)
        scharr_channels = []
        for i in range(num_channels):
            channel = x[:, i:i+1, :, :]
            scharr_output_x = F.conv2d(channel, self.scharr_x, padding=1)
            scharr_output_y = F.conv2d(channel, self.scharr_y, padding=1)
            gradient_magnitude = torch.sqrt(scharr_output_x**2 + scharr_output_y**2)
            scharr_channels.append(gradient_magnitude)
        scharr_output = torch.cat(scharr_channels, dim=1)
        return scharr_output










################# Edge detection ##################################

import torchvision.transforms.functional as TF
import cv2
import numpy as np
class EdgeDetection(nn.Module):
    def __init__(self, low_threshold, high_threshold, blur_kernel_size=3,blur_sigma=0,scharr_kernel_size=3,L2_gradient=True):
        super(EdgeDetection, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.scharr_kernel_size = scharr_kernel_size
        self.L2_gradient = L2_gradient


    def forward(self, x):
        edges = self.edge_detection(x)
        return edges

    def create_gaussian_kernel(self,blur_kernel_size,):
        kernel = torch.ones(1, 1, self.blur_kernel_size, self.blur_kernel_size)
        center = self.blur_kernel_size // 2
        for i in range(self.blur_kernel_size):
            for j in range(self.blur_kernel_size):
                x, y = i - center, j - center
                x, y = torch.tensor(x), torch.tensor(y)  # Convert x and y to tensors
                kernel[0, 0, i, j] = torch.exp(-(x**2 + y**2) / (2.0 * self.sigma**2))
        kernel /= kernel.sum()
        self.register_buffer('gaussian_kernel', kernel)
        return kernel

    def gaussian_blur(self, x):
        # Get the number of channels in the input
        kernel = self.create_gaussian_kernel()
        num_channels = x.size(1)
        # Apply Gaussian blur to each channel separately
        blurred_channels = []
        for i in range(num_channels):
            channel = x[:, i:i+1, :, :]
            blurred_channel = F.conv2d(channel, self.gaussian_kernel, padding=self.kernel_size//2)
            blurred_channels.append(blurred_channel)
        
        # Concatenate the blurred channels to form the output
        blurred_output = torch.cat(blurred_channels, dim=1)

        return blurred_output

    def edge_detection(self, images):
        #start_time=time.time()
        batch_size, channels, height, width = images.size()
        edges_batch = []

        for i in range(batch_size):
            # Convert the current image in the batch to a NumPy array
            image_np = images[i].squeeze().cpu().numpy().astype(np.uint8)
            image_np = np.transpose(image_np, (1, 2, 0))

            # Apply edge detection operations
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            blurred = self.gaussian_blur(gray_image, self.blur_kernel_size, self.blur_sigma)
            gradient_magnitude, gradient_direction = self.scharr_filters(blurred, self.scharr_kernel_size, self.L2_gradient)
            #suppressed = self.non_max_suppression(gradient_magnitude, gradient_direction)
            #strong_edges, weak_edges = self.double_thresholding(suppressed, self.low_threshold, self.high_threshold)
            #edges = self.edge_tracking_by_hysteresis(strong_edges, weak_edges)
            edges=gradient_magnitude
            # Convert the NumPy array back to a PyTorch tensor
            edges_tensor = torch.from_numpy(edges).float().unsqueeze(0).unsqueeze(0).to(images.device)

            # Repeat the single-channel tensor to make it three channels
            #edges_tensor = torch.cat([edges_tensor, edges_tensor, edges_tensor], dim=1)
            edges_tensor = torch.cat([edges_tensor] * channels, dim=1)
            edges_batch.append(edges_tensor)

        # Stack the results along the batch dimension
        edges_batch = torch.cat(edges_batch, dim=0)
        #end_time = time.time()
        #elapsed_time = end_time - start_time
        #print(f"Time taken for {batch_size} images: {elapsed_time} seconds")


        return edges_batch


    def gaussian_blur(self, image, kernel_size, sigma=0):
        # Define the variance
        if sigma == 0:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        # Define the filter
        gaussian_kernel = self.gaussian_kernel_2d(kernel_size,sigma)
        # Convolution with the filter
        blurred_image = cv2.filter2D(src=image, ddepth=-1, kernel=gaussian_kernel)
        return blurred_image

    def gaussian_kernel_2d(self, kernel_size, sigma):
        # Generate random points from the 2D Gaussian distribution
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (kernel_size, kernel_size)
        )
        # Normalize the kernel weights
        kernel /= np.sum(kernel)
        return kernel

    def scharr_filters(self, image, kernel_size, L2_gradient=False):
        scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

        if L2_gradient:
            gradient_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
            gradient_direction = np.arctan2(scharr_y, scharr_x)
        else:
            gradient_magnitude = np.abs(scharr_x) + np.abs(scharr_y)
            gradient_direction = np.arctan2(scharr_y, scharr_x)

        return gradient_magnitude, gradient_direction


    def non_max_suppression(self, gradient_magnitude, gradient_direction):
        #print(gradient_magnitude.shape)
        rows, cols = gradient_magnitude.shape
        result = np.zeros_like(gradient_magnitude)

        for i in range(1, rows - 1):
          for j in range(1, cols - 1):
            angle = gradient_direction[i, j]
            q = gradient_magnitude[i, j]
            p1, p2 = 0, 0

            if (0 <= angle < np.pi / 8) or (15 * np.pi / 8 <= angle <= 2 * np.pi):
                p1 = gradient_magnitude[i, j + 1]
                p2 = gradient_magnitude[i, j - 1]
            elif (np.pi / 8 <= angle < 3 * np.pi / 8) or (9 * np.pi / 8 <= angle < 11 * np.pi / 8):
                p1 = gradient_magnitude[i + 1, j - 1]
                p2 = gradient_magnitude[i - 1, j + 1]
            elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8) or (11 * np.pi / 8 <= angle < 13 * np.pi / 8):
                p1 = gradient_magnitude[i + 1, j]
                p2 = gradient_magnitude[i - 1, j]
            elif (5 * np.pi / 8 <= angle < 7 * np.pi / 8) or (13 * np.pi / 8 <= angle < 15 * np.pi / 8):
                p1 = gradient_magnitude[i - 1, j - 1]
                p2 = gradient_magnitude[i + 1, j + 1]

            if q >= p1 and q >= p2:
                result[i, j] = q

        return result


    def double_thresholding(self, image, low_threshold, high_threshold):
        strong_edges = (image >= high_threshold)
        weak_edges = (image >= low_threshold) & (image < high_threshold)
        return strong_edges, weak_edges

    def edge_tracking_by_hysteresis(self, strong_edges, weak_edges):
        rows, cols = strong_edges.shape
        result = np.zeros((rows, cols), dtype=np.uint8)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if strong_edges[i, j]:
                    result[i, j] = 255
                elif weak_edges[i, j]:
                    if (strong_edges[i-1:i+2, j-1:j+2] == 255).any():
                        result[i, j] = 255

        return result




###################################################################
####################-------- Gaussian module----------#########################
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create a Gaussian kernel
        self.create_gaussian_kernel()

    def create_gaussian_kernel(self):
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        center = self.kernel_size // 2
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                x, y = i - center, j - center
                x, y = torch.tensor(x), torch.tensor(y)  # Convert x and y to tensors
                kernel[0, 0, i, j] = torch.exp(-(x**2 + y**2) / (2.0 * self.sigma**2))
        kernel /= kernel.sum()
        self.register_buffer('gaussian_kernel', kernel)


    def forward(self, x):
        # Get the number of channels in the input
        num_channels = x.size(1)

        # Apply Gaussian blur to each channel separately
        blurred_channels = []
        for i in range(num_channels):
            channel = x[:, i:i+1, :, :]
            blurred_channel = F.conv2d(channel, self.gaussian_kernel, padding=self.kernel_size//2)
            blurred_channels.append(blurred_channel)
        
        # Concatenate the blurred channels to form the output
        blurred_output = torch.cat(blurred_channels, dim=1)

        return blurred_output

##############################################################################


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            core = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")
        elif paddle:  # PaddlePaddle
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi

            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix(".pdiparams")
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")
            check_requirements("tritonclient[all]")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info("Adding AutoShape... ")
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != "cpu")  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f"image{i}"  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
                files.append(Path(f).with_suffix(".jpg").name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(
                    y if self.dmb else y[0],
                    self.conf,
                    self.iou,
                    self.classes,
                    self.agnostic,
                    self.multi_label,
                    max_det=self.max_det,
                )  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
        s, crops = "", []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(", ")
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                            crops.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "cls": cls,
                                    "label": label,
                                    "im": save_one_box(box, im, file=file, save=save),
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label if labels else "", color=colors(cls))
                    im = annotator.im
            else:
                s += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display

                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip("\n")
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops

    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        return [
            Detections(
                [self.ims[i]],
                [self.pred[i]],
                [self.files[i]],
                self.times,
                self.names,
                self.s,
            )
            for i in r
        ]

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f"YOLOv5 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
