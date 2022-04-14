"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from manu_data import part2attr_np, part2attr_dict
from torch.autograd import Variable
from .blocks import conv1x1


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(norm_type='instance'):
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, opt):
        super().__init__()
        self.opt = opt
        self.label_nc = opt.label_nc
        self.norm_nc = norm_nc

        param_free_norm_type = self.opt.norm_G      # batch_norm + word_loss  or syncBN (delete word_loss)
        self.style_layer = nn.Linear(self.opt.z_dim, norm_nc * 2)

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        nhidden = 128
        self.in_mlp_dim = self.label_nc + 256

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.in_mlp_dim, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap, style=None):
        # print('x', x.size())        # [1, 1024, 8, 4]
        # print('segmap:', segmap.size())     # [1, 275, 8, 4]
        if x is not None:
            # Part 1. generate parameter-free normalized activations
            normalized = self.param_free_norm(x)

        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        style = self.style_layer(style).unsqueeze(2).unsqueeze(3)
        gamma_style, beta_style = style.chunk(2, 1)

        if x is not None:
            out = normalized * (1+ gamma + gamma_style) + beta + beta_style
            return out
        else:
            # for SEAN
            gamma = gamma + gamma_style
            beta = beta + beta_style
            return gamma, beta


class GROUP_SPADE(nn.Module):
    def __init__(self, opt, norm_nc, split_num, group_num=0):
        super(GROUP_SPADE, self).__init__()
        self.opt = opt
        if group_num == 0:
            group_num = split_num

        param_free_norm_type = self.opt.norm_G
        self.style_layer = nn.Linear(self.opt.z_dim, norm_nc * 2)

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'group':
            self.param_free_norm = nn.GroupNorm(split_num, norm_nc)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        nhidden = split_num * group_num
        self.label_nc = split_num        #  11
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(split_num, nhidden, kernel_size=3, padding=1, groups=split_num),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1, groups=group_num)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1, groups=group_num)

    def forward(self, x, segmap, style):
        """x: [1, 11 * 16, 8, 4]   segmap:  [1, 11, 8, 4]"""
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        style = self.style_layer(style).unsqueeze(2).unsqueeze(3)
        gamma_style, beta_style = style.chunk(2, 1)

        out = normalized * (1 + gamma + gamma_style) + beta + beta_style
        return out


class ACE(nn.Module):
    def __init__(self, opt, norm_nc, spade_params, status='train'):
        super(ACE, self).__init__()
        self.status = status
        self.opt = opt
        self.style_length = 256
        self.nef = 256          # text embedding dim
        self.Spade = SPADE(*spade_params)
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

        param_free_norm_type = self.opt.norm_G  # batch_norm + word_loss  or syncBN (delete word_loss)
        self.style_layer = nn.Linear(self.opt.z_dim, norm_nc * 2)

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap, att_emb, style):
        # attr_emb: [b_size, 11, 256]
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        att_emb = F.interpolate(att_emb, size=x.size()[2:], mode='bilinear')
        gamma_avg = self.conv_gamma(att_emb)
        beta_avg = self.conv_beta(att_emb)

        gamma_spade, beta_spade = self.Spade(None, segmap, style=None)
        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_avg + (1- gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade

        # style = style.unsqueeze(2).unsqueeze(3)
        style = self.style_layer(style).unsqueeze(2).unsqueeze(3)
        gamma_style, beta_style = style.chunk(2, 1)
        out = normalized * (1 + gamma_final + gamma_style) + beta_final + beta_style

        return out

