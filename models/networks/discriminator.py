"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from manu_data import split_max_num
import torch
from .blocks import Block3x3_leakRelu
from torch.nn.utils.spectral_norm import spectral_norm

##################### New Discriminator #####################
class D_GET_LOGITS(BaseNetwork):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            h_c_code = torch.cat((h_code, c_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output


class PAT_D_NET256(BaseNetwork):
    def __init__(self, opt, b_jcu=True):
        super(PAT_D_NET256, self).__init__()
        ndf = opt.ngf
        nef = opt.text_embedding_dim
        self.opt = opt
        self.conv_layer_0 = nn.Sequential(nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                                           nn.LeakyReLU(0.2, inplace=True))
        for n in range(1, 4):
            nf_mult_prev = ndf * min(2 ** (n-1), 8)
            nf_mult = ndf * min(2 ** n, 8)
            self.add_module('conv_layer_' + str(n), nn.Sequential(nn.Conv2d(nf_mult_prev, nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                                                                  nn.BatchNorm2d(nf_mult),
                                                                  nn.LeakyReLU(0.2, inplace=True)))
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef*split_max_num, bcondition=True)

    def forward(self, x_var):
        result = [x_var]
        for i in range(4):
            conv_layer = getattr(self, 'conv_layer_'+ str(i))
            intermediate = conv_layer(result[-1])
            result.append(intermediate)
        if not self.opt.no_ganFeat_loss:
            return result[1:]
        else:
            return result[-1]


class SEG_D_NET(BaseNetwork):
    def __init__(self, opt):
        super(SEG_D_NET, self).__init__()
        ndf = opt.ngf
        self.conv_layer_0 = nn.Sequential(spectral_norm(nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
                                          nn.LeakyReLU(0.2, inplace=True))
        self.conv_layer_1 = nn.Sequential(spectral_norm(nn.Conv2d(ndf, 2*ndf, kernel_size=4, stride=2, padding=1, bias=False)),
                                          nn.BatchNorm2d(2 * ndf),
                                          nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, 1, 3, padding=1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv(x)
        # [bs, 1, h//4, w//4]
        return x