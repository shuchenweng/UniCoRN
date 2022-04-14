"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, ACE, GROUP_SPADE
from urllib.parse import urlparse
from torch.hub import _download_url_to_file
import torchvision.models as models
import re
import os
import glob


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.status = opt.status        # (train or test)

        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = 'spadesyncbatch3x3'

        self.norm_0 = SPADE(spade_config_str, fin, opt)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt)

    def forward(self, x, seg, style=None, att_emb=None):
        x_s = self.shortcut(x, seg, style, att_emb)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, style=style)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, style=style)))

        out = x_s + dx
        return out

    def shortcut(self, x, seg, style=None, att_emb=None):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, style=style))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADEV2ResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, group_num=8):
        super(SPADEV2ResnetBlock, self).__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1, groups=group_num)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1, groups=group_num)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False, groups=group_num)

        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layer
        self.norm_0 = GROUP_SPADE(opt, fin, self.opt.split_num, group_num)
        self.norm_1 = GROUP_SPADE(opt, fmiddle, self.opt.split_num, group_num)
        if self.learned_shortcut:
            self.norm_s = GROUP_SPADE(opt, fin, self.opt.split_num, group_num)

    def shortcut(self, x, seg, style):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, style))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x, seg, style):
        x_s = self.shortcut(x, seg, style)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, style)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, style)))

        out = x_s + dx
        return out


########## VGG networks ###############################################
def download_model(url, dst_path):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
    hash_prefix = HASH_REGEX.search(filename).group(1)
    # model_zoo._download_url_to_file(url, os.path.join(dst_path, filename), hash_prefix, True)
    torch.hub._download_url_to_file(url, os.path.join(dst_path, filename), hash_prefix, True)
    return filename

def load_model(model_name, model_dir):
    model_urls = {
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    }
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        for url in model_urls.values():
            download_model(url, model_dir)
    model = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % model_name)
    model_path = glob.glob(path_format)[0]
    model.load_state_dict(torch.load(model_path))
    return model


class VGG19(torch.nn.Module):
    def __init__(self, opt, requires_grad=False):
        super().__init__()
        model_dir = os.path.join(opt.pretrained_dir, 'vgg')
        model = load_model('vgg19', model_dir)
        vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
