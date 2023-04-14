"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from manu_data import part2attr_np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import torch
from torchvision import models
from .blocks import conv1x1

LABEL_EMB_NUM = 512
ATT_EMB_NUM = 512
BETA = 1.0

class Conv_ImgEncoder(nn.Module):
    def __init__(self, opt):
        super(Conv_ImgEncoder, self).__init__()
        kw = 3
        self.opt = opt
        self.semantic_nc = opt.label_nc
        self.split_num = opt.split_num
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = self.split_num * 4
        norm_type = 'spectralinstance'
        norm_layer = get_nonspade_norm_layer(norm_type)
        self.layer1 = norm_layer(
            nn.Conv2d(self.split_num * 3, ndf * 2, kw, stride=2, padding=pw, groups=self.split_num))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw, groups=self.split_num))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw, groups=self.split_num))
        if opt.crop_size == 64:
            stride_list = [1, 1, 1]
        elif opt.crop_size == 256:
            if opt.dataset_mode == 'landscape' or opt.dataset_mode == 'traffic':
                stride_list = [2, 2, 1]
            else:
                stride_list = [2, 2, 2]
        elif opt.crop_size == 1024 or opt.crop_size == 512:
            stride_list = [2, 2, 2]
        else:
            raise NotImplementedError
        self.layer4 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=stride_list[0], padding=pw, groups=self.split_num))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 16, ndf * 32, kw, stride=stride_list[1], padding=pw, groups=self.split_num))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 32, ndf * 64, kw, stride=stride_list[2], padding=pw, groups=self.split_num))
        if opt.dataset_mode == 'landscape' or opt.dataset_mode == 'traffic':
            if opt.crop_size == 256:
                self.grain_layer1 = conv1x1(ndf * 16, ndf * 64)
                self.grain_layer2 = conv1x1(ndf * 8, ndf * 64)
            elif opt.crop_size == 512:
                self.grain_layer1 = conv1x1(ndf * 32, ndf * 64)
                self.grain_layer2 = conv1x1(ndf * 16, ndf * 64)
        else:
            self.grain_layer1 = conv1x1(ndf * 32, ndf * 64)
            self.grain_layer2 = conv1x1(ndf * 16, ndf * 64)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, real_image, input_semantic):
        images = self.trans_img(input_semantic, real_image)
        # images: [bs, 3 * 11, 512, 256]
        # [bs, 3 * 11, 512, 256]--> [bs, ndf, 256, 128]
        if self.opt.dataset_mode == 'vip':
            x = self.layer1(images)
            # --> [bs, ndf * 2, 128, 64]
            x = self.layer2(self.actvn(x))
            # --> [bs, ndf * 4, 64, 32]
            x = self.layer3(self.actvn(x))
            # --> [bs, ndf * 8, 32, 16]
            last_2 = self.layer4(self.actvn(x))
            # --> [bs, ndf * 8, 16, 8]
            last = self.layer5(self.actvn(last_2))
            # --> [bs, ndf * 8, 8, 4]
            x = self.layer6(self.actvn(last))
            x = self.actvn(x)
            # [bs, 256 * 11, 8, 4]
            x_1 = self.grain_layer1(self.actvn(last))
            x_1 = self.actvn(x_1)
            x_2 = self.grain_layer2(self.actvn(last_2))
            x_2 = self.actvn(x_2)
            return [x, x_1, x_2], images

        elif self.opt.dataset_mode == 'landscape' or self.opt.dataset_mode == 'traffic':
            x = self.layer1(images)
            # --> [bs, ndf * 2, 128, 64]
            x = self.layer2(self.actvn(x))
            # --> [bs, ndf * 4, 64, 32]
            out_8 = self.layer3(self.actvn(x))
            # --> [bs, ndf * 8, 32, 16]
            out_16 = self.layer4(self.actvn(out_8))
            # --> [bs, ndf * 8, 16, 8]
            out_32 = self.layer5(self.actvn(out_16))
            # --> [bs, ndf * 8, 8, 4]
            x = self.layer6(self.actvn(out_32))
            x = self.actvn(x)
            # [bs, 256 * 11, 8, 4]
            if self.opt.crop_size == 256:
                x_1 = self.grain_layer1(self.actvn(out_16))
                x_1 = self.actvn(x_1)
                x_2 = self.grain_layer2(self.actvn(out_8))
                x_2 = self.actvn(x_2)
                return [x, x_1, x_2], images
            elif self.opt.crop_size == 512:
                x_1 = self.grain_layer1(self.actvn(out_32))
                x_1 = self.actvn(x_1)
                x_2 = self.grain_layer2(self.actvn(out_16))
                x_2 = self.actvn(x_2)
                return [x, x_1, x_2], images

    def trans_img(self, input_semantic, real_image):
        # print('seg:', input_semantic.size())  # [b_size, 19, 512, 256]
        # print('img: ', real_image.size())  # [b_size, 3, 512, 256]
        # input_semantic --> [b_size, 11, 512, 256]
        batch_size = input_semantic.size(0)
        ih, iw = input_semantic.size(2), input_semantic.size(3)
        if self.opt.dataset_mode != 'landscape' and self.opt.dataset_mode != 'traffic':
            sourceL = ih * iw
            seg_mask = input_semantic.view(batch_size, self.semantic_nc, sourceL)
            max_seg = torch.zeros(batch_size, sourceL)
            for i in range(self.semantic_nc):
                nonzero = torch.nonzero(seg_mask[:, i, :])
                max_seg[nonzero[:, 0], nonzero[:, 1]] = i + 1
            max_seg = max_seg.reshape(-1).int()
            part2attr_mask = torch.from_numpy(part2attr_np[max_seg, :]).cuda()
            part2attr_mask = part2attr_mask.float()
            # --> batch x sourceL x 11
            part2attr_mask = part2attr_mask.view(batch_size, sourceL, self.split_num)
            # --> batch * 11 * sourceL --> [batch, 11, 512, 256]
            part2attr_mask = torch.transpose(part2attr_mask, 1, 2).contiguous()
            part2attr_mask = part2attr_mask.view(batch_size, self.split_num, ih, iw)    # [bs, 11, h, w]
        else:
            part2attr_mask = input_semantic
        images = None
        seg_range = self.split_num
        for i in range(batch_size):
            resize_image = None
            for n in range(0, seg_range):
                # --> [3, 512, 256]
                seg_image = real_image[i] * part2attr_mask[i][n]
                # resize seg_image --> [512, 256]
                c_sum = seg_image.sum(dim=0)        # [512 ,256]
                # --> [256]
                y_seg = c_sum.sum(dim=0)
                # -> [512]
                x_seg = c_sum.sum(dim=1)
                y_id = y_seg.nonzero()
                x_id = x_seg.nonzero()
                if y_id.size()[0] == 0 or x_id.size()[0] == 0:
                    seg_image = seg_image.unsqueeze(dim=0)
                    if resize_image is None:
                        resize_image = seg_image
                    else:
                        resize_image = torch.cat((resize_image, seg_image), dim=1)
                    continue
                y_min = y_id[0][0]
                y_max = y_id[-1][0]
                x_min = x_id[0][0]
                x_max = x_id[-1][0]
                seg_image = seg_image.unsqueeze(dim=0)
                seg_image = F.interpolate(seg_image[:, :, x_min:x_max + 1, y_min:y_max + 1], size=[ih, iw])
                # resize_image: --> [1, 11, 512, 256]
                if resize_image is None:
                    resize_image = seg_image
                else:
                    resize_image = torch.cat((resize_image, seg_image), dim=1)
            if images is None:
                images = resize_image
            else:
                images = torch.cat((images, resize_image), dim=0)
        # print('image:', images.size())      # [b_size, 3 * 11, 512, 256]
        return images

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu


###################### Label Encoder #####################################
class LabelEncoder(nn.Module):
    def __init__(self, opt):
        super(LabelEncoder, self).__init__()
        self.attr_num = opt.attr_num        # 120
        self.split_num = opt.split_num              # 11
        self.label_emb_num = LABEL_EMB_NUM        # 512
        self.attr_emb_num = ATT_EMB_NUM       # 512
        self.num_grained = opt.num_grained  # 3
        self.opt = opt

        self.define_module()
        self.beta = BETA

    def define_single_block(self, out_dim):
        self.s_emb_num = self.attr_emb_num + self.label_emb_num
        label_embedding = nn.Embedding(self.split_num, out_dim)
        attr_embedding = nn.Linear(self.attr_num, out_dim)
        hidden_layers = nn.Sequential(
            nn.Linear(self.s_emb_num, self.s_emb_num // 2),
            nn.ReLU(),
            nn.Linear(self.s_emb_num//2, self.s_emb_num//4),
        )
        initrange = 0.1
        label_embedding.weight.data.uniform_(-initrange, initrange)
        attr_embedding.weight.data.uniform_(-initrange, initrange)
        for layer in hidden_layers:
            if not isinstance(layer, nn.Linear): continue
            layer.weight.data.uniform_(-initrange, initrange)
        return label_embedding, attr_embedding, hidden_layers

    def define_module(self):
        self.label_embedding1, self.attr_embedding1, self.hidden_layers1 = self.define_single_block(self.label_emb_num)
        self.label_embedding2, self.attr_embedding2, self.hidden_layers2 = self.define_single_block(self.label_emb_num)
        self.label_embedding3, self.attr_embedding3, self.hidden_layers3 = self.define_single_block(self.label_emb_num)

    def forward(self, label, attr):
        batch_size = attr.size(0)
        attr = attr.view(batch_size * self.split_num, self.attr_num)

        concat_emb_list = []
        for n in range(1, self.num_grained+1):
            label_emb_layer = getattr(self, 'label_embedding%s'% n)
            attr_emb_layer = getattr(self, 'attr_embedding%s' % n)
            multi_hidden_layer = getattr(self, 'hidden_layers%s' % n)
            label_emb = label_emb_layer(label)
            label_emb = label_emb.expand((batch_size, self.split_num, self.label_emb_num)).contiguous()
            label_emb = label_emb.view(batch_size * self.split_num, self.label_emb_num)
            attr_emb = attr_emb_layer(attr)
            concat_emb = torch.cat([label_emb, self.beta * attr_emb], 1)
            for layer in multi_hidden_layer:
                concat_emb = layer(concat_emb)
            concat_emb = concat_emb.view(batch_size, self.split_num, self.s_emb_num//4)
            global_emb = concat_emb.mean(1)
            concat_emb = concat_emb.permute((0, 2, 1))      # [bs, 256, 11]
            concat_emb_list.append(concat_emb)
        # concat_emb_list: [[bs, 256, 11], [bs, 256, 11], [bs, 256, 11]]
        for i in range(self.num_grained):
            if i >= 1:
                concat_emb_list[i] += concat_emb_list[i - 1]
        return concat_emb_list, global_emb



################# CNN encoder (Inception-Net)##############################
class CNN_ENCODER(nn.Module):
    def __init__(self, incep_state_dict, opt):
        super(CNN_ENCODER, self).__init__()
        self.nef = opt.text_embedding_dim      # 256
        model = models.inception_v3()
        model.load_state_dict(incep_state_dict)
        for param in model.parameters():
            param.requires_grad = False

        self.num_grained = opt.num_grained
        self.define_module(model)
        self.init_trainable_weights()


    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)
        self.emb_features_3 = conv1x1(288, self.nef)
        self.emb_features_5 = conv1x1(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)
        self.emb_features_3.weight.data.uniform_(-initrange, initrange)
        self.emb_features_5.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, input_semantic=None):
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x_1 = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64

        x = F.max_pool2d(x_1, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x_2 = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x_2, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x_3 = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x_3)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x_5 = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x_5, kernel_size=8)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        grained_4 = self.emb_features_5(x_5)
        grained_3 = self.emb_features(features)     # 17 * 17
        grained_2 = self.emb_features_3(x_3)
        return [grained_4, grained_3, grained_2], cnn_code





