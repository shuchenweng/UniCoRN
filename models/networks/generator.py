"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from manu_data import part2attr_np
from .blocks import conv1x1, conv3x3, PixelNorm
from torch.autograd import Variable


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
        from models.networks.architecture import SPADEV2ResnetBlock as SPADEV2ResnetBlock
        self.opt = opt
        nf = opt.ngf
        self.cdf = 256
        self.label_nc = self.opt.label_nc
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc_in_dim = self.cdf + self.opt.label_nc
        self.out_dim = self.cdf
        self.output_mask_dim = self.cdf + self.opt.label_nc
        self.output_block_dim = self.cdf + self.opt.label_nc

        final_nc = nf

        self.h_net_att = MultiHeadAttention(opt.label_nc, self.out_dim, n_heads=opt.num_grained, opt=self.opt)
        self.proj = nn.Linear(opt.num_grained * opt.text_embedding_dim, opt.text_embedding_dim)
        self.fc = nn.Conv2d(self.fc_in_dim, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        if self.opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1*nf, nf//2, opt)
            final_nc = nf // 2
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, self.output_block_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.output_block_dim),
            nn.LeakyReLU(0.2, True))
        self.conv_mask_256 = nn.Conv2d(self.label_nc, self.output_mask_dim, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            self.num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            self.num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            self.num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)
        sw = opt.crop_size // (2**self.num_up_layers)        # (128, 64)--> (4, 2)// (512, 256)--> (16, 8)
        sh = round(sw / opt.aspect_ratio)   # aspect_ratio = 0.5        # 16
        return sw, sh

    def middle_back(self, x, seg, b_tensor, ori_seg=None, style=None):
        f_feature = F.interpolate(seg, size=(x.shape[2], x.shape[3]), mode='nearest')  # [bs, 256+19, 8, 4]
        shared_b_feature = self.conv_block_1(b_tensor[:, :3, :, :])
        shared_mask = self.conv_mask_256(ori_seg)
        b_feature = F.interpolate(shared_b_feature, size=x.size()[2:], mode='bilinear')
        mask = F.interpolate(shared_mask, size=x.size()[2:], mode='nearest')
        segmap = self.sigmoid(mask) * f_feature + (1 - self.sigmoid(mask)) * b_feature
        x = self.head_0(x, segmap, style=style)
        x = self.up(x)

        f_feature = F.interpolate(seg, size=(x.shape[2], x.shape[3]), mode='nearest')
        b_feature = F.interpolate(shared_b_feature, size=x.size()[2:], mode='bilinear')
        mask = F.interpolate(shared_mask, size=x.size()[2:], mode='nearest')
        segmap = self.sigmoid(mask) * f_feature + (1 - self.sigmoid(mask)) * b_feature
        x = self.G_middle_0(x, segmap, style=style)

        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            f_feature = F.interpolate(seg, size=x.size()[2:], mode='nearest')
            b_feature = F.interpolate(shared_b_feature, size=x.size()[2:], mode='bilinear')
            mask = F.interpolate(shared_mask, size=x.size()[2:], mode='nearest')
            segmap = self.sigmoid(mask) * f_feature + (1 - self.sigmoid(mask)) * b_feature
        x = self.G_middle_1(x, segmap, style=style)
        x = self.up(x)

        f_feature = F.interpolate(seg, size=(x.shape[2], x.shape[3]), mode='nearest')
        b_feature = F.interpolate(shared_b_feature, size=x.size()[2:], mode='bilinear')
        mask = F.interpolate(shared_mask, size=x.size()[2:], mode='nearest')
        segmap = self.sigmoid(mask) * f_feature + (1 - self.sigmoid(mask)) * b_feature
        x = self.up_0(x, segmap, style=style)
        x = self.up(x)

        f_feature = F.interpolate(seg, size=(x.shape[2], x.shape[3]), mode='nearest')
        b_feature = F.interpolate(shared_b_feature, size=x.size()[2:], mode='bilinear')
        mask = F.interpolate(shared_mask, size=x.size()[2:], mode='nearest')
        segmap = self.sigmoid(mask) * f_feature + (1 - self.sigmoid(mask)) * b_feature
        x = self.up_1(x, segmap, style=style)
        x = self.up(x)

        f_feature = F.interpolate(seg, size=(x.shape[2], x.shape[3]), mode='nearest')
        b_feature = F.interpolate(shared_b_feature, size=x.size()[2:], mode='bilinear')
        mask = F.interpolate(shared_mask, size=x.size()[2:], mode='nearest')
        segmap = self.sigmoid(mask) * f_feature + (1 - self.sigmoid(mask)) * b_feature
        x = self.up_2(x, segmap, style=style)
        x = self.up(x)

        f_feature = F.interpolate(seg, size=(x.shape[2], x.shape[3]), mode='nearest')
        if self.opt.num_upsampling_layers == 'normal' or self.opt.num_upsampling_layers == 'more':
            b_feature = shared_b_feature
            mask = shared_mask
        elif self.opt.num_upsampling_layers == 'most':
            b_feature = F.interpolate(shared_b_feature, size=x.size()[2:], mode='bilinear')
            mask = F.interpolate(shared_mask, size=x.size()[2:], mode='nearest')

        segmap = self.sigmoid(mask) * f_feature + (1 - self.sigmoid(mask)) * b_feature
        x = self.up_3(x, segmap, style=style)
        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            f_feature = F.interpolate(seg, size=x.size()[2:], mode='nearest')
            b_feature = shared_b_feature
            mask = shared_mask
            segmap = self.sigmoid(mask) * f_feature + (1 - self.sigmoid(mask)) * b_feature
            x = self.up_4(x, segmap, style=style)
        return x

    def generate_SEAN(self, x, seg, new_attr_emb, style):
        # seg:[bs, 19, 256, 128], x: [bs, 1024, 8, 4] new_attr_emb:[bs, 256, h, w] (opt.multi_text_grained)
        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        x = self.head_0(x, segmap, att_emb=new_attr_emb, style=style)
        x = self.up(x)
        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        x = self.G_middle_0(x, segmap, att_emb=new_attr_emb, style=style)
        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        x = self.G_middle_1(x, segmap, att_emb=new_attr_emb, style=style)
        x = self.up(x)
        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        x = self.up_0(x, segmap, att_emb=new_attr_emb, style=style)
        x = self.up(x)
        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        x = self.up_1(x, segmap, att_emb=new_attr_emb, style=style)
        x = self.up(x)
        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        x = self.up_2(x, segmap, att_emb=new_attr_emb, style=style)
        x = self.up(x)
        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        x = self.up_3(x, segmap, att_emb=new_attr_emb, style=style)
        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
            x = self.up_4(x, segmap, att_emb=new_attr_emb, style=style)
        return x

    def forward(self, input, z=None, att_emb=None, b_tensor=None):
        """input: [batch_size, semantic_nc, nh, nw]
            att_emb: [batch_size, cdf (256), seq_len (11)]"""
        seg = input
        # att_emb: [batch_size, 256, 11] --> [bs, 256, 11, num_grained]
        new_attr_emb = []
        for i in range(len(att_emb)):
            new_attr_emb.append(att_emb[i].unsqueeze(-1))
        new_attr_emb = torch.cat(new_attr_emb, dim=-1)  # [bs, 256, 11, 2(3)]

        h_code_att, attn = self.h_net_att(seg, new_attr_emb)
        added_emb = torch.cat((seg, h_code_att), dim=1)     # [bs, 256+19, h, w]
        seg = added_emb
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)      # [bs, 1024, 8, 4]
        x = self.middle_back(x, seg, b_tensor, input, z)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x



class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf, opt):
        super(GlobalAttentionGeneral, self).__init__()
        self.idf = idf
        self.cdf = cdf
        self.opt = opt
        self.conv_context = conv1x1(cdf, cdf)

    def forward(self, seg, att_emb):
        """
        :param seg: [batch_size, idf(19), ih, iw (queryL = ih * iw)]
        :param att_emb: [batch_size, cdf (256), sourceL (11)]
        noise_tensor: [bs, 128(cdf), 11] [bs, 2048, 11]
        :return:
        """
        ih, iw = seg.size(2), seg.size(3)
        queryL = ih * iw
        batch_size, sourceL = att_emb.size(0), att_emb.size(2)
        seg = seg.view(batch_size, self.idf, queryL)
        if self.opt.dataset_mode != 'landscape':
            max_seg = torch.zeros([batch_size, queryL])
            for i in range(self.idf):
                nonzero = torch.nonzero(seg[:, i, :])
                max_seg[nonzero[:, 0], nonzero[:, 1]] = i + 1
            max_seg = max_seg.reshape(-1).int()
            # max_seg: [batch_size * queryL]
            part2attr_mask = torch.from_numpy(part2attr_np[max_seg, :]).cuda()
            attn = part2attr_mask.float()
            # attn: --> [batch_size, queryL, sourceL]
            attn = attn.view(batch_size, queryL, sourceL)
            # attn: --> [batch_size, sourceL, queryL] (bs, ih *iw, 11)
            attn = torch.transpose(attn, 1, 2).contiguous()
        else:
            attn = seg

        # att_emb: [batch_size, cdf, sourceL] --> [batch_size, cdf, sourceL, 1]
        sourceT = att_emb.unsqueeze(3)
        # --> [batch_size, cdf, sourceL] (256->256)
        sourceT = self.conv_context(sourceT).squeeze(3)
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, self.cdf, ih, iw)      # [batch_size,cdf (256), ih, iw]
        attn = attn.view(batch_size, -1, ih, iw)
        return weightedContext, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, idf, cdf, n_heads, opt):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.opt = opt
        self.cdf = cdf
        self.idf = idf      # 19
        self.conv_context = nn.ModuleList([conv1x1(cdf, cdf) for _ in range(self.n_heads)])
        self.proj = nn.Linear(self.n_heads * self.cdf, self.cdf)

    def forward(self, seg, attr_emb):
        """
        :param seg:  [bs, idf(19), ih, iw] (queryL = ih * iw)
        :param attr_emb:[bs, cdf , sourceL, num_grained]
        noise_tensor: None or [bs, 19, cdf] 只有在开头注入Noise才会用到noise_tensor.
        :return:
        """
        bs, ih, iw = seg.shape[0], seg.shape[2], seg.shape[3]
        sourceL = attr_emb.shape[2]
        queryL = ih * iw
        seg = seg.view(bs, self.idf, queryL)
        if self.opt.dataset_mode == "vip":
            max_seg = torch.zeros([bs, queryL])
            for i in range(self.idf):
                nonzero = torch.nonzero(seg[:, i, :])
                max_seg[nonzero[:, 0], nonzero[:, 1]] = i + 1
            max_seg = max_seg.reshape(-1).int()
            # max_seg: [bs * queryL]
            part2attr_mask = torch.from_numpy(part2attr_np[max_seg, :]).cuda()
            attn = part2attr_mask.float()
            attn = attn.view(bs, queryL, sourceL)
            attn = torch.transpose(attn, 1, 2).contiguous()
            # attn: [bs, sourceL, queryL]  [bs, 11, ih *iw]
        elif self.opt.dataset_mode == 'landscape':
            attn = seg

        weightedContext_list = []
        for i in range(self.n_heads):
            attr = attr_emb[:, :, :, i]
            attr = attr.unsqueeze(-1)       # [bs, 256, 11, 1]
            sourceT = self.conv_context[i](attr).squeeze(-1)        # [bs, 256, 11]
            # (bs, cdf, sourceL)  [bs, sourceL, queryL] --> (bs, cdf, queryL)
            weightedContext = torch.bmm(sourceT, attn)
            weightedContext = torch.transpose(weightedContext, 1, 2).contiguous()
            # [bs, queryL, cdf]
            weightedContext_list.append(weightedContext)
        # [bs, queryL, cdf * n_heads] --> [bs, queryL, cdf]
        weightedContext = torch.cat(weightedContext_list, dim=-1)
        weightedContext = self.proj(weightedContext)
        weightedContext = weightedContext.transpose(1, 2).contiguous()
        # [bs, cdf, queryL]-->(bs, 256, ih, iw)
        weightedContext = weightedContext.view(bs, self.cdf, ih, iw)        # [bs, 256, ih, iw]
        attn = attn.view(bs, -1, ih, iw)        # [bs, 11, ih, iw]
        return weightedContext, attn

