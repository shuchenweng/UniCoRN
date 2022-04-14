"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
from torch.autograd import Variable
import os
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils import model_zoo
from models.networks.word_losses import words_loss
from manu_data import vip_split_attr
from manu_data_flickr import land_split_attr as split_attr_flickr
import torch.nn.functional as F
import pickle


def get_pt_version():
    raw_version = torch.__version__
    raw_version = raw_version.split('.')
    # raw_version = [int(raw_version[i])*pow(10, len(raw_version)-i-1) for i in range(len(raw_version))]
    raw_version = [int(raw_version[i])*pow(10, 3-i-1) for i in range(2)]
    version = sum(raw_version)
    return version


def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return n_params.item()


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.PT_VERSION = get_pt_version()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netSegD, self.netPatD, self.text_encoder, self.image_encoder = self.initialize_networks(opt)
        if opt.isTrain:
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(opt.gpu_ids, opt)

    def prepare_labels(self, batch_size):
        label_tensor = torch.Tensor(range(self.opt.split_num)).long().cuda()
        match_labels = Variable(torch.LongTensor(range(batch_size))).cuda()
        if self.opt.dataset_mode == 'landscape':
            match_labels = None
        return label_tensor, match_labels

    def forward(self, data, mode, correct=0, total_m=0, is_val=False):
        input_semantics, real_image, key, acts, att, class_id, b_tensor, attr_relist = self.preprocess_input(data)
        batch_size = input_semantics.size(0)
        label_tensor, match_labels = self.prepare_labels(batch_size)
        if mode == 'generator':
            with torch.no_grad():
                att_emb, _ = self.text_encoder(label_tensor, att)
                # att_emb is a list
                for i in range(len(att_emb)):
                    att_emb[i] = att_emb[i].detach()

            g_loss, generated, generated_back = self.compute_generator_loss(
                input_semantics, real_image, class_id, match_labels, att_emb, b_tensor, attr_relist)
            return g_loss, generated, generated_back

        elif mode == 'discriminator':
            with torch.no_grad():
                att_emb, _ = self.text_encoder(label_tensor, att)
                for i in range(len(att_emb)):
                    att_emb[i] = att_emb[i].detach()

            d_loss = self.compute_discriminator_loss(input_semantics, real_image, att_emb, b_tensor)
            return d_loss

        elif mode == 'inference':
            with torch.no_grad():
                att_emb, _ = self.text_encoder(label_tensor, att)
                for i in range(len(att_emb)):
                    att_emb[i] = att_emb[i].detach()

                fake_image, fake_image_background = self.generate_fake(input_semantics, real_image, att_emb, b_tensor, is_val=True)
                # add word_acc calculation.
                region_features, _ = self.image_encoder(fake_image_background, input_semantics)
                for i in range(len(region_features)):
                    region_features[i] = region_features[i].detach()

                if self.opt.dataset_mode == 'landscape':
                    word_num = len(split_attr_flickr)
                else:
                    word_num = len(vip_split_attr)
                _, _, _, correct, total_m = words_loss(region_features, input_semantics, att_emb, match_labels, word_num, class_id, correct, self.opt, attr_relist, total_m)

                return fake_image, acts, fake_image_background, correct, total_m
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        # define optimizer
        G_params = list(self.netG.parameters())
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        if opt.isTrain:
            D_params1 = list(self.netSegD.parameters())
            D_params2 = list(self.netPatD.parameters())
            optimizer_DSeg = torch.optim.Adam(D_params1, lr=D_lr, betas=(beta1, beta2))
            optimizer_DPat = torch.optim.Adam(D_params2, lr=D_lr, betas=(beta1, beta2))
            return optimizer_G, optimizer_DSeg, optimizer_DPat

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netPatD, 'patD', epoch, self.opt)
        util.save_network(self.netSegD, 'segD', epoch, self.opt)
        print('save models....')

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        if opt.isTrain:
            netSegD, netPatD = netD[0], netD[1]
        else:
            netSegD, netPatD = None, None
        if self.opt.load_G:
            load_path = os.path.join('model_weights', 'vip', 'latest_vip_pixel-wise_512_sota.pth')
            weights = torch.load(load_path, map_location=lambda storage, loc: storage)
            netG.load_state_dict(weights, strict=False)
            print('loadG:', load_path)
            # continue training
            save_patD = os.path.join('model_weights', 'vip', 'net_patD.pth')
            patd_weights = torch.load(save_patD, map_location=lambda storage, loc: storage)
            netPatD.load_state_dict(patd_weights, strict=False)
            save_segD = os.path.join('model_weights', 'vip', 'net_segD.pth')
            segd_weights = torch.load(save_segD, map_location=lambda storage, loc: storage)
            netSegD.load_state_dict(segd_weights, strict=False)

        ######### define text/image encoder ##############
        text_encoder = networks.LabelEncoder(opt).cuda()
        image_encoder = networks.Conv_ImgEncoder(opt).cuda()

        if opt.dataset_mode == 'landscape':
            if opt.crop_size == 256:
                pretrained_dir = os.path.join(opt.pretrained_dir, 'landscape-256', 'text_encoder_256_split_image_multi_True.pth')
                pretrained_dir_img = os.path.join(opt.pretrained_dir, 'landscape-256', 'image_encoder_256_split_image_multi_True.pth')
            elif opt.crop_size == 512:
                pretrained_dir = os.path.join(opt.pretrained_dir, 'landscape-512', 'text_encoder_512_multi_True_split_image.pth')
                pretrained_dir_img = os.path.join(opt.pretrained_dir, 'landscape-512', 'image_encoder_512_multi_True_split_image.pth')

        elif opt.dataset_mode == 'vip':
            pretrained_dir = os.path.join(opt.pretrained_dir, 'vip', 'text_encoder_256_data_vip_multi_True_split_image.pth')
            pretrained_dir_img = os.path.join(opt.pretrained_dir, 'vip', 'image_encoder_256_data_vip_multi_True_split_image.pth')

        state_dict = torch.load(pretrained_dir, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        state_dict_image = torch.load(pretrained_dir_img, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict_image)
        print('load pretrained text encoder from:', pretrained_dir)
        print('load pretrained image encoder from:', pretrained_dir_img)

        for p in image_encoder.parameters():
            p.requires_grad = False
        for p in text_encoder.parameters():
            p.requires_grad = False
        image_encoder.eval()
        text_encoder.eval()
        return netG, netSegD, netPatD, text_encoder, image_encoder

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['key'] = data['key']
            data['acts'] = data['acts'].cuda()
            data['att'] = data['att'].cuda()
            if self.opt.dataset_mode == 'landscape':
                data['class_id'] = None
            elif self.opt.dataset_mode == 'vip':
                data['class_id'] = data['class_id'].cpu().numpy()
            data['background'] = data['background'].cuda()
        # create one-hot label map
        input_semantics = data['label'].float()
        if self.opt.dataset_mode == 'landscape':
            # only 1 class set as foreground, others background.
            batch_size = data['att'].shape[0]
            reshape_att = data['att'].view(batch_size, -1)  # [bs, 7*70]
            index = torch.nonzero(reshape_att)
            if len(index) != batch_size:
                print('index != bs')
                raise NotImplementedError
            attr_relist = index[:, 1]  # len[] = batchsize
        else:
            attr_relist = []
        return input_semantics, data['image'], data['key'], data['acts'], data['att'], data['class_id'], \
               data['background'], attr_relist

    def compute_generator_loss(self, input_semantics, real_image, class_ids, match_label, att_emb=None, b_tensor=None, attr_relist=None):
        G_losses = {}
        fake_image, fake_image_background = self.generate_fake(input_semantics, real_image, att_emb, b_tensor)
        if torch.isnan(fake_image).sum() >0:
            print('nan')
            exit()
        if b_tensor is not None:
            mask = b_tensor[:, -1, :, :].unsqueeze(1)
            # mask: [bs, 1, 256, 128], people part is 1, background part is 0.
            if self.PT_VERSION >= 120:
                fake_background = fake_image.masked_fill(mask.bool(), 0)
                real_background = real_image.masked_fill(mask.bool(), 0)
            else:
                fake_background = fake_image.masked_fill(mask.byte(), 0)
                real_background = real_image.masked_fill(mask.byte(), 0)
            G_losses['back_L1'] = self.criterionFeat(fake_background, real_background.detach()) * self.opt.back_l1_weight
            # print('back_L1', G_losses['back_L1'])

        # add word_loss
        region_features, _ = self.image_encoder(fake_image, input_semantics)
        if self.opt.dataset_mode == 'landscape':
            word_num = len(split_attr_flickr)
        else:
            word_num = len(vip_split_attr)
        w_loss0, w_loss1, _, _, _= words_loss(region_features, input_semantics, att_emb, match_label, word_num, class_ids, 0, self.opt, attr_relist, 0)
        w_loss = (w_loss0 + w_loss1) * self.opt.word_loss_weight
        # print('w_loss', w_loss)
        G_losses['w_loss'] = w_loss

        # add vgg-loss
        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg + \
                              self.criterionVGG(fake_image_background, real_image) * self.opt.lambda_vgg
        # add Generating loss...
        condition = util.prepare_condition(self.opt, self.netG.proj, att_emb, input_semantics)      # [bs, 256, h, w]
        G_losses = networks.G_loss(self.opt, self.netPatD, self.netSegD, real_image, fake_image, fake_image_background, condition, G_losses)

        return G_losses, fake_image, fake_image_background

    def compute_discriminator_loss(self, input_semantics, real_image, att_emb=None, b_tensor=None):
        D_losses = {}
        with torch.no_grad():
            fake_image, fake_image_background= self.generate_fake(input_semantics, real_image, att_emb, b_tensor)
            fake_image = fake_image.detach()
            fake_image_background = fake_image_background.detach()

        condition = util.prepare_condition(self.opt, self.netG.proj, att_emb, input_semantics)      # [bs, 256, h, w]
        single_pat, double_pat = networks.patD_loss(self.opt, self.netPatD, real_image, fake_image, fake_image_background, condition)
        D_losses['single_pat_D'], D_losses['double_pat_D'] = single_pat, double_pat
        if self.opt.seg_loss:
            errSegD = networks.segD_loss(self.netSegD, fake_image_background, input_semantics, self.opt)
            D_losses['seg_D'] = errSegD

        return D_losses

    def generate_fake(self, input_semantics, real_image, att_emb, b_tensor, is_val=False):
        z, fixed_z = None, None

        z = Variable(torch.FloatTensor(self.opt.batchSize, self.opt.z_dim))
        fixed_z = Variable(torch.FloatTensor(self.opt.test_bs, self.opt.z_dim).normal_(0, 1))     #test_bs = 6
        z, fixed_z = z.cuda(), fixed_z.cuda()
        z.data.normal_(0, 1)

        if is_val is False:
            # training...
            fake_image = self.netG(input_semantics, z=z[:input_semantics.shape[0]], att_emb=att_emb, b_tensor=b_tensor)
        else:
            # test...
            fake_image = self.netG(input_semantics, z=fixed_z[:input_semantics.shape[0]], att_emb=att_emb, b_tensor=b_tensor)

        # paste foreground into real background .
        mask = torch.max(input_semantics, dim=1)[0]
        back_index = (mask == 0).unsqueeze(1).expand_as(real_image)
        front_index = (mask == 1).unsqueeze(1).expand_as(real_image)
        fake_image_background = torch.zeros(fake_image.shape).cuda()
        fake_image_background[back_index] = real_image[back_index]
        fake_image_background[front_index] = fake_image[front_index]

        return fake_image, fake_image_background

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
