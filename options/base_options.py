"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os
from util import util
import torch
import models.pix2pix_model
import data
import pickle


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        # datasets
        parser.add_argument('--attr_num', type=int, default=120)
        parser.add_argument('--split_num', type=int, default=11)
        parser.add_argument('--label_nc', type=int, default=19, help='# of input label classes without unknown class')

        parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size') # 384 | 768
        parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size')  # 256 | 512
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height.')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

        parser.add_argument('--dataroot', type=str, default='E:/samsung/datasets/GEN/vip') # LandscapeHD | vip
        parser.add_argument('--pretrained_dir', type=str, default='E:/samsung/pretrained/unicorn/pretrained_weights/')
        parser.add_argument('--checkpoints_dir', type=str, default='E:/samsung/modelsets/unicorn/save_weights', help='models are saved here')
        parser.add_argument('--dataset_mode', type=str, default='vip') # vip | landscape

        # networks
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--text_embedding_dim', type=int, default=256)
        parser.add_argument('--num_grained', type=int, default=3, help='must be 3')  # 3 is best
        parser.add_argument('--norm_G', type=str, default='batch', help='spade batch-norm + word_loss or syncbatch - word_loss')
        # losses
        parser.add_argument('--seg_loss', type=bool, default=True, help='whether add the seg_loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--add_wrong_errD', type=bool, default=True, help='when bs>1, whether add wrong_errD for augmentition')
        parser.add_argument('--patD_loss_weight', type=float, default=4.0, help='weight in PatD-loss')
        parser.add_argument('--segG_loss_weight', type=float, default=0.03, help='seg weight in g_loss')
        parser.add_argument('--word_loss_weight', type=float, default=2.0, help='weight for word loss')
        parser.add_argument('--smooth_gamma2', type=float, default=10.0, help='in word loss')
        parser.add_argument('--smooth_gamma3', type=float, default=5.0, help='in word loss')
        parser.add_argument('--back_l1_weight', type=float, default=1.0, help='background_l1_weight in total_loss')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')

        # training
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--batchSize', type=int, default=6, help='input batch size')
        parser.add_argument('--test_bs', type=int, default=6, help='valid or test batchsize')
        parser.add_argument('--load_G', action='store_true', help='Load pretrain-G net...')

        # noise
        parser.add_argument('--z_dim', type=int, default=2048, help='input noise_dim')       # 1032

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()
        # modify model-related parser options
        model_option_setter = models.pix2pix_model.Pix2PixModel.modify_commandline_options
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode     # vip or landscape, need to choose
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser)
        opt, unknown = parser.parse_known_args()
        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_mode)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])       # net parameter in cuda: 0

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt

