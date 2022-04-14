"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import *
from models.networks.encoder import *

def modify_commandline_options(parser, is_train):
    # merge commandline options in generator into parser.
    opt, _ = parser.parse_known_args()
    parser = SPADEGenerator.modify_commandline_options(parser)
    return parser

def define_G(opt):
    netG = SPADEGenerator(opt)
    netG.print_network()
    if len(opt.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda()
    netG.init_weights(opt.init_type, opt.init_variance)
    return netG


def define_D(opt):
    from models.networks.discriminator import PAT_D_NET256
    from models.networks.discriminator import SEG_D_NET
    netSegD = SEG_D_NET(opt)
    netPatD = PAT_D_NET256(opt)     # 所有opt.crop_size的Pat_D结构一样
    netSegD.init_weights(opt.init_type, opt.init_variance)
    netPatD.init_weights(opt.init_type, opt.init_variance)
    netSegD.cuda()
    netPatD.cuda()
    return netSegD, netPatD




