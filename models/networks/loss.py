"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19
from torch.autograd import Variable

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids, opt):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19(opt).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


############## D_loss #############################
def patD_loss(opt, netPatD, real_imgs, fake_img, fake_image_background, conditions):
    batch_size = real_imgs.size(0)
    cond_wrong_errD = 0
    if not opt.no_ganFeat_loss:
        real_features = netPatD(real_imgs)[-1]
        fake_features = netPatD(fake_img)[-1]
        fake_features_b = netPatD(fake_image_background)[-1]
    else:
        real_features = netPatD(real_imgs)
        fake_features = netPatD(fake_img)
        fake_features_b = netPatD(fake_image_background)
    if conditions is not None:
        sample_condition = nn.Upsample(size=(real_features.shape[2], real_features.shape[3]), mode='nearest')(conditions)
        cond_real_logits = netPatD.COND_DNET(real_features, sample_condition)
        cond_fake_logits = netPatD.COND_DNET(fake_features, sample_condition)
        cond_fake_logits_b = netPatD.COND_DNET(fake_features_b, sample_condition)
        real_labels = torch.FloatTensor(cond_real_logits.size()).fill_(1).cuda()
        fake_labels = torch.FloatTensor(cond_fake_logits.size()).fill_(0).cuda()
        cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
        cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
        cond_fake_errD_b = nn.BCELoss()(cond_fake_logits_b, fake_labels)
    else:
        cond_real_errD, cond_fake_errD, cond_fake_errD_b = 0, 0, 0

    if batch_size > 1:
        # augmentation.
        if opt.add_wrong_errD:
            cond_wrong_logits = netPatD.COND_DNET(real_features[:(batch_size - 1)], sample_condition[1:batch_size])
            cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    real_logits = netPatD.UNCOND_DNET(real_features)
    fake_logits = netPatD.UNCOND_DNET(fake_features)
    fake_logits_b = netPatD.UNCOND_DNET(fake_features_b)
    real_labels = torch.FloatTensor(real_logits.size()).fill_(1).cuda()
    fake_labels = torch.FloatTensor(fake_logits.size()).fill_(0).cuda()
    real_errD = nn.BCELoss()(real_logits, real_labels)
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
    fake_errD_b = nn.BCELoss()(fake_logits_b, fake_labels)
    single_errD = (real_errD / 2. + fake_errD / 3.) * opt.patD_loss_weight +\
           (cond_real_errD / 2. + cond_fake_errD / 3. + cond_wrong_errD / 3.)
    double_errD = (real_errD / 2. + fake_errD_b / 3.) * opt.patD_loss_weight + \
           (cond_real_errD / 2. + cond_fake_errD_b / 3. + cond_wrong_errD / 3.)

    return single_errD, double_errD


def segD_loss(netSegD, fake_img_background, seg_mask, opt):
    # seg_mask: [bs, 19, h, w]
    pool_seg = torch.max(seg_mask, dim=1)[0]        # [bs, h, w]
    fake_seg_result = netSegD(fake_img_background)      # [bs, 1, 256, 128]
    pool_seg = pool_seg.unsqueeze(dim=1)        # [bs, 1, h, w]
    pool_seg = nn.Upsample(size=(fake_seg_result.shape[2], fake_seg_result.shape[3]), mode='nearest')(pool_seg) # downscale=4
    seg_groundtruth = pool_seg
    errD = nn.BCELoss()(fake_seg_result, seg_groundtruth)
    return errD


############ G_Loss ########################
def G_loss(opt, netPatD, netSegD, real_image, fake_image, fake_image_background, condition, G_losses):
    if not opt.no_ganFeat_loss:
        GAN_Feat_loss = torch.FloatTensor(1).fill_(0).cuda()
        f_features = netPatD(fake_image)
        r_features = netPatD(real_image)  # len=4
        fb_features = netPatD(fake_image_background)
        for i in range(len(f_features)):
            GAN_Feat_loss += (nn.L1Loss()(f_features[i], r_features[i].detach()) +
                              nn.L1Loss()(fb_features[i], r_features[i].detach()))
        G_losses['GAN_Feat'] = GAN_Feat_loss * opt.lambda_feat
        features = f_features[-1]
        features_b = fb_features[-1]
    else:
        features = netPatD(fake_image)      # [bs, 8ndf, 8, 4]
        features_b = netPatD(fake_image_background)
    # add Pat_loss (condition + uncondition)
    # if opt.use_lae:
    if condition is not None:
        sample_condition = nn.Upsample(size=(features.shape[2], features.shape[3]), mode='nearest')(condition)  # [bs, 256(nef), h, w]
        cond_logits = netPatD.COND_DNET(features, sample_condition)
        cond_logits_b = netPatD.COND_DNET(features_b, sample_condition)
        real_labels = Variable(torch.FloatTensor(cond_logits.size()).fill_(1)).cuda()
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        cond_errG_b = nn.BCELoss()(cond_logits_b, real_labels)
    else:
        cond_errG, cond_errG_b = 0, 0

    logits = netPatD.UNCOND_DNET(features)
    logits_b = netPatD.UNCOND_DNET(features_b)
    # real_labels = Variable(torch.FloatTensor(logits.size()).fill_(1)).cuda()
    real_labels = Variable(torch.FloatTensor(logits.size()).fill_(1)).cuda()
    errG = nn.BCELoss()(logits, real_labels)
    errG_b = nn.BCELoss()(logits_b, real_labels)
    G_losses['single_pat'] = errG * opt.patD_loss_weight + cond_errG
    G_losses['double_pat'] = errG_b * opt.patD_loss_weight + cond_errG_b
    # add seg loss...
    if opt.seg_loss:
        fake_seg_result = netSegD(fake_image_background)
        fake_groundtruth = torch.zeros(fake_seg_result.shape).cuda()
        segG_loss = nn.BCELoss()(fake_seg_result, fake_groundtruth)
        G_losses['seg'] = opt.segG_loss_weight * segG_loss         # train_smooth_lambda3 =0.03

    return G_losses

