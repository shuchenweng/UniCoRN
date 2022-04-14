"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model, device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        self.generated_back = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_DSeg, self.optimizer_DPat = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated, generated_back = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.generated = generated
        self.generated_back = generated_back

    def run_discriminator_one_step(self, data):
        self.optimizer_DPat.zero_grad()
        if self.opt.seg_loss:
            self.optimizer_DSeg.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        pat_d_loss = (d_losses['single_pat_D'] + d_losses['double_pat_D'])/2
        if len(self.opt.gpu_ids) > 1:
            pat_d_loss = sum(pat_d_loss)
        pat_d_loss.backward()
        self.optimizer_DPat.step()
        if self.opt.seg_loss:
            seg_d_loss = d_losses['seg_D']
            if len(self.opt.gpu_ids) > 1:
                seg_d_loss = sum(seg_d_loss)
            seg_d_loss.backward()
            self.optimizer_DSeg.step()

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)
