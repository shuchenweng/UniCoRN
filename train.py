"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
from options.train_options import TrainOptions
from options.test_options import TestOptions
import data
import torch
from trainers.pix2pix_trainer import Pix2PixTrainer
from models.eval_model import INCEPTION_V3_FID, INCEPTION_V3, get_activations
import os
import torch.utils.model_zoo as model_zoo
import numpy as np
from util.util import compute_inception_score, \
    calculate_frechet_distance, calculate_activation_statistics, visualize_result
# import psutil
import time
import random
import sys


def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return n_params.item()

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch.set_num_threads(4)
    # parse options
    opt = TrainOptions().parse()
    print(' '.join(sys.argv))
    dataloader = data.create_dataloader(opt)
    trainer = Pix2PixTrainer(opt)

    # set for valid ...
    opt_val = TestOptions().parse()
    opt_val.load_size = opt_val.crop_size
    opt_val.batchSize = opt_val.test_bs       # fixed
    assert opt_val.load_size == opt_val.crop_size

    dataloader_val = data.create_dataloader(opt_val)
    best_fid_score = float('inf')

    ###  Evaluation model ############
    incep_path = os.path.join(opt.pretrained_dir, 'inception_v3_google-1a9a5a14.pth')
    if os.path.exists(incep_path):
        incep_state_dict = torch.load(incep_path, map_location=lambda storage, loc: storage)
    else:
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        incep_state_dict = model_zoo.load_url(url)

    inception_model = INCEPTION_V3(incep_state_dict)
    inception_model.cuda()
    inception_model.eval()
    block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
    inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
    inception_model_fid.cuda()
    inception_model_fid.eval()

    ###################################################################################
    for epoch in range(1, opt.niter+opt.niter_decay+1):     # [1, 600]
        epoch_start_time = time.time()
        for i, data_i in enumerate(dataloader):
            trainer.run_generator_one_step(data_i)
            trainer.run_discriminator_one_step(data_i)
            # break

        # ########### Evaluate ##################
        model = trainer.pix2pix_model_on_one_gpu
        model.eval()
        predictions, fake_acts_set, acts_set = [], [], []
        b_fake_acts_set = []
        correct, total_m = 0, 0
        print('len:', len(dataloader_val))
        result_dir = 'save_result_%d' % epoch
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        save_img_dir = os.path.join(result_dir, 'save_imgs')
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        for val_i, val_data_i in enumerate(dataloader_val):
            generated_val, acts_val, fake_image_background, correct, total_m = model(val_data_i, mode='inference', correct=correct, total_m=total_m, is_val=True)
            visualize_result(fake_image_background, data_i['key'], opt_val, save_img_dir)
            # test fid scores...
            with torch.no_grad():
                pred = inception_model(generated_val)
                pred = pred.data.cpu().numpy()
                predictions.append(pred)

                acts_val = acts_val.cpu().numpy()
                fake_acts = get_activations(generated_val, inception_model_fid, opt_val.batchSize)
                b_fake_acts = get_activations(fake_image_background, inception_model_fid, opt_val.batchSize)

                fake_acts_set.append(fake_acts)
                b_fake_acts_set.append(b_fake_acts)
                acts_set.append(acts_val)

            # break
        #
        w_acc = correct * 100.0 / (2 * total_m)

        predictions = np.concatenate(predictions, 0)
        mean, std = compute_inception_score(predictions, min(10, opt_val.batchSize))

        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)
        b_fake_acts_set = np.concatenate(b_fake_acts_set, 0)

        real_mu, real_sigma = calculate_activation_statistics(acts_set)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
        fake_mu2, fake_sigma2 = calculate_activation_statistics(b_fake_acts_set)

        fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        fid_score2 = calculate_frechet_distance(real_mu, real_sigma, fake_mu2, fake_sigma2)

        print('Score : mean (%f), std (%f), w_acc (%f), fid_score (%f), fid_score_new (%f) ' % (mean, std, w_acc, fid_score, fid_score2))
        save_txt_pth = os.path.join(result_dir, 'test_FID_score.txt')
        with open(save_txt_pth, 'a') as f:
            f.write('%d, %f, %f, %f, %f, %f\n' % (epoch, mean, std, w_acc, fid_score, fid_score2))
        save_pth = os.path.join(result_dir, opt.dataset_mode, 'epoch_%d_fid_%f' % (epoch, fid_score2))
        # save_pth = 'save_result/vip-256/epoch_%d_fid_%f' % (epoch, fid_score2)
        trainer.save(save_pth)

        model.train()
        model.text_encoder.eval()
        model.image_encoder.eval()
        # break

