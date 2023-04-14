"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from util.util import visualize_result
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from models.eval_model import INCEPTION_V3_FID, INCEPTION_V3, get_activations
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from util.util import compute_inception_score, calculate_frechet_distance, calculate_activation_statistics



def load_network(net, dataset_mode, crop_size):
    if dataset_mode == 'vip':
        save_path = os.path.join('E:/samsung/pretrained/unicorn/model_weights/vip/latest_vip_pixel-wise_256_sota.pth')
    elif dataset_mode == 'landscape' and crop_size == 256:
        save_path = os.path.join('E:/samsung/pretrained/unicorn/model_weights/landscape-256/latest_landscape_pixel-wise_256_sota.pth')
    elif dataset_mode == 'landscape' and crop_size == 512:
        save_path = os.path.join('E:/samsung/pretrained/unicorn/model_weights/landscape-512/latest_landscape_pixel-wise_512_sota.pth')
    elif dataset_mode == 'traffic' and crop_size == 256:
        save_path = os.path.join('E:/samsung/pretrained/unicorn/model_weights/traffic-256/latest_traffic_pixel-wise_256_sota.pth') #todo
    elif dataset_mode == 'traffic' and crop_size == 512:
        save_path = os.path.join('E:/samsung/pretrained/unicorn/model_weights/traffic-512/latest_traffic_pixel-wise_512_sota.pth') #todo
    weights = torch.load(save_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(weights, strict=False)
    return net


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    opt = TestOptions().parse()
    assert opt.batchSize == 6
    opt.load_size = opt.crop_size
    dataloader = data.create_dataloader(opt)
    model = Pix2PixModel(opt)

    model.netG = load_network(model.netG, opt.dataset_mode, opt.crop_size).cuda()
    model.netG.eval()

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
    #
    correct, total_m = 0, 0
    predictions, fake_acts_set, acts_set = [], [], []
    b_fake_acts_set = []
    #
    result_dir = 'E:/pkuproject/modelsets/unicorn'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_img_dir = os.path.join(result_dir, 'save_imgs')
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    with torch.no_grad():
        for i, data_i in enumerate(dataloader):
            print(i, len(dataloader))
            generated, acts_val, fake_image_background, correct, total_m = model(data_i, mode='inference', correct=correct, total_m=total_m, is_val=True)
            visualize_result(fake_image_background, data_i['key'], opt, save_img_dir)

            pred = inception_model(generated)
            pred = pred.data.cpu().numpy()
            predictions.append(pred)

            acts_val = acts_val.cpu().numpy()
            fake_acts = get_activations(generated, inception_model_fid, opt.batchSize)
            b_fake_acts = get_activations(fake_image_background, inception_model_fid, opt.batchSize)

            fake_acts_set.append(fake_acts)
            b_fake_acts_set.append(b_fake_acts)
            acts_set.append(acts_val)
            # break

        w_acc = correct * 100.0 / (2 * total_m)
        predictions = np.concatenate(predictions, 0)
        mean, std = compute_inception_score(predictions, min(10, opt.batchSize))

        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)
        b_fake_acts_set = np.concatenate(b_fake_acts_set, 0)

        real_mu, real_sigma = calculate_activation_statistics(acts_set)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
        fake_mu2, fake_sigma2 = calculate_activation_statistics(b_fake_acts_set)

        fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        fid_score2 = calculate_frechet_distance(real_mu, real_sigma, fake_mu2, fake_sigma2)
        print('Score : mean (%f), std (%f), w_acc (%f), fid_score (%f), fid_score_new (%f)' %
              (mean, std, w_acc, fid_score, fid_score2))
        fullpath = os.path.join(result_dir, 'eval_score_test.txt')
        with open(fullpath, 'a') as f:
            f.write('%f, %f, %f, %f, %f\n' % (mean, std, w_acc, fid_score, fid_score2))




