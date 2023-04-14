from __future__ import print_function
import os
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import csv
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.optim as optim
from models.networks.word_losses import words_loss

import data
from models.networks.encoder import LabelEncoder
from models.networks.encoder import CNN_ENCODER, Conv_ImgEncoder
from manu_data import vip_split_attr
from manu_data_flickr import land_split_attr as land_split_attr
from manu_data_traffic import traffic_split_attr as traffic_split_attr
from options.train_options import TrainOptions
from options.test_options import TestOptions
import torch.nn as nn


def get_pt_version():
    raw_version = torch.__version__
    raw_version = raw_version.split('.')
    raw_version = [int(raw_version[i]) * pow(10, 3 - i - 1) for i in range(2)]
    version = sum(raw_version)
    return version


def build_models(args):
    text_encoder = LabelEncoder(args)
    image_encoder = Conv_ImgEncoder(args)
    label_tensor = torch.Tensor(range(args.split_num)).long()

    if len(args.gpu_ids) > 1:
        text_encoder = nn.DataParallel(text_encoder, device_ids=args.gpu_ids)
        image_encoder = nn.DataParallel(image_encoder, device_ids=args.gpu_ids)
        label_tensor = label_tensor.expand(len(args.gpu_ids), args.split_num)
    text_encoder = text_encoder.cuda()
    image_encoder = image_encoder.cuda()
    label_tensor = label_tensor.cuda()
    return text_encoder, image_encoder, label_tensor


def prepare_data(data, opt):
    data['label'] = Variable(data['label']).float().cuda()      # seg
    data['image'] = data['image'].cuda()
    data['key'] = data['key']
    data['acts'] = data['acts'].cpu().numpy()
    data['att'] = data['att'].cuda()
    if opt.dataset_mode == 'landscape' or opt.dataset_mode == 'traffic':
        data['class_id'] = None
    elif opt.dataset_mode == 'vip':
        data['class_id'] = data['class_id'].cpu().numpy()
    input_semantics = data['label'].float()
    if opt.dataset_mode == 'landscape' or opt.dataset_mode == 'traffic':
        batch_size = data['att'].shape[0]
        reshape_att = data['att'].view(batch_size, -1)  # [bs, 7*70] or [bs, 6*60]
        index = torch.nonzero(reshape_att)
        if len(index) != batch_size:
            print('index != bs')
            raise NotImplementedError
        attr_relist = index[:, 1]  # len[] = batchsize
    else:
        attr_relist = []
    match_labels = Variable(torch.LongTensor(range(opt.batchSize))).cuda()

    return input_semantics, data['image'], data['key'], data['acts'], data['att'], data['class_id'], match_labels, attr_relist


def train(dataloader, image_encoder, text_encoder, label_tensor, optimizer, opt):
    image_encoder.train()
    text_encoder.train()
    w_total_loss0 = 0
    w_total_loss1 = 0
    correct, total_m = 0, 0
    for step, data in enumerate(dataloader, 0):
        print('step=(%d/%d)'% (step, len(dataloader)))
        text_encoder.zero_grad()
        image_encoder.zero_grad()
        seg, img, key, acts, att, class_id, match_label, attr_relist = prepare_data(data, opt)
        att_emb, _ = text_encoder(label_tensor, att)
        region_features, seg_image = image_encoder(img, seg)
        if opt.dataset_mode == 'landscape':
            word_num = len(land_split_attr)
        elif opt.dataset_mode == 'traffic':
            word_num = len(traffic_split_attr)
        else:
            word_num = len(vip_split_attr)
        w_loss0, w_loss1, _, correct, total_m = words_loss(region_features, seg, att_emb, match_label, word_num,
                                                  class_id, correct, opt, attr_relist, total_m)
        loss = w_loss0 + w_loss1
        w_total_loss0 += w_loss0.item()
        w_total_loss1 += w_loss1.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm(text_encoder.parameters(), 0.25)
        optimizer.step()
        # break
    return (w_total_loss0 + w_total_loss1) / len(dataloader), correct * 100.0 / (2 * total_m)


def evaluate(dataloader, image_encoder, text_encoder, label_tensor, opt):
    image_encoder.eval()
    text_encoder.eval()
    w_total_loss = 0
    correct, total_m = 0, 0
    for step, data in enumerate(dataloader, 0):
        print('val_step: (%d/%d)' % (step, len(dataloader)))
        with torch.no_grad():
            seg, img, key, acts, att, class_id, match_label, attr_relist = prepare_data(data, opt)
            att_emb, _ = text_encoder(label_tensor, att)
            region_features, _ = image_encoder(img, seg)
            if opt.dataset_mode == 'landscape':
                word_num = len(land_split_attr)
            elif opt.dataset_mode == 'traffic':
                word_num = len(traffic_split_attr)
            else:
                word_num = len(vip_split_attr)
            w_loss0, w_loss1, _, correct, total_m = words_loss(region_features, seg, att_emb, match_label, word_num,
                                                      class_id, correct, opt, attr_relist, total_m)
            w_total_loss += (w_loss0 + w_loss1).item()
            # break
    return w_total_loss / len(dataloader), correct * 100.0 / (2 * total_m)


if __name__ == '__main__':
    manualSeed = 19260817
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    PT_VERSION = get_pt_version()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    start_t = time.time()
    # prepare dataset ##
    train_opt = TrainOptions().parse()
    test_opt = TestOptions().parse()
    test_opt.batchSize = 8      # fixed
    test_opt.load_size = test_opt.crop_size

    cudnn.benchmark = True

    train_dataloader = data.create_dataloader(train_opt)
    test_dataloader = data.create_dataloader(test_opt)

    ##############Train ######################################
    text_encoder, image_encoder, label_tensor = build_models(train_opt)
    try:
        train_acc, valid_acc = [], []
        max_val_acc = None
        lr = train_opt.lr
        para = list(text_encoder.parameters())
        for v in image_encoder.parameters():
            if v.requires_grad:
                para.append(v)

        optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))

        for epoch in range(1, train_opt.niter + train_opt.niter_decay+1):
            epoch_start_time = time.time()
            # Train #
            w_loss, precision = train(train_dataloader, image_encoder, text_encoder,
                                      label_tensor, optimizer, train_opt)
            print('| epoch {: 3d} | train_loss {: 5.2f} | train_acc {: 5.2f} |'.format(epoch, w_loss, precision))
            train_acc.append(precision)
            print('-' * 89)
            # Evaluate #
            if len(test_dataloader) > 0:
                val_w_loss, val_precision = evaluate(test_dataloader, image_encoder, text_encoder,
                                                     label_tensor, test_opt)
                print('| end epoch {:3d} | valid_loss'
                      '{:5.2f} | lr {:.5f} | '
                      'valid_acc {:5.2f}|'.format(epoch, val_w_loss, lr, val_precision))

                if max_val_acc is None or val_precision > max_val_acc:
                    print('save the best model... best_val_acc: %s' % val_precision)
                    max_val_acc = val_precision
                    if PT_VERSION <= 130:
                        torch.save(image_encoder.state_dict(), 'pretrained_image_encoder.pth')
                        torch.save(text_encoder.state_dict(), 'pretrained_text_encoder.pth')
                    else:
                        torch.save(image_encoder.state_dict(), 'pretrained_image_encoder.pth', _use_new_zipfile_serialization=False)
                        torch.save(text_encoder.state_dict(), 'pretrained_text_encoder.pth', _use_new_zipfile_serialization=False)
                    print('Save Encoder Models..')
                print('max_val_acc', max_val_acc)
                valid_acc.append(val_precision)
            print('-' * 89)
            if lr > train_opt.lr / 10:
                lr *= 0.98

            with open('pretrained_valid_acc.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(valid_acc)
            # break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

