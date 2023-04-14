"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import numpy as np
from PIL import Image
import os
from scipy import linalg
from manu_data import split_max_num, part2attr_dict


def prepare_condition(opt, multi_linear, att_embs, segs):
    # att_emb: [bs, 256, 11] or [[bs, 256, 11], [bs, 256, 11], [bs, 256, 11]]--multi_grained
    re_attr_emb = att_embs      # [[bs, 256, 11], [bs, 256, 11], [bs, 256, 11]]
    condition_list = []
    batch_size = re_attr_emb[0].size(0)
    height, width = int(opt.crop_size // opt.aspect_ratio), opt.crop_size
    for i in range(len(re_attr_emb)):
        if opt.dataset_mode == "vip":
            part_emb = torch.zeros([batch_size, opt.label_nc + 1, opt.text_embedding_dim * split_max_num]).cuda()  # with background
            seg = segs.view(batch_size, opt.label_nc, height * width)  # without background
            for j in range(opt.label_nc + 1):
                label_index = np.asarray(part2attr_dict[j])
                length_label_index = label_index.shape[0]
                EX_EMBEDDING_DIM = torch.zeros([batch_size, opt.text_embedding_dim * split_max_num]).cuda()
                tmp_att_emb = re_attr_emb[i][:, :, label_index].view(batch_size, -1)
                EX_EMBEDDING_DIM[:, :length_label_index * opt.text_embedding_dim] = tmp_att_emb
                part_emb[:, j, :] = EX_EMBEDDING_DIM  # with background
            condition = torch.zeros([batch_size, height * width, opt.text_embedding_dim * split_max_num]).cuda()
            for j in range(opt.label_nc):
                part_index = torch.nonzero(seg[:, j, :])
                condition[part_index[:, 0], part_index[:, 1], :] = part_emb[part_index[:, 0], j + 1, :]
        else:
            # dataset_mode == landscape: segs: [bs, idf (7), ih, iw]; attr: [bs, cdf(256), split_Num(7)]
            # or dataset_mode == traffic : segs: [bs, idf (6), ih, iw]; attr: [bs, cdf(256), split_Num(6)]
            part_emb = re_attr_emb[i].permute(0, 2, 1)  # [bs, 7, 256]
            condition = torch.zeros([batch_size, height * width, opt.text_embedding_dim]).cuda()
            seg = segs.view(batch_size, opt.label_nc, height * width)
            for j in range(opt.label_nc):
                part_index = torch.nonzero(seg[:, j, :])
                if len(part_index) == 0:
                    continue
                condition[part_index[:, 0], part_index[:, 1], :] = part_emb[part_index[:, 0], j, :]
        condition_list.append(condition)
    concat_condition = torch.cat(condition_list, dim=-1)    # [bs, h*w, 256*3] or [bs, h*w, 256]
    # -->[bs, h * w, 256]
    concat_condition = multi_linear(concat_condition).contiguous()
    re_condition = concat_condition.view(batch_size, height, width, opt.text_embedding_dim)
    re_condition = re_condition.permute((0, 3, 1, 2))   # [bs, 256, h, w]
    return re_condition


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]     # [3, 128, 64]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    # label_tensor: [4, 19, 128, 64]
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]     # [19, 128, 64]
            one_image_np = tensor2label(one_image, n_label, imtype)
            # print('one_image_np:', one_image_np.shape)      # [3, 128, 64]???
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)       # [bs, 3, 128, 64]
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np
    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:      # 19
        label_tensor = label_tensor.max(0, keepdim=True)[1]     # [1, 128, 64]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path.replace('.jpg', '.png'))


def visualize_result(generate_fake_b, indexs, opt, img_dir):
    """保存生成图像"""
    tile=False
    ########
    trans_fake_img_b = tensor2im(generate_fake_b, tile=tile)
    if opt.dataset_mode == 'vip':
        for index in range(trans_fake_img_b.shape[0]):
            fake_b_path = os.path.join(img_dir, 'generated_img_%s.png' % (indexs[index].replace('/', '-')))
            save_image(trans_fake_img_b[index], fake_b_path)

    elif opt.dataset_mode == 'landscape':
        for index in range(trans_fake_img_b.shape[0]):
            filename = indexs[index].replace('/', '-')
            assert filename.endswith('.jpg')
            filename = filename[: -4] + '.png'
            fake_b_path = os.path.join(img_dir, filename)
            save_image(trans_fake_img_b[index], fake_b_path)
    
    elif opt.dataset_mode == 'traffic':
        for index in range(trans_fake_img_b.shape[0]):
            filename = indexs[index].replace('/', '-')
            assert filename.endswith('.png')
            fake_b_path = os.path.join(img_dir, filename)
            save_image(trans_fake_img_b[index], fake_b_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i + 1  # let's give 0 a color
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b


    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
