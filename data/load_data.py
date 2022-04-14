import os
import struct
import pickle
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import io
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from models.eval_model import INCEPTION_V3_FID, get_activations

def load_acts(dir, image_size, image_paths, filenames, opt):
    filepath = os.path.join(dir, 'acts_%d.pickle' % image_size)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            acts_dict = pickle.load(f)
        print('Load acts_dict from: %s (%d)' % (filepath, len(acts_dict)))
    else:
        acts_dict = dump_fid_acts(filepath, image_paths, filenames, opt)
    return acts_dict

def dump_fid_acts(filepath, image_paths, filenames, opt):
    incep_path = os.path.join(opt.pretrained_dir, 'inception_v3_google-1a9a5a14.pth')
    incep_state_dict = torch.load(incep_path, map_location=lambda storage, loc: storage)

    block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
    inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
    inception_model_fid.cuda()
    inception_model_fid = nn.DataParallel(inception_model_fid)
    inception_model_fid.eval()
    act_dataset = create_acts_dataset(image_paths, filenames, opt)
    preprocess_batchsize = 8
    act_dataloader = torch.utils.data.DataLoader(
        act_dataset, batch_size=preprocess_batchsize, drop_last=False,
        shuffle=False, num_workers=0)
    acts_dict = {}
    count = 0
    for step, data in enumerate(act_dataloader):
        if count % 10 == 0:
            print('%07d / %07d' % (count, act_dataloader.__len__()))
        imgs, names = prepare_acts_data(data)
        batch_size = len(names)
        acts = get_activations(imgs, inception_model_fid, batch_size)
        for batch_index in range(batch_size):
            acts_dict[names[batch_index]] = acts[batch_index]

        count += 1
    with open(filepath, 'wb') as f:
        pickle.dump(acts_dict, f)
        print('Save to: ', filepath)

    return acts_dict

def prepare_acts_data(data):
    imgs, names = data
    imgs = Variable(imgs).cuda()

    return [imgs, names]

class create_acts_dataset(data.Dataset):
    def __init__(self, img_paths, filenames, opt):
        params = get_params(opt)
        self.transform_image = get_transform(opt, params)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.img_paths = img_paths
        self.filenames = filenames

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.img_paths[index])).convert('RGB')
        image = self.transform_image(image)
        return image, self.filenames[index].replace('\\', '/')

    def __len__(self):
        return len(self.filenames)

def load_bytes_data(dir, filenames, modality):
    filepaths = os.path.join(dir, '{}.bigfile'.format(modality))
    # filepaths: datasets/vip/train/segs.bigfile (modality='segs' or 'imgs')
    fbytes = read_bytes(filenames, filepaths)
    return fbytes

def load_chosen_label(dir, img_size):
    if img_size == 256:
        filepath = os.path.join(dir, 'chosen_%d.pickle' % img_size)
    elif img_size == 512 or img_size == 1024:
        filepath = os.path.join(dir, 'chosen_1024.pickle')
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            chosen_dict = pickle.load(f)
        print('Load chosen foreground...', filepath)
    else:
        raise NotImplementedError
    return chosen_dict

def read_bytes(filenames, filepaths):
    fbytes = []
    print('start loading bigfile (%0.02f GB) into memory' % (os.path.getsize(filepaths) / 1024 / 1024 / 1024))
    with open(filepaths, 'rb') as fid:
        for index in range(len(filenames)):
            fbytes_len = struct.unpack('i', fid.read(4))[0]
            fbytes.append(fid.read(fbytes_len))
    return fbytes


def load_filenames(dir):
    # dir: datasets/vip/train/
    filepath = os.path.join(dir, 'filenames.pickle')
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def load_video_inds(dir):
    filepath = os.path.join(dir, 'videos.txt')
    with open(filepath, 'rb') as f:
        video_inds = f.readlines()
    video_inds = [int(ind.rstrip()[6:]) for ind in video_inds]
    return video_inds


def load_class_ids(dir):
    filepath = os.path.join(dir, 'class_ids.pickle')
    with open(filepath, 'rb') as f:
        class_ids = pickle.load(f)
    print('Load class_ids from : %s (%d)' % (filepath, len(class_ids)))
    return class_ids

def process_atts(dir):
    filepath = os.path.join(dir, 'atts.pickle')
    with open(filepath, 'rb') as f:
        atts = pickle.load(f)
    print('Load att_sets from : %s (%d )' % (filepath, len(atts)))
    return atts


def get_params(opt, size=None):
    if size is not None:
        w, h = size
    if opt.preprocess_mode == 'scale_width_and_crop':
        # default choice : scale_width_and_crop
        if opt.dataset_mode == 'landscape':
            if w >= h:
                new_w = int(opt.load_size * w / h)
                new_h = opt.load_size
            else:
                new_w = opt.load_size
                new_h = int(opt.load_size * h / w)
        elif opt.dataset_mode == 'vip':
            new_w = opt.load_size       # 384
            new_h = int(opt.load_size // opt.aspect_ratio)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if opt.isTrain:
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = random.randint(0, np.maximum(0, new_h - int(opt.crop_size // opt.aspect_ratio)))
    else:
        if opt.dataset_mode == 'landscape':
            if w >= h:
                x = int((new_w - opt.crop_size)/2)
                y = 0
            else:
                x = 0
                y = int((new_h - opt.crop_size) / 2)
        elif opt.dataset_mode == 'vip':
            x, y = 0, 0
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):

    def __scale_width(opt, img, target_width, method=Image.BICUBIC):
        ow, oh = img.size
        if opt.dataset_mode == 'landscape':
            if ow >= oh:
                w = int(target_width * ow / oh)
                h = target_width
            else:
                w = target_width
                h = int(target_width * oh / ow)
        elif opt.dataset_mode == 'vip':
            if (ow == target_width and oh == target_width * 2):
                return img
            w = target_width
            h = target_width * 2
        return img.resize((w, h), method)

    def __crop(opt, img, pos, size):
        x1, y1 = pos
        tw = size
        th = int(size // opt.aspect_ratio)
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def __flip(img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    transform_list = []
    if 'scale_width' in opt.preprocess_mode:        # scale_width_and_crop
        transform_list.append(transforms.Lambda(lambda img: __scale_width(opt, img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        if opt.dataset_mode == 'landscape':
            transform_list.append(transforms.Lambda(lambda img: __crop(opt, img, params['crop_pos'], opt.crop_size)))
        elif opt.dataset_mode == 'vip':
            if opt.isTrain:     # when test, load_size=crop_size, so do not need to crop.
                transform_list.append(transforms.Lambda(lambda img: __crop(opt, img, params['crop_pos'], opt.crop_size)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

