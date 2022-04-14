import os.path
import torch.utils.data as data
from .load_data import load_filenames, load_acts, load_bytes_data, load_chosen_label, process_atts
from .load_data import get_transform, get_params
import numpy as np
import io
from PIL import Image
from torch.autograd import Variable
import torch
import random

class LandscapeDataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.set_defaults(label_nc=7)
        parser.set_defaults(aspect_ratio=1)
        parser.set_defaults(split_num=7)
        parser.set_defaults(attr_num=70)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            if opt.crop_size == 256:
                parser.set_defaults(num_upsampling_layers='normal')
            elif opt.crop_size == 512:
                parser.set_defaults(num_upsampling_layers='more')
            elif opt.crop_size == 1024:
                parser.set_defaults(num_upsampling_layers='most')
        return parser

    def initialize(self, opt):
        self.opt = opt
        label_paths, image_paths, filenames, acts_dict, atts, chosen_dict = self.get_paths(opt)
        self.label_paths = label_paths
        self.image_paths = image_paths
        self.filenames = filenames
        self.acts_dict = acts_dict
        self.atts = atts
        self.chosen_dict = chosen_dict
        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else opt.phase
        dir = os.path.join(root, '%s' % phase)
        filenames = load_filenames(dir)
        label_paths = load_bytes_data(dir, filenames, 'segs')
        image_paths = load_bytes_data(dir, filenames, 'imgs')
        atts = process_atts(dir)
        if phase == 'train':
            acts_dict = None
        elif phase == 'val':
            acts_dict = load_acts(dir, self.opt.crop_size, image_paths, filenames, opt)      # [256, 256]
        chosen_dict = load_chosen_label(dir, self.opt.crop_size)
        return label_paths, image_paths, filenames, acts_dict, atts, chosen_dict

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        key = self.filenames[index]
        if self.acts_dict:
            acts = self.acts_dict[key]
        else:
            acts = torch.zeros([1])
        class_id = torch.zeros([1])
        att = self.atts[index].astype('float32')
        label = Image.open(io.BytesIO(label_path))
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, toTensor=False)
        trans_label = transform_label(label)
        re_seg = np.asarray(trans_label)

        new_seg = np.zeros([self.opt.label_nc, re_seg.shape[0], re_seg.shape[1]])

        chosen_label = self.chosen_dict[key]
        new_att = np.zeros(att.shape).astype('float32')  # [7, 70]
        new_att[chosen_label, :] = att[chosen_label, :]
        new_seg[chosen_label, re_seg == (chosen_label + 1) * 10] = 1

        label_tensor = Variable(torch.from_numpy(new_seg).float())       # [256, 128]

        image_path = self.image_paths[index]
        image = Image.open(io.BytesIO(image_path)).convert('RGB')
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image).float()
        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'key': key,
                      'acts': acts,
                      'class_id': class_id,
                      'att': new_att
                      }

        chosen_label_tensor = label_tensor[chosen_label]  # [h, w]
        mask = chosen_label_tensor.unsqueeze(0)  # [1, h, w]
        # foreground part is 1, background is 0
        back_index = (mask == 0).expand_as(image_tensor)    # [3, 256, 128]
        front_tensor = torch.zeros(image_tensor.shape)
        front_tensor[back_index] = image_tensor[back_index]
        b_tensor = torch.cat((front_tensor, mask), dim=0)
        input_dict['background'] = b_tensor

        return input_dict

    def __len__(self):
        return self.dataset_size




