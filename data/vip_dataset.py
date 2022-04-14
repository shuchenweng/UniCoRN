import os.path
import torch.utils.data as data
from .load_data import load_filenames, load_acts, load_bytes_data, load_class_ids, load_video_inds, process_atts
from .load_data import get_transform, get_params
import numpy as np
import io
from PIL import Image
from torch.autograd import Variable
import torch
import pickle

class VipDataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.set_defaults(label_nc=19)
        parser.set_defaults(aspect_ratio=0.5)
        opt, _ = parser.parse_known_args()
        return parser

    def initialize(self, opt):
        self.opt = opt
        label_paths, image_paths, filenames, acts_dict, video_inds, class_ids, atts = self.get_paths(opt)
        self.label_paths = label_paths
        self.image_paths = image_paths
        self.filenames = filenames
        self.acts_dict = acts_dict
        self.video_inds = video_inds
        self.atts = atts
        self.class_ids = class_ids
        size = len(self.label_paths)
        self.dataset_size = size


    def get_paths(self, opt):
        root = opt.dataroot
        # phase = 'val' if opt.phase == 'test' else opt.phase
        phase = opt.phase
        dir = os.path.join(root, '%s' % phase)
        filenames = load_filenames(dir)
        label_paths = load_bytes_data(dir, filenames, 'segs')
        image_paths = load_bytes_data(dir, filenames, 'imgs')
        video_inds = load_video_inds(dir)
        class_ids = load_class_ids(dir)
        atts = process_atts(dir)
        # Add acts
        if phase == 'train':
            acts_dict = None
        elif phase == 'test':
            acts_dict = load_acts(dir, self.opt.crop_size, image_paths, filenames, opt)
        return label_paths, image_paths, filenames, acts_dict, video_inds, class_ids, atts

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        key = self.filenames[index].replace('\\', '/')
        if self.acts_dict:
            acts = self.acts_dict[key]
        else:
            acts = torch.zeros([1])
        class_id = self.class_ids[index]
        att = self.atts[index].astype('float32')
        label = Image.open(io.BytesIO(label_path))
        params = get_params(self.opt)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, toTensor=False)
        trans_label = transform_label(label)
        re_seg = np.asarray(trans_label)

        new_seg = np.zeros([self.opt.label_nc, re_seg.shape[0], re_seg.shape[1]])
        max_seg = np.zeros([re_seg.shape[0], re_seg.shape[1]])
        for j in range(self.opt.label_nc):
            # [0,19)
            if j == 12:
                continue
            new_seg[j, re_seg == (j + 1) * 10] = 1
            max_seg[re_seg == (j+1) * 10] = j+1
        label_tensor = Variable(torch.from_numpy(new_seg).float())
        seg_tensor = torch.from_numpy(re_seg).float()

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
                      'att': att
                      }

        max_label = torch.max(label_tensor, dim=0)[0]   # [256, 128]
        mask = max_label.unsqueeze(0)        # [1, 256, 128]
        # foreground area is 1, background is 0
        back_index = (mask == 0).expand_as(image_tensor)    # [3, 256, 128]
        front_tensor = torch.zeros(image_tensor.shape)
        front_tensor[back_index] = image_tensor[back_index]
        b_tensor = torch.cat((front_tensor, mask), dim=0)   # [4, 256, 128]
        input_dict['background'] = b_tensor
        input_dict['seg_tensor'] = seg_tensor
        return input_dict

    def __len__(self):
        return self.dataset_size




