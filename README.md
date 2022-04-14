# UniCoRN: A Unified Conditional Image Repainting Network (official)

## Introduction
This is the author's official PyTorch UniCoRN implementation.

We present UniCoRN (A Unified Conditional Image Repainting Network), an end-to-end trainable deep neural network for Conditional Image Repainting (CIR). UniCoRN breaks the two-phase dependency assumption in the CIR task with a newly designed unified framework.

With UniCoRN, users could create or edit objects with desired color and geometry. 
<!-- ![test image size](https://github.com/shuchenweng/UniCoRN/blob/main/edit.png){:height="50%" width="50%"} -->
 <img src="https://github.com/shuchenweng/UniCoRN/blob/main/edit.png" width = "668" height = "780" alt="图片名称" align=center />
 

## Prerequisites
* Python 3.6
* PyTorch 1.10
* NVIDIA GPU + CUDA cuDNN

## Installation
Clone this repo: 
```
git clone https://github.com/shuchenweng/UniCoRN.git
```
Install PyTorch and dependencies from http://pytorch.org

Install other python requirements

## Datasets
we create a LANDSCAPE-CIR dataset with 31K high-resolution images from Flickr including 28K training images and 3K test images. We
use the pretrained segmentation network DeepLabV2 to compute the scene parsing mask for each image. Finally we label the color attribute for each image. Therefore,
at most 70 attributes are annotated in one image. We process the VIP person parsing dataset in the same way as [MISC](https://github.com/shuchenweng/MISC) and name it as PERSON-CIR, which includes 42K training images and 6K test images.

## Getting Started
Download the [Landscape-CIR dataset](https://drive.google.com/file/d/1KQqIrcLvnL89LsuxAMN5uHehNcZPzfdU/view) or [PERSON-CIR dataset](https://drive.google.com/file/d/13to7_krxUlW6bYiA2EojhsxJ5wGnF40s/view?usp=sharing).

Download the [pretrained weights](https://drive.google.com/drive/folders/1CnkJ7IetA7EBVGzbFTeAzpcgx-4tRpm-?usp=sharing) of image and text encoders

Download the [pretrained weights](https://drive.google.com/drive/folders/1hxww-Kl7BF_gd2jKhNqSJ5ToAnAEiTFL?usp=sharing) of models for finetuning or directly repainting your images.

### 1) Pretrain encoders:
PERSON-CIR dataset:
```
python pretrain_DAMSM.py --load_size 384 --crop_size 256  --dataroot  YOURS_DATASET_DIR --batchSize YOURS_BATCH_SIZE --gpu_ids YOURS_GPU_ID
```

Landscape-CIR dataset (256): 
```
python pretrain_DAMSM.py --load_size 384 --crop_size 256 --dataroot YOURS_DATASET_DIR  --dataset_mode landscape --batchSize YOURS_BATCH_SIZE --gpu_ids YOURS_GPU_ID
```

Landscape-CIR dataset (512):
```
python pretrain_DAMSM.py --load_size 768 --crop_size 512 --dataroot YOURS_DATASET_DIR  --dataset_mode landscape --batchSize YOURS_BATCH_SIZE --gpu_ids YOURS_GPU_ID
```

### 1) Training
PERSON-CIR dataset:
```
python train.py --load_size 384 --crop_size 256  --dataroot YOURS_DATASET_DIR --pretrained_dir YOURS_ENCODER_DIR --batchSize YOURS_BATCH_SIZE --test_bs 6   --gpu_ids YOURS_GPU_ID
```

Landscape-CIR dataset (256): 
```
python train.py --load_size 384 --crop_size 256 --dataroot YOURS_DATASET_DIR --pretrained_dir YOURS_ENCODER_DIR  --dataset_mode landscape --batchSize YOURS_BATCH_SIZE --test_bs 6 --gpu_ids YOURS_GPU_ID
```

Landscape-CIR dataset (512):
```
python train.py --load_size 768 --crop_size 512 --dataroot YOURS_DATASET_DIR --pretrained_dir YOURS_ENCODER_DIR  --dataset_mode landscape --batchSize YOURS_BATCH_SIZE --test_bs 6  --gpu_ids YOURS_GPU_ID
```

### 2) Testing
Modify the paths of pretrained model weights in test.py L20-25, and execute commands below.

PERSON-CIR dataset:
```
python test.py --load_size 256 --crop_size 256  --dataroot YOURS_DATASET_DIR --pretrained_dir YOURS_ENCODER_DIR --batchSize 6  --gpu_ids YOURS_GPU_ID
```

Landscape-CIR dataset (256): 
```
python test.py --load_size 384 --crop_size 256 --dataroot YOURS_DATASET_DIR --pretrained_dir YOURS_ENCODER_DIR  --dataset_mode landscape --batchSize 6 --gpu_ids YOURS_GPU_ID
```

Landscape-CIR dataset (512):
```
python test.py --load_size 768 --crop_size 512 --dataroot YOURS_DATASET_DIR --pretrained_dir YOURS_ENCODER_DIR  --dataset_mode landscape --batchSize 6 --gpu_ids YOURS_GPU_ID
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## Citation
If you use this code for your research, please cite our papers [UniCoRN: A Unified Conditional Image Repainting Network](https://ci.idm.pku.edu.cn/Weng_CVPR22c.pdf)
```
@InProceedings{UniCoRN,
  author = {Sun, Jimeng and Weng, Shuchen and Chang, Zheng and Li, Si and Shi, Boxin},
  title = {UniCoRN: A Unified Conditional Image Repainting Network},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```
