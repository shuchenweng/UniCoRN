# UniCoRN: A Unified Conditional Image Repainting Network (official)

## Introducation
This is the author's official PyTorch UniCoRN implementation.

We present UniCoRN (A Unified Conditional Image Repainting Network), an end-to-end trainable deep neural network for Conditional Image Repainting (CIR). UniCoRN breaks the two-phase dependency assumption in the CIR task with a newly designed unified framework.

With UniCoRN, users could create or edit objects with desired color and geometry. 
<!-- ![test image size](https://github.com/shuchenweng/UniCoRN/blob/main/edit.png){:height="50%" width="50%"} -->
 <img src="https://github.com/shuchenweng/UniCoRN/blob/main/edit.png" width = "668" height = "780" alt="图片名称" align=center />
 

## Prerequisites
* Python 3.6
* PyTorch 1.0
* NVIDIA GPU + CUDA cuDNN

## Installation
Clone this repo: 
```
git clone https://github.com/shuchenweng/UniCoRN.git
```
Install PyTorch and dependencies from http://pytorch.org

Install other python requirements

## Datasets
TODO

## Getting Started
Download the [proccessed Landscape dataset](https://drive.google.com/file/d/1KQqIrcLvnL89LsuxAMN5uHehNcZPzfdU/view) and copy them under DATA_DIR.

### 1) Training
```
TODO
```
### 2) Testing
```
TODO
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
