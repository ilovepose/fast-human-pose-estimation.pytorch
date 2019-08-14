# Fast Human Pose Estimation CVPR2019

## Introduction
This is an official pytorch implementation of [*Fast Human Pose Estimation*](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Fast_Human_Pose_Estimation_CVPR_2019_paper.html).  

In this work, we focus on the two problems  
1. How to reduce the model size and computation using a model-agnostic method.
2. How to improve the performance of the reduced model.

In our paper
1. We reduce the model size and computation through reducing the width and depth of a network.
2. Propose the fast pose distillation (**FPD**) to improve the performance of the reduced model.

The results on the MPII dataset demonstrate the effectiveness of our approach. We re-implemented the FPD using the HRNet codebase and provided extra evaluation on the COCO dataset.  Our method (FPD) can work without ground-truth labels, it can utilize unlabeled images. 
![Illustrating the architecture of the proposed HRNet](/figures/pose_kd.jpg)

**For the MPII dataset**
1. We first trained a teacher model (hourglass model, stacks=8, num_features=256, 90.520@MPII PCKh@0.5) and a student model (hourglass model, stacks=4, num_features=128, 89.040@MPII PCKh@0.5).
2. We then used the teacher model's prediction and the ground-truth label to co-supervisie the student model (hourglass model, stacks=4, num_features=128, 87.934@MPII PCKh@0.5).
3. Our experiment shows **1.106%** gain from FPD.

**For the COCO dataset**

1. We first trained a teacher model (HRNet-W48, input size=256x192, 75.0@COCO-Valid-Set AP) and a student model (HRNet-W32, input size=256x192, 74.4@COCO-Valid-Set AP).
2. We then used the teacher model's prediction and the ground-truth label to co-supervisie the student model (HRNet-W32, input size=256x192, 75.1@COCO-Valid-Set AP).
3. Our experiment shows **0.7%** gain from FPD.

**If you want to further improve the performance of the student model.You can remove the supervision of ground-truth label in the FPD when there are unlabeled images.**

## Main Results

### Results on MPII val

| Arch                      | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|---------------------------|------|----------|-------|-------|------|------|-------|------|----------|
| hourglass_teacher         | 97.169 | 96.382 | 90.830 | 86.466 | 90.012 | 86.802 | 82.664 | 90.520 | 38.275 |
| hourglass_student         | 96.828 | 95.194 | 87.728 | 82.919 | 87.900 | 82.551 | 78.270 | 87.934 | 34.634 |
| hourglass_student_FPD\*            | 96.385 | 94.905 | 87.847 | 81.875 | 87.225 | 81.906 | 78.955 | 87.598 | 34.359 |
| **hourglass_student_FPD**          | 96.930 | 95.550 | 89.040 | 84.444 | 88.939 | 84.021 | 80.703 | **89.040** | 36.144 |

**Note:**

- Flip test is used.
- Input size is 256x256.
- hourglass_student_FPD\* means not using pretrained students.
- Not using multi-scale test.
- Batch size is 4.
- The PCKh metric implemented in the HRNet codebase for MPII dataset is slightly different than that in our paper.
- The performance of hourglass implemented using pytorch is lower than that implemented using torch(paper).

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch               | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |    AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| pose_hrnet_w48_teacher |    256x192 | 63.6M   |   14.6 | 0.750 | 0.906 |  0.824 |  0.713 |  0.819 | 0.803 | 0.941 |  0.867 |  0.760 |  0.866 |
| pose_hrnet_w32_student |    256x192 | 28.5M   |    7.1 | 0.744 | 0.905 |  0.819 |  0.708 |  0.810 | 0.798 | 0.942 |  0.865 |  0.757 |  0.858 |
| **pose_hrnet_w32_student_FPD**  |    256x192 | 28.5M   |    7.1 | **0.751** | 0.906 |  0.823 |  0.714 |  0.820 | 0.804 | 0.943 |  0.869 |  0.762 |  0.865 |

**Note:**

- Flip test is used.
- [Person detector has person AP of 56.4 on COCO val2017 dataset](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
- GFLOPs is for convolution and linear layers only.
- Batch Size is 24.

## Development environment

The code is developed using python 3.5 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 TITAN XP GPU cards. Other platforms or GPU cards are not fully tested.  

## Quick start

### 1. Preparation

#### 1.1 Prepare the dataset
For the MPII dataset, your directory tree should look like this:   
```
$HOME/datasets/MPII
├── images
└── mpii_human_pose_v1_u12_1.mat
```
For the COCO dataset, your directory tree should look like this:   
```
$HOME/datasets/MSCOCO
├── annotations
├── images
│   ├── test2017
│   ├── train2017
│   └── val2017
└── person_detection_results
````

### 1.2 Prepare the pretrained models
Your directory tree should look like this:  
```
$HOME/datasets/models
├── pytorch
│   ├── imagenet
│   │   ├── hrnet_w32-36af842e.pth
│   │   ├── hrnet_w48-8ef0771d.pth
│   │   └── resnet50-19c8e357.pth
│   ├── pose_coco
│   │   ├── pose_hrnet_w32_256x192.pth
│   │   └── pose_hrnet_w48_256x192.pth
│   └── pose_mpii
│       ├── bs4_hourglass_128_4_1_16_0.00025_0_140_87.934_model_best.pth
│       ├── bs4_hourglass_256_8_1_16_0.00025_0_140_90.520_model_best.pth
│       ├── pose_hrnet_w32_256x256.pth
│       └── pose_hrnet_w48_256x256.pth
└── student_FPD
    ├── hourglass_student_FPD*.pth
    ├── hourglass_student_FPD.pth
    └── pose_hrnet_w32_student_FPD.pth
```

### 1.3 Prepare the environment
Setting the parameters in the file `prepare_env.sh` as follows:

```bash
# DATASET_ROOT=$HOME/datasets
# COCO_ROOT=${DATASET_ROOT}/MSCOCO
# MPII_ROOT=${DATASET_ROOT}/MPII
# MODELS_ROOT=${DATASET_ROOT}/models
```

Then execute:

```bash
bash prepare_env.sh
```

If you like, you can [**prepare the environment step by step**](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

### 2. How to train the model

#### 2.1 Download the pretrained models and place them like the section 1.2

**For MPII dataset**:  [[GoogleDrive]](https://drive.google.com/open?id=1jxL-O5TowVRCZ_xjO-PcmS7juZTYe74T) [[BaiduDrive]](https://pan.baidu.com/s/1Mm1E1G1pYDJVBW2MJTQAUw)

1. hourglass student model  

2. hourglass teacher model  

**For COCO dataset**:  [[GoogleDrive]](https://drive.google.com/open?id=1q09w7iDj_mmIVcXb-pOeLKd-n3Y8g_kA) [[BaiduDrive]](https://pan.baidu.com/s/1Mm1E1G1pYDJVBW2MJTQAUw)

1. HRNet-W32 student model  

2. HRNet-W48 teacher model  

#### 2.2 Start training

```bash
# COCO dataset training
cd scripts/fpd_coco
bash run_train_hrnet.sh

# MPII dataset training
cd scripts/fpd_mpii
bash run_train_hrnet.sh # using hrnet model
bash run_train_hg.sh # using hourglass model

# General training methods, we also provide script shell
cd scripts/mpii
bash run_train_hrnet.sh # using hrnet model
bash run_train_hg.sh # using hourglass model
bash run_train_resnet.sh # using resnet model
cd scripts/coco
bash run_train_hrnet.sh # using hrnet model
bash run_train_hg.sh # using hourglass model
bash run_train_resnet.sh # using resnet model
```

### 3. How to test the model

#### 3.1 Download the trained student models

[[GoogleDrive]](https://drive.google.com/open?id=1LRn-yEluOg4l4xjeUkslyOXYh8ljeSwP) [[BaiduDrive]](https://pan.baidu.com/s/1Mm1E1G1pYDJVBW2MJTQAUw)

**For MPII dataset:**

hourglass student FPD model  

**For COCO dataset:**

HRNet-W32 student FPD model  

#### 3.2 FPD training results and logs

[[GoogleDrive]](https://drive.google.com/open?id=1FJcXP_V9IQb_sRc3bc1Kjd82OCzSJ_-n) [[BaiduDrive]](https://pan.baidu.com/s/1Mm1E1G1pYDJVBW2MJTQAUw)

**Note:**

- coco_hrnet_w48_fpd_w32_256x256: pose_hrnet_w32_student_FPD model training resutls.

- mpii_hourglass_8_256_fpd_hg_4_128_not_pretrained: hourglass_student_FPD\* model training resutls.

- mpii_hourglass_8_256_fpd_hg_4_128_pretrained: hourglass_student_FPD model training resutls.


### Citation

If you use our code or models in your research, please cite with:

```
@InProceedings{Zhang_2019_CVPR,
author = {Zhang, Feng and Zhu, Xiatian and Ye, Mao},
title = {Fast Human Pose Estimation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

### Discussion forum
[ILovePose](http://www.ilovepose.cn:9100)

## Unoffical implementations
[Fast_Human_Pose_Estimation_Pytorch](https://github.com/yuanyuanli85/Fast_Human_Pose_Estimation_Pytorch)

## Acknowledgement
Thanks for the open-source HRNet
* [Deep High-Resolution Representation Learning for Human Pose Estimation, Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/)
