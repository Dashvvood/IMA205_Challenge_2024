---
typora-copy-images-to: ./${filename}.assets
---

# IMA205_Challenge_2024

[<img alt="Static Badge" src="https://img.shields.io/badge/IMA205-Challenge-%2320BEFF?style=flat&logo=Kaggle">](https://www.kaggle.com/competitions/ima205-challenge-2024/overview)  [<img alt="Static Badge" src="https://img.shields.io/badge/Pytorch-2.2-%23EE4C2C?style=flat&logo=pytorch">](https://pytorch.org/) [<img alt="Static Badge" src="https://img.shields.io/badge/Lightning-2.2-%23792EE5?style=flat&logo=Lightning">](https://lightning.ai/)

## Introduction

A skin lesion is defined as a superficial growth or patch of the skin that is visually different and/or has a different texture than its surrounding area. Skin lesions, such as moles or birthmarks, can degenerate and become cancer, with melanoma being the deadliest skin cancer. Its incidence has increased during the last decades, especially in the areas mostly populated by white people.

The most effective treatment is an early detection followed by surgical excision. This is why several approaches for skin cancer detection have been proposed in the last years (non-invasive computer-aided diagnosis (CAD) ).

**The goal of this challenge is to classify dermoscopic images of skin lesions among eight different diagnostic classes.**

1. Melanoma
2. Melanocytic nevus
3. Basal cell carcinoma
4. Actinic keratosis
5. Benign keratosis
6. Dermatofibroma
7. Vascular lesion
8. Squamous cell carcinoma



## Installation

```shell
conda create -n challenge python=3.11
```

```shell
source activate challenge
pip3 install -r requirement.txt
```



## Code Structure

```shell
├── allinone.sh						# fast push github
├── archive							# code and doc archived
│   ├── README.assets						
│   ├── README.md
│   └── utils.py
├── ckpt -> /media/XXX/ckpt   		# folder to store checkpoints(symbolic link)
├── config							# configuration of model or processor
│   ├── convnext_b.json
│   ├── convnext_processor.json
│   ├── convnext_s.json
│   ├── convnext_tiny.json
│   ├── convnext_v2_b.json
│   ├── vit_image_processor.json
│   └── vit_isic2019.json
├── data							# data folder
│   ├── dummy						
│   ├── ima205-challenge-2024.zip	
│   ├── ISIC2019					# ISIC2019 dataset
├── LICENSE
├── log								# store wandb log files
│   └── wandb
├── README.md
├── requirement.txt
├── script							# some scirpts for running code
│   ├── calculate_mean_std.py
│   ├── download_hf_cache.py
│   ├── lightning_logs
│   ├── test_convnext_b.sh
│   ├── test_convnext_s.sh
│   ├── test_vit_base.sh
│   ├── train_convnext_b_ddp.sh
│   ├── train_convnext_b.sh
│   ├── train_convnext_s_ddp.sh
│   ├── train_convnext_s.sh
│   └── train_vit_base.sh
└── src								# source code
    ├── args.py						# common command line parameters
    ├── constant.py					# constant, eg: INDEX2LABEL, LABEL2INDEX
    ├── dataset						# how to load ISIC2019 dataset
    ├── losses.py					# LMFLoss
    ├── model						# folder containing model code				
    ├── tester						# fodler for tester
    ├── trainer						# fodler for trainer
    └── visualizer.py				# visualizer of data	
```



## Usage

```shell
# train vit 
CUDA_VISIBLE_DEVICES=5 python ./src/trainer/vit_base_trainer.py \
--processor_config ./config/vit_image_processor.json \
--vit_config ./config/vit_isic2019.json \
--metadata ./data/ISIC2019/metadataTrain.csv \
--ckpt_dir ./ckpt/ \
--log_dir ./log/ \
--img_root ./data/ISIC2019/Train/Train \
--batch_size 32 \
--lr 6e-5
```

>metadata and img_root are parameters required in `dataset.py`

```shell
# train convnext
CUDA_VISIBLE_DEVICES=5 python ${PROJECT_ROOT}/src/trainer/convnext_b_trainer.py \
--processor_config ${PROJECT_ROOT}/config/convnext_processor.json \
--model_config ${PROJECT_ROOT}/config/convnext_b.json \
--metadata ${PROJECT_ROOT}/data/ISIC2019_medaug/metadataMedAugTrain.csv \
--ckpt_dir ${PROJECT_ROOT}/ckpt/ \
--log_dir ${PROJECT_ROOT}/log/ \
--img_root ${PROJECT_ROOT}/data/ISIC2019_medaug/Train/ \
--batch_size 32 \
--lr 6e-5

```

**use shell script**

```shell
cd script/
# set parameters in shell script
CUDA_VISIBLE_DEVICES=5 ./train_convnext_b.sh --lr 6e-5 --batch_size 16
```



**MedAugment**

- https://github.com/NUS-Tim/MedAugment.git

Transform original dataset file structure like this: 

```shell
├── classification
    ├── ISIC2019
        ├── baseline
            ├── training
            |   ├── class_1
            |   |   ├── img_1.jpg  # .png
            |   │   ├── img_2.jpg
            |   │   ├── ...
            |   ├── class_2
            |   |   ├── img_a.jpg
            |   │   ├── img_b.jpg
            |   │   ├── ...
            |   ├── ...
            └── validation
            └── test
```

>In metadata.csv, the class index starts from ONE, so we can create a empty folder class_0, so that we can use this plug-in Augmentation easily.



**Test**

```shell
# test convnext
CUDA_VISIBLE_DEVICES=5 python ./src/tester/convnext_tester.py \
--processor_config ./config/convnext_processor.json \
--model_config ./config/convnext_s.json \
--metadata ./data/ISIC2019/metadataTest.csv \
--img_root ./data/ISIC2019/Test/Test \
--batch_size 8 \
--ckpt <ckpt_path>
```



## Training curves of convnext_b

![image-20240506032220740](./README.assets/image-20240506032220740.png)




---

---
