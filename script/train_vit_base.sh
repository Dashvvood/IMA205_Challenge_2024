#!/bin/bash
PROJECT_ROOT=".."

python ${PROJECT_ROOT}/src/trainer/vit_base_trainer.py \
--processor_config ${PROJECT_ROOT}/config/vit_image_processor.json \
--vit_config ${PROJECT_ROOT}/config/vit_isic2019.json \
--metadata ${PROJECT_ROOT}/data/metadataTrain.csv \
--ckpt_dir ${PROJECT_ROOT}/ckpt/ \
--log_dir ${PROJECT_ROOT}/log/ \
--img_root ${PROJECT_ROOT}/data/Train/Train \
$@
