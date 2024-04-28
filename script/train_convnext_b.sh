#!/bin/bash
PROJECT_ROOT=".."

python ${PROJECT_ROOT}/src/trainer/convnext_b_trainer.py \
--processor_config ${PROJECT_ROOT}/config/convnext_processor.json \
--model_config ${PROJECT_ROOT}/config/convnext_b.json \
--metadata ${PROJECT_ROOT}/data/ISIC2019_medaug/metadataMedAugTrain.csv \
--ckpt_dir ${PROJECT_ROOT}/ckpt/ \
--log_dir ${PROJECT_ROOT}/log/ \
--img_root ${PROJECT_ROOT}/data/ISIC2019_medaug/Train/ \
$@
