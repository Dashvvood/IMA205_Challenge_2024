#!/bin/bash
python ../trainer/vit_base_trainer.py \
--batch_size 16 \
--device_num 1 \
--processor_config ../config/vit_image_processor.json \
--vit_config ../config/vit_isic2019.json \
--metadata ../data/metadataTrain.csv \
--ckpt_dir "../ckpt/" \
--log_dir "../log" \
--img_root "../data/Train/Train" \
--lr 1e-3 \
--max_epochs 1000 \
$@