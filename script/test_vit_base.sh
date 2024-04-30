#!/bin/bash
PROJECT_ROOT=".."

python ${PROJECT_ROOT}/src/tester/vit_base_tester.py \
--processor_config ${PROJECT_ROOT}/config/vit_image_processor.json \
--vit_config ${PROJECT_ROOT}/config/vit_isic2019.json \
--metadata ${PROJECT_ROOT}/data/ISIC2019/metadataTest.csv \
--img_root ${PROJECT_ROOT}/data/ISIC2019/Test/Test \
$@
