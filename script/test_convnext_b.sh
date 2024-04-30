#!/bin/bash
PROJECT_ROOT=".."

python ${PROJECT_ROOT}/src/tester/convnext_tester.py \
--processor_config ${PROJECT_ROOT}/config/convnext_processor.json \
--model_config ${PROJECT_ROOT}/config/convnext_b.json \
--metadata ${PROJECT_ROOT}/data/ISIC2019/metadataTest.csv \
--img_root ${PROJECT_ROOT}/data/ISIC2019/Test/Test \
$@
