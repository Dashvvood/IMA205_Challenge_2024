#!/bin/bash
PROJECT_ROOT=".."

python ${PROJECT_ROOT}/src/tester/convnext_tester.py \
--processor_config ${PROJECT_ROOT}/config/convnext_processor.json \
--vit_config ${PROJECT_ROOT}/config/convnext_s.json \
--metadata ${PROJECT_ROOT}/data/metadataTest.csv \
--ckpt_dir ${PROJECT_ROOT}/ckpt/ \
--log_dir ${PROJECT_ROOT}/log/ \
--img_root ${PROJECT_ROOT}/data/Test/Test \
$@
