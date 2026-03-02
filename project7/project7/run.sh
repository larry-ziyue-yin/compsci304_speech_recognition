#!/bin/bash

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=2
wtimit_root=/dkucc/group/courses/compsci304-2526-s3/cl688/wTIMIT

# data
manifest_dir=data
train_manifest=${manifest_dir}/train
dev_manifest=${manifest_dir}/dev
test_manifest=${manifest_dir}/test

. tools/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Data preparation"
  python tools/prepare_wtimit_manifests.py \
    --wtimit-root "${wtimit_root}" \
    --train-manifest "${train_manifest}" \
    --dev-manifest "${dev_manifest}" \
    --test-manifest "${test_manifest}"
fi


# Finetune Whisper
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  echo "stage 1: Finetune Whisper on wTIMIT"
  python main.py --config config/train_config.yaml
fi

# Test/Decoding
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python main.py --config config/test_config.yaml --test --njobs 8
fi
