#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=1

# 参考 https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1


# data
data_url=www.openslr.org/resources/33
# use your own data path
datadir=
# wav data dir
wave_data=data


. tools/parse_options.sh || exit 1;


set -e
set -u
set -o pipefail

train_set=train
dev_set=dev

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Data Download"
  local/download_and_untar_aishell.sh ${datadir} ${data_url} data_aishell
  local/download_and_untar_aishell.sh ${datadir} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Data preparation"
  local/aishell_data_prep.sh ${datadir}/data_aishell/wav \
    ${datadir}/data_aishell/transcript
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # remove the space between the text labels for Mandarin dataset
  for x in train dev test; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) \
      <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
      > data/${x}/text
    rm data/${x}/text.org
  done
fi

dict=data/dict/lang_char.txt
echo "dictionary: ${dict}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<pad> 0" > ${dict}      
  echo "<eos> 1" >> ${dict}    
  echo "<unk> 2" >> ${dict}    
  echo "<blank> 3" >> ${dict}  
  tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
    | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
    awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare data, prepare required format"
  for x in dev test ${train_set}; do
    tools/make_raw_list.py data/$x/wav.scp data/$x/text \
      data/$x/data.list
  done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  python main.py --config config/aishell/asr_example.yaml
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test
  python main.py --config config/aishell/decode_example.yaml --test --njobs 8
fi


# add LM (optional)
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  python main.py --config config/aishell/lm_example.yaml --lm
fi