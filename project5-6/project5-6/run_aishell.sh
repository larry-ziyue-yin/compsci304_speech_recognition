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

check_data_ready () {
  local missing=0
  for x in train dev test; do
    if [ ! -f "data/${x}/wav.scp" ]; then
      echo "[ERROR] Missing file: data/${x}/wav.scp"
      missing=1
    fi
    if [ ! -f "data/${x}/text" ]; then
      echo "[ERROR] Missing file: data/${x}/text"
      missing=1
    fi
    if [ ! -f "data/${x}/data.list" ]; then
      echo "[ERROR] Missing file: data/${x}/data.list"
      missing=1
    fi
  done
  if [ ! -f "${dict}" ]; then
    echo "[ERROR] Missing dictionary: ${dict}"
    missing=1
  fi
  if [ ${missing} -ne 0 ]; then
    echo "[ERROR] AISHELL data is not prepared for training."
    echo "[ERROR] Please run:"
    echo "        bash run_aishell.sh --stage 0 --stop_stage 3 --datadir <your_aishell_root>"
    exit 1
  fi
}

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
  check_data_ready
  # Training
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  python main.py --config config/aishell/asr_example.yaml
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  check_data_ready
  echo "stage 5: Decoding and WER on test set (5 configurations)"
  ref_file=data/test/text
  if [ ! -f "$ref_file" ]; then
    echo "[ERROR] Ref file not found: $ref_file. Run stage 0-3 first."
    exit 1
  fi
  for decode_cfg in config/aishell/decode_ctc_b1.yaml config/aishell/decode_ctc_b20.yaml \
                    config/aishell/decode_att_b1.yaml config/aishell/decode_att_b20.yaml \
                    config/aishell/decode_joint_b20.yaml; do
    name=$(basename "$decode_cfg" .yaml)
    echo "  Decode: $name"
    python main.py --config "$decode_cfg" --test --outdir result --njobs 8
    csv="result/${name}_test_output.csv"
    hyp="result/${name}_hyp.txt"
    if [ -f "$csv" ]; then
      python tools/prepare_wer_input.py "$csv" "$hyp"
      echo "  WER ($name):"
      python tools/compute-wer.py --char=1 --v=0 "$ref_file" "$hyp"
    else
      echo "[WARN] Decode output not found: $csv"
    fi
  done
  echo "stage 5 done. WER summaries above."
fi


# add LM (optional)
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  check_data_ready
  python main.py --config config/aishell/lm_example.yaml --lm
fi
