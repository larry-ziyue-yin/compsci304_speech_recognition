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
data_url=www.openslr.org/resources/12
# use your own data path
datadir=./data
# wav data dir
wave_data=data
outdir=result


. tools/parse_options.sh || exit 1;

# bpemode (unigram or bpe)
nbpe=5000
bpemode=bpe

set -e
set -u
set -o pipefail

train_set=train_100
dev_set=dev

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Data Download"
  for part in dev-clean test-clean train-clean-100; do
    local/download_and_untar.sh ${datadir} ${data_url} ${part}
  done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  ### Task dependent. You have to make data the following preparation part by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 0: Data preparation"
  for part in dev-clean test-clean train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep_torchaudio.sh ${datadir}/LibriSpeech/${part} $wave_data/${part//-/_}
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 1: Feature Generation"
  mkdir -p $wave_data/train_100  
  # merge total training data
  for set in train_clean_100; do
    for f in `ls $wave_data/$set`; do
      cat $wave_data/$set/$f >> $wave_data/train_100/$f
    done
  done
  mkdir -p $wave_data/dev  
  # merge total dev data
  for set in dev_clean; do
    for f in `ls $wave_data/$set`; do
      cat $wave_data/$set/$f >> $wave_data/$dev_set/$f
    done
  done

fi


dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Dictionary and BPE model preparation"
  mkdir -p data/lang_char/

  echo "<pad> 0" > ${dict}      
  echo "<eos> 1" >> ${dict}    
  echo "<unk> 2" >> ${dict}    
  echo "<blank> 3" >> ${dict}   

  cut -f 2- -d" " $wave_data/${train_set}/text > $wave_data/lang_char/input.txt
  
  tools/spm_train \
    --input=$wave_data/lang_char/input.txt \
    --vocab_size=${nbpe} \
    --model_type=bpe \
    --model_prefix=${bpemodel} \
    --pad_id=0 \
    --eos_id=1 \
    --unk_id=2 \
    --bos_id=-1 \
    --eos_piece='<eos>'
  
  tools/spm_encode --model=${bpemodel}.model --output_format=piece \
    < $wave_data/lang_char/input.txt \
    | tr ' ' '\n' \
    | sort \
    | uniq \
    | awk '{print $0 " " NR+3}' >> ${dict}
  
  wc -l ${dict}
fi


# You can train your own subword model or you can skip this step
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then 
  echo "stage 3: Generate extra subword vocab"
  python3 tools/generate_vocab_file.py --input_file $wave_data/lang_char/input.txt --mode subword --output_file $wave_data/subword-16k.model

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  python main.py --config config/libri/asr_example.yaml
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: Decoding and WER on test set (5 configurations)"
  mkdir -p "${outdir}"
  summary_file="${outdir}/wer_summary.txt"
  : > "${summary_file}"

  for name in p6_ctc_b1 p6_ctc_b20 p6_att_b1 p6_att_b20 p6_joint_b20; do
    decode_cfg="config/libri/${name}.yaml"
    csv="${outdir}/${name}_test_output.csv"
    hyp="${outdir}/${name}.hyp"
    ref="${outdir}/${name}.ref"

    if [ ! -f "${decode_cfg}" ]; then
      echo "[WARN] Missing config: ${decode_cfg}"
      echo "${name}: CONFIG_MISSING" >> "${summary_file}"
      continue
    fi

    echo "  Decode: ${name}"
    if ! python main.py --config "${decode_cfg}" --test --name "${name}" --outdir "${outdir}" --njobs 8; then
      echo "[WARN] Decode failed: ${name}"
      echo "${name}: DECODE_FAILED" >> "${summary_file}"
      continue
    fi

    if [ ! -f "${csv}" ]; then
      echo "[WARN] Decode output not found: ${csv}"
      echo "${name}: CSV_MISSING" >> "${summary_file}"
      continue
    fi

    awk -F'\t' 'NR>1{print $1" "$2}' "${csv}" > "${hyp}"
    awk -F'\t' 'NR>1{print $1" "$3}' "${csv}" > "${ref}"

    wer_output=$(python tools/compute-wer.py --v=0 "${ref}" "${hyp}" || true)
    echo "${wer_output}"
    wer_line=$(echo "${wer_output}" | rg "Overall" | tail -n 1 || true)
    if [ -z "${wer_line}" ]; then
      wer_line="Overall -> N/A"
    fi
    echo "${name}: ${wer_line}" | tee -a "${summary_file}"
  done

  echo "stage 5 done. WER summary -> ${summary_file}"
fi


# add LM (optional)
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  python main.py --config config/libri/lm_example.yaml --lm
fi
