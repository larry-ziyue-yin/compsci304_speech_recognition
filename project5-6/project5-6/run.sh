#!/bin/bash

# Assignment runner (LibriSpeech, stages 0-5)
# Task1: CTC-only (ctc_weight=1.0)
# Task2: Attention-only (ctc_weight=0.0)
# Stage 5: Decoding and WER (CTC b1/b20, Attention b1/b20, Joint CTC-Att b20)

export CUDA_VISIBLE_DEVICES="0"
stage=0
stop_stage=5
task=ctc
datadir=
wave_data=data
outdir=result

. tools/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

nbpe=5000
bpemode=bpe
train_set=train_100
dev_set=dev

if [ "${task}" = "ctc" ]; then
  asr_config=config/libri/asr_ctc.yaml
elif [ "${task}" = "att" ]; then
  asr_config=config/libri/asr_att.yaml
else
  echo "[ERROR] Unsupported --task ${task}. Use --task ctc or --task att."
  exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Data preparation"
  for part in dev-clean test-clean train-clean-100; do
    local/data_prep_torchaudio.sh ${datadir}/LibriSpeech/${part} $wave_data/${part//-/_}
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Merge train/dev splits"
  mkdir -p $wave_data/train_100
  for set in train_clean_100; do
    for f in `ls $wave_data/$set`; do
      cat $wave_data/$set/$f >> $wave_data/train_100/$f
    done
  done
  mkdir -p $wave_data/dev
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
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: (optional) extra subword vocab"
  python3 tools/generate_vocab_file.py \
    --input_file $wave_data/lang_char/input.txt \
    --mode subword \
    --output_file $wave_data/subword-16k.model
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: ASR training (${task}) with ${asr_config}"
  python main.py --config ${asr_config}
fi

# Stage 5: Decoding and WER (requires one hybrid model: train with asr_example.yaml first to get best_ctc.pth & best_att.pth)
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: Decoding and WER on test set"
  ref_file=${wave_data}/test_clean/text
  if [ ! -f "$ref_file" ]; then
    echo "[ERROR] Ref file not found: $ref_file. Run stage 0-1 first."
    exit 1
  fi
  for decode_cfg in config/libri/decode_ctc_b1.yaml config/libri/decode_ctc_b20.yaml \
                    config/libri/decode_att_b1.yaml config/libri/decode_att_b20.yaml \
                    config/libri/decode_joint_b20.yaml; do
    name=$(basename "$decode_cfg" .yaml)
    echo "  Decode: $name"
    python main.py --config "$decode_cfg" --test --outdir "$outdir" --njobs 8
    csv="${outdir}/${name}_test_output.csv"
    hyp="${outdir}/${name}_hyp.txt"
    if [ -f "$csv" ]; then
      python tools/prepare_wer_input.py "$csv" "$hyp"
      echo "  WER ($name):"
      python tools/compute-wer.py --v=0 "$ref_file" "$hyp"
    else
      echo "[WARN] Decode output not found: $csv"
    fi
  done
  echo "stage 5 done. WER summaries above."
fi
