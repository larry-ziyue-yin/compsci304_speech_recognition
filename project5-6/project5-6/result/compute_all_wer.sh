#!/bin/bash
# 从 result/*_test_output.csv 生成 hyp/ref 并计算 WER
# 在 project5-6 目录下执行: bash result/compute_all_wer.sh

set -e
cd "$(dirname "$0")/.."
outdir=result
summary="${outdir}/wer_summary.txt"
: > "${summary}"

for name in p6_ctc_b1 p6_ctc_b20 p6_att_b1 p6_att_b20 p6_joint_b20; do
  csv="${outdir}/${name}_test_output.csv"
  hyp="${outdir}/${name}_hyp.txt"
  ref="${outdir}/${name}_ref.txt"

  if [ ! -f "${csv}" ]; then
    echo "[WARN] 未找到: ${csv}"
    echo "${name}: CSV_MISSING" >> "${summary}"
    continue
  fi

  # CSV 格式: idx \t hyp \t truth → hyp/ref 格式: utt_id word1 word2 ...
  awk -F'\t' 'NR>1{print $1" "$2}' "${csv}" > "${hyp}"
  awk -F'\t' 'NR>1{print $1" "$3}' "${csv}" > "${ref}"

  echo "--- ${name} ---"
  python tools/compute-wer.py --v=0 "${ref}" "${hyp}" || true
  wer_line=$(python tools/compute-wer.py --v=0 "${ref}" "${hyp}" 2>/dev/null | grep "Overall" | tail -n 1 || echo "Overall -> N/A")
  echo "${name}: ${wer_line}" | tee -a "${summary}"
  echo ""
done

echo "WER 汇总已写入: ${summary}"
cat "${summary}"
