#!/bin/bash
# Run project6 decoding: CTC b1/b20, Attention b1/b20 (4 configs)

set -e
cd "$(dirname "$0")"

echo "=== p6_ctc_b1 ==="
python main.py --config config/libri/p6_ctc_b1.yaml  --test --name p6_ctc_b1  --outdir result --njobs 8

echo "=== p6_ctc_b20 ==="
python main.py --config config/libri/p6_ctc_b20.yaml --test --name p6_ctc_b20 --outdir result --njobs 8

echo "=== p6_att_b1 ==="
python main.py --config config/libri/p6_att_b1.yaml  --test --name p6_att_b1  --outdir result --njobs 8

echo "=== p6_att_b20 ==="
python main.py --config config/libri/p6_att_b20.yaml --test --name p6_att_b20 --outdir result --njobs 8

echo "=== p6_joint_b20 (Joint CTC-Attention beam=20) ==="
python main.py --config config/libri/p6_joint_b20.yaml --test --name p6_joint_b20 --outdir result --njobs 8

echo "=== All 5 decoding runs finished ==="
