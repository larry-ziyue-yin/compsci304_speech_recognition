# 5. Joint CTC-Attention Beam Decoding (beam size: 20)

## 配置要点（`config/libri/p6_joint_b20.yaml`）

- **src.ckpt**：指向「同时带 CTC + Attention」的 checkpoint。  
  - 正确做法：用 `config/libri/asr_hybrid.yaml` 训练一版，得到 `ckpt/asr_hybrid_sd0/best_att.pth`，此处就填这个路径。  
  - 若你目前只有 asr_att：可暂时填 `ckpt/asr_att_sd0/best_att.pth`，并把 **src.config** 改为 `config/libri/asr_att.yaml` 试跑（此时 CTC 分支未训练，效果可能不如纯 att）。
- **src.config**：必须和训练该 checkpoint 时用的 config 一致（保证 tokenizer/特征一致）。
- **decode**：  
  - `beam_size: 20`  
  - `ctc_weight: 0.3`（常用 0.3～0.5）  
  - `lm_weight: 0.0`（未用外部 LM 则保持 0）

## 如何跑 Stage 5 并补 WER 表

在 **project5-6** 目录下：

```bash
# 只跑 Joint 解码（若前 4 个已跑过）
python main.py --config config/libri/p6_joint_b20.yaml --test --name p6_joint_b20 --outdir result --njobs 8

# 或一次性跑齐 5 个（含 Joint）
./run_p6_decode.sh
```

解码会生成 `result/p6_joint_b20_test_output.csv`。

算 WER 并更新汇总表：

```bash
bash result/compute_all_wer.sh
```

脚本会从各 `*_test_output.csv` 生成 hyp/ref，调用 `tools/compute-wer.py`，并把 5 个配置的 WER 写入 `result/wer_summary.txt`。表里会多出一行 `p6_joint_b20: Overall -> xx.xx % ...`。

## 若还没有 hybrid 的 ckpt

先训练一版 hybrid，再解码：

```bash
python main.py --config config/libri/asr_hybrid.yaml
```

训练结束后会得到 `ckpt/asr_hybrid_sd0/best_att.pth`，`p6_joint_b20.yaml` 里保持 `src.ckpt: 'ckpt/asr_hybrid_sd0/best_att.pth'`、`src.config: 'config/libri/asr_hybrid.yaml'` 即可。
