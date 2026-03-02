#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 decode 输出的 CSV 生成 compute-wer 所需的 hyp 文件。
CSV 格式: idx/host_path\\thyp\\ttruth
输出 hyp 格式: utt_id word1 word2 ... (与 ref 的 utt_id 对应)
utt_id 从第一列提取：若为路径则取 basename 去掉扩展名，否则用原值。
"""
import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: python prepare_wer_input.py <decode_output.csv> <hyp.out> [ref.txt]",
              file=sys.stderr)
        print("  Reads decode CSV (header: idx\\thyp\\ttruth), writes hyp file with utt_id.",
              file=sys.stderr)
        print("  If ref.txt is given, also writes a ref file with same utt_id order (from ref content in CSV).",
              file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    hyp_path = sys.argv[2]
    write_ref = len(sys.argv) > 3
    ref_path = sys.argv[3] if write_ref else None

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("Empty CSV", file=sys.stderr)
        sys.exit(1)

    # skip header
    out_lines = []
    for line in lines[1:]:
        line = line.rstrip("\n")
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        key, hyp = parts[0], parts[1]
        # utt_id: from path take stem, else use key
        if os.path.sep in key or "/" in key:
            utt_id = Path(key).stem
        else:
            utt_id = key
        out_lines.append("{} {}".format(utt_id, hyp))

    with open(hyp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print("Wrote {} ({} lines)".format(hyp_path, len(out_lines)))


if __name__ == "__main__":
    main()
