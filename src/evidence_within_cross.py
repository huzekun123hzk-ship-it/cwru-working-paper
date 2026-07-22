#!/usr/bin/env python3
"""加分证据（机理解释）：把主实验的 margin 按"test 记录是否被 train 见过"重新框定。

事实（由 make_splits 的日志与 diagnostics 确证）：
  - 协议 R（随机窗口）：test 中 100% 的 recording 也出现在 train 中（记录被见过）。
  - 协议 S（按记录）：test 中 0% 的 recording 出现在 train 中（记录未见过）。

因此：
  within-record margin  ≡ 协议 R 的 Grad-CAM margin（模型见过该记录）
  cross-record  margin  ≡ 协议 S 的 Grad-CAM margin（模型未见过该记录）

若 within > cross，说明"记录被记忆"直接抬高了 Grad-CAM 聚焦度（虚假物理关注），
这是 ILG>0 的机理解释。本脚本从 main_results.json 汇总，避免重复训练。

产出：results/tables/within_vs_cross_margin.json
"""

from __future__ import annotations

import json
import os

import numpy as np

import config


def main():
    with open(os.path.join(config.TABLES_DIR, "main_results.json")) as f:
        results = json.load(f)

    r_margin = [results[f"R_{s}"]["pc_margin"] for s in config.SEEDS]
    s_margin = [results[f"S_{s}"]["pc_margin"] for s in config.SEEDS]

    summary = {
        "framing": "R test recordings are 100% seen in train; S test recordings are 0% seen.",
        "within_record_seen": {
            "protocol": "R",
            "margin_mean": float(np.mean(r_margin)),
            "margin_std": float(np.std(r_margin)),
            "n_seeds": len(r_margin),
        },
        "cross_record_unseen": {
            "protocol": "S",
            "margin_mean": float(np.mean(s_margin)),
            "margin_std": float(np.std(s_margin)),
            "n_seeds": len(s_margin),
        },
        "delta_within_minus_cross": float(np.mean(r_margin) - np.mean(s_margin)),
    }
    with open(os.path.join(config.TABLES_DIR, "within_vs_cross_margin.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[within/cross margin] (from main_results, no retrain)")
    print(f"  within-record (R, seen)   margin = {summary['within_record_seen']['margin_mean']:.3f}"
          f" ± {summary['within_record_seen']['margin_std']:.3f}")
    print(f"  cross-record  (S, unseen) margin = {summary['cross_record_unseen']['margin_mean']:.3f}"
          f" ± {summary['cross_record_unseen']['margin_std']:.3f}")
    print(f"  delta (memorization inflation) = {summary['delta_within_minus_cross']:.3f}")


if __name__ == "__main__":
    main()
