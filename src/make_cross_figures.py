#!/usr/bin/env python3
"""跨数据集对比图：DE vs FE 的准确率差距与 ILG 并排展示。

读取 summary.json (DE) 和 FE_summary.json，产出：
  fig_cross_dataset.png —— 两数据集的 generalization gap 与 ILG 条形对比
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def _load(name):
    with open(os.path.join(config.TABLES_DIR, name)) as f:
        return json.load(f)


def main():
    de = _load("summary.json")
    fe = _load("FE_summary.json")
    datasets = ["DE\n(drive-end)", "FE\n(fan-end)"]
    gaps = [de["generalization_gap"], fe["generalization_gap"]]
    ilgs = [de["ILG"], fe["ILG"]]

    x = np.arange(len(datasets)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, gaps, w, label="Generalization gap (acc R - acc S)",
                color="#c0504d")
    b2 = ax.bar(x + w/2, ilgs, w, label="ILG (margin R - margin S)", color="#4f81bd")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("Effect size")
    ax.set_title("Leakage effects transfer across sensor locations\n(both positive on DE and FE)")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", xy=(b.get_x()+b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.FIGURES_DIR, "fig_cross_dataset.png"), dpi=150)
    plt.close(fig)
    print("[cross-fig] wrote fig_cross_dataset.png")
    print(f"  DE: gap={de['generalization_gap']:.3f} ILG={de['ILG']:.3f}")
    print(f"  FE: gap={fe['generalization_gap']:.3f} ILG={fe['ILG']:.3f}")


if __name__ == "__main__":
    main()
