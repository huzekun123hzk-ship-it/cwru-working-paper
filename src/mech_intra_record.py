#!/usr/bin/env python3
"""机理验证A：量化同一记录内相邻窗口的相似度。

论证"为什么随机窗口划分会泄漏"：若同一记录相邻窗口高度相似，
随机划分会把近乎相同的窗口分到 train 和 test，模型可靠记忆而非泛化。

度量：
  - 相邻窗口时域皮尔逊相关系数
  - 相邻窗口频谱(幅值谱)余弦相似度
对比：同记录相邻 vs 跨记录随机配对（应显著更低）。

产出：results/tables/intra_record_similarity.json + 一张分布图。
"""

from __future__ import annotations

import json
import os

import numpy as np

import config
import datautil


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-12 else 0.0


def _spec_cosine(a, b):
    fa = np.abs(np.fft.rfft(a)); fb = np.abs(np.fft.rfft(b))
    denom = np.linalg.norm(fa) * np.linalg.norm(fb)
    return float(fa @ fb / denom) if denom > 1e-12 else 0.0


def main():
    data = datautil.load_windows()
    X = data["X"]; rec = data["recording_id"]
    rng = np.random.RandomState(0)

    adj_corr, adj_cos = [], []
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        for i in range(len(idx) - 1):
            a, b = X[idx[i]], X[idx[i + 1]]
            adj_corr.append(_pearson(a, b))
            adj_cos.append(_spec_cosine(a, b))

    # 跨记录随机配对对照
    cross_corr, cross_cos = [], []
    n_pairs = min(len(adj_corr), 5000)
    for _ in range(n_pairs):
        i, j = rng.randint(0, len(X), size=2)
        if rec[i] == rec[j]:
            continue
        cross_corr.append(_pearson(X[i], X[j]))
        cross_cos.append(_spec_cosine(X[i], X[j]))

    summary = {
        "adjacent_same_record": {
            "time_corr_mean": float(np.mean(adj_corr)),
            "time_corr_std": float(np.std(adj_corr)),
            "spec_cosine_mean": float(np.mean(adj_cos)),
            "spec_cosine_std": float(np.std(adj_cos)),
            "n": len(adj_corr),
        },
        "cross_record_random": {
            "time_corr_mean": float(np.mean(cross_corr)),
            "time_corr_std": float(np.std(cross_corr)),
            "spec_cosine_mean": float(np.mean(cross_cos)),
            "spec_cosine_std": float(np.std(cross_cos)),
            "n": len(cross_corr),
        },
    }

    os.makedirs(config.TABLES_DIR, exist_ok=True)
    out = os.path.join(config.TABLES_DIR, "intra_record_similarity.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("[mech A] adjacent same-record:")
    print(f"  time corr   = {summary['adjacent_same_record']['time_corr_mean']:.3f} "
          f"± {summary['adjacent_same_record']['time_corr_std']:.3f}")
    print(f"  spec cosine = {summary['adjacent_same_record']['spec_cosine_mean']:.3f} "
          f"± {summary['adjacent_same_record']['spec_cosine_std']:.3f}")
    print("[mech A] cross-record random:")
    print(f"  time corr   = {summary['cross_record_random']['time_corr_mean']:.3f} "
          f"± {summary['cross_record_random']['time_corr_std']:.3f}")
    print(f"  spec cosine = {summary['cross_record_random']['spec_cosine_mean']:.3f} "
          f"± {summary['cross_record_random']['spec_cosine_std']:.3f}")

    _plot(adj_cos, cross_cos)
    print(f"[mech A] wrote {out}")


def _plot(adj_cos, cross_cos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(adj_cos, bins=40, alpha=0.6, label="Adjacent, same recording", density=True)
    ax.hist(cross_cos, bins=40, alpha=0.6, label="Random, cross recording", density=True)
    ax.set_xlabel("Spectral cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Intra-record window similarity (leakage source)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.FIGURES_DIR, "fig_intra_record_similarity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
