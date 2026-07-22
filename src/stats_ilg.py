#!/usr/bin/env python3
"""ILG 的统计推断：配对检验 + bootstrap 置信区间（10 种子）。

对每个数据集：
  - 读取 main_results.json 的逐种子 pc_margin（R 与 S 按同一 seed 配对）；
  - 配对差 d_s = margin(R,s) - margin(S,s)，其均值即 ILG；
  - 报告：ILG、配对 t 检验 p 值、Wilcoxon 符号秩 p 值、
    以及对种子做 bootstrap 得到的 ILG 95% 置信区间；
  - 报告 accuracy 泛化差距的同类统计。

产出：results/tables/{prefix}stats.json
不重训，只消费已存结果，保证与主实验完全一致。
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

import config


def _paired(vals_r, vals_s, n_boot=10000, seed=0):
    d = np.asarray(vals_r) - np.asarray(vals_s)
    mean = float(np.mean(d))
    # 配对 t 检验（单样本对差）
    t_p = float(stats.ttest_rel(vals_r, vals_s).pvalue)
    # Wilcoxon 符号秩（非参数，稳健）
    try:
        w_p = float(stats.wilcoxon(vals_r, vals_s).pvalue)
    except ValueError:
        w_p = None  # 全部差相同符号且相等时会抛错
    # bootstrap 对种子重采样
    rng = np.random.RandomState(seed)
    boots = []
    n = len(d)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boots.append(np.mean(d[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n_pos = int(np.sum(d > 0))
    return {
        "mean": mean,
        "n_seeds": n,
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_seeds_positive": n_pos,
        "complete_separation": n_pos == n,
    }


def _load(prefix):
    path = os.path.join(config.TABLES_DIR, f"{prefix}main_results.json")
    with open(path) as f:
        return json.load(f)


def run_one(dataset):
    prefix = "" if dataset == config.DEFAULT_DATASET else f"{dataset}_"
    res = _load(prefix)
    seeds = config.SEEDS
    r_margin = [res[f"R_{s}"]["pc_margin"] for s in seeds]
    s_margin = [res[f"S_{s}"]["pc_margin"] for s in seeds]
    r_acc = [res[f"R_{s}"]["accuracy"] for s in seeds]
    s_acc = [res[f"S_{s}"]["accuracy"] for s in seeds]

    out = {
        "dataset": dataset,
        "n_seeds": len(seeds),
        "ILG": _paired(r_margin, s_margin),
        "generalization_gap": _paired(r_acc, s_acc),
    }
    with open(os.path.join(config.TABLES_DIR, f"{prefix}stats.json"), "w") as f:
        json.dump(out, f, indent=2)

    ilg = out["ILG"]
    gap = out["generalization_gap"]
    print(f"[stats:{dataset}] ILG = {ilg['mean']:.4f}  "
          f"95% CI [{ilg['ci95_low']:.4f}, {ilg['ci95_high']:.4f}]  "
          f"paired-t p={ilg['paired_t_p']:.2e}  "
          f"Wilcoxon p={ilg['wilcoxon_p']}  "
          f"separation {ilg['n_seeds_positive']}/{ilg['n_seeds']}")
    print(f"[stats:{dataset}] gap = {gap['mean']:.4f}  "
          f"95% CI [{gap['ci95_low']:.4f}, {gap['ci95_high']:.4f}]  "
          f"paired-t p={gap['paired_t_p']:.2e}  "
          f"separation {gap['n_seeds_positive']}/{gap['n_seeds']}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=list(config.DATASETS.keys()) + ["all"])
    args = ap.parse_args()
    datasets = list(config.DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_one(ds)


if __name__ == "__main__":
    main()
