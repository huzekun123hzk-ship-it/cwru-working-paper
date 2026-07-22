#!/usr/bin/env python3
"""主实验：包络谱 1D-CNN，协议 R vs S × 3 seeds。

产出：
  results/tables/main_results.json  —— 每 (protocol, seed) 分类指标 + 物理一致性
  results/tables/summary.json       —— 跨种子均值±标准差、ILG、泛化差距
  results/logs/run_log.txt

核心指标：
  PC_true : Grad-CAM 显著性落在故障频带内的占比（物理一致性）
  PC_neg  : 频带平移负对照占比
  ILG     : 可解释性泄漏差距 = (PC_true-PC_neg)@S - (PC_true-PC_neg)@R

用法：
    python src/run_experiments.py           # 3 seeds
    python src/run_experiments.py --quick   # 单种子冒烟
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import config
import datautil
import engine
import models
import gradcam


def physical_consistency_testset(model, test_ds, rpm_arr, freqs, max_samples=400):
    """在 test 集正确分类样本上平均物理一致性与负对照。"""
    pc_true, pc_neg = [], []
    n = min(len(test_ds), max_samples)
    for i in range(n):
        x, y = test_ds[i]
        ft, fn, pred = gradcam.physical_consistency_1d(
            model, x.unsqueeze(0), int(rpm_arr[i]), freqs, class_idx=int(y))
        if pred == int(y):  # 仅正确分类样本
            pc_true.append(ft)
            pc_neg.append(fn)
    if not pc_true:
        return 0.0, 0.0
    return float(np.mean(pc_true)), float(np.mean(pc_neg))


def run_one(protocol, seed, epochs, log, dataset=config.DEFAULT_DATASET):
    train_ds, val_ds, test_ds, cw, freqs = datautil.make_env_datasets(
        protocol, seed, dataset=dataset)
    data = datautil.load_windows(dataset)
    sp = datautil.load_splits(protocol, dataset=dataset)[str(seed)]
    rpm_arr = data["rpm"][np.asarray(sp["test"])]

    model = models.build_model("env_cnn", in_len=train_ds.x.shape[1])
    model, hist = engine.train_model(model, train_ds, val_ds, cw, seed=seed, epochs=epochs)
    metrics = engine.evaluate(model, engine.make_test_loader(test_ds))
    pc_true, pc_neg = physical_consistency_testset(model, test_ds, rpm_arr, freqs)
    metrics["phys_consistency_true"] = pc_true
    metrics["phys_consistency_negctrl"] = pc_neg
    metrics["pc_margin"] = pc_true - pc_neg
    msg = (f"[{dataset} {protocol} seed{seed}] acc={metrics['accuracy']:.4f} "
           f"macroF1={metrics['macro_f1']:.4f} "
           f"PC_true={pc_true:.3f} PC_neg={pc_neg:.3f} margin={pc_true-pc_neg:.3f}")
    print(msg); log.append(msg)
    return metrics


def aggregate(results, protocol, seeds):
    def col(k):
        return [results[f"{protocol}_{s}"][k] for s in seeds]
    out = {}
    for k in ("accuracy", "macro_f1", "phys_consistency_true",
              "phys_consistency_negctrl", "pc_margin"):
        vals = col(k)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def _tbl(name, dataset):
    prefix = "" if dataset == config.DEFAULT_DATASET else f"{dataset}_"
    return os.path.join(config.TABLES_DIR, f"{prefix}{name}")


def summarize_and_write(results, seeds, log, dataset=config.DEFAULT_DATASET):
    agg = {"R": aggregate(results, "R", seeds), "S": aggregate(results, "S", seeds)}
    # 可解释性泄漏差距 ILG = margin(R) - margin(S)
    # margin = PC_true - PC_neg（Grad-CAM 聚焦到故障频带、相对负对照的净额）。
    # ILG > 0 表示随机划分(R)的可解释性"表象"被泄漏虚高（与准确率虚高同向）。
    agg["ILG"] = agg["R"]["pc_margin_mean"] - agg["S"]["pc_margin_mean"]
    agg["ILG_pc_true"] = agg["R"]["phys_consistency_true_mean"] - agg["S"]["phys_consistency_true_mean"]
    agg["generalization_gap"] = agg["R"]["accuracy_mean"] - agg["S"]["accuracy_mean"]
    agg["dataset"] = dataset
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    with open(_tbl("summary.json", dataset), "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[summary:{dataset}] acc  R={agg['R']['accuracy_mean']:.4f}±{agg['R']['accuracy_std']:.4f}  "
          f"S={agg['S']['accuracy_mean']:.4f}±{agg['S']['accuracy_std']:.4f}")
    print(f"[summary:{dataset}] PC_margin R={agg['R']['pc_margin_mean']:.3f}  S={agg['S']['pc_margin_mean']:.3f}")
    print(f"[summary:{dataset}] ILG = margin(R)-margin(S) = {agg['ILG']:.4f}")
    print(f"[summary:{dataset}] generalization gap = {agg['generalization_gap']:.4f}")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    prefix = "" if dataset == config.DEFAULT_DATASET else f"{dataset}_"
    with open(os.path.join(config.LOGS_DIR, f"{prefix}run_log.txt"), "w") as f:
        f.write("\n".join(log))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET,
                    choices=list(config.DATASETS.keys()))
    ap.add_argument("--agg-only", action="store_true",
                    help="从已存的 main_results.json 重算 summary，不重训")
    args = ap.parse_args()
    seeds = [0] if args.quick else config.SEEDS
    epochs = 8 if args.quick else config.EPOCHS
    ds = args.dataset

    if args.agg_only:
        with open(_tbl("main_results.json", ds)) as f:
            results = json.load(f)
        summarize_and_write(results, seeds, [], dataset=ds)
        return

    log = []
    results = {}
    for protocol in ("R", "S"):
        for seed in seeds:
            results[f"{protocol}_{seed}"] = run_one(protocol, seed, epochs, log, dataset=ds)

    with open(_tbl("main_results.json", ds), "w") as f:
        json.dump(results, f, indent=2)
    summarize_and_write(results, seeds, log, dataset=ds)


if __name__ == "__main__":
    main()
