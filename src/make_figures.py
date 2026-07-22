#!/usr/bin/env python3
"""汇总图表生成：读取 results/tables/*.json，产出论文用图。

图1 记录内相似度分布（已由 mech_intra_record 生成）
图2 R vs S 准确率与物理一致性对比（条形+误差棒）
图3 准确率 × 物理一致性 margin 的 2×2 部署决策矩阵散点
图4 混淆矩阵（R vs S 各一，取 seed0）
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


def fig_bar(summary):
    R, S = summary["R"], summary["S"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Protocol R\n(random)", "Protocol S\n(record-level)"]
    # 准确率
    axes[0].bar(labels, [R["accuracy_mean"], S["accuracy_mean"]],
                yerr=[R["accuracy_std"], S["accuracy_std"]], capsize=6,
                color=["#c0504d", "#4f81bd"])
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_ylim(0.8, 1.0)
    axes[0].set_title(f"Accuracy (gap={summary['generalization_gap']:.3f})")
    # margin
    axes[1].bar(labels, [R["pc_margin_mean"], S["pc_margin_mean"]],
                yerr=[R["pc_margin_std"], S["pc_margin_std"]], capsize=6,
                color=["#c0504d", "#4f81bd"])
    axes[1].set_ylabel("Grad-CAM margin (PC_true - PC_neg)")
    axes[1].set_title(f"Interpretability (ILG={summary['ILG']:.3f})")
    fig.suptitle("Leakage inflates BOTH accuracy and interpretability appearance")
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "fig_R_vs_S_bars.png"), dpi=150)
    plt.close(fig)


def fig_decision_matrix(results):
    """每个 (protocol, seed) 点：x=accuracy, y=pc_margin。"""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for key, m in results.items():
        proto = key.split("_")[0]
        color = "#c0504d" if proto == "R" else "#4f81bd"
        marker = "o" if proto == "R" else "s"
        ax.scatter(m["accuracy"], m["pc_margin"], c=color, marker=marker, s=90,
                   edgecolors="k", zorder=3)
    # 象限分隔线（用中位数附近的参考线）
    ax.axvline(0.93, color="gray", ls="--", lw=1)
    ax.axhline(0.20, color="gray", ls="--", lw=1)
    ax.set_xlabel("Test accuracy")
    ax.set_ylabel("Grad-CAM margin (physical focus)")
    ax.set_title("Deployment decision matrix\n(R inflates focus appearance via memorization)")
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0504d",
                     markeredgecolor="k", markersize=10, label="Protocol R (random)"),
              Line2D([0], [0], marker="s", color="w", markerfacecolor="#4f81bd",
                     markeredgecolor="k", markersize=10, label="Protocol S (record-level)")]
    ax.legend(handles=legend, loc="lower left")
    ax.annotate("high acc + high apparent focus\n(inflated by leakage)",
                xy=(0.955, 0.256), xytext=(0.9, 0.28), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "fig_decision_matrix.png"), dpi=150)
    plt.close(fig)


def fig_confusion(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, proto in zip(axes, ("R", "S")):
        cm = np.array(results[f"{proto}_0"]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(config.N_CLASSES)); ax.set_yticks(range(config.N_CLASSES))
        ax.set_xticklabels(config.CLASSES); ax.set_yticklabels(config.CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Protocol {proto} (seed0)")
        for i in range(config.N_CLASSES):
            for j in range(config.N_CLASSES):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.046)
    fig.suptitle("Confusion matrices (row-normalized)")
    fig.savefig(os.path.join(config.FIGURES_DIR, "fig_confusion.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    summary = _load("summary.json")
    results = _load("main_results.json")
    fig_bar(summary)
    fig_decision_matrix(results)
    fig_confusion(results)
    print("[figures] wrote fig_R_vs_S_bars, fig_decision_matrix, fig_confusion")


if __name__ == "__main__":
    main()
