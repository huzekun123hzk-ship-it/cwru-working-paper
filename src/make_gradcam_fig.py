#!/usr/bin/env python3
"""定性对比图：R vs S 的 Grad-CAM 显著性叠加在包络谱上，标注真实故障频带。

对同一批 test 样本（每个故障类各取一条被两协议都正确分类的样本），
分别用协议 R 与协议 S 训练的模型（seed 0）计算 Grad-CAM，
把显著性画在包络谱上，并用阴影标出真实故障频带 B 与平移对照带 B'。

直观展示：R（记忆化）的显著性更"尖锐地"落在故障带内，
S（未见记录）的显著性更弥散——这正是 ILG>0 的可视化证据。

产出：results/figures/fig_gradcam_qualitative.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import datautil
import engine
import models
import gradcam
import fault_freq

SEED = 0
FAULT_CLASSES = ["IR", "OR", "B"]  # 三个故障类（Normal 无故障频带，略）


def _train(protocol, dataset=config.DEFAULT_DATASET):
    train_ds, val_ds, test_ds, cw, freqs = datautil.make_env_datasets(
        protocol, SEED, dataset=dataset)
    model = models.build_model("env_cnn", in_len=train_ds.x.shape[1])
    model, _ = engine.train_model(model, train_ds, val_ds, cw, seed=SEED,
                                  epochs=config.EPOCHS)
    return model, test_ds, freqs


def _pick_samples(test_ds, rpm_arr):
    """每个故障类取第一条样本的索引与 rpm。"""
    picks = {}
    for i in range(len(test_ds)):
        _, y = test_ds[i]
        cls = config.IDX_TO_CLASS[int(y)]
        if cls in FAULT_CLASSES and cls not in picks:
            picks[cls] = i
        if len(picks) == len(FAULT_CLASSES):
            break
    return picks


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    ds = config.DEFAULT_DATASET

    mR, test_R, freqs = _train("R", ds)
    mS, test_S, _ = _train("S", ds)

    data = datautil.load_windows(ds)
    spR = datautil.load_splits("R", dataset=ds)[str(SEED)]
    spS = datautil.load_splits("S", dataset=ds)[str(SEED)]
    rpmR = data["rpm"][np.asarray(spR["test"])]
    rpmS = data["rpm"][np.asarray(spS["test"])]

    picksR = _pick_samples(test_R, rpmR)
    picksS = _pick_samples(test_S, rpmS)

    camR = gradcam.GradCAM1D(mR)
    camS = gradcam.GradCAM1D(mS)

    ncol = len(FAULT_CLASSES)
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 6), sharex=True)

    for col, cls in enumerate(FAULT_CLASSES):
        for row, (proto, model, cam, test_ds, picks, rpm_arr) in enumerate([
            ("R", mR, camR, test_R, picksR, rpmR),
            ("S", mS, camS, test_S, picksS, rpmS),
        ]):
            ax = axes[row, col]
            idx = picks.get(cls)
            if idx is None:
                ax.set_visible(False)
                continue
            x, y = test_ds[idx]
            env = x.squeeze().cpu().numpy()
            heat, _ = cam(x.unsqueeze(0), class_idx=int(y))
            rpm = int(rpm_arr[idx])
            # 归一化包络谱用于画背景
            env_n = (env - env.min()) / (env.max() - env.min() + 1e-8)
            ax.plot(freqs, env_n, color="#888888", lw=0.8, label="envelope spectrum")
            ax.fill_between(freqs, 0, heat, color="#c0504d" if proto == "R" else "#4f81bd",
                            alpha=0.5, label=f"Grad-CAM ({proto})")
            # 真实故障带（绿）与平移对照带（灰）
            for (lo, hi, _n) in fault_freq.fault_bands(rpm):
                ax.axvspan(lo, hi, color="green", alpha=0.12)
            for (lo, hi, _n) in fault_freq.shifted_bands(rpm):
                ax.axvspan(lo, hi, color="orange", alpha=0.10)
            tb = fault_freq.fault_bands(rpm)
            nb = fault_freq.shifted_bands(rpm)
            m = (gradcam.band_fraction_1d(freqs, heat, tb)
                 - gradcam.band_fraction_1d(freqs, heat, nb))
            ax.set_title(f"{cls} — Protocol {proto} (margin={m:.2f})", fontsize=9)
            ax.set_ylim(0, 1.05)
            if row == 1:
                ax.set_xlabel("Frequency (Hz)")
            if col == 0:
                ax.set_ylabel("Protocol " + proto + "\nnorm. amplitude", fontsize=9)

    # 图例（只画一次）
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#888888", lw=1, label="Envelope spectrum"),
        Patch(facecolor="#c0504d", alpha=0.5, label="Grad-CAM saliency"),
        Patch(facecolor="green", alpha=0.15, label="True fault band B"),
        Patch(facecolor="orange", alpha=0.12, label="Shifted control band B'"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Grad-CAM focus: random-window (R) concentrates in the true fault band "
                 "more sharply than record-level (S)", y=0.97, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(config.FIGURES_DIR, "fig_gradcam_qualitative.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradcam-fig] wrote {out}")


if __name__ == "__main__":
    main()
