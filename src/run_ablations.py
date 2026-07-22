#!/usr/bin/env python3
"""消融与佐证实验（包络谱 1D-CNN）。

A. 低数据佐证：每记录仅取前 K 个窗口，比较 R vs S 准确率差距是否放大。
B. 分组消融：协议 S 的分组变量换成 load / fault_size，看泄漏控制效果。
C. 带通消融：包络谱去掉共振带带通，验证物理一致性增益是否来自带通。

产出：results/tables/ablations.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import config
import datautil
import envelope
import models
import engine
import gradcam
import fault_freq


def _train_eval(specs, y, rpm, freqs, tr, te, seed, epochs=15, do_pc=False):
    tr = np.asarray(tr); te = np.asarray(te)
    mean, std = specs[tr].mean(), specs[tr].std()
    xtr = torch.from_numpy((specs[tr]-mean)/(std+1e-8)).unsqueeze(1).float()
    xte = torch.from_numpy((specs[te]-mean)/(std+1e-8)).unsqueeze(1).float()
    ytr = torch.from_numpy(y[tr]).long(); yte = torch.from_numpy(y[te]).long()
    engine.set_seed(seed)
    m = models.build_model("env_cnn", in_len=specs.shape[1])
    crit = torch.nn.CrossEntropyLoss(); opt = torch.optim.Adam(m.parameters(), lr=config.LR)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(xte).argmax(1).numpy()
    acc = accuracy_score(yte.numpy(), pred)
    f1 = f1_score(yte.numpy(), pred, average="macro")
    margin = None
    if do_pc:
        cam = gradcam.GradCAM1D(m)
        mg = []
        for i in range(min(300, len(te))):
            yi = int(yte[i]); r = int(rpm[te[i]])
            heat, p = cam(xte[i:i+1], class_idx=yi)
            if p != yi:
                continue
            tb = fault_freq.fault_bands(r); nb = fault_freq.shifted_bands(r)
            mg.append(gradcam.band_fraction_1d(freqs, heat, tb)
                      - gradcam.band_fraction_1d(freqs, heat, nb))
        margin = float(np.mean(mg)) if mg else None
    return acc, f1, margin


def split_random(y, seed):
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); tr, te = [], []
    for c in np.unique(y):
        ci = idx[y == c]; rng.shuffle(ci); n = len(ci); nt = max(1, int(n*0.3))
        te += ci[:nt].tolist(); tr += ci[nt:].tolist()
    return tr, te


def split_group(y, group, seed):
    """按任意分组变量（group 数组）做记录级划分。"""
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); tr, te = [], []
    for c in np.unique(y):
        cm = y == c; groups = np.unique(group[cm]); rng.shuffle(groups)
        ng = len(groups)
        if ng >= 2:
            nt = max(1, int(round(ng*0.3)))
            tg = set(groups[:nt].tolist())
        else:
            tg = set()  # 无法分组则全进 train（该类退化）
        for g in groups:
            gi = idx[cm & (group == g)].tolist()
            if g in tg:
                te += gi
            else:
                tr += gi
    tr = list(set(tr) - set(te))
    return tr, te


def ablation_lowdata(specs, y, rpm, rec, freqs, out):
    print("=== A. low-data corroboration ===")
    res = {}
    for k in (10, 25, 50):
        keep = []
        for r in np.unique(rec):
            keep.extend(np.where(rec == r)[0][:k].tolist())
        keep = np.array(keep)
        yk = y[keep]; reck = rec[keep]
        accs_r, accs_s = [], []
        for seed in config.SEEDS:
            # R: 随机
            tr, te = split_random(yk, seed)
            a, _, _ = _train_eval(specs, y, rpm, freqs, keep[tr], keep[te], seed)
            accs_r.append(a)
            # S: 记录级
            tr, te = split_group(yk, reck, seed)
            a, _, _ = _train_eval(specs, y, rpm, freqs, keep[tr], keep[te], seed)
            accs_s.append(a)
        res[f"k={k}"] = {
            "R_acc_mean": float(np.mean(accs_r)), "S_acc_mean": float(np.mean(accs_s)),
            "gap": float(np.mean(accs_r) - np.mean(accs_s)),
        }
        print(f"  k={k}: R={np.mean(accs_r):.3f} S={np.mean(accs_s):.3f} gap={np.mean(accs_r)-np.mean(accs_s):.3f}")
    out["lowdata"] = res


def ablation_grouping(specs, y, rpm, rec, load, size, freqs, out):
    print("=== B. grouping variable ablation (3 seeds) ===")
    res = {}
    for name, grp in [("recording", rec), ("load", load),
                      ("fault_size", (size*1000).astype(int))]:
        accs, f1s = [], []
        for seed in config.SEEDS:
            tr, te = split_group(y, grp, seed)
            a, f, _ = _train_eval(specs, y, rpm, freqs, tr, te, seed)
            accs.append(a); f1s.append(f)
        res[name] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                     "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))}
        print(f"  group by {name}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
              f"f1={np.mean(f1s):.3f}")
    out["grouping"] = res


def ablation_bandpass(y, rpm, rec, freqs_env, out, dataset=config.DEFAULT_DATASET):
    print("=== C. bandpass ablation (envelope w/o resonance bandpass, 3 seeds) ===")
    data = datautil.load_windows(dataset)
    specs_nobp, freqs_nobp = envelope.batch_envelope(data["X"], apply_bandpass=False)
    accs, f1s, mgs = [], [], []
    for seed in config.SEEDS:
        tr, te = split_group(y, rec, seed)
        a, f, mg = _train_eval(specs_nobp, y, rpm, freqs_nobp, tr, te, seed, do_pc=True)
        accs.append(a); f1s.append(f)
        if mg is not None:
            mgs.append(mg)
    res = {"no_bandpass": {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "pc_margin_mean": float(np.mean(mgs)) if mgs else None,
        "pc_margin_std": float(np.std(mgs)) if mgs else None}}
    print(f"  no-bandpass S: acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"margin={np.mean(mgs) if mgs else None}")
    out["bandpass"] = res


def main(dataset=config.DEFAULT_DATASET):
    specs, freqs = datautil.get_envelope(dataset=dataset)
    data = datautil.load_windows(dataset)
    y = data["y"]; rpm = data["rpm"]; rec = data["recording_id"]
    load = data["load"]; size = data["fault_size"]

    out = {"dataset": dataset}
    ablation_lowdata(specs, y, rpm, rec, freqs, out)
    ablation_grouping(specs, y, rpm, rec, load, size, freqs, out)
    ablation_bandpass(y, rpm, rec, freqs, out, dataset=dataset)

    os.makedirs(config.TABLES_DIR, exist_ok=True)
    prefix = "" if dataset == config.DEFAULT_DATASET else f"{dataset}_"
    outpath = os.path.join(config.TABLES_DIR, f"{prefix}ablations.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ablations:{dataset}] wrote {outpath}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET,
                    choices=list(config.DATASETS.keys()))
    args = ap.parse_args()
    main(args.dataset)
