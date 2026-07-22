#!/usr/bin/env python3
"""诊断2：低数据 + 跨载荷设置，检验泄漏在准确率上的显现。

D. 低数据：每记录仅取前 K 个窗口，随机窗口划分 vs 按记录划分
E. 跨载荷：train=load{0,1,2}, test=load{3}（随机窗口内部仍可泄漏 vs 记录级）
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import config
import datautil
import models
import engine


def train_eval(specs, y, tr, te, seed, n_classes, epochs=8):
    tr = np.asarray(tr); te = np.asarray(te)
    mean, std = specs[tr].mean(), specs[tr].std()
    xtr = torch.from_numpy((specs[tr]-mean)/(std+1e-8)).unsqueeze(1).float()
    xte = torch.from_numpy((specs[te]-mean)/(std+1e-8)).unsqueeze(1).float()
    ytr = torch.from_numpy(y[tr]).long(); yte = torch.from_numpy(y[te]).long()
    engine.set_seed(seed)
    m = models.SpecCNN(n_classes=n_classes)
    crit = torch.nn.CrossEntropyLoss(); opt = torch.optim.Adam(m.parameters(), lr=config.LR)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(xte).argmax(1).numpy()
    return accuracy_score(yte.numpy(), pred), f1_score(yte.numpy(), pred, average="macro")


def low_data_indices(y, rec, k, seed):
    """每记录取前 k 窗口，随机 vs 记录级划分。"""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    keep = []
    for r in np.unique(rec):
        ri = idx[rec == r][:k]
        keep.extend(ri.tolist())
    keep = np.array(keep)
    yk = y[keep]; reck = rec[keep]
    # random split
    perm = rng.permutation(len(keep))
    ncut = int(len(keep)*0.7)
    tr_r = keep[perm[:ncut]]; te_r = keep[perm[ncut:]]
    # recording split
    recs = np.unique(reck); rng.shuffle(recs)
    ncut_r = int(len(recs)*0.7)
    tr_recs = set(recs[:ncut_r].tolist()); te_recs = set(recs[ncut_r:].tolist())
    tr_s = keep[np.isin(reck, list(tr_recs))]
    te_s = keep[np.isin(reck, list(te_recs))]
    return (tr_r, te_r), (tr_s, te_s)


def cross_load_indices(y, rec, load, seed):
    idx = np.arange(len(y))
    tr_mask = np.isin(load, [0, 1, 2]); te_mask = load == 3
    # 随机版：忽略 load，直接在全体上随机 70/30（会把 load3 的相邻窗口混入 train）
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y)); ncut = int(len(y)*0.7)
    tr_r = perm[:ncut]; te_r = perm[ncut:]
    # 跨载荷版：train loads 0-2, test load 3
    tr_s = idx[tr_mask]; te_s = idx[te_mask]
    return (tr_r, te_r), (tr_s, te_s)


def main():
    specs, _ = datautil.get_spectrograms()
    data = datautil.load_windows()
    y = data["y"]; rec = data["recording_id"]; load = data["load"]
    seed = 0

    print("=== D. low-data (k=12 windows/recording) ===")
    (tr_r, te_r), (tr_s, te_s) = low_data_indices(y, rec, 12, seed)
    print(f"  sizes: R train/test={len(tr_r)}/{len(te_r)}  S train/test={len(tr_s)}/{len(te_s)}")
    a, f = train_eval(specs, y, tr_r, te_r, seed, config.N_CLASSES); print(f"  R(random): acc={a:.4f} f1={f:.4f}")
    a, f = train_eval(specs, y, tr_s, te_s, seed, config.N_CLASSES); print(f"  S(record): acc={a:.4f} f1={f:.4f}")

    print("=== E. cross-load (test=load3) ===")
    (tr_r, te_r), (tr_s, te_s) = cross_load_indices(y, rec, load, seed)
    a, f = train_eval(specs, y, tr_r, te_r, seed, config.N_CLASSES); print(f"  R(random): acc={a:.4f} f1={f:.4f}")
    a, f = train_eval(specs, y, tr_s, te_s, seed, config.N_CLASSES); print(f"  S(x-load): acc={a:.4f} f1={f:.4f}")


if __name__ == "__main__":
    main()
