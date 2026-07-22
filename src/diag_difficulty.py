#!/usr/bin/env python3
"""诊断：寻找能让"随机划分 vs 记录划分"泄漏差距显现的任务设置。

对比三种设置下 协议R(随机窗口) vs 协议S(按记录) 的 test 准确率：
  A. 4 类粗分类（当前）
  B. 细粒度标签（location × fault_size，~12 类）
  C. 细粒度 + 加性噪声（SNR 较低）

只训练少 epoch、单种子，用于快速定位，不写入正式结果。
"""

from __future__ import annotations

import numpy as np
import torch

import config
import datautil
import models
import engine


def fine_labels(y, size):
    """构造 location×size 细粒度标签。"""
    keys = []
    for yi, si in zip(y, size):
        name = config.IDX_TO_CLASS[int(yi)]
        keys.append(f"{name}_{round(float(si) * 1000):03d}")
    uniq = sorted(set(keys))
    k2i = {k: i for i, k in enumerate(uniq)}
    return np.array([k2i[k] for k in keys], dtype=np.int64), uniq


def split_random(y, seed, vr, te):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y)); tr_, va_, te_ = [], [], []
    for c in np.unique(y):
        ci = idx[y == c]; rng.shuffle(ci); n = len(ci)
        nt = max(1, int(round(n * te))); nv = max(1, int(round(n * vr)))
        te_ += ci[:nt].tolist(); va_ += ci[nt:nt+nv].tolist(); tr_ += ci[nt+nv:].tolist()
    return tr_, va_, te_


def split_rec(y, rec, seed, vr, te):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y)); tr_, va_, te_ = [], [], []
    for c in np.unique(y):
        cm = y == c; recs = np.unique(rec[cm]); rng.shuffle(recs); nr = len(recs)
        if nr >= 3:
            nt = max(1, int(round(nr*te))); nv = max(1, min(int(round(nr*vr)), nr-nt-1))
            ts, vs = set(recs[:nt]), set(recs[nt:nt+nv]); trs = set(recs[nt+nv:])
        elif nr == 2:
            ts, trs, vs = {recs[0]}, {recs[1]}, {recs[1]}
        else:
            ts, trs, vs = {recs[0]}, {recs[0]}, {recs[0]}
        for r in recs:
            ri = idx[cm & (rec == r)].tolist()
            if r in ts: te_ += ri
            if r in vs: va_ += ri
            if r in trs: tr_ += ri
    tr_ = list(set(tr_) - set(te_)); va_ = list(set(va_) - set(te_))
    return tr_, va_, te_


def run(specs, y, indices_fn, seed, n_classes, noise=0.0):
    tr, va, te = indices_fn
    mean, std = specs[tr].mean(), specs[tr].std()

    def mk(ii):
        ii = np.asarray(ii)
        x = (specs[ii] - mean) / (std + 1e-8)
        if noise > 0:
            rng = np.random.RandomState(seed + 7)
            x = x + rng.randn(*x.shape).astype(np.float32) * noise
        return torch.from_numpy(x).unsqueeze(1).float(), torch.from_numpy(y[ii]).long()

    xtr, ytr = mk(tr); xte, yte = mk(te)
    from torch.utils.data import TensorDataset, DataLoader
    engine.set_seed(seed)
    m = models.SpecCNN(n_classes=n_classes)
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=config.LR)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    for _ in range(6):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(m(xb), yb); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(xte).argmax(1).numpy()
    from sklearn.metrics import accuracy_score, f1_score
    return accuracy_score(yte.numpy(), pred), f1_score(yte.numpy(), pred, average="macro")


def main():
    specs, _ = datautil.get_spectrograms()
    data = datautil.load_windows()
    y4 = data["y"]; rec = data["recording_id"]; size = data["fault_size"]
    yf, uniq = fine_labels(y4, size)
    print(f"fine classes ({len(uniq)}): {uniq}")
    seed = 0

    print("\n=== A. 4-class ===")
    for name, fn in [("R", split_random(y4, seed, 0.2, 0.2)),
                     ("S", split_rec(y4, rec, seed, 0.2, 0.2))]:
        acc, f1 = run(specs, y4, fn, seed, config.N_CLASSES)
        print(f"  {name}: acc={acc:.4f} macroF1={f1:.4f}")

    print("\n=== B. fine-grained (location x size) ===")
    for name, fn in [("R", split_random(yf, seed, 0.2, 0.2)),
                     ("S", split_rec(yf, rec, seed, 0.2, 0.2))]:
        acc, f1 = run(specs, yf, fn, seed, len(uniq))
        print(f"  {name}: acc={acc:.4f} macroF1={f1:.4f}")

    print("\n=== C. fine-grained + noise(0.8) ===")
    for name, fn in [("R", split_random(yf, seed, 0.2, 0.2)),
                     ("S", split_rec(yf, rec, seed, 0.2, 0.2))]:
        acc, f1 = run(specs, yf, fn, seed, len(uniq), noise=0.8)
        print(f"  {name}: acc={acc:.4f} macroF1={f1:.4f}")


if __name__ == "__main__":
    main()
