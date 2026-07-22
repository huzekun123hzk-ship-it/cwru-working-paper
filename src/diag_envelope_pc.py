#!/usr/bin/env python3
"""决定性验证：包络谱 1D-CNN 上，Grad-CAM 是否落在故障频带？
R vs S 的物理一致性(PC)与负对照(PC_neg)对比。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import config
import datautil
import envelope
import models
import engine
import fault_freq


def get_env_cache():
    import os
    p = os.path.join(config.DATA_MANIFEST_DIR, "env_cache.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["specs"], d["freqs"]
    data = datautil.load_windows()
    specs, freqs = envelope.batch_envelope(data["X"])
    np.savez_compressed(p, specs=specs, freqs=freqs)
    return specs, freqs


def gradcam_1d(model, x, class_idx):
    acts = {}
    grads = {}
    layer = model.last_conv_layer
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o.detach()))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0].detach()))
    logits = model(x)
    model.zero_grad()
    logits[0, class_idx].backward()
    w = grads["g"].mean(dim=2, keepdim=True)
    cam = F.relu((w * acts["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2], mode="linear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    h1.remove(); h2.remove()
    if cam.max() - cam.min() > 1e-8:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


def band_frac(freqs, sal, bands):
    total = sal.sum()
    if total < 1e-12:
        return 0.0
    s = 0.0
    for lo, hi, _ in bands:
        s += sal[(freqs >= lo) & (freqs <= hi)].sum()
    return float(s / total)


def run(protocol, seed, specs, freqs, epochs=12):
    data = datautil.load_windows()
    y = data["y"]; rpm = data["rpm"]
    sp = datautil.load_splits(protocol)[str(seed)]
    tr, va, te = np.array(sp["train"]), np.array(sp["val"]), np.array(sp["test"])
    mean, std = specs[tr].mean(), specs[tr].std()

    def mk(ii):
        x = (specs[ii] - mean) / (std + 1e-8)
        return torch.from_numpy(x).unsqueeze(1).float(), torch.from_numpy(y[ii]).long()

    xtr, ytr = mk(tr); xte, yte = mk(te)
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

    # 物理一致性（在正确分类样本上）
    pc_true, pc_neg = [], []
    n = min(300, len(te))
    for i in range(n):
        xi = xte[i:i+1]
        yi = int(yte[i])
        cam = gradcam_1d(m, xi, yi)
        r = int(rpm[te[i]])
        tb = fault_freq.fault_bands(r); nb = fault_freq.shifted_bands(r)
        pc_true.append(band_frac(freqs, cam, tb))
        pc_neg.append(band_frac(freqs, cam, nb))
    return acc, f1, float(np.mean(pc_true)), float(np.mean(pc_neg))


def main():
    specs, freqs = get_env_cache()
    print(f"env specs {specs.shape} freq res {freqs[1]-freqs[0]:.2f}Hz")
    # 先看包络谱上故障频带能量占比（不训练，纯物理）
    data = datautil.load_windows()
    y = data["y"]; rpm = data["rpm"]
    for c in range(config.N_CLASSES):
        idx = np.where(y == c)[0][:200]
        fr = []
        for i in idx:
            r = int(rpm[i]); tb = fault_freq.fault_bands(r)
            fr.append(band_frac(freqs, specs[i], tb))
        print(f"  class {config.IDX_TO_CLASS[c]}: fault-band energy frac = {np.mean(fr):.3f}")

    for proto in ("R", "S"):
        acc, f1, pct, pcn = run(proto, 0, specs, freqs)
        print(f"[{proto} seed0] acc={acc:.4f} f1={f1:.4f} PC_true={pct:.3f} PC_neg={pcn:.3f}")


if __name__ == "__main__":
    main()
