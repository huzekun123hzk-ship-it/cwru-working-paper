#!/usr/bin/env python3
"""诊断：频谱图频率分辨率、能量分布、Grad-CAM 显著性分布。

定位为何故障频带内显著性占比极低。
"""

from __future__ import annotations

import numpy as np
import torch

import config
import datautil
import models
import engine
import gradcam
import fault_freq


def main():
    specs, freqs = datautil.get_spectrograms()
    print(f"spec size {specs.shape[1:]}  freq rows={len(freqs)}")
    print(f"freq range: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    print(f"freq resolution per row: {freqs[1]-freqs[0]:.1f} Hz")
    print(f"rows below 500Hz: {(freqs<500).sum()}  below 1000Hz: {(freqs<1000).sum()}")

    ff = fault_freq.fault_frequencies(1797)
    print(f"fault freqs: BPFO={ff['BPFO']:.0f} BPFI={ff['BPFI']:.0f} BSF={ff['BSF']:.0f}")
    bands = fault_freq.fault_bands(1797)
    covered = set()
    for lo, hi, name in bands:
        rows = np.where((freqs >= lo) & (freqs <= hi))[0]
        covered.update(rows.tolist())
    print(f"rows covered by fault bands (halfwidth={config.BAND_HALFWIDTH}): {sorted(covered)}")

    # 能量分布：每行平均能量占比
    mean_energy = specs.mean(axis=(0, 2))  # (H,)
    row_frac = mean_energy / mean_energy.sum()
    top_rows = np.argsort(row_frac)[::-1][:8]
    print("\ntop energy rows (row: freqHz: frac):")
    for r in top_rows:
        print(f"  row{r}: {freqs[r]:.0f}Hz: {row_frac[r]:.3f}")

    # 训练一个 S 模型看 Grad-CAM 显著性行分布
    tr, va, te, cw = datautil.make_spec_datasets("S", 0)
    m = models.build_model("spec_cnn")
    m, _ = engine.train_model(m, tr, va, cw, seed=0, epochs=8)
    cam_engine = gradcam.GradCAM(m)
    sal_rows = np.zeros(len(freqs))
    n = min(200, len(te))
    for i in range(n):
        x, y = te[i]
        cam, _ = cam_engine(x.unsqueeze(0), class_idx=int(y))
        sal_rows += cam.sum(axis=1)
    sal_frac = sal_rows / sal_rows.sum()
    top_sal = np.argsort(sal_frac)[::-1][:8]
    print("\ntop saliency rows (row: freqHz: frac):")
    for r in top_sal:
        print(f"  row{r}: {freqs[r]:.0f}Hz: {sal_frac[r]:.3f}")


if __name__ == "__main__":
    main()
