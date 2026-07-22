#!/usr/bin/env python3
"""包络谱表示（envelope spectrum）。

流程（对齐 innovation_hints_cn.md 方向二 + manual §9.1）：
  1. 带通滤波到轴承共振带（故障冲击激起的高频共振）
  2. Hilbert 变换取解析信号 → 包络
  3. 包络去直流 → FFT 幅值谱
  4. 截取低频段 [0, FMAX]，此处故障特征频率(BPFO/BPFI/BSF)呈清晰峰值

包络谱为 1D 向量，频率分辨率 = fs / win_len ≈ 5.9 Hz（远优于原始 STFT 的 95 Hz），
使"故障频带内能量/显著性占比"成为有意义的物理一致性指标。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

import config

# 共振带（由 diag_saliency 观察到能量集中在 ~2.5-3.4kHz，取更宽的 2-4kHz）
RESONANCE_BAND = (2000.0, 4000.0)
ENV_FMAX = 500.0   # 包络谱保留的最高频率（覆盖故障频率 1-3 次谐波）


def _bandpass(signal, fs, band, order=4):
    ny = 0.5 * fs
    lo, hi = band[0] / ny, band[1] / ny
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, signal)


def envelope_spectrum(signal, fs=None, band=RESONANCE_BAND, fmax=ENV_FMAX,
                      apply_bandpass=True):
    """波形窗口 → 包络谱 1D 向量与其频率轴。

    参数 apply_bandpass=False 时跳过带通（供消融：验证增益是否来自带通）。
    返回 (spec (L,), freqs (L,))。
    """
    fs = fs or config.TARGET_FS
    x = np.asarray(signal, dtype=np.float64)
    if apply_bandpass:
        x = _bandpass(x, fs, band)
    env = np.abs(hilbert(x))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fs)
    keep = freqs <= fmax
    return spec[keep].astype(np.float32), freqs[keep]


def batch_envelope(X, fs=None, apply_bandpass=True):
    """(N, win_len) 波形 → (N, L) 包络谱，附频率轴。"""
    specs = []
    freqs = None
    for i in range(X.shape[0]):
        s, freqs = envelope_spectrum(X[i], fs=fs, apply_bandpass=apply_bandpass)
        specs.append(s)
    return np.stack(specs, axis=0), freqs


if __name__ == "__main__":
    import datautil
    data = datautil.load_windows()
    X = data["X"]; y = data["y"]
    s, freqs = envelope_spectrum(X[0])
    print(f"env spectrum len={len(s)}  freq range {freqs.min():.1f}-{freqs.max():.1f}Hz "
          f"res={freqs[1]-freqs[0]:.2f}Hz")
