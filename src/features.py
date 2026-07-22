#!/usr/bin/env python3
"""特征提取：波形窗口 → STFT 频谱图。

STFT 实现改自 experiments/ch41/utils.py（手写 rfft），保证与课程一致。
频谱图返回时同时给出频率轴，供 Grad-CAM 物理频带定位使用。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

import config


def compute_stft(signal, fs, n_fft, hop_length):
    """手写 STFT，返回 (spec_db, freqs)。spec_db 形状 (n_freqs, n_frames)。"""
    window = np.hanning(n_fft)
    n_points = len(signal)
    n_frames = (n_points - n_fft) // hop_length + 1
    n_freqs = n_fft // 2 + 1
    spec = np.zeros((n_freqs, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))
    spec_db = 20 * np.log10(spec + 1e-10)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return spec_db, freqs


def signal_to_spectrogram(signal, fs=None, n_fft=None, hop_length=None,
                          target_size=None):
    """波形 → 归一化频谱图 [0,1]，附带 resize 后的频率轴。

    返回 (spec_img (H,W), freqs_resized (H,))。
    """
    fs = fs or config.TARGET_FS
    n_fft = n_fft or config.N_FFT
    hop_length = hop_length or config.STFT_HOP
    target_size = target_size or config.SPEC_SIZE

    spec_db, freqs = compute_stft(signal, fs, n_fft, hop_length)
    s_min, s_max = spec_db.min(), spec_db.max()
    if s_max - s_min < 1e-10:
        norm = np.zeros_like(spec_db)
    else:
        norm = (spec_db - s_min) / (s_max - s_min)
    h, w = norm.shape
    spec_img = zoom(norm, (target_size[0] / h, target_size[1] / w), order=1)
    # 频率轴对应 resize：低频在上还是下取决于约定，这里频率随行号递增
    freqs_resized = np.interp(
        np.linspace(0, len(freqs) - 1, target_size[0]),
        np.arange(len(freqs)), freqs,
    )
    return spec_img.astype(np.float32), freqs_resized


def batch_spectrograms(X, fs=None):
    """把 (N, win_len) 波形批量转为 (N, H, W) 频谱图。"""
    specs = np.zeros((X.shape[0],) + config.SPEC_SIZE, dtype=np.float32)
    freqs = None
    for i in range(X.shape[0]):
        img, freqs = signal_to_spectrogram(X[i], fs=fs)
        specs[i] = img
    return specs, freqs
