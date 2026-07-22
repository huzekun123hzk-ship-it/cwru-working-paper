#!/usr/bin/env python3
"""Grad-CAM 物理一致性分析（改自 experiments/ch53）。

对 SpecCNN 的最后一层卷积做 Grad-CAM，得到频谱图上的显著性热力图。
再把热力图沿频率轴（行）聚合，计算落在理论故障频带内的显著性占比。

关键指标（本研究非精度维度）：
  physical_consistency = 故障频带内显著性 / 全部显著性
负对照：把故障频带沿频率轴平移后重算占比，用于判定"频带内高占比"是否
真实源于模型关注故障频率，而非频带定义本身的偏置。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import config
import fault_freq


class GradCAM1D:
    """针对 1D-CNN（EnvCNN）last_conv_layer 的 Grad-CAM。

    输入 x: (1,1,L)，返回 (L,) 归一化显著性向量，直接对应包络谱频率轴。
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, x, class_idx=None):
        acts, grads = {}, {}
        layer = self.model.last_conv_layer
        h1 = layer.register_forward_hook(
            lambda m, i, o: acts.__setitem__("a", o.detach()))
        h2 = layer.register_full_backward_hook(
            lambda m, gi, go: grads.__setitem__("g", go[0].detach()))
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        w = grads["g"].mean(dim=2, keepdim=True)
        cam = F.relu((w * acts["a"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2], mode="linear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        h1.remove(); h2.remove()
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx


def band_fraction_1d(freqs, saliency, bands):
    """1D 显著性向量落在频带内的占比。"""
    total = saliency.sum()
    if total <= 1e-12:
        return 0.0
    s = 0.0
    for (lo, hi, _name) in bands:
        s += saliency[(freqs >= lo) & (freqs <= hi)].sum()
    return float(s / total)


def physical_consistency_1d(model, x, rpm, freqs, class_idx=None):
    """包络谱 Grad-CAM 物理一致性：返回 (真实频带占比, 负对照占比, 预测)。"""
    engine_cam = GradCAM1D(model)
    cam, pred = engine_cam(x, class_idx=class_idx)
    true_bands = fault_freq.fault_bands(rpm)
    neg_bands = fault_freq.shifted_bands(rpm)
    return (band_fraction_1d(freqs, cam, true_bands),
            band_fraction_1d(freqs, cam, neg_bands),
            pred)


class GradCAM:
    """针对 SpecCNN.last_conv_layer 的 2D Grad-CAM（保留，用于对照实验）。"""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        layer = model.last_conv_layer
        layer.register_forward_hook(self._fwd_hook)
        layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """x: (1,1,H,W) tensor。返回 (H,W) 归一化热力图 [0,1]。"""
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        # 通道权重 = 梯度全局平均
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx


def freq_row_saliency(cam, freqs):
    """把热力图沿时间轴（列）求和 → 每个频率行的显著性向量。

    freqs 为对应每行的频率(Hz)。返回 (freqs, saliency_per_row)。
    注意：频谱图 resize 后行 0 对应最低频，与 features.py 的 freqs_resized 一致。
    """
    row_sal = cam.sum(axis=1)  # (H,)
    return freqs, row_sal


def band_fraction(freqs, row_sal, bands):
    """计算落在给定频带列表内的显著性占比。"""
    total = row_sal.sum()
    if total <= 1e-12:
        return 0.0
    in_band = 0.0
    for (lo, hi, _name) in bands:
        mask = (freqs >= lo) & (freqs <= hi)
        in_band += row_sal[mask].sum()
    return float(in_band / total)


def physical_consistency(model, x, rpm, freqs, class_idx=None):
    """返回 (真实频带占比, 平移负对照占比, 预测类别)。"""
    cam_engine = GradCAM(model)
    cam, pred = cam_engine(x, class_idx=class_idx)
    fr, row_sal = freq_row_saliency(cam, freqs)
    true_bands = fault_freq.fault_bands(rpm)
    neg_bands = fault_freq.shifted_bands(rpm)
    frac_true = band_fraction(fr, row_sal, true_bands)
    frac_neg = band_fraction(fr, row_sal, neg_bands)
    return frac_true, frac_neg, pred
