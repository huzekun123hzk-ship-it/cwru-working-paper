#!/usr/bin/env python3
"""模型定义：2D-CNN（频谱图，改自 ch41）与 1D-CNN（波形，改自 ch39）。

两个模型都保留最后一个卷积块的引用，便于 Grad-CAM 挂钩。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SpecCNN(nn.Module):
    """频谱图 2D-CNN。架构对齐 ch41 §41.2，输入 (1, H, W)。

    last_conv 暴露最后一层卷积激活，供 Grad-CAM 使用。
    """

    def __init__(self, n_classes=config.N_CLASSES, spec_size=config.SPEC_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        h = spec_size[0] // 8
        w = spec_size[1] // 8
        self.fc = nn.Linear(64 * h * w, n_classes)
        self._last_activations = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        self._last_activations = x  # 供 Grad-CAM（在 pool 前）
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    @property
    def last_conv_layer(self):
        return self.conv3


class WaveCNN(nn.Module):
    """波形 1D-CNN，输入 (1, win_len)。作为跨载体稳健性验证。"""

    def __init__(self, n_classes=config.N_CLASSES, win_len=config.WINDOW_LEN):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(16, 32, 5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, stride=2, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


class EnvCNN(nn.Module):
    """包络谱 1D-CNN，输入 (1, L)。last_conv 供 1D Grad-CAM。"""

    def __init__(self, n_classes=config.N_CLASSES, in_len=86):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)
        self._last_activations = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        self._last_activations = x
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

    @property
    def last_conv_layer(self):
        return self.conv3


def build_model(kind, **kwargs):
    if kind == "spec_cnn":
        return SpecCNN(**kwargs)
    if kind == "wave_cnn":
        return WaveCNN(**kwargs)
    if kind == "env_cnn":
        return EnvCNN(**kwargs)
    raise ValueError(f"unknown model kind: {kind}")
