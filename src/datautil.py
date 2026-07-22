#!/usr/bin/env python3
"""数据装配：加载窗口、按 split 取子集、训练集统计量归一化、缓存频谱图。

关键纪律（manual §6.3）：归一化统计量（mean/std）只在训练集上计算，
再应用到 val/test，杜绝测试信息回流。
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import config
import features


_SPEC_CACHE = None
_FREQS = None
_ENV_CACHE = None
_ENV_FREQS = None


def load_windows(dataset=config.DEFAULT_DATASET):
    data = np.load(config.ds_path(dataset, "windows"))
    return data


_ENV_CACHE_BY_DS = {}


def get_envelope(apply_bandpass=True, dataset=config.DEFAULT_DATASET):
    """全量包络谱缓存（与划分无关）。返回 (specs (N,L), freqs (L,))。"""
    import envelope
    if apply_bandpass and dataset in _ENV_CACHE_BY_DS:
        return _ENV_CACHE_BY_DS[dataset]
    cache_path = config.ds_path(dataset, "env_cache")
    if apply_bandpass and os.path.exists(cache_path):
        d = np.load(cache_path)
        _ENV_CACHE_BY_DS[dataset] = (d["specs"], d["freqs"])
        return _ENV_CACHE_BY_DS[dataset]
    data = load_windows(dataset)
    specs, freqs = envelope.batch_envelope(data["X"], apply_bandpass=apply_bandpass)
    if apply_bandpass:
        np.savez_compressed(cache_path, specs=specs, freqs=freqs)
        _ENV_CACHE_BY_DS[dataset] = (specs, freqs)
    return specs, freqs


def get_spectrograms():
    """全量频谱图缓存（与划分无关，可跨协议复用）。返回 (specs, freqs)。"""
    global _SPEC_CACHE, _FREQS
    if _SPEC_CACHE is not None:
        return _SPEC_CACHE, _FREQS
    cache_path = os.path.join(config.DATA_MANIFEST_DIR, "spec_cache.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        _SPEC_CACHE, _FREQS = d["specs"], d["freqs"]
        return _SPEC_CACHE, _FREQS
    data = load_windows()
    specs, freqs = features.batch_spectrograms(data["X"])
    np.savez_compressed(cache_path, specs=specs, freqs=freqs)
    _SPEC_CACHE, _FREQS = specs, freqs
    return specs, freqs


def load_splits(protocol, dataset=config.DEFAULT_DATASET):
    kind = "splits_R" if protocol == "R" else "splits_S"
    with open(config.ds_path(dataset, kind)) as f:
        return json.load(f)


class SpecDataset(Dataset):
    """频谱图数据集，应用训练集统计量归一化。"""

    def __init__(self, specs, y, indices, mean, std):
        self.x = specs[indices]
        self.y = y[indices].astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = (self.x[i] - self.mean) / (self.std + 1e-8)
        t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
        return t, int(self.y[i])


class WaveDataset(Dataset):
    """波形数据集，训练集统计量归一化。"""

    def __init__(self, X, y, indices, mean, std):
        self.x = X[indices]
        self.y = y[indices].astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        w = (self.x[i] - self.mean) / (self.std + 1e-8)
        t = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0).float()
        return t, int(self.y[i])


class Env1DDataset(Dataset):
    """包络谱 1D 数据集，训练集统计量归一化。"""

    def __init__(self, specs, y, indices, mean, std):
        self.x = specs[indices]
        self.y = y[indices].astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        v = (self.x[i] - self.mean) / (self.std + 1e-8)
        t = torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).float()
        return t, int(self.y[i])


def make_env_datasets(protocol, seed, apply_bandpass=True, dataset=config.DEFAULT_DATASET):
    """返回 (train_ds, val_ds, test_ds, class_weights, freqs)。"""
    specs, freqs = get_envelope(apply_bandpass=apply_bandpass, dataset=dataset)
    data = load_windows(dataset)
    y = data["y"]
    sp = load_splits(protocol, dataset=dataset)[str(seed)]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    mean = specs[tr].mean()
    std = specs[tr].std()
    train_ds = Env1DDataset(specs, y, tr, mean, std)
    val_ds = Env1DDataset(specs, y, va, mean, std)
    test_ds = Env1DDataset(specs, y, te, mean, std)
    cw = _class_weights(y[tr])
    return train_ds, val_ds, test_ds, cw, freqs


def make_spec_datasets(protocol, seed):
    """返回 (train_ds, val_ds, test_ds, class_weights)。"""
    specs, _ = get_spectrograms()
    data = load_windows()
    y = data["y"]
    sp = load_splits(protocol)[str(seed)]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    mean = specs[tr].mean()
    std = specs[tr].std()
    train_ds = SpecDataset(specs, y, tr, mean, std)
    val_ds = SpecDataset(specs, y, va, mean, std)
    test_ds = SpecDataset(specs, y, te, mean, std)
    cw = _class_weights(y[tr])
    return train_ds, val_ds, test_ds, cw


def make_wave_datasets(protocol, seed):
    data = load_windows()
    X = data["X"]
    y = data["y"]
    sp = load_splits(protocol)[str(seed)]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    mean = X[tr].mean()
    std = X[tr].std()
    train_ds = WaveDataset(X, y, tr, mean, std)
    val_ds = WaveDataset(X, y, va, mean, std)
    test_ds = WaveDataset(X, y, te, mean, std)
    cw = _class_weights(y[tr])
    return train_ds, val_ds, test_ds, cw


def _class_weights(y_train):
    """逆频率类别权重，缓解 Normal 样本过少。"""
    counts = np.bincount(y_train, minlength=config.N_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (config.N_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32)
