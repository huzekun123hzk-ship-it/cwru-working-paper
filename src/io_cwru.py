#!/usr/bin/env python3
"""CWRU 真实 .mat 数据加载与数据清单构建。

关键设计（对齐 manual §5–6 的泄漏控制）：
  - 每个 .mat 文件 = 一个 recording_id（切窗后同一文件的窗口不可跨 split）。
  - 只取 12k 驱动端(DE) 通道，采样率统一为 12000 Hz。
  - 窗口保留其来源 recording_id、标签、载荷、转速、故障尺寸。
"""

from __future__ import annotations

import json
import os

import numpy as np
import scipy.io as sio

import config


def load_file_metadata():
    """读取 cv2 已审计好的 161 文件元数据列表。"""
    with open(config.FILE_METADATA_JSON, encoding="utf-8") as f:
        return json.load(f)


def _find_channel_key(mat_dict, channel="DE"):
    """在 .mat 键中找到形如 X<id>_DE_time 的通道键。"""
    suffix = f"_{channel}_time"
    for k in mat_dict.keys():
        if k.startswith("__"):
            continue
        if k.endswith(suffix):
            return k
    return None


def load_de_signal(mat_path, channel=None):
    """加载单个 .mat 的指定通道一维信号（默认 config.CHANNEL）。

    找不到通道返回 None；文件损坏/无法读取时也返回 None 并打印告警，
    以便预处理跳过坏文件而不中断整批（manual §16 要求如实报告）。
    """
    channel = channel or config.CHANNEL
    try:
        d = sio.loadmat(mat_path)
    except (OSError, ValueError, NotImplementedError) as e:
        print(f"[warn] unreadable mat ({type(e).__name__}): {os.path.basename(mat_path)}")
        return None
    key = _find_channel_key(d, channel)
    if key is None:
        return None
    sig = np.asarray(d[key], dtype=np.float64).reshape(-1)
    return sig


def iter_windows(signal, win_len=None, hop=None, max_windows=None):
    """把一维信号切成不重叠/重叠窗口，返回 (n_win, win_len) 数组。"""
    win_len = win_len or config.WINDOW_LEN
    hop = hop or config.WINDOW_HOP
    max_windows = max_windows or config.MAX_WINDOWS_PER_FILE
    n = len(signal)
    if n < win_len:
        return np.empty((0, win_len), dtype=np.float64)
    starts = list(range(0, n - win_len + 1, hop))
    if len(starts) > max_windows:
        starts = starts[:max_windows]
    out = np.stack([signal[s:s + win_len] for s in starts], axis=0)
    return out


def build_manifest(dataset=config.DEFAULT_DATASET):
    """基于 file_metadata.json 生成某数据集的记录级清单。

    dataset in config.DATASETS（如 "DE"/"FE"）。用该数据集的故障子目录 +
    Normal 基线目录，且只保留采样率匹配的文件。channel 决定后续读哪个通道。

    返回 list[dict]：recording_id / label / label_idx / load_hp / rpm /
    fault_size / sampling_rate / channel / path / rel_path。
    """
    spec = config.DATASETS[dataset]
    channel = spec["channel"]
    fs = spec["sampling_rate"]
    keep_dirs = list(spec["fault_folders"]) + [config.NORMAL_FOLDER]

    meta = load_file_metadata()
    rows = []
    rec_id = 0
    for m in meta:
        label = m.get("fault_type")
        if label not in config.CLASS_TO_IDX:
            continue
        if int(m.get("sampling_rate", 0)) != fs:
            continue
        rel = m["file"]
        # 该数据集只接受其指定子目录（含 Normal 基线）的文件
        if not any(d in rel for d in keep_dirs):
            continue
        abs_path = os.path.join(config.CWRU_ROOT, rel)
        if not os.path.exists(abs_path):
            continue
        rows.append(
            {
                "recording_id": rec_id,
                "label": label,
                "label_idx": config.CLASS_TO_IDX[label],
                "load_hp": int(m.get("load_hp", 0)),
                "rpm": int(m.get("rpm", 0)),
                "fault_size": float(m.get("fault_size", 0.0)),
                "sampling_rate": int(m.get("sampling_rate", fs)),
                "channel": channel,
                "path": abs_path,
                "rel_path": rel,
            }
        )
        rec_id += 1
    return rows
