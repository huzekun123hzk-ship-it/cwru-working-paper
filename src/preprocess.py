#!/usr/bin/env python3
"""切窗预处理：读取 manifest，把每条记录的 DE 信号切成窗口，
保存为 windows.npz。每个窗口保留其 recording_id / label / load / rpm / fault_size。

窗口本身不做归一化——归一化统计量必须在划分后、仅用训练集计算（manual §6.3）。

用法：
    python src/preprocess.py
"""

from __future__ import annotations

import csv
import os

import numpy as np

import config
import io_cwru


def read_manifest(dataset=config.DEFAULT_DATASET):
    with open(config.ds_path(dataset, "manifest"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(dataset=config.DEFAULT_DATASET):
    rows = read_manifest(dataset)
    all_win = []
    win_rec = []      # recording_id
    win_label = []    # label_idx
    win_load = []     # load_hp
    win_rpm = []      # rpm
    win_size = []     # fault_size

    for r in rows:
        sig = io_cwru.load_de_signal(r["path"], channel=r.get("channel"))
        if sig is None:
            print(f"[warn] no DE channel: {r['rel_path']}")
            continue
        wins = io_cwru.iter_windows(sig)
        if wins.shape[0] == 0:
            print(f"[warn] too short: {r['rel_path']}")
            continue
        n = wins.shape[0]
        all_win.append(wins.astype(np.float32))
        win_rec.extend([int(r["recording_id"])] * n)
        win_label.extend([int(r["label_idx"])] * n)
        win_load.extend([int(r["load_hp"])] * n)
        win_rpm.extend([int(r["rpm"])] * n)
        win_size.extend([float(r["fault_size"])] * n)

    X = np.concatenate(all_win, axis=0)
    rec = np.asarray(win_rec, dtype=np.int64)
    y = np.asarray(win_label, dtype=np.int64)
    load = np.asarray(win_load, dtype=np.int64)
    rpm = np.asarray(win_rpm, dtype=np.int64)
    size = np.asarray(win_size, dtype=np.float32)

    os.makedirs(config.DATA_MANIFEST_DIR, exist_ok=True)
    out = config.ds_path(dataset, "windows")
    np.savez_compressed(
        out,
        X=X, recording_id=rec, y=y, load=load, rpm=rpm, fault_size=size,
    )
    print(f"[preprocess:{dataset}] windows: {X.shape} -> {out}")
    for idx, name in config.IDX_TO_CLASS.items():
        print(f"  {name:6s}: {(y == idx).sum():5d} windows "
              f"from {len(np.unique(rec[y == idx]))} recordings")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET,
                    choices=list(config.DATASETS.keys()))
    args = ap.parse_args()
    main(args.dataset)
