#!/usr/bin/env python3
"""生成两种划分协议的 train/val/test 索引（对齐 manual §6）。

协议 R（Random window，风险参照）：
    把所有窗口打乱后按比例随机划分。相邻窗口可跨 split → 泄漏。

协议 S（recording-level split，安全评估）：
    按 recording_id 分组划分，同一记录的所有窗口只进一个 split。
    分层保证每个类别在 train/val/test 都有记录（Normal 仅 3 条记录，
    强制至少各 1 条进 train/test）。

对每个 seed 各生成一套划分，写入 splits_R.json / splits_S.json。
窗口索引指向 windows.npz 的行号。

用法：
    python src/make_splits.py
"""

from __future__ import annotations

import json

import numpy as np

import config


def load_windows_meta(dataset=config.DEFAULT_DATASET):
    data = np.load(config.ds_path(dataset, "windows"))
    return data["y"], data["recording_id"], data["load"], data["fault_size"]


def split_random(y, seed, val_ratio, test_ratio):
    """协议 R：窗口级随机划分（分层按类别，保证类别比例）。"""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = np.arange(n)
    train, val, test = [], [], []
    for c in np.unique(y):
        c_idx = idx[y == c]
        rng.shuffle(c_idx)
        n_c = len(c_idx)
        n_test = max(1, int(round(n_c * test_ratio)))
        n_val = max(1, int(round(n_c * val_ratio)))
        test.extend(c_idx[:n_test].tolist())
        val.extend(c_idx[n_test:n_test + n_val].tolist())
        train.extend(c_idx[n_test + n_val:].tolist())
    return sorted(train), sorted(val), sorted(test)


def split_by_recording(y, rec, seed, val_ratio, test_ratio):
    """协议 S：按 recording_id 分组划分，分层保证每类每 split 有记录。"""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    train, val, test = [], [], []

    # 每个类别下的 recording 列表
    for c in np.unique(y):
        c_mask = y == c
        recs = np.unique(rec[c_mask])
        rng.shuffle(recs)
        n_r = len(recs)
        if n_r >= 3:
            n_test = max(1, int(round(n_r * test_ratio)))
            n_val = max(1, int(round(n_r * val_ratio)))
            # 保证 train 至少 1 条
            n_val = min(n_val, n_r - n_test - 1)
            n_val = max(n_val, 0)
            test_recs = set(recs[:n_test].tolist())
            val_recs = set(recs[n_test:n_test + n_val].tolist())
            train_recs = set(recs[n_test + n_val:].tolist())
        elif n_r == 2:
            # 只能 train + test，各 1 条；val 借用 train（记录级仍不泄漏 test）
            test_recs = {recs[0]}
            train_recs = {recs[1]}
            val_recs = {recs[1]}
        else:
            # 仅 1 条记录：train=test=val 同一条（无法避免，记为局限）
            test_recs = {recs[0]}
            train_recs = {recs[0]}
            val_recs = {recs[0]}

        for r in recs:
            r_idx = idx[c_mask & (rec == r)].tolist()
            if r in test_recs:
                test.extend(r_idx)
            if r in val_recs:
                val.extend(r_idx)
            if r in train_recs:
                train.extend(r_idx)

    # 去重（val 可能与 train 借用重叠时，保证 test 独立即可）
    train = sorted(set(train) - set(test))
    val = sorted(set(val) - set(test))
    test = sorted(set(test))
    return train, val, test


def summarize(name, y, rec, train, val, test):
    def dist(ii):
        vals, cnts = np.unique(y[ii], return_counts=True)
        return {config.IDX_TO_CLASS[int(v)]: int(c) for v, c in zip(vals, cnts)}

    # 检查 test 与 train 是否有 recording 重叠（协议 S 应为 0）
    rec_train = set(rec[train].tolist())
    rec_test = set(rec[test].tolist())
    overlap = rec_train & rec_test
    print(f"[{name}] train/val/test = {len(train)}/{len(val)}/{len(test)}")
    print(f"       test dist: {dist(test)}")
    print(f"       recording overlap train∩test = {len(overlap)}")


def main(dataset=config.DEFAULT_DATASET):
    y, rec, load, size = load_windows_meta(dataset)
    splits_R = {}
    splits_S = {}
    for seed in config.SEEDS:
        tr, va, te = split_random(y, seed, config.VAL_RATIO, config.TEST_RATIO)
        splits_R[str(seed)] = {"train": tr, "val": va, "test": te}
        summarize(f"{dataset} R seed{seed}", y, rec, tr, va, te)

        tr, va, te = split_by_recording(y, rec, seed, config.VAL_RATIO, config.TEST_RATIO)
        splits_S[str(seed)] = {"train": tr, "val": va, "test": te}
        summarize(f"{dataset} S seed{seed}", y, rec, tr, va, te)

    rpath = config.ds_path(dataset, "splits_R")
    spath = config.ds_path(dataset, "splits_S")
    with open(rpath, "w") as f:
        json.dump(splits_R, f)
    with open(spath, "w") as f:
        json.dump(splits_S, f)
    print(f"[splits:{dataset}] wrote {rpath}")
    print(f"[splits:{dataset}] wrote {spath}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET,
                    choices=list(config.DATASETS.keys()))
    args = ap.parse_args()
    main(args.dataset)
