#!/usr/bin/env python3
"""生成 data_manifest/manifest.csv —— 记录级数据清单（含 recording_id）。

用法：
    python src/make_manifest.py
"""

from __future__ import annotations

import csv
import os

import config
import io_cwru


FIELDS = [
    "recording_id",
    "label",
    "label_idx",
    "load_hp",
    "rpm",
    "fault_size",
    "sampling_rate",
    "channel",
    "rel_path",
    "path",
]


def main(dataset=config.DEFAULT_DATASET):
    os.makedirs(config.DATA_MANIFEST_DIR, exist_ok=True)
    rows = io_cwru.build_manifest(dataset=dataset)
    out = config.ds_path(dataset, "manifest")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in FIELDS})

    by_class = {}
    by_load = {}
    for r in rows:
        by_class[r["label"]] = by_class.get(r["label"], 0) + 1
        by_load[r["load_hp"]] = by_load.get(r["load_hp"], 0) + 1
    print(f"[manifest:{dataset}] wrote {len(rows)} recordings -> {out}")
    print(f"[manifest:{dataset}] by class: {by_class}")
    print(f"[manifest:{dataset}] by load : {by_load}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET,
                    choices=list(config.DATASETS.keys()))
    args = ap.parse_args()
    main(args.dataset)
