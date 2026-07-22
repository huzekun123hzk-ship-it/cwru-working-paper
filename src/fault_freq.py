#!/usr/bin/env python3
"""轴承故障特征频率计算（SKF6205 驱动端）。

转频 fr = rpm / 60；再乘以 bearing_params.json 的倍数得到 BPFO/BPFI/BSF。
供 Grad-CAM 物理一致性分析定义参考频带。
"""

from __future__ import annotations

import json

import config


def load_bearing_params():
    with open(config.BEARING_PARAMS_JSON, encoding="utf-8") as f:
        return json.load(f)["SKF6205"]


def fault_frequencies(rpm):
    """给定转速(rpm)，返回 dict{BPFO,BPFI,BSF,FTF,fr}（单位 Hz）。"""
    fr = rpm / 60.0
    return {
        "fr": fr,
        "BPFO": fr * config.BPFO_RATIO,
        "BPFI": fr * config.BPFI_RATIO,
        "BSF": fr * config.BSF_RATIO,
        "FTF": fr * config.FTF_RATIO,
    }


def fault_bands(rpm, n_harmonics=3, halfwidth=None):
    """返回故障特征频率及其谐波的 (low, high) 频带列表。

    包含 BPFO/BPFI/BSF 的 1..n 次谐波，各自 ± halfwidth。
    """
    halfwidth = halfwidth if halfwidth is not None else config.BAND_HALFWIDTH
    ff = fault_frequencies(rpm)
    bands = []
    for key in ("BPFO", "BPFI", "BSF"):
        base = ff[key]
        for h in range(1, n_harmonics + 1):
            center = base * h
            bands.append((center - halfwidth, center + halfwidth, f"{key}x{h}"))
    return bands


def shifted_bands(rpm, shift=None, n_harmonics=3, halfwidth=None):
    """负对照：把每个故障频带沿频率轴平移 shift Hz。"""
    shift = shift if shift is not None else config.NEG_CONTROL_SHIFT
    bands = fault_bands(rpm, n_harmonics, halfwidth)
    return [(lo + shift, hi + shift, name + "_shift") for (lo, hi, name) in bands]


if __name__ == "__main__":
    for rpm in (1797, 1772, 1750, 1730):
        ff = fault_frequencies(rpm)
        print(f"rpm={rpm}: fr={ff['fr']:.2f} BPFO={ff['BPFO']:.1f} "
              f"BPFI={ff['BPFI']:.1f} BSF={ff['BSF']:.1f}")
