#!/usr/bin/env python3
"""全局配置：路径、常量、类别定义、实验超参数。

所有脚本从这里读取配置，保证协议 R 与协议 S 共享完全相同的
架构 / 优化器 / epoch / 窗口参数，仅划分协议不同（manual §10.1）。
"""

from __future__ import annotations

import os

# ============================================================
# 路径（相对本文件定位，避免依赖工作目录）
# ============================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
WORKSPACE_DIR = os.path.dirname(REPO_DIR)

# 原始 CWRU 数据与已有元数据（复用 cv2 下已审计好的资源）
CWRU_ROOT = os.path.join(
    WORKSPACE_DIR, "cv2", "2025E", "cwru_experiment"
)
SOURCE_DATASETS = os.path.join(CWRU_ROOT, "Source_Datasets")
FILE_METADATA_JSON = os.path.join(CWRU_ROOT, "references", "file_metadata.json")
BEARING_PARAMS_JSON = os.path.join(
    CWRU_ROOT, "bearing-vibration-explorer", "references", "bearing_params.json"
)

# 输出目录
DATA_MANIFEST_DIR = os.path.join(REPO_DIR, "data_manifest")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# ============================================================
# 多数据集定义（DE = 主数据集，FE = 第二数据集，不同传感器位置）
# channel = .mat 通道后缀；folders = 该数据集故障文件所在子目录（Normal 通用）
# ============================================================
DATASETS = {
    "DE": {
        "channel": "DE",
        "sampling_rate": 12000,
        "fault_folders": ["02_12k_Drive_End_Fault"],
        "desc": "CWRU 12kHz drive-end",
    },
    "FE": {
        "channel": "FE",
        "sampling_rate": 12000,
        "fault_folders": ["04_12k_Fan_End_Fault"],
        "desc": "CWRU 12kHz fan-end (second dataset, different sensor location)",
    },
}
DEFAULT_DATASET = "DE"
NORMAL_FOLDER = "01_Normal_Baseline"


def ds_path(dataset, kind):
    """返回某数据集的产物路径。DE 用无前缀名（保持向后兼容），其余加前缀。"""
    prefix = "" if dataset == DEFAULT_DATASET else f"{dataset}_"
    names = {
        "manifest": f"{prefix}manifest.csv",
        "windows": f"{prefix}windows.npz",
        "splits_R": f"{prefix}splits_R.json",
        "splits_S": f"{prefix}splits_S.json",
        "env_cache": f"{prefix}env_cache.npz",
    }
    return os.path.join(DATA_MANIFEST_DIR, names[kind])


# 向后兼容的 DE 默认路径别名
MANIFEST_CSV = os.path.join(DATA_MANIFEST_DIR, "manifest.csv")
WINDOWS_NPZ = os.path.join(DATA_MANIFEST_DIR, "windows.npz")
SPLITS_R_JSON = os.path.join(DATA_MANIFEST_DIR, "splits_R.json")
SPLITS_S_JSON = os.path.join(DATA_MANIFEST_DIR, "splits_S.json")
ENV_CACHE_NPZ = os.path.join(DATA_MANIFEST_DIR, "env_cache.npz")
SPEC_CACHE_NPZ = os.path.join(DATA_MANIFEST_DIR, "spec_cache.npz")

# ============================================================
# 类别定义（4 类，与 cv2 审计一致）
# ============================================================
CLASSES = ["Normal", "IR", "OR", "B"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# ============================================================
# 数据 / 窗口 / STFT 参数（全实验固定）
# ============================================================
TARGET_FS = 12000            # 只用 12k 驱动端(DE)，采样率统一
CHANNEL = "DE"               # 驱动端通道
WINDOW_LEN = 2048            # 窗口长度（点）
WINDOW_HOP = 1024            # 步长（50% 重叠）
MAX_WINDOWS_PER_FILE = 200   # 每文件窗口上限（控制不平衡与规模）

N_FFT = 256                  # STFT 窗长（2048 点信号 → 合理时频分辨率）
STFT_HOP = 64                # STFT 跳步
SPEC_SIZE = (64, 64)         # 频谱图 resize 目标（轻量、CPU 友好）

# ============================================================
# 训练超参数（协议 R / S 共享）
# ============================================================
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # 10 个随机种子（支持配对统计与置信区间）
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
VAL_RATIO = 0.2
TEST_RATIO = 0.2
DEVICE = "cpu"

# ============================================================
# 轴承（SKF6205 驱动端）故障频率倍数（相对转频 fr）
# 来源：cv2 bearing_params.json
# ============================================================
BPFO_RATIO = 3.5848
BPFI_RATIO = 5.4152
BSF_RATIO = 2.3581
FTF_RATIO = 0.3983
# 故障频带半宽（Hz）：在特征频率 ± 该值内累计显著性
BAND_HALFWIDTH = 15.0
# 负对照频带平移量（Hz）
NEG_CONTROL_SHIFT = 120.0
