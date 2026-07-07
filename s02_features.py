# s03_extract_feature_pool.py
# -*- coding: utf-8 -*-

"""
步骤3：Stage2 特征池提取，增强鲁棒预处理版。

输入支持两种形态：
- 3D 预切窗 PPG：直接逐个使用 H5 中已有窗口，不再二次滑窗。
- 连续时序 PPG：通过 Stage1 后降采样到 25Hz，再按 5s/1s 滑窗（可显式切到 3s）。
- grouped-window H5：一个 record 下多个窗口 group，窗口名末尾为 *_w20_1；
  读取时按 w 后数字排序，label 来自最后一段。

功能：
1. 读取 artifacts/splits.json
2. 读取 artifacts/stage1_threshold.json
3. 对通过 Stage1 IR DC/ACDC 阈值的样本提取 5s/25Hz Stage2 特征
4. 复用原始 H5 读取方式
5. 复用原始绿光通道构建方式：
   - mode=1: ch3/ch4/ch5 为三通道绿光
   - mode=2: ch2 作为绿光，退化为 g1=g2=g3
6. 加入鲁棒预处理：
   - 去毛刺
   - 去跳变
   - median filter
   - moving average
   - bandpass
7. 输出特征池 CSV：
   - feature_pool_train.csv
   - feature_pool_valid.csv
   - feature_pool_test.csv
"""

import os
import json
import argparse
import re
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd

from scipy.signal import resample_poly  # only for polyphase downsampling (C has equivalent)

# =========================================================
# 基本配置
# =========================================================

EPS = 1e-12
DEFAULT_FS = 100.0
STAGE1_PRIMITIVE_SEC = 1.0
STAGE1_DECISION_SEC = 3.0
STAGE1_FS = 5
STAGE1_GATE_K = int(round(STAGE1_DECISION_SEC / STAGE1_PRIMITIVE_SEC))
DEFAULT_STAGE1_DC_THRESHOLD = 0.3e6
DEFAULT_STAGE1_AC_DC_THRESHOLD = 1.0
DEFAULT_SKIP_INITIAL_WINDOWS = 3
DEFAULT_USE_STAGE2_IR = False
COMMERCIAL_8_FEATURE_NAMES = [
    "GREEN_CORR",
    "GREEN_AC",
    "AMB_AC",
    "ACC_YSUM",
    "GREEN_DC",
    "AMB_DC",
    "GREEN_XCORR",
    "FFT_PEAK_MEDIAN_RATIO",
]

WINDOW_NAME_RE = re.compile(r"(?:^|_)w(?P<index>\d+)_(?P<label>[01])$")


def apply_stage2_ir_policy(ir, use_stage2_ir=DEFAULT_USE_STAGE2_IR):
    """Return the IR signal used by Stage2 features according to the pipeline switch."""
    ir = np.asarray(ir, dtype=np.float64)
    if use_stage2_ir:
        return ir
    return np.zeros_like(ir, dtype=np.float64)


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_n_workers(n_workers=None, n_items=None, cap=4):
    """Resolve a conservative worker count for server-safe batch runs."""
    if _env_flag("WL_FORCE_SERIAL"):
        return 1
    if n_workers is None:
        n_workers = max(1, min(cap, (os.cpu_count() or cap) // 2))
    try:
        resolved = max(1, int(n_workers))
    except (TypeError, ValueError):
        resolved = 1
    if n_items is not None and int(n_items) <= 2:
        return 1
    return resolved


def multiprocessing_context_from_env():
    """Return an mp context when WL_MP_START_METHOD is set, otherwise default."""
    method = os.environ.get("WL_MP_START_METHOD", "").strip()
    if not method:
        return None
    import multiprocessing as mp
    return mp.get_context(method)

# 已知高冗余特征（节省计算，尤其 Entropy/Derivative 类 O(N²) 操作）
# 这些特征在 s04 清洗阶段也会被 VIF/高相关移除，提前在 s03 跳过以加速提取
_REDUNDANT_FEATURES = {
    # -- ApEn (skip_apen=True, 不再计算) --
    # -- 二阶导数 (compute_d2=False, 不再计算) --
    # -- valley_ratio ≈ 1 - peak_ratio --
    "GREEN_Temporal_valley_ratio", "IRX_Temporal_valley_ratio", "AMBX_Temporal_valley_ratio",
    # -- AC_RMS ≈ 1.48 × AC_MAD (MAD 更鲁棒，保留 MAD) --
    "IRX_AC_RMS",
    "G1_AC_RMS", "G2_AC_RMS", "G3_AC_RMS",
    # -- AUTO_CORR_LAG_SEC ≈ 1/DOM_FREQ --
    "G1_AUTO_CORR_LAG_SEC", "G2_AUTO_CORR_LAG_SEC", "G3_AUTO_CORR_LAG_SEC",
    # -- DOM_FREQ 跨通道不变（同一心跳），consensus range/cv 始终为 0 --
    "G_consensus_DOM_FREQ_min", "G_consensus_DOM_FREQ_max",
    "G_consensus_DOM_FREQ_range", "G_consensus_DOM_FREQ_cv",
    "G_consensus_DOM_FREQ_top2_mean",
    # -- Hjorth_Activity = var(bp) ≈ (AC_RMS)²，仅 IRX/AMBX 砍 --
    "IRX_Hjorth_Activity", "AMBX_Hjorth_Activity",
    # -- Hjorth_Complexity 二阶导出，极不稳定 --
    "GREEN_Hjorth_Complexity", "IRX_Hjorth_Complexity", "AMBX_Hjorth_Complexity",
    # -- Entropy_Shannon 对连续信号区分度低，仅保留 GREEN --
    "IRX_Entropy_Shannon", "AMBX_Entropy_Shannon",
    # -- 与 IR_over_Gmean_mean 信息重叠 --
    "log_IR_Gmean_mean",
    # -- ≈ IRX_DERIV_MAD --
    "IR_diff_std",
    # -- Hjorth_Mobility ≈ Deriv_d1_std / AC_RMS，ratio 型 --
    "GREEN_Hjorth_Mobility", "IRX_Hjorth_Mobility", "AMBX_Hjorth_Mobility",
    # -- 手表佩戴姿态固定，区分度低 --
    "ACC_GRAVITY_DOM_RATIO",
    # Not in the deployment formula surface; keep them out of the final pool.
    "G_consensus_AC_MAD_range",
    "GREEN_FFT_harmonic_ratio", "GREEN_FFT_harmonic_present",
    # -- AC_DC_RATIO = AC_RMS / |DC|；当前单通道函数仍正确计算 AC_RMS 中间量，
    #    只是最终特征池中优先保留更鲁棒的 AC_MAD / |DC| 等价信息。--
}

# 低可解释性特征：这些量通常难以用“绿光稳定性 / 三通道一致性 / 环境光泄漏 /
# ACC 运动强度”直接解释，且在汇报和部署排查时成本较高。新增 Stage2 模型只从
# 可解释候选池中选特征，商用 s01_model.py 的 8 个特征和树参数不受影响。
_EXPLAINABILITY_BLOCKED_EXACT = {
    "G_MIN_CHANNEL_ID",
    "G_DROPOUT_ANGLE",
    "G_TOP2_CHANNEL_COUNT",
    "G_TOP2_WORST_IDX",
    "G_SPATIAL_STABILITY_SCORE",
    "G_TOP2_RANK_STABILITY",
    "G_TOP2_SWITCH_RATE",
    "corr_Gmean_vmag",
    "corr_Ambient_vmag",
    "ACC_STILL_X_GREEN_STABILITY",
    "ACC_DIFF_TO_GTOP2_DIFF_RATIO",
    "ACC_STILL_GREEN_MISMATCH",
}

_EXPLAINABILITY_BLOCKED_PATTERNS = (
    "_Entropy_",
    "_Hjorth_",
    "_bp_skewness",
    "_bp_kurtosis",
    "spatial_vmag",
    "SPATIAL_VMAG",
    "FFT_peak_width",
    "FFT_SNR",
    "harmonic_",
)


def is_explainable_stage2_feature(name):
    """Return True for feature names that are suitable for explainable Stage2 use."""
    if name in _EXPLAINABILITY_BLOCKED_EXACT:
        return False
    return not any(pattern in name for pattern in _EXPLAINABILITY_BLOCKED_PATTERNS)

# =========================================================
# H5 读取与 Stage1 工具
# =========================================================

def normalize_ppg_array(arr):
    """Normalize H5 PPG arrays whose last axis is points to (T, C) or (N_win, T_win, C)."""
    x = np.asarray(arr)
    if x.ndim == 2:
        if x.shape[0] != 40:
            raise ValueError(f"expected PPG shape (C,T) with C=40, got {x.shape}")
        return x.T
    if x.ndim == 3:
        if x.shape[1] != 40:
            raise ValueError(f"expected pre-windowed PPG shape (N,C,T) with C=40, got {x.shape}")
        return np.transpose(x, (0, 2, 1))
    raise ValueError(f"unsupported PPG ndim={x.ndim}, shape={x.shape}")


def normalize_acc_array(arr):
    """Normalize H5 ACC arrays whose last axis is points to (T, C) or (N_win, T_win, C)."""
    x = np.asarray(arr)
    if x.ndim == 2:
        if x.shape[0] > 6:
            raise ValueError(f"expected ACC shape (C,T), got {x.shape}")
        return x.T
    if x.ndim == 3:
        if x.shape[1] > 6:
            raise ValueError(f"expected pre-windowed ACC shape (N,C,T), got {x.shape}")
        return np.transpose(x, (0, 2, 1))
    raise ValueError(f"unsupported ACC ndim={x.ndim}, shape={x.shape}")


def is_prewindowed_signal(arr):
    return np.asarray(arr).ndim == 3


def flatten_prewindowed_signal(arr):
    x = np.asarray(arr)
    if x.ndim != 3:
        return x
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2])


def parse_grouped_window_name(name):
    """Return (window_index, label) parsed from names ending in *_w20_1."""
    match = WINDOW_NAME_RE.search(str(name))
    if not match:
        return None
    return int(match.group("index")), int(match.group("label"))


def _sorted_grouped_window_items(group):
    items = []
    for child_name in group.keys():
        parsed = parse_grouped_window_name(child_name)
        if parsed is None:
            continue
        child = group[child_name]
        if not isinstance(child, h5py.Group) or "ppg" not in child:
            continue
        window_index, label = parsed
        items.append((window_index, label, child_name, child))
    return sorted(items, key=lambda item: item[0])


def load_grouped_window_metadata(sample):
    if sample.get("window_layout") != "grouped_windows":
        return None
    indices = sample.get("window_indices")
    labels = sample.get("window_labels")
    names = sample.get("window_names")
    if indices is not None and labels is not None:
        return {
            "window_indices": [int(x) for x in indices],
            "window_labels": [int(x) for x in labels],
            "window_names": [str(x) for x in names] if names is not None else None,
        }
    with h5py.File(sample["h5_file"], "r") as f:
        items = _sorted_grouped_window_items(f[sample["sample_name"]])
    return {
        "window_indices": [int(item[0]) for item in items],
        "window_labels": [int(item[1]) for item in items],
        "window_names": [str(item[2]) for item in items],
    }


def load_ppg(sample):
    """
    Read PPG as continuous (T, C) or pre-windowed (N_win, T_win, C).
    Old H5 layout (C, T) remains supported.
    """
    with h5py.File(sample["h5_file"], "r") as f:
        grp = f[sample["sample_name"]]
        if sample.get("window_layout") == "grouped_windows" or "ppg" not in grp:
            windows = []
            for _idx, _label, _name, child in _sorted_grouped_window_items(grp):
                windows.append(normalize_ppg_array(child["ppg"][:]))
            if not windows:
                raise KeyError(f"sample {sample['sample_name']} has no grouped PPG windows")
            ppg = np.stack(windows, axis=0)
        else:
            ppg = normalize_ppg_array(grp["ppg"][:])
    return ppg


def load_acc(sample):
    """
    读取ACC数据：
        f[sample_name]['acc'][:].T
    即原始为 (3, N)，转成 (N, 3)
    如果没有acc数据，返回None
    """
    with h5py.File(sample["h5_file"], "r") as f:
        grp = f[sample["sample_name"]]
        if sample.get("window_layout") == "grouped_windows" or "ppg" not in grp:
            acc_windows = []
            for _idx, _label, _name, child in _sorted_grouped_window_items(grp):
                if "acc" not in child:
                    return None
                acc_windows.append(normalize_acc_array(child["acc"][:]))
            if not acc_windows:
                return None
            return np.stack(acc_windows, axis=0)
        if "acc" not in grp:
            return None
        acc = normalize_acc_array(grp["acc"][:])
    return acc

def _is_25hz_sample(sample):
    """检测样本是否已经是 25Hz 原生数据（名称含 sleep_25hz）。"""
    name = sample.get("sample_name", "") if isinstance(sample, dict) else str(sample)
    return "sleep_25hz" in name.lower()
def stage1_ambient_check(ppg, ambient_ratio_threshold=0.8):
    """Stage1 环境光检查: median(ambient) / median(ir) < threshold。

    比对值过高说明环境光异常强或传感器未贴紧皮肤。
    此函数保留用于极端硬过滤；比值本身作为特征写入特征池，让模型学习决策边界。
    """
    if is_prewindowed_signal(ppg):
        ppg = flatten_prewindowed_signal(ppg)
    if ppg is None or ppg.shape[1] < 2:
        return True  # no ambient channel, pass
    ir_dc = float(np.median(ppg[:, 0]))
    amb_dc = float(np.median(ppg[:, 1]))
    # 只过滤极端无效情况（IR 完全无信号，或环境光绝对淹没了IR）
    if ir_dc < 1e2:
        return False
    try:
        ambient_ratio_threshold = float(ambient_ratio_threshold)
    except (TypeError, ValueError):
        ambient_ratio_threshold = 0.8
    return (amb_dc / ir_dc) < ambient_ratio_threshold

def compute_ambient_stage1_features(raw_window):
    """计算环境光 Stage1 软特征（供模型学习，不硬过滤）。

    返回 dict 含 AMB_STAGE1_RATIO / AMB_STAGE1_PASS / IR_DC_LEVEL。
    用于区分"真离腕"和"佩戴但环境光强"。"""
    feats = {}
    if raw_window is None:
        feats["AMB_STAGE1_RATIO"] = 0.0
        feats["AMB_STAGE1_PASS"] = 1.0
        feats["IR_DC_LEVEL"] = 0.0
        return feats
    ppg_flat = flatten_prewindowed_signal(raw_window) if is_prewindowed_signal(raw_window) else raw_window
    if ppg_flat is None or ppg_flat.shape[1] < 2:
        feats["AMB_STAGE1_RATIO"] = 0.0
        feats["AMB_STAGE1_PASS"] = 1.0
        feats["IR_DC_LEVEL"] = 0.0
        return feats
    ir_dc = float(np.median(ppg_flat[:, 0]))
    amb_dc = float(np.median(ppg_flat[:, 1]))
    ratio = float(amb_dc / max(ir_dc, EPS))
    feats["AMB_STAGE1_RATIO"] = ratio
    feats["AMB_STAGE1_PASS"] = 1.0 if stage1_ambient_check(raw_window) else 0.0
    feats["IR_DC_LEVEL"] = ir_dc
    return feats


def downsample_to_5hz(signal, fs_original=100, fs_target=5):
    if fs_original == fs_target:
        return signal

    gcd = np.gcd(fs_original, fs_target)
    up = fs_target // gcd
    down = fs_original // gcd

    return resample_poly(signal, up, down)

def stage1_sample_pass(ppg, dc_threshold, ac_dc_threshold, ppg_fs=100):
    """
    Stage1 逻辑（与 s02 固定阈值配置保持一致）：
    IR 通道 ppg[:, 0]
    ppg_fs -> 5Hz
    3s窗口，15点
    只要任意一个 3s 窗口通过，就认为该 sample 进入第二阶段。

    pass 条件：
        dc > dc_threshold and ac_dc_ratio < ac_dc_threshold

    DC = min(neighbor_mean) where neighbor_mean[i] = (x[i] + x[i+1]) / 2
    AC = median(|diff(x)|)
    """
    if is_prewindowed_signal(ppg):
        ppg = flatten_prewindowed_signal(ppg)
    ir = ppg[:, 0]
    ir_5hz = downsample_to_5hz(ir, ppg_fs, 5)

    win = int(round(STAGE1_PRIMITIVE_SEC * STAGE1_FS))
    stride = win
    pass_count = 0

    for i in range(0, len(ir_5hz) - win + 1, stride):
        x = ir_5hz[i:i + win]

        # DC: min(邻均值) —— 对单点毛刺鲁棒（与 s02 一致）
        if len(x) >= 2:
            neighbor_mean = (x[:-1] + x[1:]) / 2.0
            dc = float(np.min(neighbor_mean))
        else:
            dc = float(np.mean(x))

        # AC: 邻差 MAD —— 与 s02 一致
        if len(x) >= 2:
            ac = float(np.median(np.abs(np.diff(x))))
        else:
            ac = 0.0

        ac_dc_ratio = ac / (np.abs(dc) + EPS)

        if dc > dc_threshold and ac_dc_ratio < ac_dc_threshold:
            pass_count += 1
        else:
            pass_count = 0

        if pass_count >= STAGE1_GATE_K:
            return True

    return False

# =========================================================
# 绿光通道构建逻辑
# =========================================================

def detect_green_mode(ppg):
    """
    直接复用你原来的模式识别逻辑。

    if mode1_var > mode2_var and mode1_var > 1e6:
        mode = 1
    else:
        mode = 2
    """
    if is_prewindowed_signal(ppg):
        ppg = flatten_prewindowed_signal(ppg)
    if ppg.shape[1] >= 6:
        var_ch0 = np.var(ppg[:, 0])
        var_ch3 = np.var(ppg[:, 3])
        var_ch4 = np.var(ppg[:, 4])
        var_ch5 = np.var(ppg[:, 5])

        mode1_var = (var_ch3 + var_ch4 + var_ch5) / 3.0
        mode2_var = var_ch0

        if mode1_var > mode2_var and mode1_var > 1e6:
            return 1
        else:
            return 2

    return 2

def get_channels_from_window(window, mode):
    """
    通道选择逻辑：
    
    IR: ch0
    
    Ambient: ch1（如果没有则退化为ch0）
    
    绿光（3通道独立）：
        mode=1 且通道数>=6:
            g1=ch3, g2=ch4, g3=ch5
        mode=2 且通道数>=16:
            g1=(ch7+ch10+ch13)/3
            g2=(ch8+ch11+ch14)/3
            g3=(ch9+ch12+ch15)/3
        否则:
            ch2作为绿光，退化为g1=g2=g3
    """
    ir = window[:, 0]

    if window.shape[1] > 1:
        ambient = window[:, 1]
    else:
        ambient = window[:, 0]

    if mode == 1 and window.shape[1] >= 6:
        g1 = window[:, 3]
        g2 = window[:, 4]
        g3 = window[:, 5]
    elif mode == 2 and window.shape[1] >= 16:
        g1 = (window[:, 6] + window[:, 9] + window[:, 12]) / 3.0
        g2 = (window[:, 7] + window[:, 10] + window[:, 13]) / 3.0
        g3 = (window[:, 8] + window[:, 11] + window[:, 14]) / 3.0
    else:
        if window.shape[1] >= 3:
            g = window[:, 2]
        else:
            g = window[:, 0]

        g1 = g
        g2 = g
        g3 = g

    return ir, ambient, g1, g2, g3

# =========================================================
# 边界情况处理
# =========================================================

def validate_window(ppg_window, min_channels=6, min_length=100):
    """
    验证窗口有效性
    
    参数:
        ppg_window: PPG窗口数据，shape (N, C)
        min_channels: 最小通道数要求，默认6
        min_length: 最小长度要求，默认100
        
    返回:
        bool: 窗口是否有效
    """
    if ppg_window is None:
        return False
    
    ppg = np.asarray(ppg_window, dtype=np.float64)
    
    if ppg.ndim == 1:
        ppg = ppg.reshape(-1, 1)
    
    if len(ppg) < min_length:
        return False
    
    if ppg.shape[1] < min_channels:
        return False
    
    return True


def validate_h5_file(h5_file, sample_name):
    """
    验证H5文件是否可读
    
    参数:
        h5_file: H5文件路径
        sample_name: 样本名称
        
    返回:
        tuple: (是否有效, 错误信息)
    """
    try:
        with h5py.File(h5_file, "r") as f:
            if sample_name not in f:
                return False, f"样本 {sample_name} 不存在于H5文件中"

            grp = f[sample_name]
            if "ppg" not in grp:
                items = _sorted_grouped_window_items(grp)
                if not items:
                    return False, f"样本 {sample_name} 缺少PPG数据"
                ppg = normalize_ppg_array(items[0][3]["ppg"][:])
                if ppg is None or len(ppg) == 0:
                    return False, f"样本 {sample_name} PPG数据为空"
                return True, None

            ppg = normalize_ppg_array(grp["ppg"][:])
            if ppg is None or len(ppg) == 0:
                return False, f"样本 {sample_name} PPG数据为空"
                
        return True, None
    except Exception as e:
        return False, f"H5文件读取失败: {str(e)}"


# =========================================================
# 鲁棒基础工具函数
# =========================================================

def safe_div(a, b, eps=EPS):
    return float(a) / (float(b) + eps)


STAGE2_IR_FEATURE_PREFIXES = (
    "IR_",
    "IRX_",
    "GREEN_IR_",
    "IR_AMB_",
    "IR_over_",
    "log_IR_",
    "corr_IR_",
    "ACC_IR_",
)

STAGE2_IR_FEATURE_NAMES = {
    "corr_Ambient_IR",
    "AMB_STAGE1_RATIO",
    "AMB_STAGE1_PASS",
    "IR_DC_LEVEL",
}


def is_stage2_ir_feature(name):
    """Return True for features that use the IR channel and must not enter Stage2."""
    n = str(name)
    return n in STAGE2_IR_FEATURE_NAMES or any(n.startswith(p) for p in STAGE2_IR_FEATURE_PREFIXES)


def filter_stage2_ir_features(features):
    """Remove all IR-derived Stage2 features while preserving input order."""
    if hasattr(features, "items"):
        return OrderedDict((k, v) for k, v in features.items() if not is_stage2_ir_feature(k))
    return [f for f in features if not is_stage2_ir_feature(f)]


DEPLOYMENT_ALLOWED_NON_FFT_FEATURES = {
    "SQI_FLAT_RATIO", "SQI_SPIKE_RATIO",
    "GREEN_ROBUST_RANGE_RATIO", "AMB_ROBUST_RANGE_RATIO",
    "GREEN_SEG_ACDC_CV", "AMB_SEG_ACDC_CV",
    "G_mean_mean", "G_mean_std", "G_mean_diff_std", "G_mean_acdc",
    "GREEN_DC_MEDIAN", "GREEN_DC_IQR", "GREEN_AC_RMS", "GREEN_AC_MAD",
    "GREEN_AC_DC_RATIO", "GREEN_DERIV_MAD",
    "Ambient_mean", "Ambient_std", "Ambient_p95", "corr_Ambient_Gmean",
    "AMBX_DC_MEDIAN", "AMBX_DC_IQR", "AMBX_AC_RMS", "AMBX_AC_MAD",
    "AMBX_AC_DC_RATIO", "AMBX_DERIV_MAD",
    "GREEN_AC", "AMB_AC", "GREEN_DC", "AMB_DC", "GREEN_CORR", "GREEN_XCORR",
    "AMB_AC_TO_GREEN_AC", "AMB_DC_TO_GREEN_DC",
    "GREEN_AMB_BP_CORR", "GREEN_AMB_ENV_CORR", "GREEN_AMB_LEAK",
    "G_imbalance_mean", "G_imbalance_p90", "G_imbalance_iqr",
    "G_rangeNorm_mean", "G_rangeNorm_p90",
    "G_spatial_vmag_mean", "G_spatial_vmag_p90", "G_spatial_vmag_iqr",
    "G_spatial_vmag_std", "G_ch_dc_cv", "G_ch_dc_max_min_ratio",
    "GCH_DC_RANGE_RATIO", "GCH_AC_RANGE_RATIO",
    "G_2OF3_AC_SUPPORT", "G_TOP2_TO_ALL_AC_RATIO", "G_TOP2_CORR_MIN",
    "G_WEAK_CHANNEL_GAP", "G_SPATIAL_STABILITY_SCORE",
    "G_TOP1_TO_TOP2_AC_RATIO", "G_TOP2_RANK_STABILITY", "G_TOP2_SWITCH_RATE",
    "G_SPATIAL_VMAG_RANGE", "GREEN_AMB_SEG_CORR_RANGE",
    "GTOP2_ROBUST_RANGE_RATIO", "GTOP2_SEG_ACDC_CV",
    "GTOP2_DC_MEDIAN", "GTOP2_DC_IQR", "GTOP2_AC_RMS", "GTOP2_AC_MAD",
    "GTOP2_AC_DC_RATIO", "GTOP2_DERIV_MAD",
    "GTOP2_bp_skewness", "GTOP2_bp_kurtosis",
    "GTOP2_zero_cross_rate", "GTOP2_abs_diff_ratio",
    "GTOP2_HALF_ACDC_DELTA", "GTOP2_SEG_ACDC_RANGE",
    "GREEN_AMB_LEAK_STABILITY",
    "ACC_MAG_MEAN", "ACC_MAG_STD", "ACC_MAG_MAD", "ACC_AXIS_STD_SUM",
    "ACC_GRAVITY_DOM_RATIO", "ACC_BP_RMS", "ACC_DIFF_MAD", "ACC_STILL_SCORE",
    "ACC_MAG_P50", "ACC_MAG_P90", "ACC_YSUM",
    "ACC_X_MEAN", "ACC_Y_MEAN", "ACC_Z_MEAN",
    "ACC_X_STD", "ACC_Y_STD", "ACC_Z_STD",
    "ACC_X_ENERGY", "ACC_Y_ENERGY", "ACC_Z_ENERGY",
    "ACC_AXIS_MEAN_SUM", "ACC_MAG_ENERGY", "ACC_MAG_P2P",
    "ACC_TILT_ANGLE", "ACC_DOM_AXIS", "ACC_GRAVITY_RATIO",
    "ACC_ENERGY_TO_GREEN_AC", "ACC_GREEN_BP_CORR",
    "ACC_TO_GTOP2_AC_RATIO", "ACC_STILL_X_GREEN_STABILITY",
    "ACC_DIFF_TO_GTOP2_DIFF_RATIO", "ACC_STILL_GREEN_MISMATCH",
    "SIG_LEN", "SIG_SEC", "mode",
    "TOTAL_INVALID_COUNT", "PPG_INVALID_COUNT", "GREEN_INVALID_COUNT",
}

DEPLOYMENT_ALLOWED_FFT_FEATURES = {
    "GTOP2_BAND_ENERGY_RATIO",
    "GTOP2_FFT_PEAK_MEDIAN_RATIO",
    "GTOP2_DOM_FREQ",
    "GREEN_BAND_ENERGY_RATIO",
    "GREEN_FFT_PEAK_MEDIAN_RATIO",
    "GREEN_DOM_FREQ",
    "FFT_PEAK_MEDIAN_RATIO",
    "AMB_BAND_ENERGY_RATIO",
    "AMB_FFT_PEAK_MEDIAN_RATIO",
    "AMB_DOM_FREQ",
    "AMBX_FFT_PEAK_MEDIAN_RATIO",
    "AMBX_DOM_FREQ",
}

DEPLOYMENT_ALLOWED_NON_FFT_FEATURES = {
    name for name in DEPLOYMENT_ALLOWED_NON_FFT_FEATURES
    if is_explainable_stage2_feature(name)
}
DEPLOYMENT_ALLOWED_FFT_FEATURES = {
    name for name in DEPLOYMENT_ALLOWED_FFT_FEATURES
    if is_explainable_stage2_feature(name)
}


def is_deployment_friendly_stage2_feature(name):
    """Return True for Stage2 features that are both deployment-practical and explainable."""
    return is_explainable_stage2_feature(name)


def filter_deployment_friendly_stage2_features(features):
    """Filter Stage2 features by the deployment-friendly allowlist."""
    filtered = filter_stage2_ir_features(features)
    if hasattr(filtered, "items"):
        return OrderedDict(
            (k, v) for k, v in filtered.items()
            if is_deployment_friendly_stage2_feature(k)
        )
    return [f for f in filtered if is_deployment_friendly_stage2_feature(f)]


def robust_mad(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0

    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def robust_iqr(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0

    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def safe_corr(x, y):
    """计算相关系数（带缓存优化）"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = min(len(x), len(y))
    if n < 8:
        return 0.0

    x = x[:n]
    y = y[:n]

    x = x - np.mean(x)
    y = y - np.mean(y)

    sx = np.std(x)
    sy = np.std(y)

    if sx < EPS or sy < EPS:
        return 0.0

    v = np.mean((x / sx) * (y / sy))

    if not np.isfinite(v):
        return 0.0

    return float(v)


def moving_average_filter(x, window_size=5):
    x = np.asarray(x, dtype=np.float64)

    if len(x) < window_size or window_size < 2:
        return x.copy()

    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(x, kernel, mode="same")

# =========================================================
# 鲁棒预处理
# =========================================================

def remove_burr(x, burr_k=6.0):
    """
    去毛刺：
    当前点与左右点都差异很大时，用邻点均值替换。

    向量化：判定/替换都基于原始邻点值，无序贯依赖。
    """
    x = np.asarray(x, dtype=np.float64).copy()

    if len(x) < 3:
        return x

    d = np.diff(x)
    mad_d = robust_mad(d)
    thr = max(burr_k * mad_d, EPS)

    left = x[:-2]
    mid = x[1:-1]
    right = x[2:]
    bad = (np.abs(mid - left) > thr) & (np.abs(mid - right) > thr)
    if bad.any():
        replaced = 0.5 * (left + right)
        x[1:-1] = np.where(bad, replaced, mid)

    return x

def remove_step(x, step_k=10.0):
    """
    去跳变：
    相邻点差异过大时，钳制为前一点。

    注意：
    这个规则偏保守，能抑制突发跳点；
    但如果真实戴摘瞬间进入窗口，也会被平滑掉一部分。
    对 3s 活体窗口通常是可以接受的。
    """
    x = np.asarray(x, dtype=np.float64).copy()

    if len(x) < 2:
        return x

    d = np.diff(x)
    mad_d = robust_mad(d)
    thr = max(step_k * mad_d, EPS)

    for i in range(1, len(x)):
        if abs(x[i] - x[i - 1]) > thr:
            x[i] = x[i - 1]

    return x

# 全局缓存 FIR bandpass 核，避免每窗重算。C 可直译：sinc + Hamming 窗 + 卷积。
_FIR_BANDPASS_CACHE = {}


def _get_fir_bandpass_kernel(fs, lowcut, highcut, numtaps=65):
    """Windowed-sinc FIR bandpass kernel. Zero-phase via forward-backward convolve."""
    key = (float(fs), float(lowcut), float(highcut), int(numtaps))
    if key in _FIR_BANDPASS_CACHE:
        return _FIR_BANDPASS_CACHE[key]
    nyq = 0.5 * fs
    lo = lowcut / nyq
    hi = highcut / nyq
    # Ideal bandpass = lowpass(hi) - lowpass(lo)
    t = np.arange(numtaps, dtype=np.float64) - (numtaps - 1) / 2.0
    h = np.sinc(hi * t) * hi - np.sinc(lo * t) * lo
    h *= np.hamming(numtaps)
    h /= np.sum(np.abs(h)) + 1e-12
    _FIR_BANDPASS_CACHE[key] = h
    return h


def bandpass_filter(x, fs, lowcut=0.4, highcut=6.0, order=None, numtaps=65):
    """Windowed-sinc FIR bandpass (zero-phase via forward-backward convolve).

    Equivalent C pattern:
      - Design kernel once (sinc + Hamming window).
      - Convolve forward, then convolve reversed for zero-phase.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < numtaps:
        return x.copy()
    h = _get_fir_bandpass_kernel(fs, lowcut, highcut, numtaps)
    # Forward-backward convolve for zero-phase (like filtfilt)
    forward = np.convolve(x, h, mode='same')
    return np.convolve(forward[::-1], h, mode='same')[::-1]


def _median_filter_np(x, kernel_size):
    """Numpy median filter - directly portable to C (rolling window median)."""
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    half = k // 2
    n = len(x)
    out = np.empty(n, dtype=x.dtype)
    if n == 0:
        return out
    if n >= k:
        windows = np.lib.stride_tricks.sliding_window_view(x, k)
        out[half:n - half] = np.median(windows, axis=1)
    for i in list(range(0, min(half, n))) + list(range(max(half, n - half), n)):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out

def preprocess_signal(x, fs):
    """
    返回：
        raw_clean: 清理后的原始信号，用于 DC / IQR / raw corr
        bp: 带通信号，用于 AC / 频谱 / 相关 / 自相关
        dc: 直流中值

    滤波核为时间自适应（fs 变化时保持一致的时间尺度）：
        median_filter ≈ 50ms, moving_avg ≈ 30ms (all numpy, C-portable)
    """
    x = np.asarray(x, dtype=np.float64).copy()

    x = remove_burr(x, burr_k=6.0)
    x = remove_step(x, step_k=10.0)

    # 时间自适应: 50ms 中值滤波 (min 3)
    mf_kernel = max(3, int(round(0.05 * fs)))
    if mf_kernel % 2 == 0:
        mf_kernel += 1
    if len(x) >= mf_kernel:
        try:
            x = _median_filter_np(x, kernel_size=mf_kernel)
        except Exception:
            pass

    # 时间自适应: 30ms 滑动平均 (min 2)
    ma_win = max(2, int(round(0.03 * fs)))
    x = moving_average_filter(x, window_size=ma_win)

    dc = float(np.median(x))

    bp = bandpass_filter(x, fs, lowcut=0.4, highcut=6.0, order=4)
    bp = moving_average_filter(bp, window_size=ma_win)

    return x, bp, dc

# =========================================================
# 周期性 / 频域特征
# =========================================================

def fft_peak_features(x, fs, fmin=0.5, fmax=5.0):
    """
    返回：
        peak_median_ratio
        dom_freq
    """
    x = np.asarray(x, dtype=np.float64)

    if len(x) < 16:
        return 0.0, 0.0

    x = x - np.mean(x)
    xw = x * np.hamming(len(x))

    nfft = 1
    while nfft < len(x):
        nfft <<= 1

    nfft = max(256, nfft)

    spec = np.abs(np.fft.rfft(xw, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(mask):
        return 0.0, 0.0

    band_spec = spec[mask]
    band_freqs = freqs[mask]

    med = np.median(band_spec)

    if med < EPS:
        peak_ratio = 0.0
    else:
        peak_ratio = float(np.max(band_spec) / (med + EPS))

    dom_freq = float(band_freqs[np.argmax(band_spec)])

    return peak_ratio, dom_freq


def compute_fft_cache(x, fs, fmin=0.5, fmax=5.0):
    """
    一次性计算FFT，返回所有需要的信息（避免重复计算）
    
    返回：
        dict: 包含 peak_ratio, dom_freq, spec, freqs 等
    """
    x = np.asarray(x, dtype=np.float64)
    
    result = {
        'peak_ratio': 0.0,
        'dom_freq': 0.0,
        'spec': None,
        'freqs': None,
        'band_spec': None,
        'band_freqs': None
    }
    
    if len(x) < 16:
        return result
    
    x = x - np.mean(x)
    xw = x * np.hamming(len(x))
    
    nfft = 1
    while nfft < len(x):
        nfft <<= 1
    
    nfft = max(256, nfft)
    
    spec = np.abs(np.fft.rfft(xw, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    
    mask = (freqs >= fmin) & (freqs <= fmax)
    
    if not np.any(mask):
        result['spec'] = spec
        result['freqs'] = freqs
        return result
    
    band_spec = spec[mask]
    band_freqs = freqs[mask]
    
    med = np.median(band_spec)
    
    if med < EPS:
        peak_ratio = 0.0
    else:
        peak_ratio = float(np.max(band_spec) / (med + EPS))
    
    dom_freq = float(band_freqs[np.argmax(band_spec)])
    
    result['peak_ratio'] = peak_ratio
    result['dom_freq'] = dom_freq
    result['spec'] = spec
    result['freqs'] = freqs
    result['band_spec'] = band_spec
    result['band_freqs'] = band_freqs
    
    return result

def normalized_autocorr(x):
    x = np.asarray(x, dtype=np.float64)

    if len(x) < 4:
        return np.zeros(1, dtype=np.float64)

    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(x) - 1:]

    if corr[0] < EPS:
        return np.zeros_like(corr)

    return corr / (corr[0] + EPS)

def autocorr_periodicity_features(x, fs, bpm_min=40.0, bpm_max=180.0):
    """
    返回：
        ac_peak
        ac_lag_sec
    """
    x = np.asarray(x, dtype=np.float64)

    if len(x) < int(fs * 1.5):
        return 0.0, 0.0

    ac = normalized_autocorr(x)

    lag_min = int(fs * 60.0 / bpm_max)
    lag_max = int(fs * 60.0 / bpm_min)

    lag_min = max(1, lag_min)
    lag_max = min(len(ac) - 1, lag_max)

    if lag_max <= lag_min:
        return 0.0, 0.0

    seg = ac[lag_min:lag_max + 1]

    if len(seg) == 0:
        return 0.0, 0.0

    idx = int(np.argmax(seg))
    peak = float(seg[idx])
    lag = lag_min + idx
    lag_sec = float(lag / fs)

    return peak, lag_sec

def max_norm_xcorr(x, y, max_lag_samples):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = min(len(x), len(y))

    if n < 8:
        return 0.0

    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])

    sx = np.std(x)
    sy = np.std(y)

    if sx < EPS or sy < EPS:
        return 0.0

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-n + 1, n)

    mask = np.abs(lags) <= max_lag_samples

    corr = corr[mask]
    corr = corr / (n * sx * sy + EPS)

    if len(corr) == 0:
        return 0.0

    return float(np.max(np.abs(corr)))

def smooth_envelope(x, fs, win_sec=0.25):
    x = np.abs(np.asarray(x, dtype=np.float64))

    win = max(3, int(round(win_sec * fs)))

    if win % 2 == 0:
        win += 1

    kernel = np.ones(win, dtype=np.float64) / win

    return np.convolve(x, kernel, mode="same")


def finite_signal(x, fill_value=0.0):
    """Replace NaN/Inf with the finite median so robust features stay finite."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if finite.any():
        fill = float(np.median(arr[finite]))
    else:
        fill = float(fill_value)
    return np.where(finite, arr, fill).astype(np.float64, copy=False)


# =========================================================
# 5s / 25Hz window robust features
# =========================================================

def robust_range_ratio(x):
    """Robust dynamic range normalized by median level."""
    x = finite_signal(x)
    if len(x) < 4:
        return 0.0
    p95, p5 = np.percentile(x, [95, 5])
    med = float(np.median(x))
    return safe_div(float(p95 - p5), abs(med) + EPS)


def segment_acdc_cv(raw, n_segments=3):
    """CV of segment AC/DC values (window divided into n_segments equal parts)."""
    raw = finite_signal(raw)
    if len(raw) < n_segments * 4:
        return 0.0
    seg_len = max(1, len(raw) // n_segments)
    vals = []
    for i in range(n_segments):
        seg = raw[i * seg_len:(i + 1) * seg_len] if i < n_segments - 1 else raw[i * seg_len:]
        if len(seg) < 4:
            continue
        dc = float(np.median(seg))
        ac = robust_mad(np.diff(seg)) if len(seg) > 1 else 0.0
        vals.append(safe_div(ac, abs(dc) + EPS))
    if len(vals) < 2:
        return 0.0
    vals = np.asarray(vals, dtype=np.float64)
    return safe_div(float(np.std(vals)), abs(float(np.mean(vals))) + EPS)


def segment_acdc_values(raw, n_segments=3):
    """Return segment AC/DC values using the same primitive as segment_acdc_cv."""
    raw = finite_signal(raw)
    if len(raw) < n_segments * 4:
        return []
    seg_len = max(1, len(raw) // n_segments)
    vals = []
    for i in range(n_segments):
        seg = raw[i * seg_len:(i + 1) * seg_len] if i < n_segments - 1 else raw[i * seg_len:]
        if len(seg) < 4:
            continue
        dc = float(np.median(seg))
        ac = robust_mad(np.diff(seg)) if len(seg) > 1 else 0.0
        vals.append(safe_div(ac, abs(dc) + EPS))
    return vals


def zero_cross_rate(x):
    x = finite_signal(x)
    if len(x) < 2:
        return 0.0
    centered = x - np.median(x)
    signs = np.sign(centered)
    return float(np.mean(signs[1:] * signs[:-1] < 0))


def bp_shape_features(bp, prefix):
    bp = finite_signal(bp)
    feat = OrderedDict()
    if len(bp) < 3:
        feat[f"{prefix}_bp_skewness"] = 0.0
        feat[f"{prefix}_bp_kurtosis"] = 0.0
        feat[f"{prefix}_zero_cross_rate"] = 0.0
        feat[f"{prefix}_abs_diff_ratio"] = 0.0
        return feat
    mu = float(np.mean(bp))
    sigma = float(np.std(bp))
    if sigma > EPS:
        z = (bp - mu) / sigma
        feat[f"{prefix}_bp_skewness"] = float(np.mean(z ** 3))
        feat[f"{prefix}_bp_kurtosis"] = float(np.mean(z ** 4))
    else:
        feat[f"{prefix}_bp_skewness"] = 0.0
        feat[f"{prefix}_bp_kurtosis"] = 0.0
    feat[f"{prefix}_zero_cross_rate"] = zero_cross_rate(bp)
    feat[f"{prefix}_abs_diff_ratio"] = safe_div(
        float(np.mean(np.abs(np.diff(bp)))),
        float(np.mean(np.abs(bp - np.median(bp)))) + EPS,
    )
    return feat


def segment_stability_features(raw, prefix):
    feat = OrderedDict()
    half_vals = segment_acdc_values(raw, n_segments=2)
    third_vals = segment_acdc_values(raw, n_segments=3)
    if len(half_vals) == 2:
        feat[f"{prefix}_HALF_ACDC_DELTA"] = safe_div(
            abs(float(half_vals[0]) - float(half_vals[1])),
            abs(float(np.mean(half_vals))) + EPS,
        )
    else:
        feat[f"{prefix}_HALF_ACDC_DELTA"] = 0.0
    if len(third_vals) >= 2:
        feat[f"{prefix}_SEG_ACDC_RANGE"] = safe_div(
            float(np.max(third_vals) - np.min(third_vals)),
            abs(float(np.mean(third_vals))) + EPS,
        )
    else:
        feat[f"{prefix}_SEG_ACDC_RANGE"] = 0.0
    return feat


def ambient_green_leak_stability(amb_raw, green_raw):
    amb_vals = segment_acdc_values(amb_raw, n_segments=3)
    green_vals = segment_acdc_values(green_raw, n_segments=3)
    n = min(len(amb_vals), len(green_vals))
    if n < 2:
        return 0.0
    ratios = [safe_div(amb_vals[i], green_vals[i] + EPS) for i in range(n)]
    return safe_div(float(np.std(ratios)), abs(float(np.mean(ratios))) + EPS)


def segment_corr_range(x, y, n_segments=3):
    x = finite_signal(x)
    y = finite_signal(y)
    n = min(len(x), len(y))
    if n < n_segments * 4:
        return 0.0
    x = x[:n]
    y = y[:n]
    seg_len = max(1, n // n_segments)
    vals = []
    for i in range(n_segments):
        xs = x[i * seg_len:(i + 1) * seg_len] if i < n_segments - 1 else x[i * seg_len:]
        ys = y[i * seg_len:(i + 1) * seg_len] if i < n_segments - 1 else y[i * seg_len:]
        if len(xs) >= 4 and len(ys) >= 4:
            vals.append(safe_corr(xs, ys))
    if len(vals) < 2:
        return 0.0
    return float(np.max(vals) - np.min(vals))


def band_energy_ratio_from_fft_cache(fft_cache, low=0.7, high=3.0):
    """Physiological-band energy ratio using the cached short-window FFT."""
    spec = fft_cache.get("spec")
    freqs = fft_cache.get("freqs")
    if spec is None or freqs is None or len(spec) == 0:
        return 0.0
    spec = finite_signal(spec)
    freqs = finite_signal(freqs)
    total_mask = (freqs >= 0.5) & (freqs <= 5.0)
    band_mask = (freqs >= low) & (freqs <= high)
    total = float(np.sum(spec[total_mask] ** 2))
    band = float(np.sum(spec[band_mask] ** 2))
    return safe_div(band, total + EPS)


def _diff_flat_ratio(x):
    x = finite_signal(x)
    if len(x) < 2:
        return 1.0
    d = np.abs(np.diff(x))
    scale = max(abs(float(np.median(x))), 1.0)
    tol = max(scale * 1e-7, 1e-6)
    return float(np.mean(d <= tol))


def _diff_spike_ratio(x, k=6.0):
    x = finite_signal(x)
    if len(x) < 3:
        return 0.0
    d = np.abs(np.diff(x))
    mad = robust_mad(d)
    if mad <= EPS:
        return 0.0
    return float(np.mean(d > k * mad))


def short_window_sqi_features(ir_raw_in, amb_raw_in, g_raw_in):
    """Fast Stage2 SQI features from ambient and green only.

    IR is intentionally ignored here: Stage1 owns IR gating, and Stage2 must not
    carry hidden IR information through generic SQI feature names.
    """
    channels = [
        finite_signal(amb_raw_in),
        finite_signal(g_raw_in),
    ]
    return OrderedDict([
        ("SQI_FLAT_RATIO", float(np.mean([_diff_flat_ratio(x) for x in channels]))),
        ("SQI_SPIKE_RATIO", float(np.mean([_diff_spike_ratio(x) for x in channels]))),
    ])

# =========================================================
# 单通道特征
# =========================================================

def extract_single_channel_features(raw, bp, dc, fs, prefix, fft_cache=None):
    """
    单通道特征提取
    
    参数：
        fft_cache: 可选的FFT缓存字典，避免重复计算
    """
    feat = OrderedDict()

    raw = np.asarray(raw, dtype=np.float64)
    bp = np.asarray(bp, dtype=np.float64)

    if len(raw) == 0 or len(bp) == 0:
        return feat

    ac_rms = float(np.sqrt(np.mean(bp ** 2)))
    ac_mad = robust_mad(bp)
    dc_iqr = robust_iqr(raw)

    deriv = np.diff(bp) if len(bp) > 1 else np.array([0.0])
    deriv_mad = robust_mad(deriv)

    # 使用缓存的FFT结果，避免重复计算
    if fft_cache is not None:
        fft_peak_ratio = fft_cache.get('peak_ratio', 0.0)
        dom_freq = fft_cache.get('dom_freq', 0.0)
    else:
        fft_peak_ratio, dom_freq = fft_peak_features(bp, fs, fmin=0.5, fmax=5.0)
    
    ac_peak, ac_lag_sec = autocorr_periodicity_features(bp, fs, bpm_min=40.0, bpm_max=180.0)

    feat[f"{prefix}_DC_MEDIAN"] = float(dc)
    feat[f"{prefix}_DC_IQR"] = dc_iqr
    feat[f"{prefix}_AC_RMS"] = ac_rms
    feat[f"{prefix}_AC_MAD"] = ac_mad
    feat[f"{prefix}_AC_DC_RATIO"] = safe_div(ac_rms, abs(dc) + EPS)
    feat[f"{prefix}_DERIV_MAD"] = deriv_mad
    feat[f"{prefix}_FFT_PEAK_MEDIAN_RATIO"] = fft_peak_ratio
    feat[f"{prefix}_DOM_FREQ"] = dom_freq
    feat[f"{prefix}_AUTO_CORR_PEAK"] = ac_peak
    feat[f"{prefix}_AUTO_CORR_LAG_SEC"] = ac_lag_sec

    return feat

# =========================================================
# 三通道绿光空间特征
# =========================================================

def extract_green_spatial_features(g1_raw, g2_raw, g3_raw, g1_bp, g2_bp, g3_bp,
                                   g1_input=None, g2_input=None, g3_input=None):
    feat = OrderedDict()
    eps = 1e-8

    g_stack = np.vstack([g1_raw, g2_raw, g3_raw])
    # ---------- 空间不均衡 ----------
    g_spatial_std = np.std(g_stack, axis=0)
    g_spatial_mean = np.mean(g_stack, axis=0)

    g_imbalance = g_spatial_std / (np.abs(g_spatial_mean) + eps)

    feat["G_imbalance_mean"] = float(np.mean(g_imbalance))
    feat["G_imbalance_p90"] = float(np.percentile(g_imbalance, 90))
    feat["G_imbalance_iqr"] = robust_iqr(g_imbalance)

    # ---------- 归一化极差 ----------
    g_max = np.max(g_stack, axis=0)
    g_min = np.min(g_stack, axis=0)

    g_range_norm = (g_max - g_min) / (
        np.abs(g1_raw) + np.abs(g2_raw) + np.abs(g3_raw) + eps
    )

    feat["G_rangeNorm_mean"] = float(np.mean(g_range_norm))
    feat["G_rangeNorm_p90"] = float(np.percentile(g_range_norm, 90))

    # ---------- 中心对称空间向量 ----------
    vx = g1_raw - 0.5 * g2_raw - 0.5 * g3_raw
    vy = (np.sqrt(3) / 2.0) * (g2_raw - g3_raw)

    vmag = np.sqrt(vx ** 2 + vy ** 2) / (
        np.abs(g1_raw) + np.abs(g2_raw) + np.abs(g3_raw) + eps
    )

    feat["G_spatial_vmag_mean"] = float(np.mean(vmag))
    feat["G_spatial_vmag_p90"] = float(np.percentile(vmag, 90))
    feat["G_spatial_vmag_iqr"] = robust_iqr(vmag)
    feat["G_spatial_vmag_std"] = float(np.std(vmag))
    feat["G_SPATIAL_VMAG_RANGE"] = float(np.percentile(vmag, 90) - np.percentile(vmag, 10))

    # ---------- 三通道 DC 相对强弱 ----------
    ch_dc = np.array([
        np.median(g1_raw),
        np.median(g2_raw),
        np.median(g3_raw)
    ], dtype=np.float64)

    feat["G_ch_dc_cv"] = float(np.std(ch_dc) / (np.abs(np.mean(ch_dc)) + eps))
    feat["G_ch_dc_max_min_ratio"] = float(
        np.max(np.abs(ch_dc)) / (np.min(np.abs(ch_dc)) + eps)
    )

    # ---------- 三通道 BP 一致性 ----------
    c12 = safe_corr(g1_bp, g2_bp)
    c23 = safe_corr(g2_bp, g3_bp)
    c31 = safe_corr(g3_bp, g1_bp)

    feat["G_bp_corr_mean"] = float(np.mean([c12, c23, c31]))
    feat["G_bp_corr_min"] = float(np.min([c12, c23, c31]))
    feat["G_bp_corr_std"] = float(np.std([c12, c23, c31]))

    # ---------- 三绿光可靠性 ----------
    # These scalar features keep the 120-degree green-channel geometry deployable:
    # no voting state, just compact reliability cues for XGBoost.
    ch_bp = [g1_bp, g2_bp, g3_bp]
    ch_raw = [
        g1_raw if g1_input is None else np.asarray(g1_input, dtype=np.float64),
        g2_raw if g2_input is None else np.asarray(g2_input, dtype=np.float64),
        g3_raw if g3_input is None else np.asarray(g3_input, dtype=np.float64),
    ]
    ch_ac = np.array([
        float(np.sqrt(np.mean((x - np.median(x)) ** 2))) for x in ch_raw
    ], dtype=np.float64)
    max_ac = float(np.max(ch_ac)) if len(ch_ac) else 0.0
    total_ac = float(np.sum(ch_ac))
    if max_ac <= eps:
        feat["G_2OF3_AC_SUPPORT"] = 0.0
        feat["G_TOP2_TO_ALL_AC_RATIO"] = 0.0
        feat["G_TOP2_CORR_MIN"] = 0.0
        feat["G_WEAK_CHANNEL_GAP"] = 0.0
        feat["G_SPATIAL_STABILITY_SCORE"] = 0.0
        feat["G_TOP1_TO_TOP2_AC_RATIO"] = 0.0
        feat["G_TOP2_RANK_STABILITY"] = 0.0
        feat["G_TOP2_SWITCH_RATE"] = 0.0
    else:
        support_count = int(np.sum(ch_ac >= 0.5 * max_ac))
        top2_idx = np.argsort(ch_ac)[-2:]
        top2_ac = ch_ac[top2_idx]
        top2_corr = safe_corr(ch_bp[int(top2_idx[0])], ch_bp[int(top2_idx[1])])
        top2_mean_ac = float(np.mean(top2_ac))
        weak_ac = float(np.min(ch_ac))
        weak_gap = safe_div(top2_mean_ac - weak_ac, top2_mean_ac + eps)
        corr_quality = max(0.0, float(top2_corr))
        spatial_penalty = 1.0 / (1.0 + float(np.mean(vmag)))

        feat["G_2OF3_AC_SUPPORT"] = float(support_count / 3.0)
        feat["G_TOP2_TO_ALL_AC_RATIO"] = safe_div(float(np.sum(top2_ac)), total_ac + eps)
        feat["G_TOP2_CORR_MIN"] = float(top2_corr)
        feat["G_WEAK_CHANNEL_GAP"] = float(max(0.0, weak_gap))
        feat["G_SPATIAL_STABILITY_SCORE"] = float(
            feat["G_2OF3_AC_SUPPORT"] * corr_quality * spatial_penalty
        )
        feat["G_TOP1_TO_TOP2_AC_RATIO"] = safe_div(max_ac, top2_mean_ac + eps)
        global_top2 = set(int(i) for i in top2_idx)
        n_seg = 3
        n = min(len(ch_raw[0]), len(ch_raw[1]), len(ch_raw[2]))
        seg_len = max(1, n // n_seg)
        switches = []
        for i in range(n_seg):
            lo = i * seg_len
            hi = (i + 1) * seg_len if i < n_seg - 1 else n
            if hi - lo < 4:
                continue
            seg_ac = np.array([
                float(np.sqrt(np.mean((np.asarray(ch_raw[j][lo:hi]) - np.median(ch_raw[j][lo:hi])) ** 2)))
                for j in range(3)
            ], dtype=np.float64)
            seg_top2 = set(int(j) for j in np.argsort(seg_ac)[-2:])
            switches.append(0.0 if seg_top2 == global_top2 else 1.0)
        switch_rate = float(np.mean(switches)) if switches else 0.0
        feat["G_TOP2_SWITCH_RATE"] = switch_rate
        feat["G_TOP2_RANK_STABILITY"] = float(1.0 - switch_rate)

    return feat, g_imbalance, vmag

# =========================================================
# 通道间特征
# =========================================================

def extract_cross_channel_features(g_raw, g_bp, g_dc,
                                   ir_raw, ir_bp, ir_dc,
                                   amb_raw, amb_bp,
                                   fs,
                                   fft_cache_green=None,
                                   fft_cache_ir=None):
    """
    通道间特征提取
    
    参数：
        fft_cache_green: 绿光FFT缓存
        fft_cache_ir: IR FFT缓存
    """
    feat = OrderedDict()

    g_env = smooth_envelope(g_bp, fs)
    amb_env = smooth_envelope(amb_bp, fs)

    # 使用缓存的FFT结果，避免重复计算
    if fft_cache_green is not None:
        g_dom = fft_cache_green.get('dom_freq', 0.0)
    else:
        _, g_dom = fft_peak_features(g_bp, fs, fmin=0.5, fmax=5.0)
    
    feat["GREEN_AMB_BP_CORR"] = safe_corr(g_bp, amb_bp)
    feat["GREEN_AMB_ENV_CORR"] = safe_corr(g_env, amb_env)

    g_rms = np.sqrt(np.mean(g_bp ** 2)) + EPS
    amb_rms = np.sqrt(np.mean(amb_bp ** 2)) + EPS

    feat["GREEN_AMB_LEAK"] = abs(feat["GREEN_AMB_BP_CORR"]) * safe_div(amb_rms, g_rms)

    return feat

# =========================================================
# ACC 特征（轻量版）
# =========================================================

def _acc_magnitude(acc_window):
    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    return np.sqrt(np.sum(acc * acc, axis=1) + 1e-12)


def extract_acc_features(acc_window, fs=100.0, prefix="ACC"):
    feats = OrderedDict()

    _all_keys = ["MAG_MEAN", "MAG_STD", "MAG_MAD", "AXIS_STD_SUM",
                 "GRAVITY_DOM_RATIO", "BP_RMS", "DIFF_MAD", "STILL_SCORE",
                 "MAG_P50", "MAG_P90", "YSUM",
                 "X_MEAN", "X_STD", "X_ENERGY",
                 "Y_MEAN", "Y_STD", "Y_ENERGY",
                 "Z_MEAN", "Z_STD", "Z_ENERGY",
                 "AXIS_MEAN_SUM", "MAG_ENERGY", "MAG_P2P",
                 "TILT_ANGLE", "DOM_AXIS", "GRAVITY_RATIO"]

    if acc_window is None or len(acc_window) < 4:
        for k in _all_keys:
            feats[f"{prefix}_{k}"] = 0.0
        return feats

    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)

    mag = _acc_magnitude(acc)
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    mag_mad = robust_mad(mag)
    mag_energy = float(np.sum(mag ** 2))
    mag_p2p = float(np.max(mag) - np.min(mag))

    axis_std = np.std(acc, axis=0)
    axis_std_sum = float(np.sum(axis_std))

    axis_mean_abs = np.abs(np.mean(acc, axis=0))
    dom_axis_ratio = float(np.max(axis_mean_abs) / (np.sum(axis_mean_abs) + 1e-8))

    # Per-axis features (Tier 1): mean, std, energy
    n_axes = acc.shape[1]
    axis_labels = ["X", "Y", "Z"]
    axis_means = np.mean(acc, axis=0)
    axis_energies = np.sum(acc ** 2, axis=0)
    for i in range(min(n_axes, 3)):
        lbl = axis_labels[i]
        feats[f"{prefix}_{lbl}_MEAN"] = float(axis_means[i])
        feats[f"{prefix}_{lbl}_STD"] = float(axis_std[i])
        feats[f"{prefix}_{lbl}_ENERGY"] = float(axis_energies[i])
    for i in range(n_axes, 3):
        lbl = axis_labels[i]
        feats[f"{prefix}_{lbl}_MEAN"] = 0.0
        feats[f"{prefix}_{lbl}_STD"] = 0.0
        feats[f"{prefix}_{lbl}_ENERGY"] = 0.0
    feats[f"{prefix}_AXIS_MEAN_SUM"] = float(np.sum(np.abs(axis_means)))

    # ACC orientation (Tier 2): tilt angle from gravity vector
    # When worn, one axis typically points down (gravity); off-wrist orientation is random.
    _grav = axis_means.copy()
    _grav_norm = float(np.sqrt(np.sum(_grav ** 2)))
    if _grav_norm > 1e-8:
        _grav_unit = _grav / _grav_norm
        # Tilt angle: angle between gravity vector and vertical (Z-up assumed)
        # cos(theta) = |gz| / |g|, theta=0 means device flat, theta=90 means vertical
        feats[f"{prefix}_TILT_ANGLE"] = float(np.degrees(np.arccos(np.clip(np.abs(_grav_unit[2]) if n_axes >= 3 else 0.0, 0.0, 1.0))))
        # Dominant axis index (0=X, 1=Y, 2=Z)
        feats[f"{prefix}_DOM_AXIS"] = float(np.argmax(np.abs(_grav_unit)))
        # Gravity concentration: how much of total ACC energy is in the gravity component
        feats[f"{prefix}_GRAVITY_RATIO"] = float(_grav_norm / (mag_mean + 1e-8))
    else:
        feats[f"{prefix}_TILT_ANGLE"] = 0.0
        feats[f"{prefix}_DOM_AXIS"] = 0.0
        feats[f"{prefix}_GRAVITY_RATIO"] = 0.0

    mag_centered = mag - np.mean(mag)
    try:
        mag_bp = bandpass_filter(mag_centered, fs, lowcut=0.5, highcut=5.0, order=2)
    except Exception:
        mag_bp = mag_centered
    bp_rms = float(np.sqrt(np.mean(mag_bp ** 2)))

    diff_mad = robust_mad(np.diff(mag)) if len(mag) > 1 else 0.0

    rel_std = mag_std / (abs(mag_mean) + 1e-6)
    still_score = float(1.0 / (1.0 + 50.0 * rel_std))

    feats[f"{prefix}_MAG_MEAN"] = mag_mean
    feats[f"{prefix}_YSUM"] = mag_mean
    feats[f"{prefix}_MAG_STD"] = mag_std
    feats[f"{prefix}_MAG_MAD"] = mag_mad
    feats[f"{prefix}_MAG_ENERGY"] = mag_energy
    feats[f"{prefix}_MAG_P2P"] = mag_p2p
    feats[f"{prefix}_AXIS_STD_SUM"] = axis_std_sum
    feats[f"{prefix}_GRAVITY_DOM_RATIO"] = dom_axis_ratio
    feats[f"{prefix}_BP_RMS"] = bp_rms
    feats[f"{prefix}_DIFF_MAD"] = diff_mad
    feats[f"{prefix}_STILL_SCORE"] = still_score
    feats[f"{prefix}_MAG_P50"] = float(np.percentile(mag, 50))
    feats[f"{prefix}_MAG_P90"] = float(np.percentile(mag, 90))

    return feats


def extract_acc_ppg_cross_features(acc_window, green_bp, ir_bp=None, fs=100.0):
    feats = OrderedDict()

    if acc_window is None or len(acc_window) < 4:
        feats["ACC_GREEN_BP_CORR"] = 0.0
        return feats

    mag = _acc_magnitude(acc_window)
    mag_centered = mag - np.mean(mag)

    try:
        mag_bp = bandpass_filter(mag_centered, fs, lowcut=0.5, highcut=5.0, order=2)
    except Exception:
        mag_bp = mag_centered

    n = min(len(mag_bp), len(green_bp))
    if n < 8:
        feats["ACC_GREEN_BP_CORR"] = 0.0
        return feats

    feats["ACC_GREEN_BP_CORR"] = abs(safe_corr(mag_bp[:n], green_bp[:n]))

    return feats


def extract_acc_green_coupling_features(acc_window, green_raw, green_bp):
    feats = OrderedDict()
    keys = [
        "ACC_TO_GTOP2_AC_RATIO",
        "ACC_STILL_X_GREEN_STABILITY",
        "ACC_DIFF_TO_GTOP2_DIFF_RATIO",
        "ACC_STILL_GREEN_MISMATCH",
    ]
    if acc_window is None or green_raw is None or green_bp is None or len(acc_window) < 4:
        for k in keys:
            feats[k] = 0.0
        return feats
    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    mag = _acc_magnitude(acc)
    green_raw = finite_signal(green_raw)
    green_bp = finite_signal(green_bp)
    n = min(len(mag), len(green_raw), len(green_bp))
    if n < 4:
        for k in keys:
            feats[k] = 0.0
        return feats
    mag = mag[:n]
    green_raw = green_raw[:n]
    green_bp = green_bp[:n]
    acc_diff_mad = robust_mad(np.diff(mag)) if len(mag) > 1 else 0.0
    green_diff_mad = robust_mad(np.diff(green_raw)) if len(green_raw) > 1 else 0.0
    green_ac = float(np.sqrt(np.mean(green_bp ** 2)))
    green_stability = 1.0 / (1.0 + segment_acdc_cv(green_raw))
    still_score = 1.0 / (1.0 + float(np.std(mag)) + acc_diff_mad)
    feats["ACC_TO_GTOP2_AC_RATIO"] = safe_div(acc_diff_mad, green_ac + EPS)
    feats["ACC_STILL_X_GREEN_STABILITY"] = float(still_score * green_stability)
    feats["ACC_DIFF_TO_GTOP2_DIFF_RATIO"] = safe_div(acc_diff_mad, green_diff_mad + EPS)
    feats["ACC_STILL_GREEN_MISMATCH"] = float(still_score * safe_div(green_ac, acc_diff_mad + EPS))
    return feats


# =========================================================
# Hjorth 参数特征
# =========================================================

def extract_hjorth_parameters(x, prefix=""):
    """
    计算 Hjorth 三参数: Activity, Mobility, Complexity

    Activity: 信号方差（功率）
    Mobility: 一阶导数标准差与信号标准差的比值
    Complexity: 二阶导数 mobility 与 一阶导数 mobility 的比值

    参数:
        x: 输入信号 (numpy array)
        prefix: 通道前缀 (如 "GREEN", "IRX", "AMBX")

    返回:
        OrderedDict: {'{prefix}_Hjorth_Activity': ..., ...}
    """
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""

    if len(x) < 4:
        feat[f"{pf}Hjorth_Activity"] = 0.0
        feat[f"{pf}Hjorth_Mobility"] = 0.0
        feat[f"{pf}Hjorth_Complexity"] = 0.0
        return feat

    # Activity = 方差
    activity = float(np.var(x))
    feat[f"{pf}Hjorth_Activity"] = activity

    # 一阶导数
    d1 = np.diff(x)
    if len(d1) < 2:
        feat[f"{pf}Hjorth_Mobility"] = 0.0
        feat[f"{pf}Hjorth_Complexity"] = 0.0
        return feat

    # Mobility = sqrt(var(d1) / var(x))
    var_d1 = np.var(d1)
    if activity > EPS:
        mobility = float(np.sqrt(var_d1 / activity))
    else:
        mobility = 0.0
    feat[f"{pf}Hjorth_Mobility"] = mobility
    
    # 二阶导数 -> Complexity
    d2 = np.diff(d1)
    if len(d2) < 2:
        feat[f"{pf}Hjorth_Complexity"] = 0.0
        return feat
    
    var_d2 = np.var(d2)
    if var_d1 > EPS:
        complexity = float(np.sqrt(var_d2 / var_d1))
    else:
        complexity = 0.0
    feat[f"{pf}Hjorth_Complexity"] = complexity

    return feat


# =========================================================
# 熵特征
# =========================================================

def extract_entropy_features(x, r_std_ratio=0.2, m=2, prefix="", skip_apen=True):
    """
    计算熵特征: Shannon熵, 近似熵(ApEn), 样本熵(SampEn)

    参数:
        x: 输入信号 (numpy array)
        r_std_ratio: ApEn/SampEn 的阈值参数（std的倍数），默认0.2
        m: 嵌入维度，默认2
        prefix: 通道前缀 (如 "GREEN", "IRX", "AMBX")

    返回:
        OrderedDict: {'{prefix}_Entropy_Shannon': ..., ...}
    """
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""

    if len(x) < 10:
        feat[f"{pf}Entropy_Shannon"] = 0.0
        feat[f"{pf}Entropy_ApEn"] = 0.0
        feat[f"{pf}Entropy_SampEn"] = 0.0
        return feat
    
    # ---------- Shannon 熵 ----------
    try:
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        shannon_entropy = float(-np.sum(hist * np.log(hist + EPS)))
    except Exception:
        shannon_entropy = 0.0
    feat[f"{pf}Entropy_Shannon"] = shannon_entropy
    # ---------- ApEn (O(N^2)) — skip_apen=True 跳过 ----------
    if not skip_apen:
        try:
            N = len(x)
            r = r_std_ratio * np.std(x)
            if r < EPS:
                apen = 0.0
            else:
                def phi(m_val):
                    patterns = np.array([x[i:i + m_val] for i in range(N - m_val)])
                    if len(patterns) == 0:
                        return 0.0
                    distances = np.max(np.abs(patterns[:, np.newaxis, :] - patterns[np.newaxis, :, :]), axis=2)
                    count = np.sum(distances <= r, axis=1)
                    count = count / (N - m_val)
                    count = count[count > 0]
                    if len(count) == 0:
                        return 0.0
                    return np.mean(np.log(count + EPS))
                phi_m = phi(m)
                phi_m1 = phi(m + 1)
                apen = float(phi_m - phi_m1)
                if not np.isfinite(apen):
                    apen = 0.0
        except Exception:
            apen = 0.0
        feat[f"{pf}Entropy_ApEn"] = apen

    
    # ---------- 样本熵 (SampEn) ----------
    # SampEn(m, r) = -ln( A / B )
    #   B = #{ pairs (i,j), i!=j, max|x[i:i+m]-x[j:j+m]| <= r }
    #   A = #{ pairs (i,j), i!=j, max|x[i:i+m+1]-x[j:j+m+1]| <= r }
    # 部署端 s07 _sample_entropy 与此实现保持一致。
    try:
        N = len(x)
        r = r_std_ratio * np.std(x)
        if r < EPS:
            sampen = 0.0
        else:
            def sampen_count(m_val):
                patterns = np.array([x[i:i + m_val] for i in range(N - m_val)])
                if len(patterns) < 2:
                    return 0.0
                distances = np.max(np.abs(patterns[:, np.newaxis, :] - patterns[np.newaxis, :, :]), axis=2)
                np.fill_diagonal(distances, np.inf)
                return float(np.sum(distances <= r))

            B_m = sampen_count(m)         # 长度 m 的匹配对数
            B_m1 = sampen_count(m + 1)    # 长度 m+1 的匹配对数

            if B_m > 0 and B_m1 > 0:
                sampen = float(-np.log(B_m1 / (B_m + EPS)))
            else:
                sampen = 0.0
            if not np.isfinite(sampen):
                sampen = 0.0
    except Exception:
        sampen = 0.0
    feat[f"{pf}Entropy_SampEn"] = sampen
    
    return feat


# =========================================================
# 导数特征
# =========================================================

def extract_derivative_features(x, fs=100.0, prefix="", compute_d2=False):
    """
    计算导数特征: 一阶/二阶导数统计, 零交叉率

    参数:
        x: 输入信号 (numpy array)
        fs: 采样率，默认100Hz
        prefix: 通道前缀 (如 "GREEN", "IRX", "AMBX")

    返回:
        OrderedDict: 包含一阶/二阶导数的均值、标准差、零交叉率等
    """
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""
    
    if len(x) < 4:
        feat[f"{pf}Deriv_d1_mean"] = 0.0
        feat[f"{pf}Deriv_d1_std"] = 0.0
        feat[f"{pf}Deriv_d1_max"] = 0.0
        feat[f"{pf}Deriv_d1_min"] = 0.0
        feat[f"{pf}Deriv_d1_zcr"] = 0.0
        feat[f"{pf}Deriv_d2_mean"] = 0.0
        feat[f"{pf}Deriv_d2_std"] = 0.0
        feat[f"{pf}Deriv_d2_max"] = 0.0
        feat[f"{pf}Deriv_d2_min"] = 0.0
        feat[f"{pf}Deriv_d2_zcr"] = 0.0
        return feat
    
    # 一阶导数
    d1 = np.diff(x)
    if len(d1) > 0:
        feat[f"{pf}Deriv_d1_mean"] = float(np.mean(d1))
        feat[f"{pf}Deriv_d1_std"] = float(np.std(d1))
        feat[f"{pf}Deriv_d1_max"] = float(np.max(d1))
        feat[f"{pf}Deriv_d1_min"] = float(np.min(d1))
        zcr_d1 = np.sum(np.abs(np.diff(np.sign(d1)))) / (2.0 * len(d1))
        feat[f"{pf}Deriv_d1_zcr"] = float(zcr_d1)
    else:
        feat[f"{pf}Deriv_d1_mean"] = 0.0
        feat[f"{pf}Deriv_d1_std"] = 0.0
        feat[f"{pf}Deriv_d1_max"] = 0.0
        feat[f"{pf}Deriv_d1_min"] = 0.0
        feat[f"{pf}Deriv_d1_zcr"] = 0.0
    
    # 二阶导数 (compute_d2=False 跳过)
    if compute_d2:
        d2 = np.diff(d1) if len(d1) > 1 else np.array([])
        if len(d2) > 0:
            feat[f"{pf}Deriv_d2_mean"] = float(np.mean(d2))
            feat[f"{pf}Deriv_d2_std"] = float(np.std(d2))
            feat[f"{pf}Deriv_d2_max"] = float(np.max(d2))
            feat[f"{pf}Deriv_d2_min"] = float(np.min(d2))
            zcr_d2 = np.sum(np.abs(np.diff(np.sign(d2)))) / (2.0 * len(d2))
            feat[f"{pf}Deriv_d2_zcr"] = float(zcr_d2)
        else:
            feat[f"{pf}Deriv_d2_mean"] = 0.0
            feat[f"{pf}Deriv_d2_std"] = 0.0
            feat[f"{pf}Deriv_d2_max"] = 0.0
            feat[f"{pf}Deriv_d2_min"] = 0.0
            feat[f"{pf}Deriv_d2_zcr"] = 0.0
    
    return feat


# =========================================================
# 时序动态特征
# =========================================================

def extract_temporal_dynamic_features(x, fs=100.0, prefix=""):
    """
    计算时序动态特征: 信号斜率, 峰值突出度
    
    参数:
        x: 输入信号 (numpy array)
        fs: 采样率，默认100Hz
    
    返回:
        OrderedDict: 包含斜率、峰值突出度等特征
    """
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""
    
    if len(x) < 4:
        feat[f"{pf}Temporal_slope_mean"] = 0.0
        feat[f"{pf}Temporal_slope_std"] = 0.0
        feat[f"{pf}Temporal_peak_prominence"] = 0.0
        feat[f"{pf}Temporal_peak_ratio"] = 0.0
        feat[f"{pf}Temporal_valley_ratio"] = 0.0
        return feat
    
    # ---------- 信号斜率 (基于线性回归) ----------
    t = np.arange(len(x))
    t_mean = np.mean(t)
    x_mean = np.mean(x)
    
    slope_numerator = np.sum((t - t_mean) * (x - x_mean))
    slope_denominator = np.sum((t - t_mean) ** 2)
    
    if slope_denominator > EPS:
        slope = slope_numerator / slope_denominator
    else:
        slope = 0.0
    
    fitted = x_mean + slope * (t - t_mean)
    residuals = x - fitted
    slope_std = float(np.std(residuals))
    
    feat[f"{pf}Temporal_slope_mean"] = float(slope)
    feat[f"{pf}Temporal_slope_std"] = slope_std
    
    # Keep this legacy helper numpy-only; the deployment-friendly pool no longer
    # calls temporal peak features, but standalone use should not require scipy.
    if len(x) >= 3:
        peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
        valleys = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
        feat[f"{pf}Temporal_peak_ratio"] = float(len(peaks) / len(x))
        feat[f"{pf}Temporal_valley_ratio"] = float(len(valleys) / len(x))
        if len(peaks) > 0:
            local_base = np.maximum(x[peaks - 1], x[peaks + 1])
            feat[f"{pf}Temporal_peak_prominence"] = float(np.mean(np.maximum(0.0, x[peaks] - local_base)))
        else:
            feat[f"{pf}Temporal_peak_prominence"] = 0.0
    else:
        feat[f"{pf}Temporal_peak_prominence"] = 0.0
        feat[f"{pf}Temporal_peak_ratio"] = 0.0
        feat[f"{pf}Temporal_valley_ratio"] = 0.0
    
    return feat


# =========================================================
# 统一窗口级特征提取函数（供训练和部署共用）
# =========================================================

def extract_window_features(ppg_window, fs=25.0, acc_window=None,
                            use_stage2_ir=DEFAULT_USE_STAGE2_IR):
    """
    统一的窗口级特征提取函数。
    训练（s03）和部署（s06）都调用此函数，保证一致性。

    参数:
        ppg_window: shape (N, C)，C 至少包含 6 通道
                    约定通道顺序: [IR, Ambient, G1, G2, G3, ...]
        fs: 采样率，默认100Hz
        acc_window: shape (M, 3) 加速度计数据，可选

    返回:
        OrderedDict 形式的特征字典
    """
    ppg = np.asarray(ppg_window, dtype=np.float64)
    if ppg.ndim == 1:
        ppg = ppg.reshape(-1, 1)

    if ppg.shape[1] < 6:
        raise ValueError(f"ppg_window 需要至少6通道，当前只有{ppg.shape[1]}通道")

    ir = apply_stage2_ir_policy(ppg[:, 0], use_stage2_ir=use_stage2_ir)
    ambient = ppg[:, 1] if ppg.shape[1] > 1 else np.zeros_like(ir)
    g1 = ppg[:, 2] if ppg.shape[1] > 2 else np.zeros_like(ir)
    g2 = ppg[:, 3] if ppg.shape[1] > 3 else g1
    g3 = ppg[:, 4] if ppg.shape[1] > 4 else g1

    # 用 return_preprocessed 复用 g1_bp / ir_bp，避免再调两次 preprocess_signal
    feat, preprocessed = extract_feature_pool_from_window(
        ir, ambient, g1, g2, g3, fs, return_preprocessed=True
    )

    feat.update(extract_acc_features(acc_window, fs=fs, prefix="ACC"))

    green_bp = preprocessed.get("g_top2_bp") if ppg.shape[1] > 2 else None
    green_raw = preprocessed.get("g_top2_raw") if ppg.shape[1] > 2 else None
    if green_bp is not None:
        feat.update(extract_acc_ppg_cross_features(acc_window, green_bp, fs=fs))
        _g_ac = float(np.sqrt(np.mean(green_bp ** 2)))
        feat["ACC_ENERGY_TO_GREEN_AC"] = safe_div(feat.get("ACC_MAG_ENERGY", 0.0), _g_ac)
    else:
        feat["ACC_ENERGY_TO_GREEN_AC"] = 0.0
    if green_raw is not None and green_bp is not None:
        feat.update(extract_acc_green_coupling_features(acc_window, green_raw, green_bp))

    # filter_deployment_friendly_stage2_features disabled: all features C-friendly
    return feat


def align_acc_window(acc, ppg_len, start_ppg, win_ppg, fs_ppg=100.0, fs_acc=None):
    """
    按时间比例对齐ACC窗口。
    ACC采样率默认与PPG相同（100Hz）。
    """
    if acc is None or len(acc) == 0:
        return None

    if fs_acc is None:
        fs_acc = fs_ppg

    acc_per_ppg = fs_acc / fs_ppg
    start_acc = int(start_ppg * acc_per_ppg)
    end_acc = start_acc + int(win_ppg * acc_per_ppg)

    if start_acc >= len(acc):
        return None
    return acc[start_acc:min(end_acc, len(acc))]


# =========================================================
# 单窗口特征提取主函数（保留原有接口，内部被extract_window_features调用）
# =========================================================

def extract_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=25, return_preprocessed=False):
    """
    输入：
        ir, ambient, g1, g2, g3: 5秒窗口信号 (默认已降采样至 25Hz)
        fs: 采样率 (默认 25)
        return_preprocessed: 是否返回预处理结果（供复用）

    输出：
        如果 return_preprocessed=False: OrderedDict 特征
        如果 return_preprocessed=True: (OrderedDict 特征, dict 预处理结果)
        
    优化：
        - 使用FFT缓存避免重复计算
        - 一次性计算所有需要的FFT结果
        - 可返回预处理结果供ACC交叉特征使用，避免重复计算
    """
    feat = OrderedDict()

    ir = np.asarray(ir, dtype=np.float64).reshape(-1)
    ambient = np.asarray(ambient, dtype=np.float64).reshape(-1)
    g1 = np.asarray(g1, dtype=np.float64).reshape(-1)
    g2 = np.asarray(g2, dtype=np.float64).reshape(-1)
    g3 = np.asarray(g3, dtype=np.float64).reshape(-1)

    n = min(len(ir), len(ambient), len(g1), len(g2), len(g3))

    if n < int(1.0 * fs):
        raise ValueError(f"窗口太短: n={n}, 需要 >= {int(fs)}")

    ir = ir[:n]
    ambient = ambient[:n]
    g1 = g1[:n]
    g2 = g2[:n]
    g3 = g3[:n]
    g1_input = g1.copy()
    g2_input = g2.copy()
    g3_input = g3.copy()
    g_mean_input = (g1 + g2 + g3) / 3.0

    # Short-window SQI is computed on the raw window before artifact removal.
    feat.update(short_window_sqi_features(ir, ambient, g_mean_input))

    # ---------- 鲁棒预处理 ----------
    ir_raw, ir_bp, ir_dc = preprocess_signal(ir, fs)
    amb_raw, amb_bp, amb_dc = preprocess_signal(ambient, fs)

    g1_raw, g1_bp, g1_dc = preprocess_signal(g1, fs)
    g2_raw, g2_bp, g2_dc = preprocess_signal(g2, fs)
    g3_raw, g3_bp, g3_dc = preprocess_signal(g3, fs)

    g_mean_raw = (g1_raw + g2_raw + g3_raw) / 3.0
    g_mean_bp = (g1_bp + g2_bp + g3_bp) / 3.0
    g_mean_dc = float(np.median(g_mean_raw))

    # =====================================================
    # ★ 优化：一次性计算所有FFT，避免重复
    # =====================================================
    fft_cache = {
        'green': compute_fft_cache(g_mean_bp, fs, fmin=0.5, fmax=5.0),
        'amb': compute_fft_cache(amb_bp, fs, fmin=0.5, fmax=5.0),
        'g1': compute_fft_cache(g1_bp, fs, fmin=0.5, fmax=5.0),
        'g2': compute_fft_cache(g2_bp, fs, fmin=0.5, fmax=5.0),
        'g3': compute_fft_cache(g3_bp, fs, fmin=0.5, fmax=5.0),
    }

    feat["GREEN_ROBUST_RANGE_RATIO"] = robust_range_ratio(g_mean_raw)
    feat["AMB_ROBUST_RANGE_RATIO"] = robust_range_ratio(amb_raw)
    feat["GREEN_SEG_ACDC_CV"] = segment_acdc_cv(g_mean_raw)
    feat["AMB_SEG_ACDC_CV"] = segment_acdc_cv(amb_raw)
    feat["GREEN_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(fft_cache['green'])
    feat["AMB_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(fft_cache['amb'])

    # =====================================================
    # 1. 基础长度
    # =====================================================
    feat["SIG_LEN"] = float(n)
    feat["SIG_SEC"] = float(n / fs)

    # =====================================================
    # 2. 与之前流程兼容的核心低维特征名
    # =====================================================

    # IR 类
    # G_mean 类
    feat["G_mean_mean"] = float(np.mean(g_mean_raw))
    feat["G_mean_std"] = float(np.std(g_mean_raw))
    feat["G_mean_diff_std"] = float(np.std(np.diff(g_mean_raw)))
    feat["G_mean_acdc"] = safe_div(np.sqrt(np.mean(g_mean_bp ** 2)), abs(g_mean_dc) + EPS)

    # IR-G 关系
    # Ambient
    feat["Ambient_mean"] = float(np.mean(amb_raw))
    feat["Ambient_std"] = float(np.std(amb_raw))
    feat["Ambient_p95"] = float(np.percentile(amb_raw, 95))
    feat["corr_Ambient_Gmean"] = safe_corr(amb_raw, g_mean_raw)

    # IR / Ambient 比值 — 皮肤接触指示器（佩戴时皮肤遮挡Ambient，比值变化显著）
    # =====================================================
    # 3. 单通道增强特征（使用FFT缓存）
    # =====================================================
    feat.update(extract_single_channel_features(
        g_mean_raw, g_mean_bp, g_mean_dc, fs, "GREEN", fft_cache=fft_cache['green']
    ))

    feat.update(extract_single_channel_features(
        amb_raw, amb_bp, amb_dc, fs, "AMBX", fft_cache=fft_cache['amb']
    ))

    # =====================================================
    # 3b. 波形形态: 偏度/峰度 — 真实 PPG 有特征性不对称（收缩峰尖锐、舒张缓慢）
    # =====================================================
    for _pf, _bp in [("GREEN", g_mean_bp)]:
        _m = float(np.mean(_bp))
        _s = float(np.std(_bp))
        if _s > EPS:
            feat[f"{_pf}_bp_skewness"] = float(np.mean((_bp - _m) ** 3) / (_s ** 3))
            feat[f"{_pf}_bp_kurtosis"] = float(np.mean((_bp - _m) ** 4) / (_s ** 4))
        else:
            feat[f"{_pf}_bp_skewness"] = 0.0
            feat[f"{_pf}_bp_kurtosis"] = 0.0

    # =====================================================
    # 4. 绿光三通道空间特征
    # =====================================================
    spatial_feat, g_imbalance, g_vmag = extract_green_spatial_features(
        g1_raw, g2_raw, g3_raw,
        g1_bp, g2_bp, g3_bp,
        g1_input=g1_input,
        g2_input=g2_input,
        g3_input=g3_input,
    )

    feat.update(spatial_feat)

    # =====================================================
    # 4a2. GREEN_TOP2 aggregation signal (Tier 3)
    #       Per-channel quality → select best 2 of 3 channels.
    #       TOP2 is robust to single-channel dropout (watch tilt):
    #       the worst channel is excluded, preserving signal quality.
    # =====================================================
    _ch_ac_rms = np.array([
        float(np.sqrt(np.mean(g1_bp ** 2))),
        float(np.sqrt(np.mean(g2_bp ** 2))),
        float(np.sqrt(np.mean(g3_bp ** 2))),
    ])
    _top2_idx = np.argsort(_ch_ac_rms)[-2:]  # indices of 2 highest AC channels
    _ch_bp = [g1_bp, g2_bp, g3_bp]
    _ch_raw = [g1_raw, g2_raw, g3_raw]
    g_top2_bp = np.mean([_ch_bp[i] for i in _top2_idx], axis=0)
    g_top2_raw = np.mean([_ch_raw[i] for i in _top2_idx], axis=0)
    feat["G_TOP2_CHANNEL_COUNT"] = float(len(_top2_idx))  # always 2, for consistency
    feat["G_TOP2_WORST_IDX"] = float(np.argmin(_ch_ac_rms))  # which channel was excluded

    # =====================================================
    # 4b. Per-channel G1/G2/G3 individual features (Tier 3)
    #     Extract the same lightweight features from each
    #     green channel independently. Combined with consensus
    #     statistics below, this captures directional tilt
    #     patterns without needing 3 separate models.
    # =====================================================
    _ch_data = [
        ("G1", g1_raw, g1_bp, g1_dc, fft_cache['g1']),
        ("G2", g2_raw, g2_bp, g2_dc, fft_cache['g2']),
        ("G3", g3_raw, g3_bp, g3_dc, fft_cache['g3']),
    ]
    for _pf, _raw, _bp, _dc, _fc in _ch_data:
        feat.update(extract_single_channel_features(
            _raw, _bp, _dc, fs, _pf, fft_cache=_fc
        ))

    # =====================================================
    # 4c. Cross-channel consensus statistics (Tier 3)
    #     For each base feature, compute min/max/range/cv
    #     across G1/G2/G3. The 120-degree symmetric LED layout
    #     means the *pattern* of channel deviations encodes
    #     tilt direction — e.g. G1 low + G2/G3 normal = tilted
    #     toward G1 direction (still on skin), while all 3 low
    #     = genuinely off-wrist.
    # =====================================================
    _consensus_base = [
        "DC_MEDIAN", "AC_RMS", "AC_MAD", "AC_DC_RATIO",
        "DERIV_MAD", "FFT_PEAK_MEDIAN_RATIO",
        "AUTO_CORR_PEAK",
    ]
    for _base in _consensus_base:
        _vals = []
        for _pf, _, _, _, _ in _ch_data:
            _v = feat.get(f"{_pf}_{_base}", 0.0)
            _vals.append(float(_v))
        _arr = np.array(_vals, dtype=np.float64)
        # _mean skipped: numerically ≈ GREEN_* / G_mean_* (already extracted)
        feat[f"G_consensus_{_base}_min"] = float(np.min(_arr))
        feat[f"G_consensus_{_base}_max"] = float(np.max(_arr))
        feat[f"G_consensus_{_base}_range"] = float(np.max(_arr) - np.min(_arr))
        _mean_abs = np.mean(np.abs(_arr))
        feat[f"G_consensus_{_base}_cv"] = float(np.std(_arr) / (_mean_abs + EPS))
        # top2_mean: mean of best 2 channels, robust to single-channel dropout
        _sorted = np.sort(_arr)
        feat[f"G_consensus_{_base}_top2_mean"] = float(np.mean(_sorted[-2:]))

    # =====================================================
    # 4d. Channel dropout indicators (Tier 3)
    #     Count how many channels appear "dark" and which
    #     direction the dropout vector points.
    # =====================================================
    _ch_ac = np.array([feat.get(f"{p}_AC_RMS", 0.0) for p in ["G1", "G2", "G3"]])
    _ch_ac_thr = 0.05 * np.max(_ch_ac) if np.max(_ch_ac) > EPS else EPS
    _dropout_mask = _ch_ac < _ch_ac_thr
    feat["G_DROPOUT_COUNT"] = float(np.sum(_dropout_mask))
    feat["G_MIN_CHANNEL_ID"] = float(np.argmin(_ch_ac))  # always report weakest channel
    # Dropout direction: angle of the vector from G1/G2/G3 AC values
    # in the 120-degree coordinate system (same transform as spatial vmag)
    _vx_drop = _ch_ac[0] - 0.5 * _ch_ac[1] - 0.5 * _ch_ac[2]
    _vy_drop = (np.sqrt(3.0) / 2.0) * (_ch_ac[1] - _ch_ac[2])
    feat["G_DROPOUT_ANGLE"] = float(np.degrees(np.arctan2(_vy_drop, _vx_drop + EPS)))

    # =====================================================
    # 5. 光学通道交叉特征（使用FFT缓存）
    # =====================================================
    feat.update(extract_cross_channel_features(
        g_mean_raw, g_mean_bp, g_mean_dc,
        ir_raw, ir_bp, ir_dc,
        amb_raw, amb_bp,
        fs,
        fft_cache_green=fft_cache['green'],
        fft_cache_ir=None
    ))


    # =====================================================
    # 5b. GREEN_CORR — bp 与自身平滑版本的相关性 (商用8特征之一)
    # =====================================================
    _ma_win_gc = max(2, int(round(0.15 * fs)))
    feat["GREEN_CORR"] = safe_corr(g_mean_bp,
                                   moving_average_filter(g_mean_bp, window_size=_ma_win_gc))
    feat["GREEN_AC"] = float(
        0.5 * np.sqrt(np.mean(g_mean_bp ** 2)) + 0.5 * robust_mad(g_mean_bp) * 1.4826
    )
    feat["AMB_AC"] = float(
        0.5 * np.sqrt(np.mean(amb_bp ** 2)) + 0.5 * robust_mad(amb_bp) * 1.4826
    )
    feat["GREEN_DC"] = float(np.median(g_mean_raw))
    feat["AMB_DC"] = float(np.median(ambient))
    _ac = normalized_autocorr(g_mean_bp)
    _lag_min = max(1, int(fs * 60.0 / 180.0))
    _lag_max = min(len(_ac) - 1, int(fs * 60.0 / 40.0))
    feat["GREEN_XCORR"] = float(np.max(_ac[_lag_min:_lag_max + 1])) if _lag_max > _lag_min else 0.0
    feat["FFT_PEAK_MEDIAN_RATIO"] = float(fft_cache["green"].get("peak_ratio", 0.0))

    # =====================================================
    # 6b. FFT 峰值宽度 + 绿光三通道相位一致性
    # =====================================================
    _fc = fft_cache['green']
    if _fc.get('band_spec') is not None and len(_fc['band_spec']) > 0:
        _bs = _fc['band_spec']
        _bf = _fc['band_freqs']
        _peak_val = np.max(_bs)
        _above_half = _bs > _peak_val * 0.5
        feat["GREEN_FFT_peak_width_Hz"] = float(_bf[_above_half][-1] - _bf[_above_half][0]) if np.any(_above_half) else 0.0
        # SNR: 带内功率 / 带外功率
        _in_band = float(np.sum(_bs ** 2))
        _out_band = float(np.sum(_fc['spec'] ** 2)) - _in_band
        feat["GREEN_FFT_SNR"] = float(_in_band / (_out_band + EPS))
    else:
        feat["GREEN_FFT_peak_width_Hz"] = 0.0
        feat["GREEN_FFT_SNR"] = 0.0

    # 谐波比: 2倍频功率 / 基频功率 (真实 PPG 有谐波结构，噪声没有)
    _fc = fft_cache['green']
    if _fc.get('band_spec') is not None and len(_fc['band_spec']) > 0:
        _bs = _fc['band_spec']
        _bf = _fc['band_freqs']
        _f0_idx = int(np.argmax(_bs))
        _f0 = _bf[_f0_idx]
        _f0_power = float(_bs[_f0_idx] ** 2)
        # 搜索 2*f0 附近 (±0.3Hz)
        _h2_mask = (_bf >= _f0 * 2 - 0.3) & (_bf <= _f0 * 2 + 0.3)
        if np.any(_h2_mask):
            _h2_power = float(np.max(_bs[_h2_mask] ** 2))
            feat["GREEN_FFT_harmonic_ratio"] = float(_h2_power / (_f0_power + EPS))
            feat["GREEN_FFT_harmonic_present"] = 1.0 if _h2_power > _f0_power * 0.1 else 0.0
        else:
            feat["GREEN_FFT_harmonic_ratio"] = 0.0
            feat["GREEN_FFT_harmonic_present"] = 0.0
    else:
        feat["GREEN_FFT_harmonic_ratio"] = 0.0
        feat["GREEN_FFT_harmonic_present"] = 0.0

    # 绿光三通道互相关峰值位置一致性（同一生理源 → lag 一致）
    _c12 = np.correlate(g1_bp - np.mean(g1_bp), g2_bp - np.mean(g2_bp), mode='same')
    _c23 = np.correlate(g2_bp - np.mean(g2_bp), g3_bp - np.mean(g3_bp), mode='same')
    _lag12 = int(np.argmax(np.abs(_c12)) - len(_c12) // 2) if len(_c12) > 0 else 0
    _lag23 = int(np.argmax(np.abs(_c23)) - len(_c23) // 2) if len(_c23) > 0 else 0
    feat["G_bp_lag_std"] = float(np.std([_lag12, _lag23]))

    # =====================================================
    # 7. 空间-光强耦合特征
    # =====================================================
    feat["corr_Gmean_G_imbalance"] = safe_corr(g_mean_raw, g_imbalance)
    feat["corr_Gmean_vmag"] = safe_corr(g_mean_raw, g_vmag)
    feat["corr_Ambient_vmag"] = safe_corr(amb_raw, g_vmag)

    # =====================================================
    # 7a2. 坏接触/离腕质量比值特征 (Tier 3)
    #      Ratio 型特征，跨设备/肤色鲁棒，区分：
    #      贴腕静止 vs 离腕环境光干扰 vs 松戴漏光
    # =====================================================
    _g_ac_rms = float(np.sqrt(np.mean(g_mean_bp ** 2)))
    _amb_ac_rms = float(np.sqrt(np.mean(amb_bp ** 2)))
    _g_dc = float(np.median(g_mean_raw))
    _amb_dc = float(np.median(amb_raw))
    _ch_dc_vals = np.array([float(np.median(g1_raw)), float(np.median(g2_raw)), float(np.median(g3_raw))])
    _ch_ac_vals = np.array([float(np.sqrt(np.mean(g1_bp ** 2))), float(np.sqrt(np.mean(g2_bp ** 2))), float(np.sqrt(np.mean(g3_bp ** 2)))])
    feat["AMB_AC_TO_GREEN_AC"] = safe_div(_amb_ac_rms, _g_ac_rms)
    feat["AMB_DC_TO_GREEN_DC"] = safe_div(_amb_dc, abs(_g_dc))
    feat["GCH_DC_RANGE_RATIO"] = safe_div(float(np.max(_ch_dc_vals) - np.min(_ch_dc_vals)), abs(float(np.mean(_ch_dc_vals))))
    feat["GCH_AC_RANGE_RATIO"] = safe_div(float(np.max(_ch_ac_vals) - np.min(_ch_ac_vals)), abs(float(np.mean(_ch_ac_vals))))

    # =====================================================
    # 7c. AMB spectral features (Tier 2)
    #     Ambient light spectrum helps distinguish skin-contact (light blocked)
    #     from off-wrist (ambient fluctuations visible).
    # =====================================================
    _fc_amb = fft_cache['amb']
    if _fc_amb.get('band_spec') is not None and len(_fc_amb['band_spec']) > 0:
        _bs = _fc_amb['band_spec']
        _bf = _fc_amb['band_freqs']
        feat["AMB_DOM_FREQ"] = float(_bf[int(np.argmax(_bs))])
        feat["AMB_FFT_PEAK_MEDIAN_RATIO"] = float(_fc_amb.get('peak_ratio', 0.0))
    else:
        feat["AMB_DOM_FREQ"] = 0.0
        feat["AMB_FFT_PEAK_MEDIAN_RATIO"] = 0.0

    # =====================================================
    # 7d. Per-channel skewness/kurtosis for AMB (Tier 2)
    #     GREEN and IRX already have these; AMB waveform shape
    #     indicates ambient light modulation patterns.
    # =====================================================
    _m_amb = float(np.mean(amb_bp))
    _s_amb = float(np.std(amb_bp))
    if _s_amb > EPS:
        feat["AMBX_bp_skewness"] = float(np.mean((amb_bp - _m_amb) ** 3) / (_s_amb ** 3))
        feat["AMBX_bp_kurtosis"] = float(np.mean((amb_bp - _m_amb) ** 4) / (_s_amb ** 4))
    else:
        feat["AMBX_bp_skewness"] = 0.0
        feat["AMBX_bp_kurtosis"] = 0.0

    # =====================================================
    # 7e. Signal quality indicators (Tier 2)
    #     Detect sensor saturation, clipping, and poor contact.
    # =====================================================
    for _ch_label, _ch_raw in [("GREEN", g_mean_raw)]:
        _sat_thr = 0.98 * np.max(_ch_raw)
        feat[f"{_ch_label}_SAT_FRAC"] = float(np.mean(_ch_raw >= _sat_thr))
        _d = np.diff(_ch_raw)
        _clip_eps = 1e-10
        feat[f"{_ch_label}_CLIP_RATE"] = float(np.mean(np.abs(_d) < _clip_eps))

    # GTOP2 基础特征：端侧只需要这一组 top-2 绿光统计和 1 个 FFT 来源。
    _gtop2_dc = float(np.median(g_top2_raw))
    _gtop2_cache = compute_fft_cache(g_top2_bp, fs, fmin=0.5, fmax=5.0)
    feat["GTOP2_ROBUST_RANGE_RATIO"] = robust_range_ratio(g_top2_raw)
    feat["GTOP2_SEG_ACDC_CV"] = segment_acdc_cv(g_top2_raw)
    feat["GTOP2_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(_gtop2_cache)
    feat.update(extract_single_channel_features(
        g_top2_raw, g_top2_bp, _gtop2_dc, fs, "GTOP2", fft_cache=_gtop2_cache
    ))
    feat.update(bp_shape_features(g_top2_bp, "GTOP2"))
    feat.update(segment_stability_features(g_top2_raw, "GTOP2"))
    feat["GREEN_AMB_LEAK_STABILITY"] = ambient_green_leak_stability(amb_raw, g_top2_raw)
    feat["GREEN_AMB_SEG_CORR_RANGE"] = segment_corr_range(amb_raw, g_top2_raw)

    # =====================================================
    # 12. 清理异常数值 + 移除已知冗余特征
    # =====================================================
    # 注意: 此处将 None / NaN / inf 统一替换为 0.0，这是 s03 层面的"兜底"处理。
    # 这意味着下游 s04/s05 从 CSV 读取特征时通常不会看到 NaN/inf（已被 0.0 替代）。
    # 0.0 不一定是该特征的合理默认值，但后续会被 s05 clip_outliers 裁剪到 IQR 下界，
    # 且 s05/s06 的 fill+clip 链会在此基础上做额外的防御性处理。
    # 训练与推理使用相同的 s03 代码，因此这一行为在两端一致，不构成训练-部署 gap。

    # feat = filter_deployment_friendly_stage2_features(feat)  # disabled: all features C-friendly

    # ---- Invalid feature count (before NaN→0.0 fill) ----
    # Stage2 starts after Stage1 IR gating. IR-derived keys must not be created
    # in the model-facing pool; failing here catches new leaks at the source.
    _stage2_ir_leaks = [k for k in feat.keys() if is_stage2_ir_feature(k)]
    if _stage2_ir_leaks:
        raise ValueError(
            "IR-derived Stage2 features generated in s03; remove them at source: "
            + ", ".join(_stage2_ir_leaks[:10])
        )

    # Count features that could not be computed. This lets XGBoost
    # distinguish "genuinely low value" from "computation failed".
    # Computed BEFORE the 0.0 fill below so we capture the true invalid count.
    _invalid_total = 0
    _invalid_ppg = 0
    _invalid_green = 0
    for k, v in feat.items():
        if v is None or not np.isfinite(v):
            _invalid_total += 1
            if any(k.startswith(p) for p in ["GREEN", "G_", "G1", "G2", "G3", "GTOP2", "AMBX", "AMB_", "Ambient", "corr_"]):
                if any(k.startswith(p) for p in ["GREEN", "G_", "G1", "G2", "G3", "GTOP2"]):
                    _invalid_green += 1
                _invalid_ppg += 1
    feat["TOTAL_INVALID_COUNT"] = float(_invalid_total)
    feat["PPG_INVALID_COUNT"] = float(_invalid_ppg)
    feat["GREEN_INVALID_COUNT"] = float(_invalid_green)

    for k in list(feat.keys()):
        if k in _REDUNDANT_FEATURES or not is_explainable_stage2_feature(k):
            del feat[k]
            continue
        v = feat[k]
        if v is None or not np.isfinite(v):
            feat[k] = 0.0
        else:
            feat[k] = float(v)

    # 如果需要返回预处理结果（供ACC交叉特征复用）
    if return_preprocessed:
        preprocessed = {
            'g1_bp': g1_bp, 'g2_bp': g2_bp, 'g3_bp': g3_bp,
            'g_top2_bp': g_top2_bp, 'g_top2_raw': g_top2_raw,
            'amb_bp': amb_bp,
            'g_mean_bp': g_mean_bp
        }
        return feat, preprocessed
    
    return feat

# =========================================================
# split 级别批量特征提取
# =========================================================

def _downsample_ppg(ppg, src_fs=100, tgt_fs=25):
    """将整个 ppg 从 src_fs 降采样到 tgt_fs。分批处理以控制内存。"""
    if src_fs == tgt_fs:
        return ppg
    gcd = np.gcd(src_fs, tgt_fs)
    up = tgt_fs // gcd
    down = src_fs // gcd
    ppg = ppg.astype(np.float32, copy=False)  # float32 省一半内存
    n_cols = ppg.shape[1]
    batch_size = 8  # 每次处理 8 个通道，控制峰值内存
    out_parts = []
    for c in range(0, n_cols, batch_size):
        batch = ppg[:, c:c + batch_size]
        part = resample_poly(batch, up, down, axis=0)
        out_parts.append(part.astype(np.float64))
    return np.concatenate(out_parts, axis=1)


def _extract_rows_for_sample(sample, dc_threshold, ac_dc_threshold,
                              window_len, stride_len, fs,
                              target_aware_stride, stride_neg, stride_pos,
                              skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
                              use_stage2_ir=DEFAULT_USE_STAGE2_IR):
    """
    单样本抽窗特征。返回 rows list（失败时返回 []）。

    3D 预切窗样本直接使用已有窗口；连续时序样本在 Stage1 后降采样
    ppg 到 25Hz 再滑窗，大幅降低计算量。
    fs 参数为降采样后目标采样率 (25Hz)。
    """
    FEATURE_FS = 25  # 特征提取统一采样率

    try:
        ppg = load_ppg(sample)
        acc = load_acc(sample)
    except Exception as e:
        print(f"读取失败 {sample.get('sample_name')}: {e}")
        return []

    if is_prewindowed_signal(ppg):
        window_meta = load_grouped_window_metadata(sample)
        window_indices = window_meta.get("window_indices") if window_meta else None
        window_labels = window_meta.get("window_labels") if window_meta else None
        native_25hz = _is_25hz_sample(sample) or int(ppg.shape[1]) == int(round((window_len / max(fs, 1)) * FEATURE_FS))
        ppg_src_fs = 25 if native_25hz else 100
        mode = detect_green_mode(ppg)
        rows = []
        first_idx = max(0, int(skip_initial_windows))
        for win_idx in range(first_idx, ppg.shape[0]):
            window_number = int(window_indices[win_idx]) if window_indices and win_idx < len(window_indices) else int(win_idx)
            window_target = int(window_labels[win_idx]) if window_labels and win_idx < len(window_labels) else int(sample["target"])
            raw_window = ppg[win_idx]
            if raw_window.shape[0] < 2:
                continue
            if not stage1_sample_pass(raw_window, dc_threshold, ac_dc_threshold, ppg_fs=ppg_src_fs):
                continue
            if native_25hz:
                window = raw_window.astype(np.float64, copy=False)
            else:
                window = _downsample_ppg(raw_window, src_fs=100, tgt_fs=FEATURE_FS)
            acc_seg = None
            if acc is not None:
                try:
                    if is_prewindowed_signal(acc) and win_idx < acc.shape[0]:
                        raw_acc = acc[win_idx]
                        acc_seg = raw_acc.astype(np.float64, copy=False) if native_25hz else resample_poly(
                            raw_acc.astype(np.float64), FEATURE_FS, 100, axis=0
                        )
                    elif not is_prewindowed_signal(acc) and len(acc) > 0:
                        raw_start = int(win_idx * stride_len)
                        raw_acc = acc[raw_start:raw_start + window_len]
                        if len(raw_acc) > 0:
                            acc_seg = raw_acc.astype(np.float64, copy=False) if native_25hz else resample_poly(
                                raw_acc.astype(np.float64), FEATURE_FS, 100, axis=0
                            )
                except Exception:
                    acc_seg = None
            try:
                ir, ambient, g1, g2, g3 = get_channels_from_window(window, mode)
                ir = apply_stage2_ir_policy(ir, use_stage2_ir=use_stage2_ir)
                feat, preprocessed = extract_feature_pool_from_window(
                    ir=ir, ambient=ambient, g1=g1, g2=g2, g3=g3,
                    fs=FEATURE_FS, return_preprocessed=True
                )
                if acc_seg is not None and len(acc_seg) > 0:
                    feat.update(extract_acc_features(acc_seg, fs=FEATURE_FS, prefix="ACC"))
                    green_bp = preprocessed.get("g_top2_bp")
                    ir_bp = preprocessed.get("ir_bp")
                    if green_bp is not None and ir_bp is not None:
                        feat.update(extract_acc_ppg_cross_features(acc_seg, green_bp, ir_bp, fs=FEATURE_FS))
                    if green_bp is not None:
                        _g_ac = float(np.sqrt(np.mean(green_bp ** 2)))
                        _acc_energy = float(feat.get("ACC_MAG_ENERGY", 0.0))
                        feat["ACC_ENERGY_TO_GREEN_AC"] = safe_div(_acc_energy, _g_ac)
                        feat.update(extract_acc_green_coupling_features(
                            acc_seg,
                            preprocessed.get("g_top2_raw"),
                            green_bp,
                        ))
                    else:
                        feat["ACC_ENERGY_TO_GREEN_AC"] = 0.0
                feat.update(compute_ambient_stage1_features(raw_window))
                # feat = filter_deployment_friendly_stage2_features(feat)  # disabled: all features C-friendly
                feat["sample_name"] = sample["sample_name"]
                feat["h5_file"] = sample["h5_file"]
                feat["target"] = int(window_target)
                feat["start_100hz"] = int(window_number * stride_len)
                feat["start_sec"] = float(window_number * stride_len / max(fs, 1))
                feat["window_index"] = int(window_number)
                feat["mode"] = int(mode)
                rows.append(feat)
            except Exception as e:
                print(f"特征提取失败: sample={sample.get('sample_name')}, "
                      f"window_idx={win_idx}, error={e}")
                continue
        return rows

    if len(ppg) < window_len:
        return []

    # 检测是否 25Hz 原生数据
    native_25hz = _is_25hz_sample(sample)
    ppg_src_fs = 25 if native_25hz else 100

    if not stage1_sample_pass(ppg, dc_threshold, ac_dc_threshold, ppg_fs=ppg_src_fs):
        return []
    mode = detect_green_mode(ppg)
    sample_target = int(sample.get("target", 0))

    # 降采样至 25Hz（原生 25Hz 直接使用）
    if native_25hz:
        ppg_25 = ppg
        acc_25 = acc if (acc is not None and len(acc) > 0) else None
    else:
        ppg_25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS)
        if acc is not None and len(acc) > 0:
            acc_25 = np.zeros((0, 3), dtype=np.float64)
            try:
                acc_25 = resample_poly(acc.astype(np.float64), FEATURE_FS, 100, axis=0)
            except Exception:
                acc_25 = None
        else:
            acc_25 = None

    # 25Hz 下的窗长和步长（从调用参数推导，而非硬编码）
    window_sec_actual = window_len / max(fs, 1)
    stride_sec_actual = stride_len / max(fs, 1)
    win_25 = int(window_sec_actual * FEATURE_FS)
    stride_25 = int(stride_sec_actual * FEATURE_FS)
    if target_aware_stride:
        # stride_neg/pos 是从 fs=100 计算出来的原始值，需要换算到 25Hz
        stride_neg_25 = int(stride_neg * FEATURE_FS / max(fs, 1))
        stride_pos_25 = int(stride_pos * FEATURE_FS / max(fs, 1))
        stride_25 = stride_neg_25 if sample_target == 0 else stride_pos_25

    if len(ppg_25) < win_25:
        return []

    rows = []
    first_start = max(0, int(skip_initial_windows)) * stride_25
    for start in range(first_start, len(ppg_25) - win_25 + 1, stride_25):
        window = ppg_25[start:start + win_25, :]
        try:
            ir, ambient, g1, g2, g3 = get_channels_from_window(window, mode)
            ir = apply_stage2_ir_policy(ir, use_stage2_ir=use_stage2_ir)
            feat, preprocessed = extract_feature_pool_from_window(
                ir=ir, ambient=ambient, g1=g1, g2=g2, g3=g3,
                fs=FEATURE_FS, return_preprocessed=True
            )
            if acc_25 is not None and len(acc_25) > 0:
                acc_seg = align_acc_window(acc_25, len(ppg_25), start, win_25,
                                           fs_ppg=FEATURE_FS, fs_acc=FEATURE_FS)
                feat.update(extract_acc_features(acc_seg, fs=FEATURE_FS, prefix="ACC"))
                green_bp = preprocessed.get("g_top2_bp")
                ir_bp = preprocessed.get("ir_bp")
                if green_bp is not None and ir_bp is not None:
                    feat.update(extract_acc_ppg_cross_features(
                        acc_seg, green_bp, ir_bp, fs=FEATURE_FS
                    ))
                if green_bp is not None:
                    _g_ac = float(np.sqrt(np.mean(green_bp ** 2)))
                    _acc_energy = float(feat.get("ACC_MAG_ENERGY", 0.0))
                    feat["ACC_ENERGY_TO_GREEN_AC"] = safe_div(_acc_energy, _g_ac)
                    feat.update(extract_acc_green_coupling_features(
                        acc_seg,
                        preprocessed.get("g_top2_raw"),
                        green_bp,
                    ))
                else:
                    feat["ACC_ENERGY_TO_GREEN_AC"] = 0.0

            feat.update(compute_ambient_stage1_features(window))
            # feat = filter_deployment_friendly_stage2_features(feat)  # disabled: all features C-friendly
            feat["sample_name"] = sample["sample_name"]
            feat["h5_file"] = sample["h5_file"]
            feat["target"] = int(sample["target"])
            feat["start_100hz"] = int(start * (fs / FEATURE_FS))  # 映射回原始 fs 坐标
            feat["start_sec"] = float(start / FEATURE_FS)
            feat["mode"] = int(mode)
            rows.append(feat)
        except Exception as e:
            print(f"特征提取失败: sample={sample.get('sample_name')}, "
                  f"start={start}, error={e}")
            continue
    return rows


def _worker_extract(args_tuple):
    """子进程入口。"""
    (sample, dc_threshold, ac_dc_threshold, window_len, stride_len, fs,
     target_aware_stride, stride_neg, stride_pos, skip_initial_windows,
     use_stage2_ir) = args_tuple
    return _extract_rows_for_sample(
        sample, dc_threshold, ac_dc_threshold, window_len, stride_len, fs,
        target_aware_stride, stride_neg, stride_pos, skip_initial_windows,
        use_stage2_ir=use_stage2_ir,
    )


def extract_features_for_split(samples,
                               dc_threshold,
                               ac_dc_threshold,
                               window_sec=5,
                               stride_sec=1,
                               fs=100,
                               target_aware_stride=False,
                               target_ratio=5.0,
                               skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
                               use_stage2_ir=DEFAULT_USE_STAGE2_IR,
                               n_workers=None):
    """
    提取特征池（样本级并行）。

    参数:
        samples: 样本列表
        dc_threshold: Stage1 DC阈值
        ac_dc_threshold: Stage1 AC/DC阈值
        window_sec: 窗口秒数
        stride_sec: 默认步长秒数
        fs: 采样率
        target_aware_stride: 是否启用target感知stride
        target_ratio: 目标正负样本比例 (neg/pos)
        n_workers: 并行 worker 数；None=自动(cpu_count-1)，1=单进程
    """
    window_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)
    stride_neg = int(1 * fs)
    stride_pos = int(3 * fs)

    n_workers = resolve_n_workers(n_workers, n_items=len(samples))

    args_list = [
        (s, dc_threshold, ac_dc_threshold, window_len, stride_len, fs,
         target_aware_stride, stride_neg, stride_pos, skip_initial_windows,
         use_stage2_ir)
        for s in samples
    ]

    all_rows = []
    if n_workers == 1:
        for i, a in enumerate(args_list, 1):
            all_rows.extend(_worker_extract(a))
            if len(args_list) >= 10 and (i % max(1, len(args_list) // 10) == 0 or i == len(args_list)):
                print(f"  s03 progress: {i}/{len(args_list)} samples", flush=True)
    else:
        pool_kwargs = {"max_workers": n_workers}
        mp_ctx = multiprocessing_context_from_env()
        if mp_ctx is not None:
            pool_kwargs["mp_context"] = mp_ctx
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {ex.submit(_worker_extract, a): i for i, a in enumerate(args_list)}
            total = len(futures)
            print(f"  s03 parallel extraction: {total} samples, workers={n_workers}", flush=True)
            for done_count, fut in enumerate(as_completed(futures), 1):
                sample_idx = futures[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    sample_name = samples[sample_idx].get("sample_name", f"idx={sample_idx}")
                    print(f"  [WARN] s03 worker failed sample={sample_name}: {e}", flush=True)
                    rows = []
                all_rows.extend(rows)
                if done_count % max(1, total // 10) == 0 or done_count == total:
                    print(f"  s03 progress: {done_count}/{total} samples", flush=True)

    return pd.DataFrame(all_rows)

# =========================================================
# main
# =========================================================
def _stage1_threshold_pair(payload):
    payload = payload or {}
    return (
        float(payload.get("dc_threshold", DEFAULT_STAGE1_DC_THRESHOLD)),
        float(payload.get("ac_dc_threshold", DEFAULT_STAGE1_AC_DC_THRESHOLD)),
    )


def resolve_stage1_thresholds(th):
    """
    兼容新旧 stage1_threshold.json。

    新版：
        th["deploy_stage1_threshold"]
        th["train_stage1_threshold"]

    旧版：
        th["dc_threshold"]
        th["ac_dc_threshold"]
    """
    th = th or {}
    if "deploy_stage1_threshold" in th:
        deploy_dc, deploy_acdc = _stage1_threshold_pair(th["deploy_stage1_threshold"])
    else:
        deploy_dc, deploy_acdc = _stage1_threshold_pair(th)

    if "train_stage1_threshold" in th:
        train_dc, train_acdc = _stage1_threshold_pair(th["train_stage1_threshold"])
    else:
        # 旧版没有宽松阈值时，退化为 deploy 阈值
        train_dc = deploy_dc
        train_acdc = deploy_acdc

    return {
        "deploy": {
            "dc_threshold": float(deploy_dc),
            "ac_dc_threshold": float(deploy_acdc),
        },
        "train": {
            "dc_threshold": float(train_dc),
            "ac_dc_threshold": float(train_acdc),
        }
    }


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--window_sec", type=int, default=5, choices=[3, 5],
                        help="Stage2 窗口秒数：3s (75点@25Hz) 或 5s (125点@25Hz)")
    parser.add_argument("--stride_sec", type=int, default=1)
    parser.add_argument("--skip_initial_windows", type=int, default=DEFAULT_SKIP_INITIAL_WINDOWS,
                        help="drop this many leading Stage2 windows per sample")
    parser.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_USE_STAGE2_IR,
                        help="legacy compatibility flag; Stage2 model features are always ambient/green/ACC only")
    
    # target 感知 stride 参数
    # 注意：默认 False。train/deploy 都用统一 1s stride 才能保证窗口分布一致。
    # 启用此参数会让 target=0 用 1s、target=1 用 3s，制造 neg:pos 5:1 的窗口比，
    # 配合 s05 旧版 scale_pos_weight=neg/pos 会双倍偏置模型预测正类、抬高 FP，
    # 与"FP 代价高"的产品目标冲突。
    parser.add_argument("--target_aware_stride", action="store_true",
                        default=False,
                        help="[不推荐] 启用 target 感知 stride（pos=3s, neg=1s）。"
                             "与部署 1s stride 分布不一致，且与 scale_pos_weight 叠加双倍偏置。"
                             "保留仅供对照。")
    parser.add_argument("--target_ratio", type=float, default=5.0,
                        help="target 感知 stride 启用时的目标 neg/pos 比例")

    parser.add_argument(
        "--test_gate",
        type=str,
        default="deploy",
        choices=["deploy", "train"],
        help="test 特征池提取使用哪个 Stage1 门控。防过拟合推荐 deploy。"
    )

    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")

    if args is None:
        args = parser.parse_args()

    split_path = os.path.join(args.artifact_dir, "splits.json")
    th_path = os.path.join(args.artifact_dir, "stage1_threshold.json")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    with open(th_path, "r", encoding="utf-8") as f:
        th = json.load(f)

    thresholds = resolve_stage1_thresholds(th)

    print("=" * 80)
    print("Stage1 thresholds")
    print("=" * 80)
    print("deploy threshold:")
    print(f"  dc_threshold    = {thresholds['deploy']['dc_threshold']}")
    print(f"  acdc_threshold  = {thresholds['deploy']['ac_dc_threshold']}")
    print("train/feature threshold:")
    print(f"  dc_threshold    = {thresholds['train']['dc_threshold']}")
    print(f"  acdc_threshold  = {thresholds['train']['ac_dc_threshold']}")

    for part in ["train", "valid", "test"]:
        print("=" * 80)
        print(f"提取 {part} 特征")
        print("=" * 80)

        if part in ["train", "valid"]:
            gate_name = "train"
        else:
            gate_name = args.test_gate

        dc_threshold = thresholds[gate_name]["dc_threshold"]
        ac_dc_threshold = thresholds[gate_name]["ac_dc_threshold"]

        print(f"{part} 使用 Stage1 gate: {gate_name}")
        print(f"  dc_threshold    = {dc_threshold}")
        print(f"  ac_dc_threshold  = {ac_dc_threshold}")
        
        if args.target_aware_stride:
            print("  ⚠ target_aware_stride: 启用 (不推荐)")
            print("    target=0 stride=1s, target=1 stride=3s -> 制造窗口级不平衡")
            print("    若 s05 同时启用 scale_pos_weight=neg/pos 会双倍偏置正类，"
                  "与 FP 高代价目标冲突。")
        else:
            print(f"  target_aware_stride: 禁用 (统一 stride={args.stride_sec}s，"
                  f"与部署 1s stride 分布一致)")

        df = extract_features_for_split(
            samples=split[part],
            dc_threshold=dc_threshold,
            ac_dc_threshold=ac_dc_threshold,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            fs=100,
            target_aware_stride=args.target_aware_stride,
            target_ratio=args.target_ratio,
            skip_initial_windows=args.skip_initial_windows,
            use_stage2_ir=args.use_stage2_ir,
            n_workers=args.n_workers,
        )

        out_path = os.path.join(args.artifact_dir, f"feature_pool_{part}.csv")
        df.to_csv(out_path, index=False)

        print(f"{part} 特征提取完成: {len(df)} windows")
        print(f"保存到: {out_path}")

        if len(df) > 0 and "target" in df.columns:
            print(f"  target=0: {np.sum(df['target'].values == 0)}")
            print(f"  target=1: {np.sum(df['target'].values == 1)}")
            meta_cols = ["sample_name", "h5_file", "target", "start_100hz", "start_sec"]
            print(f"  特征列数: {len([c for c in df.columns if c not in meta_cols])}")

if __name__ == "__main__":
    main()
