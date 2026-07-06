# s04_feature_selection.py
# -*- coding: utf-8 -*-

"""
步骤4：稳定性特征筛选，防过拟合版（优化）

主要改进：
1. stability_selection 把 (seed × fold) 笛卡尔积扁平化后并行训练 + permutation importance
2. permutation_importance 内部线程数由 WL_INNER_N_JOBS 控制，默认 1
3. fit 失败时 log warning，避免 silent fallback 掩盖问题
4. 合并 select_by_group / select_by_group_from_combined

公共接口 / CLI / 输出 schema 完全不变。
"""

import os
import sys
import json
import argparse
import pickle
import logging
import time
import re
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from s02_features import (
    DEPLOYMENT_ALLOWED_FFT_FEATURES as S03_DEPLOYMENT_ALLOWED_FFT_FEATURES,
    DEPLOYMENT_ALLOWED_NON_FFT_FEATURES as S03_DEPLOYMENT_ALLOWED_NON_FFT_FEATURES,
    filter_deployment_friendly_stage2_features as s03_filter_deployment_friendly_stage2_features,
    filter_stage2_ir_features,
    is_deployment_friendly_stage2_feature as s03_is_deployment_friendly_stage2_feature,
)

logger = logging.getLogger(__name__)


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
    if n_items is not None and int(n_items) <= 4:
        return 1
    return resolved


def get_inner_n_jobs(default=1):
    """Thread count for XGBoost/permutation jobs inside each worker."""
    try:
        return max(1, int(os.environ.get("WL_INNER_N_JOBS", default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def multiprocessing_context_from_env():
    method = os.environ.get("WL_MP_START_METHOD", "").strip()
    if not method:
        return None
    import multiprocessing as mp
    return mp.get_context(method)


def _risk_level(score):
    if score >= 3:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def estimate_s04_workload(n_train_rows, n_valid_rows, n_features, n_samples,
                          n_workers=1, skip_vif=False, shap_available=False,
                          run_subset_search=False):
    """Return a coarse, ordered bottleneck estimate for s04 feature selection."""
    n_train_rows = int(n_train_rows or 0)
    n_valid_rows = int(n_valid_rows or 0)
    n_features = int(n_features or 0)
    n_samples = int(n_samples or 0)
    n_workers = max(1, int(n_workers or 1))
    n_splits = min(5, n_samples) if n_samples > 0 else 0
    n_folds = 3 * n_splits

    estimates = []

    if skip_vif:
        vif_score = 1
        vif_note = "skip_vif=True; VIF is bypassed, so STEP 1 usually only does missing/variance/correlation filtering."
    else:
        vif_score = 1 + int(n_features >= 40) + int(n_features >= 80) + int(n_train_rows >= 10000)
        vif_note = (
            "Runs before STEP 1 completes; if logs stop before [s04] STEP 1/5 completes, "
            "VIF/correlation cleaning is the likely bottleneck."
        )
    estimates.append({
        "step": "STEP 1 clean/VIF",
        "risk": _risk_level(vif_score),
        "work_items": int(n_features),
        "note": vif_note,
    })

    stability_score = 1 + int(n_folds >= 12) + int(n_features >= 60) + int(n_train_rows >= 10000)
    estimates.append({
        "step": "STEP 3 stability/permutation",
        "risk": _risk_level(stability_score),
        "work_items": int(n_folds),
        "note": (
            f"Runs about {n_folds} GroupKFold jobs; first progress line appears only after a fold returns "
            f"when multiprocessing is enabled ({n_workers} workers)."
        ),
    })

    shap_score = 0 if not shap_available else 1 + int(n_train_rows >= 10000) + int(n_features >= 60)
    estimates.append({
        "step": "STEP 4 SHAP",
        "risk": _risk_level(shap_score),
        "work_items": int(n_train_rows * max(n_features, 1)) if shap_available else 0,
        "note": "SHAP is unavailable and will be skipped." if not shap_available else
                "TreeExplainer runs on train rows and selected feature candidates; this can be slow on large pools.",
    })

    diag_score = 1 + int(n_features >= 80) + int((n_train_rows + n_valid_rows) >= 20000)
    estimates.append({
        "step": "post-selection diagnostics",
        "risk": _risk_level(diag_score),
        "work_items": int(n_features),
        "note": "Exports per-feature drift/AUC/FP-proxy diagnostics after final ranking.",
    })

    if run_subset_search:
        estimates.append({
            "step": "EXTRA subset search",
            "risk": "medium",
            "work_items": 6,
            "note": "Only runs with --run_subset_search; trains fixed XGBoost models for candidate feature subsets.",
        })

    return estimates


def print_s04_workload_estimate(df_train, df_valid, feature_cols, args):
    n_samples = df_train["sample_name"].nunique() if "sample_name" in df_train.columns else 0
    estimates = estimate_s04_workload(
        n_train_rows=len(df_train),
        n_valid_rows=len(df_valid),
        n_features=len(feature_cols),
        n_samples=n_samples,
        n_workers=args.n_workers,
        skip_vif=args.skip_vif,
        shap_available=SHAP_AVAILABLE,
        run_subset_search=args.run_subset_search,
    )
    print("\n[s04] workload estimate")
    print(f"  train_rows={len(df_train)}, valid_rows={len(df_valid)}, "
          f"features={len(feature_cols)}, train_samples={n_samples}, workers={args.n_workers}")
    for item in estimates:
        print(f"  - {item['step']}: risk={item['risk']}, work_items={item['work_items']}; {item['note']}")
    sys.stdout.flush()


def _elapsed(start):
    return time.perf_counter() - start

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed, skipping SHAP importance")


META_COLS = [
    "sample_name",
    "h5_file",
    "target",
    "start_100hz",
    "start_sec",
    "window_index",
]


FEATURE_GROUPS = {
    "commercial_baseline": [
        "GREEN_CORR",
        "GREEN_AC",
        "AMB_AC",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
    ],
    # -- 3s/75点短窗信号质量与鲁棒性 --
    "signal_quality": [
        "SQI_FLAT_RATIO", "SQI_SPIKE_RATIO",
        "GREEN_ROBUST_RANGE_RATIO", "AMB_ROBUST_RANGE_RATIO",
    ],
    "short_window_stability": [
        "GREEN_SEG_ACDC_CV", "AMB_SEG_ACDC_CV",
        "GTOP2_HALF_ACDC_DELTA", "GTOP2_SEG_ACDC_RANGE",
        "GREEN_AMB_LEAK_STABILITY",
    ],
    "short_window_frequency": [
        "GREEN_BAND_ENERGY_RATIO", "AMB_BAND_ENERGY_RATIO",
    ],
    # -- Green 单通道: 原始(4) + 鲁棒 DC/AC(6) = 10 - limit 2 --
    "green_stats": [
        "G_mean_mean", "G_mean_std", "G_mean_diff_std", "G_mean_acdc",
        "GREEN_DC_MEDIAN", "GREEN_DC_IQR", "GREEN_AC_RMS", "GREEN_AC_MAD",
        "GREEN_AC_DC_RATIO", "GREEN_DERIV_MAD",
    ],
    # -- Ambient 统计(11) - limit 1 --
    "ambient_stats": [
        "Ambient_mean", "Ambient_std", "Ambient_p95",
        "corr_Ambient_Gmean",
        "AMBX_DC_MEDIAN", "AMBX_DC_IQR", "AMBX_AC_RMS", "AMBX_AC_MAD",
        "AMBX_AC_DC_RATIO", "AMBX_DERIV_MAD",
    ],
    # -- 绿光三通道空间(13) - limit 2 --
    "green_spatial": [
        "G_imbalance_mean", "G_imbalance_p90", "G_imbalance_iqr",
        "G_rangeNorm_mean", "G_rangeNorm_p90",
        "G_spatial_vmag_mean", "G_spatial_vmag_p90",
        "G_spatial_vmag_iqr", "G_spatial_vmag_std",
        "G_ch_dc_cv", "G_ch_dc_max_min_ratio",
        "GCH_DC_RANGE_RATIO", "GCH_AC_RANGE_RATIO",
        "G_WEAK_CHANNEL_GAP", "G_SPATIAL_STABILITY_SCORE",
        "G_TOP1_TO_TOP2_AC_RATIO", "G_SPATIAL_VMAG_RANGE",
    ],
    # -- 绿光三通道一致性(3) - limit 1 --
    "green_3ch_consistency": [
        "G_bp_corr_mean", "G_bp_corr_min", "G_bp_corr_std",
        "G_bp_lag_std",
        "G_2OF3_AC_SUPPORT", "G_TOP2_TO_ALL_AC_RATIO", "G_TOP2_CORR_MIN",
        "G_TOP2_RANK_STABILITY", "G_TOP2_SWITCH_RATE",
    ],
    # -- Ambient 交叉泄露(5) - limit 1 --
    "amb_cross": [
        "GREEN_AMB_BP_CORR",
        "GREEN_AMB_ENV_CORR",
        "GREEN_AMB_LEAK",
        "GREEN_AMB_SEG_CORR_RANGE",
        "AMB_AC_TO_GREEN_AC", "AMB_DC_TO_GREEN_DC",
    ],
    # -- 周期性/频域: FFT + Autocorr + 谐波 (limit 2) --
    "frequency": [
        "GTOP2_BAND_ENERGY_RATIO", "GTOP2_FFT_PEAK_MEDIAN_RATIO", "GTOP2_DOM_FREQ",
        "GREEN_FFT_PEAK_MEDIAN_RATIO", "GREEN_DOM_FREQ",
        "GREEN_AUTO_CORR_PEAK", "GREEN_AUTO_CORR_LAG_SEC",
        "AMBX_FFT_PEAK_MEDIAN_RATIO", "AMBX_DOM_FREQ",
        "AMBX_AUTO_CORR_PEAK", "AMBX_AUTO_CORR_LAG_SEC",
        "GREEN_FFT_peak_width_Hz", "GREEN_FFT_SNR",
        "GREEN_FFT_harmonic_ratio", "GREEN_FFT_harmonic_present",
        "AMB_DOM_FREQ", "AMB_FFT_PEAK_MEDIAN_RATIO",
    ],
    # -- 空间-光强耦合(5) - limit 1 --
    "spatial_coupling": [
        "corr_Gmean_G_imbalance", "corr_Gmean_vmag",
        "corr_IR_G_imbalance", "corr_IR_vmag", "corr_Ambient_vmag",
    ],
    # -- 信号复杂度: Hjorth(3) + Entropy(3) = 6 - limit 2 --
    "signal_complexity": [
        "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
        "Entropy_Shannon", "Entropy_ApEn", "Entropy_SampEn",
    ],
    # -- 波形形态: Derivative(10) + Temporal(5) = 15 - limit 2 --
    "waveform_morphology": [
        "GTOP2_bp_skewness", "GTOP2_bp_kurtosis",
        "GTOP2_zero_cross_rate", "GTOP2_abs_diff_ratio",
        "Deriv_d1_mean", "Deriv_d1_std", "Deriv_d1_max", "Deriv_d1_min", "Deriv_d1_zcr",
        "Deriv_d2_mean", "Deriv_d2_std", "Deriv_d2_max", "Deriv_d2_min", "Deriv_d2_zcr",
        "Temporal_slope_mean", "Temporal_slope_std",
        "Temporal_peak_prominence", "Temporal_peak_ratio", "Temporal_valley_ratio",
        "GREEN_bp_skewness", "IRX_bp_skewness",
        "GREEN_bp_kurtosis", "IRX_bp_kurtosis",
    ],
    # -- ACC 核心 (13) - limit 2 --
    "acc_features": [
        "ACC_MAG_MEAN", "ACC_MAG_STD", "ACC_MAG_MAD",
        "ACC_AXIS_STD_SUM", "ACC_GRAVITY_DOM_RATIO",
        "ACC_BP_RMS", "ACC_DIFF_MAD", "ACC_STILL_SCORE",
        "ACC_MAG_P50", "ACC_MAG_P90",
        "ACC_GREEN_BP_CORR", "ACC_IR_BP_CORR",
        "ACC_ENERGY_TO_GREEN_AC",
        "ACC_TO_GTOP2_AC_RATIO", "ACC_STILL_X_GREEN_STABILITY",
        "ACC_DIFF_TO_GTOP2_DIFF_RATIO", "ACC_STILL_GREEN_MISMATCH",
    ],
    # -- ACC 分轴统计 (12) - limit 1 --
    "acc_per_axis": [
        "ACC_X_MEAN", "ACC_Y_MEAN", "ACC_Z_MEAN",
        "ACC_X_STD", "ACC_Y_STD", "ACC_Z_STD",
        "ACC_X_ENERGY", "ACC_Y_ENERGY", "ACC_Z_ENERGY",
        "ACC_AXIS_MEAN_SUM", "ACC_MAG_ENERGY", "ACC_MAG_P2P",
    ],
    # -- ACC 震颤检测 (4) - limit 1 --
    "acc_tremor": [
        "ACC_TREMOR_PEAK_FREQ", "ACC_TREMOR_PEAK_POWER",
        "ACC_TREMOR_POWER_RATIO", "ACC_LOW_MOTION_RATIO",
    ],
    # -- ACC 姿态/重力 (3) - limit 1 --
    "acc_orientation": [
        "ACC_TILT_ANGLE", "ACC_DOM_AXIS", "ACC_GRAVITY_RATIO",
    ],
    # -- Meta --
    "meta": ["SIG_LEN", "SIG_SEC"],
    "mode": ["mode"],
}

GROUP_LIMITS_DEFAULT = {
    "commercial_baseline": 8,
    "signal_quality": 2,
    "short_window_stability": 2,
    "short_window_frequency": 1,
    "green_stats": 2,
    "ambient_stats": 2,
    "green_spatial": 3,
    "green_3ch_consistency": 3,
    "amb_cross": 2,
    "frequency": 4,
    "spatial_coupling": 1,
    "signal_complexity": 2,
    "waveform_morphology": 3,
    "acc_features": 1,
    "acc_per_axis": 1,
    "acc_tremor": 1,
    "acc_orientation": 1,
    "meta": 0,
    "mode": 1,
    "other": 2,
}

GROUP_LIMITS_ACCURACY_FIRST = {
    **GROUP_LIMITS_DEFAULT,
    "green_stats": 4,
    "green_spatial": 4,
    "green_3ch_consistency": 4,
    "frequency": 5,
    "amb_cross": 3,
    # ACC reduced: motion features less informative for wearing detection
    "acc_features": 1,
    "acc_per_axis": 1,
    "acc_tremor": 1,
    "acc_orientation": 1,
}


def group_limits_for_ranking_objective(ranking_objective):
    objective = str(ranking_objective or "balanced").strip().lower()
    if objective in {"window_accuracy", "accuracy", "accuracy_first"}:
        return GROUP_LIMITS_ACCURACY_FIRST
    return GROUP_LIMITS_DEFAULT

DEPLOYMENT_ALLOWED_NON_FFT_FEATURES = {
    # Signal quality and robust scalar features.
    "SQI_FLAT_RATIO", "SQI_SPIKE_RATIO",
    "GREEN_ROBUST_RANGE_RATIO", "AMB_ROBUST_RANGE_RATIO",
    "GREEN_SEG_ACDC_CV", "AMB_SEG_ACDC_CV",
    # Green and ambient statistics.
    "G_mean_mean", "G_mean_std", "G_mean_diff_std", "G_mean_acdc",
    "GREEN_DC_MEDIAN", "GREEN_DC_IQR", "GREEN_AC_RMS", "GREEN_AC_MAD",
    "GREEN_AC_DC_RATIO", "GREEN_DERIV_MAD",
    "Ambient_mean", "Ambient_std", "Ambient_p95", "corr_Ambient_Gmean",
    "AMBX_DC_MEDIAN", "AMBX_DC_IQR", "AMBX_AC_RMS", "AMBX_AC_MAD",
    "AMBX_AC_DC_RATIO", "AMBX_DERIV_MAD",
    "GREEN_AC", "AMB_AC", "GREEN_DC", "AMB_DC", "GREEN_CORR",
    "AMB_AC_TO_GREEN_AC", "AMB_DC_TO_GREEN_DC",
    "GREEN_AMB_BP_CORR", "GREEN_AMB_ENV_CORR", "GREEN_AMB_LEAK",
    # Three-green spatial and reliability features.
    "G_imbalance_mean", "G_imbalance_p90", "G_imbalance_iqr",
    "G_rangeNorm_mean", "G_rangeNorm_p90",
    "G_spatial_vmag_mean", "G_spatial_vmag_p90", "G_spatial_vmag_iqr",
    "G_spatial_vmag_std", "G_ch_dc_cv", "G_ch_dc_max_min_ratio",
    "GCH_DC_RANGE_RATIO", "GCH_AC_RANGE_RATIO",
    "G_2OF3_AC_SUPPORT", "G_TOP2_TO_ALL_AC_RATIO", "G_TOP2_CORR_MIN",
    "G_WEAK_CHANNEL_GAP", "G_SPATIAL_STABILITY_SCORE",
    "G_TOP1_TO_TOP2_AC_RATIO", "G_TOP2_RANK_STABILITY", "G_TOP2_SWITCH_RATE",
    "G_SPATIAL_VMAG_RANGE", "GREEN_AMB_SEG_CORR_RANGE",
    # Top-2 green non-FFT statistics.
    "GTOP2_ROBUST_RANGE_RATIO", "GTOP2_SEG_ACDC_CV",
    "GTOP2_DC_MEDIAN", "GTOP2_DC_IQR", "GTOP2_AC_RMS", "GTOP2_AC_MAD",
    "GTOP2_AC_DC_RATIO", "GTOP2_DERIV_MAD",
    "GTOP2_bp_skewness", "GTOP2_bp_kurtosis",
    "GTOP2_zero_cross_rate", "GTOP2_abs_diff_ratio",
    "GTOP2_HALF_ACDC_DELTA", "GTOP2_SEG_ACDC_RANGE",
    "GREEN_AMB_LEAK_STABILITY",
    # ACC features that stay simple for endpoint deployment.
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
    # Metadata.
    "SIG_LEN", "SIG_SEC", "mode",
    "TOTAL_INVALID_COUNT", "PPG_INVALID_COUNT", "GREEN_INVALID_COUNT",
    # GREEN/GTOP2/AMBX waveform features (C-friendly: diff, var, sqrt, polyfit)
    "GREEN_Deriv_d1_mean", "GREEN_Deriv_d1_std", "GREEN_Deriv_d1_max",
    "GREEN_Deriv_d1_min", "GREEN_Deriv_d1_zcr",
    "GREEN_Temporal_slope_mean", "GREEN_Temporal_slope_std",
    "GREEN_Temporal_peak_prominence", "GREEN_Temporal_peak_ratio",
    "GREEN_Hjorth_Activity", "GREEN_Hjorth_Mobility",
    "GREEN_Entropy_SampEn",
    "GREEN_bp_skewness", "GREEN_bp_kurtosis",
    "GREEN_FFT_SNR", "GREEN_FFT_harmonic_ratio", "GREEN_FFT_harmonic_present",
    "GREEN_FFT_peak_width_Hz", "GREEN_XCORR",
    "GREEN_BAND_ENERGY_RATIO", "GREEN_FFT_PEAK_MEDIAN_RATIO", "GREEN_DOM_FREQ",
    "FFT_PEAK_MEDIAN_RATIO",
    "GREEN_SAT_FRAC", "GREEN_CLIP_RATE",
    "GTOP2_Deriv_d1_mean", "GTOP2_Deriv_d1_std", "GTOP2_Deriv_d1_max",
    "GTOP2_Deriv_d1_min", "GTOP2_Deriv_d1_zcr",
    "GTOP2_Temporal_slope_mean", "GTOP2_Temporal_slope_std",
    "GTOP2_Temporal_peak_prominence", "GTOP2_Temporal_peak_ratio",
    "GTOP2_Hjorth_Activity", "GTOP2_Hjorth_Mobility",
    "AMBX_Deriv_d1_mean", "AMBX_Deriv_d1_std", "AMBX_Deriv_d1_max",
    "AMBX_Deriv_d1_min", "AMBX_Deriv_d1_zcr",
    "AMBX_Temporal_slope_mean", "AMBX_Temporal_slope_std",
    "AMBX_Temporal_peak_prominence", "AMBX_Temporal_peak_ratio",
    "AMBX_bp_skewness", "AMBX_bp_kurtosis",
    # ACC tremor (C-friendly: FFT + band sum)
    "ACC_TREMOR_PEAK_FREQ", "ACC_TREMOR_PEAK_POWER",
    "ACC_TREMOR_POWER_RATIO", "ACC_LOW_MOTION_RATIO",
    "ACC_SAT_FRAC", "ACC_CLIP_RATE",
    # Green dropout and spatial correlation
    "G_DROPOUT_COUNT", "G_DROPOUT_ANGLE", "G_MIN_CHANNEL_ID",
    "G_TOP2_CHANNEL_COUNT", "G_TOP2_WORST_IDX",
    "G_bp_corr_mean", "G_bp_corr_min", "G_bp_corr_std", "G_bp_lag_std",
    "corr_Ambient_vmag", "corr_Gmean_G_imbalance", "corr_Gmean_vmag",
}

DEPLOYMENT_FFT_FEATURE_SOURCES = {
    "GTOP2_BAND_ENERGY_RATIO": "green_top2",
    "GTOP2_FFT_PEAK_MEDIAN_RATIO": "green_top2",
    "GTOP2_DOM_FREQ": "green_top2",
    "AMB_BAND_ENERGY_RATIO": "ambient",
    "AMB_FFT_PEAK_MEDIAN_RATIO": "ambient",
    "AMB_DOM_FREQ": "ambient",
    "AMBX_FFT_PEAK_MEDIAN_RATIO": "ambient",
    "AMBX_DOM_FREQ": "ambient",
    "GREEN_BAND_ENERGY_RATIO": "green",
    "GREEN_FFT_PEAK_MEDIAN_RATIO": "green",
    "GREEN_DOM_FREQ": "green",
    "FFT_PEAK_MEDIAN_RATIO": "green",
}

DEPLOYMENT_FORBIDDEN_TOKENS = (
    "coherence",  # Welch PSD + cross-spectrum, C-port hard
)

DEPLOYMENT_ALLOWED_FFT_SOURCES = {"green_top2", "green", "ambient"}

# s03 is the source of truth for the model-facing deployment-friendly feature pool.
# s04 may rank and select features, but it must not define a second allowlist.
DEPLOYMENT_ALLOWED_NON_FFT_FEATURES = S03_DEPLOYMENT_ALLOWED_NON_FFT_FEATURES
DEPLOYMENT_ALLOWED_FFT_FEATURES = S03_DEPLOYMENT_ALLOWED_FFT_FEATURES


def _deployment_fft_source_rank(source):
    order = {"green_top2": 0, "ambient": 1}
    return order.get(source, 99)


def deployment_fft_source_for_feature(feature):
    return DEPLOYMENT_FFT_FEATURE_SOURCES.get(str(feature))


def is_deployment_allowed_feature(feature):
    """Return whether s03 exposes this feature in the deployment-friendly pool."""
    return bool(s03_is_deployment_friendly_stage2_feature(feature))


def filter_features_for_deployment(features):
    """Filter by the s03 source-of-truth deployment-friendly Stage2 policy."""
    return list(s03_filter_deployment_friendly_stage2_features(list(features)))


def summarize_deployment_feature_costs(features):
    fft_sources = sorted({
        deployment_fft_source_for_feature(f)
        for f in features
        if deployment_fft_source_for_feature(f) is not None
    }, key=_deployment_fft_source_rank)
    forbidden_selected = [
        f for f in features
        if not is_deployment_allowed_feature(f)
    ]
    return {
        "feature_set": "deployment_friendly",
        "feature_count": int(len(features)),
        "fft_sources": fft_sources,
        "fft_source_count": int(len(fft_sources)),
        "forbidden_selected": forbidden_selected,
        "forbidden_selected_count": int(len(forbidden_selected)),
    }


def get_feature_cols(df):
    exclude = set(META_COLS)
    cols = [c for c in df.columns if c not in exclude]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def feature_to_group(feature):
    for g, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return g
    # 前缀匹配：GREEN_/IRX_/AMBX_ + 基础特征名 → 对应组
    for pf in ("GREEN_", "IRX_", "AMBX_"):
        if feature.startswith(pf):
            base = feature[len(pf):]
            for g, cols in FEATURE_GROUPS.items():
                if base in cols:
                    return g
            break
    return "other"


# =========================================================
# 阶段1：组内快速预筛
# =========================================================

def fast_group_preselection(df, feature_cols, group_limits=None, preselect_top=4):
    if group_limits is None:
        group_limits = GROUP_LIMITS_DEFAULT

    y = df["target"].values.astype(int)
    group_features = defaultdict(list)
    for f in feature_cols:
        group_features[feature_to_group(f)].append(f)

    selected = {}
    for group_name, features_in_group in group_features.items():
        if len(features_in_group) < 2:
            for f in features_in_group:
                selected[f] = {"group": group_name, "importance": 1.0, "method": "only"}
            continue

        limit = group_limits.get(group_name, 1)
        actual_select = min(preselect_top, len(features_in_group), limit * 2)
        print(f"  [{group_name}] {len(features_in_group)} features, top {actual_select}...")
        try:
            X_group = df[features_in_group].values.astype(float)
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=20, reg_lambda=10, reg_alpha=1,
                objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=get_inner_n_jobs(), verbosity=0,
            )
            model.fit(X_group, y)

            importance_dict = model.get_booster().get_score(importance_type='gain')
            importance = {}
            for idx_str, imp in importance_dict.items():
                try:
                    idx = int(idx_str[1:])
                except Exception:
                    continue
                if 0 <= idx < len(features_in_group):
                    importance[features_in_group[idx]] = imp

            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for i, (f, imp) in enumerate(sorted_features[:actual_select]):
                selected[f] = {
                    "group": group_name,
                    "importance": float(imp),
                    "method": "gain",
                    "rank": i + 1,
                }
            print("done")
        except Exception as e:
            logger.warning(f"fast_group_preselection: group={group_name} 训练失败({e})；"
                           f"回退为顺序保留前 {actual_select} 个，importance 标为 0。")
            for i, f in enumerate(features_in_group[:actual_select]):
                selected[f] = {
                    "group": group_name,
                    "importance": 0.0,
                    "method": "fallback_after_fit_error",
                    "rank": i + 1,
                }
            print("fallback")

    return selected


# =========================================================
# 数据清洗（按 train）
# =========================================================

def clean_features_by_train(df_train, df_valid, feature_cols, missing_thresh=0.3,
                             var_thresh=1e-8, corr_thresh=0.95, skip_vif=False):
    df_train = df_train.copy()
    df_valid = df_valid.copy()
    removed = {"missing": [], "low_variance": [], "high_corr": []}

    # inf -> nan（向量化）
    df_train[feature_cols] = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)
    in_valid = [c for c in feature_cols if c in df_valid.columns]
    if in_valid:
        df_valid[in_valid] = df_valid[in_valid].replace([np.inf, -np.inf], np.nan)

    # 1. high missing
    miss_rate = df_train[feature_cols].isna().mean()
    kept = miss_rate[miss_rate <= missing_thresh].index.tolist()
    removed["missing"] = miss_rate[miss_rate > missing_thresh].index.tolist()

    # 2. median fill
    fill_values = {}
    med_series = df_train[kept].median()
    for c in kept:
        med = med_series[c]
        if not np.isfinite(med):
            med = 0.0
        fill_values[c] = float(med)
    df_train[kept] = df_train[kept].fillna(med_series.fillna(0.0))
    valid_kept = [c for c in kept if c in df_valid.columns]
    if valid_kept:
        df_valid[valid_kept] = df_valid[valid_kept].fillna(med_series[valid_kept].fillna(0.0))

    # 3. low variance
    var_series = df_train[kept].var()
    kept2 = var_series[np.isfinite(var_series) & (var_series > var_thresh)].index.tolist()
    removed["low_variance"] = [c for c in kept if c not in kept2]

    # 4. high corr: 对于高相关对，保留与 target 更相关的那个
    if len(kept2) > 1:
        y = df_train["target"].values
        corr = df_train[kept2].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_remove = set()

        def safe_abs_corr(values, target):
            values = np.asarray(values, dtype=float)
            target = np.asarray(target, dtype=float)
            mask = np.isfinite(values) & np.isfinite(target)
            if np.sum(mask) < 2:
                return 0.0
            values, target = values[mask], target[mask]
            if np.var(values) <= 0.0 or np.var(target) <= 0.0:
                return 0.0
            corr_value = np.corrcoef(values, target)[0, 1]
            return float(abs(corr_value)) if np.isfinite(corr_value) else 0.0

        for col_i in upper.columns:
            high_corr_with = [col_j for col_j in upper.index if upper.at[col_j, col_i] > corr_thresh]
            if not high_corr_with:
                continue
            # 对每个高相关对，保留与 target 更相关的特征
            for col_j in high_corr_with:
                if col_i in to_remove or col_j in to_remove:
                    continue
                # 用中位数填充可能的 NaN 再算 target corr
                vi = df_train[col_i].fillna(df_train[col_i].median()).values
                vj = df_train[col_j].fillna(df_train[col_j].median()).values
                corr_i = safe_abs_corr(vi, y)
                corr_j = safe_abs_corr(vj, y)
                if corr_i >= corr_j:
                    to_remove.add(col_j)
                else:
                    to_remove.add(col_i)
        kept3 = [c for c in kept2 if c not in to_remove]
        removed["high_corr"] = sorted(list(to_remove))
    else:
        kept3 = kept2

    # 5. VIF (Variance Inflation Factor) — 批量剔除 VIF > 10 的特征
    removed["high_vif"] = []
    if skip_vif:
        print("  VIF: skipped (--skip_vif)")
    elif len(kept3) > 2:
        from sklearn.linear_model import Ridge
        _kept = list(kept3)
        _X = df_train[_kept].values.astype(float)
        _mu = np.nanmean(_X, axis=0)
        _sd = np.nanstd(_X, axis=0)
        _sd[~np.isfinite(_sd) | (_sd < 1e-12)] = 1.0
        _X = (_X - _mu) / _sd
        _X = np.nan_to_num(_X, nan=0.0, posinf=0.0, neginf=0.0)
        _max_iter = max(len(_kept) // 2, 1)
        _orig_n = len(_kept)
        for _iter in range(_max_iter):
            nf = _X.shape[1]
            if nf <= 2:
                break
            print(f"  VIF round {_iter+1}: computing VIF for {nf} features...")
            vif_vals = np.full(nf, np.inf)
            col_mask = np.ones(nf, dtype=bool)
            for j in range(nf):
                col_mask[j] = False
                X_rest = _X[:, col_mask]  # 列索引切片，无复制
                y_j = _X[:, j]
                try:
                    # VIF uses highly collinear feature blocks by design; lsqr avoids
                    # Cholesky/normal-equation warnings on nearly singular matrices.
                    ridge = Ridge(alpha=1.0, fit_intercept=False, solver="lsqr")
                    ridge.fit(X_rest, y_j)
                    y_pred = ridge.predict(X_rest)
                    ss_res = np.sum((y_j - y_pred) ** 2)
                    ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                    r2 = max(0.0, min(r2, 0.999))
                    vif_vals[j] = 1.0 / (1.0 - r2)
                except Exception:
                    vif_vals[j] = 100.0
                col_mask[j] = True

            # 一次性移除所有 VIF > 10 的特征
            high_vif_mask = vif_vals > 10.0
            n_remove = int(np.sum(high_vif_mask))
            if n_remove == 0:
                print("  done (all VIF <= 10)")
                break
            print(f"  remove {n_remove}")
            for j in range(nf - 1, -1, -1):
                if high_vif_mask[j]:
                    removed["high_vif"].append(_kept[j])
                    _kept.pop(j)
            _X = df_train[_kept].values.astype(float)  # 重建（仅当有删除时）
            _mu = np.nanmean(_X, axis=0)
            _sd = np.nanstd(_X, axis=0)
            _sd[~np.isfinite(_sd) | (_sd < 1e-12)] = 1.0
            _X = (_X - _mu) / _sd
            _X = np.nan_to_num(_X, nan=0.0, posinf=0.0, neginf=0.0)
        kept3 = _kept
        print(f"  VIF: {_orig_n} -> {len(kept3)} features")

    return df_train, df_valid, kept3, removed, fill_values


# =========================================================
# 稳定性选择（并行）
# =========================================================

_WORKER_DATA = None


def _init_stab_worker(data_pickle):
    global _WORKER_DATA
    _WORKER_DATA = pickle.loads(data_pickle)


def _run_one_fold(args_tuple):
    """
    单 (seed, fold) 训练 + permutation importance。
    返回 (fold_info, top_k 列表 或 [])。
    """
    seed, fold_id, n_folds, tr_idx, va_idx, top_k = args_tuple
    data = _WORKER_DATA
    X = data["X"]
    y = data["y"]
    feature_cols = data["feature_cols"]

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    n_tr = len(y_tr)
    n_va = len(y_va)
    info = {"seed": seed, "fold": fold_id + 1, "n_folds": n_folds,
            "n_train": n_tr, "n_valid": n_va, "auc": None, "kept": False}

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return info, []

    model = xgb.XGBClassifier(
        n_estimators=30, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=10, reg_alpha=1,
        objective="binary:logistic", eval_metric="logloss",
        random_state=seed, n_jobs=1, verbosity=0,
    )

    try:
        model.fit(X_tr, y_tr)
    except Exception as e:
        logger.warning(f"fold seed={seed} 训练失败: {e}")
        return info, []

    try:
        proba = model.predict_proba(X_va)[:, 1]
        val_auc = roc_auc_score(y_va, proba)
    except Exception:
        val_auc = 0.5

    info["auc"] = float(val_auc)
    min_fold_auc = data.get("min_fold_auc", 0.55)
    if not np.isfinite(val_auc) or val_auc < min_fold_auc:
        return info, []

    try:
        result = permutation_importance(
            model, X_va, y_va,
            scoring="roc_auc",
            n_repeats=3,
            random_state=seed,
            n_jobs=get_inner_n_jobs(),
        )
    except Exception as e:
        logger.warning(f"fold seed={seed} permutation 失败: {e}")
        return info, []

    imps = result.importances_mean
    order = np.argsort(imps)[::-1]
    k = min(top_k, len(feature_cols))
    out = []
    for rank, idx in enumerate(order[:k]):
        out.append((int(idx), float(imps[idx]), int(rank + 1)))
    info["kept"] = True
    return info, out


def _run_one_fold_serial(args_tuple, data):
    """单进程路径下直接调用。"""
    global _WORKER_DATA
    _WORKER_DATA = data
    try:
        return _run_one_fold(args_tuple)
    finally:
        _WORKER_DATA = None


def stability_selection(df, feature_cols, max_splits=5, seeds=None, n_workers=None, min_fold_auc=0.55):
    if seeds is None:
        seeds = [1, 7, 42]  # 3 seeds (was 5)

    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(int)
    groups = df["sample_name"].values

    unique_groups = np.unique(groups)
    n_splits = min(max_splits, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("sample group 数量不足，无法 GroupKFold。")

    # 小数据自动放宽 AUC 门槛
    if X.shape[0] < 200:
        min_fold_auc = max(0.50, min_fold_auc - 0.10)

    # 构造任务: (seed, fold_id, n_folds, tr, va, top_k)
    n_folds_total = len(seeds) * n_splits
    tasks = []
    for seed in seeds:
        gkf = GroupKFold(n_splits=n_splits)
        for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
            tasks.append((seed, fold_id, n_splits,
                          np.asarray(tr_idx, dtype=np.int64),
                          np.asarray(va_idx, dtype=np.int64),
                          min(15, len(feature_cols))))

    n_workers = resolve_n_workers(n_workers, n_items=len(tasks))

    data = {"X": X, "y": y, "feature_cols": feature_cols, "min_fold_auc": min_fold_auc}

    # 小数据集走单进程，跳过 pickle 序列化开销
    use_mp = n_workers > 1 and len(tasks) > 4
    print(f"\n  稳定性选择: {n_folds_total} folds (seeds={len(seeds)} x splits={n_splits}), "
          f"workers={'mp' if use_mp else 1}")

    fold_results = []
    if use_mp:
        data_pickle = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        pool_kwargs = {
            "max_workers": n_workers,
            "initializer": _init_stab_worker,
            "initargs": (data_pickle,),
        }
        mp_ctx = multiprocessing_context_from_env()
        if mp_ctx is not None:
            pool_kwargs["mp_context"] = mp_ctx
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {ex.submit(_run_one_fold, t): i for i, t in enumerate(tasks)}
            done_count = 0
            kept_count = 0
            total = len(futures)
            print(f"  (parallel, {total} folds)")
            for fut in as_completed(futures):
                try:
                    info, fold_out = fut.result()
                except Exception as e:
                    logger.warning(f"stability fold worker crashed: {e}")
                    info = {"seed": "?", "fold": "?", "n_folds": n_splits, "auc": None}
                    fold_out = []
                done_count += 1
                if fold_out:
                    kept_count += 1
                auc_str = f"auc={info['auc']:.3f}" if info['auc'] else "auc=?"
                tag = "KEPT" if fold_out else "DROP"
                print(f"  [{done_count}/{total}] seed={info['seed']} fold={info['fold']}/{info['n_folds']} "
                      f"{auc_str} {tag}")
                fold_results.append((info, fold_out))
    else:
        for i, t in enumerate(tasks):
            seed, fold_id, n_folds = t[0], t[1], t[2]
            print(f"  [{i+1}/{len(tasks)}] seed={seed} fold={fold_id+1}/{n_folds} training...")
            info, fold_out = _run_one_fold_serial(t, data)
            auc_str = f"auc={info['auc']:.3f}" if info['auc'] else "auc=?"
            tag = "KEPT" if fold_out else "DROP"
            print(f"{auc_str} {tag}")
            fold_results.append((info, fold_out))

    # 聚合结果
    selected_count = defaultdict(int)
    importance_sum = defaultdict(float)
    rank_sum = defaultdict(float)
    total_runs = 0
    for _info, fold_out in fold_results:
        if not fold_out:
            continue
        total_runs += 1
        for idx, imp, rank in fold_out:
            f = feature_cols[idx]
            selected_count[f] += 1
            importance_sum[f] += imp
            rank_sum[f] += rank

    print(f"  有效 folds: {total_runs}/{n_folds_total}")

    summary = []
    for f in feature_cols:
        count = selected_count[f]
        freq = count / max(total_runs, 1)
        avg_imp = importance_sum[f] / max(count, 1)
        avg_rank = rank_sum[f] / max(count, 1)
        summary.append({
            "feature": f,
            "group": feature_to_group(f),
            "freq": float(freq),
            "avg_importance": float(avg_imp),
            "avg_rank": float(avg_rank),
            "count": int(count),
        })

    summary = sorted(
        summary,
        key=lambda x: (x["freq"], x["avg_importance"], -x["avg_rank"]),
        reverse=True,
    )
    return summary


# =========================================================
# SHAP
# =========================================================

def shap_importance(df, feature_cols, selected_features=None):
    if not SHAP_AVAILABLE:
        return {}
    if selected_features is None:
        selected_features = feature_cols

    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(int)
    if len(np.unique(y)) < 2:
        return {}

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, n_jobs=get_inner_n_jobs(), verbosity=0,
    )
    model.fit(X, y)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return {f: float(mean_abs_shap[i]) for i, f in enumerate(feature_cols)}
    except Exception as e:
        print(f"SHAP计算失败: {e}")
        return {}


def shap_consistency_check(df_train, df_valid, feature_cols):
    """
    在 train 上训练模型后，对 train 与 valid 分别算 SHAP，比较排序一致性。
    返回 dict 含两套 SHAP 与 Spearman 相关、Top-K 重合度。
    用途：识别 train-only 强但泛化弱的"假重要特征"。
    """
    if not SHAP_AVAILABLE:
        return {"available": False}

    X_tr = df_train[feature_cols].values.astype(float)
    y_tr = df_train[["target"]].values.astype(int).ravel()
    X_va = df_valid[feature_cols].values.astype(float)
    if len(np.unique(y_tr)) < 2 or len(X_va) == 0:
        return {"available": False, "reason": "insufficient_data"}

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, n_jobs=get_inner_n_jobs(), verbosity=0,
    )
    model.fit(X_tr, y_tr)

    def _shap_of(X):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        return np.abs(sv).mean(axis=0)

    try:
        s_tr = _shap_of(X_tr)
        s_va = _shap_of(X_va)
    except Exception as e:
        return {"available": False, "reason": f"shap_fail: {e}"}

    # Spearman 相关
    from scipy.stats import spearmanr
    rho, _ = spearmanr(s_tr, s_va)

    # Top-K 重合
    k = min(10, len(feature_cols))
    top_tr = set(np.argsort(s_tr)[-k:])
    top_va = set(np.argsort(s_va)[-k:])
    overlap = len(top_tr & top_va) / float(k)

    shap_train = {f: float(s_tr[i]) for i, f in enumerate(feature_cols)}
    shap_valid = {f: float(s_va[i]) for i, f in enumerate(feature_cols)}

    # 哪些特征 train 高但 valid 低（可疑 train-only）
    suspicious = []
    rank_tr = {f: int(r) for r, f in enumerate(sorted(feature_cols, key=lambda f: shap_train[f], reverse=True))}
    rank_va = {f: int(r) for r, f in enumerate(sorted(feature_cols, key=lambda f: shap_valid[f], reverse=True))}
    for f in feature_cols:
        if rank_tr[f] < k and rank_va[f] > 2 * k:
            suspicious.append({
                "feature": f,
                "rank_train": rank_tr[f],
                "rank_valid": rank_va[f],
                "shap_train": shap_train[f],
                "shap_valid": shap_valid[f],
            })

    return {
        "available": True,
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "topk_overlap": float(overlap),
        "k": int(k),
        "shap_train": shap_train,
        "shap_valid": shap_valid,
        "suspicious_train_only": suspicious,
    }


def export_shap_importance_plot(shap_train, shap_valid, spearman_rho, topk_overlap,
                                suspicious, selected_features, artifact_dir):
    """Export SHAP feature importance and consistency visualisation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip SHAP plot: {e}")
        return None

    out_dir = os.path.join(str(artifact_dir), "report_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s04_shap_importance.png")

    shap_items_tr = sorted(shap_train.items(), key=lambda kv: kv[1], reverse=True) if shap_train else []
    top_n = min(20, len(shap_items_tr))
    selected_set = set(selected_features or [])

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)

    # (0,0) SHAP bar — top 20 features from train
    ax_bar = fig.add_subplot(gs[0, 0])
    if shap_items_tr:
        top_items = shap_items_tr[:top_n][::-1]
        names = [x[0] for x in top_items]
        vals = [x[1] for x in top_items]
        colors = [
            "#2f6f73" if name in selected_set else "#9aa6ac"
            for name in names
        ]
        ax_bar.barh(np.arange(len(names)), vals, color=colors, height=0.68)
        ax_bar.set_yticks(np.arange(len(names)))
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.set_xlabel("mean(|SHAP|)")
        ax_bar.set_title(f"SHAP Feature Importance (Top {top_n})")
        ax_bar.grid(axis="x", alpha=0.18)
    else:
        ax_bar.text(0.5, 0.5, "No SHAP data", ha="center", va="center")
        ax_bar.set_axis_off()
        ax_bar.set_title("SHAP Feature Importance")

    # (0,1) Train vs Valid SHAP scatter
    ax_sc = fig.add_subplot(gs[0, 1])
    common = [f for f in shap_train if f in shap_valid]
    if common and len(common) >= 2:
        x_vals = [shap_train[f] for f in common]
        y_vals = [shap_valid[f] for f in common]
        min_v = min(min(x_vals), min(y_vals)) * 0.9
        max_v = max(max(x_vals), max(y_vals)) * 1.1
        ax_sc.scatter(x_vals, y_vals, c="#4c78a8", alpha=0.7, s=22, edgecolors="none")
        ax_sc.plot([min_v, max_v], [min_v, max_v], color="#9aa6ac", linewidth=1.2, linestyle="--")
        rho_text = f"ρ = {spearman_rho:.4f}" if spearman_rho is not None else "ρ = N/A"
        ax_sc.text(0.05, 0.95, rho_text, transform=ax_sc.transAxes, fontsize=11, va="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4f6f7", edgecolor="#d9dee2"))
        ax_sc.set_xlim(min_v, max_v)
        ax_sc.set_ylim(min_v, max_v)
    else:
        ax_sc.text(0.5, 0.5, "Insufficient features for comparison",
                   ha="center", va="center")
        ax_sc.set_axis_off()
    ax_sc.set_xlabel("Train mean(|SHAP|)")
    ax_sc.set_ylabel("Valid mean(|SHAP|)")
    ax_sc.set_title("Train vs Valid SHAP Consistency")
    ax_sc.grid(alpha=0.18)

    # (1,0) Top-K overlap bar
    ax_ov = fig.add_subplot(gs[1, 0])
    k_vals = [3, 5, 10, 15, 20]
    bar_vals = []
    for kval in k_vals:
        k_eff = min(kval, len(common)) if common else 0
        if k_eff <= 0:
            bar_vals.append(0.0)
            continue
        top_tr_k = set(sorted(common, key=lambda f: shap_train.get(f, 0.0), reverse=True)[:k_eff])
        top_va_k = set(sorted(common, key=lambda f: shap_valid.get(f, 0.0), reverse=True)[:k_eff])
        bar_vals.append(len(top_tr_k & top_va_k) / float(k_eff))
    colors_ov = ["#2f6f73" if v >= 0.6 else "#d35f2d" if v < 0.4 else "#8172b2" for v in bar_vals]
    ax_ov.bar(range(len(k_vals)), bar_vals, color=colors_ov, width=0.6)
    ax_ov.set_xticks(range(len(k_vals)))
    ax_ov.set_xticklabels([f"k={k}" for k in k_vals])
    ax_ov.set_ylim(0, 1.05)
    ax_ov.set_ylabel("Overlap Ratio")
    ax_ov.set_title("Top-K Feature Overlap (Train ∩ Valid)")
    ax_ov.grid(axis="y", alpha=0.18)
    for i, v in enumerate(bar_vals):
        ax_ov.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # (1,1) Suspicious train-only features
    ax_sus = fig.add_subplot(gs[1, 1])
    if suspicious:
        sus_names = [s["feature"] for s in suspicious[:15]][::-1]
        sus_vals = [s["shap_train"] for s in suspicious[:15]][::-1]
        ax_sus.barh(np.arange(len(sus_names)), sus_vals, color="#c44e52", height=0.6)
        ax_sus.set_yticks(np.arange(len(sus_names)))
        ax_sus.set_yticklabels(sus_names, fontsize=8)
        ax_sus.set_xlabel("Train mean(|SHAP|)")
        ax_sus.set_title(f"Train-only Strong Features ({len(suspicious)} found)")
    else:
        ax_sus.text(0.5, 0.5, "No suspicious train-only features",
                    ha="center", va="center", fontsize=13, color="#2f6f73")
        ax_sus.set_axis_off()
        ax_sus.set_title("Train-only Strong Features")
    ax_sus.grid(axis="x", alpha=0.18)

    fig.suptitle("SHAP Feature Importance Report", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] s04 SHAP plot -> {out_path}")
    return out_path


# 标量绝对量纲（依赖原始 ADC 数值大小）的特征清单。
# 它们随肤色 / 传感器 / 老化漂移，应优先用 ratio/correlation 替代或在线上做基线自适应。
SCALE_DEPENDENT_FEATURES = {
    # ir_basic
    "IR_mean", "IR_std", "IR_p95", "IR_diff_std",
    # green_basic
    "G_mean_mean", "G_mean_std", "G_mean_diff_std",
    # ambient
    "Ambient_mean", "Ambient_std", "Ambient_p95",
    # robust_ac_dc - DC 绝对值
    "GREEN_DC_MEDIAN", "GREEN_DC_IQR", "GREEN_AC_RMS", "GREEN_AC_MAD", "GREEN_DERIV_MAD",
    "IRX_DC_MEDIAN", "IRX_DC_IQR", "IRX_AC_RMS", "IRX_AC_MAD", "IRX_DERIV_MAD",
    "AMBX_DC_MEDIAN", "AMBX_DC_IQR", "AMBX_AC_RMS", "AMBX_AC_MAD", "AMBX_DERIV_MAD",
    "PPG_GREEN_DC", "PPG_AMB_DC", "PPG_GREEN_AC", "PPG_AMB_AC",
    "GREEN_AC", "AMB_AC", "GREEN_DC", "AMB_DC", "ACC_YSUM",
    # acc 部分
    "ACC_MAG_MEAN", "ACC_AXIS_STD_SUM",
}


def annotate_scale_dependency(selected_features):
    """返回 (scale_dep_list, scale_inv_list)。仅元信息，不影响模型本身。"""
    scale_dep = [f for f in selected_features if f in SCALE_DEPENDENT_FEATURES]
    scale_inv = [f for f in selected_features if f not in SCALE_DEPENDENT_FEATURES]
    return scale_dep, scale_inv


def compute_drift_metrics(x_train, x_valid, n_bins=10):
    """
    计算单个特征 train/valid 分布漂移指标。

    Parameters:
        x_train: 1D array-like, train 特征值（已清洗、finite）
        x_valid: 1D array-like, valid 特征值（已清洗、finite）
        n_bins: PSI 分箱数

    Returns:
        dict: {ks_stat, ks_pvalue, psi, mean_shift, std_ratio}
        若数据不足则返回 None 值。
    """
    from scipy import stats as _stats

    x_train = np.asarray(x_train, dtype=np.float64)
    x_valid = np.asarray(x_valid, dtype=np.float64)

    mask_tr = np.isfinite(x_train)
    mask_va = np.isfinite(x_valid)
    if mask_tr.sum() < 4 or mask_va.sum() < 4:
        return {"ks_stat": None, "ks_pvalue": None, "psi": None,
                "mean_shift": None, "std_ratio": None}

    x_tr = x_train[mask_tr]
    x_va = x_valid[mask_va]

    # KS 双样本检验
    ks_stat, ks_pvalue = _stats.ks_2samp(x_tr, x_va)

    # PSI (Population Stability Index)
    bin_edges = np.percentile(x_tr, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    tr_counts, _ = np.histogram(x_tr, bins=bin_edges)
    va_counts, _ = np.histogram(x_va, bins=bin_edges)

    eps_psi = 1e-6
    tr_pct = np.clip(tr_counts / max(len(x_tr), 1), eps_psi, 1.0)
    va_pct = np.clip(va_counts / max(len(x_va), 1), eps_psi, 1.0)
    psi = float(np.sum((va_pct - tr_pct) * np.log(va_pct / tr_pct)))

    # 均值偏移（以 train std 为单位）
    tr_mean = float(np.mean(x_tr))
    tr_std = float(np.std(x_tr)) or 1e-12
    va_mean = float(np.mean(x_va))
    mean_shift = float((va_mean - tr_mean) / tr_std)

    # 标准差比
    va_std = float(np.std(x_va)) or 1e-12
    std_ratio = float(va_std / tr_std)

    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "psi": psi,
        "mean_shift": mean_shift,
        "std_ratio": std_ratio,
    }


DEPLOYMENT_COST_BY_GROUP = {
    "commercial_baseline": 1.2,
    "signal_quality": 0.8,
    "short_window_stability": 1.0,
    "short_window_frequency": 1.6,
    "ir_stats": 1.0,
    "green_stats": 1.0,
    "ambient_stats": 1.0,
    "ir_g_amplitude": 1.2,
    "ir_g_correlation": 1.4,
    "amb_cross": 1.4,
    "green_spatial": 1.5,
    "green_3ch_consistency": 1.6,
    "spatial_coupling": 1.8,
    "acc_features": 2.0,
    "frequency": 2.8,
    "waveform_morphology": 2.2,
    "signal_complexity": 3.2,
    "mode": 0.5,
    "meta": 0.2,
    "other": 2.0,
}


def deployment_feature_summary(feature):
    """Return deployment-practicality metadata for a candidate feature.

    s04 runs before the final s05/s06 model and state machine exist, so this is
    a deployment proxy: prefer lower-cost, scale-robust features when train-only
    importance is otherwise similar.
    """
    group = feature_to_group(feature)
    cost = float(DEPLOYMENT_COST_BY_GROUP.get(group, DEPLOYMENT_COST_BY_GROUP["other"]))
    reasons = [f"group={group}", f"cost={cost:g}"]

    if "SampEn" in feature or "Entropy" in feature:
        cost = max(cost, 3.5)
        reasons.append("entropy/O(N^2)-risk")
    if "FFT" in feature or "DOM_FREQ" in feature or "AUTO_CORR" in feature:
        cost = max(cost, 2.8)
        reasons.append("frequency-domain")

    scale_dependent = feature in SCALE_DEPENDENT_FEATURES
    reasons.append("scale-dependent" if scale_dependent else "scale-robust")

    cost_penalty = min(cost / 4.0, 1.0) * 0.45
    scale_penalty = 0.25 if scale_dependent else 0.0
    fit = float(max(0.0, min(1.0, 1.0 - cost_penalty - scale_penalty)))

    return {
        "deployment_group": group,
        "deployment_cost": cost,
        "deployment_scale_dependent": scale_dependent,
        "deployment_fit": fit,
        "deployment_reason": ", ".join(reasons),
    }


def add_deployment_scores(summary, deployment_score_weight=0.25):
    """Blend train-only importance with deployment practicality."""
    w = float(max(0.0, min(1.0, deployment_score_weight)))
    out = []
    for item in summary:
        enriched = dict(item)
        deployment_meta = deployment_feature_summary(enriched["feature"])
        enriched.update(deployment_meta)
        base = float(enriched.get("combined_score", 0.0))
        enriched["deployment_score"] = float((1.0 - w) * base + w * deployment_meta["deployment_fit"])
        enriched["deployment_score_weight"] = w
        out.append(enriched)
    return sorted(out, key=lambda x: x["deployment_score"], reverse=True)


def _max_consecutive_true(mask):
    best = 0
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _fp_proxy_for_feature(df, feature, recall_floor=0.95, state_k_on=3):
    """Train-only proxy for sample/state-machine FP risk of one feature.

    A feature can look useful at window level yet produce isolated or sustained
    false positive windows inside negative samples. This proxy chooses the
    feature direction and threshold from train positives, then measures how
    often train negative samples would trigger any-window FP and K-consecutive
    state-machine-style FP.
    """
    if feature not in df.columns or "target" not in df.columns or "sample_name" not in df.columns:
        return {
            "threshold": None,
            "direction": 1,
            "sample_fp_rate": 0.0,
            "state_fp_rate": 0.0,
            "fit": 1.0,
            "available": False,
            "reason": "missing_columns",
        }

    x = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = df["target"].astype(int)
    finite = x.notna()
    if finite.sum() < 4 or y[finite].nunique() < 2:
        return {
            "threshold": None,
            "direction": 1,
            "sample_fp_rate": 0.0,
            "state_fp_rate": 0.0,
            "fit": 1.0,
            "available": False,
            "reason": "insufficient_data",
        }

    fill = float(x[finite].median())
    score = x.fillna(fill).astype(float).to_numpy()
    y_arr = y.to_numpy()
    pos = score[y_arr == 1]
    neg = score[y_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {
            "threshold": None,
            "direction": 1,
            "sample_fp_rate": 0.0,
            "state_fp_rate": 0.0,
            "fit": 1.0,
            "available": False,
            "reason": "single_class",
        }

    direction = 1 if float(np.nanmedian(pos)) >= float(np.nanmedian(neg)) else -1
    directed = score * direction
    pos_directed = directed[y_arr == 1]
    q = max(0.0, min(1.0, 1.0 - float(recall_floor)))
    threshold = float(np.quantile(pos_directed, q))

    neg_df = pd.DataFrame({
        "sample_name": df["sample_name"].values,
        "target": y_arr,
        "score": directed,
    })
    neg_df = neg_df[neg_df["target"] == 0]
    if neg_df.empty:
        sample_fp_rate = 0.0
        state_fp_rate = 0.0
    else:
        sample_hits = []
        state_hits = []
        for _name, g in neg_df.groupby("sample_name", sort=False):
            hit = g["score"].to_numpy() >= threshold
            sample_hits.append(bool(np.any(hit)))
            state_hits.append(_max_consecutive_true(hit) >= max(1, int(state_k_on)))
        sample_fp_rate = float(np.mean(sample_hits)) if sample_hits else 0.0
        state_fp_rate = float(np.mean(state_hits)) if state_hits else 0.0

    penalty = 0.5 * sample_fp_rate + 0.5 * state_fp_rate
    return {
        "threshold": threshold,
        "direction": int(direction),
        "sample_fp_rate": sample_fp_rate,
        "state_fp_rate": state_fp_rate,
        "fit": float(max(0.0, min(1.0, 1.0 - penalty))),
        "available": True,
        "reason": "ok",
    }


def add_fp_cost_proxy_scores(summary, df_train, fp_cost_weight=0.25,
                             recall_floor=0.95, state_k_on=3):
    """Penalize features that create sample/state-level FP risk on train."""
    w = float(max(0.0, min(1.0, fp_cost_weight)))
    out = []
    for item in summary:
        enriched = dict(item)
        feature = enriched["feature"]
        proxy = _fp_proxy_for_feature(
            df_train, feature,
            recall_floor=recall_floor,
            state_k_on=state_k_on,
        )
        base = float(enriched.get("deployment_score", enriched.get("combined_score", 0.0)))
        enriched["fp_proxy_available"] = bool(proxy["available"])
        enriched["fp_proxy_reason"] = proxy["reason"]
        enriched["fp_proxy_threshold"] = proxy["threshold"]
        enriched["fp_proxy_direction"] = proxy["direction"]
        enriched["fp_proxy_sample_fp_rate"] = proxy["sample_fp_rate"]
        enriched["fp_proxy_state_fp_rate"] = proxy["state_fp_rate"]
        enriched["fp_proxy_fit"] = proxy["fit"]
        enriched["fp_cost_weight"] = w
        enriched["deployment_score_before_fp_cost"] = base
        enriched["deployment_score"] = float((1.0 - w) * base + w * proxy["fit"])
        out.append(enriched)
    return sorted(out, key=lambda x: x["deployment_score"], reverse=True)


def compute_all_feature_diagnostics(df_train, df_valid, feature_cols, fill_values=None,
                                    kept_features=None, removed_map=None):
    """
    对所有候选特征生成完整诊断表（train-only 评估 + train/valid 漂移）。

    返回 DataFrame，列：
      feature, group, missing_rate, variance, train_auc,
      fp_proxy_fit, fp_proxy_sample_fp_rate, fp_proxy_state_fp_rate,
      deployment_cost, is_scale_dependent,
      ks_stat, ks_pvalue, psi, mean_shift, std_ratio,
      removed, removed_reason

    Parameters:
        df_train: 训练集 DataFrame（已清洗或未清洗）
        df_valid: 验证集 DataFrame
        feature_cols: 所有候选特征列名
        fill_values: dict feature→fill_value，若为 None 则自动用 train 中位数
        kept_features: 可选，复用主流程 clean_features_by_train 的保留特征
        removed_map: 可选，复用主流程 clean_features_by_train 的移除原因
    """
    rows = []
    y_tr = df_train["target"].values.astype(int)

    # 优先复用主流程的清洗结果，避免诊断导出阶段重复运行相关性/VIF。
    if kept_features is None or removed_map is None:
        _, _, kept_features, removed_map, _ = clean_features_by_train(
            df_train, df_valid, feature_cols,
            missing_thresh=0.3, var_thresh=1e-8, corr_thresh=0.90, skip_vif=False,
        )
    kept_set = set(kept_features)
    removed_reasons = {}
    for reason, flist in removed_map.items():
        for f in flist:
            removed_reasons[f] = reason

    if fill_values is None:
        fill_values = {}
        for f in feature_cols:
            if f in df_train.columns:
                x_raw = pd.to_numeric(df_train[f], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan)
                fill_values[f] = float(x_raw.median()) if x_raw.notna().any() else 0.0

    for f in feature_cols:
        group = feature_to_group(f)
        profile = deployment_feature_summary(f)
        is_removed = f not in kept_set
        removed_reason = removed_reasons.get(f, "")

        # 缺失率
        x_tr_raw = pd.to_numeric(df_train[f], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
        missing_rate = float(x_tr_raw.isna().mean())

        # 填充后的方差
        fv = fill_values.get(f, 0.0)
        x_tr_clean = x_tr_raw.fillna(fv).values.astype(float)
        variance = float(np.var(x_tr_clean)) if len(x_tr_clean) > 0 else 0.0

        # 单特征 train AUC
        train_auc = None
        if not is_removed and len(np.unique(y_tr)) >= 2:
            try:
                train_auc = float(roc_auc_score(y_tr, x_tr_clean))
            except Exception:
                pass

        # FP proxy (train only)
        fp_proxy = _fp_proxy_for_feature(df_train, f)

        # 漂移指标
        drift = {}
        if f in df_valid.columns:
            x_va_raw = pd.to_numeric(df_valid[f], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
            x_va_clean = x_va_raw.fillna(fv).values.astype(float)
            drift = compute_drift_metrics(x_tr_clean, x_va_clean)

        row = {
            "feature": f,
            "group": group,
            "missing_rate": missing_rate,
            "variance": variance,
            "train_auc": train_auc,
            "fp_proxy_fit": fp_proxy.get("fit"),
            "fp_proxy_sample_fp_rate": fp_proxy.get("sample_fp_rate"),
            "fp_proxy_state_fp_rate": fp_proxy.get("state_fp_rate"),
            "deployment_cost": profile["deployment_cost"],
            "is_scale_dependent": int(profile["deployment_scale_dependent"]),
            "ks_stat": drift.get("ks_stat"),
            "ks_pvalue": drift.get("ks_pvalue"),
            "psi": drift.get("psi"),
            "mean_shift": drift.get("mean_shift"),
            "std_ratio": drift.get("std_ratio"),
            "removed": int(is_removed),
            "removed_reason": removed_reason,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 候选特征子集搜索
# ─────────────────────────────────────────────────────────────

def _commercial_8_feature_names():
    """返回商业8特征名称列表（从 FEATURE_GROUPS 中获取）。"""
    return list(FEATURE_GROUPS.get("commercial_baseline", [
        "GREEN_CORR", "GREEN_AC", "AMB_AC", "ACC_YSUM",
        "GREEN_DC", "AMB_DC", "GREEN_XCORR", "FFT_PEAK_MEDIAN_RATIO",
    ]))


def _select_top_n_by_group(sorted_items, n, group_limits=None):
    """从排序后的特征列表中按组限制选取 top-n。"""
    if group_limits is None:
        group_limits = GROUP_LIMITS_DEFAULT
    selected = []
    group_count = defaultdict(int)
    for item in sorted_items:
        f = item["feature"]
        g = item.get("group", feature_to_group(f))
        limit = group_limits.get(g, 2)
        if limit <= 0:
            continue
        if group_count[g] < limit:
            selected.append(f)
            group_count[g] += 1
        if len(selected) >= n:
            break
    return selected, dict(group_count)


# XGBoost 固定配置（与 s05 一致，用于公平对比）
SUBSET_SEARCH_XGB_CONFIG = {
    "n_estimators": 40,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_lambda": 10,
    "reg_alpha": 1,
    "scale_pos_weight": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}


def generate_feature_subset_candidates(combined_summary, kept_features, max_features=15):
    """
    生成 6 组候选特征子集。

    每组通过 _select_top_n_by_group() 恪守 GROUP_LIMITS_DEFAULT。

    返回 OrderedDict: {subset_name: {"features": [...], "description": "..."}}
    """
    candidates = OrderedDict()

    # 为每个 combined_summary 条目确保 group 字段存在
    enriched = []
    for item in combined_summary:
        e = dict(item)
        if "group" not in e:
            e["group"] = feature_to_group(e["feature"])
        if "deployment_cost" not in e:
            e["deployment_cost"] = DEPLOYMENT_COST_BY_GROUP.get(e["group"], 2.0)
        if "fp_proxy_sample_fp_rate" not in e:
            e["fp_proxy_sample_fp_rate"] = 0.5
        enriched.append(e)

    # 1. accuracy_first_15: 按 combined_score 降序
    acc_sorted = sorted(enriched, key=lambda x: x.get("combined_score", 0), reverse=True)
    acc_feats, _ = _select_top_n_by_group(acc_sorted, max_features)
    candidates["accuracy_first_15"] = {
        "features": acc_feats,
        "description": "默认 s04 combined_score 排序（permutation + SHAP + deployment + FP proxy）",
    }

    # 2. fp_safe_15: 按 fp_proxy_sample_fp_rate 升序
    fp_sorted = sorted(enriched,
                       key=lambda x: (x.get("fp_proxy_sample_fp_rate", 1.0),
                                      -x.get("combined_score", 0)))
    fp_feats, _ = _select_top_n_by_group(fp_sorted, max_features)
    candidates["fp_safe_15"] = {
        "features": fp_feats,
        "description": "优先低 FP proxy 样本误报率",
    }

    # 3. deployment_light_15: 按 deployment_cost 升序
    deploy_sorted = sorted(enriched,
                           key=lambda x: (x.get("deployment_cost", 10.0),
                                          -x.get("combined_score", 0)))
    deploy_feats, _ = _select_top_n_by_group(deploy_sorted, max_features)
    candidates["deployment_light_15"] = {
        "features": deploy_feats,
        "description": "优先低部署成本",
    }

    # 4. balanced_15: 准确率 + FP安全 + 成本 加权
    max_cost = max(DEPLOYMENT_COST_BY_GROUP.values()) if DEPLOYMENT_COST_BY_GROUP else 4.0
    balanced_items = []
    for item in enriched:
        cs = item.get("combined_score", 0)
        fp_rate = item.get("fp_proxy_sample_fp_rate", 0.5)
        cost = item.get("deployment_cost", 2.0)
        balanced_score = 0.4 * cs + 0.3 * (1.0 - fp_rate) + 0.3 * (1.0 - cost / max_cost)
        balanced_items.append({**item, "_balanced_score": balanced_score})
    balanced_sorted = sorted(balanced_items, key=lambda x: x["_balanced_score"], reverse=True)
    balanced_feats, _ = _select_top_n_by_group(balanced_sorted, max_features)
    candidates["balanced_15"] = {
        "features": balanced_feats,
        "description": "加权: 0.4×accuracy + 0.3×(1-FP) + 0.3×(1-cost/max)",
    }

    focus_groups = [
        "green_stats", "green_spatial", "green_3ch_consistency",
        "frequency", "short_window_stability", "amb_cross",
        "acc_features", "acc_per_axis", "acc_orientation",
    ]
    by_group = defaultdict(list)
    for item in acc_sorted:
        by_group[item.get("group", feature_to_group(item["feature"]))].append(item)
    beam_specs = [
        ("accuracy_beam_green_top2", ["green_stats", "frequency", "green_3ch_consistency", "green_spatial", "acc_features", "amb_cross"]),
        ("accuracy_beam_stability", ["short_window_stability", "green_spatial", "green_3ch_consistency", "green_stats", "frequency", "acc_features"]),
        ("accuracy_beam_motion_light", ["acc_features", "acc_per_axis", "acc_orientation", "green_stats", "frequency", "amb_cross"]),
    ]
    for name, preferred_groups in beam_specs:
        ordered = []
        seen_features = set()
        for group in preferred_groups + [g for g in focus_groups if g not in preferred_groups]:
            for item in by_group.get(group, [])[:3]:
                if item["feature"] not in seen_features:
                    ordered.append(item)
                    seen_features.add(item["feature"])
        for item in acc_sorted:
            if item["feature"] not in seen_features:
                ordered.append(item)
                seen_features.add(item["feature"])
        feats, _ = _select_top_n_by_group(ordered, max_features)
        if feats:
            candidates[name] = {
                "features": feats,
                "description": "accuracy-first beam candidate with group-diverse deployment-friendly features",
            }

    # 5. balanced_12: 同 balanced，12 特征
    balanced_12_feats, _ = _select_top_n_by_group(balanced_sorted, 12)
    candidates["balanced_12"] = {
        "features": balanced_12_feats,
        "description": "Balanced 策略压缩到 12 特征",
    }

    # 6. commercial_8_baseline: 硬编码商业8特征
    comm_8_names = _commercial_8_feature_names()
    comm_8 = [f for f in comm_8_names if f in kept_features]
    candidates["commercial_8_baseline"] = {
        "features": comm_8,
        "description": "商业 8 特征 baseline（参照组）",
    }

    return candidates


def evaluate_feature_subsets(df_train, df_valid, candidates):
    """
    用固定 XGBoost 配置训练并评估每组候选特征（仅在 valid 上）。

    所有候选使用 SUBSET_SEARCH_XGB_CONFIG（与 s05 一致）。
    绝不读取 test split。

    返回 DataFrame。
    """
    y_tr = df_train["target"].values.astype(int)
    y_va = df_valid["target"].values.astype(int)

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        print("[WARN] evaluate_feature_subsets: train 或 valid 只有单类别，跳过")
        return pd.DataFrame()

    results = []
    for name, info in candidates.items():
        feats = info["features"]
        if len(feats) == 0:
            print(f"  [{name}] 无有效特征，跳过")
            continue

        # 填充值：从 train 中位数计算
        fvs = {}
        for f in feats:
            if f in df_train.columns:
                x_raw = pd.to_numeric(df_train[f], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan)
                fvs[f] = float(x_raw.median()) if x_raw.notna().any() else 0.0
            else:
                fvs[f] = 0.0

        # 构建 X
        col_list = []
        for f in feats:
            col = pd.to_numeric(df_train[f], errors="coerce").replace(
                [np.inf, -np.inf], np.nan).fillna(fvs[f]).values.astype(float)
            col_list.append(col)
        X_tr = np.column_stack(col_list)

        va_cols = []
        for f in feats:
            if f in df_valid.columns:
                col = pd.to_numeric(df_valid[f], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan).fillna(fvs[f]).values.astype(float)
            else:
                col = np.full(len(df_valid), fvs[f], dtype=float)
            va_cols.append(col)
        X_va = np.column_stack(va_cols)

        # 训练
        model = xgb.XGBClassifier(**SUBSET_SEARCH_XGB_CONFIG)
        try:
            model.fit(X_tr, y_tr, verbose=False)
        except Exception as e:
            print(f"  [{name}] 训练失败: {e}")
            continue

        # 评估（阈值 0.5，公平对比）
        p = model.predict_proba(X_va)[:, 1]
        pred = (p >= 0.5).astype(int)

        acc = float(accuracy_score(y_va, pred))
        prec = float(precision_score(y_va, pred, zero_division=0))
        rec = float(recall_score(y_va, pred, zero_division=0))
        f1 = float(f1_score(y_va, pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) >= 2 else None
        except Exception:
            auc = None

        tn, fp, fn, tp = confusion_matrix(y_va, pred, labels=[0, 1]).ravel()

        # 组多样性
        group_count = defaultdict(int)
        for f in feats:
            group_count[feature_to_group(f)] += 1

        deploy_costs = [DEPLOYMENT_COST_BY_GROUP.get(feature_to_group(f), 2.0)
                        for f in feats]
        scale_dep = sum(1 for f in feats if f in SCALE_DEPENDENT_FEATURES)

        denom = max(tn + fp, 1)
        fp_rate = fp / denom
        score = acc + 0.1 * rec - 0.1 * fp_rate - 0.01 * np.mean(deploy_costs)

        results.append({
            "subset_name": name,
            "feature_names": ",".join(feats),
            "n_features": len(feats),
            "group_count": len(group_count),
            "deployment_cost_mean": float(np.mean(deploy_costs)),
            "scale_dependent_count": int(scale_dep),
            "valid_accuracy": acc,
            "valid_precision": prec,
            "valid_recall": rec,
            "valid_f1": f1,
            "valid_auc": auc,
            "valid_fp": int(fp),
            "valid_fn": int(fn),
            "score": score,
        })
        print(f"  [{name}] n={len(feats)}, acc={acc:.4f}, prec={prec:.4f}, "
              f"rec={rec:.4f}, f1={f1:.4f}, auc={auc if auc else 'N/A'}, "
              f"fp={fp}, fn={fn}")

    return pd.DataFrame(results)


def run_feature_subset_search(df_train, df_valid, feature_cols, combined_summary,
                               output_dir, max_features=15):
    """
    完整特征子集搜索流程。

    输出文件（全部在 output_dir 下）：
    - subset_candidates.csv      候选描述
    - subset_eval_valid.csv      评估结果
    - subset_eval_summary.json   汇总 + 最佳候选
    - best_subset_features.json  最佳特征列表

    返回:
        dict: {"name": best_name, "features": [...], "valid_accuracy": float}
        若无有效结果则返回 None
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成候选
    candidates = generate_feature_subset_candidates(
        combined_summary, feature_cols, max_features
    )

    # 保存候选描述
    cand_records = []
    for name, info in candidates.items():
        cand_records.append({
            "subset_name": name,
            "n_features": len(info["features"]),
            "features": info["features"],
            "description": info["description"],
        })
    cand_df = pd.DataFrame(cand_records)
    cand_path = os.path.join(output_dir, "subset_candidates.csv")
    cand_df.to_csv(cand_path, index=False)
    print(f"[s04] 候选描述 -> {cand_path}")

    # 2. 评估
    eval_df = evaluate_feature_subsets(df_train, df_valid, candidates)
    if len(eval_df) == 0:
        print("[WARN] 所有候选评估失败")
        return None

    eval_path = os.path.join(output_dir, "subset_eval_valid.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"[s04] 评估结果 -> {eval_path}")

    # 3. 选择最佳（valid accuracy 最高）
    eval_sorted = eval_df.sort_values("valid_accuracy", ascending=False)
    best_row = eval_sorted.iloc[0]
    best_name = best_row["subset_name"]
    best_features = candidates[best_name]["features"]

    best_out = {
        "best_subset_name": best_name,
        "best_subset_method": "max_valid_accuracy",
        "n_features": len(best_features),
        "features": best_features,
        "group_count": {g: sum(1 for f in best_features
                               if feature_to_group(f) == g)
                        for g in set(feature_to_group(f) for f in best_features)},
        "evaluated_on": "valid_only",
        "test_not_used": True,
    }
    best_path = os.path.join(output_dir, "best_subset_features.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_out, f, indent=2, ensure_ascii=False)
    print(f"[s04] 最佳子集 -> {best_path}")

    # 汇总
    summary_out = {
        "best_subset_name": best_name,
        "best_valid_accuracy": float(best_row["valid_accuracy"]),
        "all_results": eval_df.to_dict(orient="records"),
        "note": "所有候选仅在 valid split 上评估，未使用 test。",
    }
    summary_path = os.path.join(output_dir, "subset_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    print(f"[s04] 搜索汇总 -> {summary_path}")

    return {
        "name": best_name,
        "features": best_features,
        "valid_accuracy": float(best_row["valid_accuracy"]),
    }



def _select_by_group_impl(summary, max_features=10, group_limits=None, min_acc_features=1):
    """summary 中只要每项含 'feature' 和 'group' 即可。"""
    if group_limits is None:
        group_limits = GROUP_LIMITS_DEFAULT

    selected = []
    group_count = defaultdict(int)

    for item in summary:
        f = item["feature"]
        g = item["group"]
        limit = group_limits.get(g, 1)
        if limit <= 0:
            continue
        if group_count[g] < limit:
            selected.append(f)
            group_count[g] += 1
        if len(selected) >= max_features:
            break

    acc_selected = [f for f in selected if f in FEATURE_GROUPS["acc_features"]]
    if len(acc_selected) < min_acc_features and min_acc_features > 0:
        acc_candidates = [item["feature"] for item in summary if item["group"] == "acc_features"]
        for f in acc_candidates:
            if f not in selected:
                if len(selected) >= max_features:
                    removed = selected.pop()
                    removed_group = feature_to_group(removed)
                    if removed_group in group_count:
                        group_count[removed_group] = max(0, group_count[removed_group] - 1)
                selected.append(f)
                group_count["acc_features"] = group_count.get("acc_features", 0) + 1
                break

    return selected, dict(group_count)


def select_by_group_from_combined(summary, max_features=10, group_limits=None, min_acc_features=1):
    return _select_by_group_impl(summary, max_features, group_limits, min_acc_features)


# =========================================================
# valid 摘要
# =========================================================

def summarize_valid_selected(df_valid, selected_features):
    out = {}
    for f in selected_features:
        if f not in df_valid.columns:
            out[f] = {"exists": False}
            continue
        x = df_valid[f].replace([np.inf, -np.inf], np.nan)
        out[f] = {
            "exists": True,
            "missing_rate": float(x.isna().mean()),
            "mean": float(x.mean()) if x.notna().any() else None,
            "std": float(x.std()) if x.notna().any() else None,
        }
    return out


def _plot_label(name, max_len=34):
    text = str(name)
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def export_feature_selection_report_plot(result, artifact_dir):
    """Export a report-style PNG for feature ranking and FP proxy risk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip s04 plot: {e}")
        return None

    out_dir = os.path.join(str(artifact_dir), "report_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s04_feature_selection_report.png")

    summary = list(result.get("combined_summary", []))[:20]
    selected = set(result.get("selected_features", []))
    if not summary:
        summary = [{"feature": "no features", "deployment_score": 0.0, "combined_score": 0.0, "group": "none"}]

    labels = [_plot_label(x.get("feature", "")) for x in summary][::-1]
    deploy = [float(x.get("deployment_score", 0.0)) for x in summary][::-1]
    combined = [float(x.get("combined_score", 0.0)) for x in summary][::-1]
    colors = ["#2f6f73" if x.get("feature") in selected else "#9aa6ac" for x in summary][::-1]

    group_count = result.get("group_count", {}) or {}
    fp_sample = [float(x.get("fp_proxy_sample_fp_rate", 0.0)) for x in summary[:12]][::-1]
    fp_state = [float(x.get("fp_proxy_state_fp_rate", 0.0)) for x in summary[:12]][::-1]
    fp_labels = [_plot_label(x.get("feature", ""), 28) for x in summary[:12]][::-1]

    fig = plt.figure(figsize=(15, 9), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.9])
    ax_rank = fig.add_subplot(gs[:, 0])
    ax_group = fig.add_subplot(gs[0, 1])
    ax_fp = fig.add_subplot(gs[1, 1])

    y = np.arange(len(labels))
    ax_rank.barh(y, deploy, color=colors, height=0.72, label="deployment score")
    ax_rank.plot(combined, y, color="#d35f2d", linewidth=2, marker="o", markersize=3, label="raw combined")
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(labels, fontsize=8)
    ax_rank.set_xlabel("score")
    ax_rank.set_title("Top Feature Ranking")
    ax_rank.grid(axis="x", alpha=0.18)
    ax_rank.legend(loc="lower right", frameon=False)

    if group_count:
        g_names = list(group_count.keys())
        g_vals = [group_count[k] for k in g_names]
        ax_group.bar(g_names, g_vals, color="#4c78a8")
        ax_group.set_title("Selected Feature Groups")
        ax_group.set_ylabel("count")
        ax_group.tick_params(axis="x", rotation=35, labelsize=8)
        ax_group.grid(axis="y", alpha=0.18)
    else:
        ax_group.text(0.5, 0.5, "No group counts", ha="center", va="center")
        ax_group.set_axis_off()

    yy = np.arange(len(fp_labels))
    ax_fp.barh(yy - 0.18, fp_sample, height=0.34, color="#c44e52", label="sample FP proxy")
    ax_fp.barh(yy + 0.18, fp_state, height=0.34, color="#8172b2", label="state FP proxy")
    ax_fp.set_yticks(yy)
    ax_fp.set_yticklabels(fp_labels, fontsize=8)
    ax_fp.set_xlim(0, 1)
    ax_fp.set_title("False Positive Proxy Risk")
    ax_fp.grid(axis="x", alpha=0.18)
    ax_fp.legend(frameon=False)

    fig.suptitle("Feature Selection Report", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] s04 report plot -> {out_path}")
    return out_path


# =========================================================
# main
# =========================================================

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--max_features", type=int, default=15)
    parser.add_argument("--missing_thresh", type=float, default=0.3)
    parser.add_argument("--var_thresh", type=float, default=1e-8)
    parser.add_argument("--corr_thresh", type=float, default=0.90)
    parser.add_argument("--min_fold_auc", type=float, default=0.55,
                        help="稳定性选择中丢弃 AUC 低于此值的 fold（默认 0.55）")
    parser.add_argument("--skip_vif", action="store_true",
                        help="跳过 VIF 步骤（相关性+方差已去除大部分冗余）")
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")
    parser.add_argument("--deployment_score_weight", type=float, default=0.25,
                        help="部署导向重排权重。0=保持原始重要性排序，建议 0.2-0.35。")
    parser.add_argument("--fp_cost_weight", type=float, default=0.25,
                        help="sample/state-machine FP cost proxy reranking weight.")
    parser.add_argument("--fp_proxy_recall_floor", type=float, default=0.95,
                        help="Positive-window recall floor used by the train-only FP proxy.")
    parser.add_argument("--fp_proxy_state_k_on", type=int, default=3,
                        help="Consecutive windows needed to count a state-machine FP proxy hit.")
    parser.add_argument("--ranking_objective", type=str, default="balanced",
                        choices=["balanced", "window_accuracy"],
                        help="Feature ranking/selection objective. window_accuracy relaxes group caps for Stage2 window accuracy.")
    parser.add_argument("--run_subset_search", action="store_true",
                        help="运行候选特征子集搜索，结果覆写 selected_features.json")
    parser.add_argument("--subset_search_max_features", type=int, default=15,
                        help="子集搜索中每个候选的最大特征数")

    if args is None:
        args = parser.parse_args()

    df_train = pd.read_csv(os.path.join(args.artifact_dir, "feature_pool_train.csv"))
    df_valid = pd.read_csv(os.path.join(args.artifact_dir, "feature_pool_valid.csv"))

    feature_cols = get_feature_cols(df_train)

    # Stage2 candidates are ambient/green/ACC only; remove stale IR-derived
    # columns from old feature_pool CSVs before selection.
    _before = len(feature_cols)
    feature_cols = filter_stage2_ir_features(feature_cols)
    _dropped = _before - len(feature_cols)
    if _dropped > 0:
        print(f"[s04 IR strip] 从候选池移除 {_dropped} 个 IR 特征 (保留 {len(feature_cols)} 个)")
    _deploy_before = len(feature_cols)
    feature_cols = filter_features_for_deployment(feature_cols)
    _deploy_dropped = _deploy_before - len(feature_cols)
    if _deploy_dropped > 0:
        print(
            f"[s04 deployment filter] 移除 {_deploy_dropped} 个不适合端侧部署的复杂特征 "
            f"(保留 {len(feature_cols)} 个)"
        )
    if not feature_cols:
        raise ValueError(
            "deployment-friendly feature filter left no Stage2 candidate features; rerun s03."
        )

    print_s04_workload_estimate(df_train, df_valid, feature_cols, args)

    step_start = time.perf_counter()
    print("\n[s04] STEP 1/5: 数据清洗/VIF 开始...")
    sys.stdout.flush()
    df_train_clean, df_valid_clean, kept_features, removed, fill_values = clean_features_by_train(
        df_train, df_valid, feature_cols,
        missing_thresh=args.missing_thresh,
        var_thresh=args.var_thresh,
        corr_thresh=args.corr_thresh,
        skip_vif=args.skip_vif,
    )

    print(f"[s04] STEP 1/5: 数据清洗完成. 原始={len(feature_cols)}, 清洗后={len(kept_features)}, "
          f"elapsed={_elapsed(step_start):.1f}s")
    sys.stdout.flush()

    step_start = time.perf_counter()
    print("\n[s04] STEP 2/5: 组内快速预筛...")
    sys.stdout.flush()
    preselected = fast_group_preselection(df_train_clean, kept_features, preselect_top=4)
    preselected_features = list(preselected.keys())
    print(f"预选后特征数: {len(preselected_features)}")
    for f, info in sorted(preselected.items(), key=lambda x: x[1]['importance'], reverse=True):
        print(f"  {f}: {info['method']}, importance={info['importance']:.2f}, group={info['group']}")
    print(f"[s04] STEP 2/5: 组内快速预筛完成. elapsed={_elapsed(step_start):.1f}s")
    sys.stdout.flush()

    step_start = time.perf_counter()
    print(f"\n[s04] STEP 3/5: 稳定性选择 (permutation importance, {args.n_workers} workers)...")
    sys.stdout.flush()
    summary = stability_selection(
        df_train_clean, preselected_features,
        max_splits=5, n_workers=args.n_workers,
        min_fold_auc=args.min_fold_auc,
    )
    print("\n稳定性排序 Top20:")
    for i, item in enumerate(summary[:20]):
        print(f"{i+1:02d}. {item['feature']} | group={item['group']} | "
              f"freq={item['freq']:.3f} | imp={item['avg_importance']:.6f} | "
              f"rank={item['avg_rank']:.2f}")
    print(f"[s04] STEP 3/5: 稳定性选择完成. elapsed={_elapsed(step_start):.1f}s")
    sys.stdout.flush()

    step_start = time.perf_counter()
    print("\n[s04] STEP 4/5: SHAP 二次确认 (可能较慢)...")
    sys.stdout.flush()
    shap_imp = {}
    if SHAP_AVAILABLE:
        shap_imp = shap_importance(df_train_clean, preselected_features)
        print("SHAP Top10:")
        for i, (f, v) in enumerate(sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {i+1}. {f}: {v:.6f}")
    else:
        print("SHAP不可用，跳过")
    print(f"[s04] STEP 4/5: SHAP 二次确认完成. elapsed={_elapsed(step_start):.1f}s")
    sys.stdout.flush()

    step_start = time.perf_counter()
    print("\n[s04] STEP 5/5: 综合排序 (复用 STEP3 + STEP4 结果)...")
    sys.stdout.flush()
    # 直接复用已有的 stability_selection 结果和 SHAP 结果，不再重跑
    combined = []
    for item in summary:
        f = item["feature"]
        sv = shap_imp.get(f, 0.0)
        combined.append({**item, "shap_imp": sv})
    perm_scores = np.array([it["freq"] * it["avg_importance"] for it in combined])
    shap_scores = np.array([shap_imp.get(it["feature"], 0.0) for it in combined])
    pmx, smx = perm_scores.max() or 1.0, shap_scores.max() or 1.0
    perm_n = perm_scores / pmx if pmx > 1e-12 else perm_scores
    shap_n = shap_scores / smx if smx > 1e-12 else shap_scores
    for i, it in enumerate(combined):
        has_shap = it["feature"] in shap_imp and shap_imp[it["feature"]] > 0
        it["combined_score"] = float(0.5 * perm_n[i] + 0.5 * shap_n[i]) if has_shap else float(perm_n[i])
    combined_summary = sorted(combined, key=lambda x: x["combined_score"], reverse=True)
    combined_summary = add_deployment_scores(
        combined_summary,
        deployment_score_weight=args.deployment_score_weight,
    )
    combined_summary = add_fp_cost_proxy_scores(
        combined_summary,
        df_train_clean,
        fp_cost_weight=args.fp_cost_weight,
        recall_floor=args.fp_proxy_recall_floor,
        state_k_on=args.fp_proxy_state_k_on,
    )
    print("综合+部署导向排序 Top20:")
    for i, item in enumerate(combined_summary[:20]):
        shap_str = f", shap={item.get('shap_imp', 0):.6f}" if item.get('shap_imp', 0) > 0 else ""
        print(f"{i+1:02d}. {item['feature']} | group={item['group']} | "
              f"freq={item.get('freq',0):.3f}, combined={item['combined_score']:.6f}, "
              f"deploy={item['deployment_score']:.6f}, fit={item['deployment_fit']:.3f}, "
              f"sampleFP={item.get('fp_proxy_sample_fp_rate',0):.3f}, "
              f"stateFP={item.get('fp_proxy_state_fp_rate',0):.3f}{shap_str}")

    group_limits = group_limits_for_ranking_objective(args.ranking_objective)
    selected, group_count = select_by_group_from_combined(
        combined_summary,
        max_features=args.max_features,
        group_limits=group_limits,
    )
    selected = filter_features_for_deployment(selected)
    if len(selected) < min(args.max_features, len(combined_summary)):
        selected, group_count = select_by_group_from_combined(
            [
                item for item in combined_summary
                if item["feature"] in set(filter_features_for_deployment([item["feature"]]))
            ],
            max_features=args.max_features,
            group_limits=group_limits,
        )

    print("\n最终选择特征:")
    for i, f in enumerate(selected):
        shap_v = shap_imp.get(f, 0.0)
        shap_str = f", shap={shap_v:.4f}" if shap_v > 0 else ""
        print(f"{i+1}. {f} | group={feature_to_group(f)}{shap_str}")

    valid_summary = summarize_valid_selected(df_valid_clean, selected)

    # train vs valid SHAP 一致性，挑出 train-only 可疑特征
    sub_start = time.perf_counter()
    print("\n【SHAP train vs valid 一致性检查】")
    sys.stdout.flush()
    shap_check = shap_consistency_check(df_train_clean, df_valid_clean, preselected_features)
    if shap_check.get("available"):
        print(f"  Spearman 相关={shap_check['spearman_rho']}, "
              f"Top{shap_check['k']} 重合度={shap_check['topk_overlap']:.2f}")
        if shap_check["suspicious_train_only"]:
            print(f"  [WARN] train 排名高但 valid 不高的可疑特征 ({len(shap_check['suspicious_train_only'])}):")
            for s in shap_check["suspicious_train_only"][:5]:
                print(f"    {s['feature']}: train rank={s['rank_train']}, "
                      f"valid rank={s['rank_valid']}")
    else:
        print(f"  跳过：{shap_check.get('reason', 'shap unavailable')}")
    print(f"[s04] SHAP 一致性检查完成. elapsed={_elapsed(sub_start):.1f}s")
    sys.stdout.flush()

    # 选中特征里的尺度依赖标记
    scale_dep_in_sel, scale_inv_in_sel = annotate_scale_dependency(selected)
    if scale_dep_in_sel:
        print(f"\n[WARN] 选中特征中含 {len(scale_dep_in_sel)} 个绝对量纲特征"
              f"（随肤色/传感器漂移，部署前考虑加基线自适应）:")
        for f in scale_dep_in_sel:
            print(f"    {f}")

    # =========================================================
    # 特征诊断表导出（始终输出）
    # =========================================================
    sub_start = time.perf_counter()
    print("\n[s04] 诊断表导出开始...")
    sys.stdout.flush()
    try:
        diag_df = compute_all_feature_diagnostics(
            df_train_clean, df_valid_clean, feature_cols,
            fill_values=fill_values,
            kept_features=kept_features,
            removed_map=removed,
        )
        diag_path = os.path.join(args.artifact_dir, "feature_diagnostics.csv")
        diag_df.to_csv(diag_path, index=False)
        print(f"\n[s04] 特征诊断表 -> {diag_path}")
        # 标记三类风险特征
        high_drift = diag_df[
            diag_df["psi"].notna() & (diag_df["psi"] > 0.25)
        ].sort_values("psi", ascending=False)
        if len(high_drift) > 0:
            print(f"  [WARN] {len(high_drift)} 个特征 PSI > 0.25（train/valid 漂移显著）:")
            for _, row in high_drift.head(5).iterrows():
                print(f"    {row['feature']}: PSI={row['psi']:.3f}, KS={row['ks_stat']:.3f}")
        high_fp = diag_df[
            diag_df["fp_proxy_sample_fp_rate"].notna()
            & (diag_df["fp_proxy_sample_fp_rate"] > 0.3)
        ].sort_values("fp_proxy_sample_fp_rate", ascending=False)
        if len(high_fp) > 0:
            print(f"  [WARN] {len(high_fp)} 个特征 FP proxy > 0.3（易造成非佩戴误触）:")
            for _, row in high_fp.head(5).iterrows():
                print(f"    {row['feature']}: fp_sample_rate={row['fp_proxy_sample_fp_rate']:.3f}")
    except Exception as e:
        print(f"  [WARN] 特征诊断表导出失败: {e}")
    print(f"[s04] 诊断表导出完成. elapsed={_elapsed(sub_start):.1f}s")
    sys.stdout.flush()

    # =========================================================
    # 候选特征子集搜索（仅当 --run_subset_search 时）
    # =========================================================
    if args.run_subset_search:
        print("\n" + "=" * 80)
        print("[s04] EXTRA: 候选特征子集搜索")
        print("=" * 80)
        search_dir = os.path.join(args.artifact_dir, "feature_subset_search")
        try:
            best_info = run_feature_subset_search(
                df_train_clean, df_valid_clean, kept_features,
                combined_summary, search_dir,
                max_features=args.subset_search_max_features,
            )
            if best_info and best_info.get("features"):
                selected = best_info["features"]
                selected = filter_features_for_deployment(selected)
                group_count = {}
                for f in selected:
                    g = feature_to_group(f)
                    group_count[g] = group_count.get(g, 0) + 1
                print(f"\n[s04] 子集搜索完成，最佳候选 '{best_info['name']}' "
                      f"valid_acc={best_info.get('valid_accuracy', 0):.4f}")
                print(f"[s04] 已用最佳候选覆写 selected_features ({len(selected)} 特征)")
                # 重新标注尺度依赖
                scale_dep_in_sel, scale_inv_in_sel = annotate_scale_dependency(selected)
        except Exception as e:
            print(f"[WARN] 子集搜索失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"[s04] STEP 5/5: 综合排序与输出准备完成. elapsed={_elapsed(step_start):.1f}s")
    sys.stdout.flush()

    result = {
        "selected_features": selected,
        "max_features": args.max_features,
        "selection_policy": {
            "selection_data": "train_only",
            "valid_used_for_selection": False,
            "test_used_for_selection": False,
            "group_kfold_group": "sample_name",
            "use_shap": SHAP_AVAILABLE,
            "use_permutation": True,
            "deployment_score_weight": float(args.deployment_score_weight),
            "fp_cost_weight": float(args.fp_cost_weight),
            "fp_proxy_recall_floor": float(args.fp_proxy_recall_floor),
            "fp_proxy_state_k_on": int(args.fp_proxy_state_k_on),
            "ranking_objective": str(args.ranking_objective),
            "ranking_target": "train_only_importance + deployment_proxy_fit + train_only_sample_state_fp_proxy",
        },
        "permutation_summary": summary,
        "shap_importance": shap_imp if SHAP_AVAILABLE else {},
        "shap_consistency": shap_check,
        "combined_summary": combined_summary,
        "removed_features": removed,
        "group_count": group_count,
        "group_limits": group_limits,
        "train_fill_values": fill_values,
        "valid_selected_feature_summary": valid_summary,
        "selected_deployment_summary": {
            f: deployment_feature_summary(f) for f in selected
        },
        "deployment_feature_cost_summary": summarize_deployment_feature_costs(selected),
        "scale_dependency": {
            "scale_dependent_selected": scale_dep_in_sel,
            "scale_invariant_selected": scale_inv_in_sel,
            "ratio_invariant": (len(scale_inv_in_sel) / max(len(selected), 1)),
            "note": "scale_dependent 特征依赖原始 ADC 数值，建议线上做基线自适应或换成 ratio。",
        },
    }

    out_path = os.path.join(args.artifact_dir, "selected_features.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n特征选择结果已保存: {out_path}")

    # 输出完整排序列表（供 s05 搜参时测试不同 max_features）
    ranked = sorted(combined_summary, key=lambda x: x["combined_score"], reverse=True)
    ranked = [
        r for r in ranked
        if r["feature"] in set(filter_features_for_deployment([r["feature"]]))
    ]
    ranked_path = os.path.join(args.artifact_dir, "ranked_features.json")
    with open(ranked_path, "w", encoding="utf-8") as f:
        json.dump([{
            "feature": r["feature"],
            "group": r["group"],
            "combined_score": r["combined_score"],
            "freq": r.get("freq", 0),
            "avg_importance": r.get("avg_importance", 0),
            "shap_imp": r.get("shap_imp", 0.0),
        } for r in ranked], f, indent=2, ensure_ascii=False)
    print(f"特征排序列表已保存: {ranked_path}")


    export_feature_selection_report_plot(result, args.artifact_dir)

    # Export SHAP importance plot when SHAP is available
    if SHAP_AVAILABLE:
        shap_train = result.get("shap_importance", {}) or {}
        shap_check = result.get("shap_consistency", {}) or {}
        if shap_train and shap_check.get("available"):
            try:
                export_shap_importance_plot(
                    shap_train=shap_train,
                    shap_valid=shap_check.get("shap_valid", {}),
                    spearman_rho=shap_check.get("spearman_rho"),
                    topk_overlap=shap_check.get("topk_overlap", 0.0),
                    suspicious=shap_check.get("suspicious_train_only", []),
                    selected_features=result.get("selected_features", []),
                    artifact_dir=args.artifact_dir,
                )
            except Exception as e:
                print(f"[WARN] SHAP plot export failed: {e}")
    else:
        print("(s04 SHAP plot: shap unavailable, skipped)")


# =========================================================
# Feature embedding report utilities (formerly s11).
# =========================================================
FEATURE_POOL_FILES = {
    "train": "feature_pool_train.csv",
    "valid": "feature_pool_valid.csv",
    "test": "feature_pool_test.csv",
}

META_COLUMNS = {
    "sample_name",
    "h5_file",
    "target",
    "start_100hz",
    "start_sec",
    "window_index",
    "mode",
    "split",
}

LABEL_COLORS = {
    0: "#4C78A8",
    1: "#D65F5F",
}

METHOD_TITLES = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "umap": "UMAP",
}


def _normalize_list(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(str(v).strip().lower() for v in values if str(v).strip())


def _parse_csv_list(value: str, cast=str) -> Tuple:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return tuple(cast(p) for p in parts)


def _set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "ps.fonttype": 42,
        }
    )


def load_feature_pools(artifact_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    for split, filename in FEATURE_POOL_FILES.items():
        path = artifact_dir / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(
                f"failed to read {path}: {exc}. "
                "The feature_pool CSV is malformed or partially written; "
                "rerun s03_extract_feature_pool.py for this artifact_dir."
            ) from exc
        if len(df) == 0:
            continue
        df = df.copy()
        df["split"] = split
        frames.append(df)

    if not frames:
        expected = ", ".join(FEATURE_POOL_FILES.values())
        raise FileNotFoundError(f"No non-empty feature pools found in {artifact_dir}; expected {expected}")

    data = pd.concat(frames, ignore_index=True, sort=False)
    if "target" not in data.columns:
        raise ValueError("Feature pools must contain a 'target' column")

    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric_cols if c not in META_COLUMNS]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after excluding metadata columns")

    return data, feature_cols


def _sample_rows(df: pd.DataFrame, max_points: int, random_state: int) -> pd.DataFrame:
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df.copy()

    group_cols = [c for c in ["target", "split"] if c in df.columns]
    if not group_cols:
        return df.sample(n=max_points, random_state=random_state).sort_index().reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    pieces = []
    grouped = list(df.groupby(group_cols, dropna=False, sort=False))
    total = len(df)
    remaining = max_points
    for idx, (_, group) in enumerate(grouped):
        if idx == len(grouped) - 1:
            take = min(len(group), remaining)
        else:
            take = max(1, int(round(max_points * len(group) / total)))
            take = min(len(group), take, remaining - (len(grouped) - idx - 1))
        remaining -= take
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        pieces.append(group.sample(n=take, random_state=seed))

    return pd.concat(pieces).sort_index().reset_index(drop=True)


def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    max_points: int,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    sampled = _sample_rows(df, int(max_points), int(random_state))
    raw = sampled.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(raw)
    scaled = StandardScaler().fit_transform(imputed)
    return sampled.reset_index(drop=True), scaled


def _pad_components(values: np.ndarray, dim: int) -> np.ndarray:
    if values.shape[1] >= dim:
        return values[:, :dim]
    padded = np.zeros((values.shape[0], dim), dtype=float)
    padded[:, : values.shape[1]] = values
    return padded


def _pca_axis_labels(explained_variance_ratio: Sequence[float], dims: Sequence[int]) -> List[str]:
    max_dim = max([int(d) for d in dims if int(d) in {2, 3}] or [2])
    labels = []
    for idx in range(max_dim):
        if idx < len(explained_variance_ratio):
            labels.append(f"PC{idx + 1} ({100.0 * float(explained_variance_ratio[idx]):.1f}%)")
        else:
            labels.append(f"PC{idx + 1}")
    return labels


def compute_embeddings(
    x: np.ndarray,
    methods: Sequence[str],
    dims: Sequence[int],
    random_state: int,
    perplexity: float,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Mapping[str, object]]]:
    requested_methods = _normalize_list(methods)
    requested_dims = tuple(sorted({int(d) for d in dims if int(d) in {2, 3}}))
    embeddings: Dict[str, Dict[int, np.ndarray]] = {}
    status: Dict[str, Mapping[str, object]] = {}

    if x.shape[0] < 3:
        raise ValueError("At least 3 windows are required for 2D/3D embedding figures")

    if "pca" in requested_methods:
        n_components = min(3, x.shape[0], x.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        coords = pca.fit_transform(x)
        embeddings["pca"] = {dim: _pad_components(coords, dim) for dim in requested_dims}
        explained = [float(v) for v in pca.explained_variance_ratio_]
        status["pca"] = {
            "status": "ok",
            "explained_variance_ratio": explained,
            "axis_labels": _pca_axis_labels(explained, requested_dims),
        }

    if "tsne" in requested_methods:
        if x.shape[0] < 5:
            status["tsne"] = {"status": "skipped", "reason": "need at least 5 windows for stable t-SNE"}
        else:
            method_embeddings = {}
            effective_perplexity = min(float(perplexity), max(2.0, (x.shape[0] - 1) / 3.0))
            for dim in requested_dims:
                init = "pca" if x.shape[1] >= dim else "random"
                model = TSNE(
                    n_components=dim,
                    init=init,
                    learning_rate="auto",
                    perplexity=effective_perplexity,
                    random_state=random_state,
                    metric="euclidean",
                )
                method_embeddings[dim] = model.fit_transform(x)
            embeddings["tsne"] = method_embeddings
            status["tsne"] = {"status": "ok", "perplexity": float(effective_perplexity)}

    if "umap" in requested_methods:
        try:
            from umap import UMAP  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            status["umap"] = {
                "status": "skipped",
                "reason": f"umap-learn is not installed ({exc.__class__.__name__})",
            }
        else:  # pragma: no cover - optional package is absent in CI by default
            n_neighbors = min(30, max(2, x.shape[0] - 1))
            method_embeddings = {}
            for dim in requested_dims:
                model = UMAP(
                    n_components=dim,
                    n_neighbors=n_neighbors,
                    min_dist=0.12,
                    metric="euclidean",
                    random_state=random_state,
                )
                method_embeddings[dim] = model.fit_transform(x)
            embeddings["umap"] = method_embeddings
            status["umap"] = {"status": "ok", "n_neighbors": int(n_neighbors), "min_dist": 0.12}

    return embeddings, status


def _axis_label(method: str, axis_idx: int, method_info: Optional[Mapping[str, object]] = None) -> str:
    if method == "pca":
        if method_info:
            labels = method_info.get("axis_labels")
            if isinstance(labels, Sequence) and not isinstance(labels, str) and axis_idx < len(labels):
                return str(labels[axis_idx])
        return f"PC{axis_idx + 1}"
    if method == "tsne":
        return f"t-SNE {axis_idx + 1}"
    return f"UMAP {axis_idx + 1}"


def _format_axes_2d(ax, method: str, method_info: Optional[Mapping[str, object]] = None) -> None:
    ax.set_xlabel(_axis_label(method, 0, method_info))
    ax.set_ylabel(_axis_label(method, 1, method_info))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.25, color="#D9D9D9", alpha=0.55)
    ax.set_axisbelow(True)


def _format_axes_3d(ax, method: str, method_info: Optional[Mapping[str, object]] = None) -> None:
    ax.set_xlabel(_axis_label(method, 0, method_info), labelpad=-2)
    ax.set_ylabel(_axis_label(method, 1, method_info), labelpad=-2)
    ax.set_zlabel(_axis_label(method, 2, method_info), labelpad=-2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.25, color="#D9D9D9", alpha=0.45)
    ax.view_init(elev=24, azim=38)


def _plot_points(ax, coords: np.ndarray, labels: np.ndarray, dim: int) -> None:
    unique_labels = sorted(pd.Series(labels).dropna().unique().tolist())
    for label in unique_labels:
        mask = labels == label
        color = LABEL_COLORS.get(int(label), "#6F6F6F") if str(label).lstrip("-").isdigit() else "#6F6F6F"
        label_text = f"label={label}"
        if dim == 2:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=6,
                marker="o",
                c=color,
                edgecolors="white",
                linewidths=0.18,
                alpha=0.76,
                label=label_text,
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                s=5,
                marker="o",
                c=color,
                edgecolors="white",
                linewidths=0.12,
                alpha=0.72,
                depthshade=False,
                label=label_text,
            )


def _save_figure(fig, stem: Path, formats: Sequence[str], dpi: int) -> List[str]:
    paths = []
    out = stem.with_suffix(".png")
    fig.savefig(out, dpi=dpi, facecolor="white")
    paths.append(str(out))
    return paths


def _safe_feature_name(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("._")
    return safe or "feature"


def load_selected_features(artifact_dir: Path, available_features: Sequence[str]) -> Tuple[List[str], Mapping[str, object]]:
    path = artifact_dir / "selected_features.json"
    if not path.exists():
        return [], {"status": "skipped", "reason": "selected_features.json not found"}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        selected = [str(x) for x in payload]
    else:
        selected = [str(x) for x in payload.get("selected_features", [])]

    available = set(available_features)
    present = [name for name in selected if name in available]
    missing = [name for name in selected if name not in available]
    status = {
        "status": "ok" if present else "skipped",
        "n_features": len(present),
        "selected_features": present,
        "missing_features": missing,
    }
    if not present:
        status["reason"] = "selected_features.json did not contain any features present in the feature pools"
    return present, status


def _plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    feature_index: int,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    random_state: int,
) -> List[str]:
    plot_df = df.loc[:, ["target", feature]].copy()
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["target", feature])

    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = sorted(plot_df["target"].dropna().unique().tolist())
        rng = np.random.default_rng(int(random_state) + int(feature_index))
        positions = np.arange(len(labels), dtype=float)
        groups = [plot_df.loc[plot_df["target"] == label, feature].to_numpy(dtype=float) for label in labels]

        violin = ax.violinplot(
            groups,
            positions=positions,
            widths=0.68,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor("#E8E8E8")
            body.set_edgecolor("#A0A0A0")
            body.set_linewidth(0.5)
            body.set_alpha(0.55)

        box = ax.boxplot(
            groups,
            positions=positions,
            widths=0.24,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 0.9},
            boxprops={"facecolor": "white", "edgecolor": "#404040", "linewidth": 0.65},
            whiskerprops={"color": "#404040", "linewidth": 0.6},
            capprops={"color": "#404040", "linewidth": 0.6},
        )
        for patch in box["boxes"]:
            patch.set_alpha(0.82)

        for pos, label, values in zip(positions, labels, groups):
            jitter = rng.normal(0.0, 0.045, size=len(values))
            color = LABEL_COLORS.get(int(label), "#6F6F6F") if str(label).lstrip("-").isdigit() else "#6F6F6F"
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=4,
                marker="o",
                c=color,
                edgecolors="white",
                linewidths=0.12,
                alpha=0.55,
                label=f"label={label}",
                zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([f"label={label}" for label in labels])
        ax.legend(frameon=False, loc="best", handletextpad=0.25, borderpad=0.2)

    ax.set_title(feature)
    ax.set_xlabel("Target label")
    ax.set_ylabel("Feature value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.25, color="#D9D9D9", alpha=0.55)
    ax.set_axisbelow(True)

    stem = out_dir / f"feature_distribution_{feature_index:02d}_{_safe_feature_name(feature)}"
    paths = _save_figure(fig, stem, formats, dpi)
    plt.close(fig)
    return paths


def summarize_selected_feature_distributions(
    df: pd.DataFrame,
    selected_features: Sequence[str],
) -> Dict[str, Mapping[str, object]]:
    stats: Dict[str, Mapping[str, object]] = {}
    if "target" not in df.columns:
        return stats

    y = pd.to_numeric(df["target"], errors="coerce")
    for feature in selected_features:
        values = pd.to_numeric(df.get(feature), errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = pd.DataFrame({"target": y, "value": values}).dropna()
        if valid.empty:
            stats[feature] = {"status": "empty"}
            continue
        labels = sorted(valid["target"].astype(int).unique().tolist())
        row: Dict[str, object] = {
            "status": "ok",
            "n": int(len(valid)),
            "labels": labels,
        }
        for label in labels:
            group = valid.loc[valid["target"].astype(int) == int(label), "value"].to_numpy(dtype=float)
            row[f"label_{label}_n"] = int(len(group))
            row[f"label_{label}_mean"] = float(np.mean(group)) if len(group) else None
            row[f"label_{label}_median"] = float(np.median(group)) if len(group) else None
        if len(labels) == 2:
            lo, hi = labels[0], labels[1]
            lo_values = valid.loc[valid["target"].astype(int) == int(lo), "value"].to_numpy(dtype=float)
            hi_values = valid.loc[valid["target"].astype(int) == int(hi), "value"].to_numpy(dtype=float)
            if len(lo_values) and len(hi_values):
                row["mean_diff_label_high_minus_low"] = float(np.mean(hi_values) - np.mean(lo_values))
                row["median_diff_label_high_minus_low"] = float(np.median(hi_values) - np.median(lo_values))
                try:
                    auc = float(roc_auc_score(valid["target"].astype(int), valid["value"].to_numpy(dtype=float)))
                    row["auc"] = auc
                    row["auc_separation"] = float(max(auc, 1.0 - auc))
                except Exception:
                    row["auc"] = None
                    row["auc_separation"] = None
        stats[feature] = row
    return stats


def _write_selected_feature_distribution_source(
    out_dir: Path,
    df: pd.DataFrame,
    selected_features: Sequence[str],
) -> Optional[Path]:
    if not selected_features:
        return None
    columns = [c for c in ["split", "sample_name", "h5_file", "target", "start_sec", "window_index", "mode"] if c in df.columns]
    source = df.loc[:, columns + list(selected_features)].copy()
    path = out_dir / "selected_feature_distribution_source_data.csv"
    source.to_csv(path, index=False)
    return path


def plot_selected_feature_distributions(
    out_dir: Path,
    df: pd.DataFrame,
    selected_features: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    random_state: int,
) -> Tuple[Dict[str, List[str]], Optional[Path]]:
    figure_paths: Dict[str, List[str]] = {}
    for idx, feature in enumerate(selected_features, start=1):
        key = f"feature_distribution_{idx:02d}_{_safe_feature_name(feature)}"
        figure_paths[key] = _plot_feature_distribution(
            df=df,
            feature=feature,
            feature_index=idx,
            out_dir=out_dir,
            formats=formats,
            dpi=dpi,
            random_state=random_state,
        )
    source_path = _write_selected_feature_distribution_source(out_dir, df, selected_features)
    return figure_paths, source_path


def _plot_selected_feature_correlation_heatmap(
    out_dir: Path,
    df: pd.DataFrame,
    selected_features: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    if len(selected_features) < 2:
        return []
    matrix = df.loc[:, list(selected_features)].apply(pd.to_numeric, errors="coerce")
    corr = matrix.replace([np.inf, -np.inf], np.nan).corr(method="pearson")
    if corr.empty:
        return []

    fig_size = max(3.2, 0.34 * len(selected_features) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title("Selected feature correlation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    paths = _save_figure(fig, out_dir / "selected_feature_correlation_heatmap", formats, dpi)
    plt.close(fig)
    return paths


def _plot_pca_loading_top_features(
    out_dir: Path,
    x: np.ndarray,
    feature_cols: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    top_n: int = 15,
) -> List[str]:
    if x.shape[0] < 2 or x.shape[1] < 1:
        return []
    n_components = min(2, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(x)
    loading_strength = np.sum(np.abs(pca.components_), axis=0)
    order = np.argsort(loading_strength)[::-1][: min(int(top_n), len(feature_cols))]
    names = [str(feature_cols[i]) for i in order][::-1]
    values = [float(loading_strength[i]) for i in order][::-1]

    fig, ax = plt.subplots(figsize=(4.2, max(2.6, 0.22 * len(names) + 1.0)))
    ax.barh(np.arange(len(names)), values, color="#4C78A8", height=0.68)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("|PC loading| sum")
    ax.set_title("Top PCA loading features")
    ax.grid(True, axis="x", linewidth=0.25, color="#D9D9D9", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    paths = _save_figure(fig, out_dir / "pca_loading_top_features", formats, dpi)
    plt.close(fig)
    return paths


def _split_auc_table(df: pd.DataFrame, selected_features: Sequence[str]) -> pd.DataFrame:
    if "split" not in df.columns or "target" not in df.columns or not selected_features:
        return pd.DataFrame()
    rows = []
    for split, sub in df.groupby("split", dropna=False, sort=True):
        y = pd.to_numeric(sub["target"], errors="coerce")
        for feature in selected_features:
            values = pd.to_numeric(sub.get(feature), errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = pd.DataFrame({"target": y, "value": values}).dropna()
            if valid["target"].nunique() < 2 or len(valid) < 3:
                auc_sep = np.nan
            else:
                try:
                    auc = float(roc_auc_score(valid["target"].astype(int), valid["value"].to_numpy(dtype=float)))
                    auc_sep = max(auc, 1.0 - auc)
                except Exception:
                    auc_sep = np.nan
            rows.append({"split": str(split), "feature": feature, "auc_separation": auc_sep})
    return pd.DataFrame(rows)


def _plot_feature_split_auc_heatmap(
    out_dir: Path,
    df: pd.DataFrame,
    selected_features: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[str], Dict[str, Dict[str, Optional[float]]]]:
    table = _split_auc_table(df, selected_features)
    if table.empty:
        return [], {}
    pivot = table.pivot(index="feature", columns="split", values="auc_separation")
    if pivot.empty or pivot.shape[1] < 2:
        return [], {}

    fig_w = max(3.4, 0.45 * pivot.shape[1] + 2.2)
    fig_h = max(2.6, 0.26 * pivot.shape[0] + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=0)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=6)
    ax.set_title("Selected feature AUC by split")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUC separation")
    fig.tight_layout()
    paths = _save_figure(fig, out_dir / "selected_feature_split_auc_heatmap", formats, dpi)
    plt.close(fig)

    summary = {
        str(feature): {
            str(split): (None if pd.isna(value) else float(value))
            for split, value in row.items()
        }
        for feature, row in pivot.iterrows()
    }
    return paths, summary


def _plot_single(
    coords: np.ndarray,
    labels: np.ndarray,
    method: str,
    dim: int,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    method_info: Optional[Mapping[str, object]] = None,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> List[str]:
    n_total = int(coords.shape[0])
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    title = (
        f"{METHOD_TITLES.get(method, method.upper())} {dim}D{title_suffix}\n"
        f"(n={n_total}, pos={n_pos}, neg={n_neg})"
    )
    if dim == 2:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        _plot_points(ax, coords, labels, dim=2)
        _format_axes_2d(ax, method, method_info)
        ax.set_title(title, fontsize=8)
    else:
        fig = plt.figure(figsize=(3.5, 3.1))
        ax = fig.add_subplot(111, projection="3d")
        _plot_points(ax, coords, labels, dim=3)
        _format_axes_3d(ax, method, method_info)
        ax.set_title(title, pad=8, fontsize=8)

    ax.legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)
    paths = _save_figure(fig, out_dir / f"{method}_{dim}d{filename_suffix}", formats, dpi)
    plt.close(fig)
    return paths


def _plot_panel(
    embeddings: Mapping[str, Mapping[int, np.ndarray]],
    labels: np.ndarray,
    dim: int,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    method_status: Optional[Mapping[str, Mapping[str, object]]] = None,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> List[str]:
    methods = [m for m in ("pca", "tsne", "umap") if dim in embeddings.get(m, {})]
    if not methods:
        return []

    n_total = int(labels.shape[0])
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    count_str = f"n={n_total} pos={n_pos} neg={n_neg}"
    if dim == 2:
        fig, axes = plt.subplots(1, len(methods), figsize=(3.15 * len(methods), 2.8), squeeze=False)
        for ax, method in zip(axes[0], methods):
            _plot_points(ax, embeddings[method][dim], labels, dim=2)
            _format_axes_2d(ax, method, (method_status or {}).get(method))
            ax.set_title(f"{METHOD_TITLES.get(method, method.upper())} 2D{title_suffix}\n({count_str})", fontsize=8)
        axes[0, -1].legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)
    else:
        fig = plt.figure(figsize=(3.25 * len(methods), 3.0))
        for idx, method in enumerate(methods, start=1):
            ax = fig.add_subplot(1, len(methods), idx, projection="3d")
            _plot_points(ax, embeddings[method][dim], labels, dim=3)
            _format_axes_3d(ax, method, (method_status or {}).get(method))
            ax.set_title(f"{METHOD_TITLES.get(method, method.upper())} 3D{title_suffix}\n({count_str})", pad=6, fontsize=8)
            if idx == len(methods):
                ax.legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)

    fig.tight_layout(w_pad=1.0)
    paths = _save_figure(fig, out_dir / f"embedding_panel_{dim}d{filename_suffix}", formats, dpi)
    plt.close(fig)
    return paths


def _write_source_data(
    out_dir: Path,
    sampled: pd.DataFrame,
    embeddings: Mapping[str, Mapping[int, np.ndarray]],
) -> Path:
    columns = [c for c in ["split", "sample_name", "h5_file", "target", "start_sec", "window_index", "mode"] if c in sampled.columns]
    source = sampled.loc[:, columns].copy()
    for method, by_dim in embeddings.items():
        for dim, coords in by_dim.items():
            for idx in range(dim):
                source[f"{method}_{dim}d_{idx + 1}"] = coords[:, idx]
    path = out_dir / "embedding_source_data.csv"
    source.to_csv(path, index=False)
    return path


def _write_source_data_balanced(
    out_dir: Path,
    sampled: pd.DataFrame,
    embeddings: Mapping[str, Mapping[int, np.ndarray]],
) -> Path:
    columns = [c for c in ["split", "sample_name", "h5_file", "target", "start_sec", "window_index", "mode"] if c in sampled.columns]
    source = sampled.loc[:, columns].copy()
    for method, by_dim in embeddings.items():
        for dim, coords in by_dim.items():
            for idx in range(dim):
                source[f"{method}_{dim}d_{idx + 1}"] = coords[:, idx]
    path = out_dir / "embedding_source_data_balanced.csv"
    source.to_csv(path, index=False)
    return path


def _write_report(
    out_dir: Path,
    summary: Mapping[str, object],
    figure_paths: Mapping[str, List[str]],
) -> Path:
    lines = [
        "# Feature Embedding Report",
        "",
        "This report visualizes Stage2 window features after robust median imputation and z-score scaling.",
        "Each point is one window. Color encodes the binary target label.",
        f"Output directory: `{out_dir.name}`.",
        "",
        "## Data",
        f"- Windows plotted: {summary['n_rows']}",
        f"- Embedding feature source: {summary.get('embedding_feature_source', 'unknown')}",
        f"- Numeric features: {summary['n_features']}",
        f"- Labels: {summary['label_counts']}",
        "",
        "## Figures",
        "- PCA 2D/3D: linear separability and dominant variance directions.",
        "- t-SNE 2D/3D: local neighborhood structure.",
        "- UMAP 2D/3D: global/local manifold structure when `umap-learn` is installed.",
        "- `*_balanced`: 正样本随机降采样至与负样本同等数量后的降维图，用于消除类别比例失调对视觉判断的影响。",
        "",
    ]
    balanced_keys = set()
    for key, paths in figure_paths.items():
        if not paths:
            continue
        rel = [Path(p).name for p in paths]
        if "_balanced" in key:
            balanced_keys.add(key)
        else:
            lines.append(f"- {key}: " + ", ".join(rel))
    if balanced_keys:
        lines.extend(["", "### Balanced (正样本降采样至与负样本同数量)", ""])
        bal_info = summary.get("balanced", {})
        if bal_info.get("status") == "ok":
            lines.append(f"- n_neg={bal_info.get('n_neg')}, n_pos_downsampled={bal_info.get('n_pos_downsampled')}, n_total={bal_info.get('n_total_balanced')}")
        for key in sorted(balanced_keys):
            paths = figure_paths[key]
            rel = [Path(p).name for p in paths]
            lines.append(f"- {key}: " + ", ".join(rel))
    explainer_paths = summary.get("explainer_figures", {})
    if isinstance(explainer_paths, Mapping) and explainer_paths:
        lines.extend(
            [
                "",
                "## Feature-Space Explainability",
            ]
        )
        for key, paths in explainer_paths.items():
            if not paths:
                continue
            rel = [Path(p).name for p in paths]
            lines.append(f"- {key}: " + ", ".join(rel))
    dist_info = summary.get("selected_feature_distributions", {})
    dist_figures = dist_info.get("figures", {}) if isinstance(dist_info, Mapping) else {}
    if dist_figures:
        lines.extend(
            [
                "",
                "## Selected Feature Distributions",
                "Each selected feature is plotted separately with label-colored jittered window points, a violin density envelope, and a boxplot summary.",
            ]
        )
        for key, paths in dist_figures.items():
            rel = [Path(p).name for p in paths]
            lines.append(f"- {key}: " + ", ".join(rel))
    lines.extend(
        [
            "",
            "## Method Status",
        ]
    )
    for method, info in summary["methods"].items():
        lines.append(f"- {METHOD_TITLES.get(method, method.upper())}: {info}")
    lines.append("")

    path = out_dir / "embedding_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _balance_by_target(
    sampled: pd.DataFrame,
    x: np.ndarray,
    feature_cols: Sequence[str],
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    """将正样本随机降采样到与负样本同等数量，返回平衡后的 DataFrame 和特征矩阵。

    负样本 (target=0) 全部保留，正样本 (target=1) 随机降采样至负样本数量。
    对平衡后的数据重新做中位数填充和 z-score 标准化。
    """
    if "target" not in sampled.columns:
        return sampled, x, {"status": "skipped", "reason": "no target column"}

    neg_mask = sampled["target"] == 0
    pos_mask = sampled["target"] == 1
    n_neg = int(neg_mask.sum())
    n_pos = int(pos_mask.sum())

    if n_neg == 0 or n_pos == 0:
        return sampled, x, {"status": "skipped", "reason": f"single class: neg={n_neg}, pos={n_pos}"}

    if n_pos <= n_neg:
        return sampled, x, {"status": "skipped", "reason": f"pos({n_pos}) <= neg({n_neg}), already balanced"}

    rng = np.random.default_rng(int(random_state))
    pos_indices = np.where(pos_mask)[0]
    keep_pos = rng.choice(pos_indices, size=n_neg, replace=False)
    keep_idx = np.sort(np.concatenate([np.where(neg_mask)[0], keep_pos]))
    balanced = sampled.iloc[keep_idx].reset_index(drop=True)

    raw = balanced.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(raw)
    scaled = StandardScaler().fit_transform(imputed)

    return balanced, scaled, {
        "status": "ok",
        "n_neg": n_neg,
        "n_pos_original": n_pos,
        "n_pos_downsampled": n_neg,
        "n_total_balanced": n_neg * 2,
    }


def run_embedding_report(
    artifact_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    methods: Sequence[str] = ("pca", "tsne"),
    dims: Sequence[int] = (2, 3),
    formats: Sequence[str] = ("png",),
    max_points: int = 0,
    random_state: int = 42,
    perplexity: float = 30.0,
    dpi: int = 600,
) -> Mapping[str, object]:
    artifact_dir = Path(artifact_dir)
    out_dir = Path(output_dir) if output_dir is not None else artifact_dir / "feature_embedding_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_nature_style()
    formats = ("png",)

    df, all_feature_cols = load_feature_pools(artifact_dir)
    selected_features, selected_status = load_selected_features(artifact_dir, all_feature_cols)
    if selected_features:
        feature_cols = list(selected_features)
        embedding_feature_source = "selected_features"
    else:
        feature_cols = list(all_feature_cols)
        embedding_feature_source = "all_numeric_features"

    sampled, x = prepare_matrix(df, feature_cols, int(max_points), int(random_state))
    embeddings, method_status = compute_embeddings(
        x=x,
        methods=methods,
        dims=dims,
        random_state=int(random_state),
        perplexity=float(perplexity),
    )

    labels = sampled["target"].to_numpy()
    figure_paths: Dict[str, List[str]] = {}
    for method in ("pca", "tsne", "umap"):
        for dim in sorted(embeddings.get(method, {})):
            key = f"{method}_{dim}d"
            figure_paths[key] = _plot_single(
                embeddings[method][dim],
                labels,
                method,
                dim,
                out_dir,
                formats,
                int(dpi),
                method_info=method_status.get(method),
            )
    for dim in sorted({int(d) for d in dims if int(d) in {2, 3}}):
        figure_paths[f"embedding_panel_{dim}d"] = _plot_panel(
            embeddings,
            labels,
            dim,
            out_dir,
            formats,
            int(dpi),
            method_status=method_status,
        )

    # ── 平衡版本：正样本随机降采样至与负样本同等数量 ──
    balanced_sampled, balanced_x, balance_info = _balance_by_target(
        sampled, x, feature_cols, int(random_state)
    )
    if balance_info.get("status") == "ok":
        balanced_embeddings, balanced_method_status = compute_embeddings(
            x=balanced_x,
            methods=methods,
            dims=dims,
            random_state=int(random_state),
            perplexity=float(perplexity),
        )
        balanced_labels = balanced_sampled["target"].to_numpy()
        _bal_suffix = " (balanced)"
        for method in ("pca", "tsne", "umap"):
            for dim in sorted(balanced_embeddings.get(method, {})):
                key = f"{method}_{dim}d_balanced"
                figure_paths[key] = _plot_single(
                    balanced_embeddings[method][dim],
                    balanced_labels,
                    method,
                    dim,
                    out_dir,
                    formats,
                    int(dpi),
                    method_info=balanced_method_status.get(method),
                    title_suffix=_bal_suffix,
                    filename_suffix="_balanced",
                )
        for dim in sorted({int(d) for d in dims if int(d) in {2, 3}}):
            figure_paths[f"embedding_panel_{dim}d_balanced"] = _plot_panel(
                balanced_embeddings,
                balanced_labels,
                dim,
                out_dir,
                formats,
                int(dpi),
                method_status=balanced_method_status,
                title_suffix=_bal_suffix,
                filename_suffix="_balanced",
            )
        # 平衡版的 source data
        _write_source_data_balanced(out_dir, balanced_sampled, balanced_embeddings)
    else:
        balanced_embeddings = {}
        balanced_method_status = balance_info

    source_path = _write_source_data(out_dir, sampled, embeddings)
    distribution_paths, distribution_source_path = plot_selected_feature_distributions(
        out_dir=out_dir,
        df=sampled,
        selected_features=selected_features,
        formats=formats,
        dpi=int(dpi),
        random_state=int(random_state),
    )
    distribution_statistics = summarize_selected_feature_distributions(sampled, selected_features)
    split_auc_paths, split_auc_summary = _plot_feature_split_auc_heatmap(
        out_dir=out_dir,
        df=sampled,
        selected_features=selected_features,
        formats=formats,
        dpi=int(dpi),
    )
    explainer_figures = {
        "selected_feature_correlation_heatmap": _plot_selected_feature_correlation_heatmap(
            out_dir=out_dir,
            df=sampled,
            selected_features=selected_features,
            formats=formats,
            dpi=int(dpi),
        ),
        "selected_feature_split_auc_heatmap": split_auc_paths,
        "pca_loading_top_features": _plot_pca_loading_top_features(
            out_dir=out_dir,
            x=x,
            feature_cols=feature_cols,
            formats=formats,
            dpi=int(dpi),
        ),
    }
    label_counts = sampled["target"].value_counts().sort_index()
    split_counts = sampled["split"].value_counts().sort_index() if "split" in sampled.columns else pd.Series(dtype=int)
    summary = {
        "n_rows": int(len(sampled)),
        "n_rows_available": int(len(df)),
        "max_points": int(max_points),
        "n_features": int(len(feature_cols)),
        "feature_columns": list(feature_cols),
        "all_numeric_feature_count": int(len(all_feature_cols)),
        "embedding_feature_source": embedding_feature_source,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "methods": method_status,
        "balanced": {
            "status": balance_info.get("status", "skipped"),
            "n_neg": balance_info.get("n_neg"),
            "n_pos_original": balance_info.get("n_pos_original"),
            "n_pos_downsampled": balance_info.get("n_pos_downsampled"),
            "n_total_balanced": balance_info.get("n_total_balanced"),
            "methods": balanced_method_status,
            "source_data_balanced": str(out_dir / "embedding_source_data_balanced.csv"),
        },
        "source_data": str(source_path),
        "figures": figure_paths,
        "explainer_figures": explainer_figures,
        "selected_feature_distributions": {
            **selected_status,
            "source_data": str(distribution_source_path) if distribution_source_path else None,
            "figures": distribution_paths,
            "statistics": distribution_statistics,
            "split_auc_separation": split_auc_summary,
        },
    }
    summary_path = out_dir / "embedding_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(out_dir, summary, figure_paths)

    return {
        "output_dir": out_dir,
        "report_path": report_path,
        "summary_path": summary_path,
        "source_path": source_path,
        "summary": summary,
    }


if __name__ == "__main__":
    main()
