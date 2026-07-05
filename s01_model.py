# -*- coding: utf-8 -*-
"""
Frozen commercial model + 8-feature extraction + Stage1 gate.

This is the complete commercial liveness detection module.
MUST NOT BE MODIFIED - all enhancements go in s05-s09.
"""

import hashlib
import json

import numpy as np

# ======================== Commercial model (from 1.txt / C code) ========================

FEATURE_NAMES = [
    'green_corr', 'green_ac', 'amb_ac', 'acc_ysum',
    'green_dc', 'amb_dc', 'green_xcorr', 'fft_peak_med',
]
TREE_NUM, TREE_NODE, DETECT_TREE_THRESH = 18, 29, -1000
GOOD_CORR_THRESH, GOOD_CORR_THRESH_SLEEP, GOOD_AC_THRESH, MIN_AC_THRESH = 8000, 6000, 1000, 10
LIVE_FLAG_DELAY, UN_LIVE_FLAG_DELAY = 5, 5

TREE_INDEX = [
    [7, 1, 0, -1, 1, -1, -1, 5, 4, -1, -1, 3, -1, -1, 1, 5, 1, -1, -1, 4, -1, -1, 4, 3, -1, -1, 4, -1, -1],
    [1, 3, 1, -1, 4, -1, -1, 1, 7, -1, -1, 4, -1, -1, 7, 3, 1, -1, -1, 4, -1, -1, 7, 6, -1, -1, 3, -1, -1],
    [7, 3, 1, -1, 2, -1, -1, 0, 0, -1, -1, 1, -1, -1, 3, 5, 5, -1, -1, 0, -1, -1, 0, 5, -1, -1, 7, -1, -1],
    [6, 3, 1, 5, -1, -1, 0, -1, -1, 4, 4, -1, -1, 1, -1, -1, 7, 3, 1, -1, -1, 0, -1, -1, 5, -1, 5, -1, -1],
    [4, 2, 7, 3, -1, -1, 2, -1, -1, 2, -1, -1, 1, 2, 1, -1, -1, 5, -1, -1, 0, 4, -1, 5, -1, -1, 5, -1, -1],
    [2, 4, 1, 5, -1, -1, 4, -1, -1, 4, 5, -1, -1, 1, -1, -1, 7, 1, 4, -1, -1, 4, -1, -1, 3, 3, -1, -1, -1],
    [1, 4, 5, 4, -1, -1, 5, -1, -1, 1, 4, -1, -1, 4, -1, -1, 4, 2, 0, -1, -1, 4, -1, -1, 4, 3, -1, -1, -1],
    [4, 4, 0, -1, 7, -1, -1, 2, 4, -1, -1, 3, -1, -1, 4, 3, 5, -1, -1, 3, -1, -1, 2, 1, -1, -1, 2, -1, -1],
    [6, 3, -1, 7, 5, -1, 2, -1, -1, 7, 1, -1, -1, 0, -1, -1, 7, 4, 6, -1, -1, 3, -1, -1, 3, 3, -1, -1, -1],
    [6, 3, 1, 5, -1, -1, 5, -1, -1, 1, 2, -1, -1, 4, -1, -1, 7, 7, 7, -1, -1, 3, -1, -1, 3, 4, -1, -1, -1],
    [3, 1, 1, 5, -1, -1, 2, -1, -1, 3, -1, 4, -1, -1, 1, 5, 4, -1, -1, 4, -1, -1, 7, 3, -1, -1, 2, -1, -1],
    [5, 5, 5, 4, -1, -1, 2, -1, -1, 3, -1, 4, -1, -1, 4, 3, 2, -1, -1, 4, -1, -1, 5, 4, -1, -1, 5, -1, -1],
    [0, 2, 5, 4, -1, -1, 4, -1, -1, 5, 5, -1, -1, 4, -1, -1, 4, 4, 5, -1, -1, -1, 4, 5, -1, -1, 7, -1, -1],
    [4, 4, 4, 4, -1, -1, 1, -1, -1, 2, 5, -1, -1, 4, -1, -1, 1, 7, -1, 4, -1, -1, 5, 4, -1, -1, 2, -1, -1],
    [5, 4, 4, 3, -1, -1, 4, -1, -1, 1, -1, 3, -1, -1, 0, 7, 6, -1, -1, 1, -1, -1, 4, 4, -1, -1, 4, -1, -1],
    [3, 4, 4, 5, -1, -1, 3, -1, -1, 4, 4, -1, -1, 5, -1, -1, 2, 3, 7, -1, -1, -1, 0, 0, -1, -1, 3, -1, -1],
    [5, 5, 0, 7, -1, -1, 1, -1, -1, 4, 4, -1, -1, 3, -1, -1, 5, 4, 4, -1, -1, 4, -1, -1, 1, 5, -1, -1, -1],
    [2, 5, 5, 7, -1, -1, 7, -1, -1, 5, 2, -1, -1, 2, -1, -1, 1, 7, 3, -1, -1, -1, 3, 1, -1, -1, 4, -1, -1],
]

TREE_VALUE = [
    [347, 2016, 8, 9057, 766, -9749, -6549, 28548, 8370, 7573, -7357, 11419, -5768, 6272, 492, 13732, 446, -9453, -3421, 1840, 1086, -9545, 943, 2063, -9130, 2755, 7869, 9418, -7105],
    [1060, 2973, 422, -9591, 1716, -1514, -8643, 350, 1, 9594, -5674, 541, -9978, 3462, 693, 2839, 17585, -4177, 6921, 8483, 4284, -7486, 12302, 9714, 5108, 9210, 2505, -8894, 9209],
    [127, 547, 2981, -9552, 12862, -2887, 9920, 11, 5, 9398, 3539, 525, -6540, -1919, 2458, 11475, 2671, 248, -8980, 9464, 417, 9010, 9197, 2714, 4853, 187, 898, 6310, 9672],
    [9866, 467, 3949, 971, -9143, -6206, 4650, -9403, 8482, 3345, 617, -3467, 1489, 5946, -8159, 2518, 6696, 382, 5634, -8161, 10000, 8809, 2915, 8353, 2476, 10000, 3857, -9667, 4695],
    [2, 180, 5, 4795, -10000, 9445, 2, 3968, 7613, 29519, -10000, 10000, 201, 556, 120, -9844, -8240, 50849, 2265, -9349, 9569, 1742, 1637, 2709, 2197, -3719, 2536, 8653, 561],
    [8180, 947, 209, 35705, -1085, 8714, 540, -9966, -7187, 6065, 124, 6847, 4, 30132, -7641, 3050, 348, 9428, 1810, 5627, 1012, 2140, 3060, 8750, 4261, 1303, 2080, -4850, 2465],
    [13107, 1939, 20563, 2, 2976, -2585, 47584, 7046, 1047, 3684, 2110, -6435, -2157, 2053, -3694, 2390, 2064, 54935, 7534, -5461, -620, 2037, 9304, -1918, 21648, 9138, 6395, 1543, -9474],
    [691, 2, 2, 10000, 5, 5527, -1317, 13702, 538, -9738, -3245, 1814, -8254, 5938, 6390, 1391, 932, -552, 4927, 2271, -2382, 998, 61, 1052, -10000, 8935, 52049, -5972, 6747],
    [9946, 105, -10000, 34, 2097, 0, 1541, -6962, -2352, 347, 2024, 2678, -57, 9473, -1967, 2596, 5997, 553, 9948, 10000, -8712, 248, -8193, 8686, 2580, 1251, 8618, -6364, 10000],
    [9588, 15015, 145, 86, 4755, -5734, 246, 2482, -206, 2001, 1239, -218, 4692, 6898, -4167, 6520, 4749, 1400, 1398, 844, -9455, 2932, 2142, 6752, 2720, 1973, 689, -7294, 7361],
    [557, 1378, 444, 1170, -368, -8718, 33, 9817, -8267, 157, -9859, 1958, 6308, 303, 21701, 972, 1367, -1533, 1948, 1162, 2271, -857, 3484, 5808, 4856, 674, 427, 8804, -6467],
    [97, 71, 33, 1041, 1718, 9710, 40, -6452, 4300, 343, -9991, 2775, 8020, -1201, 403, 2557, 26644, -9307, -4377, 2, 8260, -5243, 225, 1765, -162, -4590, 243, 5732, 171],
    [9844, 452, 18823, 1363, -2038, 41, 1217, 2765, -7923, 1105, 338, 8692, -7937, 8544, 542, -4134, 1975, 777, 274, -9593, 8073, 9877, 2005, 2551, 9941, -8343, 9255, 9809, -2790],
    [4752, 2505, 2122, 1958, 483, -1690, 34760, 3009, -5240, 34109, 328, -5450, -1611, 3434, 1106, 7636, 896, 257, -10000, 6001, 547, -10000, 5795, 9299, 7631, 2486, 10594, -3857, 4607],
    [205, 1282, 1, 1022110, 9403, 1381, 880, -9603, 372, 284, -9198, 1704, 9389, 3208, 6402, 661, 9459, -798, -4002, 18649, -5985, 1065, 1521, 1478, -287, -4998, 1742, 3885, -84],
    [5691002, 1268, 1185, 53942, -1211, 2496, 1824, 6545, -5716, 1478, 1405, 625, 6372, 1836, 783, -390, 2, 6391238, 3, 10000, -10000, 7223, 35, 17, 9367, 2039, 5736934, -10000, 9735],
    [8708, 2146, 8348, 129, 386, -3036, 2915, -209, 3598, 2005, 1975, -2989, -8025, 2618, 2921, -2579, 15730, 3554, 1321, 2365, 6880, 8893, -7424, 8852, 35633, 23091, -3578, 185, 6759],
    [9285, 578, 421, 348, 764, -2045, 432, -6147, 5199, 687, 156, 7977, -5975, 49, -5116, 646, 637, 5169, 1946, -9402, -6995, 10000, 944, 1998, -4685, 8181, 8, -9480, -831],
]


def _stable_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def commercial_model_manifest():
    """Return immutable metadata for the frozen commercial feature/model contract."""
    return {
        "model_name": "frozen_commercial_adaboost",
        "feature_names": list(FEATURE_NAMES),
        "tree_num": int(TREE_NUM),
        "tree_node": int(TREE_NODE),
        "detect_tree_threshold": int(DETECT_TREE_THRESH),
        "good_corr_threshold": int(GOOD_CORR_THRESH),
        "good_corr_threshold_sleep": int(GOOD_CORR_THRESH_SLEEP),
        "good_ac_threshold": int(GOOD_AC_THRESH),
        "min_ac_threshold": int(MIN_AC_THRESH),
        "live_flag_delay": int(LIVE_FLAG_DELAY),
        "un_live_flag_delay": int(UN_LIVE_FLAG_DELAY),
        "feature_fs": int(FEATURE_FS),
        "commercial_win_sec": int(COMMERCIAL_WIN_SEC),
        "commercial_stride_sec": int(COMMERCIAL_STRIDE_SEC),
        "stage1_primitive_sec": float(STAGE1_PRIMITIVE_SEC),
        "stage1_decision_sec": float(STAGE1_DECISION_SEC),
        "stage1_fs": int(STAGE1_FS),
        "stage1_gate_k": int(STAGE1_GATE_K),
        "tree_index_sha256": _stable_sha256(TREE_INDEX),
        "tree_value_sha256": _stable_sha256(TREE_VALUE),
        "frozen": True,
    }


def _dt_branch_process(index, branch_index, length):
    j = 0; neg = 0; pos = 0; i = branch_index
    while j < length:
        j += 1
        if index[i + j] >= 0: pos += 1
        else: neg += 1
        if neg > pos: i += pos + neg + 1; break
    return i


def _dt_classify(x, index, value, length):
    confidence = 0; i = 0
    while i < length:
        if index[i] < 0: confidence = value[i]; return -index[i], confidence
        elif x[index[i]] <= value[i]: i += 1
        else: i = _dt_branch_process(index, i, length)
    return 0, confidence


class OldLivenessModel:
    def __init__(self, is_sleep=False, all_channel_saturate=False):
        self.is_sleep = is_sleep; self.all_channel_saturate = all_channel_saturate
        self.pre_live_flag = 1; self.wear_num = 0; self.no_wear_num = 0; self.tick = 0

    def reset(self): self.pre_live_flag = 1; self.wear_num = 0; self.no_wear_num = 0; self.tick = 0

    def dt_score(self, x):
        return sum(_dt_classify(x, TREE_INDEX[i], TREE_VALUE[i], TREE_NODE)[1] for i in range(TREE_NUM))

    def logical_judge(self, feature):
        th = GOOD_CORR_THRESH_SLEEP if self.is_sleep else GOOD_CORR_THRESH
        return feature[0] > th and feature[6] > th and feature[1] > GOOD_AC_THRESH

    def flag_transition(self, is_live, is_after_move=False, no_need_calc_feature=False):
        if is_live == 1:
            self.no_wear_num = max(0, self.no_wear_num - 1) if self.no_wear_num > 0 else 0
            if self.no_wear_num == 0: self.wear_num = min(LIVE_FLAG_DELAY, self.wear_num + 1)
        else:
            self.wear_num = max(0, self.wear_num - 1) if self.wear_num > 0 else 0
            if self.wear_num == 0: self.no_wear_num = min(UN_LIVE_FLAG_DELAY, self.no_wear_num + 1)
        if self.wear_num >= LIVE_FLAG_DELAY and self.pre_live_flag == 0: is_live = 1
        elif self.no_wear_num >= UN_LIVE_FLAG_DELAY and self.pre_live_flag == 1: is_live = 0
        else: is_live = self.pre_live_flag
        self.pre_live_flag = is_live; return is_live

    def predict(self, feature, use_flag_transition=True, is_after_move=False, no_need_calc_feature=False):
        self.tick += 1; info = {}
        if feature[1] < MIN_AC_THRESH or feature[4] == 0 or self.all_channel_saturate:
            is_live = 1; info['reason'] = 'abnormal_signal_protect'; info['dt_score'] = None; info['dt_live'] = None
        else:
            score = self.dt_score(feature); dt_live = 1 if score > DETECT_TREE_THRESH else 0
            info['dt_score'] = score; info['dt_live'] = dt_live; is_live = dt_live
            if self.logical_judge(feature): is_live = 1; info['logical_override'] = True
        info['pre_flag_transition'] = is_live
        if use_flag_transition: is_live = self.flag_transition(is_live, is_after_move, no_need_calc_feature)
        info['is_live'] = is_live; info['wear_num'] = self.wear_num; info['no_wear_num'] = self.no_wear_num; info['tick'] = self.tick
        return is_live, info

    def predict_raw(self, feature):
        if feature[1] < MIN_AC_THRESH or feature[4] == 0 or self.all_channel_saturate: return 1, None, None, False
        score = self.dt_score(feature); dt_live = 1 if score > DETECT_TREE_THRESH else 0
        is_live = dt_live; override = False
        if self.logical_judge(feature): is_live = 1; override = True
        return is_live, score, dt_live, override

    def score_to_probability(self, score, temperature=5000.0):
        if score is None: return 1.0
        z = (float(score) - DETECT_TREE_THRESH) / temperature; z = max(-50.0, min(50.0, z))
        return float(1.0 / (1.0 + np.exp(-z)))


# ======================== Commercial 8-feature extraction (from s09 / C code) ========================

EPS = 1e-12
FEATURE_FS, COMMERCIAL_WIN_SEC, COMMERCIAL_STRIDE_SEC = 25, 5, 1
STAGE1_PRIMITIVE_SEC, STAGE1_DECISION_SEC, STAGE1_FS = 1.0, 3.0, 5
STAGE1_GATE_K = int(round(STAGE1_DECISION_SEC / STAGE1_PRIMITIVE_SEC))

from s02_features import (
    preprocess_signal, safe_corr, robust_mad,
    moving_average_filter, normalized_autocorr, fft_peak_features,
    downsample_to_5hz,
)


def _safe_float(value, default=0.0):
    if value is None or not np.isfinite(value): return float(default)
    return float(value)


class CommercialStage1Gate:
    def __init__(self, dc_threshold, K=3):
        self.dc_threshold = float(dc_threshold); self.K = int(K)
        self.stage2_enabled = False; self.pass_count = 0; self.fail_count = 0

    def _check_one(self, ir5):
        x = np.asarray(ir5, dtype=float)
        dc = float(np.min((x[:-1] + x[1:]) / 2.0)) if len(x) >= 2 else (float(x[0]) if len(x) == 1 else 0.0)
        return dc > self.dc_threshold

    def update(self, ir5):
        if self._check_one(ir5): self.pass_count += 1; self.fail_count = 0
        else: self.fail_count += 1; self.pass_count = 0
        if self.pass_count >= self.K: self.stage2_enabled = True
        elif self.fail_count >= self.K: self.stage2_enabled = False
        return self.stage2_enabled


def extract_8_commercial_features(ir, ambient, g1, g2, g3, acc_window=None, fs=FEATURE_FS):
    _, amb_bp, _ = preprocess_signal(ambient, fs)
    g1_raw, g1_bp, _ = preprocess_signal(g1, fs); g2_raw, g2_bp, _ = preprocess_signal(g2, fs)
    g3_raw, g3_bp, _ = preprocess_signal(g3, fs)
    g_mean_raw = (g1_raw + g2_raw + g3_raw) / 3.0; g_mean_bp = (g1_bp + g2_bp + g3_bp) / 3.0
    ma_win = max(2, int(round(0.15 * fs))); g_smooth = moving_average_filter(g_mean_bp, window_size=ma_win)
    green_corr = safe_corr(g_mean_bp, g_smooth)
    green_ac = 0.5 * float(np.sqrt(np.mean(g_mean_bp**2))) + 0.5 * robust_mad(g_mean_bp) * 1.4826
    amb_ac = 0.5 * float(np.sqrt(np.mean(amb_bp**2))) + 0.5 * robust_mad(amb_bp) * 1.4826
    if acc_window is not None and len(acc_window) >= 4:
        acc_arr = np.asarray(acc_window, dtype=float); acc_mag = np.sqrt(np.sum(acc_arr**2, axis=1) + EPS)
        acc_ysum = float(np.mean(acc_mag))
    else: acc_ysum = 0.0
    ac = normalized_autocorr(g_mean_bp); lag_min = max(1, int(fs * 60.0 / 180.0))
    lag_max = min(len(ac) - 1, int(fs * 60.0 / 40.0))
    green_xcorr = float(np.max(ac[lag_min:lag_max + 1])) if lag_max > lag_min else 0.0
    peak_ratio, _ = fft_peak_features(g_mean_bp, fs, fmin=0.5, fmax=5.0)
    return [_safe_float(v) for v in [green_corr, green_ac, amb_ac, acc_ysum,
            float(np.median(g_mean_raw)), float(np.median(np.asarray(ambient, dtype=float))),
            green_xcorr, peak_ratio]]


def advance_stage1_gate(gate, ir5, s1_win, s1_stride, last_s1_step, target_s1_step):
    enabled, step = gate.stage2_enabled, last_s1_step
    for s1_step in range(last_s1_step + 1, target_s1_step + 1):
        start = s1_step * s1_stride; seg = ir5[start:start + s1_win]
        if len(seg) < s1_win: break
        enabled = bool(gate.update(seg)); step = s1_step
    return enabled, step
