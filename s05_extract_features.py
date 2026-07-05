# -*- coding: utf-8 -*-
"""
S05: Extract BOTH commercial scores and new features for ALL windows.

Output: {artifact_dir}/features_{train,valid,test}.csv
"""

import argparse, json, os, time
import numpy as np, pandas as pd

from s01_model import OldLivenessModel, extract_8_commercial_features, FEATURE_FS, COMMERCIAL_WIN_SEC, COMMERCIAL_STRIDE_SEC
from s01_model import commercial_model_manifest
from s02_features import load_ppg, load_acc, get_channels_from_window, detect_green_mode
from s02_features import is_prewindowed_signal, _downsample_ppg, _is_25hz_sample, extract_feature_pool_from_window, validate_h5_file
from s04_data import load_splits


def _to_25hz(s, ppg, acc):
    if _is_25hz_sample(s): return (np.asarray(ppg, dtype=np.float64),
        np.asarray(acc, dtype=np.float64) if acc is not None and len(acc) > 0 else None, 25)
    ppg25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS); acc25 = None
    if acc is not None and len(acc) > 0:
        from scipy.signal import resample_poly
        acc25 = resample_poly(np.asarray(acc, dtype=np.float32), FEATURE_FS, 100, axis=0).astype(np.float64)
    return ppg25, acc25, 100


def _prewindow_to_25hz(s, w, ws):
    n = int(w.shape[0])
    if (_is_25hz_sample(s) or n == int(round(float(ws) * FEATURE_FS)) or (n <= 200 and n > 0 and n % FEATURE_FS == 0)):
        return np.asarray(w, dtype=np.float64), 25
    return _downsample_ppg(np.asarray(w, dtype=np.float64), src_fs=100, tgt_fs=FEATURE_FS), 100


def extract_sample(sample, model):
    base = {"sample_name": sample.get("sample_name", "unknown"), "target": int(sample.get("target", 0))}
    try:
        ppg, acc = load_ppg(sample), load_acc(sample)
        ok, err = validate_h5_file(sample["h5_file"], base["sample_name"])
        if not ok: raise ValueError(err)
    except Exception as exc:
        return [{**base, "window_idx": -1, "commercial_score": None, "commercial_pred": 0,
                 "fallback": True, "fallback_reason": str(exc)}]
    rows = []
    if is_prewindowed_signal(ppg):
        mode = detect_green_mode(ppg)
        for idx in range(3, ppg.shape[0]):
            win25, _ = _prewindow_to_25hz(sample, ppg[idx], COMMERCIAL_WIN_SEC)
            try:
                ir, amb, g1, g2, g3 = get_channels_from_window(win25, mode)
                acc_seg = None
                if acc is not None and is_prewindowed_signal(acc) and idx < acc.shape[0]:
                    acc_seg, _ = _prewindow_to_25hz(sample, acc[idx], COMMERCIAL_WIN_SEC)
                _, score, _, _ = model.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, acc_seg))
                is_live = 1 if (score is not None and score > -1000) else 0
                nf = extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS)
                r = {**base, "window_idx": idx, "commercial_score": float(score) if score is not None else -2000.0,
                     "commercial_pred": is_live, "fallback": False, "fallback_reason": None}
                r.update(nf); rows.append(r)
            except Exception: continue
        return rows
    ppg25, acc25, _ = _to_25hz(sample, ppg, acc); mode = detect_green_mode(ppg)
    sw, ss = int(round(COMMERCIAL_WIN_SEC * FEATURE_FS)), int(round(COMMERCIAL_STRIDE_SEC * FEATURE_FS))
    for step in range(3, max(0, (len(ppg25) - sw) // ss + 1)):
        win = ppg25[step * ss:step * ss + sw, :]
        try:
            ir, amb, g1, g2, g3 = get_channels_from_window(win, mode)
            _, score, _, _ = model.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, None))
            is_live = 1 if (score is not None and score > -1000) else 0
            nf = extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS)
            r = {**base, "window_idx": step, "commercial_score": float(score) if score is not None else -2000.0,
                 "commercial_pred": is_live, "fallback": False, "fallback_reason": None}
            r.update(nf); rows.append(r)
        except Exception: continue
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--splits_dir", default="artifacts"); p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args(); os.makedirs(args.artifact_dir, exist_ok=True)
    with open(os.path.join(args.artifact_dir, "commercial_model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(commercial_model_manifest(), f, indent=2, ensure_ascii=False)
    splits = load_splits(args.splits_dir); model = OldLivenessModel(); t0 = time.time()
    for name in ["train", "valid", "test"]:
        samples = splits[name][:args.max_samples] if args.max_samples else splits[name]
        rows = []
        for i, s in enumerate(samples):
            if len(samples) >= 10 and (i + 1) % max(1, len(samples) // 10) == 0: print(f"[{name}] {i+1}/{len(samples)}")
            rows.extend(extract_sample(s, model))
        df = pd.DataFrame(rows); df.to_csv(os.path.join(args.artifact_dir, f"features_{name}.csv"), index=False)
        print(f"[{name}] {len(df)} rows")
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__": main()
