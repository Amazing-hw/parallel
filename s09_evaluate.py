# -*- coding: utf-8 -*-
"""
S09: End-to-end evaluation for the parallel soft guard.

Output: {artifact_dir}/evaluation_report.json
"""

import argparse, json, os, time
import numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from s01_model import OldLivenessModel, extract_8_commercial_features, DETECT_TREE_THRESH
from s01_model import FEATURE_FS, COMMERCIAL_WIN_SEC, COMMERCIAL_STRIDE_SEC
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


def score_to_prob(score):
    if score is None or not np.isfinite(score): return 1.0
    z = (float(score) - DETECT_TREE_THRESH) / 5000.0
    return float(1.0 / (1.0 + np.exp(-max(-50.0, min(50.0, z)))))


def predict_new(new_feats, bundle):
    feats, fills = bundle["selected_features"], bundle["fill_values"]
    X = np.array([[new_feats.get(f, 0.0) for f in feats]], dtype=float)
    for i, c in enumerate(feats):
        if not np.isfinite(X[0, i]): X[0, i] = fills.get(c, 0.0)
    return float(bundle["model"].predict_proba(X)[0, 1])


def fuse_veto(P_c, P_n, hi=0.80, lo=0.20):
    P_c, P_n = np.asarray(P_c, float), np.asarray(P_n, float)
    r = P_c.copy(); r[(P_c >= 0.5) & (P_n < lo)] = P_n[(P_c >= 0.5) & (P_n < lo)]
    return r


def metric(yt, yp):
    cm = confusion_matrix(yt, yp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    return {"n": len(yt), "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


GUARD_MODES = ("bypass", "shadow", "soft_guard", "hard_veto")


def _summarize_risks(veto_risks, veto_threshold):
    arr = np.asarray(veto_risks, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.asarray([0.0], dtype=float)
    high = arr >= float(veto_threshold)
    return {
        "veto_risk": float(np.max(arr)),
        "risk_count": int(np.sum(high)),
        "window_count": int(arr.size),
        "risk_ratio": float(np.mean(high)),
    }


def make_guard_decision(
    commercial_pred,
    veto_risks,
    guard_mode="shadow",
    veto_threshold=0.8,
    min_veto_windows=2,
    min_veto_ratio=0.4,
):
    commercial_pred = int(commercial_pred)
    if guard_mode not in GUARD_MODES:
        raise ValueError(f"unknown guard_mode: {guard_mode}")

    summary = _summarize_risks(veto_risks, veto_threshold)
    hard_candidate = (
        commercial_pred == 1
        and summary["risk_count"] >= int(min_veto_windows)
        and summary["risk_ratio"] >= float(min_veto_ratio)
    )
    decision = {
        "final_pred": commercial_pred,
        "guard_action": "pass",
        "decision_source": "commercial",
        **summary,
    }
    if commercial_pred == 0:
        return decision
    if guard_mode == "bypass":
        return decision
    if summary["risk_count"] > 0:
        decision["guard_action"] = "extend_detection"
    if guard_mode == "soft_guard":
        decision["decision_source"] = "soft_guard" if decision["guard_action"] != "pass" else "commercial"
        return decision
    if guard_mode == "shadow":
        return decision
    if hard_candidate:
        decision["final_pred"] = 0
        decision["guard_action"] = "hard_veto"
        decision["decision_source"] = "hard_veto"
    return decision


def apply_guard_decision(commercial_pred, veto_risks, guard_mode="shadow", veto_threshold=0.8):
    return make_guard_decision(
        commercial_pred, veto_risks, guard_mode=guard_mode, veto_threshold=veto_threshold
    )["final_pred"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--splits_dir", default="artifacts"); p.add_argument("--split", default="test")
    p.add_argument("--guard_mode", default="shadow", choices=GUARD_MODES)
    p.add_argument("--min_veto_windows", type=int, default=2)
    p.add_argument("--min_veto_ratio", type=float, default=0.4)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    splits = load_splits(args.splits_dir); samples = splits[args.split]
    bundle = joblib.load(os.path.join(args.artifact_dir, "new_model_bundle.pkl"))
    with open(os.path.join(args.artifact_dir, "fusion_config.json")) as f: fcfg = json.load(f)
    strat = fcfg["chosen_strategy"]; vp = fcfg.get("veto_params", {})
    com = OldLivenessModel(); t0 = time.time(); results = []
    for sample in samples:
        sn, target = sample.get("sample_name", "unknown"), int(sample.get("target", 0))
        try:
            ppg, acc = load_ppg(sample), load_acc(sample)
            ok, err = validate_h5_file(sample["h5_file"], sn)
            if not ok: raise ValueError(err)
        except Exception:
            results.append({"sample_name": sn, "target": target, "commercial_pred": 0,
                            "parallel_pred": 0, "bypass_pred": 0, "fallback": True}); continue
        Pc, Pn = [], []
        if is_prewindowed_signal(ppg):
            mode = detect_green_mode(ppg)
            for idx in range(3, ppg.shape[0]):
                win25, _ = _prewindow_to_25hz(sample, ppg[idx], COMMERCIAL_WIN_SEC)
                try:
                    ir, amb, g1, g2, g3 = get_channels_from_window(win25, mode)
                    acc_seg = None
                    if acc is not None and is_prewindowed_signal(acc) and idx < acc.shape[0]:
                        acc_seg, _ = _prewindow_to_25hz(sample, acc[idx], COMMERCIAL_WIN_SEC)
                    _, score, _, _ = com.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, acc_seg))
                    Pc.append(score_to_prob(score))
                    Pn.append(predict_new(extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS), bundle))
                except Exception: Pc.append(0.0); Pn.append(0.5)
        else:
            ppg25, acc25, _ = _to_25hz(sample, ppg, acc); mode = detect_green_mode(ppg)
            sw, ss = int(round(COMMERCIAL_WIN_SEC * FEATURE_FS)), int(round(COMMERCIAL_STRIDE_SEC * FEATURE_FS))
            for step in range(3, max(0, (len(ppg25) - sw) // ss + 1)):
                win = ppg25[step * ss:step * ss + sw, :]
                try:
                    ir, amb, g1, g2, g3 = get_channels_from_window(win, mode)
                    _, score, _, _ = com.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, None))
                    Pc.append(score_to_prob(score))
                    Pn.append(predict_new(extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS), bundle))
                except Exception: Pc.append(0.0); Pn.append(0.5)
        if not Pc:
            results.append({"sample_name": sn, "target": target, "commercial_pred": 0,
                            "parallel_pred": 0, "bypass_pred": 0, "fallback": True}); continue
        Pca, Pna = np.array(Pc), np.array(Pn)
        com_pred = int(np.mean(Pca) >= 0.5)
        if strat != "veto":
            raise ValueError(f"unsupported commercial guard strategy: {strat}")
        Pf = fuse_veto(Pca, Pna, vp.get("p_c_high", 0.8), vp.get("p_n_low", 0.2))
        risk_series = 1.0 - Pna if com_pred else [0.0]
        decision = make_guard_decision(
            com_pred, risk_series, guard_mode=args.guard_mode, veto_threshold=1.0 - float(vp.get("p_n_low", 0.2)),
            min_veto_windows=args.min_veto_windows, min_veto_ratio=args.min_veto_ratio
        )
        par_pred = decision["final_pred"]
        results.append({"sample_name": sn, "target": target, "commercial_pred": com_pred,
                        "parallel_pred": par_pred, "bypass_pred": com_pred,
                        "parallel_mean_prob": float(np.mean(Pf)), "veto_risk": decision["veto_risk"],
                        "risk_count": decision["risk_count"], "window_count": decision["window_count"],
                        "risk_ratio": decision["risk_ratio"], "guard_action": decision["guard_action"],
                        "decision_source": decision["decision_source"],
                        "guard_mode": args.guard_mode, "fallback": False})
    print(f"Inference ({time.time()-t0:.1f}s)")
    cm = metric([r["target"] for r in results], [r["commercial_pred"] for r in results])
    pm = metric([r["target"] for r in results], [r["parallel_pred"] for r in results])
    disc = [r for r in results if r["commercial_pred"] != r["parallel_pred"]]
    fixed = sum(1 for d in disc if d["parallel_pred"] == d["target"] and d["commercial_pred"] != d["target"])
    broken = sum(1 for d in disc if d["commercial_pred"] == d["target"] and d["parallel_pred"] != d["target"])
    print(f"Commercial: acc={cm['accuracy']:.4f} prec={cm['precision']:.4f} rec={cm['recall']:.4f} f1={cm['f1']:.4f}")
    print(f"Parallel:   acc={pm['accuracy']:.4f} prec={pm['precision']:.4f} rec={pm['recall']:.4f} f1={pm['f1']:.4f}")
    print(f"Disagreements: {len(disc)}/{len(results)} (fixed={fixed}, broken={broken})")
    report = {"split": args.split, "n": len(results), "fusion": strat, "guard_mode": args.guard_mode,
              "min_veto_windows": args.min_veto_windows, "min_veto_ratio": args.min_veto_ratio,
              "commercial": cm, "parallel": pm,
              "bypass": cm, "n_disagreements": len(disc), "fixed": fixed, "broken": broken}
    with open(os.path.join(args.artifact_dir, "evaluation_report.json"), "w") as f: json.dump(report, f, indent=2)
    pd.DataFrame(results).to_csv(os.path.join(args.artifact_dir, "evaluation_samples.csv"), index=False)
    pd.DataFrame([{"metric": m, "commercial": cm[m], "parallel": pm[m], "delta": pm[m] - cm[m]}
                  for m in ["accuracy", "precision", "recall", "f1"]])\
      .to_csv(os.path.join(args.artifact_dir, "evaluation_comparison.csv"), index=False)
    print("Done")

if __name__ == "__main__": main()
