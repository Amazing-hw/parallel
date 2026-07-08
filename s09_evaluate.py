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


def resolve_feature_pool_path(artifact_dir, split):
    standard = os.path.join(artifact_dir, f"feature_pool_{split}.csv")
    legacy = os.path.join(artifact_dir, f"features_{split}.csv")
    return standard if os.path.exists(standard) else legacy


def _to_25hz(s, ppg, acc):
    if _is_25hz_sample(s): return (np.asarray(ppg, dtype=np.float64),
        np.asarray(acc, dtype=np.float64) if acc is not None and len(acc) > 0 else None, 25)
    ppg25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS); acc25 = None
    if acc is not None and len(acc) > 0:
        from scipy.signal import resample_poly
        acc25 = resample_poly(np.asarray(acc, dtype=np.float32), FEATURE_FS, 100, axis=0).astype(np.float64)
    return ppg25, acc25, 100


def _slice_acc(acc25, start, size):
    if acc25 is None or start >= len(acc25):
        return None
    return acc25[start:start + size]


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


def build_feature_matrix(df, features, fills):
    if not features:
        return np.empty((len(df), 0), dtype=float)
    matrix = df.reindex(columns=features)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    X = matrix.to_numpy(dtype=float)
    for i, feature in enumerate(features):
        invalid = ~np.isfinite(X[:, i])
        X[invalid, i] = fills.get(feature, 0.0)
    return X


def predict_new_many(df, bundle):
    feats, fills = bundle["selected_features"], bundle["fill_values"]
    X = build_feature_matrix(df, feats, fills)
    return np.asarray(bundle["model"].predict_proba(X)[:, 1], dtype=float)


def fuse_veto(P_c, P_n, hi=0.80, lo=0.20):
    P_c, P_n = np.asarray(P_c, float), np.asarray(P_n, float)
    r = P_c.copy(); r[(P_c >= 0.5) & (P_n < lo)] = P_n[(P_c >= 0.5) & (P_n < lo)]
    return r


def commercial_probabilities_from_rows(df):
    if "commercial_pred" in df.columns:
        return pd.to_numeric(df["commercial_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return df["commercial_score"].apply(score_to_prob).to_numpy(dtype=float)


def metric(yt, yp):
    cm = confusion_matrix(yt, yp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    return {"n": len(yt), "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def confusion_matrix_rows(model_name, metrics):
    c = metrics["confusion"]
    return [
        {"model": model_name, "true_label": 0, "pred_0": int(c["TN"]), "pred_1": int(c["FP"])},
        {"model": model_name, "true_label": 1, "pred_0": int(c["FN"]), "pred_1": int(c["TP"])},
    ]


def print_confusion_matrix(title, metrics):
    c = metrics["confusion"]
    print(f"{title} confusion matrix (rows=true label, cols=pred label)")
    print("              pred_0  pred_1")
    print(f"  true_0      {int(c['TN']):6d}  {int(c['FP']):6d}")
    print(f"  true_1      {int(c['FN']):6d}  {int(c['TP']):6d}")


ERROR_SAMPLE_COLUMNS = [
    "sample_name", "target", "commercial_pred", "final_pred",
    "commercial_wrong", "final_wrong", "change_type", "guard_action",
    "decision_source", "guard_mode", "fallback", "veto_risk", "risk_count",
    "window_count", "risk_ratio",
]


def _prediction_audit_rows(results, final_key):
    rows = []
    for r in results:
        target = int(r.get("target", 0))
        commercial_pred = int(r.get("commercial_pred", 0))
        final_pred = int(r.get(final_key, commercial_pred))
        commercial_wrong = commercial_pred != target
        final_wrong = final_pred != target
        if final_wrong and commercial_wrong:
            change_type = "still_wrong"
        elif final_wrong:
            change_type = "broken_by_full_scheme"
        elif commercial_wrong:
            change_type = "fixed_by_full_scheme"
        else:
            change_type = "correct"
        rows.append({
            "sample_name": r.get("sample_name", ""),
            "target": target,
            "commercial_pred": commercial_pred,
            "final_pred": final_pred,
            "commercial_wrong": bool(commercial_wrong),
            "final_wrong": bool(final_wrong),
            "change_type": change_type,
            "guard_action": r.get("guard_action", ""),
            "decision_source": r.get("decision_source", ""),
            "guard_mode": r.get("guard_mode", ""),
            "fallback": bool(r.get("fallback", False)),
            "veto_risk": r.get("veto_risk", ""),
            "risk_count": r.get("risk_count", ""),
            "window_count": r.get("window_count", ""),
            "risk_ratio": r.get("risk_ratio", ""),
        })
    return rows


def write_error_sample_outputs(artifact_dir, results, final_key):
    rows = _prediction_audit_rows(results, final_key)
    audit = pd.DataFrame(rows, columns=ERROR_SAMPLE_COLUMNS)
    audit[audit["final_wrong"] == True].to_csv(
        os.path.join(artifact_dir, "evaluation_error_samples.csv"), index=False
    )
    audit[audit["change_type"] == "fixed_by_full_scheme"].to_csv(
        os.path.join(artifact_dir, "evaluation_fixed_samples.csv"), index=False
    )
    audit.to_csv(os.path.join(artifact_dir, "evaluation_prediction_audit.csv"), index=False)


GUARD_MODES = ("bypass", "shadow", "soft_guard", "hard_veto")


def _row_pred_for_guard_mode(row, mode, min_veto_windows=2, min_veto_ratio=0.4):
    commercial_pred = int(row.get("commercial_pred", 0))
    if mode in {"bypass", "shadow", "soft_guard"} or commercial_pred == 0:
        return commercial_pred
    risk_count = int(row.get("risk_count", 0) or 0)
    risk_ratio = float(row.get("risk_ratio", 0.0) or 0.0)
    if risk_count >= int(min_veto_windows) and risk_ratio >= float(min_veto_ratio):
        return 0
    return commercial_pred


def evaluate_all_guard_modes_from_rows(rows, pred_key="parallel_pred", min_veto_windows=2, min_veto_ratio=0.4):
    results = {}
    y_true = [int(r.get("target", 0)) for r in rows]
    commercial_pred = [int(r.get("commercial_pred", 0)) for r in rows]
    for mode in GUARD_MODES:
        y_pred = [_row_pred_for_guard_mode(r, mode, min_veto_windows, min_veto_ratio) for r in rows]
        metrics = metric(y_true, y_pred) if rows else metric([], [])
        disagreements = [i for i, (c, p) in enumerate(zip(commercial_pred, y_pred)) if c != p]
        fixed = sum(1 for i in disagreements if y_pred[i] == y_true[i] and commercial_pred[i] != y_true[i])
        broken = sum(1 for i in disagreements if commercial_pred[i] == y_true[i] and y_pred[i] != y_true[i])
        results[mode] = {
            "metrics": metrics,
            "n_disagreements": int(len(disagreements)),
            "fixed": int(fixed),
            "broken": int(broken),
        }
    return results


def write_guard_mode_comparison(artifact_dir, rows, pred_key="parallel_pred", min_veto_windows=2, min_veto_ratio=0.4):
    comparison = evaluate_all_guard_modes_from_rows(
        rows,
        pred_key=pred_key,
        min_veto_windows=min_veto_windows,
        min_veto_ratio=min_veto_ratio,
    )
    out_rows = []
    for mode, payload in comparison.items():
        metrics = payload["metrics"]
        out_rows.append({
            "guard_mode": mode,
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "TN": metrics["confusion"]["TN"],
            "FP": metrics["confusion"]["FP"],
            "FN": metrics["confusion"]["FN"],
            "TP": metrics["confusion"]["TP"],
            "n_disagreements": payload["n_disagreements"],
            "fixed": payload["fixed"],
            "broken": payload["broken"],
        })
    pd.DataFrame(out_rows).to_csv(os.path.join(artifact_dir, "evaluation_guard_modes.csv"), index=False)
    with open(os.path.join(artifact_dir, "evaluation_guard_modes.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    return comparison


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


def _normal_window_mask(df):
    if "fallback" not in df.columns:
        return pd.Series(True, index=df.index)
    raw = df["fallback"].fillna(False)
    if raw.dtype == bool:
        return ~raw
    text = raw.astype(str).str.strip().str.lower()
    return ~text.isin(["1", "true", "yes", "y"])


def evaluate_cached_feature_rows(
    df,
    bundle,
    fcfg,
    guard_mode="shadow",
    min_veto_windows=2,
    min_veto_ratio=0.4,
):
    if len(df) == 0:
        return []
    if fcfg.get("chosen_strategy") != "veto":
        raise ValueError(f"unsupported commercial guard strategy: {fcfg.get('chosen_strategy')}")

    vp = fcfg.get("veto_params", {})
    p_c_high, p_n_low = float(vp.get("p_c_high", 0.8)), float(vp.get("p_n_low", 0.2))
    work = df.copy()
    work["P_c"] = commercial_probabilities_from_rows(work)
    work["P_n"] = predict_new_many(work, bundle)

    results = []
    for sn, group in work.groupby("sample_name"):
        target = int(pd.to_numeric(group["target"], errors="coerce").dropna().iloc[0])
        normal = group[_normal_window_mask(group)]
        if len(normal) == 0:
            results.append({"sample_name": sn, "target": target, "commercial_pred": 0,
                            "parallel_pred": 0, "bypass_pred": 0, "fallback": True})
            continue
        Pca, Pna = normal["P_c"].to_numpy(float), normal["P_n"].to_numpy(float)
        com_pred = int(np.mean(Pca) >= 0.5)
        Pf = fuse_veto(Pca, Pna, p_c_high, p_n_low)
        risk_series = 1.0 - Pna if com_pred else [0.0]
        decision = make_guard_decision(
            com_pred, risk_series, guard_mode=guard_mode, veto_threshold=1.0 - p_n_low,
            min_veto_windows=min_veto_windows, min_veto_ratio=min_veto_ratio
        )
        results.append({"sample_name": sn, "target": target, "commercial_pred": com_pred,
                        "parallel_pred": decision["final_pred"], "bypass_pred": com_pred,
                        "parallel_mean_prob": float(np.mean(Pf)), "veto_risk": decision["veto_risk"],
                        "risk_count": decision["risk_count"], "window_count": decision["window_count"],
                        "risk_ratio": decision["risk_ratio"], "guard_action": decision["guard_action"],
                        "decision_source": decision["decision_source"],
                        "guard_mode": guard_mode, "fallback": False})
    return results


def write_evaluation_outputs(artifact_dir, split, strat, guard_mode, min_veto_windows, min_veto_ratio, results):
    cm = metric([r["target"] for r in results], [r["commercial_pred"] for r in results])
    pm = metric([r["target"] for r in results], [r["parallel_pred"] for r in results])
    disc = [r for r in results if r["commercial_pred"] != r["parallel_pred"]]
    fixed = sum(1 for d in disc if d["parallel_pred"] == d["target"] and d["commercial_pred"] != d["target"])
    broken = sum(1 for d in disc if d["commercial_pred"] == d["target"] and d["parallel_pred"] != d["target"])
    print(f"Commercial: acc={cm['accuracy']:.4f} prec={cm['precision']:.4f} rec={cm['recall']:.4f} f1={cm['f1']:.4f}")
    print_confusion_matrix("Commercial baseline", cm)
    print(f"Parallel:   acc={pm['accuracy']:.4f} prec={pm['precision']:.4f} rec={pm['recall']:.4f} f1={pm['f1']:.4f}")
    print_confusion_matrix("Parallel final", pm)
    print(f"Disagreements: {len(disc)}/{len(results)} (fixed={fixed}, broken={broken})")
    guard_mode_comparison = write_guard_mode_comparison(
        artifact_dir, results, pred_key="parallel_pred",
        min_veto_windows=min_veto_windows, min_veto_ratio=min_veto_ratio,
    )
    report = {"split": split, "n": len(results), "fusion": strat, "guard_mode": guard_mode,
              "min_veto_windows": min_veto_windows, "min_veto_ratio": min_veto_ratio,
              "commercial": cm, "parallel": pm,
              "guard_mode_comparison": guard_mode_comparison,
              "bypass": cm, "n_disagreements": len(disc), "fixed": fixed, "broken": broken}
    with open(os.path.join(artifact_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(results).to_csv(os.path.join(artifact_dir, "evaluation_samples.csv"), index=False)
    write_error_sample_outputs(artifact_dir, results, "parallel_pred")
    pd.DataFrame([{"metric": m, "commercial": cm[m], "parallel": pm[m], "delta": pm[m] - cm[m]}
                  for m in ["accuracy", "precision", "recall", "f1"]])\
      .to_csv(os.path.join(artifact_dir, "evaluation_comparison.csv"), index=False)
    pd.DataFrame(
        confusion_matrix_rows("commercial", cm) + confusion_matrix_rows("parallel", pm)
    ).to_csv(os.path.join(artifact_dir, "evaluation_confusion_matrices.csv"), index=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--splits_dir", default="artifacts"); p.add_argument("--split", default="test")
    p.add_argument("--guard_mode", default="shadow", choices=GUARD_MODES)
    p.add_argument("--min_veto_windows", type=int, default=2)
    p.add_argument("--min_veto_ratio", type=float, default=0.4)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    bundle = joblib.load(os.path.join(args.artifact_dir, "new_model_bundle.pkl"))
    with open(os.path.join(args.artifact_dir, "fusion_config.json")) as f: fcfg = json.load(f)
    strat = fcfg["chosen_strategy"]; vp = fcfg.get("veto_params", {})
    t0 = time.time(); results = []
    feature_path = resolve_feature_pool_path(args.artifact_dir, args.split)
    if os.path.exists(feature_path):
        print(f"Using cached features: {feature_path}")
        results = evaluate_cached_feature_rows(
            pd.read_csv(feature_path), bundle, fcfg, guard_mode=args.guard_mode,
            min_veto_windows=args.min_veto_windows, min_veto_ratio=args.min_veto_ratio
        )
    else:
        splits = load_splits(args.splits_dir); samples = splits[args.split]
        com = OldLivenessModel()
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
                        is_live, score, _, _ = com.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, acc_seg))
                        Pc.append(float(is_live))
                        Pn.append(predict_new(extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS), bundle))
                    except Exception: Pc.append(0.0); Pn.append(0.5)
            else:
                ppg25, acc25, _ = _to_25hz(sample, ppg, acc); mode = detect_green_mode(ppg)
                sw, ss = int(round(COMMERCIAL_WIN_SEC * FEATURE_FS)), int(round(COMMERCIAL_STRIDE_SEC * FEATURE_FS))
                for step in range(3, max(0, (len(ppg25) - sw) // ss + 1)):
                    start = step * ss
                    win = ppg25[start:start + sw, :]
                    try:
                        ir, amb, g1, g2, g3 = get_channels_from_window(win, mode)
                        is_live, score, _, _ = com.predict_raw(
                            extract_8_commercial_features(ir, amb, g1, g2, g3, _slice_acc(acc25, start, sw))
                        )
                        Pc.append(float(is_live))
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
    write_evaluation_outputs(
        args.artifact_dir, args.split, strat, args.guard_mode,
        args.min_veto_windows, args.min_veto_ratio, results
    )
    print("Done")

if __name__ == "__main__": main()
