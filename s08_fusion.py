# -*- coding: utf-8 -*-
"""
S08: Fusion configuration for the parallel veto-risk guard.

Output: {artifact_dir}/fusion_config.json
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from s01_model import DETECT_TREE_THRESH


def resolve_feature_pool_path(artifact_dir, split):
    standard = os.path.join(artifact_dir, f"feature_pool_{split}.csv")
    legacy = os.path.join(artifact_dir, f"features_{split}.csv")
    return standard if os.path.exists(standard) else legacy


def score_to_prob(score):
    if score is None or not np.isfinite(score):
        return 1.0
    z = (float(score) - DETECT_TREE_THRESH) / 5000.0
    return float(1.0 / (1.0 + np.exp(-max(-50.0, min(50.0, z)))))


def fuse_veto(P_c, P_n, hi=0.80, lo=0.20):
    P_c, P_n = np.asarray(P_c, float), np.asarray(P_n, float)
    r = P_c.copy()
    veto_mask = (P_c >= 0.5) & (P_n < lo)
    r[veto_mask] = P_n[veto_mask]
    return r


def eval_fusion(df, col, thr=0.5):
    agg = df.groupby("sample_name").agg(target=("target", "first"), mp=(col, "mean"))
    agg["pred"] = (agg["mp"] >= thr).astype(int)
    yt, yp = agg["target"].values, agg["pred"].values
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"n": len(yt), "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def metric_from_sample_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def _parse_grid(raw, cast):
    if isinstance(raw, (list, tuple)):
        return [cast(x) for x in raw]
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    if not values:
        raise ValueError(f"empty grid: {raw}")
    return values


def _sample_guard_frame(df):
    rows = []
    for sample_name, group in df.groupby("sample_name"):
        target = int(pd.to_numeric(group["target"], errors="coerce").dropna().iloc[0])
        p_c = pd.to_numeric(group["P_c"], errors="coerce").fillna(0.0).to_numpy(float)
        p_n = pd.to_numeric(group["P_n"], errors="coerce").fillna(0.5).to_numpy(float)
        commercial_pred = int(np.mean(p_c) >= 0.5)
        rows.append({
            "sample_name": sample_name,
            "target": target,
            "commercial_pred": commercial_pred,
            "P_n": p_n,
        })
    return rows


def evaluate_veto_guard_params(sample_rows, p_n_low, min_veto_windows, min_veto_ratio):
    threshold = 1.0 - float(p_n_low)
    y_true, y_commercial, y_final = [], [], []
    for row in sample_rows:
        risk = 1.0 - np.asarray(row["P_n"], dtype=float)
        high = risk >= threshold
        risk_count = int(np.sum(high))
        risk_ratio = float(np.mean(high)) if len(high) else 0.0
        commercial_pred = int(row["commercial_pred"])
        final_pred = commercial_pred
        if (
            commercial_pred == 1
            and risk_count >= int(min_veto_windows)
            and risk_ratio >= float(min_veto_ratio)
        ):
            final_pred = 0
        y_true.append(int(row["target"]))
        y_commercial.append(commercial_pred)
        y_final.append(final_pred)
    commercial_metrics = metric_from_sample_predictions(y_true, y_commercial)
    metrics = metric_from_sample_predictions(y_true, y_final)
    return metrics, commercial_metrics


def search_veto_guard_params(
    df,
    p_n_lows=None,
    min_veto_windows_values=None,
    min_veto_ratio_values=None,
    max_fn_increase=1,
):
    p_n_lows = _parse_grid(p_n_lows or "0.05,0.10,0.15,0.20,0.25,0.30", float)
    min_veto_windows_values = _parse_grid(min_veto_windows_values or "1,2,3", int)
    min_veto_ratio_values = _parse_grid(min_veto_ratio_values or "0.2,0.3,0.4,0.5", float)
    sample_rows = _sample_guard_frame(df)
    if not sample_rows:
        raise ValueError("no valid rows for veto guard search")

    records = []
    for p_n_low in p_n_lows:
        for min_veto_windows in min_veto_windows_values:
            for min_veto_ratio in min_veto_ratio_values:
                metrics, commercial = evaluate_veto_guard_params(
                    sample_rows, p_n_low, min_veto_windows, min_veto_ratio
                )
                fp_reduction = commercial["confusion"]["FP"] - metrics["confusion"]["FP"]
                fn_increase = metrics["confusion"]["FN"] - commercial["confusion"]["FN"]
                score = (
                    10.0 * fp_reduction
                    - 6.0 * max(0, fn_increase)
                    + float(metrics["accuracy"])
                    + 0.2 * float(metrics["f1"])
                )
                if fn_increase > int(max_fn_increase):
                    score -= 100.0 + 10.0 * (fn_increase - int(max_fn_increase))
                records.append({
                    "p_c_high": 0.8,
                    "p_n_low": float(p_n_low),
                    "min_veto_windows": int(min_veto_windows),
                    "min_veto_ratio": float(min_veto_ratio),
                    "score": float(score),
                    "fp_reduction": int(fp_reduction),
                    "fn_increase": int(fn_increase),
                    "metrics": metrics,
                    "commercial_metrics": commercial,
                })
    best = sorted(
        records,
        key=lambda r: (
            -float(r["score"]),
            int(r["fn_increase"]),
            -int(r["fp_reduction"]),
            int(r["min_veto_windows"]),
            float(r["min_veto_ratio"]),
            float(r["p_n_low"]),
        ),
    )[0]
    return records, best


def build_feature_matrix(df, features, fills):
    if not features:
        return np.empty((len(df), 0), dtype=float)
    matrix = df.reindex(columns=features)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    X = np.array(matrix.to_numpy(dtype=float, copy=True), dtype=float, copy=True)
    for i, feature in enumerate(features):
        invalid = ~np.isfinite(X[:, i])
        X[invalid, i] = fills.get(feature, 0.0)
    return X


def evaluate_strategies(df, model, features, fills, new_threshold, veto_high, veto_low):
    df = df.copy()
    df["P_c"] = df["commercial_score"].apply(score_to_prob)
    df["P_n"] = model.predict_proba(build_feature_matrix(df, features, fills))[:, 1]

    P_c = df["P_c"].values
    P_n = df["P_n"].values
    df["P_com_dec"] = (P_c >= 0.5).astype(float)
    df["P_new_dec"] = (P_n >= new_threshold).astype(float)
    df["P_veto"] = fuse_veto(P_c, P_n, veto_high, veto_low)

    return {
        "commercial": eval_fusion(df, "P_com_dec", 0.5),
        "new_model": eval_fusion(df, "P_new_dec", new_threshold),
        "veto": eval_fusion(df, "P_veto", 0.5),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--strategy", default="veto", choices=["veto"])
    p.add_argument("--veto_high", type=float, default=0.80)
    p.add_argument("--veto_low", type=float, default=0.20)
    p.add_argument("--search_p_n_lows", default="0.05,0.10,0.15,0.20,0.25,0.30")
    p.add_argument("--search_min_veto_windows", default="1,2,3")
    p.add_argument("--search_min_veto_ratios", default="0.2,0.3,0.4,0.5")
    p.add_argument("--max_fn_increase", type=int, default=1)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)

    bp = os.path.join(args.artifact_dir, "new_model_bundle.pkl")
    if not os.path.exists(bp):
        print("ERROR: bundle not found")
        return
    bundle = joblib.load(bp)
    model = bundle["model"]
    feats = bundle["selected_features"]
    fills = bundle["fill_values"]
    new_thr = bundle["threshold"]

    vp = resolve_feature_pool_path(args.artifact_dir, "valid")
    if not os.path.exists(vp):
        print("ERROR: valid features not found")
        return
    df = pd.read_csv(vp)
    if "fallback" in df.columns:
        df = df[df["fallback"] == 0]
    if df.empty:
        print(
            f"ERROR: {vp} has no non-fallback validation rows. "
            "Run S05 on valid H5 data, or inspect feature_pool_valid.csv and fallback_reason."
        )
        sys.exit(1)

    search_df = df.copy()
    search_df["P_c"] = search_df["commercial_score"].apply(score_to_prob)
    search_df["P_n"] = model.predict_proba(build_feature_matrix(search_df, feats, fills))[:, 1]
    search_records, best_guard = search_veto_guard_params(
        search_df,
        p_n_lows=args.search_p_n_lows,
        min_veto_windows_values=args.search_min_veto_windows,
        min_veto_ratio_values=args.search_min_veto_ratios,
        max_fn_increase=args.max_fn_increase,
    )
    results = evaluate_strategies(df, model, feats, fills, new_thr, args.veto_high, best_guard["p_n_low"])
    results["veto_guard"] = best_guard["metrics"]
    print("Fusion comparison (valid):")
    for n, m in results.items():
        extra = f", alpha={m.get('alpha',0):.2f}" if "alpha" in m else ""
        print(f"  {n:<15s}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}{extra}")
    best_name = "veto"
    print(
        "Best veto guard params: "
        f"p_n_low={best_guard['p_n_low']:.3f}, "
        f"min_veto_windows={best_guard['min_veto_windows']}, "
        f"min_veto_ratio={best_guard['min_veto_ratio']:.3f}, "
        f"FP_reduction={best_guard['fp_reduction']}, FN_increase={best_guard['fn_increase']}"
    )
    config = {"best_strategy": best_name, "chosen_strategy": "veto",
              "strategies": results,
              "veto_params": {"p_c_high": args.veto_high, "p_n_low": best_guard["p_n_low"],
                              "min_veto_windows": best_guard["min_veto_windows"],
                              "min_veto_ratio": best_guard["min_veto_ratio"]},
              "veto_search": {"enabled": True, "max_fn_increase": args.max_fn_increase,
                              "selection_objective": "minimize false-wear FP first, constrain FN increase, then accuracy/F1",
                              "best": best_guard, "candidate_count": len(search_records),
                              "grid": {"p_n_lows": _parse_grid(args.search_p_n_lows, float),
                                       "min_veto_windows": _parse_grid(args.search_min_veto_windows, int),
                                       "min_veto_ratios": _parse_grid(args.search_min_veto_ratios, float)}},
              "new_model_threshold": float(new_thr)}
    pd.DataFrame(search_records).to_csv(os.path.join(args.artifact_dir, "fusion_search_results.csv"), index=False)
    with open(os.path.join(args.artifact_dir, "fusion_search_results.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best_guard, "records": search_records}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.artifact_dir, "fusion_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Best: {best_name} (F1={results[best_name]['f1']:.4f})")
    print("Done")

if __name__ == "__main__":
    main()
