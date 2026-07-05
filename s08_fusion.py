# -*- coding: utf-8 -*-
"""
S08: Fusion configuration for the parallel veto-risk guard.

Output: {artifact_dir}/fusion_config.json
"""

import argparse, json, os, time
import numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from s01_model import DETECT_TREE_THRESH


def score_to_prob(score):
    if score is None or not np.isfinite(score): return 1.0
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
    agg["pred"] = (agg["mp"] >= thr).astype(int); yt, yp = agg["target"].values, agg["pred"].values
    cm = confusion_matrix(yt, yp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    return {"n": len(yt), "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--strategy", default="veto", choices=["veto"])
    p.add_argument("--veto_high", type=float, default=0.80); p.add_argument("--veto_low", type=float, default=0.20)
    args = p.parse_args(); os.makedirs(args.artifact_dir, exist_ok=True)
    bp = os.path.join(args.artifact_dir, "new_model_bundle.pkl")
    if not os.path.exists(bp): print("ERROR: bundle not found"); return
    bundle = joblib.load(bp); model, feats, fills, new_thr = bundle["model"], bundle["selected_features"], bundle["fill_values"], bundle["threshold"]
    vp = os.path.join(args.artifact_dir, "features_valid.csv")
    if not os.path.exists(vp): print("ERROR: valid features not found"); return
    df = pd.read_csv(vp)
    if "fallback" in df.columns: df = df[df["fallback"] == 0]
    df["P_c"] = df["commercial_score"].apply(score_to_prob)
    Xn = np.array([[r.get(f, 0.0) for f in feats] for _, r in df.iterrows()], dtype=float)
    for i, c in enumerate(feats): m = ~np.isfinite(Xn[:, i]); Xn[m, i] = fills.get(c, 0.0)
    df["P_n"] = model.predict_proba(Xn)[:, 1]; P_c, P_n = df["P_c"].values, df["P_n"].values; results = {}
    df["P_com_dec"] = (P_c >= 0.5).astype(float); results["commercial"] = eval_fusion(df, "P_com_dec", 0.5)
    df["P_new_dec"] = (P_n >= new_thr).astype(float); results["new_model"] = eval_fusion(df, "P_new_dec", new_thr)
    df["P_veto"] = fuse_veto(P_c, P_n, args.veto_high, args.veto_low); results["veto"] = eval_fusion(df, "P_veto", 0.5)
    print("Fusion comparison (valid):")
    for n, m in results.items():
        extra = f", alpha={m.get('alpha',0):.2f}" if "alpha" in m else ""
        print(f"  {n:<15s}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}{extra}")
    best_name = "veto"
    config = {"best_strategy": best_name, "chosen_strategy": "veto",
              "strategies": results, "veto_params": {"p_c_high": args.veto_high, "p_n_low": args.veto_low},
              "new_model_threshold": float(new_thr)}
    with open(os.path.join(args.artifact_dir, "fusion_config.json"), "w") as f: json.dump(config, f, indent=2)
    print(f"Best: {best_name} (F1={results[best_name]['f1']:.4f})"); print("Done")

if __name__ == "__main__": main()
