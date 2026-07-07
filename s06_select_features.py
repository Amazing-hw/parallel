# -*- coding: utf-8 -*-
"""
S06: Feature selection on ALL data (independent of commercial).

Output: {artifact_dir}/selected_features.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s03_selection import (
    stability_selection, select_by_group_from_combined,
    get_feature_cols, fast_group_preselection, clean_features_by_train,
    feature_to_group, add_deployment_scores,
)

LEAKAGE_AND_META_COLUMNS = {
    "sample_name",
    "target",
    "should_veto",
    "commercial_pred",
    "window_idx",
    "commercial_score",
    "is_error",
    "fallback",
    "fallback_reason",
    "h5_file",
    "Unnamed: 0",
}


SELECTION_CACHE_VERSION = 1


def get_candidate_feature_cols(df):
    return [
        c for c in df.columns
        if c not in LEAKAGE_AND_META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _safe_auc(y, values):
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = y.notna() & x.notna()
    if ok.sum() < 3 or y[ok].nunique() < 2:
        return 0.5
    try:
        auc = float(roc_auc_score(y[ok].astype(int), x[ok].astype(float)))
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def feature_selection_cache_params(max_features, n_workers=None):
    return {
        "script": "parallel_s06_select_features",
        "version": SELECTION_CACHE_VERSION,
        "max_features": int(max_features),
        "missing_thresh": 0.5,
        "corr_thresh": 0.95,
        "preselect_top": 6,
        "seeds": [1, 7, 42],
        "min_fold_auc": 0.55,
        "deployment_score_weight": 0.15,
        "commercial_score_included": False,
        "n_workers": None if n_workers is None else int(n_workers),
    }


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_feature_selection_cache_key(input_paths, params):
    inputs = []
    for path in input_paths:
        if path and os.path.exists(path):
            inputs.append({"name": os.path.basename(path), "sha256": _file_sha256(path)})
    payload = {"inputs": inputs, "params": params}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def load_feature_selection_cache(artifact_dir, input_paths, params):
    selected_path = os.path.join(artifact_dir, "selected_features.json")
    cache_path = os.path.join(artifact_dir, "feature_review", "selection_cache.json")
    if not (os.path.exists(selected_path) and os.path.exists(cache_path)):
        return None
    cache_key, _ = compute_feature_selection_cache_key(input_paths, params)
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("cache_key") != cache_key:
            return None
        with open(selected_path, "r", encoding="utf-8") as f:
            selected_payload = json.load(f)
    except Exception:
        return None
    cache["selected_features"] = list(selected_payload.get("selected_features", []))
    return cache


def write_feature_selection_cache(artifact_dir, input_paths, params, selected_features):
    out_dir = os.path.join(artifact_dir, "feature_review")
    os.makedirs(out_dir, exist_ok=True)
    cache_key, payload = compute_feature_selection_cache_key(input_paths, params)
    cache = {
        "cache_key": cache_key,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_features": list(selected_features),
        "payload": payload,
    }
    with open(os.path.join(out_dir, "selection_cache.json"), "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    return cache


def write_feature_review(artifact_dir, train_df, valid_df, ranked, selected_features, max_features):
    out_dir = os.path.join(artifact_dir, "feature_review")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    selected_set = set(selected_features)
    for i, item in enumerate(ranked, start=1):
        feature = item["feature"]
        train_values = train_df[feature] if feature in train_df else pd.Series(dtype=float)
        valid_values = valid_df[feature] if feature in valid_df else pd.Series(dtype=float)
        auc_train = _safe_auc(train_df["target"], train_values)
        auc_valid = _safe_auc(valid_df["target"], valid_values)
        missing_rate = float(pd.to_numeric(train_values, errors="coerce").isna().mean()) if len(train_values) else 1.0
        stability = 1.0 - abs(auc_train - auc_valid)
        rows.append({
            "rank": i,
            "feature": feature,
            "group": item.get("group", feature_to_group(feature)),
            "auc_train": auc_train,
            "auc_valid": auc_valid,
            "stability_score": float(stability),
            "missing_rate": missing_rate,
            "freq": float(item.get("freq", 0.0)),
            "avg_importance": float(item.get("avg_importance", 0.0)),
            "combined_score": float(item.get("combined_score", 0.0)),
            "selected_auto": feature in selected_set,
            "recommendation": "keep" if feature in selected_set else "review",
        })
    ranked_df = pd.DataFrame(rows)
    ranked_df.to_csv(os.path.join(out_dir, "ranked_features.csv"), index=False)
    with open(os.path.join(out_dir, "ranked_features.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, "ranked_features.md"), "w", encoding="utf-8") as f:
        f.write("# Ranked Features\n\n")
        f.write("| rank | feature | auc_train | auc_valid | stability | selected_auto |\n")
        f.write("|---:|---|---:|---:|---:|---|\n")
        for row in rows[:max(30, max_features)]:
            f.write(f"| {row['rank']} | {row['feature']} | {row['auc_train']:.4f} | {row['auc_valid']:.4f} | {row['stability_score']:.4f} | {row['selected_auto']} |\n")
    template = {
        "selected_features": selected_features,
        "excluded_features": {},
        "notes": "Edit selected_features, save as manual_feature_selection.json, then pass --manual_features to training.",
    }
    with open(os.path.join(out_dir, "manual_feature_selection_template.json"), "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--max_features", type=int, default=12)
    p.add_argument("--n_workers", type=int, default=None)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    tp = os.path.join(args.artifact_dir, "features_train.csv")
    if not os.path.exists(tp):
        print(f"ERROR: {tp} not found")
        sys.exit(1)
    vp = os.path.join(args.artifact_dir, "features_valid.csv")
    input_paths = [tp] + ([vp] if os.path.exists(vp) else [])
    cache_params = feature_selection_cache_params(args.max_features, n_workers=args.n_workers)
    cache = load_feature_selection_cache(args.artifact_dir, input_paths, cache_params)
    if cache is not None:
        print(f"Feature selection cache hit: {cache['cache_key']}")
        print(f"Selected ({len(cache['selected_features'])}):")
        for i, feature in enumerate(cache["selected_features"], start=1):
            print(f"  {i}. {feature}")
        print("Done (0.0s)")
        return
    t0 = time.time()
    dt = pd.read_csv(tp)
    dv = pd.read_csv(vp) if os.path.exists(vp) else dt.copy()
    fcols = get_candidate_feature_cols(dt)
    dtc, dvc, kept, _, _ = clean_features_by_train(dt, dv, fcols, missing_thresh=0.5, corr_thresh=0.95, skip_vif=True)
    presel = fast_group_preselection(dtc, kept, preselect_top=6)
    presel_cols = sorted(presel.keys())
    stab = stability_selection(dtc, presel_cols, max_splits=min(5, dtc["sample_name"].nunique()),
                               seeds=[1, 7, 42], n_workers=args.n_workers, min_fold_auc=0.55)
    if not stab:
        stab = [{"feature": f, "freq": 1.0, "avg_importance": 0.01, "avg_rank": i, "group": feature_to_group(f)}
                for i, f in enumerate(presel_cols)]
    for row in stab:
        row["combined_score"] = row.get("freq", 0.5) * row.get("avg_importance", 0.01)
    stab.sort(key=lambda r: r["combined_score"], reverse=True)
    stab = add_deployment_scores(stab, deployment_score_weight=0.15)
    sel = select_by_group_from_combined(stab, max_features=args.max_features, min_acc_features=1)
    feats = sel[0] if isinstance(sel, tuple) else sel
    feats = [f for f in feats if f != "commercial_score"]
    write_feature_review(args.artifact_dir, dtc, dvc, stab, feats, args.max_features)
    print(f"Selected ({len(feats)}):")
    for i, feature in enumerate(feats, start=1):
        print(f"  {i}. {feature}")
    with open(os.path.join(args.artifact_dir, "selected_features.json"), "w") as f:
        json.dump({"selected_features": feats, "n_features": len(feats), "commercial_score_included": False}, f, indent=2)
    write_feature_selection_cache(args.artifact_dir, input_paths, cache_params, feats)
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
