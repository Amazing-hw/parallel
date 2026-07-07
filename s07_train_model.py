# -*- coding: utf-8 -*-
"""
S07: Train tiny independent XGBoost on ALL data.

Output: {artifact_dir}/new_model.json, new_model_bundle.pkl
"""

import argparse, hashlib, json, os, platform, sys, time
import numpy as np, pandas as pd, xgboost as xgb, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

LEAKAGE_FEATURES = {
    "target",
    "should_veto",
    "commercial_pred",
    "is_error",
    "fallback",
}


def sha256_head(path, head_bytes=4 * 1024 * 1024):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()


def build_training_fingerprint(artifact_dir, feature_pool_train_path=None, splits_path=None):
    fingerprint = {
        "train_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xgboost": xgb.__version__,
        "splits_sha256_head": sha256_head(splits_path or os.path.join(os.path.dirname(artifact_dir), "splits.json")),
        "feature_pool_train_sha256_head": sha256_head(
            feature_pool_train_path or os.path.join(artifact_dir, "feature_pool_train.csv")
        ),
        "selection_policy": {
            "selection_data": "train_only",
            "valid_used_for_selection": False,
            "test_used_for_selection": False,
            "test_role": "final_closed_evaluation_only",
        },
    }
    return fingerprint


def resolve_feature_pool_path(artifact_dir, split):
    standard = os.path.join(artifact_dir, f"feature_pool_{split}.csv")
    legacy = os.path.join(artifact_dir, f"features_{split}.csv")
    return standard if os.path.exists(standard) else legacy


def resolve_feature_list(auto_path, manual_path=None):
    source = {"source": "auto", "path": auto_path}
    path = auto_path
    if manual_path and os.path.exists(manual_path):
        path = manual_path
        source = {"source": "manual", "path": manual_path}
    elif manual_path:
        print(f"[WARN] manual feature file not found, using auto: {manual_path}")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    features = [str(x) for x in payload.get("selected_features", [])]
    if not features:
        raise ValueError(f"no selected_features in {path}")
    leaked = [f for f in features if f in LEAKAGE_FEATURES]
    if leaked:
        raise ValueError(f"label leakage features are not allowed in selected_features: {leaked}")
    source["payload"] = payload
    return features, source


def prepare(df, features):
    X = df[features].values.astype(float); y = df["target"].values.astype(int); fills = {}
    for i, c in enumerate(features):
        ok = np.isfinite(X[:, i]); fills[c] = float(np.median(X[:, i][ok])) if ok.sum() > 0 else 0.0
        X[~ok, i] = fills[c]
    return X, y, fills


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--n_estimators", type=int, default=10); p.add_argument("--max_depth", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--manual_features", default=None)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    feats, feature_source = resolve_feature_list(
        os.path.join(args.artifact_dir, "selected_features.json"), args.manual_features
    )
    tp = resolve_feature_pool_path(args.artifact_dir, "train")
    vp = resolve_feature_pool_path(args.artifact_dir, "valid")
    if not os.path.exists(tp): print("ERROR: train not found"); sys.exit(1)
    fingerprint = build_training_fingerprint(args.artifact_dir, tp)
    with open(os.path.join(args.artifact_dir, "model_fingerprint.json"), "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2, ensure_ascii=False)
    dt = pd.read_csv(tp); dv = pd.read_csv(vp) if os.path.exists(vp) else dt.copy()
    if "fallback" in dt.columns: dt = dt[dt["fallback"] == 0]
    if "fallback" in dv.columns: dv = dv[dv["fallback"] == 0]
    np_, nn_ = int(dt["target"].sum()), len(dt) - int(dt["target"].sum())
    sw = max(0.5, nn_ / max(1, np_)) if np_ > 0 else 1.0
    Xt, yt, fills = prepare(dt, feats); Xv, yv, _ = prepare(dv, feats); t0 = time.time()
    model = xgb.XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                              learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=20, reg_lambda=10, reg_alpha=1,
                              objective="binary:logistic", eval_metric="logloss",
                              random_state=42, scale_pos_weight=sw, n_jobs=1)
    model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    pv = model.predict_proba(Xv)[:, 1]; best = {"threshold": 0.5, "score": -np.inf}
    for t in np.linspace(0.05, 0.95, 91):
        s = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if s > best["score"]: best = {"threshold": float(t), "score": float(s)}
    thr = best["threshold"]
    cm = confusion_matrix(yt, model.predict_proba(Xt)[:, 1] >= thr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel(); tm = {"auc": float(roc_auc_score(yt, model.predict_proba(Xt)[:, 1])) if len(np.unique(yt))>1 else 0.5}
    cm = confusion_matrix(yv, pv >= thr, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    vm = {"accuracy": float(accuracy_score(yv, (pv >= thr).astype(int))),
          "f1": float(f1_score(yv, (pv >= thr).astype(int), zero_division=0)),
          "auc": float(roc_auc_score(yv, pv)) if len(np.unique(yv))>1 else 0.5}
    nn = sum(1 for l in model.get_booster().get_dump() if "leaf" in l or "yes=" in l)
    print(f"Trees={args.n_estimators} Depth={args.max_depth} Nodes={nn} Thr={thr:.3f}")
    print(f"Train AUC={tm['auc']:.4f}  Valid AUC={vm['auc']:.4f} F1={vm['f1']:.4f}")
    model.get_booster().save_model(os.path.join(args.artifact_dir, "new_model.json"))
    cfg = {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "n_nodes": nn,
           "feature_source": feature_source,
           "fingerprint": fingerprint,
           "selected_features": feats, "threshold": float(thr),
           "fill_values": {k: float(v) for k, v in fills.items()}, "train_metrics": tm, "valid_metrics": vm}
    joblib.dump({"model": model, "selected_features": feats, "threshold": thr, "fill_values": fills,
                 "fingerprint": fingerprint, "config": cfg},
                os.path.join(args.artifact_dir, "new_model_bundle.pkl"))
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__": main()
