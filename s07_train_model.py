# -*- coding: utf-8 -*-
"""
S07: Train tiny independent XGBoost on ALL data.

Output: {artifact_dir}/new_model.json, new_model_bundle.pkl
"""

import argparse, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

LEAKAGE_FEATURES = {
    "target",
    "should_veto",
    "commercial_pred",
    "is_error",
    "fallback",
}


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
    tp = os.path.join(args.artifact_dir, "features_train.csv")
    vp = os.path.join(args.artifact_dir, "features_valid.csv")
    if not os.path.exists(tp): print("ERROR: train not found"); sys.exit(1)
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
           "selected_features": feats, "threshold": float(thr),
           "fill_values": {k: float(v) for k, v in fills.items()}, "train_metrics": tm, "valid_metrics": vm}
    joblib.dump({"model": model, "selected_features": feats, "threshold": thr, "fill_values": fills, "config": cfg},
                os.path.join(args.artifact_dir, "new_model_bundle.pkl"))
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__": main()
