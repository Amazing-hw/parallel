# -*- coding: utf-8 -*-
"""
S07: Train tiny independent XGBoost on ALL data.

Output: {artifact_dir}/new_model.json, new_model_bundle.pkl
"""

import argparse, hashlib, json, os, platform, sys, time
from itertools import product
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


def parse_model_search_values(raw, cast, name):
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _freeze_params(params):
    return tuple(sorted(params.items()))


def build_xgb_params(scale_pos_weight, n_estimators=10, max_depth=2, learning_rate=0.05,
                     min_child_weight=20, reg_lambda=10, reg_alpha=1, n_jobs=1):
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": int(min_child_weight),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "scale_pos_weight": float(scale_pos_weight),
        "n_jobs": max(1, int(n_jobs)),
        "verbosity": 0,
    }


def build_model_search_candidates(args, scale_pos_weight=1.0):
    axes = {
        "n_estimators": parse_model_search_values(args.model_search_n_estimators, int, "model_search_n_estimators"),
        "max_depth": parse_model_search_values(args.model_search_max_depth, int, "model_search_max_depth"),
        "learning_rate": parse_model_search_values(args.model_search_learning_rate, float, "model_search_learning_rate"),
        "min_child_weight": parse_model_search_values(args.model_search_min_child_weight, int, "model_search_min_child_weight"),
        "reg_lambda": parse_model_search_values(args.model_search_reg_lambda, float, "model_search_reg_lambda"),
        "reg_alpha": parse_model_search_values(args.model_search_reg_alpha, float, "model_search_reg_alpha"),
    }
    keys = list(axes.keys())
    default_params = build_xgb_params(
        scale_pos_weight,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_jobs=getattr(args, "n_jobs", 1),
    )
    grid = []
    for values in product(*(axes[k] for k in keys)):
        params = build_xgb_params(scale_pos_weight, n_jobs=getattr(args, "n_jobs", 1))
        params.update(dict(zip(keys, values)))
        grid.append(params)
    max_candidates = max(1, int(args.model_search_max_candidates))
    if len(grid) > max_candidates:
        rng = np.random.default_rng(int(args.model_search_random_state))
        keep = sorted(rng.choice(len(grid), size=max_candidates, replace=False).tolist())
        grid = [grid[i] for i in keep]
    grid.append(default_params)
    seen = set()
    candidates = []
    for params in grid:
        frozen = _freeze_params(params)
        if frozen in seen:
            continue
        seen.add(frozen)
        candidates.append({
            "rank_input_order": len(candidates) + 1,
            "params": params,
            "is_default_params": frozen == _freeze_params(default_params),
        })
    return candidates


def count_xgb_nodes(model):
    return sum(1 for dump in model.get_booster().get_dump() for line in dump.splitlines() if "leaf" in line or "yes=" in line)


def eval_binary(model, X, y, thr):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
    }


def search_f1_threshold(y_true, prob):
    best = {"threshold": 0.5, "score": -np.inf, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best["score"]:
            best = {"threshold": float(t), "score": f1, "precision": precision, "recall": recall, "f1": f1}
    return best


def evaluate_model_search_candidate(candidate, X_train, y_train, X_valid, y_valid, size_cost=0.002):
    model = xgb.XGBClassifier(**candidate["params"])
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    prob = model.predict_proba(X_valid)[:, 1]
    threshold = search_f1_threshold(y_valid, prob)
    metrics = eval_binary(model, X_valid, y_valid, threshold["threshold"])
    nodes = count_xgb_nodes(model)
    score = float(metrics["f1"]) + 0.05 * float(metrics["auc"]) - float(size_cost) * nodes
    record = {
        "rank_input_order": candidate["rank_input_order"],
        "is_default_params": candidate["is_default_params"],
        "score": float(score),
        "threshold": float(threshold["threshold"]),
        "threshold_selection": threshold,
        "valid_metrics": metrics,
        "n_nodes": int(nodes),
        "params": candidate["params"],
    }
    return model, record


def select_best_model_search_record(records):
    return sorted(
        records,
        key=lambda r: (
            -float(r["score"]),
            int(r["n_nodes"]),
            not bool(r["is_default_params"]),
            int(r["rank_input_order"]),
        ),
    )[0]


def write_model_search_outputs(artifact_dir, records, best_record):
    rows = []
    safe_records = []
    for record in records:
        safe = dict(record)
        safe_records.append(safe)
        row = {
            "rank_input_order": record["rank_input_order"],
            "is_default_params": record["is_default_params"],
            "chosen": record is best_record,
            "score": record["score"],
            "threshold": record["threshold"],
            "n_nodes": record["n_nodes"],
            "valid_accuracy": record["valid_metrics"]["accuracy"],
            "valid_precision": record["valid_metrics"]["precision"],
            "valid_recall": record["valid_metrics"]["recall"],
            "valid_f1": record["valid_metrics"]["f1"],
            "valid_auc": record["valid_metrics"]["auc"],
        }
        row.update({f"param_{k}": v for k, v in record["params"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(artifact_dir, "model_search_results.csv"), index=False)
    summary = {
        "enabled": True,
        "selection_objective": "valid_f1_auc_with_size_penalty",
        "candidate_count": len(records),
        "best": best_record,
        "top_candidates": sorted(safe_records, key=lambda r: r["score"], reverse=True)[:10],
    }
    with open(os.path.join(artifact_dir, "model_search_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def search_model(args, X_train, y_train, X_valid, y_valid, scale_pos_weight):
    candidates = build_model_search_candidates(args, scale_pos_weight=scale_pos_weight)
    models = []
    records = []
    print(f"Model search: {len(candidates)} tiny candidates")
    for i, candidate in enumerate(candidates, start=1):
        model, record = evaluate_model_search_candidate(
            candidate, X_train, y_train, X_valid, y_valid,
            size_cost=args.model_search_size_cost,
        )
        models.append(model)
        records.append(record)
        print(
            f"  [{i}/{len(candidates)}] trees={record['params']['n_estimators']} "
            f"depth={record['params']['max_depth']} score={record['score']:.4f} "
            f"nodes={record['n_nodes']} f1={record['valid_metrics']['f1']:.4f}"
        )
    best_record = select_best_model_search_record(records)
    best_idx = records.index(best_record)
    summary = write_model_search_outputs(args.artifact_dir, records, best_record)
    return models[best_idx], best_record, summary


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--n_estimators", type=int, default=10); p.add_argument("--max_depth", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--model_search_n_estimators", default="6,8,10,12,16,20")
    p.add_argument("--model_search_max_depth", default="1,2,3")
    p.add_argument("--model_search_learning_rate", default="0.03,0.05")
    p.add_argument("--model_search_min_child_weight", default="10,20")
    p.add_argument("--model_search_reg_lambda", default="5,10")
    p.add_argument("--model_search_reg_alpha", default="0,1")
    p.add_argument("--model_search_max_candidates", type=int, default=32)
    p.add_argument("--model_search_random_state", type=int, default=42)
    p.add_argument("--model_search_size_cost", type=float, default=0.002)
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
    model, best_record, model_search_summary = search_model(args, Xt, yt, Xv, yv, sw)
    pv = model.predict_proba(Xv)[:, 1]
    best = best_record["threshold_selection"]
    thr = best_record["threshold"]
    cm = confusion_matrix(yt, model.predict_proba(Xt)[:, 1] >= thr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel(); tm = {"auc": float(roc_auc_score(yt, model.predict_proba(Xt)[:, 1])) if len(np.unique(yt))>1 else 0.5}
    cm = confusion_matrix(yv, pv >= thr, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    vm = {"accuracy": float(accuracy_score(yv, (pv >= thr).astype(int))),
          "f1": float(f1_score(yv, (pv >= thr).astype(int), zero_division=0)),
          "auc": float(roc_auc_score(yv, pv)) if len(np.unique(yv))>1 else 0.5}
    nn = count_xgb_nodes(model)
    best_params = best_record["params"]
    print(f"Trees={best_params['n_estimators']} Depth={best_params['max_depth']} Nodes={nn} Thr={thr:.3f}")
    print(f"Train AUC={tm['auc']:.4f}  Valid AUC={vm['auc']:.4f} F1={vm['f1']:.4f}")
    model.get_booster().save_model(os.path.join(args.artifact_dir, "new_model.json"))
    cfg = {"n_estimators": int(best_params["n_estimators"]), "max_depth": int(best_params["max_depth"]),
           "learning_rate": float(best_params["learning_rate"]), "n_jobs": max(1, int(args.n_jobs)), "n_nodes": nn,
           "model_search": model_search_summary,
           "feature_source": feature_source,
           "fingerprint": fingerprint,
           "selected_features": feats, "threshold": float(thr),
           "fill_values": {k: float(v) for k, v in fills.items()}, "train_metrics": tm, "valid_metrics": vm}
    joblib.dump({"model": model, "selected_features": feats, "threshold": thr, "fill_values": fills,
                 "fingerprint": fingerprint, "config": cfg},
                os.path.join(args.artifact_dir, "new_model_bundle.pkl"))
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__": main()
