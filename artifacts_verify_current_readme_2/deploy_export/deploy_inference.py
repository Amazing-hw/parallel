# -*- coding: utf-8 -*-
"""
Minimal deployment reference for the exported cascade/parallel guard package.

This file documents the feature order, fill-value behavior, and JSON model load
path for engineering integration. It is intentionally small and independent of
the training pipeline.
"""

import json
from pathlib import Path

import numpy as np


def load_method(package_dir="."):
    package_dir = Path(package_dir)
    with open(package_dir / "method.json", encoding="utf-8") as f:
        return json.load(f)


def build_feature_vector(feature_dict, method):
    values = []
    fills = method.get("fill_values", {})
    for name in method["selected_features"]:
        raw = feature_dict.get(name, fills.get(name, 0.0))
        try:
            value = float(raw)
        except Exception:
            value = float(fills.get(name, 0.0))
        if not np.isfinite(value):
            value = float(fills.get(name, 0.0))
        values.append(value)
    return np.asarray(values, dtype=float).reshape(1, -1)


def predict_guard_probability(feature_dict, package_dir="."):
    package_dir = Path(package_dir)
    method = load_method(package_dir)
    constant_probability = method["model"].get("constant_probability")
    if constant_probability is not None:
        return float(constant_probability)

    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(package_dir / "model.json"))
    x = build_feature_vector(feature_dict, method)
    return float(booster.predict(xgb.DMatrix(x))[0])


def apply_guard(commercial_pred, feature_dict, package_dir=".", guard_mode=None):
    method = load_method(package_dir)
    guard_mode = guard_mode or method["guard"]["default_mode"]
    p_new = predict_guard_probability(feature_dict, package_dir)
    commercial_pred = int(commercial_pred)

    if method["project_type"] == "parallel":
        veto = method["parallel"]["veto_params"]
        risk = 1.0 - p_new
        should_veto = commercial_pred == 1 and risk >= (1.0 - float(veto["p_n_low"]))
        if guard_mode in ("bypass", "shadow", "soft_guard") or not should_veto:
            return {"final_pred": commercial_pred, "new_probability": p_new, "guard_action": "record" if should_veto else "pass"}
        if guard_mode == "hard_veto":
            return {"final_pred": 0, "new_probability": p_new, "guard_action": "hard_veto"}
        return {"final_pred": commercial_pred, "new_probability": p_new, "guard_action": "pass"}

    raise ValueError(f"unsupported project_type for this package: {method['project_type']}")
