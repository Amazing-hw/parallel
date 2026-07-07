# -*- coding: utf-8 -*-
"""
S12: Export a self-contained deployment handoff package for the parallel guard.

Output: {artifact_dir}/deploy_export by default.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib


PROJECT_TYPE = "parallel"
MODEL_SOURCE = "new_model.json"
BUNDLE_SOURCE = "new_model_bundle.pkl"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def sanitize_config(config):
    if not isinstance(config, dict):
        return {}
    cleaned = json.loads(json.dumps(config, ensure_ascii=False, default=str))
    feature_source = cleaned.get("feature_source")
    if isinstance(feature_source, dict) and "path" in feature_source:
        feature_source["path"] = os.path.basename(str(feature_source["path"]))
    return cleaned


def deploy_inference_source():
    return r'''# -*- coding: utf-8 -*-
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
'''


def readme_text():
    return """# Parallel 部署交接包

这个目录是从训练/分析工程中导出的独立部署交接包，可以直接交给工程化同事。

## 文件说明

- `model.json`：新增并联 XGBoost 模型 JSON。
- `method.json`：部署方法配置，包含特征顺序、填充值、阈值、guard 模式、fusion/veto 参数。
- `selected_features.json`：训练时确认的最终特征列表。
- `fill_values.json`：每个特征的缺失值/异常值填充值。
- `feature_extractor.py`：部署侧参考特征提取脚本，来源于项目 `s02_features.py`。
- `s02_features.py`：兼容文件名，供 `commercial_model.py` 内部导入使用。
- `commercial_model.py`：冻结商用模型脚本，来源于项目 `s01_model.py`。
- `commercial_model_manifest.json`：商用模型冻结证据，用于核对树参数和特征是否变化。
- `deploy_inference.py`：最小 Python 推理参考，用于说明模型加载、特征顺序和 veto 逻辑。
- `deploy_manifest.json`：导出清单和文件 SHA256。

## 工程化重点

1. 商用模型仍然保留，并联模型只提供独立风险复核信号。
2. 默认 `shadow` 不改变最终输出，只记录风险和分歧。
3. `fusion_config.json` 的核心内容已经写入 `method.json` 的 `parallel` 字段。
4. 真正上线前应由端侧工程按 `method.json` 重写为目标语言实现，并用本目录文件做一致性核对。
"""


def build_method(artifact_dir, bundle, model_config, fusion_config):
    selected_features = [str(x) for x in bundle.get("selected_features", model_config.get("selected_features", []))]
    fill_values = {str(k): float(v) for k, v in bundle.get("fill_values", model_config.get("fill_values", {})).items()}
    threshold = float(bundle.get("threshold", model_config.get("threshold", fusion_config.get("new_model_threshold", 0.5))))
    constant_probability = bundle.get("constant_probability", model_config.get("constant_probability"))
    method = {
        "project_type": PROJECT_TYPE,
        "package_version": 1,
        "model": {
            "file": "model.json",
            "source_file": MODEL_SOURCE,
            "runtime": "constant_probability" if constant_probability is not None else "xgboost_json",
            "threshold": threshold,
            "constant_probability": None if constant_probability is None else float(constant_probability),
        },
        "selected_features": selected_features,
        "fill_values": fill_values,
        "feature_extractor": {
            "file": "feature_extractor.py",
            "compat_file": "s02_features.py",
            "source_file": "s02_features.py",
            "primary_window_function": "extract_feature_pool_from_window",
        },
        "commercial_model": {
            "file": "commercial_model.py",
            "source_file": "s01_model.py",
            "manifest_file": "commercial_model_manifest.json",
            "frozen": True,
        },
        "guard": {
            "default_mode": "shadow",
            "supported_modes": ["bypass", "shadow", "soft_guard", "hard_veto"],
            "recommended_first_release": "shadow",
        },
        "parallel": {
            "fusion_strategy": fusion_config.get("chosen_strategy", "veto"),
            "veto_params": fusion_config.get("veto_params", {"p_c_high": 0.8, "p_n_low": 0.2}),
            "new_model_threshold": float(fusion_config.get("new_model_threshold", threshold)),
            "final_decision_rule": "default shadow keeps commercial_pred; hard_veto can reject commercial-positive samples when new-model risk is persistent",
        },
        "training_config": sanitize_config(model_config),
        "fingerprint": sanitize_config(model_config.get("fingerprint", {})),
        "fusion_config": sanitize_config(fusion_config),
    }
    return method


def export_deploy_package(artifact_dir, output_dir=None):
    artifact_dir = os.path.abspath(artifact_dir)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = output_dir or os.path.join(artifact_dir, "deploy_export")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_path = require_file(os.path.join(artifact_dir, MODEL_SOURCE))
    bundle_path = require_file(os.path.join(artifact_dir, BUNDLE_SOURCE))
    selected_path = require_file(os.path.join(artifact_dir, "selected_features.json"))
    manifest_path = require_file(os.path.join(artifact_dir, "commercial_model_manifest.json"))
    fusion_path = require_file(os.path.join(artifact_dir, "fusion_config.json"))
    fingerprint_path = os.path.join(artifact_dir, "model_fingerprint.json")
    feature_script = require_file(os.path.join(project_dir, "s02_features.py"))
    commercial_script = require_file(os.path.join(project_dir, "s01_model.py"))

    bundle = joblib.load(bundle_path)
    model_config = bundle.get("config") or {}
    fusion_config = read_json(fusion_path)
    method = build_method(artifact_dir, bundle, model_config, fusion_config)

    write_json(os.path.join(output_dir, "model.json"), sanitize_config(read_json(model_path)))
    shutil.copyfile(selected_path, os.path.join(output_dir, "selected_features.json"))
    shutil.copyfile(manifest_path, os.path.join(output_dir, "commercial_model_manifest.json"))
    if os.path.exists(fingerprint_path):
        shutil.copyfile(fingerprint_path, os.path.join(output_dir, "model_fingerprint.json"))
    shutil.copyfile(feature_script, os.path.join(output_dir, "feature_extractor.py"))
    shutil.copyfile(feature_script, os.path.join(output_dir, "s02_features.py"))
    shutil.copyfile(commercial_script, os.path.join(output_dir, "commercial_model.py"))
    write_json(os.path.join(output_dir, "method.json"), method)
    write_json(os.path.join(output_dir, "fill_values.json"), method["fill_values"])
    Path(output_dir, "deploy_inference.py").write_text(deploy_inference_source(), encoding="utf-8")
    Path(output_dir, "README_DEPLOY.md").write_text(readme_text(), encoding="utf-8")

    files = sorted(p.name for p in Path(output_dir).iterdir() if p.is_file())
    deploy_manifest = {
        "project_type": PROJECT_TYPE,
        "independent_package": True,
        "files": files,
        "sha256": {name: sha256_file(os.path.join(output_dir, name)) for name in files},
    }
    write_json(os.path.join(output_dir, "deploy_manifest.json"), deploy_manifest)
    return output_dir


def main():
    p = argparse.ArgumentParser(description="Export parallel deployment handoff package")
    p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()
    out = export_deploy_package(args.artifact_dir, args.output_dir)
    print(f"Deploy package exported: {out}")


if __name__ == "__main__":
    main()
