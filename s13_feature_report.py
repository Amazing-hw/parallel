# -*- coding: utf-8 -*-
"""
S13: Lightweight feature-pool report for explainability review.

Input:  {artifact_dir}/feature_pool_test.csv
Output: {artifact_dir}/feature_report/*
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {
    "sample_name", "h5_file", "target", "should_veto", "commercial_pred",
    "window_idx", "window_index", "start_sec", "start_100hz", "commercial_score",
    "is_error", "fallback", "fallback_reason", "mode", "split", "Unnamed: 0",
}


def _safe_auc(y, x):
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        auc = float(roc_auc_score(y, x))
    except Exception:
        return 0.5
    return max(auc, 1.0 - auc)


def _load_selected_features(artifact_dir, candidates):
    path = Path(artifact_dir) / "selected_features.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = [str(x) for x in payload.get("selected_features", [])]
        return [x for x in selected if x in candidates]
    except Exception:
        return []


def _numeric_feature_cols(df):
    return [
        c for c in df.columns
        if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _prepare_matrix(df, features, max_points, random_state):
    data = df.copy()
    if max_points and len(data) > max_points:
        data = data.sample(n=int(max_points), random_state=int(random_state))
    x = data[features].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    return data.reset_index(drop=True), x.to_numpy(dtype=float)


def _plot_auc(out_dir, ranking, dpi):
    top = ranking.head(20).iloc[::-1]
    plt.figure(figsize=(8, max(3, 0.28 * len(top))))
    plt.barh(top["feature"], top["auc"])
    plt.axvline(0.5, color="#666666", linewidth=1)
    plt.xlabel("AUC or 1-AUC")
    plt.title("Top Feature Separability on Test")
    plt.tight_layout()
    path = Path(out_dir) / "feature_auc_ranking.png"
    plt.savefig(path, dpi=dpi)
    plt.close()
    return str(path)


def _plot_pca(out_dir, x, y, dpi):
    if len(x) < 2 or x.shape[1] < 2:
        return None
    z = StandardScaler().fit_transform(x)
    emb = PCA(n_components=2, random_state=0).fit_transform(z)
    plt.figure(figsize=(6.2, 5.2))
    for label, color, name in [(0, "#D55E00", "target=0"), (1, "#0072B2", "target=1")]:
        mask = y == label
        if np.any(mask):
            plt.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.75, c=color, label=name)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Feature Pool")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = Path(out_dir) / "pca_2d.png"
    plt.savefig(path, dpi=dpi)
    plt.close()
    return str(path)


def _plot_corr(out_dir, df, features, dpi):
    selected = list(features[: min(15, len(features))])
    if len(selected) < 2:
        return None
    corr = df[selected].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)
    plt.figure(figsize=(8, 7))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(selected)), selected, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(selected)), selected, fontsize=8)
    plt.title("Selected Feature Correlation")
    plt.tight_layout()
    path = Path(out_dir) / "selected_feature_correlation.png"
    plt.savefig(path, dpi=dpi)
    plt.close()
    return str(path)


def generate_feature_report(artifact_dir, output_dir=None, max_points=1000, dpi=200, random_state=42):
    artifact_dir = Path(artifact_dir)
    out_dir = Path(output_dir) if output_dir else artifact_dir / "feature_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = artifact_dir / "feature_pool_test.csv"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    df = pd.read_csv(feature_path)
    if "target" not in df.columns:
        raise ValueError("feature_pool_test.csv must contain target")
    candidates = _numeric_feature_cols(df)
    selected = _load_selected_features(artifact_dir, candidates)
    report_features = selected or candidates
    sampled, x = _prepare_matrix(df, report_features, max_points, random_state)
    y = sampled["target"].astype(int).to_numpy()

    rows = []
    for feature in candidates:
        values = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rows.append({"feature": feature, "auc": _safe_auc(df["target"].astype(int).to_numpy(), values.to_numpy())})
    ranking = pd.DataFrame(rows).sort_values("auc", ascending=False)
    ranking.to_csv(out_dir / "feature_auc_ranking.csv", index=False)
    sampled[["sample_name", "target"] + [c for c in report_features if c in sampled.columns]].to_csv(
        out_dir / "feature_report_source.csv", index=False
    )

    figures = {
        "feature_auc_ranking": _plot_auc(out_dir, ranking, dpi),
        "pca_2d": _plot_pca(out_dir, x, y, dpi),
        "selected_feature_correlation": _plot_corr(out_dir, sampled, report_features, dpi),
    }
    summary = {
        "n_rows": int(len(df)),
        "n_rows_sampled": int(len(sampled)),
        "n_candidate_features": int(len(candidates)),
        "n_report_features": int(len(report_features)),
        "feature_source": "selected_features" if selected else "all_numeric_features",
        "label_counts": {str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()},
        "figures": {k: v for k, v in figures.items() if v},
    }
    (out_dir / "feature_report_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Feature report exported: {out_dir}")
    return {"output_dir": str(out_dir), "summary": summary}


def main():
    p = argparse.ArgumentParser(description="Generate feature-pool explainability report")
    p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_points", type=int, default=1000)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    generate_feature_report(args.artifact_dir, args.output_dir, args.max_points, args.dpi)


if __name__ == "__main__":
    main()
