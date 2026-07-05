# -*- coding: utf-8 -*-
"""
S11: Explainability reports for the parallel guard.

Outputs comparison reports, tree exports, and error-path traces without
changing model behavior.
"""

import argparse
import json
import os
import shutil
import subprocess

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PRED_COL = "parallel_pred"
BUNDLE_NAME = "new_model_bundle.pkl"
FEATURE_FILE_PREFIX = "features"
PALETTE = {
    "commercial": "#484878",
    "guard": "#E4CCD8",
    "delta_up": "#2E9E44",
    "delta_down": "#B64342",
    "neutral": "#606060",
}


def apply_publication_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def save_publication_figure(fig, stem, dpi=600):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    path = f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _to_rows(rows):
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict("records")
    return list(rows)


def _metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def build_comparison_report(rows, guard_pred_col=PRED_COL):
    data = _to_rows(rows)
    y = [int(r["target"]) for r in data]
    commercial = [int(r["commercial_pred"]) for r in data]
    guard = [int(r[guard_pred_col]) for r in data]
    fixed = [
        str(r.get("sample_name", i)) for i, r in enumerate(data)
        if int(r["commercial_pred"]) != int(r["target"]) and int(r[guard_pred_col]) == int(r["target"])
    ]
    broken = [
        str(r.get("sample_name", i)) for i, r in enumerate(data)
        if int(r["commercial_pred"]) == int(r["target"]) and int(r[guard_pred_col]) != int(r["target"])
    ]
    return {
        "n": int(len(data)),
        "commercial": _metrics(y, commercial),
        "guard": _metrics(y, guard),
        "fixed_count": int(len(fixed)),
        "broken_count": int(len(broken)),
        "fixed_samples": fixed,
        "broken_samples": broken,
    }


def _child_by_id(node, node_id):
    for child in node.get("children", []):
        if int(child.get("nodeid")) == int(node_id):
            return child
    return None


def trace_tree_path(tree, feature_values):
    node = tree
    path = []
    while "leaf" not in node:
        feature = node["split"]
        threshold = float(node["split_condition"])
        raw_value = feature_values.get(feature)
        value = float(raw_value) if raw_value is not None and np.isfinite(float(raw_value)) else None
        if value is None:
            direction, child_id = "missing", node["missing"]
        elif value < threshold:
            direction, child_id = "yes", node["yes"]
        else:
            direction, child_id = "no", node["no"]
        path.append({
            "node_id": int(node["nodeid"]),
            "feature": feature,
            "threshold": threshold,
            "value": value,
            "direction": direction,
            "child_id": int(child_id),
        })
        node = _child_by_id(node, child_id)
        if node is None:
            break
    return {
        "path": path,
        "leaf_id": int(node.get("nodeid", -1)) if node else -1,
        "leaf_value": float(node.get("leaf", 0.0)) if node else 0.0,
    }


def _tree_to_dot(node, lines):
    node_id = int(node["nodeid"])
    if "leaf" in node:
        lines.append(f'  {node_id} [label="leaf={float(node["leaf"]):.6g}", shape=box];')
        return
    label = f'{node["split"]} < {float(node["split_condition"]):.6g}'
    lines.append(f'  {node_id} [label="{label}"];')
    for child in node.get("children", []):
        child_id = int(child["nodeid"])
        edge_label = []
        if child_id == int(node.get("yes", -999)):
            edge_label.append("yes")
        if child_id == int(node.get("no", -999)):
            edge_label.append("no")
        if child_id == int(node.get("missing", -999)):
            edge_label.append("missing")
        lines.append(f'  {node_id} -> {child_id} [label="{"/".join(edge_label)}"];')
        _tree_to_dot(child, lines)


def export_trees(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    booster = model.get_booster()
    dumps = booster.get_dump(dump_format="json", with_stats=True)
    text_dump = booster.get_dump(dump_format="text", with_stats=True)
    with open(os.path.join(out_dir, "all_trees.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_dump))
    rows = []
    for i, payload in enumerate(dumps):
        tree = json.loads(payload)
        json_path = os.path.join(out_dir, f"tree_{i:03d}.json")
        dot_path = os.path.join(out_dir, f"tree_{i:03d}.dot")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2)
        lines = [
            "digraph tree {",
            '  graph [rankdir=TB, bgcolor="white", margin=0.04];',
            '  node [shape=ellipse, style="rounded,filled", fillcolor="#F7F7FA", color="#606060", fontname="Arial", fontsize=10];',
            '  edge [color="#767676", fontname="Arial", fontsize=9];',
        ]
        _tree_to_dot(tree, lines)
        lines.append("}")
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        png_path = os.path.join(out_dir, f"tree_{i:03d}.png")
        if shutil.which("dot"):
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=False)
        rows.append({"tree_id": i, "json_path": json_path, "dot_path": dot_path, "png_path": png_path if os.path.exists(png_path) else ""})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "model_structure_summary.csv"), index=False)
    return rows


def export_error_paths(model, features, eval_df, feature_df, pred_col, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    errors = eval_df[eval_df[pred_col].astype(int) != eval_df["target"].astype(int)].copy()
    errors.to_csv(os.path.join(out_dir, "error_samples.csv"), index=False)
    if errors.empty or feature_df.empty:
        pd.DataFrame().to_csv(os.path.join(out_dir, "error_tree_paths.csv"), index=False)
        return []
    trees = [json.loads(x) for x in model.get_booster().get_dump(dump_format="json", with_stats=True)]
    rows = []
    error_names = set(errors["sample_name"].astype(str))
    for _, row in feature_df[feature_df["sample_name"].astype(str).isin(error_names)].iterrows():
        values = {f: row.get(f) for f in features}
        for tree_id, tree in enumerate(trees):
            trace = trace_tree_path(tree, values)
            for step_idx, step in enumerate(trace["path"]):
                rows.append({
                    "sample_name": row["sample_name"],
                    "window_idx": row.get("window_idx", -1),
                    "tree_id": tree_id,
                    "step_idx": step_idx,
                    "node_id": step["node_id"],
                    "feature": step["feature"],
                    "threshold": step["threshold"],
                    "value": step["value"],
                    "direction": step["direction"],
                    "leaf_id": trace["leaf_id"],
                    "leaf_value": trace["leaf_value"],
                })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "error_tree_paths.csv"), index=False)
    write_error_path_summary_figure(rows, out_dir)
    return rows


def write_comparison_outputs(report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    rows = []
    for metric in ["accuracy", "precision", "recall", "f1"]:
        rows.append({
            "metric": metric,
            "commercial": report["commercial"][metric],
            "guard": report["guard"][metric],
            "delta": report["guard"][metric] - report["commercial"][metric],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparison_report.csv"), index=False)
    with open(os.path.join(out_dir, "comparison_report.md"), "w", encoding="utf-8") as f:
        f.write("# Commercial vs Guard Comparison\n\n")
        f.write(f"- Samples: {report['n']}\n")
        f.write(f"- Fixed samples: {report['fixed_count']}\n")
        f.write(f"- Broken samples: {report['broken_count']}\n")
        for row in rows:
            f.write(f"- {row['metric']}: commercial={row['commercial']:.6f}, guard={row['guard']:.6f}, delta={row['delta']:.6f}\n")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    source_path = os.path.join(fig_dir, "comparison_metrics_source.csv")
    df.to_csv(source_path, index=False)
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df["commercial"], width, label="Commercial", color=PALETTE["commercial"], edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, df["guard"], width, label="Full guard", color=PALETTE["guard"], edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="#A8A8A8", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df["metric"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Commercial baseline vs guard")
    ax.legend(loc="lower right")
    for xi, delta in zip(x, df["delta"]):
        color = PALETTE["delta_up"] if delta >= 0 else PALETTE["delta_down"]
        ax.text(xi, 1.01, f"{delta:+.2f}", ha="center", va="bottom", color=color, fontsize=6)
    fig_paths = save_publication_figure(fig, os.path.join(fig_dir, "comparison_metrics"))
    return {"comparison_figure": fig_paths, "source_data": source_path}


def write_error_path_summary_figure(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(os.path.join(out_dir, "error_path_node_frequency_source.csv"), index=False)
        return []
    freq = (
        df.groupby(["node_id", "feature"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    freq["label"] = freq.apply(lambda r: f"node {int(r['node_id'])}: {r['feature']}", axis=1)
    source_path = os.path.join(out_dir, "error_path_node_frequency_source.csv")
    freq.to_csv(source_path, index=False)
    apply_publication_style()
    fig_h = max(2.2, 0.22 * len(freq) + 0.8)
    fig, ax = plt.subplots(figsize=(3.55, fig_h))
    plot_df = freq.iloc[::-1]
    ax.barh(plot_df["label"], plot_df["count"], color=PALETTE["commercial"], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Occurrences in error paths")
    ax.set_title("Frequent error-path split nodes")
    ax.grid(axis="x", color="#D8D8D8", lw=0.5)
    return save_publication_figure(fig, os.path.join(out_dir, "error_path_node_frequency"))


def _as_dataframe(rows):
    return rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))


def _sample_count(df):
    return int(df["sample_name"].nunique()) if "sample_name" in df.columns else int(len(df))


def write_sample_flow_funnel(rows, out_dir, pred_col=PRED_COL):
    df = _as_dataframe(rows)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    if df.empty:
        source = pd.DataFrame(columns=["stage", "count"])
    else:
        action = df.get("guard_action", pd.Series(["pass"] * len(df))).fillna("pass").astype(str)
        stages = [
            ("all evaluated samples", _sample_count(df)),
            ("commercial positives", int((df["commercial_pred"].astype(int) == 1).sum())),
            ("guard review suggested", int(action.isin(["extend_detection", "hard_veto"]).sum())),
            ("hard veto candidates", int(action.eq("hard_veto").sum())),
            ("final errors", int((df[pred_col].astype(int) != df["target"].astype(int)).sum())),
        ]
        source = pd.DataFrame([{"stage": s, "count": c} for s, c in stages])
    source_path = os.path.join(fig_dir, "sample_flow_funnel_source.csv")
    source.to_csv(source_path, index=False)
    apply_publication_style()
    fig_h = max(2.1, 0.32 * max(1, len(source)) + 0.7)
    fig, ax = plt.subplots(figsize=(3.55, fig_h))
    plot_df = source.iloc[::-1]
    ax.barh(plot_df["stage"], plot_df["count"], color=PALETTE["commercial"], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Samples")
    ax.set_title("Sample flow through commercial guard")
    ax.grid(axis="x", color="#D8D8D8", lw=0.5)
    for y, value in enumerate(plot_df["count"]):
        ax.text(value, y, f" {int(value)}", va="center", fontsize=6, color=PALETTE["neutral"])
    fig_paths = save_publication_figure(fig, os.path.join(fig_dir, "sample_flow_funnel"))
    return {"figure": fig_paths, "source_data": source_path}


def _with_error_type(df, pred_col):
    out = df.copy()
    target = out["target"].astype(int)
    pred = out[pred_col].astype(int)
    out["error_type"] = "correct"
    out.loc[(target == 0) & (pred == 1), "error_type"] = "false_wearing"
    out.loc[(target == 1) & (pred == 0), "error_type"] = "false_reject"
    if "veto_risk" not in out.columns:
        out["veto_risk"] = 0.0
    if "risk_ratio" not in out.columns:
        out["risk_ratio"] = 0.0
    out["veto_risk"] = pd.to_numeric(out["veto_risk"], errors="coerce").fillna(0.0).clip(0, 1)
    out["risk_ratio"] = pd.to_numeric(out["risk_ratio"], errors="coerce").fillna(0.0).clip(0, 1)
    return out


def write_error_distribution_figures(rows, out_dir, pred_col=PRED_COL):
    df = _with_error_type(_as_dataframe(rows), pred_col)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    source_path = os.path.join(fig_dir, "error_distribution_source.csv")
    df.to_csv(source_path, index=False)
    apply_publication_style()
    paths = []

    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    bins = np.linspace(0, 1, 11)
    colors = {"correct": "#D8D8D8", "false_wearing": PALETTE["delta_down"], "false_reject": PALETTE["commercial"]}
    for label in ["correct", "false_wearing", "false_reject"]:
        vals = df.loc[df["error_type"] == label, "veto_risk"].to_numpy(dtype=float)
        if len(vals):
            ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.55, label=label, color=colors[label], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Guard risk")
    ax.set_ylabel("Samples")
    ax.set_title("Error distribution by guard risk")
    ax.legend(loc="upper left")
    paths.extend(save_publication_figure(fig, os.path.join(fig_dir, "error_distribution_by_guard_risk")))

    counts = df["error_type"].value_counts().reindex(["false_wearing", "false_reject", "correct"], fill_value=0).reset_index()
    counts.columns = ["error_type", "count"]
    counts.to_csv(os.path.join(fig_dir, "error_distribution_by_error_type_source.csv"), index=False)
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    bar_colors = [colors[x] for x in counts["error_type"]]
    ax.bar(counts["error_type"], counts["count"], color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Samples")
    ax.set_title("Final prediction outcomes")
    ax.tick_params(axis="x", rotation=20)
    paths.extend(save_publication_figure(fig, os.path.join(fig_dir, "error_distribution_by_error_type")))
    return {"figures": paths, "source_data": source_path}


def write_error_escape_rules(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = _as_dataframe(rows)
    csv_path = os.path.join(out_dir, "error_escape_rules.csv")
    md_path = os.path.join(out_dir, "error_escape_rules.md")
    if df.empty:
        pd.DataFrame(columns=["node_id", "feature", "threshold", "direction", "count", "n_samples"]).to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Error Escape Rules\n\nNo error tree paths were available.\n")
        return md_path
    grouped = (
        df.groupby(["node_id", "feature", "threshold", "direction"], dropna=False)
        .agg(count=("node_id", "size"), n_samples=("sample_name", "nunique"))
        .reset_index()
        .sort_values(["count", "n_samples"], ascending=False)
    )
    grouped.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Error Escape Rules\n\n")
        f.write("High-frequency split nodes traversed by final-model error samples.\n\n")
        for _, row in grouped.head(20).iterrows():
            f.write(
                f"- node {int(row['node_id'])}: {row['feature']} {row['direction']} "
                f"threshold={float(row['threshold']):.6g}; occurrences={int(row['count'])}; "
                f"samples={int(row['n_samples'])}\n"
            )
    return md_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--split", default="test")
    args = p.parse_args()
    eval_path = os.path.join(args.artifact_dir, "evaluation_samples.csv")
    bundle_path = os.path.join(args.artifact_dir, BUNDLE_NAME)
    feature_path = os.path.join(args.artifact_dir, f"{FEATURE_FILE_PREFIX}_{args.split}.csv")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"{eval_path} not found; run s09_evaluate.py first")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"{bundle_path} not found; train the guard model first")
    eval_df = pd.read_csv(eval_path)
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    features = bundle["selected_features"]
    feature_df = pd.read_csv(feature_path) if os.path.exists(feature_path) else pd.DataFrame()
    report = build_comparison_report(eval_df, guard_pred_col=PRED_COL)
    write_comparison_outputs(report, args.artifact_dir)
    write_sample_flow_funnel(eval_df, args.artifact_dir, pred_col=PRED_COL)
    write_error_distribution_figures(eval_df, args.artifact_dir, pred_col=PRED_COL)
    export_trees(model, os.path.join(args.artifact_dir, "tree_export"))
    error_rows = export_error_paths(model, features, eval_df, feature_df, PRED_COL, os.path.join(args.artifact_dir, "error_trace"))
    write_error_escape_rules(error_rows, os.path.join(args.artifact_dir, "error_trace"))
    print("Explainability reports written")


if __name__ == "__main__":
    main()
