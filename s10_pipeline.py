# -*- coding: utf-8 -*-
"""
Standalone parallel pipeline runner.

Usage:
    python s10_pipeline.py --dataset_dir path/to/h5_dataset --guard_mode shadow --explain
"""

import argparse
import os
import subprocess
import sys
import time

from s04_data import save_splits, scan_h5_samples, split_samples, summarize_split


PROJECT_NAME = "parallel"
DRY_RUN_PATH_KEYS = ("base_dir", "dataset_dir", "splits_dir", "artifact_dir")


def _abs_path(path, base_dir):
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def resolve_pipeline_paths(splits_dir=None, artifact_dir=None, dataset_dir="dataset", base_dir=None):
    base_dir = os.path.abspath(base_dir or os.path.dirname(os.path.abspath(__file__)))
    splits_dir = _abs_path(splits_dir, base_dir) if splits_dir else os.path.join(base_dir, "artifacts")
    artifact_dir = _abs_path(artifact_dir, base_dir) if artifact_dir else os.path.join(base_dir, "artifacts", PROJECT_NAME)
    dataset_dir = _abs_path(dataset_dir, base_dir)
    return {"base_dir": base_dir, "splits_dir": splits_dir, "artifact_dir": artifact_dir, "dataset_dir": dataset_dir}


def print_dry_run_paths(paths):
    for key in DRY_RUN_PATH_KEYS:
        print(f"[DRY RUN] {key}={paths[key]}")


def ensure_splits(splits_dir, dataset_dir, valid_size=0.15, test_size=0.15, random_state=42, n_workers=None, force_split=False):
    splits_path = os.path.join(splits_dir, "splits.json")
    if os.path.exists(splits_path) and not force_split:
        print(f"Using existing splits: {splits_path}")
        return {"created": False, "path": splits_path}
    samples = scan_h5_samples(dataset_dir, n_workers=n_workers)
    if not samples:
        raise FileNotFoundError(f"no supported H5 samples found in dataset_dir={dataset_dir}")
    print(f"Total samples: {len(samples)}")
    print(f"target=0: {sum(1 for s in samples if int(s['target']) == 0)}")
    print(f"target=1: {sum(1 for s in samples if int(s['target']) == 1)}")
    splits = split_samples(samples, valid_size=valid_size, test_size=test_size, random_state=random_state)
    summarize_split(splits)
    save_splits(splits, splits_dir)
    print(f"Saved splits: {splits_path}")
    return {"created": True, "path": splits_path}


def format_duration(seconds):
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def print_timing_summary(records, total_elapsed, pipeline_status):
    print("\n" + "=" * 60)
    print(f"  Pipeline timing summary (status={pipeline_status}, total={total_elapsed:.1f}s / {format_duration(total_elapsed)})")
    print("=" * 60)
    for i, record in enumerate(records, start=1):
        elapsed = float(record.get("elapsed_sec", 0.0))
        status = record.get("status", "unknown")
        detail = record.get("detail")
        suffix = f" [{detail}]" if detail else ""
        print(f"  {i:02d}. {record.get('step', 'unknown'):<24} {elapsed:8.1f}s  {format_duration(elapsed):>9}  {status}{suffix}")


def run(script, args_list, desc, cwd):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    cmd = [sys.executable, script] + args_list
    print(f"  {subprocess.list2cmdline(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=cwd).returncode
    elapsed = time.time() - t0
    if rc == 0:
        print(f"  Done ({elapsed:.1f}s / {format_duration(elapsed)})")
    else:
        print(f"\n[FAILED] {desc} ({elapsed:.1f}s / {format_duration(elapsed)})")
    return {
        "step": desc,
        "status": "success" if rc == 0 else "failed",
        "elapsed_sec": round(elapsed, 3),
        "return_code": int(rc),
        "command": subprocess.list2cmdline(cmd),
    }


def split_timing_record(elapsed, split_result):
    return {
        "step": "S04-Split data",
        "status": "success",
        "elapsed_sec": round(elapsed, 3),
        "return_code": 0,
        "command": "internal ensure_splits",
        "detail": "created" if split_result.get("created") else "reused",
        "output": split_result.get("path"),
    }


def parse_skip_tokens(values):
    tokens = set()
    for value in values or []:
        for part in str(value).replace(";", ",").split(","):
            token = part.strip().lower()
            if token:
                tokens.add(token)
    return tokens


def should_skip_step(desc, script, skip_tokens):
    if not skip_tokens:
        return False
    script_name = os.path.basename(script).lower()
    script_stem = os.path.splitext(script_name)[0]
    desc_norm = str(desc).lower().replace("-", " ")
    aliases = {script_name, script_stem}
    for part in desc_norm.split():
        aliases.add(part)
    if desc_norm.startswith("s"):
        aliases.add(desc_norm.split()[0])
    return any(token in aliases for token in skip_tokens)


def main():
    d = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Standalone parallel pipeline")
    p.add_argument("--dataset_dir", default="dataset")
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--artifact_dir", default=None)
    p.add_argument("--valid_size", type=float, default=0.15)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=None)
    p.add_argument("--force_split", action="store_true")
    p.add_argument("--max_features", type=int, default=12)
    p.add_argument("--preselect_top", type=int, default=4)
    p.add_argument("--stability_splits", type=int, default=4)
    p.add_argument("--stability_seeds", default="1,7")
    p.add_argument("--stability_max_rows", type=int, default=5000)
    p.add_argument("--permutation_repeats", type=int, default=3)
    p.add_argument("--rank_only", action="store_true")
    p.add_argument("--n_estimators", type=int, default=10)
    p.add_argument("--max_depth", type=int, default=2)
    p.add_argument("--strategy", default="veto", choices=["veto"])
    p.add_argument("--guard_mode", default="shadow", choices=["bypass", "shadow", "soft_guard", "hard_veto"])
    p.add_argument("--min_veto_windows", type=int, default=2)
    p.add_argument("--min_veto_ratio", type=float, default=0.4)
    p.add_argument("--manual_features", default=None)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--feature_report", action="store_true",
                   help="Generate S13 feature-pool explainability report from feature_pool_test.csv")
    p.add_argument("--plot_mode", default="full", choices=["basic", "full"])
    p.add_argument("-skip", "--skip", action="append", default=[],
                   help="Skip pipeline stages by stage id or script name, e.g. -skip s06 or --skip s06,s11_explain.py")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--eval_split", default="test")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    paths = resolve_pipeline_paths(args.splits_dir, args.artifact_dir, args.dataset_dir, d)
    if args.dry_run:
        print_dry_run_paths(paths)
    timing_records = []
    pipeline_t0 = time.time()
    if not args.dry_run:
        split_t0 = time.time()
        split_result = ensure_splits(
            paths["splits_dir"], paths["dataset_dir"],
            valid_size=args.valid_size, test_size=args.test_size,
            random_state=args.random_state, n_workers=args.n_workers,
            force_split=args.force_split,
        )
        split_elapsed = time.time() - split_t0
        timing_records.append(split_timing_record(split_elapsed, split_result))
        print(f"[TIMING] S04-Split data: {split_elapsed:.1f}s / {format_duration(split_elapsed)}")

    extract_args = ["--splits_dir", paths["splits_dir"], "--artifact_dir", paths["artifact_dir"]]
    if args.max_samples:
        extract_args += ["--max_samples", str(args.max_samples)]
    if args.n_workers is not None:
        extract_args += ["--n_workers", str(args.n_workers)]
    select_args = [
        "--artifact_dir", paths["artifact_dir"],
        "--max_features", str(args.max_features),
        "--preselect_top", str(args.preselect_top),
        "--stability_splits", str(args.stability_splits),
        "--stability_seeds", args.stability_seeds,
        "--stability_max_rows", str(args.stability_max_rows),
        "--permutation_repeats", str(args.permutation_repeats),
    ]
    if args.rank_only:
        select_args.append("--rank_only")
    if args.n_workers is not None:
        select_args += ["--n_workers", str(args.n_workers)]
    train_args = ["--artifact_dir", paths["artifact_dir"], "--n_estimators", str(args.n_estimators), "--max_depth", str(args.max_depth)]
    if args.manual_features:
        train_args += ["--manual_features", _abs_path(args.manual_features, d)]
    steps = [
        ("S05-Extract", os.path.join(d, "s05_extract_features.py"), extract_args),
        ("S06-Select", os.path.join(d, "s06_select_features.py"), select_args),
        ("S07-Train", os.path.join(d, "s07_train_model.py"), train_args),
        ("S08-Fusion", os.path.join(d, "s08_fusion.py"), ["--artifact_dir", paths["artifact_dir"], "--strategy", args.strategy]),
        ("S09-Evaluate", os.path.join(d, "s09_evaluate.py"),
         ["--artifact_dir", paths["artifact_dir"], "--splits_dir", paths["splits_dir"], "--split", args.eval_split,
          "--guard_mode", args.guard_mode, "--min_veto_windows", str(args.min_veto_windows),
          "--min_veto_ratio", str(args.min_veto_ratio)]),
    ]
    if args.explain:
        steps.append(("S11-Explain", os.path.join(d, "s11_explain.py"), [
            "--artifact_dir", paths["artifact_dir"], "--split", args.eval_split,
            "--plot_mode", args.plot_mode,
        ]))
    if args.feature_report:
        steps.append(("S13-Feature report", os.path.join(d, "s13_feature_report.py"), [
            "--artifact_dir", paths["artifact_dir"],
        ]))

    skip_tokens = parse_skip_tokens(args.skip)
    for desc, script, step_args in steps:
        if should_skip_step(desc, script, skip_tokens):
            print(f"[DRY RUN] SKIP {desc}" if args.dry_run else f"[SKIP] {desc}")
            continue
        if args.dry_run:
            print(f"[DRY RUN] {script} {' '.join(step_args)}")
        else:
            record = run(script, step_args, desc, d)
            timing_records.append(record)
            if record["return_code"] != 0:
                total_elapsed = time.time() - pipeline_t0
                print_timing_summary(timing_records, total_elapsed, "failed")
                sys.exit(record["return_code"])
    total_elapsed = time.time() - pipeline_t0
    if not args.dry_run:
        print_timing_summary(timing_records, total_elapsed, "success")
    print(f"\n{'='*60}\n  Parallel done ({total_elapsed:.1f}s / {format_duration(total_elapsed)})\n{'='*60}")


if __name__ == "__main__":
    main()
