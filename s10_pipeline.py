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


def run(script, args_list, desc, cwd):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    cmd = [sys.executable, script] + args_list
    print(f"  {subprocess.list2cmdline(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=cwd).returncode
    if rc != 0:
        print(f"\n[FAILED] {desc}")
        sys.exit(rc)
    print(f"  Done ({time.time()-t0:.1f}s)")


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
    p.add_argument("--n_estimators", type=int, default=10)
    p.add_argument("--max_depth", type=int, default=2)
    p.add_argument("--strategy", default="veto", choices=["veto"])
    p.add_argument("--guard_mode", default="shadow", choices=["bypass", "shadow", "soft_guard", "hard_veto"])
    p.add_argument("--min_veto_windows", type=int, default=2)
    p.add_argument("--min_veto_ratio", type=float, default=0.4)
    p.add_argument("--manual_features", default=None)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--eval_split", default="test")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    paths = resolve_pipeline_paths(args.splits_dir, args.artifact_dir, args.dataset_dir, d)
    if args.dry_run:
        print(f"[DRY RUN] base_dir={paths['base_dir']}")
        print(f"[DRY RUN] dataset_dir={paths['dataset_dir']}")
        print(f"[DRY RUN] splits_dir={paths['splits_dir']}")
        print(f"[DRY RUN] artifact_dir={paths['artifact_dir']}")
    else:
        ensure_splits(
            paths["splits_dir"], paths["dataset_dir"],
            valid_size=args.valid_size, test_size=args.test_size,
            random_state=args.random_state, n_workers=args.n_workers,
            force_split=args.force_split,
        )

    extract_args = ["--splits_dir", paths["splits_dir"], "--artifact_dir", paths["artifact_dir"]]
    if args.max_samples:
        extract_args += ["--max_samples", str(args.max_samples)]
    train_args = ["--artifact_dir", paths["artifact_dir"], "--n_estimators", str(args.n_estimators), "--max_depth", str(args.max_depth)]
    if args.manual_features:
        train_args += ["--manual_features", _abs_path(args.manual_features, d)]
    steps = [
        ("S05-Extract", os.path.join(d, "s05_extract_features.py"), extract_args),
        ("S06-Select", os.path.join(d, "s06_select_features.py"), ["--artifact_dir", paths["artifact_dir"], "--max_features", str(args.max_features)]),
        ("S07-Train", os.path.join(d, "s07_train_model.py"), train_args),
        ("S08-Fusion", os.path.join(d, "s08_fusion.py"), ["--artifact_dir", paths["artifact_dir"], "--strategy", args.strategy]),
        ("S09-Evaluate", os.path.join(d, "s09_evaluate.py"),
         ["--artifact_dir", paths["artifact_dir"], "--splits_dir", paths["splits_dir"], "--split", args.eval_split,
          "--guard_mode", args.guard_mode, "--min_veto_windows", str(args.min_veto_windows),
          "--min_veto_ratio", str(args.min_veto_ratio)]),
    ]
    if args.explain:
        steps.append(("S11-Explain", os.path.join(d, "s11_explain.py"), ["--artifact_dir", paths["artifact_dir"], "--split", args.eval_split]))

    t0 = time.time()
    for desc, script, step_args in steps:
        if args.dry_run:
            print(f"[DRY RUN] {script} {' '.join(step_args)}")
        else:
            run(script, step_args, desc, d)
    print(f"\n{'='*60}\n  Parallel done ({time.time()-t0:.1f}s)\n{'='*60}")


if __name__ == "__main__":
    main()
