# -*- coding: utf-8 -*-
"""Standalone data loading and splitting utilities for the parallel project."""

import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


WINDOW_NAME_RE = re.compile(r"(?:^|_)w(?P<index>\d+)_(?P<label>[01])$")


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_n_workers(n_workers=None, n_items=None, cap=4):
    if _env_flag("WL_FORCE_SERIAL"):
        return 1
    if n_workers is None:
        n_workers = max(1, min(cap, (os.cpu_count() or cap) // 2))
    try:
        resolved = max(1, int(n_workers))
    except (TypeError, ValueError):
        resolved = 1
    if n_items is not None and int(n_items) <= 1:
        return 1
    return resolved


def multiprocessing_context_from_env():
    method = os.environ.get("WL_MP_START_METHOD", "").strip()
    if not method:
        return None
    import multiprocessing as mp
    return mp.get_context(method)


def is_supported_ppg_shape(shape):
    """Accept PPG arrays stored as (40, T) or pre-windowed (N, 40, T)."""
    if len(shape) == 2:
        return int(shape[0]) == 40
    if len(shape) == 3:
        return int(shape[1]) == 40
    return False


def find_h5_files(dataset_dir):
    files = glob.glob(os.path.join(dataset_dir, "*.h5"))
    if not files:
        files = glob.glob(os.path.join("..", dataset_dir, "*.h5"))
    return sorted(files)


def parse_window_name(name):
    match = WINDOW_NAME_RE.search(str(name))
    if not match:
        return None
    return int(match.group("index")), int(match.group("label"))


def _read_ppg_config(group, fallback=None):
    if "ppg_config" not in group:
        return fallback
    try:
        return int(group["ppg_config"][()])
    except (TypeError, ValueError):
        return fallback


def _target_from_window_labels(labels):
    if not labels:
        return 0
    if len(set(labels)) == 1:
        return int(labels[0])
    return int(np.mean(labels) >= 0.5)


def _scan_grouped_window_sample(h5_file, sample_name, group, filtered):
    parent_cfg = _read_ppg_config(group)
    windows = []
    for child_name in group.keys():
        parsed = parse_window_name(child_name)
        if parsed is None:
            continue
        child = group[child_name]
        if not isinstance(child, h5py.Group) or "ppg" not in child:
            continue
        shape = child["ppg"].shape
        if not is_supported_ppg_shape(shape):
            filtered["channel_count"] += 1
            continue
        window_index, label = parsed
        windows.append((window_index, label, child_name, shape, _read_ppg_config(child, parent_cfg)))
    if not windows:
        return None
    windows.sort(key=lambda item: item[0])
    labels = [int(item[1]) for item in windows]
    return {
        "sample_name": sample_name,
        "h5_file": h5_file,
        "target": _target_from_window_labels(labels),
        "ppg_shape": [len(windows)] + list(windows[0][3]),
        "ppg_cfg": None if windows[0][4] is None else int(windows[0][4]),
        "window_layout": "grouped_windows",
        "window_names": [str(item[2]) for item in windows],
        "window_indices": [int(item[0]) for item in windows],
        "window_labels": labels,
        "window_label_counts": {
            "target0": int(sum(1 for x in labels if x == 0)),
            "target1": int(sum(1 for x in labels if x == 1)),
        },
    }


def _scan_one_h5(h5_file):
    samples = []
    filtered = {"ppg_cfg": 0, "channel_count": 0}
    try:
        with h5py.File(h5_file, "r") as f:
            for sample_name in f.keys():
                group = f[sample_name]
                if "ppg" not in group:
                    grouped = _scan_grouped_window_sample(h5_file, sample_name, group, filtered)
                    if grouped is not None:
                        samples.append(grouped)
                    continue
                if "target" not in group:
                    continue
                shape = group["ppg"].shape
                if not is_supported_ppg_shape(shape):
                    filtered["channel_count"] += 1
                    continue
                try:
                    target = int(group["target"][()])
                except (TypeError, ValueError, KeyError):
                    continue
                samples.append({
                    "sample_name": sample_name,
                    "h5_file": h5_file,
                    "target": target,
                    "ppg_shape": list(shape),
                    "ppg_cfg": _read_ppg_config(group),
                })
    except Exception as exc:
        print(f"Scan error {h5_file}: {exc}")
    return samples, filtered


def scan_h5_samples(dataset_dir, n_workers=None):
    h5_files = find_h5_files(dataset_dir)
    print(f"Found {len(h5_files)} H5 files")
    print(f"Using H5 files: {h5_files}")
    n_workers = resolve_n_workers(n_workers, n_items=len(h5_files))
    samples = []
    filtered_total = {"ppg_cfg": 0, "channel_count": 0}
    if n_workers == 1 or len(h5_files) <= 1:
        for h5_file in h5_files:
            rows, filtered = _scan_one_h5(h5_file)
            samples.extend(rows)
            filtered_total["ppg_cfg"] += filtered["ppg_cfg"]
            filtered_total["channel_count"] += filtered["channel_count"]
    else:
        pool_kwargs = {"max_workers": n_workers}
        mp_ctx = multiprocessing_context_from_env()
        if mp_ctx is not None:
            pool_kwargs["mp_context"] = mp_ctx
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {ex.submit(_scan_one_h5, h5_file): h5_file for h5_file in h5_files}
            for done_count, fut in enumerate(as_completed(futures), 1):
                try:
                    rows, filtered = fut.result()
                except Exception as exc:
                    print(f"Scan error {futures[fut]}: {exc}")
                    rows, filtered = [], {"ppg_cfg": 0, "channel_count": 0}
                samples.extend(rows)
                filtered_total["ppg_cfg"] += filtered["ppg_cfg"]
                filtered_total["channel_count"] += filtered["channel_count"]
                if len(futures) >= 10 and (done_count % max(1, len(futures) // 10) == 0 or done_count == len(futures)):
                    print(f"  split scan progress: {done_count}/{len(futures)} files", flush=True)
    if filtered_total["channel_count"] > 0:
        print(f"Filtered samples with unsupported PPG shape: {filtered_total['channel_count']}")
    samples.sort(key=lambda s: (s["h5_file"], s["sample_name"]))
    return samples


def split_samples(samples, valid_size=0.15, test_size=0.15, random_state=42):
    y = np.array([s["target"] for s in samples])
    idx = np.arange(len(samples))
    train_valid_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )
    valid_ratio = valid_size / (1.0 - test_size)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_ratio,
        random_state=random_state,
        stratify=y[train_valid_idx],
    )
    return {
        "train": [samples[i] for i in train_idx],
        "valid": [samples[i] for i in valid_idx],
        "test": [samples[i] for i in test_idx],
    }


def load_splits(artifact_dir):
    with open(os.path.join(artifact_dir, "splits.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def save_splits(splits, artifact_dir):
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)


def summarize_split(splits):
    for part in ["train", "valid", "test"]:
        rows = splits[part]
        n0 = sum(1 for s in rows if int(s["target"]) == 0)
        n1 = sum(1 for s in rows if int(s["target"]) == 1)
        print(f"  {part}: total={len(rows)}, target0={n0}, target1={n1}")
