# -*- coding: utf-8 -*-
"""
S05: Extract BOTH commercial scores and new features for ALL windows.

Output: {artifact_dir}/features_{train,valid,test}.csv
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import time

import numpy as np
import pandas as pd

from s01_model import OldLivenessModel, extract_8_commercial_features, FEATURE_FS, COMMERCIAL_WIN_SEC, COMMERCIAL_STRIDE_SEC
from s01_model import commercial_model_manifest
from s02_features import load_ppg, load_acc, get_channels_from_window, detect_green_mode
from s02_features import is_prewindowed_signal, _downsample_ppg, _is_25hz_sample, extract_feature_pool_from_window, validate_h5_file
from s04_data import load_splits, multiprocessing_context_from_env, resolve_n_workers

MIN_AUTO_PARALLEL_SAMPLES = 32
MIN_CHUNKED_MAP_SAMPLES = 200
_THREADPOOL_LIMITER = None
FEATURE_RESULT_COLUMNS = [
    "sample_name",
    "target",
    "window_idx",
    "commercial_score",
    "commercial_pred",
    "fallback",
    "fallback_reason",
]


def _format_duration(seconds):
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _print_progress(split_name, done, total, start_time, rows_count):
    if total <= 0:
        return
    elapsed = max(1e-9, time.time() - start_time)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    pct = 100.0 * done / total
    print(
        f"[{split_name}] {done}/{total} ({pct:5.1f}%) "
        f"speed={rate:.2f} samples/s eta={_format_duration(eta)} rows={rows_count}",
        flush=True,
    )


def _progress_interval(total):
    if total <= 20:
        return 1
    return max(1, total // 20)


def _safe_rate(count, elapsed):
    return round(float(count) / float(elapsed), 6) if elapsed > 0 else None


def _write_commercial_manifest(artifact_dir):
    with open(os.path.join(artifact_dir, "commercial_model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(commercial_model_manifest(), f, indent=2, ensure_ascii=False)


def _print_timing_summary(rows):
    print("\n[S05 TIMING] split summary")
    for row in rows:
        print(
            f"  {row['split']:<5} samples={row['samples']:>5} rows={row['rows']:>6} "
            f"elapsed={row['elapsed_sec']:>7.1f}s "
            f"speed={row['samples_per_sec'] or 0:.2f} samples/s"
        )


def _parallel_sample_worker(args):
    idx, sample = args
    return idx, extract_sample(sample, OldLivenessModel())


def _init_feature_worker():
    """Limit nested BLAS/NumExpr threads inside each process-pool worker."""
    global _THREADPOOL_LIMITER
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        _THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except Exception:
        _THREADPOOL_LIMITER = None


def _parallel_chunksize(total, n_workers):
    return max(1, int(total) // max(1, int(n_workers) * 8))


def _use_chunked_map(total, n_workers):
    return int(n_workers) > 1 and int(total) >= MIN_CHUNKED_MAP_SAMPLES


def write_feature_outputs(artifact_dir, split_name, df):
    """Write legacy and standard feature-pool names for the same cached rows."""
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame(columns=FEATURE_RESULT_COLUMNS)
    df.to_csv(os.path.join(artifact_dir, f"features_{split_name}.csv"), index=False)
    df.to_csv(os.path.join(artifact_dir, f"feature_pool_{split_name}.csv"), index=False)


def _resolve_s05_workers(n_workers, total):
    if n_workers is None:
        if total < MIN_AUTO_PARALLEL_SAMPLES:
            return 1
        return resolve_n_workers(None, n_items=total)
    return resolve_n_workers(n_workers, n_items=total)


def _iter_sample_results(name, samples, model, n_workers, split_t0):
    total = len(samples)
    interval = _progress_interval(total)
    n_workers = _resolve_s05_workers(n_workers, total)
    if n_workers <= 1 or total <= 1:
        for i, sample in enumerate(samples, start=1):
            yield i - 1, extract_sample(sample, model)
            if i == 1 or i == total or i % interval == 0:
                yield "progress", i
        return

    pool_kwargs = {"max_workers": n_workers, "initializer": _init_feature_worker}
    mp_ctx = multiprocessing_context_from_env()
    if mp_ctx is not None:
        pool_kwargs["mp_context"] = mp_ctx
    use_chunked = _use_chunked_map(total, n_workers)
    chunksize = _parallel_chunksize(total, n_workers)
    suffix = f", chunksize={chunksize}" if use_chunked else ""
    print(f"[{name}] parallel workers={n_workers}, mp_start={mp_ctx.get_start_method()}{suffix}", flush=True)
    ordered = [None] * total
    done = 0
    with ProcessPoolExecutor(**pool_kwargs) as executor:
        if use_chunked:
            args_iter = ((idx, sample) for idx, sample in enumerate(samples))
            result_iter = executor.map(_parallel_sample_worker, args_iter, chunksize=chunksize)
            for idx, rows in result_iter:
                ordered[idx] = rows
                done += 1
                if done == 1 or done == total or done % interval == 0:
                    current_rows = sum(len(part) for part in ordered if part is not None)
                    _print_progress(name, done, total, split_t0, current_rows)
        else:
            futures = [
                executor.submit(_parallel_sample_worker, (idx, sample))
                for idx, sample in enumerate(samples)
            ]
            for future in as_completed(futures):
                idx, rows = future.result()
                ordered[idx] = rows
                done += 1
                if done == 1 or done == total or done % interval == 0:
                    current_rows = sum(len(part) for part in ordered if part is not None)
                    _print_progress(name, done, total, split_t0, current_rows)
    for idx, rows in enumerate(ordered):
        yield idx, rows or []


def _run_split(name, samples, model, artifact_dir, n_workers=1):
    rows = []
    split_t0 = time.time()
    print(f"[{name}] start: {len(samples)} samples", flush=True)
    for idx, result in _iter_sample_results(name, samples, model, n_workers, split_t0):
        if idx == "progress":
            _print_progress(name, result, len(samples), split_t0, len(rows))
        else:
            rows.extend(result)
    df = pd.DataFrame(rows)
    write_feature_outputs(artifact_dir, name, df)
    elapsed = time.time() - split_t0
    print(f"[{name}] {len(df)} rows, elapsed={_format_duration(elapsed)}")
    return {
        "split": name,
        "samples": len(samples),
        "rows": len(df),
        "elapsed_sec": round(elapsed, 3),
        "samples_per_sec": _safe_rate(len(samples), elapsed),
        "rows_per_sec": _safe_rate(len(df), elapsed),
    }


def _to_25hz(s, ppg, acc):
    if _is_25hz_sample(s):
        return (
            np.asarray(ppg, dtype=np.float64),
            np.asarray(acc, dtype=np.float64) if acc is not None and len(acc) > 0 else None,
            25,
        )
    ppg25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS)
    acc25 = None
    if acc is not None and len(acc) > 0:
        from scipy.signal import resample_poly
        acc25 = resample_poly(np.asarray(acc, dtype=np.float32), FEATURE_FS, 100, axis=0).astype(np.float64)
    return ppg25, acc25, 100


def _slice_acc(acc25, start, size):
    if acc25 is None or start >= len(acc25):
        return None
    return acc25[start:start + size]


def _prewindow_to_25hz(s, w, ws):
    n = int(w.shape[0])
    if (_is_25hz_sample(s) or n == int(round(float(ws) * FEATURE_FS)) or (n <= 200 and n > 0 and n % FEATURE_FS == 0)):
        return np.asarray(w, dtype=np.float64), 25
    return _downsample_ppg(np.asarray(w, dtype=np.float64), src_fs=100, tgt_fs=FEATURE_FS), 100


def extract_sample(sample, model):
    base = {"sample_name": sample.get("sample_name", "unknown"), "target": int(sample.get("target", 0))}
    try:
        ppg, acc = load_ppg(sample), load_acc(sample)
        ok, err = validate_h5_file(sample["h5_file"], base["sample_name"])
        if not ok: raise ValueError(err)
    except Exception as exc:
        return [{**base, "window_idx": -1, "commercial_score": None, "commercial_pred": 0,
                 "fallback": True, "fallback_reason": str(exc)}]
    rows = []
    if is_prewindowed_signal(ppg):
        mode = detect_green_mode(ppg)
        for idx in range(3, ppg.shape[0]):
            win25, _ = _prewindow_to_25hz(sample, ppg[idx], COMMERCIAL_WIN_SEC)
            try:
                ir, amb, g1, g2, g3 = get_channels_from_window(win25, mode)
                acc_seg = None
                if acc is not None and is_prewindowed_signal(acc) and idx < acc.shape[0]:
                    acc_seg, _ = _prewindow_to_25hz(sample, acc[idx], COMMERCIAL_WIN_SEC)
                is_live, score, _, _ = model.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, acc_seg))
                nf = extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS)
                r = {**base, "window_idx": idx, "commercial_score": float(score) if score is not None else -2000.0,
                     "commercial_pred": is_live, "fallback": False, "fallback_reason": None}
                r.update(nf)
                rows.append(r)
            except Exception:
                continue
        return rows
    ppg25, acc25, _ = _to_25hz(sample, ppg, acc)
    mode = detect_green_mode(ppg)
    sw, ss = int(round(COMMERCIAL_WIN_SEC * FEATURE_FS)), int(round(COMMERCIAL_STRIDE_SEC * FEATURE_FS))
    for step in range(3, max(0, (len(ppg25) - sw) // ss + 1)):
        start = step * ss
        win = ppg25[start:start + sw, :]
        try:
            ir, amb, g1, g2, g3 = get_channels_from_window(win, mode)
            acc_seg = _slice_acc(acc25, start, sw)
            is_live, score, _, _ = model.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, acc_seg))
            nf = extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS)
            r = {**base, "window_idx": step, "commercial_score": float(score) if score is not None else -2000.0,
                 "commercial_pred": is_live, "fallback": False, "fallback_reason": None}
            r.update(nf)
            rows.append(r)
        except Exception:
            continue
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", default="artifacts/parallel")
    p.add_argument("--splits_dir", default="artifacts")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--n_workers", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    _write_commercial_manifest(args.artifact_dir)
    splits = load_splits(args.splits_dir)
    model = OldLivenessModel()
    t0 = time.time()
    timing_rows = []
    for name in ["train", "valid", "test"]:
        samples = splits[name][:args.max_samples] if args.max_samples else splits[name]
        timing_rows.append(_run_split(name, samples, model, args.artifact_dir, args.n_workers))
    total_elapsed = time.time() - t0
    timing_rows.append({
        "split": "total",
        "samples": int(sum(r["samples"] for r in timing_rows)),
        "rows": int(sum(r["rows"] for r in timing_rows)),
        "elapsed_sec": round(total_elapsed, 3),
        "samples_per_sec": _safe_rate(sum(r["samples"] for r in timing_rows), total_elapsed),
        "rows_per_sec": _safe_rate(sum(r["rows"] for r in timing_rows), total_elapsed),
    })
    _print_timing_summary(timing_rows)
    print(f"Done ({total_elapsed:.1f}s / {_format_duration(total_elapsed)})")

if __name__ == "__main__":
    main()
