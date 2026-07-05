# Parallel Commercial Guard

This folder is a standalone parallel solution for PPG/ACC watch
wearing-liveness detection. It keeps the frozen commercial feature/model
contract and trains an independent lightweight XGBoost model for veto-risk
review.

## Standalone Boundary

You can copy this `parallel/` folder to another location and run it directly.
It does not import scripts from the parent `new_codex_1` directory.

Runtime inputs still come from outside the folder:

- an H5 dataset directory passed by `--dataset_dir`;
- Python packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `h5py`,
  `joblib`, `matplotlib`, and `scipy`;
- optional Graphviz `dot` for PNG tree rendering.

The pipeline creates its own local `artifacts/` directory by default.

## Commercial Contract

The commercial model lives in `s01_model.py` and is treated as frozen.
`commercial_model_manifest()` records:

- commercial feature names;
- AdaBoost `tree_num`, `tree_node`, and `detect_tree_threshold`;
- Stage1 timing/gate parameters;
- SHA256 hashes for commercial tree index/value arrays;
- `frozen=True`.

The pipeline writes `artifacts/parallel/commercial_model_manifest.json` as
acceptance evidence. The independent model must not alter the commercial model
or commercial feature extraction.

## Data Format

The dataset directory should contain `.h5` files. Supported sample layouts:

- normal sample group containing `ppg`, `target`, and optional `acc`;
- grouped-window sample where child groups are named like `*_w20_1` and each
  child contains `ppg` and optional `acc`.

Supported PPG shapes:

- `(40, T)`;
- `(N_windows, 40, T_window)`.

The pipeline scans H5 files, creates `artifacts/splits.json`, then reuses that
split on later runs unless `--force_split` is provided.

## Quick Start

```bash
cd parallel
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --guard_mode shadow --explain
```

Use `--dry_run` to print commands without executing:

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --dry_run
```

Regenerate the split:

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --force_split
```

## Pipeline

`s10_pipeline.py` runs:

1. Auto split: scan H5 files and write `artifacts/splits.json` when needed.
2. `s05_extract_features.py`: extract commercial scores and new PPG/ACC
   features for all windows.
3. `s06_select_features.py`: rank and select deployment-friendly independent
   model features.
4. `s07_train_model.py`: train a tiny independent XGBoost model.
5. `s08_fusion.py`: write veto fusion configuration.
6. `s09_evaluate.py`: compare commercial-only output with selected guard mode.
7. `s11_explain.py`: optional explainability reports when `--explain` is set.

## Guard Modes

- `bypass`: final output equals commercial output.
- `shadow`: final output equals commercial output; guard risk is logged.
- `soft_guard`: final output equals commercial output; high risk requests
  extended detection.
- `hard_veto`: can change a commercial positive to negative only when risk is
  persistent.

Default mode is `shadow`.

Persistent veto requires:

```text
risk_count >= min_veto_windows
risk_ratio >= min_veto_ratio
```

Defaults:

```text
min_veto_windows = 2
min_veto_ratio = 0.4
```

The guard never promotes a commercial negative to positive.

## Difference From Cascade

Parallel trains an independent model on all extracted windows and uses it as a
veto-risk reviewer. It is not limited to commercial-positive error candidates.

This makes parallel useful for:

- shadow-mode disagreement analysis;
- feature research;
- sample-distribution and model-branch explainability;
- independent veto-risk review.

For conservative commercial launch, cascade is usually the safer first-line
architecture because it is easier to explain as a serial safety layer after the
commercial model.

## Outputs

Main artifacts:

- `artifacts/splits.json`
- `artifacts/parallel/commercial_model_manifest.json`
- `artifacts/parallel/features_{train,valid,test}.csv`
- `artifacts/parallel/selected_features.json`
- `artifacts/parallel/new_model.json`
- `artifacts/parallel/new_model_bundle.pkl`
- `artifacts/parallel/fusion_config.json`
- `artifacts/parallel/evaluation_report.json`
- `artifacts/parallel/evaluation_samples.csv`
- `artifacts/parallel/evaluation_comparison.csv`

Explainability artifacts:

- `artifacts/parallel/feature_review/*`
- `artifacts/parallel/figures/*.png`
- `artifacts/parallel/tree_export/*`
- `artifacts/parallel/error_trace/*`

Current image policy is high-resolution PNG only. CSV, JSON, DOT, and Markdown
source/audit files are retained.

## Manual Feature Review

Feature selection writes:

```text
artifacts/parallel/feature_review/ranked_features.csv
artifacts/parallel/feature_review/ranked_features.json
artifacts/parallel/feature_review/ranked_features.md
artifacts/parallel/feature_review/manual_feature_selection_template.json
```

Edit the template, save it as `manual_feature_selection.json`, then run:

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --manual_features artifacts/parallel/feature_review/manual_feature_selection.json
```

Label and label-proxy columns are rejected from manual feature files.

## Important Parameters

- Stage1 feature extraction defaults in `s02_features.py`:
  - `DEFAULT_STAGE1_DC_THRESHOLD = 0.3e6`
  - `DEFAULT_STAGE1_AC_DC_THRESHOLD = 1.0`

If strict online-commercial threshold parity is required, pass or restore the
online threshold explicitly and keep it recorded in the produced artifacts.

## Recommended Use

Use parallel as a shadow and analysis solution first:

1. Run `shadow` to collect independent disagreement data.
2. Review `feature_review/`, `figures/`, `tree_export/`, and `error_trace/`.
3. Compare with cascade before any intervention mode.
4. Use `hard_veto` only for offline evaluation or tightly controlled gray
   release.
