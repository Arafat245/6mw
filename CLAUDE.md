# CLAUDE.md

## Project Overview

Predicting 6-Minute Walk Distance (6MWD) from hip-worn accelerometer data in Pediatric-Onset Multiple Sclerosis (POMS). n=101 (POMS=38, Healthy=63) after excluding M22 (data quality) and M44 (too-short recording).

## Key Rules

### Data & Analysis
- **Always n=101.** M22 and M44 are excluded from all analyses.
- **LOO CV only.** Leave-One-Subject-Out cross-validation. Never use L5SO or random splits.
- **Report in meters.** Convert all 6MWD predictions and true values from feet to meters (×0.3048) before evaluating. MAE, RMSE, etc. should be in meters (m), not feet.
- **Demographics = basic only.** Demo features: Age, Sex, Height, BMI, cohort_POMS. Never include clinical scores (BDI, MFIS, etc.) as predictors — they require clinic visits.
- **No CCPT.** Do not use predicted 6MWD as input to predict 6MWD (circular).
- **Home pipeline is fully clinic-free.** `home/step0→step3` pipeline. No clinic data used anywhere.
- **Full recording for home.** No daytime filtering — keeping all data (including sleep/sedentary) improves prediction (R²=0.454 vs 0.365 with daytime only).
- **Hardcoded FS=30 for all subjects.** Do NOT resample non-30Hz subjects. Resampling hurts performance (resample_poly R²=0.342, linear interp R²=0.373, hardcoded 30Hz R²=0.454).
- **Keep all walking bouts.** Do not filter/verify bouts — verified bouts give worse results (R²=0.423 vs 0.454).
- **Match clinic files by subject key** (e.g., `C61`), not full filename — `target_6mwd.csv` years don't match `csv_raw2/`/`csv_preprocessed2/` filenames.

### Sampling Frequencies
- 91 subjects at 30 Hz, 8 at 60 Hz (M26, M27, M29, M31, M32, M43, M45, C75), 2 at 100 Hz (M48, M49)
- Sampling rate is in GT3X metadata (`info.txt` inside ZIP, field `Sample Rate`)
- pygt3x 0.7.1 returns float64 timestamps (Unix epoch seconds) on both Linux and Windows

### Coding & Experiments
- **New experiments go in `temporary_experiments/`.** All new scripts, cached features, cloned repos, and outputs must live in `temporary_experiments/` until they produce results that improve on current best. Only then promote to `home/`, `analysis/`, etc.
- **Cache features.** Save extracted features to `.npz` or `.csv` so re-extraction is instant. Use `feats/` directory for finalized features only.
- **Use cached data.** Read from `home_full_recording_npz/` (not GT3X) when possible. Use `csv_preprocessed2/` (not csv_raw2) for clinic.
- **Keep experiments fast.** Prefer small focused tests (5-10 configs) over exhaustive grid search. Kill experiments early if trends are clear.
- **No unnecessary runs.** Don't re-run pipelines that haven't changed. Don't recompute features that are already cached.
- **Be honest about time estimates.**

### Paper (LaTeX in `POMS/`)
- **Max 7 figures, max 7 tables.** Both budgets are currently full — warn before adding new ones.
- **All outputs go to `feats/`.** Tables as CSV, figures as PNG/SVG.
- **Tables go to `POMS/tables/`, figures to `POMS/figures/`.**

### Models
- **Ridge regression** is the primary model. Current best alphas: Clinic=5, Home=20.
- **Foundation models** (MOMENT, Chronos, LimuBERT) are comparison baselines only. Handcrafted features beat them.
- **DL models** (TCN, LSTM, Transformer) are used separately from foundation models, not mixed.

## Current Best Results

| Setting | Features | R² | MAE (m) | ρ |
|---|---|---|---|---|
| Clinic | Gait+CWT+WalkSway+Demo (55f) | 0.806 | 31.1 | 0.880 |
| Home (clinic-free) | PerBout-Top20+Demo(4) (24f, Spearman inside LOO) | 0.454 | 55.5 | 0.659 |

## Home Pipeline (step0 → step3)

```bash
python home/step0_gt3x_to_npz.py          # GT3X → NPZ (~60 min, one-time)
python home/step1_detect_walking_bouts.py --save-csv  # Detect + save bouts (~18 min)
python home/step2_extract_features.py      # Extract 153 features (~12 min)
python home/step3_predict.py               # LOO CV → R²=0.454 (<1 sec)
```

Quick reproduction: `python home/step3_predict.py` (cached features)
From saved bouts: `python home/reproduce_from_bouts.py [--bout-dir walking_bouts]`
Full results table: `python analysis/results_table_final.py` (clinic + home)

## Key Scripts

| Script | Purpose |
|---|---|
| `home/step0_gt3x_to_npz.py` | GT3X → full recording NPZ (no filtering) |
| `home/step1_detect_walking_bouts.py` | Walking bout detection + optional CSV saving |
| `home/step2_extract_features.py` | Per-bout gait + activity feature extraction |
| `home/step3_predict.py` | LOO CV prediction (R²=0.454) |
| `home/reproduce_from_bouts.py` | Reproduce from saved walking bout CSVs |
| `clinic/reproduce_c2.py` | Clinic preprocessing + Gait/CWT extraction |
| `clinic/extract_walking_sway.py` | Clinic WalkSway features |
| `analysis/results_table_final.py` | Full results table (clinic + home, all 101 subjects) |

## Directory Layout

- `home_full_recording_npz/` — Full recording NPZ files (101 subjects, ~2.4 GB)
- `walking_bouts/` — Walking bout CSVs per subject (Timestamp, X, Y, Z, 186K files)
- `feats/` — All extracted features, outputs, tables, figures
- `csv_raw2/` — Clinic 6MWT raw data
- `csv_preprocessed2/` — Clinic preprocessed data
- `Accel files/` — Raw home GT3X files
- `POMS/` — Paper LaTeX source, figures, tables
- `archive/` — Old experimental scripts (NOT part of final pipeline)
- `temporary_experiments/` — Scratch space for new experiments
- `notebooks/` — Exploratory/legacy Jupyter notebooks

## Experiment History

- **Clinic per-bout (60s windows):** PerBout-Top20+Demo(4) R²=0.675 — use 60s windows, not 30s
- **Home daytime only:** R²=0.365 — worse than full recording
- **Home worn-time filter only:** R²=0.407 — worse than no filter
- **Resampling non-30Hz to 30Hz:** resample_poly R²=0.342, linear interp R²=0.373 — both worse
- **Walking bout verification:** R²=0.423 — worse than keeping all bouts
