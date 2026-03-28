# CLAUDE.md

## Project Overview

Predicting 6-Minute Walk Distance (6MWD) from hip-worn accelerometer data in Pediatric-Onset Multiple Sclerosis (POMS). n=101 (POMS=38, Healthy=63) after excluding M22 (data quality) and M44 (too-short recording).

## Key Rules

### Data & Analysis
- **Always n=101.** M22 and M44 are excluded from all analyses.
- **LOO CV only.** Leave-One-Subject-Out cross-validation. Never use L5SO or random splits.
- **Demographics = basic only.** Demo features: Age, Sex, Height, BMI, cohort_POMS. Never include clinical scores (BDI, MFIS, etc.) as predictors — they require clinic visits.
- **No CCPT.** Do not use predicted 6MWD as input to predict 6MWD (circular).
- **Home pipeline is fully clinic-free.** `home/extract_clinicfree_features.py` — ENMO+HR walking detection, per-bout feature extraction, robust aggregation → `feats/home_clinicfree_top20.npz`. No clinic data used anywhere.
- Legacy: `home/home_hybrid_models_v2.py` (clinic-informed, archived) and `home/preprocess_raw.py` (secondary).

### Coding & Experiments
- **New experiments go in `temporary_experiments/`.** All new scripts, cached features, cloned repos, and outputs must live in `temporary_experiments/` until they produce results that improve on current best. Only then promote to `feats/`, `analysis/`, etc.
- **Cache features.** Save extracted features to `.npz` or `.csv` so re-extraction is instant. Use `feats/` directory for finalized features only.
- **Use cached data.** Read from `csv_home_daytime/` (not GT3X) and `csv_preprocessed2/` (not csv_raw2) when possible.
- **Keep experiments fast.** Prefer small focused tests (5-10 configs) over exhaustive grid search. Kill experiments early if trends are clear. LOO with XGBoost on 101 subjects takes ~0.5-1s per config.
- **No unnecessary runs.** Don't re-run pipelines that haven't changed. Don't recompute features that are already cached.
- **Be honest about time estimates.**
- **Push to GitHub on improvement.** Whenever results improve or a better model is achieved:
  1. Update `README.md` — Best Results table, Results History, and Step H3 reproduction instructions.
  2. Update `CLAUDE.md` — Current Best Results table.
  3. Save all new features/outputs to `feats/`.
  4. Add detailed reproduction steps in README.md (script name, inputs, outputs, runtime).
  5. Commit and push all changes to GitHub.

### Paper (LaTeX in `POMS/`)
- **Max 7 figures, max 7 tables.** Both budgets are currently full — warn before adding new ones.
- **All outputs go to `feats/`.** Tables as CSV, figures as PNG/SVG.
- **Tables go to `POMS/tables/`, figures to `POMS/figures/`.**

### Models
- **Ridge regression** is the primary model. Current best alphas: Clinic=10, Home=5.
- **Foundation models** (MOMENT, Chronos, LimuBERT) are comparison baselines only. Handcrafted features beat them.
- **DL models** (TCN, LSTM, Transformer) are used separately from foundation models, not mixed.

## Current Best Results

| Setting | Features | R² | MAE (ft) | ρ |
|---|---|---|---|---|
| Clinic | Gait+CWT+WalkSway+Demo (55f) | 0.806 | 102 | 0.880 |
| Home (clinic-free) | FwdSel+CWT-PerBout+Demo (30f) | 0.736 | 132 | 0.823 |

## Reproducing Home Best (R²=0.555)

```bash
conda activate pygt3x-env  # Python 3.11 with pygt3x
# Step 0 (one-time, ~2.5h): GT3X -> daytime NPZ
python temporary_experiments/step0_gt3x_to_npz.py
# Step 1 (~8 min): Extract 153 features, reproduce R²=0.441
python temporary_experiments/step1_extract_and_predict.py
# Step 2 (~30 min): Forward selection -> R²=0.555
python temporary_experiments/step2_goldman_features.py
```

Inputs: `Accel files/*/\*.gt3x`, `6mw_segmented_walk_data_dict.pkl`, `Accel files/PedMSWalkStudy_Demographic.xlsx`
Outputs: `feats/home_clinicfree_features.csv`, `feats/goldman_features.csv`, `feats/target_6mwd.csv`

## Key Scripts

| Script | Purpose |
|---|---|
| `temporary_experiments/step0_gt3x_to_npz.py` | GT3X -> daytime NPZ (one-time) |
| `temporary_experiments/step1_extract_and_predict.py` | Clinic-free feature extraction + Top-20 model |
| `temporary_experiments/step2_goldman_features.py` | Goldman features + forward selection (best home model) |
| `clinic/reproduce_c2.py` | Clinic preprocessing + Gait/CWT extraction |
| `clinic/extract_walking_sway.py` | Clinic WalkSway features |
| `home/extract_clinicfree_features.py` | Home clinic-free features (reference implementation) |
| `analysis/results_table_final.py` | Main results table (all feature combos) |

## Directory Layout

- `feats/` — All extracted features, outputs, tables, figures
- `csv_home_daytime_npz/` — Cached home daytime data (NPZ, from GT3X)
- `POMS/` — Paper LaTeX source, figures, tables
- `archive/` — Old experimental scripts (NOT part of final pipeline)
- `temporary_experiments/` — Scratch space for new experiments
- `notebooks/` — Exploratory/legacy Jupyter notebooks
