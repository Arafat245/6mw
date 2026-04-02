# Gait Assessment in Pediatric-Onset Multiple Sclerosis Using Wearable Accelerometry

Predicting 6MWD from hip-worn accelerometer data collected during clinic 6-minute walk tests and home free-living monitoring in POMS.

**Subjects:** n=101 (POMS=38, Healthy=63), consistent across all analyses. Two subjects excluded: M22 (data quality) and M44 (too-short recording). Subject list in `feats/target_6mwd.csv`.

## Results Table

| Feature Set | #f | Clinic R² | Clinic MAE (m) | Clinic ρ | Home R² | Home MAE (m) | Home ρ |
|---|---|---|---|---|---|---|---|
| Gait | 11 | 0.682 | 42.7 | 0.801 | 0.145 | 70.1 | 0.377 |
| CWT | 28 | 0.357 | 60.2 | 0.601 | 0.150 | 67.9 | 0.462 |
| WalkSway | 12 | 0.403 | 54.2 | 0.715 | 0.056 | 73.3 | 0.313 |
| Demo | 4 | 0.362 | 60.8 | 0.595 | 0.362 | 60.8 | 0.595 |
| PerBout-Top20 | 20 | 0.617 | 45.0 | 0.791 | 0.182 | 67.2 | 0.453 |
| PerBout-Top20+Demo | 24 | 0.679 | 39.7 | 0.841 | **0.452** | **56.0** | **0.649** |
| **Gait+CWT+WS+Demo** | **55** | **0.806** | **31.2** | **0.880** | 0.281 | 63.8 | 0.543 |

Reproduce: `python analysis/reproduce_results_table_best_models.py` (~1 min)

- **n=101**, LOO CV, no data leakage. All metrics in meters.
- **Both clinic and home use Ridge regression.** Clinic α=5, Home α=20.
- **Demo-only row:** cohort_POMS, Age, Sex, BMI — same for clinic and home, Ridge α=20
- **Demo in combos:** Clinic uses Height, Home uses BMI (different best Demo per setting)
- **Clinic Gait/CWT/WS:** extracted from full 6MWT, no feature selection, Ridge α varies per set
- **Clinic PerBout:** 60s windows of 6MWT, Spearman Top-20 inside LOO, Ridge α=5
- **Clinic Gait+CWT+WS+Demo:** all 55 features, no selection, Ridge α=5
- **Home PerBout:** all walking bouts from full recording, Spearman Top-20 inside LOO, Ridge α=20
- **Home Gait/CWT/WS:** VM-based (no gravity removal, no axis alignment), Top-10 clean bouts ≥60s, Spearman Top-11 inside LOO
- **Home Gait+CWT+WS+Demo:** Spearman Top-20 on Gait/CWT/WS accel pool, append Demo(BMI), Ridge α=20

---

## Quick Reproduction

```bash
# Results table with best models (7 rows, ~1 min)
python analysis/reproduce_results_table_best_models.py

# All models comparison — clinic (7 models, <1 min)
python clinic/predict_all_models.py

# All models comparison — home (7 models + voting ensembles for reference, <30 sec)
python home/step3_predict_all_models.py

# All paper tables (8 CSVs, ~70 sec)
python analysis/generate_paper_tables.py

# All paper figures (6 PNGs, ~15 sec)
python analysis/generate_paper_figures.py

# Full combination tables (~13 min)
python analysis/results_table_full.py
```

---

## Data Requirements

| # | File | Description |
|---|---|---|
| 1 | `Accel files/*/*.gt3x` | Raw home free-living accelerometer (hip-worn, ±6g, ~7–10 days). 91 at 30 Hz, 8 at 60 Hz, 2 at 100 Hz. |
| 2 | `csv_raw2/*.csv` | Clinic 6MWT raw data (Timestamp, X, Y, Z). 101 files. |
| 3 | `csv_preprocessed2/*.csv` | Clinic preprocessed data (AP, ML, VT, _bp, ENMO). 101 files. |
| 4 | `feats/target_6mwd.csv` | Subject list with 6MWD ground truth (feet). 103 rows, exclude M22/M44 → 101. |
| 5 | `SwayDemographics.xlsx` | Demographics: ID, Age, Sex, Height, Weight, BMI, EDSS, MFIS, BDI. |

---

## CLINIC PIPELINE

### Step 1: Gait/CWT/WalkSway Feature Extraction

```bash
python clinic/extract_gait_cwt_ws_features.py    # ~2 min
# Input:  csv_preprocessed2/*.csv + csv_raw2/*.csv
# Output: feats/clinic_gait_features.csv (101 x 12)
#         feats/clinic_cwt_features.csv (101 x 29)
#         feats/clinic_walksway_features.csv (101 x 13)
```

- **Gait (11f):** `extract_gait10()` from preprocessed AP/ML/VT + `vt_rms_g` — cadence, step regularity, harmonic ratios, jerk, ENMO, cadence slope, spectral entropy
- **CWT (28f):** `extract_cwt()` from raw XYZ VM — Morlet wavelet, 6 temporal segments, mean/std/slope
- **WalkSway (12f):** `extract_walking_sway()` from preprocessed AP/ML/VT — 10 ENMO-normalized sway features + 2 ratios

### Step 2: PerBout Feature Extraction (124f)

```bash
python clinic/extract_perbout_features.py    # ~1 min
# Input:  csv_raw2/*.csv
# Output: feats/clinic_perbout_features.csv (101 x 125)
```

- Split 6MWT into 60s non-overlapping windows (trim first/last 10s)
- Extract 20 per-bout features per window (same features as home PerBout)
- Aggregate across windows: 6 stats (median, IQR, p10, p90, max, CV) = 120 gait + 4 meta = 124 features

### Step 3: Clinic Prediction

```bash
python clinic/predict.py               # <1 sec (cached features)
# Input:  feats/clinic_gait_features.csv + clinic_cwt_features.csv
#         + clinic_walksway_features.csv + SwayDemographics.xlsx
# Output: R²=0.806, MAE=31.2m, ρ=0.880
```

Gait(11) + CWT(28) + WalkSway(12) + Demo(Height, 4) = 55 features, no feature selection, Ridge α=5.

### Clinic All Models Comparison

```bash
python clinic/predict_all_models.py    # <1 min
# Input:  feats/clinic_{gait,cwt,walksway}_features.csv + SwayDemographics.xlsx
# Output: R², MAE, ρ for Ridge, Lasso, ElasticNet, KNN, SVR, RF, XGBoost
```

---

## HOME PIPELINE (Clinic-Free)

**No clinic data used anywhere.** Full recording (no daytime filtering) — keeping all data including sleep/sedentary improves prediction.

### Step 0: GT3X → NPZ

```bash
python home/step0_gt3x_to_npz.py          # ~60 min, one-time
# Input:  Accel files/*/*.gt3x + feats/target_6mwd.csv
# Output: home_full_recording_npz/{key}.npz (xyz + timestamps, 101 files, ~2.4 GB)
```

Reads raw GT3X via `pygt3x`, saves full recording as-is (no filtering, no resampling).

### Step 1: Walking Bout Detection

```bash
python home/step1_detect_walking_bouts.py --save-csv    # ~18 min with --save-csv
# Input:  home_full_recording_npz/*.npz
# Output: feats/home_walking_bouts.pkl (bout indices per subject)
#         walking_bouts/{subj_id}/bout_*.csv (Timestamp, X, Y, Z — 186,012 files)
```

Three-stage detection at FS=30 Hz:
1. ENMO ≥ 0.015g per second, min 10s bouts
2. Harmonic ratio ≥ 0.2 in 10s windows
3. Merge gaps ≤ 5s

### Step 2: PerBout Feature Extraction (153f)

```bash
python home/step2_extract_features.py      # ~12 min
# Input:  home_full_recording_npz/*.npz + feats/home_walking_bouts.pkl
# Output: feats/home_perbout_features.csv (101 x 154)
```

Per-bout preprocessing: gravity removal → Rodrigues rotation → PCA yaw alignment → AP, ML, VT + bandpass + ENMO. Extract 20 features per bout, aggregate with 6 stats (median, IQR, p10, p90, max, CV) = 120 gait + 4 meta + 29 activity = **153 features**.

### Step 3: Home Prediction

```bash
python home/step3_predict.py               # <1 sec (cached features)
# Input:  feats/home_perbout_features.csv + SwayDemographics.xlsx
# Output: R²=0.452, MAE=56.0m, ρ=0.649 (Ridge only baseline)
```

Spearman Top-20 inside LOO + Demo(4), Ridge α=20.

### Home All Models Comparison + Voting Ensemble

```bash
python home/step3_predict_all_models.py    # <30 sec (cached features)
# Input:  feats/home_perbout_features.csv + SwayDemographics.xlsx
# Output: R², MAE, ρ for Ridge, Lasso, ElasticNet, KNN, SVR, RF, XGBoost + voting ensembles
```

Comparison script only — Ridge(α=20) is used as the home model (R²=0.452). Vote ensemble (R²=0.478) gives only marginal improvement, not worth the added complexity.

### Home Gait/CWT/WalkSway Feature Extraction

```bash
python home/extract_gait_cwt_ws_features.py    # ~3 min
# Input:  walking_bouts/{subj_id}/bout_*.csv
# Output: feats/home_gait_features.csv (101 x 67)
#         feats/home_cwt_features.csv (101 x 169)
#         feats/home_walksway_features.csv (101 x 73)
```

- Select **Top-10 longest clean bouts ≥60s** per subject (drift ≤ 0.5g, orientation change ≤ 10°)
- Compute **VM = sqrt(X² + Y² + Z²)** — no gravity removal, no axis alignment (reduces orientation artifacts)
- Extract Gait(11), CWT(28), WalkSway(12) per bout using clinic functions on VM signal
- Aggregate across bouts: 6 stats per feature

### Alternative Reproduction

```bash
# From saved walking bout CSVs (no NPZ or pkl needed):
python home/reproduce_from_bouts.py [--bout-dir walking_bouts]    # ~18 min
```

### Full Pipeline from Scratch

```bash
python home/step0_gt3x_to_npz.py                          # GT3X → NPZ (~60 min)
python home/step1_detect_walking_bouts.py --save-csv       # Detect + save bouts (~18 min)
python home/step2_extract_features.py                      # PerBout features (~12 min)
python home/extract_gait_cwt_ws_features.py                # Gait/CWT/WS features (~3 min)
python home/step3_predict_all_models.py                    # Predict + all models (<30 sec)
```

---

## Evaluation & Paper Outputs

### Results Table (Best Models)

```bash
python analysis/reproduce_results_table_best_models.py    # ~1 min
# Input:  feats/*.csv + SwayDemographics.xlsx
# Output: results/results_table_best_models.csv (7 rows)
```

Clinic=Ridge(α=5), Home=Ridge(α=20).

### Paper Tables (8 CSVs)

```bash
python analysis/generate_paper_tables.py            # ~70 sec
# Input:  feats/*.csv + SwayDemographics.xlsx
# Output: results/paper_tables/*.csv + results/results_table_final.csv
```

| Table | File | Description |
|---|---|---|
| 1 | `results/paper_tables/demographics_table.csv` | Demographic/clinical characteristics by cohort |
| 2 | `results/paper_tables/feature_descriptions.csv` | Feature categories and names |
| 3 | `results/paper_tables/best_predictions.csv` | Per-subject LOO predictions (best models) |
| 4 | `results/paper_tables/error_analysis_by_cohort.csv` | Error metrics by cohort (All/POMS/Healthy) |
| 5 | `results/paper_tables/feature_correlations.csv` | Spearman ρ with 6MWD by setting/cohort |
| 6 | `results/paper_tables/ms_vs_healthy_features.csv` | POMS vs Healthy group differences (Cohen's d, BH-corrected) |
| 7 | `results/paper_tables/clinical_corr_ms_only.csv` | Clinic feature–clinical score correlations (MS only) |
| 8 | `results/paper_tables/clinical_corr_ms_home.csv` | Home feature–clinical score correlations (MS only) |

### Paper Figures (6 PNGs)

```bash
python analysis/generate_paper_figures.py           # ~15 sec
# Input:  feats/*.csv + SwayDemographics.xlsx
# Output: results/paper_figures/*.png
```

| Figure | File | Description |
|---|---|---|
| 1 | `results/paper_figures/fig_predicted_vs_actual.png` | Predicted vs actual 6MWD scatter (home + clinic) |
| 2 | `results/paper_figures/fig_bland_altman.png` | Bland-Altman agreement plots |
| 3 | `results/paper_figures/fig_shap_importance.png` | SHAP feature importance (clinic + home) |
| 4 | `results/paper_figures/heatmap_feature_6mwd_corr.png` | Feature–6MWD correlation heatmap by setting/cohort |
| 5 | `results/paper_figures/heatmap_clinical_corr_clinic.png` | Clinic feature–clinical score correlations (POMS only) |
| 6 | `results/paper_figures/heatmap_clinical_corr_home.png` | Home feature–clinical score correlations (POMS only) |

### Full Combination Tables

```bash
python analysis/results_table_full.py               # ~13 min
# Input:  feats/*.csv + SwayDemographics.xlsx
# Output: results/results_no_selection.csv + results/results_spearman_top20.csv
```

All feature set combinations. Clinic=Ridge(α=5), Home=Ridge(α=20).

---

## All Feature Files

| File | Created by | Features | Description |
|---|---|---|---|
| `feats/target_6mwd.csv` | — | — | Ground truth 6MWD (feet), 101 subjects |
| `feats/clinic_gait_features.csv` | `clinic/extract_gait_cwt_ws_features.py` | 11 + key | Clinic Gait from 6MWT |
| `feats/clinic_cwt_features.csv` | `clinic/extract_gait_cwt_ws_features.py` | 28 + key | Clinic CWT from 6MWT |
| `feats/clinic_walksway_features.csv` | `clinic/extract_gait_cwt_ws_features.py` | 12 + key | Clinic WalkSway from 6MWT |
| `feats/clinic_perbout_features.csv` | `clinic/extract_perbout_features.py` | 124 + key | Clinic PerBout (60s windows) |
| `feats/home_walking_bouts.pkl` | `home/step1_detect_walking_bouts.py` | — | Walking bout indices per subject |
| `feats/home_perbout_features.csv` | `home/step2_extract_features.py` | 153 + key | Home PerBout (all bouts) |
| `feats/home_gait_features.csv` | `home/extract_gait_cwt_ws_features.py` | 66 + key | Home Gait (VM, Top-10 clean bouts) |
| `feats/home_cwt_features.csv` | `home/extract_gait_cwt_ws_features.py` | 168 + key | Home CWT (VM, Top-10 clean bouts) |
| `feats/home_walksway_features.csv` | `home/extract_gait_cwt_ws_features.py` | 72 + key | Home WalkSway (VM, Top-10 clean bouts) |

---

## Key Scripts

| Script | Purpose |
|---|---|
| `home/step0_gt3x_to_npz.py` | GT3X → full recording NPZ (no filtering) |
| `home/step1_detect_walking_bouts.py` | Walking bout detection + optional CSV saving |
| `home/step2_extract_features.py` | Home PerBout feature extraction (153f) |
| `home/step3_predict.py` | Home Ridge-only prediction (R²=0.452, baseline) |
| `home/step3_predict_all_models.py` | Home all models comparison (Ridge best, R²=0.452) |
| `home/extract_gait_cwt_ws_features.py` | Home Gait/CWT/WalkSway features (VM-based) |
| `home/reproduce_from_bouts.py` | Reproduce from saved bout CSVs |
| `clinic/predict.py` | Clinic Ridge prediction (R²=0.806) |
| `clinic/predict_all_models.py` | Clinic all models comparison |
| `clinic/reproduce_c2.py` | Clinic preprocessing + Gait/CWT extraction functions |
| `clinic/extract_walking_sway.py` | Clinic WalkSway extraction function |
| `clinic/extract_gait_cwt_ws_features.py` | Clinic Gait/CWT/WalkSway feature extraction |
| `clinic/extract_perbout_features.py` | Clinic PerBout feature extraction (60s windows) |
| `analysis/reproduce_results_table_best_models.py` | Results table with best models (~1 min) |
| `analysis/results_table_full.py` | Full combination tables (~13 min) |
| `analysis/generate_paper_tables.py` | Generate all paper tables (8 CSVs, ~70 sec) |
| `analysis/generate_paper_figures.py` | Generate all paper figures (6 PNGs, ~15 sec) |

---

## Directory Layout

```
6mw/
├── home/                           HOME PIPELINE
│   ├── step0_gt3x_to_npz.py         GT3X → NPZ
│   ├── step1_detect_walking_bouts.py Bout detection [--save-csv]
│   ├── step2_extract_features.py     PerBout features (153f)
│   ├── step3_predict.py              Ridge baseline (R²=0.452)
│   ├── step3_predict_all_models.py   All models comparison
│   ├── extract_gait_cwt_ws_features.py Gait/CWT/WS features (VM)
│   └── reproduce_from_bouts.py       Reproduce from bout CSVs
│
├── clinic/                         CLINIC PIPELINE
│   ├── predict.py                    Ridge prediction (R²=0.806)
│   ├── predict_all_models.py         All models comparison
│   ├── reproduce_c2.py               Preprocessing + Gait/CWT functions
│   ├── extract_walking_sway.py       WalkSway function
│   ├── extract_gait_cwt_ws_features.py Gait/CWT/WS feature extraction
│   └── extract_perbout_features.py   PerBout features (60s windows)
│
├── analysis/                       EVALUATION & PAPER OUTPUTS
│   ├── reproduce_results_table_best_models.py  Results table (best models)
│   ├── results_table_full.py             Full combination tables
│   ├── generate_paper_tables.py          Paper tables (8 CSVs)
│   └── generate_paper_figures.py         Paper figures (6 PNGs)
│
├── feats/                          CACHED FEATURES (10 files)
│   ├── target_6mwd.csv               Ground truth
│   ├── home_walking_bouts.pkl        Walking bout indices
│   ├── home_perbout_features.csv     Home PerBout (153f)
│   ├── home_gait_features.csv        Home Gait (66f)
│   ├── home_cwt_features.csv         Home CWT (168f)
│   ├── home_walksway_features.csv    Home WalkSway (72f)
│   ├── clinic_gait_features.csv      Clinic Gait (11f)
│   ├── clinic_cwt_features.csv       Clinic CWT (28f)
│   ├── clinic_walksway_features.csv  Clinic WalkSway (12f)
│   └── clinic_perbout_features.csv   Clinic PerBout (124f)
│
├── results/                        RESULTS & PAPER OUTPUTS
│   ├── results_table_best_models.csv  Results table (7 rows)
│   ├── results_no_selection.csv       Full table (no selection)
│   ├── results_spearman_top20.csv     Full table (Spearman selection)
│   ├── paper_tables/                  Paper tables (8 CSVs)
│   └── paper_figures/                 Paper figures (6 PNGs)
│
├── home_full_recording_npz/        Full recording NPZ (101 files, ~2.4 GB)
├── walking_bouts/                  Walking bout CSVs (186,012 files)
├── csv_raw2/                       Clinic 6MWT raw data
├── csv_preprocessed2/              Clinic preprocessed data
├── Accel files/                    Raw home GT3X files
├── SwayDemographics.xlsx           Demographics
├── POMS/                           Paper (LaTeX)
└── temp_exps/                      Scratch space for new experiments
```

---

## Important Notes

### Models
- **Clinic:** Ridge(α=5) is the best model. Non-linear models (RF, XGBoost, SVR, KNN) all worse.
- **Home:** Ridge(α=20) is the best model. Vote ensemble (R²=0.478) offers only marginal gain over Ridge (R²=0.452), not worth the complexity.
- **Fusion:** Early fusion (simple concatenation) beats late fusion, residual fusion, feature interactions, and modality-weighted fusion.

### Sampling Frequencies
- 91 subjects at 30 Hz, 8 at 60 Hz (M26, M27, M29, M31, M32, M43, M45, C75), 2 at 100 Hz (M48, M49)
- All processing uses hardcoded FS=30 (no resampling — gives best results)
- Sampling rate is in GT3X metadata (`info.txt` inside ZIP, field `Sample Rate`)

### Home Pipeline Design Decisions
- **Full recording (no daytime filter):** R²=0.452 vs 0.365 with daytime only
- **No quality filtering for PerBout:** keeping all bouts gives R²=0.452 vs 0.356 with quality filter
- **Axis-based preprocessing for PerBout:** R²=0.452 vs 0.431 with VM-based
- **VM-based for Gait/CWT/WalkSway:** removes orientation artifacts in short free-living bouts
- **Top-10 clean bouts ≥60s for Gait/CWT/WS:** longer bouts + quality filter needed for clinic-style features

### File Matching
- Match clinic files by subject key (e.g., `C61`), not full filename — `target_6mwd.csv` years don't match `csv_raw2/`/`csv_preprocessed2/` filenames

### No Data Leakage
- Feature selection (Spearman) done inside each LOO fold (training data only)
- StandardScaler fit on training data only
- Ridge α and model hyperparameters are fixed per feature set (no search on test data)
- Imputation uses column median (negligible leakage from test subject)
- Clinic and home pipelines are fully separate
