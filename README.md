# Gait Assessment in Pediatric-Onset Multiple Sclerosis Using Wearable Accelerometry

Predicting 6MWD from hip-worn accelerometer data collected during clinic 6-minute walk tests and home free-living monitoring in POMS.

**Subjects:** n=101 (POMS=38, Healthy=63), consistent across all analyses. Two subjects excluded: M22 (data quality) and M44 (too-short recording). Subject list in `feats/target_6mwd.csv`.

## Results Table

| Feature Set | #f | Clinic R² [95% CI] | Clinic MAE (m) [95% CI] | Clinic r [95% CI] | Home R² [95% CI] | Home MAE (m) [95% CI] | Home r [95% CI] |
|---|---|---|---|---|---|---|---|
| Gait | 11 | 0.68 [0.51, 0.78] | 42.7 [35.8, 50.1] | 0.83 [0.74, 0.89] | 0.15 [−0.10, 0.31] | 70.1 [58.5, 83.1] | 0.41 [0.18, 0.57] |
| CWT | 28 | 0.36 [0.19, 0.47] | 60.2 [50.1, 71.4] | 0.60 [0.49, 0.70] | 0.15 [−0.04, 0.31] | 67.9 [56.6, 80.8] | 0.40 [0.23, 0.57] |
| WalkSway | 12 | 0.40 [0.23, 0.57] | 54.2 [43.8, 65.7] | 0.63 [0.49, 0.77] | 0.06 [−0.20, 0.19] | 73.3 [61.1, 86.6] | 0.29 [0.12, 0.45] |
| Demo | 4 | 0.36 [0.21, 0.46] | 60.8 [50.9, 71.3] | 0.60 [0.49, 0.70] | 0.36 [0.21, 0.46] | 60.8 [50.9, 71.3] | 0.60 [0.49, 0.70] |
| Bout+Act-Top20 | 20 | 0.63 [0.49, 0.73] | 44.3 [36.5, 53.0] | 0.79 [0.72, 0.86] | 0.18 [0.01, 0.30] | 67.2 [55.9, 80.3] | 0.43 [0.28, 0.56] |
| Bout+Act-Top20+Demo | 24 | 0.69 [0.58, 0.77] | 39.6 [32.0, 48.2] | 0.83 [0.77, 0.88] | **0.45 [0.30, 0.55]** | **56.0 [46.8, 65.5]** | **0.67 [0.57, 0.76]** |
| **Gait+CWT+WS+Demo** | **55** | **0.81 [0.71, 0.88]** | **31.2 [25.5, 37.1]** | **0.90 [0.85, 0.94]** | 0.28 [0.04, 0.44] | 63.8 [52.8, 75.3] | 0.54 [0.39, 0.67] |

> **Footnote on `Bout+Act-Top20`.** The pool combines two feature families per setting: (1) **Bout features** — 20 per-segment gait features × 6 aggregation statistics (median, IQR, p10, p90, max, CV) = 120 columns, plus 2 bout-meta columns (total walk seconds, mean bout duration). Segments are 60 s windows of the 6MWT for clinic, and free-living walking bouts (≥10 s) for home. (2) **Act features** — 29 whole-recording activity features (no segmentation) computed from 1-s ENMO of the same recording. Spearman correlations are computed inside each LOO training fold; the top 20 are used to fit Ridge. Pool sizes: 151 (home and clinic both, after adding clinic activity features and deduplicating g_bout_dur_cv). Constant features (e.g., `act_daily_cv` for clinic, since 6MWT < 1 day) get correlation 0 and are never selected.

Reproduce: `python analysis/reproduce_results_table_best_models.py` (~1 min)

- **n=101**, LOO CV, no data leakage. All metrics in meters. Correlation column is Pearson **r** between predicted and true 6MWD (Spearman ρ available from scripts; ρ used only for feature selection inside folds). 95% CIs are percentile bootstrap over the LOO predictions (B=2000, seed=42).
- **Both clinic and home use Ridge regression.** Clinic α=5, Home α=20.
- **Demo-only row:** cohort_POMS, Age, Sex, BMI — same for clinic and home, Ridge α=20
- **Demo in combos:** Clinic uses Height, Home uses BMI (different best Demo per setting)
- **Clinic Gait/CWT/WS:** extracted from full 6MWT, no feature selection, Ridge α varies per set
- **Clinic Bout+Act:** 60s windows of 6MWT (gait+meta) + activity features from full 6MWT, Spearman Top-20 inside LOO, Ridge α=5
- **Clinic Gait+CWT+WS+Demo:** all 55 features, no selection, Ridge α=5
- **Home Bout+Act:** all walking bouts from full recording (gait+meta) + activity features from full multi-day recording, Spearman Top-20 inside LOO, Ridge α=20
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

### Step 2: Bout+Act Feature Extraction (151f)

```bash
python clinic/extract_perbout_features.py    # ~1 min
# Input:  csv_raw2/*.csv
# Output: feats/clinic_perbout_features.csv (101 x 152)
```

- Split 6MWT into 60s non-overlapping windows (trim first/last 10s)
- Extract 20 per-bout features per window (same features as home Bout+Act)
- Aggregate across windows: 6 stats (median, IQR, p10, p90, max, CV) = 120 gait + 2 meta
- Activity features (29) from second-by-second ENMO of trimmed 6MWT (no segmentation)
- Total: 120 gait + 2 meta + 29 activity = **151 features**

### Step 3: Clinic Prediction

```bash
python clinic/predict.py               # <1 sec (cached features)
# Input:  feats/clinic_gait_features.csv + clinic_cwt_features.csv
#         + clinic_walksway_features.csv + SwayDemographics.xlsx
# Output: R²=0.806, MAE=31.2m, r=0.898
```

Gait(11) + CWT(28) + WalkSway(12) + Demo(Height, 4) = 55 features, no feature selection, Ridge α=5.

### Clinic All Models Comparison

```bash
python clinic/predict_all_models.py    # <1 min
# Input:  feats/clinic_{gait,cwt,walksway}_features.csv + SwayDemographics.xlsx
# Output: R², MAE, r for Ridge, Lasso, ElasticNet, KNN, SVR, RF, XGBoost
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

### Step 2: Bout+Act Feature Extraction (151f)

```bash
python home/step2_extract_features.py      # ~12 min
# Input:  home_full_recording_npz/*.npz + feats/home_walking_bouts.pkl
# Output: feats/home_perbout_features.csv (101 x 152)
```

Per-bout preprocessing: gravity removal → Rodrigues rotation → PCA yaw alignment → AP, ML, VT + bandpass + ENMO. Extract 20 features per bout, aggregate with 6 stats (median, IQR, p10, p90, max, CV) = 120 gait + 2 meta + 29 activity = **151 features**.

#### 20 Per-Bout Gait Features

Each feature is extracted per walking bout, then aggregated across all bouts using 6 statistics (median, IQR, p10, p90, max, CV) → 20 × 6 = 120 columns.

| # | Feature | Description |
|---|---|---|
| 1 | `cadence_hz` | Dominant walking cadence from VT power spectrum (0.5–2.5 Hz) |
| 2 | `cadence_power` | Peak spectral power at cadence frequency |
| 3 | `acf_step_reg` | Step regularity — autocorrelation of VT at step lag |
| 4 | `hr_ap` | Harmonic ratio, anteroposterior axis |
| 5 | `hr_vt` | Harmonic ratio, vertical axis |
| 6 | `hr_ml` | Harmonic ratio, mediolateral axis |
| 7 | `stride_time_mean` | Mean stride interval (seconds) |
| 8 | `stride_time_std` | Stride interval standard deviation |
| 9 | `stride_time_cv` | Stride interval coefficient of variation |
| 10 | `ap_rms` | RMS acceleration, anteroposterior |
| 11 | `ml_rms` | RMS acceleration, mediolateral |
| 12 | `vt_rms` | RMS acceleration, vertical |
| 13 | `enmo_mean` | Mean ENMO (Euclidean Norm Minus One) |
| 14 | `enmo_p95` | 95th percentile ENMO |
| 15 | `vm_std` | Standard deviation of dynamic vector magnitude |
| 16 | `vt_range` | Range (peak-to-peak) of vertical acceleration |
| 17 | `ml_range` | Range (peak-to-peak) of mediolateral acceleration |
| 18 | `jerk_mean` | Mean jerk — rate of change of vector magnitude |
| 19 | `signal_energy` | Mean squared dynamic vector magnitude |
| 20 | `duration_sec` | Bout duration in seconds |

#### 2 Meta Features

| # | Feature | Description |
|---|---|---|
| 1 | `g_total_walk_sec` | Total walking duration across all bouts (seconds) |
| 2 | `g_mean_bout_dur` | Mean bout duration (seconds) |

(Note: `g_bout_dur_cv` was removed from the pool because it is mathematically identical to `g_duration_sec_cv` — the CV aggregation of the per-bout `duration_sec` feature. VIF = 10¹² confirmed the exact collinearity.)

#### 29 Activity Features (whole recording)

Computed from second-by-second ENMO of the full multi-day recording (no bout segmentation).

| # | Feature | Description |
|---|---|---|
| 1 | `act_enmo_mean` | Mean ENMO (1-sec epochs) |
| 2 | `act_enmo_std` | Standard deviation of ENMO |
| 3 | `act_enmo_median` | Median ENMO |
| 4 | `act_enmo_p5` | 5th percentile ENMO |
| 5 | `act_enmo_p25` | 25th percentile ENMO |
| 6 | `act_enmo_p75` | 75th percentile ENMO |
| 7 | `act_enmo_p95` | 95th percentile ENMO |
| 8 | `act_enmo_iqr` | Interquartile range of ENMO |
| 9 | `act_enmo_skew` | Skewness of ENMO distribution |
| 10 | `act_enmo_kurtosis` | Kurtosis of ENMO distribution |
| 11 | `act_enmo_entropy` | Entropy of ENMO histogram (20 bins) |
| 12 | `act_pct_sedentary` | Fraction of time sedentary (ENMO < 0.02g) |
| 13 | `act_pct_light` | Fraction of time in light activity (0.02–0.06g) |
| 14 | `act_pct_moderate` | Fraction of time in moderate activity (0.06–0.1g) |
| 15 | `act_pct_vigorous` | Fraction of time in vigorous activity (≥ 0.1g) |
| 16 | `act_mvpa_min_per_hr` | MVPA minutes per hour (ENMO ≥ 0.06g) |
| 17 | `act_n_bouts` | Number of active bouts (≥ 5s, ENMO ≥ 0.02g) |
| 18 | `act_bouts_per_hr` | Active bouts per hour |
| 19 | `act_bout_mean_dur` | Mean active bout duration (seconds) |
| 20 | `act_bout_dur_cv` | CV of active bout durations |
| 21 | `act_longest_bout` | Longest active bout duration (seconds) |
| 22 | `act_astp` | Active-to-sedentary transition probability |
| 23 | `act_satp` | Sedentary-to-active transition probability |
| 24 | `act_fragmentation` | Fragmentation index (ASTP + SATP) |
| 25 | `act_early_enmo` | Mean ENMO in first third of recording |
| 26 | `act_mid_enmo` | Mean ENMO in middle third of recording |
| 27 | `act_late_enmo` | Mean ENMO in last third of recording |
| 28 | `act_early_late_ratio` | Ratio of early-to-late ENMO |
| 29 | `act_daily_cv` | Day-to-day CV of mean ENMO |

#### Bout+Act-Top20: Spearman-Selected Features

The 20 accelerometer features selected by Spearman correlation inside each LOO fold (after the `g_bout_dur_cv` dedup). 15 features are selected in all 101 folds; the remaining 5 slots vary slightly across folds.

| # | Feature | Folds Selected | Spearman ρ |
|---|---|---|---|
| 1 | `g_duration_sec_max` | 101/101 | +0.375 |
| 2 | `act_pct_vigorous` | 101/101 | +0.370 |
| 3 | `g_duration_sec_cv` | 101/101 | +0.367 |
| 4 | `act_enmo_p95` | 101/101 | +0.366 |
| 5 | `g_ap_rms_med` | 101/101 | +0.355 |
| 6 | `g_enmo_mean_p10` | 101/101 | +0.350 |
| 7 | `g_ap_rms_cv` | 101/101 | −0.347 |
| 8 | `g_jerk_mean_med` | 101/101 | +0.345 |
| 9 | `g_acf_step_reg_max` | 101/101 | +0.344 |
| 10 | `g_ml_rms_cv` | 101/101 | −0.342 |
| 11 | `g_acf_step_reg_p90` | 101/101 | +0.337 |
| 12 | `g_vm_std_med` | 101/101 | +0.333 |
| 13 | `g_vm_std_cv` | 101/101 | −0.333 |
| 14 | `g_enmo_p95_med` | 101/101 | +0.330 |
| 15 | `g_ml_range_med` | 101/101 | +0.328 |
| 16 | `g_enmo_mean_med` | 100/101 | +0.326 |
| 17 | `g_signal_energy_med` | 100/101 | +0.324 |
| 18 | `g_ml_range_cv` | 92/101 | −0.325 |
| 19 | `g_vt_range_med` | 82/101 | +0.319 |
| 20 | `g_ml_rms_med` | 79/101 | +0.319 |

These 20 features + Demo(4) = 24 features → Ridge(α=20) → R²=0.452.

### Step 3: Home Prediction

```bash
python home/step3_predict.py               # <1 sec (cached features)
# Input:  feats/home_perbout_features.csv + SwayDemographics.xlsx
# Output: R²=0.452, MAE=56.0m, r=0.672 (Ridge only baseline)
```

Spearman Top-20 inside LOO + Demo(4), Ridge α=20.

### Home All Models Comparison + Voting Ensemble

```bash
python home/step3_predict_all_models.py    # <30 sec (cached features)
# Input:  feats/home_perbout_features.csv + SwayDemographics.xlsx
# Output: R², MAE, r for Ridge, Lasso, ElasticNet, KNN, SVR, RF, XGBoost + voting ensembles
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
python home/step2_extract_features.py                      # Bout+Act features (~12 min)
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

### POMS/tables/ — paper-side tables

Every table below is auto-mirrored from `results/paper_tables/` (or `results/`) into `POMS/tables/` so the LaTeX build always reads the up-to-date version.

| File (in `POMS/tables/`) | Generating script | Description |
|---|---|---|
| `results_table_final.csv` | `analysis/reproduce_results_table_best_models.py` | 7-row results table with 95% bootstrap CIs (clinic + home) |
| `model_comparison.csv` | `analysis/model_comparison_table.py` | 7 models × clinic/home, R²/MAE/r with CIs, worst→best |
| `demographics_table.csv` | `analysis/generate_paper_tables.py` | Cohort demographics + clinical scores |
| `feature_descriptions.csv` | `analysis/generate_paper_tables.py` | Feature categories and names |
| `best_predictions.csv` | `analysis/generate_paper_tables.py` | Per-subject LOO predictions (best models) |
| `error_analysis_by_cohort.csv` | `analysis/generate_paper_tables.py` | Error metrics by cohort (POMS/Healthy) |
| `feature_correlations.csv` | `analysis/generate_paper_tables.py` | Spearman ρ with 6MWD by setting/cohort |
| `ms_vs_healthy_features.csv` | `analysis/generate_paper_tables.py` | POMS vs Healthy differences (Cohen's d, BH-corrected) |
| `clinical_corr_ms_only.csv` | `analysis/generate_paper_tables.py` | Clinic feature–clinical score correlations (MS only) |
| `clinical_corr_ms_home.csv` | `analysis/generate_paper_tables.py` | Home feature–clinical score correlations (MS only) |

```bash
python analysis/reproduce_results_table_best_models.py  # results_table_final.csv (~1 min)
python analysis/model_comparison_table.py               # model_comparison.csv (<1 min)
python analysis/generate_paper_tables.py                # 8 tables (~70 sec)
```

### POMS/figures/ — paper-side figures

| File (in `POMS/figures/`) | Generating script | Description |
|---|---|---|
| `heatmap_feature_6mwd_corr_top10.png` | `analysis/generate_feature_6mwd_heatmaps.py` | Top-10 features × {All, POMS, Healthy}, clinic + home side-by-side (Spearman ρ) |
| `heatmap_clinical_corr_clinic.png` (Fig 3) | `analysis/generate_clinical_corr_heatmaps.py` | Top-10 clinic features vs clinical scores (POMS only) |
| `heatmap_clinical_corr_home.png` (Fig 4) | `analysis/generate_clinical_corr_heatmaps.py` | Top-10 home features (Bout+Act) vs clinical scores (POMS only) |
| `bout_distribution_overview.png` | `analysis/generate_bout_distribution_figure.py` | 3-panel: violin+jitter cohort comparison / joint scatter / pooled survival |
| `bout_duration_boxplots_per_subject.png` | `analysis/generate_bout_distribution_figure.py` | Per-subject box plots (POMS top, Healthy bottom), sorted by median — paper Supplementary S2 |
| `fig_supp_wear_median_bout.png` | `analysis/generate_supp_wear_median_bout.py` | 2-panel: (a) 7-day device wear-time violin + MW-U test, (b) scatter of bout count × median bout duration sized by 6MWD — paper Supplementary S1 |
| `heatmap_clinic_home_feature_corr.png` | `analysis/generate_clinic_home_corr_heatmap.py` | 5×5 clinic↔home Gait feature correlation — All/POMS/Healthy panels; shows p90 home column as the densest cluster (6MWT ≈ near-peak of free-living) |
| `fig_model_comparison_mae.png` | `analysis/generate_model_comparison_barchart.py` | Grouped bar chart of MAE (m) per model, Clinic (blue) vs Home (red), error bars = σ-equivalent of 2000-bootstrap 95% CI |
| `fig_overview_diagram.png` | *(manual — Figma/PPT export, not regenerated by any script)* | Study pipeline overview |

```bash
python analysis/generate_feature_6mwd_heatmaps.py      # Fig 2 (6MWD heatmap)
python analysis/generate_clinical_corr_heatmaps.py     # Figs 3 & 4 (clinical scores)
python analysis/generate_bout_distribution_figure.py   # bout overview + per-subject boxplots
python analysis/generate_supp_wear_median_bout.py      # supplementary wear-time + median-bout scatter
python analysis/generate_clinic_home_corr_heatmap.py   # 5×5 clinic↔home correlation heatmap
python analysis/generate_model_comparison_barchart.py  # MAE bar chart per model
```

`analysis/generate_paper_figures.py` (legacy) still emits `fig_predicted_vs_actual.png`, `fig_bland_altman.png`, `fig_shap_importance.png`, and the full-feature `heatmap_feature_6mwd_corr.png` / `heatmap_clinical_corr_*.png` into `results/paper_figures/`. Those older variants are superseded for the paper by the dedicated scripts listed above but are kept for reference.

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
| `feats/clinic_perbout_features.csv` | `clinic/extract_perbout_features.py` | 151 + key | Clinic Bout+Act (60s windows + activity) |
| `feats/home_walking_bouts.pkl` | `home/step1_detect_walking_bouts.py` | — | Walking bout indices per subject |
| `feats/home_perbout_features.csv` | `home/step2_extract_features.py` | 151 + key | Home Bout+Act (all bouts + activity) |
| `feats/home_gait_features.csv` | `home/extract_gait_cwt_ws_features.py` | 66 + key | Home Gait (VM, Top-10 clean bouts) |
| `feats/home_cwt_features.csv` | `home/extract_gait_cwt_ws_features.py` | 168 + key | Home CWT (VM, Top-10 clean bouts) |
| `feats/home_walksway_features.csv` | `home/extract_gait_cwt_ws_features.py` | 72 + key | Home WalkSway (VM, Top-10 clean bouts) |

---

## Key Scripts

| Script | Purpose |
|---|---|
| `home/step0_gt3x_to_npz.py` | GT3X → full recording NPZ (no filtering) |
| `home/step1_detect_walking_bouts.py` | Walking bout detection + optional CSV saving |
| `home/step2_extract_features.py` | Home Bout+Act feature extraction (151f) |
| `home/step3_predict.py` | Home Ridge-only prediction (R²=0.452, baseline) |
| `home/step3_predict_all_models.py` | Home all models comparison (Ridge best, R²=0.452) |
| `home/extract_gait_cwt_ws_features.py` | Home Gait/CWT/WalkSway features (VM-based) |
| `home/reproduce_from_bouts.py` | Reproduce from saved bout CSVs |
| `clinic/predict.py` | Clinic Ridge prediction (R²=0.806) |
| `clinic/predict_all_models.py` | Clinic all models comparison |
| `clinic/reproduce_c2.py` | Clinic preprocessing + Gait/CWT extraction functions |
| `clinic/extract_walking_sway.py` | Clinic WalkSway extraction function |
| `clinic/extract_gait_cwt_ws_features.py` | Clinic Gait/CWT/WalkSway feature extraction |
| `clinic/extract_perbout_features.py` | Clinic Bout+Act feature extraction (60s windows + activity) |
| `analysis/reproduce_results_table_best_models.py` | Results table with best models (~1 min) |
| `analysis/results_table_full.py` | Full combination tables (~13 min) |
| `analysis/generate_paper_tables.py` | Paper tables → `POMS/tables/` (8 CSVs, ~70 sec) |
| `analysis/reproduce_results_table_best_models.py` | `results_table_final.csv` with bootstrap CIs → `POMS/tables/` |
| `analysis/model_comparison_table.py` | `model_comparison.csv` (7 models × clinic/home) → `POMS/tables/` |
| `analysis/generate_feature_6mwd_heatmaps.py` | `heatmap_feature_6mwd_corr_top10.png` → `POMS/figures/` |
| `analysis/generate_clinical_corr_heatmaps.py` | Figs 3 & 4 (clinic/home × clinical scores) → `POMS/figures/` |
| `analysis/generate_bout_distribution_figure.py` | 2 home-bout figures (overview + per-subject boxplots) → `POMS/figures/` |
| `analysis/generate_supp_wear_median_bout.py` | Supplementary figure (wear-time + median-bout scatter) → `POMS/figures/` |
| `analysis/generate_clinic_home_corr_heatmap.py` | `heatmap_clinic_home_feature_corr.png` (5×5, All/POMS/Healthy) → `POMS/figures/` |
| `analysis/generate_model_comparison_barchart.py` | `fig_model_comparison_mae.png` (MAE bar chart, 7 models × clinic/home) → `POMS/figures/` |
| `analysis/generate_paper_figures.py` | Legacy paper figures → `results/paper_figures/` only |

---

## Directory Layout

```
6mw/
├── home/                           HOME PIPELINE
│   ├── step0_gt3x_to_npz.py         GT3X → NPZ
│   ├── step1_detect_walking_bouts.py Bout detection [--save-csv]
│   ├── step2_extract_features.py     Bout+Act features (151f)
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
│   └── extract_perbout_features.py   Bout+Act features (60s windows + activity)
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
│   ├── home_perbout_features.csv     Home Bout+Act (151f)
│   ├── home_gait_features.csv        Home Gait (66f)
│   ├── home_cwt_features.csv         Home CWT (168f)
│   ├── home_walksway_features.csv    Home WalkSway (72f)
│   ├── clinic_gait_features.csv      Clinic Gait (11f)
│   ├── clinic_cwt_features.csv       Clinic CWT (28f)
│   ├── clinic_walksway_features.csv  Clinic WalkSway (12f)
│   └── clinic_perbout_features.csv   Clinic Bout+Act (151f)
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
- **No quality filtering for Bout+Act:** keeping all bouts gives R²=0.452 vs 0.356 with quality filter
- **Axis-based preprocessing for Bout+Act:** R²=0.452 vs 0.431 with VM-based
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
