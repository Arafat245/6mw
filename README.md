# Gait Assessment in Pediatric-Onset Multiple Sclerosis Using Wearable Accelerometry

Predicting 6MWD from hip-worn accelerometer data collected during clinic 6-minute walk tests and home free-living monitoring in POMS.

**Subjects:** n=101 (POMS=38, Healthy=63), consistent across all analyses.

## Best Results (Current)

| Model | R² | MAE (ft) | r | ρ |
|---|---|---|---|---|
| **Clinic:** Gait+CWT+WalkSway+Demo (55f), Ridge α=10 | **0.806** | 100 | 0.898 | 0.889 |
| **Home (with PLS):** PLS(Gait)+Demo(5) (7f), Ridge α=20 | **0.519** | 172 | 0.720 | 0.703 |
| **Home (no PLS):** Gait+Demo(5) (16f), Ridge α=50 | **0.488** | 175 | 0.700 | 0.692 |

## Folder Summary

| Folder | Contents | Count |
|---|---|---|
| `clinic/` | Clinic pipeline scripts | 2 scripts |
| `home/` | Home pipeline scripts | 4 scripts |
| `analysis/` | Analysis & results scripts | 3 scripts |
| `POMS/` | Paper (LaTeX, figures, tables) | — |
| `feats/` | All extracted features & outputs | — |
| `references/` | PDF reference papers | 15 PDFs |
| `notebooks/` | Jupyter notebooks (exploratory/legacy) | 12 notebooks |
| `old_figures/` | Old figures from early analysis | 18 PNGs |
| `data/misc/` | Misc CSV/Excel/data files | ~15 files |
| `data/legacy/` | Old data directories (csv_raw, csv_preprocessed) | 3 dirs |
| `archive/` | Old experimental scripts (not in final pipeline) | ~20 scripts |
| `temporary_experiments/` | Scratch space for new experiments before finalizing | — |

## Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn openpyxl pywt xgboost shap statsmodels pygt3x
# Foundation models (optional):
pip install momentfm torch
```

## Project Structure

```
6mw/
├── README.md
│
├── clinic/                         CLINIC PIPELINE SCRIPTS
│   ├── reproduce_c2.py               Preprocessing + Gait/CWT extraction
│   └── extract_walking_sway.py       WalkSway feature extraction
│
├── home/                           HOME PIPELINE SCRIPTS
│   ├── home_hybrid_models_v2.py      Home gait extraction (PRIMARY)
│   ├── preprocess_raw.py             GT3X → daytime → walking_segments
│   ├── extract_walking_sway.py       Home WalkSway features
│   └── extract_agd_features.py       AGD epoch features
│
├── analysis/                       ANALYSIS & RESULTS SCRIPTS
│   ├── results_table_final.py        Main results table
│   ├── reproduce_best_results.py     Quick validation
│   └── home_clinic_preproc.py        Preprocessing comparison
│
├── POMS/                           PAPER
│   ├── main.tex                      LaTeX source
│   ├── references.bib                Bibliography
│   ├── figures/                      Paper figures
│   └── tables/                       Paper tables
│
├── feats/                          EXTRACTED FEATURES & OUTPUTS
│   ├── target_6mwd.csv               Ground truth (cohort, subj_id, year, sixmwd)
│   ├── home_hybrid_v2_features.npz   Home features (X_gait, X_act, X_cent, X_cwt)
│   ├── best_predictions.csv          LOO predictions from best models
│   ├── results_table_final.csv       Main results table
│   └── *.csv, *.png                  All tables and figures
│
├── csv_raw2/                       CLINIC DATA — raw 6MWT (Timestamp, X, Y, Z)
├── csv_preprocessed2/              CLINIC DATA — preprocessed (AP, ML, VT, _bp, ENMO)
├── csv_home_daytime/               HOME DATA — daytime accelerometer (X, Y, Z)
├── results_raw_pipeline/           HOME DATA — walking segments + embeddings
│   ├── walking_segments/             Walking bouts (AP, ML, VT)
│   └── emb_*.npy                     Foundation model embeddings
├── Accel files/                    RAW DATA — GT3X files + AGD + wear diary
├── SwayDemographics.xlsx           DEMOGRAPHICS — clinical scores
│
├── references/                     REFERENCE PAPERS (PDFs)
├── notebooks/                      JUPYTER NOTEBOOKS (exploratory/legacy)
├── old_figures/                    OLD FIGURES (from early analysis)
├── data/
│   ├── misc/                         Misc CSV/Excel/data files
│   └── legacy/                       Old data directories (csv_raw, csv_preprocessed)
├── limubert_repo/                  LIMUBERT MODEL CODE
└── archive/                        OLD SCRIPTS (experimental, not in final pipeline)
```

---

## CLINIC PIPELINE

### Step C0: GT3X → Raw CSV (already done)

Clinic 6MWT recordings were extracted from GT3X files to `csv_raw2/` with columns: Timestamp, X, Y, Z.

### Step C1: Preprocessing

**Script:** `reproduce_c2.py` → function `preprocess_file()`
**Input:** `csv_raw2/*.csv`
**Output:** `csv_preprocessed2/*.csv`

```bash
python clinic/reproduce_c2.py
```

1. Get sampling rate from timestamps (~30 Hz)
2. Trim first/last 10 seconds (transition artifacts)
3. Resample to uniform 30 Hz
4. Gravity removal: 0.25 Hz 4th-order Butterworth lowpass → subtract
5. Rodrigues rotation: align gravity vector with Z-axis
6. PCA yaw alignment: PCA on horizontal plane → AP, ML, VT
7. Bandpass filter: 0.25–2.5 Hz → AP_bp, ML_bp, VT_bp
8. Compute VM_dyn, VM_raw, ENMO

**Output columns:** AP, ML, VT, AP_bp, ML_bp, VT_bp, VM_dyn, VM_raw, ENMO + metadata

### Step C2: Clinic Feature Extraction

**Gait (11 features)** — from `csv_preprocessed2/`:
- Script: `reproduce_c2.py` → `extract_gait10()` + `compute_vt_rms()` + `add_sway_ratios()`
- Features: cadence_hz, step_time_cv_pct, acf_step_regularity, hr_ap, hr_vt, ml_rms_g, ml_spectral_entropy, jerk_mean_abs_gps, enmo_mean_g, cadence_slope_per_min, vt_rms_g

**CWT (28 features)** — from `csv_raw2/`:
- Script: `reproduce_c2.py` → `extract_cwt()`
- Morlet wavelet on VM signal, 6 temporal segments, aggregated as mean/std/slope

**WalkSway (12 features)** — from `csv_preprocessed2/`:
- Script: `extract_walking_sway.py` → `extract_walking_sway()`
- 10 ENMO-normalized sway features + ml_over_enmo + ml_over_vt

**Demo (4 features):** cohort_POMS, Age, Sex, Height from `SwayDemographics.xlsx`

### Step C3: Clinic Prediction

**Script:** `results_table_final.py`
**Model:** Ridge α=10, LOO CV, n=101
**Features:** Gait(11) + CWT(28) + WalkSway(12) + Demo(4) = **55 features**

```
Best Clinic Result:
  All:     R²=0.8055, MAE=100.1 ft, r=0.898, ρ=0.889
  POMS:    R²=0.874, MAE=93.2 ft
  Healthy: R²=0.637, MAE=104.2 ft
```

**To reproduce:**
```bash
python analysis/results_table_final.py  # row: Gait+CWT+WalkSway+Demo
```

---

## HOME PIPELINE

### Step H0: GT3X → Home Daytime CSV (already done)

**Script:** `preprocess_raw.py` → `load_gt3x()`, `extract_daytime()`
**Input:** `Accel files/*/*.gt3x`
**Output:** `csv_home_daytime/*.csv` (X, Y, Z, ~4M samples per subject)

```bash
python home/preprocess_raw.py
```

1. Read GT3X binary → timestamps + X, Y, Z at 30 Hz
2. Extract daytime (7 AM – 10 PM) with worn-time detection (rolling std > 0.01)

### Step H1: Home Gait Feature Extraction (PRIMARY pipeline)

**IMPORTANT:** Home gait features are created by `home_hybrid_models_v2.py`, NOT `preprocess_raw.py`.

**Script:** `home_hybrid_models_v2.py`
**Input:** `csv_home_daytime/*.csv` + `csv_raw2/*.csv` (clinic data as reference)
**Output:** `feats/home_hybrid_v2_features.npz`

```bash
python home/home_hybrid_models_v2.py
# Then save: np.savez('feats/home_hybrid_v2_features.npz', X_gait=X_gait, X_act=X_act, X_cent=X_cent, X_cwt=X_cwt, X_demo=X_demo, y=y)
```

Steps:

1. **`detect_active_bouts(xyz, fs, min_bout_sec=30)`**
   - ENMO per second ≥ 0.015 → active
   - Merge consecutive active seconds into bouts (min 30 sec)

2. **`refine_with_hr(xyz, fs, bouts, hr_threshold=0.2)`**
   - Bandpass VM at 0.5–3.0 Hz
   - 10-sec windows: FFT → harmonic ratio (even/odd harmonics)
   - Keep windows with HR ≥ 0.2 (periodic walking confirmed)
   - Merge into refined bouts (min 30 sec)

3. **`select_walking_segment(xyz, fs, bouts, target_sec=360, clinic_xyz, clinic_fs)`**
   - Compute walking signature per bout: [mean_ENMO, std_ENMO, cadence, step_regularity, ...]
   - Compute clinic 6MWT walking signature (loaded from `csv_raw2/`)
   - Rank bouts by **cosine similarity to clinic** signature
   - Select most clinic-like bouts up to **360 seconds** (6 minutes)
   - **Uses clinic data as reference for bout selection**

4. **`preprocess_walking(walking_xyz, fs)`**
   - Gravity removal (0.25 Hz lowpass)
   - **Rodrigues rotation** (same as clinic)
   - PCA yaw → AP, ML, VT
   - Bandpass for _bp variants + ENMO

5. **`extract_gait13(preprocessed_df)`**
   - 13 features: cadence_hz, step_time_cv_pct, acf_step_regularity, hr_ap, hr_vt, ml_rms_g, ml_spectral_entropy, jerk_mean_abs_gps, enmo_mean_g, cadence_slope_per_min, vt_rms_g, ml_over_enmo, ml_over_vt

**Output NPZ contents:**

| Key | Shape | Description |
|---|---|---|
| X_gait | (102, 13) | 11 gait + 2 sway ratios |
| X_act | (102, 15) | Activity profile features |
| X_cent | (102, 18) | 6-minute activity centile features |
| X_cwt | (102, 28) | CWT features |
| X_demo | (102, 4) | Demographics |
| y | (102,) | 6MWD targets |

**Verified:** Reproduced features correlate r=1.0000 with cached npz.

### Step H1b: Alternative Home Walking Segments (SECONDARY pipeline)

**Script:** `preprocess_raw.py`
**Output:** `results_raw_pipeline/walking_segments/*.csv`

Different from H1:
- Detection: RMS + std + autocorrelation (min 20 sec)
- Preprocessing: gravity projection (not Rodrigues), no bandpass
- No clinic-informed bout selection
- **Used only for WalkSway features**, NOT for main gait features

### Step H2: Home WalkSway Features

**Script:** `extract_walking_sway.py`
**Input:** `results_raw_pipeline/walking_segments/*.csv`
**Output:** `feats/home_walking_sway.csv`

```bash
python clinic/extract_walking_sway.py  # also: python home/extract_walking_sway.py
```

12 ENMO-normalized sway features (same function as clinic, different input data).

### Step H3: Home Prediction

**Best Home with PLS (R²=0.519):**
- Features: PLS(2 components from Gait11) + Demo(5) = 7 features
- Gait from: `home_hybrid_v2_features.npz` X_gait[:, :11]
- PLS target: clinic gait features from `reproduce_c2.py:extract_gait10()`
- Model: Ridge α=20, LOO CV
- **Requires paired clinic data** for PLS training

**Best Home without PLS (R²=0.488):**
- Features: Gait(11) + Demo(5) = 16 features
- Model: Ridge α=50, LOO CV
- **Note:** Home gait features still use clinic data as reference during bout selection (Step H1.3)

```
Best Home Result (PLS):
  All:     R²=0.519, MAE=172 ft, r=0.720, ρ=0.703
  POMS:    R²=0.405
  Healthy: R²=0.354

Best Home Result (no PLS):
  All:     R²=0.488, MAE=175 ft, r=0.700, ρ=0.692
  POMS:    R²=0.331
  Healthy: R²=0.346
```

**To reproduce:**
```bash
python analysis/results_table_final.py  # rows: PLS(Gait)+Demo(5), Gait+Demo(5) [no clinic]
```

---

## Outputs

### Tables

| # | File | Description |
|---|---|---|
| T1 | `feats/demographics_table.csv` | Demographics / cohort characteristics |
| T2 | `feats/results_table_final.csv` | Main results: R² by feature set |
| T3 | `feats/ms_vs_healthy_features.csv` | POMS vs Healthy comparison (Mann-Whitney, BH-corrected) |
| T4 | `feats/feature_descriptions.csv` | Feature & clinical score names by category |
| T5 | `feats/error_analysis_by_cohort.csv` | Error analysis by cohort |
| T6 | `feats/clinic_home_feature_correlation.csv` | Clinic-home feature correlations |
| T7 | `feats/literature_comparison.csv` | Literature comparison |

### Figures

| # | File | Description |
|---|---|---|
| F1 | `feats/fig_overview_diagram.svg` | Pipeline overview diagram |
| F2 | `feats/heatmap_feature_6mwd_corr.png` | Feature-6MWD correlations |
| F3 | `feats/heatmap_clinical_corr_clinic.png` | Clinic features vs clinical scores (POMS) |
| F4 | `feats/heatmap_clinical_corr_home.png` | Home features vs clinical scores (POMS) |
| F5 | `feats/fig_predicted_vs_actual.png` | Predicted vs Actual 6MWD |
| F6 | `feats/fig_bland_altman.png` | Bland-Altman agreement |
| F7 | `feats/fig_shap_importance.png` | SHAP feature importance |

---

## Feature Categories

### Gait (11 features)

| Feature | Description | Unit |
|---|---|---|
| cadence_hz | Walking cadence | Hz |
| step_time_cv_pct | Step time variability | % |
| acf_step_regularity | Autocorrelation step regularity | — |
| hr_ap | Harmonic ratio (AP) | — |
| hr_vt | Harmonic ratio (VT) | — |
| ml_rms_g | Mediolateral RMS | g |
| ml_spectral_entropy | ML spectral entropy | — |
| jerk_mean_abs_gps | Mean absolute jerk | g/s |
| enmo_mean_g | ENMO (intensity) | g |
| cadence_slope_per_min | Cadence fatigue trend | Hz/min |
| vt_rms_g | Vertical RMS | g |

### Walking Sway (12 features)
ENMO-normalized. Higher = more instability.

ml_range_norm, ml_path_length_norm, ml_jerk_rms_norm, ap_rms_norm, ap_range_norm, sway_ellipse_norm, ml_velocity_rms_norm, stride_ml_cv, ml_ap_ratio, hr_ml, ml_over_enmo, ml_over_vt

### CWT (28 features)
Continuous wavelet transform: mean_energy, high_freq_energy, dominant_freq, estimated_cadence, max_power_freq, freq_variability, freq_cv, wavelet_entropy, fundamental_freq, harmonic_ratio — each mean/std + temporal slopes.

### Demographics
- **Demo(3):** cohort_POMS, Age, Sex
- **Demo(5):** cohort_POMS, Age, Sex, Height, BMI

---

## Key Scripts

| Script | Purpose |
|---|---|
| `clinic/reproduce_c2.py` | Clinic preprocessing + Gait/CWT extraction |
| `clinic/extract_walking_sway.py` | Clinic WalkSway features |
| `home/home_hybrid_models_v2.py` | **Home gait feature extraction** (creates home_hybrid_v2_features.npz) |
| `home/preprocess_raw.py` | GT3X → daytime CSV → walking_segments |
| `home/extract_walking_sway.py` | Home WalkSway features |
| `home/extract_agd_features.py` | AGD epoch features |
| `analysis/results_table_final.py` | Main results table (all feature combinations) |

---

## Important Notes

### Two Home Pipelines
1. **`home/home_hybrid_models_v2.py`** (PRIMARY): Clinic-informed bout selection, Rodrigues rotation, bandpass → `home_hybrid_v2_features.npz` → **all home gait features**
2. **`home/preprocess_raw.py`** (SECONDARY): Simpler detection, gravity projection, no bandpass → `walking_segments/` → **WalkSway features only**

### Clinic-Informed Home Features
Home gait features use clinic data as reference during bout selection (`select_walking_segment()` cosine similarity). Even the "no PLS" model implicitly uses clinic data during feature extraction.

### Archive
Old experimental scripts (exp1-exp11, run_all_models, predict_6mwd_*, etc.) are in `archive/`. These were used during development but are NOT part of the final pipeline.

### Clinic-Home Date Gaps
64% of subjects had clinic and home within 3 days. 14 subjects had >30 days gap (max 559 days).

## Excluded Subjects
- **M22:** Data quality issues
- **M44:** Too-short clinic recording (601 samples)
- All analyses: n=101

---

## Results History

| Date | Change | Clinic R² | Home R² (PLS) | Home R² (no PLS) |
|---|---|---|---|---|
| Initial | Gait13+CWT+Demo, PLS(Gait)+Demo(3) | 0.792 | 0.483 | 0.428 |
| +WalkSway | Gait11+CWT+WalkSway+Demo | 0.806 | 0.483 | 0.428 |
| +Demo(5)+α | Added Height+BMI, tuned α | 0.806 | **0.519** | **0.488** |
