# Gait Assessment in Pediatric-Onset Multiple Sclerosis Using Wearable Accelerometry

Predicting 6MWD from hip-worn accelerometer data collected during clinic 6-minute walk tests and home free-living monitoring. Compares handcrafted gait/sway/wavelet features, foundation model embeddings, and PLS-based home-to-clinic domain adaptation.

**Subjects:** n=101 (POMS=38, Healthy=63), consistent across all analyses.

## Best Results (Current)

| Model | R² | MAE (ft) | Needs Clinic? |
|---|---|---|---|
| **Clinic:** Gait+CWT+WalkSway+Demo (55f) | **0.806** | 100 | N/A |
| **Home (with clinic):** PLS(Gait)+Demo(5) (7f) | **0.519** | 172 | Yes (PLS training) |
| **Home (no clinic):** Gait+Demo(5), Ridge α=50 (16f) | **0.488** | 175 | No |

## Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn openpyxl pywt xgboost shap statsmodels pygt3x
# Foundation models (optional, only needed for MOMENT/LimuBERT rows):
pip install momentfm torch
```

## Data Layout

```
Accel files/                        Raw GT3X files + AGD epoch data + wear diary
  C01_OPT/
    *.gt3x                          Raw tri-axial acceleration (30 Hz, binary)
    *60sec.agd                      SQLite: 60-sec epoch data (activity counts, steps, inclinometer)
  HomeAccelerometer_ontimes.xlsx    Self-reported wear diary (65 POMS subjects)
csv_raw2/                           Clinic 6MWT recordings (Timestamp, X, Y, Z) — extracted from GT3X
csv_preprocessed2/                  Preprocessed clinic data (AP, ML, VT axes)
csv_home_daytime/                   Home daytime accelerometer data (X, Y, Z, fs=30)
results_raw_pipeline/
  walking_segments/                 Home walking bouts (preprocessed AP, ML, VT) — from preprocess_raw.py
  emb_limubert_clinic.npy           LimuBERT embeddings (clinic)
  emb_limubert_home.npy             LimuBERT embeddings (home)
SwayDemographics.xlsx               Demographics and clinical scores
feats/
  target_6mwd.csv                   Ground truth (cohort, subj_id, year, sixmwd)
  home_hybrid_v2_features.npz       Home features (X_gait:13, X_act:15, X_cent:18, X_cwt:28, X_demo:4, y)
  home_cwt_hybrid.csv               Home CWT features (28 features)
  home_agd_features.csv             Home AGD epoch features (22 features)
  home_walking_bout_indices.npz     Cached walking bout boundaries
  moment_clinic_raw.npy             MOMENT embeddings (clinic)
  moment_home_raw.npy               MOMENT embeddings (home)
  best_predictions.csv              LOO predictions from best models
  results_table_final.csv           Main results table
```

## Full Pipeline (Step by Step)

### Phase 0: GT3X → Raw CSV (already done)

**Input:** `Accel files/*/\*.gt3x` (binary accelerometer files from ActiGraph GT3X)
**Script:** `preprocess_raw.py` (functions: `load_gt3x()`, `extract_daytime()`)

```bash
python preprocess_raw.py
```

1. Read GT3X binary → extract timestamps + X, Y, Z acceleration at 30 Hz
2. Extract daytime segments (7 AM – 10 PM) with worn-time detection (rolling std > 0.01)
3. **Output:** `csv_home_daytime/*.csv` (X, Y, Z columns, ~4M samples per subject)

**Note:** Clinic raw data in `csv_raw2/` was extracted separately (Timestamp, X, Y, Z from 6MWT recordings).

### Phase 1: Clinic Preprocessing

**Input:** `csv_raw2/*.csv`
**Script:** `reproduce_c2.py` (function: `preprocess_file()`)

```bash
python reproduce_c2.py
```

Steps:
1. Get sampling rate from timestamps (~30 Hz)
2. Trim first/last 10 seconds (transition artifacts)
3. Resample to uniform 30 Hz
4. Gravity removal: 0.25 Hz 4th-order Butterworth lowpass → subtract gravity estimate
5. Rodrigues rotation: align gravity vector with Z-axis
6. PCA yaw alignment: PCA on horizontal plane → AP (anteroposterior), ML (mediolateral), VT (vertical)
7. Bandpass filter: 0.25–2.5 Hz on AP, ML, VT → AP_bp, ML_bp, VT_bp
8. Compute VM_dyn, VM_raw, ENMO

**Output:** `csv_preprocessed2/*.csv` (18 columns: AP, ML, VT, AP_bp, ML_bp, VT_bp, VM_dyn, VM_raw, ENMO, metadata)

### Phase 2: Home Walking Detection and Feature Extraction

**IMPORTANT:** The home gait features in `home_hybrid_v2_features.npz` are created by `home_hybrid_models_v2.py`, NOT by `preprocess_raw.py`. The pipelines differ significantly.

**Input:** `csv_home_daytime/*.csv` + `csv_raw2/*.csv` (clinic data used as reference)
**Script:** `home_hybrid_models_v2.py`

```bash
python home_hybrid_models_v2.py
```

Steps for home gait feature extraction:

1. **`detect_active_bouts(xyz, fs, min_bout_sec=30)`**
   - Compute ENMO per second
   - Mark seconds with ENMO ≥ 0.015 as active
   - Merge consecutive active seconds into bouts (minimum 30 seconds)

2. **`refine_with_hr(xyz, fs, bouts, hr_threshold=0.2)`**
   - Bandpass filter VM at 0.5–3.0 Hz
   - In 10-second windows within each bout, compute FFT
   - Calculate harmonic ratio (even/odd harmonics) at detected cadence
   - Keep only windows with HR ≥ 0.2 (confirms periodic walking)
   - Merge passing windows into refined bouts (minimum 30 seconds)

3. **`select_walking_segment(xyz, fs, bouts, target_sec=360, clinic_xyz, clinic_fs)`**
   - Compute walking signature for each bout: [mean_ENMO, std_ENMO, cadence, step_regularity, ...]
   - Compute walking signature for clinic 6MWT data (loaded from `csv_raw2/`)
   - Rank home bouts by cosine similarity to clinic signature
   - Select most clinic-like bouts up to 360 seconds total (matching 6MWT duration)
   - **NOTE:** This step uses clinic data as reference for bout selection

4. **`preprocess_walking(walking_xyz, fs)`**
   - Gravity removal: 0.25 Hz lowpass → subtract
   - Rodrigues rotation (same as clinic preprocessing)
   - PCA yaw alignment → AP, ML, VT
   - Bandpass filter for _bp variants
   - Compute ENMO

5. **`extract_gait13(preprocessed_df)`**
   - Extract 13 features: cadence_hz, step_time_cv_pct, acf_step_regularity, hr_ap, hr_vt, ml_rms_g, ml_spectral_entropy, jerk_mean_abs_gps, enmo_mean_g, cadence_slope_per_min, vt_rms_g, ml_over_enmo, ml_over_vt

**Output:** `feats/home_hybrid_v2_features.npz` containing:
- `X_gait` (102, 13): Gait features (first 11 used as "Gait", last 2 as WalkSway ratios)
- `X_act` (102, 15): Activity profile features
- `X_cent` (102, 18): 6-minute activity centile features
- `X_cwt` (102, 28): CWT features
- `X_demo` (102, 4): Demographics
- `y` (102,): 6MWD targets

**To reproduce `home_hybrid_v2_features.npz` from scratch:**
```python
# After running home_hybrid_models_v2.py, save manually:
np.savez('feats/home_hybrid_v2_features.npz',
         X_gait=X_gait, X_act=X_act, X_cent=X_cent,
         X_cwt=X_cwt, X_demo=X_demo, y=y)
```

### Phase 2b: Alternative Home Walking Segments (preprocess_raw.py)

**Note:** `results_raw_pipeline/walking_segments/` was created by a DIFFERENT pipeline in `preprocess_raw.py`:
- Different walking detection (RMS + std + autocorrelation, min 20 sec)
- Different preprocessing (gravity projection, not Rodrigues rotation)
- No bandpass filter
- No clinic-informed bout selection

These walking_segments are used for WalkSway feature extraction but NOT for the main gait features. The gait features come from `home_hybrid_models_v2.py`.

### Phase 3: Clinic Feature Extraction

**Input:** `csv_preprocessed2/*.csv`, `csv_raw2/*.csv`
**Scripts:** `reproduce_c2.py`, `extract_walking_sway.py`

**Clinic Gait (11 features)** — from `csv_preprocessed2/`:
```python
from reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios
# extract_gait10(df) → 10 features from preprocessed AP/ML/VT + _bp signals
# compute_vt_rms() → vt_rms_g (11th feature)
# add_sway_ratios() → ml_over_enmo, ml_over_vt (moved to WalkSway)
```

**Clinic CWT (28 features)** — from `csv_raw2/`:
```python
from reproduce_c2 import extract_cwt
# extract_cwt(raw_xyz) → 28 features using Morlet wavelet on VM signal
# Aggregated across 6 temporal segments: mean, std, slope, slope_r
```

**Clinic WalkSway (12 features)** — from `csv_preprocessed2/`:
```python
from extract_walking_sway import extract_walking_sway
# extract_walking_sway(AP, ML, VT) → 10 ENMO-normalized sway features
# + ml_over_enmo, ml_over_vt from add_sway_ratios() = 12 total
```

**Clinic Demo (4 features):** cohort_POMS, Age, Sex, Height from `SwayDemographics.xlsx`

**Best clinic model:** All 55 features (11+28+12+4), Ridge α=10, LOO CV
```
R²=0.8055, MAE=100.1 ft, r=0.898, ρ=0.889
  POMS:    R²=0.874, MAE=93.2 ft
  Healthy: R²=0.637, MAE=104.2 ft
```

**Reproduced by:** `python results_table_final.py` (row: Gait+CWT+WalkSway+Demo)

### Phase 4: Walking Sway Features

**Input:** `csv_preprocessed2/*.csv` (clinic), `results_raw_pipeline/walking_segments/*.csv` (home)
**Script:** `extract_walking_sway.py`

```bash
python extract_walking_sway.py
```

Extracts 10 ENMO-normalized walking sway features + 2 sway ratios (ml_over_enmo, ml_over_vt) = 12 total.
Normalization by ENMO removes walking speed confound → higher = more instability.

**Output:** `feats/clinic_walking_sway.csv`, `feats/home_walking_sway.csv`

### Phase 5: Generate Results

**Script:** `results_table_final.py`

```bash
python results_table_final.py
```

Runs LOO CV for all feature set combinations. Uses Ridge α=10 for clinic, α=50 for best home.

**Output:** `feats/results_table_final.csv`

## Reproducing All Outputs

### Tables

| # | File | Description |
|---|---|---|
| T1 | `feats/demographics_table.csv` | Demographics / cohort characteristics |
| T2 | `feats/results_table_final.csv` | Main results: R² by feature set (`python results_table_final.py`) |
| T3 | `feats/ms_vs_healthy_features.csv` | POMS vs Healthy feature comparison (Mann-Whitney, BH-corrected) |
| T4 | `feats/feature_descriptions.csv` | Feature & clinical score names by category |
| T5 | `feats/error_analysis_by_cohort.csv` | Error analysis (R², MAE, RMSE, ρ) by cohort |
| T6 | `feats/clinic_home_feature_correlation.csv` | Clinic-home feature correlations |
| T7 | `feats/literature_comparison.csv` | Literature comparison table |

### Figures

| # | File | Description |
|---|---|---|
| F1 | `feats/fig_overview_diagram.svg` | Pipeline overview diagram |
| F2 | `feats/heatmap_feature_6mwd_corr.png` | Feature-6MWD Spearman correlations by cohort/setting |
| F3 | `feats/heatmap_clinical_corr_clinic.png` | Clinic wearable features vs clinical scores (POMS only) |
| F4 | `feats/heatmap_clinical_corr_home.png` | Home wearable features vs clinical scores (POMS only) |
| F5 | `feats/fig_predicted_vs_actual.png` | Predicted vs Actual 6MWD scatter (best models) |
| F6 | `feats/fig_bland_altman.png` | Bland-Altman agreement plots (best models) |
| F7 | `feats/fig_shap_importance.png` | SHAP top 10 feature importance beeswarm (Clinic + Home) |

## Feature Categories

### Gait (11 features)
Extracted from aligned AP/ML/VT acceleration during walking.

| Feature | Description | Unit |
|---|---|---|
| cadence_hz | Walking cadence | Hz |
| step_time_cv_pct | Step time coefficient of variation | % |
| acf_step_regularity | Autocorrelation step regularity | — |
| hr_ap | Harmonic ratio (anteroposterior) | — |
| hr_vt | Harmonic ratio (vertical) | — |
| ml_rms_g | Mediolateral RMS acceleration | g |
| ml_spectral_entropy | Mediolateral spectral entropy | — |
| jerk_mean_abs_gps | Mean absolute jerk of 3D velocity | g/s |
| enmo_mean_g | ENMO (activity intensity) | g |
| cadence_slope_per_min | Cadence trend over walk (fatigue) | Hz/min |
| vt_rms_g | Vertical RMS acceleration | g |

### Walking Sway (12 features)
ENMO-normalized trunk control features during walking. Higher = more instability.

| Feature | Description |
|---|---|
| ml_range_norm | ML peak-to-peak range / ENMO |
| ml_path_length_norm | Cumulative ML displacement / ENMO |
| ml_jerk_rms_norm | ML jerk RMS / ENMO |
| ap_rms_norm | AP RMS / ENMO |
| ap_range_norm | AP peak-to-peak range / ENMO |
| sway_ellipse_norm | 95% ML×AP ellipse area / ENMO² |
| ml_velocity_rms_norm | ML velocity RMS / ENMO |
| stride_ml_cv | Stride-to-stride ML peak CV |
| ml_ap_ratio | ML RMS / AP RMS |
| hr_ml | Harmonic ratio (mediolateral) |
| ml_over_enmo | ML RMS / ENMO |
| ml_over_vt | ML RMS / VT RMS |

### CWT (28 features)
Continuous wavelet transform features from raw accelerometer signal, computed per segment then aggregated (mean/std/slope).

### Demographics
- **Demo(3):** cohort_POMS, Age, Sex
- **Demo(5):** cohort_POMS, Age, Sex, Height, BMI (better for home prediction)

## Key Methods

- **LOO CV:** Leave-one-out cross-validation with StandardScaler inside the loop
- **Ridge:** α=10 for clinic models, α=50 for home models
- **PLS:** Partial Least Squares maps home gait features into clinic gait feature space (2 components)
- **Walking sway normalization:** Raw sway ÷ ENMO removes walking speed confound
- **SHAP:** Feature importance analysis on sensor features (excluding demographics)

## Key Scripts

| Script | Purpose |
|---|---|
| `preprocess_raw.py` | GT3X → daytime segments → walking_segments (alternative pipeline) |
| `reproduce_c2.py` | Clinic preprocessing + Gait/CWT feature extraction |
| `home_hybrid_models_v2.py` | **Home gait feature extraction** (creates home_hybrid_v2_features.npz) |
| `extract_walking_sway.py` | Walking sway feature extraction (clinic + home) |
| `extract_agd_features.py` | AGD epoch features (activity counts, inclinometer, steps) |
| `results_table_final.py` | Main results table (all feature combinations) |

## Important Notes

### Home Feature Pipeline Distinction
There are TWO different home preprocessing pipelines:
1. **`home_hybrid_models_v2.py`** (PRIMARY): Clinic-informed bout selection, Rodrigues rotation, bandpass filter → creates `home_hybrid_v2_features.npz` → **used for all home gait features**
2. **`preprocess_raw.py`** (SECONDARY): Simpler detection, gravity projection, no bandpass → creates `walking_segments/` → **used only for WalkSway features**

### Clinic-Informed Home Features
The home gait features use clinic data as a reference during bout selection (`select_walking_segment()` computes cosine similarity to clinic walking signature). This means even the "no PLS" home model implicitly uses clinic data during feature extraction.

### Clinic-Home Date Gaps
Most subjects (64%) had clinic and home recordings within 3 days. However, 14 subjects had gaps >30 days (max 559 days), which may introduce noise in home prediction.

## Excluded Subjects

- **M22:** Excluded due to data quality issues
- **M44:** Too-short clinic recording (601 samples)
- All analyses use n=101 (intersection of valid clinic + home data)

## Reproducing Best Results

### Best Clinic (R²=0.806)

```bash
python results_table_final.py  # row: Gait+CWT+WalkSway+Demo
```
- Features: Gait(11) from `reproduce_c2.py:extract_gait10()` + CWT(28) from `reproduce_c2.py:extract_cwt()` + WalkSway(12) from `extract_walking_sway.py` + Demo(4) from `SwayDemographics.xlsx`
- Model: Ridge α=10, LOO CV, n=101
- Input data: `csv_preprocessed2/`, `csv_raw2/`, `SwayDemographics.xlsx`

### Best Home with PLS (R²=0.519)

```bash
# Gait features from home_hybrid_models_v2.py (first 11 of X_gait)
# PLS(nc=2) maps home gait → clinic gait space
# Demo(5): cohort, age, sex, height, BMI
# Ridge α=20
```
- Features: PLS(2 components) + Demo(5) = 7 features
- Gait source: `feats/home_hybrid_v2_features.npz` X_gait[:, :11]
- Clinic target for PLS: `reproduce_c2.py:extract_gait10()` from `csv_preprocessed2/`
- **Requires paired clinic data** for PLS training

### Best Home without PLS (R²=0.488)

```bash
# Same gait features + Demo(5), Ridge α=50
```
- Features: Gait(11) + Demo(5) = 16 features
- Gait source: `feats/home_hybrid_v2_features.npz` X_gait[:, :11]
- **Note:** Home gait features use clinic data as reference during bout selection in `home_hybrid_models_v2.py:select_walking_segment()`

### Reproducing home_hybrid_v2_features.npz from scratch

```bash
python home_hybrid_models_v2.py
# Then manually save:
# np.savez('feats/home_hybrid_v2_features.npz', X_gait=X_gait, X_act=X_act, X_cent=X_cent, X_cwt=X_cwt, X_demo=X_demo, y=y)
```
- Verified: reproduced features correlate r=1.0000 with cached npz
- Pipeline: detect_active_bouts → refine_with_hr → select_walking_segment (clinic-informed) → preprocess_walking (Rodrigues) → extract_gait13

## Results History

| Date | Change | Clinic R² | Home R² (PLS) | Home R² (no PLS) |
|---|---|---|---|---|
| Initial | Gait13+CWT+Demo, PLS(Gait)+Demo(3) | 0.792 | 0.483 | 0.428 |
| +WalkSway | Gait11+CWT+WalkSway+Demo | 0.806 | 0.483 | 0.428 |
| +Demo(5) | Added Height+BMI to demographics | 0.806 | **0.519** | **0.488** |
| +α tuning | Ridge α=50 for home, PLS α=20 | 0.806 | 0.519 | 0.488 |
