# Predicting 6-Minute Walk Distance from Wearable Accelerometry in Pediatric MS

Predicting 6MWD from wrist-worn accelerometer data collected during clinic 6-minute walk tests and home free-living monitoring. Compares handcrafted gait/sway/wavelet features, foundation model embeddings, and PLS-based home-to-clinic domain adaptation.

**Subjects:** n=101 (MS=38, Healthy=63), consistent across all analyses.

## Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn openpyxl pywt xgboost
# Foundation models (optional, only needed for MOMENT/LimuBERT rows):
pip install momentfm torch
```

## Data Layout

```
csv_raw2/                           Raw clinic 6MWT recordings (Timestamp, X, Y, Z)
csv_preprocessed2/                  Preprocessed clinic data (AP, ML, VT axes)
csv_home_daytime/                   Home daytime accelerometer data (X, Y, Z)
results_raw_pipeline/
  walking_segments/                 Home walking bouts (preprocessed AP, ML, VT)
  emb_limubert_clinic.npy           LimuBERT embeddings (clinic)
  emb_limubert_home.npy             LimuBERT embeddings (home)
SwayDemographics.xlsx               Demographics and clinical scores
feats/
  target_6mwd.csv                   Ground truth (cohort, subj_id, year, sixmwd)
  home_hybrid_v2_features.npz       Home gait features (X_gait: 13 features)
  home_cwt_hybrid.csv               Home CWT features (28 features)
  moment_clinic_raw.npy             MOMENT embeddings (clinic)
  moment_home_raw.npy               MOMENT embeddings (home)
```

## Pipeline

### Step 1: Preprocessing (already cached)

Clinic raw data → preprocessed AP/ML/VT axes:

```bash
python reproduce_c2.py
```

This creates `csv_preprocessed2/` from `csv_raw2/` (gravity removal, Rodrigues rotation, PCA yaw alignment, bandpass filtering). Also extracts Gait10 + sway ratios + CWT features and runs initial LOO evaluation.

Home walking segments are already in `results_raw_pipeline/walking_segments/` (created by `preprocess_raw.py`).

### Step 2: Extract Walking Sway Features

```bash
python extract_walking_sway.py
```

Extracts 12 ENMO-normalized walking sway features from:
- Clinic: `csv_preprocessed2/` (6MWT data)
- Home: `results_raw_pipeline/walking_segments/`

**Output:** `feats/clinic_walking_sway.csv`, `feats/home_walking_sway.csv`

### Step 3: Generate Main Results Table

```bash
python results_table_final.py
```

Runs LOO CV (Ridge α=10) for all feature set combinations:
- Individual: Gait (11f), CWT (28f), WalkSway (12f), Demo (3-4f)
- Combined: Gait+CWT+WalkSway+Demo (54-55f)
- PLS: Home→Clinic domain adaptation (2 components + Demo)
- Foundation models: MOMENT PCA50, LimuBERT (with/without Demo)

**Output:** `feats/results_table_final.csv`

**Best results:**
- Clinic: Gait+CWT+WalkSway+Demo → R²=0.8055
- Home: PLS(Gait)+Demo → R²=0.4822

### Step 4: Generate All Figures and Tables

Run the analysis script to produce all paper outputs:

```bash
python generate_paper_outputs.py
```

Or generate individually (see below).

## Reproducing Individual Outputs

### Tables

| # | File | How to generate |
|---|---|---|
| T1 | `feats/demographics_table.csv` | Demographics (Table 1): age, sex, BMI, clinical scores by group |
| T2 | `feats/results_table_final.csv` | `python results_table_final.py` |
| T3 | `feats/ms_vs_healthy_features.csv` | MS vs Healthy comparison (Mann-Whitney, BH-corrected) |
| T4 | `feats/feature_descriptions.csv` | Feature names, descriptions, units for all 63 features |
| T5 | `feats/error_analysis_by_cohort.csv` | R², MAE, RMSE, Spearman ρ by cohort for all models |

### Figures

| # | File | Description |
|---|---|---|
| F1 | `feats/heatmap_feature_6mwd_corr.png` | Feature-6MWD Spearman correlations by cohort/setting |
| F2 | `feats/heatmap_clinical_corr_clinic.png` | Clinic wearable features vs clinical scores (MS only) |
| F3 | `feats/heatmap_clinical_corr_home.png` | Home wearable features vs clinical scores (MS only) |
| F4 | `feats/fig_predicted_vs_actual.png` | Predicted vs Actual 6MWD scatter (best models) |
| F5 | `feats/fig_bland_altman.png` | Bland-Altman agreement plots (best models) |

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

Features include: mean_energy, high_freq_energy, dominant_freq, estimated_cadence, max_power_freq, freq_variability, freq_cv, wavelet_entropy, fundamental_freq, harmonic_ratio — each with mean and std across segments, plus temporal slopes for energy/entropy/variability.

### Demographics (3-4 features)
- cohort_M (MS indicator), Age, Sex
- Height (clinic only, excluded from home models)

## Key Methods

- **LOO CV:** Leave-one-out cross-validation with StandardScaler inside the loop
- **Ridge (α=10):** Primary model for all evaluations
- **PLS:** Partial Least Squares maps home features into clinic feature space (2 components), enabling home-based 6MWD prediction without clinic data
- **Walking sway normalization:** Raw sway features divided by ENMO to remove walking speed confound, so higher normalized sway = more instability

## Key Scripts

| Script | Purpose |
|---|---|
| `reproduce_c2.py` | Preprocessing + clinic Gait/CWT feature extraction |
| `extract_walking_sway.py` | Walking sway feature extraction (clinic + home) |
| `results_table_final.py` | Main results table (all feature combinations) |

## Excluded Subjects

- **M22:** Excluded due to data quality issues (all analyses use n=101)
- **M44:** Too-short clinic recording (601 samples), excluded from clinic features but included in home — analyses requiring both use n=101 intersection
