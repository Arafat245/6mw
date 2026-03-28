# Gait Assessment in Pediatric-Onset Multiple Sclerosis Using Wearable Accelerometry

Predicting 6MWD from hip-worn accelerometer data collected during clinic 6-minute walk tests and home free-living monitoring in POMS.

**Subjects:** n=101 (POMS=38, Healthy=63), consistent across all analyses.

## Best Results (Current)

| Feature Set | #f | Home MAE (ft) | Home ρ | Home R² | Clinic MAE (ft) | Clinic ρ | Clinic R² |
|---|---|---|---|---|---|---|---|
| Gait | H:11, C:11 | 242 | 0.245 | 0.036 | 140 | 0.801 | 0.682 |
| CWT | H:28, C:28 | 243 | 0.269 | 0.034 | 197 | 0.601 | 0.357 |
| WalkSway | H:12, C:12 | 244 | -0.038 | -0.030 | 178 | 0.715 | 0.403 |
| PerBout-Top20 | 20 | 220 | 0.423 | 0.162 | — | — | — |
| Demo | H:4, C:4 | 203 | 0.588 | 0.355 | 218 | 0.524 | 0.305 |
| **PerBout-Top20+Demo (best home)** | **H:24, C:4** | **187** | **0.661** | **0.462** | 218 | 0.524 | 0.305 |
| **Gait+CWT+WalkSway+Demo** | **H:56, C:55** | 216 | 0.521 | 0.232 | **102** | **0.880** | **0.806** |

- **Home best**: Spearman Top-20 selected inside LOO (no data leakage) + Demo(4) = 24 features. Ridge α=20.
- **Clinic**: Full 6MWT. Demo(4) without BMI. Ridge α=10.
- **n=101**, LOO CV. Home is fully **clinic-free** — no clinic data used anywhere.

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

## HOME PIPELINE (Clinic-Free)

**No clinic data used anywhere.** The home pipeline is fully self-contained.

### Step H0: GT3X → Home Daytime CSV (already done)

**Script:** `home/preprocess_raw.py` → `load_gt3x()`, `extract_daytime()`
**Input:** `Accel files/*/*.gt3x` — raw ActiGraph GT3X binary files (30 Hz, ±6g, hip-worn)
**Output:** `csv_home_daytime/*.csv` — columns: X, Y, Z, fs (102 files, ~35 GB total)

```bash
python home/preprocess_raw.py
```

1. Read GT3X binary via `pygt3x` → timestamps + X, Y, Z at 30 Hz
2. Extract daytime only: keep hours 7:00 AM – 10:00 PM
3. Worn-time detection: rolling 5-second std of VM > 0.01 (remove non-worn periods)
4. Save as CSV with columns X, Y, Z, fs=30

**Output per subject:** 36–107 hours of daytime data (~1–4 million samples)

**Naming convention:** `{cohort}{subj_id:02d}_{year}_{sixmwd}.csv` (e.g., `C01_2016_2147.csv`)

### Step H1: Walking Bout Detection

**Script:** `home/extract_clinicfree_features.py` → `detect_walking_bouts()`
**Input:** `csv_home_daytime/*.csv` (X, Y, Z at 30 Hz)

Three-stage detection with no clinic reference:

**Stage 1 — ENMO Thresholding:**
1. Compute vector magnitude: VM = sqrt(X² + Y² + Z²)
2. Compute ENMO = max(VM - 1.0, 0) — removes gravity, keeps dynamic acceleration
3. Average ENMO into 1-second bins (30 samples per bin)
4. Mark seconds with mean ENMO ≥ **0.015g** as "active"
5. Group consecutive active seconds into bouts
6. Discard bouts shorter than **10 seconds**

**Stage 2 — Harmonic Ratio Refinement:**
1. Bandpass filter VM at **0.5–3.0 Hz** (4th-order Butterworth, zero-phase `filtfilt`)
2. Slide **10-second non-overlapping windows** through each active bout
3. Per window:
   - FFT on bandpass-filtered VM segment
   - Find dominant frequency in **0.8–3.5 Hz** band
   - Compute harmonic ratio: HR = Σ(even harmonics at 2f, 4f, ...) / Σ(odd harmonics at f, 3f, 5f, ...) up to 10 harmonics
4. Keep windows with HR ≥ **0.2** (confirms periodic walking pattern)
5. Merge adjacent passing windows; discard merged segments < **10 seconds**
6. If no windows pass HR filter in a bout, keep the original bout as fallback

**Stage 3 — Merge Adjacent Bouts:**
1. If gap between two refined bouts ≤ **5 seconds** → merge into one bout
2. Bouts farther apart remain separate

**Output per subject:** List of (start_sample, end_sample) tuples. Typically 374–1,552 walking bouts per subject (median bout duration: 14 seconds).

### Step H2: Per-Bout Preprocessing

**Script:** `home/extract_clinicfree_features.py` → `extract_bout_features()` → `preprocess_segment()`

For each walking bout independently (no concatenation of distant bouts):

1. **Gravity removal:**
   - 0.25 Hz 4th-order Butterworth lowpass → estimates gravity component
   - Subtract gravity estimate from raw signal → dynamic acceleration

2. **Rodrigues rotation:**
   - Compute mean gravity vector from lowpass estimate
   - Compute rotation angle between gravity vector and Z-axis [0,0,1]
   - Apply Rodrigues rotation formula to align gravity with vertical axis
   - Result: gravity-aligned coordinate system

3. **PCA yaw alignment:**
   - Take horizontal plane (X-Y) of rotated signal
   - Compute 2×2 covariance matrix
   - Eigenvector of largest eigenvalue = anterior-posterior (AP) direction
   - Rotate by yaw angle → AP, ML (mediolateral), VT (vertical) axes

4. **Bandpass filter:**
   - 0.25–2.5 Hz, 4th-order Butterworth, zero-phase → AP_bp, ML_bp, VT_bp
   - Isolates step-frequency band

5. **Derived signals:**
   - VM_dyn = norm(AP, ML, VT)
   - ENMO = max(norm(raw_resampled) - 1.0, 0)

### Step H3: Per-Bout Feature Extraction (20 features)

**Script:** `home/extract_clinicfree_features.py` → `extract_bout_features()`

Minimum bout length for feature extraction: **10 seconds** (300 samples at 30 Hz).
Bouts with estimated cadence < **1.0 Hz** are rejected (not true walking).

**20 features extracted from each valid bout:**

| # | Feature | How Computed |
|---|---|---|
| 1 | `cadence_hz` | Peak frequency of Welch PSD of VT_bp in 0.5–3.5 Hz (nperseg=max(4×fs, 256)) |
| 2 | `cadence_power` | Power at dominant cadence peak |
| 3 | `acf_step_reg` | Autocorrelation of VT_bp at step lag (lag = round(fs/cadence)) |
| 4 | `hr_ap` | Harmonic ratio of AP_bp: Σ(even harmonics) / Σ(odd harmonics) at cadence multiples, up to 10 harmonics |
| 5 | `hr_vt` | Harmonic ratio of VT_bp (same method) |
| 6 | `hr_ml` | Harmonic ratio of ML_bp (same method) |
| 7 | `stride_time_mean` | Mean inter-peak interval from VT_bp peak detection (min_distance=0.5×fs/cadence, prominence=0.5×std) |
| 8 | `stride_time_std` | Std of inter-peak intervals (ddof=1) |
| 9 | `stride_time_cv` | CV = std/mean of inter-peak intervals |
| 10 | `ml_rms` | RMS of ML axis: sqrt(mean(ML²)) |
| 11 | `vt_rms` | RMS of VT axis |
| 12 | `ap_rms` | RMS of AP axis |
| 13 | `enmo_mean` | Mean ENMO of bout |
| 14 | `enmo_p95` | 95th percentile ENMO of bout |
| 15 | `vm_std` | Std of VM_dyn |
| 16 | `vt_range` | Peak-to-peak range of VT (max - min) |
| 17 | `ml_range` | Peak-to-peak range of ML |
| 18 | `jerk_mean` | Mean absolute jerk: mean(|diff(VM_dyn)| × fs) |
| 19 | `signal_energy` | Mean of VM_dyn² |
| 20 | `duration_sec` | Bout duration in seconds |

### Step H4: Aggregation Across Bouts (124 gait features + 4 meta)

**Script:** `home/extract_clinicfree_features.py` → `extract_all_features()`

For each of the 20 per-bout features, compute **6 robust statistics** across all valid bouts:

| Stat | Suffix | What it captures |
|---|---|---|
| Median | `_med` | Typical bout value |
| IQR | `_iqr` | Spread (75th - 25th percentile) |
| 10th percentile | `_p10` | Worst/lowest bouts |
| 90th percentile | `_p90` | Best/highest bouts |
| Maximum | `_max` | Peak capacity |
| CV | `_cv` | Consistency across bouts (std/mean) |

→ 20 features × 6 stats = **120 gait-aggregated features**

Plus **4 bout meta-features:**

| Feature | Description |
|---|---|
| `g_n_valid_bouts` | Number of valid walking bouts (after cadence filter) |
| `g_total_walk_sec` | Total walking seconds across all valid bouts |
| `g_mean_bout_dur` | Mean bout duration (seconds) |
| `g_bout_dur_cv` | CV of bout durations (std/mean) |

→ **124 gait features total**

### Step H5: Activity Features (29 features, whole recording)

**Script:** `home/extract_clinicfree_features.py` → `extract_activity_features()`

Computed from the **entire daytime recording** (not just walking bouts):

**ENMO distribution (11 features):**

| Feature | Description |
|---|---|
| `act_enmo_mean` | Mean per-second ENMO across entire recording |
| `act_enmo_std` | Std of per-second ENMO |
| `act_enmo_median` | Median per-second ENMO |
| `act_enmo_p5` | 5th percentile |
| `act_enmo_p25` | 25th percentile |
| `act_enmo_p75` | 75th percentile |
| `act_enmo_p95` | 95th percentile |
| `act_enmo_iqr` | IQR (p75 - p25) |
| `act_enmo_skew` | Skewness of ENMO distribution |
| `act_enmo_kurtosis` | Kurtosis of ENMO distribution |
| `act_enmo_entropy` | Shannon entropy of 20-bin histogram |

**Intensity zones (5 features):**

| Feature | Threshold | Description |
|---|---|---|
| `act_pct_sedentary` | ENMO < 0.02g | Fraction of time sedentary |
| `act_pct_light` | 0.02 ≤ ENMO < 0.06g | Fraction in light activity |
| `act_pct_moderate` | 0.06 ≤ ENMO < 0.1g | Fraction in moderate activity |
| `act_pct_vigorous` | ENMO ≥ 0.1g | Fraction in vigorous activity |
| `act_mvpa_min_per_hr` | ENMO ≥ 0.06g | MVPA minutes per hour |

**Bout patterns (5 features):**

| Feature | Description |
|---|---|
| `act_n_bouts` | Number of active bouts (ENMO ≥ 0.02g, min 5s) |
| `act_bouts_per_hr` | Active bouts per hour |
| `act_bout_mean_dur` | Mean active bout duration (seconds) |
| `act_bout_dur_cv` | CV of active bout durations |
| `act_longest_bout` | Duration of longest active bout (seconds) |

**Transition dynamics (3 features):**

| Feature | Description |
|---|---|
| `act_astp` | Active-to-sedentary transition probability |
| `act_satp` | Sedentary-to-active transition probability |
| `act_fragmentation` | ASTP + SATP (overall fragmentation index) |

**Diurnal patterns (5 features):**

| Feature | Description |
|---|---|
| `act_early_enmo` | Mean ENMO of first third of recording |
| `act_mid_enmo` | Mean ENMO of middle third |
| `act_late_enmo` | Mean ENMO of last third |
| `act_early_late_ratio` | Early/late ENMO ratio |
| `act_daily_cv` | Day-to-day CV of mean ENMO (if ≥2 days) |

→ **29 activity features**

**Grand total: 153 accelerometry features** (124 gait + 29 activity)

### Step H6: Demographics (4 features)

**Source:** `SwayDemographics.xlsx`

| Feature | Column | Description |
|---|---|---|
| `cohort_POMS` | Derived from ID | 1 if M (POMS), 0 if C (Healthy) |
| `Age` | Age | Age in years at time of study |
| `Sex` | Sex | 1=Male, 2=Female |
| `BMI` | BMI | Body Mass Index (kg/m²) |

Height is NOT used (redundant with BMI). Missing values imputed with column median.

### Step H7: Feature Selection + Prediction (no data leakage)

**Script:** `analysis/reproduce_home_result.py`
**Input:** `feats/home_clinicfree_features.csv` (153 features), `SwayDemographics.xlsx`, `feats/target_6mwd.csv`

**Leave-One-Subject-Out Cross-Validation with nested feature selection:**

```
For each of 101 LOO folds:
    1. Hold out subject i (never seen during selection or training)
    2. On 100 TRAINING subjects only:
       a. For each of 153 accelerometry features:
          - Compute |Spearman ρ| between feature and 6MWD
       b. Rank features by |ρ| descending
       c. Select top K=20 features
    3. Combine 20 selected accelerometry + 4 demographic = 24 features
    4. StandardScaler: fit on 100 training subjects, transform both train and test
    5. Ridge regression (α=20): fit on 100 training subjects
    6. Predict held-out subject i's 6MWD

Collect 101 predictions
Compute metrics: R²=0.462, MAE=187 ft, ρ=0.661
```

**Key properties:**
- **No data leakage:** Feature selection uses only training data in each fold
- **No clinic data:** All features from home accelerometer + demographics
- **Reproducible:** Same result every run (deterministic algorithm)

```
Best Home Result (clinic-free, no leakage):
  R²=0.462, MAE=187 ft, ρ=0.661
```

**Top predictive features (all clinic-free, ranked by typical Spearman |ρ| with 6MWD):**
- Bout meta: longest bout duration (ρ=0.42), bout duration CV (ρ=0.41)
- Activity: peak daily ENMO (ρ=0.40), vigorous activity % (ρ=0.39)
- Gait quality: best step regularity (ρ=0.38), median jerk (ρ=0.34)

### To Reproduce

```bash
# Step 1: Extract 153 clinic-free features (one-time, ~15 min)
python home/extract_clinicfree_features.py
# Input:  csv_home_daytime/*.csv (home accelerometer, 101 subjects)
# Output: feats/home_clinicfree_features.csv (153 features × 101 subjects)
#         feats/home_clinicfree_top20.npz (pre-selected top-20)

# Step 2: Reproduce R²=0.462 with Spearman inside LOO (no leakage, ~1 min)
python analysis/reproduce_home_result.py
# Input:  feats/home_clinicfree_features.csv
#         feats/target_6mwd.csv
#         SwayDemographics.xlsx
# Output: R²=0.462, MAE=187, ρ=0.661

# Step 3: Full results table (home + clinic, ~30s)
python analysis/results_table_final.py
# Output: feats/results_table_final.csv
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
| `home/extract_clinicfree_features.py` | **Home clinic-free feature extraction** (creates home_clinicfree_top20.npz) |
| `home/home_hybrid_models_v2.py` | Home gait extraction (legacy, clinic-informed) |
| `home/preprocess_raw.py` | GT3X → daytime CSV → walking_segments |
| `home/extract_walking_sway.py` | Home WalkSway features |
| `home/extract_agd_features.py` | AGD epoch features |
| `analysis/results_table_final.py` | Main results table (all feature combinations) |

---

## Important Notes

### Clinic-Free Home Pipeline
The home model uses **no clinic data at all**. Walking bouts are detected using ENMO thresholding + harmonic ratio refinement (no cosine similarity to clinic signature). Features are extracted per-bout and aggregated with robust statistics (median, IQR, percentiles, CV). Feature selection (Spearman Top-20) is done **inside each LOO fold** to avoid data leakage. See `home/extract_clinicfree_features.py`.

### Legacy Home Pipelines (archived)
1. **`home/home_hybrid_models_v2.py`**: Clinic-informed bout selection via cosine similarity — no longer used for best results
2. **`home/preprocess_raw.py`**: Simpler detection → `walking_segments/` → WalkSway features

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

| Date | Change | Clinic R² | Home R² |
|---|---|---|---|
| Initial | Gait13+CWT+Demo | 0.792 | 0.428 |
| +WalkSway | Gait11+CWT+WalkSway+Demo | 0.806 | 0.428 |
| +Demo(5)+α | Added Height+BMI, tuned α | 0.806 | 0.488 |
| Clinic-free (fixed Top-20) | PerBout-Top20 + Demo(4), fixed selection | 0.806 | 0.447 |
| **Clinic-free (nested LOO)** | Spearman inside LOO + Demo(4), α=20, no leakage | 0.806 | **0.462** |
