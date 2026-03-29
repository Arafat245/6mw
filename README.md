# Predicting 6-Minute Walk Distance from Home Accelerometry in Pediatric-Onset MS

Predicting 6MWD from hip-worn accelerometer data in Pediatric-Onset Multiple Sclerosis (POMS).

**Subjects:** n=101 (POMS=38, Healthy=63). M22 (data quality) and M44 (too-short recording) excluded.

## Best Results

| Setting | Features | R² | MAE (ft) | rho |
|---|---|---|---|---|
| **Home (clinic-free)** | **Per-bout(20) + Demo(4) = 24f** | **0.451** | **183** | **0.658** |
| Clinic | Gait+CWT+WalkSway+Demo (55f) | 0.806 | 102 | 0.880 |

- Home: Spearman inside LOO (no data leakage), Ridge alpha=20, LOO CV
- Demo(4): cohort_POMS, Age, Sex, BMI
- Fully clinic-free — no clinic data used anywhere

## Prerequisites

```bash
# Conda environment with Python 3.11 (pygt3x requires <= 3.13)
conda create -n pygt3x-env python=3.11
conda activate pygt3x-env
pip install numpy pandas scipy scikit-learn openpyxl pygt3x PyWavelets
```

## Quick Start

```bash
conda activate pygt3x-env

# Step 0: GT3X -> daytime NPZ (one-time, ~2.5 hours)
# Reads raw GT3X files, extracts daytime (7AM-10PM), filters worn-time, saves compressed NPZ
# Input:  Accel files/*/*.gt3x + 6mw_segmented_walk_data_dict.pkl
# Output: csv_home_daytime_npz/*.npz (101 files) + _subjects.csv
python temporary_experiments/step0_gt3x_to_npz.py

# Step 1: Extract features + evaluate -> R²=0.451 (~8 minutes)
# Detects walking bouts (ENMO+HR), extracts 153 per-bout gait + activity features,
# loads Demo(4) from demographics, runs Spearman inside LOO with Ridge (no leakage)
# Input:  csv_home_daytime_npz/*.npz + Accel files/PedMSWalkStudy_Demographic.xlsx
# Output: feats/home_clinicfree_features.csv + feats/target_6mwd.csv
python temporary_experiments/step1_extract_and_predict.py
```

---

## Detailed Reproduction Pipeline

### Step 0: GT3X to Daytime NPZ (~2.5 hours, one-time)

**Script:** `temporary_experiments/step0_gt3x_to_npz.py`

**Input:**
- `Accel files/*/*.gt3x` — raw GT3X home accelerometer files (101 subjects)
- `6mw_segmented_walk_data_dict.pkl` — clinic 6MWT data with 6MWD targets (distance field)

**Process:**
1. Build subject list (101 subjects):
   - Load pkl: contains clinic 6MWT data with `distance` (6MWD in feet) per subject
   - Match pkl keys (C01, C02...M60) to GT3X files in `Accel files/`
   - Exclude M22 and M44
   - Sort: C subjects first, then M, by ID number
2. For each subject:
   - Load GT3X using `pygt3x.reader.FileReader` -> timestamps + X, Y, Z at 30 Hz
   - Extract daytime: keep 7AM-10PM hours only
   - Wear-time filter: rolling 5-second std of VM > 0.01 (remove non-worn periods)
   - If < 60s daytime data, use full recording as fallback
   - Save as compressed NPZ

**Output:**
- `csv_home_daytime_npz/{key}.npz` — daytime XYZ arrays (101 files, ~2.4GB total)
- `csv_home_daytime_npz/_subjects.csv` — subject list (key, cohort, subj_id, year, sixmwd, gt3x_path)

```bash
conda activate pygt3x-env
python temporary_experiments/step0_gt3x_to_npz.py
```

### Step 1: Feature Extraction + Prediction (~8 minutes)

**Script:** `temporary_experiments/step1_extract_and_predict.py`

**Input:**
- `csv_home_daytime_npz/*.npz` — daytime XYZ data
- `csv_home_daytime_npz/_subjects.csv` — subject list
- `Accel files/PedMSWalkStudy_Demographic.xlsx` — demographics (Age, Sex, BMI)

**Process:**

#### 1a. Walking Bout Detection (ENMO + Harmonic Ratio, 3 stages)

**Stage 1 — ENMO threshold:**
- Compute VM = sqrt(X² + Y² + Z²)
- ENMO = max(VM - 1.0, 0)
- Average ENMO per 1-second chunk
- Mark seconds with mean ENMO >= 0.015g as "active"
- Group consecutive active seconds into bouts (min 10s)

**Stage 2 — Harmonic Ratio refinement:**
- Bandpass filter VM at 0.5-3.0 Hz (4th order Butterworth)
- In 10-second non-overlapping windows within each bout:
  - FFT on bandpass signal
  - Find dominant frequency in 0.8-3.5 Hz
  - Compute harmonic ratio: sum(even harmonics) / sum(odd harmonics)
  - Accept window if HR >= 0.2
- Merge surviving windows into refined bouts (min 10s)

**Stage 3 — Merge adjacent:**
- If gap between two bouts < 5 seconds, merge them
- Result: list of (start_sample, end_sample) per subject

#### 1b. Per-Bout Gait Feature Extraction

For each walking bout (min 300 samples = 10s at 30Hz):

**Preprocessing:**
- Gravity removal: 0.25 Hz lowpass -> subtract
- Rodrigues rotation: align gravity vector with Z-axis
- PCA yaw alignment: PCA on horizontal plane -> AP, ML, VT axes
- Bandpass: 0.25-2.5 Hz -> AP_bp, ML_bp, VT_bp
- Compute ENMO and VM_dyn

**20 raw gait features per bout:**

| Feature | Description |
|---|---|
| cadence_hz | Dominant frequency from Welch PSD of VT_bp (0.5-3.5 Hz) |
| cadence_power | Power at dominant frequency |
| acf_step_reg | Autocorrelation at step lag |
| hr_ap, hr_vt, hr_ml | Harmonic ratios (AP, VT, ML axes) |
| stride_time_mean | Mean stride time from peak detection |
| stride_time_std | Stride time standard deviation |
| stride_time_cv | Stride time coefficient of variation |
| ml_rms, vt_rms, ap_rms | RMS amplitude per axis |
| enmo_mean | Mean ENMO intensity |
| enmo_p95 | 95th percentile ENMO |
| vm_std | VM variability |
| vt_range, ml_range | Peak-to-peak range |
| jerk_mean | Mean absolute jerk |
| signal_energy | Mean VM squared |
| duration_sec | Bout duration in seconds |

Bout rejected if cadence < 1.0 Hz.

#### 1c. Aggregation (6 stats per raw feature)

For each of the 20 raw features, compute across all valid bouts:
- `_med`: median
- `_iqr`: interquartile range (p75 - p25)
- `_p10`: 10th percentile
- `_p90`: 90th percentile
- `_max`: maximum
- `_cv`: coefficient of variation (std / mean)

Plus 4 bout meta-features:
- `g_n_valid_bouts`: number of valid bouts
- `g_total_walk_sec`: total walking duration
- `g_mean_bout_dur`: mean bout duration
- `g_bout_dur_cv`: bout duration CV

**Total: 124 gait features** (20 x 6 + 4)

#### 1d. Activity Features (whole recording)

Computed from full daytime recording (not bout-specific):
- ENMO distribution: mean, std, median, p5, p25, p75, p95, IQR, skew, kurtosis, entropy (11)
- Activity levels: % sedentary, light, moderate, vigorous, MVPA min/hr (5)
- Bout patterns: n_bouts, bouts_per_hr, bout_mean_dur, bout_dur_cv, longest_bout (5)
- Fragmentation: ASTP, SATP, fragmentation (3)
- Diurnal: early/mid/late ENMO, early_late_ratio, daily_cv (5)

**Total: 29 activity features**

**Grand total: 153 accelerometry features** (124 gait + 29 activity)

#### 1e. Demographics

Loaded from `Accel files/PedMSWalkStudy_Demographic.xlsx`:
- cohort_POMS (1 if MS, 0 if healthy)
- Age
- Sex
- BMI

Height dropped (redundant with BMI — BMI already encodes height).

**Demo(4): 4 features, always included (not part of feature selection)**

#### 1f. Imputation

For each feature column:
- If all NaN -> fill with 0
- If some NaN -> fill with column median

#### 1g. Evaluation (Spearman inside LOO, no data leakage)

```
For each of 101 LOO folds:
    Hold out subject i (never seen during selection or training)
    On 100 TRAINING subjects only:
        Compute |Spearman rho| of each of 153 accel features with y
        Rank by |rho|, select top 20
    Combine 20 selected + 4 demo = 24 features
    StandardScaler: fit on training, transform both
    Ridge(alpha=20): fit on training, predict held-out subject i
Collect 101 predictions
Compute: R² = 0.451, MAE = 183 ft, rho = 0.658
```

**Output:**
- `feats/home_clinicfree_features.csv` — 153 features for 101 subjects
- `feats/home_clinicfree_top20_reproduced.npz` — Top-20 feature matrix
- `feats/home_walking_bouts.pkl` — walking bout indices per subject
- `feats/target_6mwd.csv` — subject list with 6MWD targets

```bash
python temporary_experiments/step1_extract_and_predict.py
```

## Feature Selection Comparison

All methods evaluated inside LOO (no data leakage), K=20 + Demo(4), Ridge alpha=20:

| Method | R² | MAE (ft) | rho | Time |
|---|---|---|---|---|
| **Spearman** | **0.451** | **183** | **0.658** | **4s** |
| ANOVA-F | 0.416 | 193 | 0.620 | 0s |
| Pearson | 0.416 | 193 | 0.620 | 2s |
| ReliefF | 0.364 | 201 | 0.607 | 0s |
| Chi-Square | 0.358 | 201 | 0.589 | 42s |
| Decision Tree | 0.355 | 199 | 0.599 | 1s |
| Mutual Info | 0.338 | 204 | 0.562 | 10s |
| PCA (20 comp) | 0.077 | 237 | 0.341 | 1s |

## Walking Detection Comparison

Same per-bout features and evaluation, only walking detection method differs:

| Detection Method | K=10+Demo4 | K=20+Demo4 |
|---|---|---|
| **ENMO+HR (ours)** | R²=0.342 | **R²=0.451** |
| find_walking CWT — Python (cmor) | R²=0.384 | R²=0.243 |
| find_walking CWT — MATLAB (Morse) | R²=0.355 | R²=0.128 |

ENMO+HR detection outperforms CWT-based find_walking (Straczkiewicz 2023) for 6MWD prediction. The more permissive ENMO threshold (0.015g vs 0.3g) captures slow/impaired walking important for POMS.

## Feature Count Sweep

| K + Demo(4) | Total | R² | MAE | rho |
|---|---|---|---|---|
| 10 + 4 | 14 | 0.342 | 205 | 0.545 |
| 15 + 4 | 19 | 0.392 | 195 | 0.598 |
| **20 + 4** | **24** | **0.451** | **183** | **0.658** |
| 25 + 4 | 29 | 0.441 | 185 | 0.643 |
| 30 + 4 | 34 | 0.442 | 184 | 0.653 |

K=20 is the sweet spot.

## Key Data Dependencies

| File | Purpose |
|---|---|
| `Accel files/*/*.gt3x` | Raw home accelerometer data (GT3X format) |
| `6mw_segmented_walk_data_dict.pkl` | Clinic 6MWT data: subject keys + 6MWD (distance field) |
| `Accel files/PedMSWalkStudy_Demographic.xlsx` | Demographics: ID, Age, Sex, Height, Weight, BMI |

## Key Scripts

| Script | Purpose |
|---|---|
| `temporary_experiments/step0_gt3x_to_npz.py` | GT3X -> daytime NPZ (one-time, ~2.5h) |
| `temporary_experiments/step1_extract_and_predict.py` | Feature extraction + Top-20 prediction (R²=0.451) |
| `temporary_experiments/compare_feature_selection_v2.py` | Compare 8 feature selection methods |
| `temporary_experiments/step7_exact_findwalking.py` | find_walking CWT comparison |
| `home/extract_clinicfree_features.py` | Reference implementation of clinic-free features |

## Excluded Subjects
- **M22:** Data quality issues
- **M44:** Too-short clinic recording (601 samples)
- All analyses: n=101

## Results History

| Change | Home R² |
|---|---|
| Initial Top-20 (correlation on all data) | 0.441 |
| Spearman inside LOO + Demo(4) (no leakage) | **0.451** |
| find_walking CWT bouts (MATLAB Morse) | 0.128 |
| find_walking CWT bouts (Python cmor) | 0.243 |
| Forward selection (data leakage — invalid) | 0.555-0.736* |

*Forward selection results were inflated by data leakage (feature selection saw test data).
