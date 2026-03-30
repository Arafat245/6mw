# Gait Assessment in Pediatric-Onset Multiple Sclerosis Using Wearable Accelerometry

Predicting 6MWD from hip-worn accelerometer data collected during clinic 6-minute walk tests and home free-living monitoring in POMS.

**Subjects:** n=101 (POMS=38, Healthy=63), consistent across all analyses. Two subjects excluded: M22 (data quality) and M44 (too-short recording). Subject list in `feats/target_6mwd.csv`.

## Results Table

| Feature Set | #f | Clinic R² | Clinic MAE (m) | Clinic ρ | Home R² | Home MAE (m) | Home ρ |
|---|---|---|---|---|---|---|---|
| Gait | 11 | 0.682 | 42.7 | 0.801 | 0.145 | 70.1 | 0.377 |
| CWT | 28 | 0.357 | 60.2 | 0.601 | 0.150 | 68.0 | 0.461 |
| WalkSway | 12 | 0.403 | 54.2 | 0.715 | 0.061 | 72.7 | 0.315 |
| Demo | 4 | 0.362 | 60.8 | 0.595 | 0.362 | 60.8 | 0.595 |
| PerBout-Top20 | 20 | 0.583 | 46.8 | 0.778 | 0.184 | 67.5 | 0.451 |
| PerBout-Top20+Demo | 24 | 0.675 | 40.3 | 0.841 | **0.454** | **55.5** | **0.659** |
| **Gait+CWT+WS+Demo** | **55** | **0.806** | **31.2** | **0.880** | 0.238 | 65.7 | 0.529 |

- **n=101**, LOO CV, no data leakage. All metrics in meters.
- **Demo(4):** cohort_POMS, Age, Sex, BMI (same for clinic and home, Ridge α=20)
- **Clinic Gait/CWT/WS:** extracted from full 6MWT, Ridge with fixed α per feature set
- **Clinic PerBout:** 60s windows of 6MWT, Spearman Top-20 inside LOO, Ridge α=20
- **Home PerBout:** all walking bouts from full recording, Spearman Top-20 inside LOO, Ridge α=20
- **Home Gait/CWT/WS:** VM-based extraction from Top-10 clean bouts ≥60s, Spearman Top-11 inside LOO

---

## Data Requirements

| # | File | Description |
|---|---|---|
| 1 | `Accel files/*/*.gt3x` | Raw home free-living accelerometer (hip-worn, ±6g, ~7–10 days). 91 at 30 Hz, 8 at 60 Hz, 2 at 100 Hz. |
| 2 | `csv_raw2/*.csv` | Clinic 6MWT raw data (Timestamp, X, Y, Z). 101 files. |
| 3 | `csv_preprocessed2/*.csv` | Clinic preprocessed data (AP, ML, VT, _bp, ENMO). 101 files. |
| 4 | `feats/target_6mwd.csv` | Subject list with 6MWD ground truth (feet). 103 rows, exclude M22/M44 → 101. |
| 5 | `SwayDemographics.xlsx` | Demographics: ID, Age, Sex, Height, Weight, BMI. |

---

## CLINIC PIPELINE

### Feature Extraction

**Gait (11f), CWT (28f), WalkSway (12f):**
```bash
python clinic/extract_gait_cwt_ws_features.py
# Input:  csv_preprocessed2/*.csv + csv_raw2/*.csv
# Output: feats/clinic_gait_features.csv (101 x 12)
#         feats/clinic_cwt_features.csv (101 x 29)
#         feats/clinic_walksway_features.csv (101 x 13)
```

- **Gait:** `extract_gait10()` from preprocessed AP/ML/VT + `vt_rms_g` — cadence, step regularity, harmonic ratios, jerk, ENMO, cadence slope, spectral entropy
- **CWT:** `extract_cwt()` from raw XYZ VM — Morlet wavelet, 6 temporal segments, mean/std/slope
- **WalkSway:** `extract_walking_sway()` from preprocessed AP/ML/VT — 10 ENMO-normalized sway features + 2 ratios

**PerBout (124f) — 60s windows:**
```bash
python clinic/extract_perbout_features.py
# Input:  csv_raw2/*.csv
# Output: feats/clinic_perbout_features.csv (101 x 125)
```

- Split 6MWT into 60s non-overlapping windows (trim first/last 10s)
- Extract 20 per-bout features per window (same features as home PerBout)
- Aggregate across windows: 6 stats (median, IQR, p10, p90, max, CV) + 4 meta = 124 features

### Evaluation

```bash
python analysis/results_table_final.py
# Produces: Clinic Gait/CWT/WS/Combined + Home PerBout results
```

- **Gait:** Ridge α=5, LOO CV → R²=0.682
- **CWT:** Ridge α=20, LOO CV → R²=0.357
- **WalkSway:** Ridge α=5, LOO CV → R²=0.403
- **Gait+CWT+WS+Demo:** Ridge α=5, LOO CV → **R²=0.806**
- **PerBout-Top20+Demo:** Spearman Top-20 inside LOO, Ridge α=20 → R²=0.675

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

Per-bout preprocessing: gravity removal → Rodrigues rotation → PCA yaw alignment → AP, ML, VT + bandpass + ENMO. Extract 20 features per bout, aggregate with 6 stats = 124 gait + 4 meta + 29 activity = **153 features**.

### Step 3: Home PerBout Prediction

```bash
python home/step3_predict.py               # <1 sec (cached features)
# Input:  feats/home_perbout_features.csv + SwayDemographics.xlsx
# Output: R²=0.454, MAE=55.5m, ρ=0.659
```

Spearman Top-20 inside LOO + Demo(4), Ridge α=20.

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
python home/step3_predict.py                               # Predict (<1 sec)
```

---

## All Feature Files

| File | Created by | Features | Description |
|---|---|---|---|
| `feats/clinic_gait_features.csv` | `clinic/extract_gait_cwt_ws_features.py` | 11 + key | Clinic Gait from 6MWT |
| `feats/clinic_cwt_features.csv` | `clinic/extract_gait_cwt_ws_features.py` | 28 + key | Clinic CWT from 6MWT |
| `feats/clinic_walksway_features.csv` | `clinic/extract_gait_cwt_ws_features.py` | 12 + key | Clinic WalkSway from 6MWT |
| `feats/clinic_perbout_features.csv` | `clinic/extract_perbout_features.py` | 124 + key | Clinic PerBout (60s windows) |
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
| `home/step3_predict.py` | Home PerBout prediction (R²=0.454, instant) |
| `home/extract_gait_cwt_ws_features.py` | Home Gait/CWT/WalkSway features (VM-based) |
| `home/reproduce_from_bouts.py` | Reproduce from saved bout CSVs [--bout-dir] |
| `clinic/reproduce_c2.py` | Clinic preprocessing + Gait/CWT extraction functions |
| `clinic/extract_walking_sway.py` | Clinic WalkSway extraction function |
| `clinic/extract_gait_cwt_ws_features.py` | Clinic Gait/CWT/WalkSway feature extraction |
| `clinic/extract_perbout_features.py` | Clinic PerBout feature extraction (60s windows) |
| `analysis/results_table_final.py` | Full results table (clinic + home) |

---

## Directory Layout

```
6mw/
├── home/                           HOME PIPELINE
│   ├── step0_gt3x_to_npz.py         GT3X → NPZ
│   ├── step1_detect_walking_bouts.py Bout detection [--save-csv]
│   ├── step2_extract_features.py     PerBout features (153f)
│   ├── step3_predict.py              PerBout prediction (R²=0.454)
│   ├── extract_gait_cwt_ws_features.py Gait/CWT/WS features (VM)
│   └── reproduce_from_bouts.py       Reproduce from bout CSVs
│
├── clinic/                         CLINIC PIPELINE
│   ├── reproduce_c2.py               Preprocessing + Gait/CWT functions
│   ├── extract_walking_sway.py       WalkSway function
│   ├── extract_gait_cwt_ws_features.py Gait/CWT/WS feature extraction
│   └── extract_perbout_features.py   PerBout features (60s windows)
│
├── analysis/                       EVALUATION
│   └── results_table_final.py        Full results table
│
├── feats/                          CACHED FEATURES
│   ├── target_6mwd.csv               Ground truth
│   ├── home_perbout_features.csv     Home PerBout (153f)
│   ├── home_gait_features.csv        Home Gait (66f)
│   ├── home_cwt_features.csv         Home CWT (168f)
│   ├── home_walksway_features.csv    Home WalkSway (72f)
│   ├── clinic_gait_features.csv      Clinic Gait (11f)
│   ├── clinic_cwt_features.csv       Clinic CWT (28f)
│   ├── clinic_walksway_features.csv  Clinic WalkSway (12f)
│   └── clinic_perbout_features.csv   Clinic PerBout (124f)
│
├── results/                        RESULTS
│   └── results_table_final.csv       Final results table
│
├── home_full_recording_npz/        Full recording NPZ (101 files, ~2.4 GB)
├── walking_bouts/                  Walking bout CSVs (186,012 files)
├── csv_raw2/                       Clinic 6MWT raw data
├── csv_preprocessed2/              Clinic preprocessed data
├── Accel files/                    Raw home GT3X files
├── SwayDemographics.xlsx           Demographics
├── POMS/                           Paper (LaTeX)
├── archive/                        Old scripts (not in pipeline)
├── temporary_experiments/          Scratch space
└── notebooks/                      Exploratory notebooks
```

---

## Important Notes

### Sampling Frequencies
- 91 subjects at 30 Hz, 8 at 60 Hz (M26, M27, M29, M31, M32, M43, M45, C75), 2 at 100 Hz (M48, M49)
- All processing uses hardcoded FS=30 (no resampling — gives best results)
- Sampling rate is in GT3X metadata (`info.txt` inside ZIP, field `Sample Rate`)

### Home Pipeline Design Decisions
- **Full recording (no daytime filter):** R²=0.454 vs 0.365 with daytime only
- **No quality filtering for PerBout:** keeping all bouts gives R²=0.454 vs 0.356 with quality filter
- **Axis-based preprocessing for PerBout:** R²=0.454 vs 0.431 with VM-based
- **VM-based for Gait/CWT/WalkSway:** removes orientation artifacts in short free-living bouts
- **Top-10 clean bouts ≥60s for Gait/CWT/WS:** longer bouts + quality filter needed for clinic-style features

### File Matching
- Match clinic files by subject key (e.g., `C61`), not full filename — `target_6mwd.csv` years don't match `csv_raw2/`/`csv_preprocessed2/` filenames

### No Data Leakage
- Feature selection (Spearman) done inside each LOO fold (training data only)
- StandardScaler fit on training data only
- Ridge α is fixed per feature set (no alpha search)
- Imputation uses column median (negligible leakage from test subject)
- Clinic and home pipelines are fully separate
