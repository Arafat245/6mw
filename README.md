# Gait Assessment in Pediatric-Onset Multiple Sclerosis Using Wearable Accelerometry

Predicting 6MWD from hip-worn accelerometer data collected during clinic 6-minute walk tests and home free-living monitoring in POMS.

**Subjects:** n=101 (POMS=38, Healthy=63), consistent across all analyses. Two subjects excluded: M22 (data quality) and M44 (too-short clinic recording, 601 samples). Subject list in `feats/target_6mwd.csv` (103 rows including M22/M44 — filter with `cohort=='M' & subj_id in [22,44]`).

## Best Results (Current)

| Setting | Features | R² | MAE (ft) | ρ |
|---|---|---|---|---|
| **Clinic** | Gait+CWT+WalkSway+Demo (55f) | **0.806** | **102** | **0.880** |
| **Home (clinic-free)** | PerBout-Top20+Demo(4) (24f, Spearman inside LOO) | **0.454** | **182** | **0.659** |

- **Home best**: Full recording (no daytime filter), Spearman Top-20 selected inside LOO (no data leakage) + Demo(4) = 24 features. Ridge α=20.
- **Clinic**: Full 6MWT. Demo(4) without BMI. Ridge α=10.
- **n=101**, LOO CV. Home is fully **clinic-free** — no clinic data used anywhere.

## Folder Summary

| Folder | Contents | Count |
|---|---|---|
| `clinic/` | Clinic pipeline scripts | 2 scripts |
| `home/` | Home pipeline scripts (step0–step3 + reproduce) | 7 scripts |
| `analysis/` | Analysis & results scripts | 3 scripts |
| `POMS/` | Paper (LaTeX, figures, tables) | — |
| `feats/` | All extracted features & outputs | — |
| `home_full_recording_npz/` | Full recording NPZ files (101 subjects, ~2.4 GB) | 101 files |
| `walking_bouts/` | Walking bout CSVs per subject (Timestamp, X, Y, Z) | 186,012 files |
| `references/` | PDF reference papers | 15 PDFs |
| `notebooks/` | Jupyter notebooks (exploratory/legacy) | 12 notebooks |
| `old_figures/` | Old figures from early analysis | 18 PNGs |
| `data/misc/` | Misc CSV/Excel/data files | ~15 files |
| `data/legacy/` | Old data directories (csv_raw, csv_preprocessed) | 3 dirs |
| `archive/` | Old experimental scripts (not in final pipeline) | ~20 scripts |
| `temporary_experiments/` | Scratch space for new experiments before finalizing | — |

## Data Requirements

Three source files are needed to reproduce everything from scratch:

| # | File | Description | Format |
|---|---|---|---|
| 1 | `Accel files/*/*.gt3x` | Raw home free-living accelerometer recordings (hip-worn, ±6g, ~7–10 days). One GT3X file per subject inside subject folders (e.g., `C01_OPT/`, `M05_OPT-2016/`). 91 subjects at 30 Hz, 8 at 60 Hz, 2 at 100 Hz. | ActiGraph GT3X binary |
| 2 | `feats/target_6mwd.csv` | Subject list with 6MWD ground truth. 103 rows with columns: `cohort` (C=Healthy, M=POMS), `subj_id`, `year`, `sixmwd` (6-minute walk distance in feet). Exclude M22 and M44 → 101 subjects. | CSV |
| 3 | `SwayDemographics.xlsx` | Demographics and clinical scores. Columns: `ID` (e.g., C01, M05), `Age`, `Sex` (1=M, 2=F), `Height` (cm), `Weight` (kg), `BMI`, plus clinical scores (BDI, MFIS, EDSS — not used as predictors). | Excel |

**Pre-computed intermediate files** (generated from the above, already cached):

| File | Created by | Description |
|---|---|---|
| `csv_raw2/*.csv` | GT3X extraction | Clinic 6MWT raw data (Timestamp, X, Y, Z). 122 files. |
| `csv_preprocessed2/*.csv` | `clinic/reproduce_c2.py` | Clinic preprocessed (AP, ML, VT, _bp, ENMO). 120 files. |
| `home_full_recording_npz/*.npz` | `home/step0_gt3x_to_npz.py` | Full home recording (xyz + timestamps). 101 files, ~2.4 GB. |
| `walking_bouts/{subj_id}/bout_*.csv` | `home/step1_detect_walking_bouts.py` | Walking bout CSVs (Timestamp, X, Y, Z). 186,012 files. |
| `feats/home_clinicfree_features.csv` | `home/step2_extract_features.py` | 153 clinic-free features × 101 subjects. |
| `feats/home_walking_bouts.pkl` | `home/step1_detect_walking_bouts.py` | Bout indices per subject (pickle). |

If intermediate files are missing, run the corresponding script to regenerate them.

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
│   ├── step0_gt3x_to_npz.py         GT3X → full recording NPZ
│   ├── step1_detect_walking_bouts.py Walking bout detection [--save-csv]
│   ├── step2_extract_features.py     Per-bout + activity feature extraction
│   ├── step3_predict.py              LOO CV prediction → R²=0.454
│   ├── reproduce_from_bouts.py       Reproduce from saved bout CSVs [--bout-dir]
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
├── home_full_recording_npz/        HOME DATA — full recording NPZ (xyz + timestamps)
├── walking_bouts/                  HOME DATA — walking bout CSVs per subject
├── results_raw_pipeline/           HOME DATA — walking segments + embeddings
│   ├── walking_segments/             Walking bouts (AP, ML, VT)
│   └── emb_*.npy                     Foundation model embeddings
├── Accel files/                    RAW DATA — home GT3X files + AGD + wear diary
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

**No clinic data used anywhere.** The home pipeline is fully self-contained. Uses full recording (no daytime filtering) — keeping all data including sleep/sedentary periods improves prediction.

### Step H0: GT3X → NPZ

**Script:** `home/step0_gt3x_to_npz.py`
**Input:** `Accel files/*/*.gt3x` + `feats/target_6mwd.csv`
**Output:** `home_full_recording_npz/{key}.npz` — compressed NPZ with `xyz` (float32) + `timestamps` (float64). 101 files, ~2.4 GB total.

```bash
python home/step0_gt3x_to_npz.py
# ~60 min, one-time
```

Reads raw GT3X via `pygt3x`, saves full recording as-is (no filtering). NPZ format is ~15x smaller than CSV.

### Step H1: Walking Bout Detection

**Script:** `home/step1_detect_walking_bouts.py`
**Input:** `home_full_recording_npz/*.npz`
**Output:**
- `feats/home_walking_bouts.pkl` — bout indices (start, end) per subject
- `walking_bouts/{subj_id}/bout_*.csv` — individual bout CSVs with Timestamp, X, Y, Z (optional, with `--save-csv`)

```bash
python home/step1_detect_walking_bouts.py --save-csv
# ~18 min with --save-csv, ~3 min without
```

Three-stage detection (all at fs=30 Hz):

**Stage 1 — ENMO Thresholding:**
1. VM = sqrt(X² + Y² + Z²), ENMO = max(VM - 1.0, 0)
2. Average ENMO into 1-second bins
3. Mark seconds with mean ENMO ≥ **0.015g** as "active"
4. Group consecutive active seconds, discard bouts < **10 seconds**

**Stage 2 — Harmonic Ratio Refinement:**
1. Bandpass filter VM at **0.5–3.0 Hz** (4th-order Butterworth)
2. **10-second non-overlapping windows**: FFT → dominant frequency in 0.8–3.5 Hz → HR = Σ(even harmonics) / Σ(odd harmonics)
3. Keep windows with HR ≥ **0.2**; merge adjacent; discard < 10s
4. If no windows pass, keep original bout as fallback

**Stage 3 — Merge:** gaps ≤ **5 seconds** → merge

**Output:** 186,012 total bouts across 101 subjects.

### Step H2: Feature Extraction (153 features)

**Script:** `home/step2_extract_features.py`
**Input:** `home_full_recording_npz/*.npz` + `feats/home_walking_bouts.pkl`
**Output:** `feats/home_clinicfree_features.csv` — 153 features × 101 subjects

```bash
python home/step2_extract_features.py
# ~12 min
```

**Per-bout preprocessing** (for each bout independently):
1. Gravity removal (0.25 Hz lowpass → subtract)
2. Rodrigues rotation (align gravity with Z-axis)
3. PCA yaw alignment → AP, ML, VT axes
4. Bandpass 0.25–2.5 Hz → AP_bp, ML_bp, VT_bp
5. Derived: VM_dyn, ENMO

**20 per-bout gait features:** cadence_hz, cadence_power, acf_step_reg, hr_ap, hr_vt, hr_ml, stride_time_mean/std/cv, ml_rms, vt_rms, ap_rms, enmo_mean, enmo_p95, vm_std, vt_range, ml_range, jerk_mean, signal_energy, duration_sec

**Aggregation:** 6 stats per feature (median, IQR, p10, p90, max, CV) = 120 gait + 4 meta = **124 gait features**

**29 activity features** (from full recording): ENMO distribution (11), intensity zones (5), bout patterns (5), transition dynamics (3), diurnal patterns (5)

**Total: 153 features** (124 gait + 29 activity)

### Step H3: Prediction

**Script:** `home/step3_predict.py`
**Input:** `feats/home_clinicfree_features.csv` + `SwayDemographics.xlsx` + `home_full_recording_npz/_subjects.csv`
**Output:** Prints R², MAE, ρ

```bash
python home/step3_predict.py
# <1 sec
```

**LOO CV with nested feature selection:**
1. Hold out subject i
2. On 100 training subjects: rank 153 features by |Spearman ρ| with 6MWD
3. Select top K=20 + Demo(4) = 24 features
4. StandardScaler → Ridge (α=20) → predict held-out subject

**Result: R²=0.454, MAE=182 ft, ρ=0.659**

**Alternative:** Reproduce from saved walking bout CSVs (no need for NPZ or pkl):
```bash
python home/reproduce_from_bouts.py [--bout-dir walking_bouts]
# ~18 min, same result
```

### To Reproduce

```bash
# From scratch (one-time, ~90 min total):
python home/step0_gt3x_to_npz.py          # GT3X → NPZ (~60 min)
python home/step1_detect_walking_bouts.py --save-csv  # Detect + save bouts (~18 min)
python home/step2_extract_features.py      # Extract 153 features (~12 min)
python home/step3_predict.py               # LOO CV → R²=0.454 (<1 sec)

# With cached features (instant):
python home/step3_predict.py               # R²=0.454

# From saved walking bout CSVs:
python home/reproduce_from_bouts.py        # R²=0.454 (~18 min)
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
| `home/step0_gt3x_to_npz.py` | GT3X → full recording NPZ |
| `home/step1_detect_walking_bouts.py` | Walking bout detection + optional CSV saving |
| `home/step2_extract_features.py` | Per-bout gait + activity feature extraction |
| `home/step3_predict.py` | LOO CV prediction (R²=0.454) |
| `home/reproduce_from_bouts.py` | Reproduce from saved walking bout CSVs |
| `clinic/reproduce_c2.py` | Clinic preprocessing + Gait/CWT extraction |
| `clinic/extract_walking_sway.py` | Clinic WalkSway features |
| `analysis/results_table_final.py` | Main results table (all feature combinations) |

---

## Important Notes

### Clinic-Free Home Pipeline
The home model uses **no clinic data at all**. Full recording is used (no daytime filtering — keeping sleep/sedentary data improves prediction). Walking bouts are detected using ENMO thresholding + harmonic ratio refinement. Features are extracted per-bout and aggregated with robust statistics. Feature selection (Spearman Top-20) is done **inside each LOO fold** to avoid data leakage. See `home/step1_detect_walking_bouts.py`, `home/step2_extract_features.py`, `home/step3_predict.py`.

### Legacy Home Pipelines (archived)
1. **`home/home_hybrid_models_v2.py`**: Clinic-informed bout selection via cosine similarity — no longer used
2. **`home/preprocess_raw.py`**: Daytime filtering + simpler detection → `walking_segments/`
3. **`home/extract_clinicfree_features.py`**: Older feature extraction (daytime only, R²=0.365)

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
| Clinic-free (nested LOO, daytime) | Spearman inside LOO + Demo(4), α=20, daytime only | 0.806 | 0.365 |
| **Clinic-free (nested LOO, full recording)** | Spearman inside LOO + Demo(4), α=20, no daytime filter | 0.806 | **0.454** |
