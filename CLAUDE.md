# CLAUDE.md

## Project Overview

Predicting 6-Minute Walk Distance (6MWD) from hip-worn accelerometer data in Pediatric-Onset Multiple Sclerosis (POMS). n=101 (POMS=38, Healthy=63) after excluding M22 (data quality) and M44 (too-short recording).

## Key Rules

### Data & Analysis
- **Always n=101.** M22 and M44 are excluded from all analyses.
- **LOO CV only.** Leave-One-Subject-Out cross-validation. Never use L5SO or random splits.
- **Report in meters.** Convert all 6MWD predictions and true values from feet to meters (×0.3048) before evaluating. MAE, RMSE, etc. should be in meters (m), not feet.
- **Demographics = basic only.** Demo features: Age, Sex, Height, BMI, cohort_POMS. Never include clinical scores (BDI, MFIS, etc.) as predictors — they require clinic visits.
- **No CCPT.** Do not use predicted 6MWD as input to predict 6MWD (circular).
- **Home pipeline is fully clinic-free.** `home/step0→step3` pipeline. No clinic data used anywhere.
- **Full recording for home.** No daytime filtering — keeping all data (including sleep/sedentary) improves prediction (R²=0.452 vs 0.365 with daytime only).
- **Hardcoded FS=30 for all subjects.** Do NOT resample non-30Hz subjects. Resampling hurts performance (resample_poly R²=0.342, linear interp R²=0.373, hardcoded 30Hz R²=0.452).
- **Keep all walking bouts.** Do not filter/verify bouts — verified bouts give worse results (R²=0.423 vs 0.452).
- **Match clinic files by subject key** (e.g., `C61`), not full filename — `target_6mwd.csv` years don't match `csv_raw2/`/`csv_preprocessed2/` filenames.

### Sampling Frequencies
- 91 subjects at 30 Hz, 8 at 60 Hz (M26, M27, M29, M31, M32, M43, M45, C75), 2 at 100 Hz (M48, M49)
- Sampling rate is in GT3X metadata (`info.txt` inside ZIP, field `Sample Rate`)
- pygt3x 0.7.1 returns float64 timestamps (Unix epoch seconds) on both Linux and Windows

### Coding & Experiments
- **New experiments go in `temp_exps/`.** All new scripts, cached features, cloned repos, and outputs must live in `temp_exps/` until they produce results that improve on current best. Only then promote to `home/`, `analysis/`, etc.
- **Cache features.** Save extracted features to `.npz` or `.csv` so re-extraction is instant. Use `feats/` directory for finalized features only.
- **Use cached data.** Read from `home_full_recording_npz/` (not GT3X) when possible. Use `csv_preprocessed2/` (not csv_raw2) for clinic.
- **Keep experiments fast.** Prefer small focused tests (5-10 configs) over exhaustive grid search. Kill experiments early if trends are clear.
- **No unnecessary runs.** Don't re-run pipelines that haven't changed. Don't recompute features that are already cached.
- **Be honest about time estimates.**

### Paper (LaTeX in `POMS/`)
- **Max 7 figures, max 7 tables.** Both budgets are currently full — warn before adding new ones.
- **All outputs go to `feats/`.** Tables as CSV, figures as PNG/SVG.
- **Tables go to `POMS/tables/`, figures to `POMS/figures/`.**

### Models
- **Clinic:** Ridge(α=5) is the best model. Non-linear models (RF, XGBoost, SVR, KNN, GPR) all worse.
- **Home:** Ridge(α=20) is the best model. Vote ensemble (R²=0.478) offers only marginal gain over Ridge (R²=0.452), not worth the complexity.
- **Foundation models** (MOMENT, Chronos, LimuBERT) are comparison baselines only. Handcrafted features beat them.
- **DL models** (TCN, LSTM, Transformer) are used separately from foundation models, not mixed.

## Current Best Results

| Setting | Features | Model | R² | MAE (m) | r |
|---|---|---|---|---|---|
| Clinic | Gait+CWT+WalkSway+Demo (55f) | Ridge(α=5) | 0.806 | 31.2 | 0.898 |
| Home (clinic-free) | Bout+Act-Top20+Demo(4) (24f, Spearman inside LOO) | Ridge(α=20) | 0.452 | 56.0 | 0.672 |

## Home Pipeline (step0 → step3)

```bash
python home/step0_gt3x_to_npz.py          # GT3X → NPZ (~60 min, one-time)
python home/step1_detect_walking_bouts.py --save-csv  # Detect + save bouts (~18 min)
python home/step2_extract_features.py      # Extract 153 features (~12 min)
python home/step3_predict.py               # LOO CV → R²=0.452 (<1 sec)
```

Quick reproduction: `python home/step3_predict.py` (cached features)
From saved bouts: `python home/reproduce_from_bouts.py [--bout-dir walking_bouts]`
Full results table: `python analysis/results_table_final.py` (clinic + home)

## Key Scripts

| Script | Purpose |
|---|---|
| `home/step0_gt3x_to_npz.py` | GT3X → full recording NPZ (no filtering) |
| `home/step1_detect_walking_bouts.py` | Walking bout detection + optional CSV saving |
| `home/step2_extract_features.py` | Per-bout gait + activity feature extraction |
| `home/step3_predict.py` | LOO CV prediction (R²=0.452) |
| `home/reproduce_from_bouts.py` | Reproduce from saved walking bout CSVs |
| `clinic/reproduce_c2.py` | Clinic preprocessing + Gait/CWT extraction |
| `clinic/extract_walking_sway.py` | Clinic WalkSway features |
| `analysis/results_table_final.py` | Full results table (clinic + home, all 101 subjects) |

## Directory Layout

- `home_full_recording_npz/` — Full recording NPZ files (101 subjects, ~2.4 GB)
- `walking_bouts/` — Walking bout CSVs per subject (Timestamp, X, Y, Z, 186K files)
- `feats/` — All extracted features, outputs, tables, figures
- `csv_raw2/` — Clinic 6MWT raw data
- `csv_preprocessed2/` — Clinic preprocessed data
- `Accel files/` — Raw home GT3X files
- `POMS/` — Paper LaTeX source, figures, tables
- `archive/` — Old experimental scripts (NOT part of final pipeline)
- `temp_exps/` — Scratch space for new experiments
- `notebooks/` — Exploratory/legacy Jupyter notebooks

## Experiment History

- **Clinic per-bout (60s windows):** Bout+Act-Top20+Demo(4) R²=0.69 (after adding 29 activity features from full 6MWT to clinic feature pool, 2026-04-19; was R²=0.679 with 123-feature gait+meta-only pool). Use 60s windows, not 30s.
- **Home daytime only:** R²=0.365 — worse than full recording
- **Home worn-time filter only:** R²=0.407 — worse than no filter
- **Resampling non-30Hz to 30Hz:** resample_poly R²=0.342, linear interp R²=0.373 — both worse
- **Walking bout verification:** R²=0.423 — worse than keeping all bouts
- **Circadian/temporal features (IS, IV, L5/M10, cosinor, bout timing):** 18 new features from clock-time analysis of full recordings. Too weakly correlated with 6MWD — Spearman never selects them over gait features. Standalone R²=0.337.
- **MRMR selection (inside LOO):** R²=0.423 on Bout+Act, R²=0.431 on Bout+Act+Goldman — worse than Spearman Top-20 (0.452). Reducing redundancy hurts.
- **Goldman PAM features (MSR, HWSR, daily steps):** Only 92/101 subjects have data. With imputation, Spearman-20 R²=0.387, MRMR-20 R²=0.431 — both worse.
- **Combined pools (Bout+Act+Circadian+Goldman):** Spearman-20 R²=0.387, MRMR-20 R²=0.401 — adding features beyond Bout+Act hurts or doesn't help.
- **Forward selection (outside LOO):** Previously reported R²=0.555 — inflated by selection leakage (feature selection not inside LOO folds). Not valid.
- **Kernel Ridge Regression (RBF):** All negative R² — severely overfits at n=101. Even with inner 5-fold CV: R²=0.06.
- **SVR (RBF):** Best individual R²=0.428 (C=1000, γ=0.01) — close but worse than Ridge.
- **Gait complexity features (sample entropy, spectral entropy, permutation entropy):** 18 features from top-10 longest bouts. Too weakly correlated — Spearman never selects them.
- **Ridge+SVR blending:** 0.5*Ridge(α=20) + 0.5*SVR(rbf, C=500, γ=0.05) on same Spearman-20 + Demo(4): R²=0.472, MAE=53.8m, ρ=0.691. Modest improvement over R²=0.452 baseline.
- **Vote(Ridge+Lasso+SVR):** Best ensemble for home. R²=0.478, MAE=53.6m, ρ=0.674. Only +0.024 over Ridge (0.452) — not worth the complexity, so Ridge is used as the final model.
- **Stacking regressor (inner 5-fold CV, Ridge meta):** All combos worse than voting at n=101 — inner CV meta-features are too noisy.
- **Pearson selection (inside LOO) — both settings:** Re-ran with current feature set (152f home, 123f clinic, after `g_n_valid_bouts` removal) and Pearson Top-20 + Demo, Ridge same α. **Home:** Pearson R²=0.406, MAE=59.4m, r=0.638 vs Spearman baseline R²=0.452, MAE=56.0m, r=0.672 (ΔR²=−0.046). **Clinic:** Pearson R²=0.570, MAE=41.5m, r=0.763 vs Spearman R²=0.663, MAE=40.3m, r=0.818 (ΔR²=−0.093). Mean selection overlap: 14.7/20 home, 19.0/20 clinic — clinic differs by only 1 feature yet costs 9 R² points, suggesting Pearson preferentially picks one strongly-linear feature over a more informative monotonic one. Earlier session entry (R²=0.416 Ridge, 0.429 blend) used a slightly different feature set; conclusion is consistent. Note: reporting Pearson r as the predicted-vs-actual *metric* is independent from selection method — Ridge produces linear outputs (so r is the right metric), but features benefit from monotonic-rank selection because gait↔6MWD relationships saturate (e.g., max bout duration plateaus, ENMO percentiles non-linear). Keep Spearman selection as the canonical pipeline.
- **Clinic non-linear models:** KNN R²=0.460, RF R²=0.652, XGBoost R²=0.663, SVR R²=0.642 — all worse than Ridge R²=0.806. Voting/stacking also worse for clinic.
- **Late fusion (separate Demo + Bout+Act models):** Best R²=0.385 (0.7*Demo+0.3*B+A) — worse than early fusion (0.452). Loses cross-modal information.
- **Residual fusion (Demo→residuals→Bout+Act):** R²=0.363 — worse. Two-stage approach loses information vs joint model.
- **Residual-guided feature selection:** Select Bout+Act by correlation with Demo residuals, then early fuse. R²=0.388 — worse than standard Spearman selection (0.452).
- **Feature interactions (Bout+Act×Demo cross-terms):** 80 interaction features + Ridge α=100: R²=0.387. Overfits at n=101.
- **Modality-weighted fusion (inner CV for weight):** R²=0.424 — worse than unweighted concatenation.
- **Gaussian Process Regression:** Pure non-linear kernels (RBF R²=0.09, Matern R²=0.20) overfit badly. DotProduct+RBF R²=0.450 ≈ Ridge (essentially linear). No improvement over Ridge.
- **Transfer learning from external 6MWT dataset (Nature Sci Data 2025, 60 healthy adults):** Trained Ridge on per-path gait features→6MWD from 60 subjects' controlled 6MWT (lower-back IMU, 100Hz→30Hz). Applied to home bouts as "virtual clinic assessment." Transfer-only+Demo R²=0.323, transfer median+Demo R²=0.356. Pooled with Bout+Act features: R²=0.453 (+0.001) — negligible. Domain gap too large (controlled vs free-living, healthy vs POMS).
- **LLM few-shot prediction (Claude Haiku):** 100 training subjects as in-context examples with 8 gait features + demographics, predict held-out 6MWD. R²=0.341, MAE=62.4m, ρ=0.602. Ensembling 0.3*LLM+0.7*Ridge: R²=0.441 — worse than Ridge alone. LLMs can't match Ridge's numeric precision.
- **RAG-inspired patient similarity (k-NN + Ridge blend):** Inverse-distance-weighted k-NN in standardized feature space, blended with Ridge. All configs (k=3/5/7/10, α=0.5–0.8) worse than pure Ridge. k-NN adds noise at n=101.
- **Bout-weighted aggregation (type-7 weighted quantile across 6 stats):** Pre-registered primary = `sqrt(duration_sec) * max(0,acf_step_reg) / (1+stride_time_cv)`, plus 6 exploratory weight schemes (sqrt(dur), dur, sqrt(cadence_power), acf alone, sqrt(dur)×acf, primary with p95-replaces-max). Uniform W0 reproduces baseline exactly (R²=0.4516). All 7 weighted variants worse by 0.05–0.13 R² — primary R²=0.329. Weighting bouts by length/rhythm/stability systematically downweights short/noisy bouts that carry the between-subject capacity signal, erasing the variance Ridge relies on.
- **Bout-burden features (explicit encoding of "bad bouts"):** 20 burden features appended to baseline 153 — 12 fold-agnostic (short-bout fractions dur<15/<30/<60, shape ratios p10/med, p25/med, (p90-p10)/med for acf_step_reg, stride_time_cv, duration_sec) + 8 fold-local (rhythm/unstable/weak-cadence burdens at TRAIN p25/p50/p75/p90, plus 2 joint burdens). Designed to encode the capacity signal that weighting erased. Global-only (165f) identical to baseline (Spearman Top-20 never selected any burden feature — outranked by gait features). Fold-local added (173f, primary) R²=0.429, −0.023 worse. Fold-local burdens rank high enough to displace stronger features; burden content is redundant with existing `g_stride_time_cv_med` / `g_acf_step_reg_max`. Baseline already efficiently captures bout heterogeneity.
- **Multi-α Ridge ensemble (same feature/selection protocol):** Equal-weight average of Ridge predictions at α ∈ {5, 10, 20, 50, 100} on the same Spearman Top-20 + Demo(4) features, pre-registered as confirmatory. Ensemble R²=0.4424, −0.009 vs baseline. High-α models (50: 0.417; 100: 0.366) over-regularize toward the mean and drag the equal-weight ensemble down. Standalone α=10 gave R²=0.4555 (marginal +0.004), but post-hoc α-selection is the same leakage pattern as forward selection — not promoted. Confirms α=20 is near-optimal and the plateau at ~0.452 is real for this representation at n=101.
- **Inverted pendulum distance (Zijlstra & Hof 2003) — clinic:** Direct biomechanical estimate of 6MWD from VT_bp via per-step double integration. Steps detected as VT_bp peaks (≥0.25s spacing); per-step h = peak-to-peak vertical CoM displacement after high-pass-detrended cumulative integration; SL = 1.25·2·√(2·L·h − h²) with L = 0.53·Height; total distance summed across steps and scaled to 360s. Result: R²=−0.427, MAE=88.7m, r=0.613, ρ=0.612 (n=101, mean pred 555m vs true 638m). Negative R² from biased global scaling (K=1.25, L=0.53·H are population heuristics, not per-subject calibrated); even with optimal linear calibration the ceiling is r²≈0.376 — worse than Demo-only (0.362) and far below Ridge (0.806). Confirms direct integration of hip accel cannot recover absolute walked distance: residual gravity leakage, bandpass removing DC velocity, and hip ≠ CoM kinematics each cost more accuracy than they save. Script deleted from `temp_exps/`.
- **Cadence-only clinic baselines (parsimonious models):** Single-feature and mechanistic-physics baselines using only `cadence_hz` from `clinic_gait_features.csv`. (A) Cadence alone (Ridge, 1f) R²=0.471, MAE=57.1m — already beats Demo-only (4f, R²=0.362). (B) Cadence + Height (2f) R²=0.680, MAE=45.3m. (C) Cadence + Demo(4) (5f) R²=0.712, MAE=43.3m. (D) Pure mechanistic `d = K · cadence(Hz) · Height(m) · 360s` with LOO-calibrated scalar K (0 learned features besides K) R²=0.656, MAE=46.8m. (E) Mechanistic + Demo residual Ridge R²=0.711. Cadence alone explains ~47% of variance — confirms cadence is the dominant single gait signal for 6MWD, and the speed = cadence × stride-length identity (proxied by Height) is the right physics. Full Ridge (55f, R²=0.806) still best, but cadence + Demo(4) reaches 88% of full-model variance with 1/11 the features. Script `temp_exps/cadence_only_clinic.py` retained as the parsimonious-baseline reference.
- **Cadence-only home baselines (collapse vs clinic):** Same cadence-only protocol on `home_perbout_features.csv` (12 cadence stats: hz/power × 6 aggregations). Spearman ρ of `g_cadence_hz_med` with 6MWD is only +0.163 (vs clinic ~+0.6); `g_cadence_hz_max` ρ=+0.082 (peak home cadence is not a max-effort proxy). Results (Ridge α=20, LOO): cadence_med alone (1f) R²=−0.001; all 6 cadence_hz stats R²=+0.023; all 12 cadence stats R²=+0.024; cadence_med + Demo(4) (5f) R²=+0.363, MAE=61.5m — **statistically tied with Demo-only (R²=0.362)**, cadence adds ~zero information at home; mechanistic `K · cadence_med · Height · 360s` R²=+0.118 (vs 0.656 clinic). The signal that works at home is **duration + intensity** (`g_duration_sec_max` ρ=0.375, `act_pct_vigorous` ρ=0.370), not cadence — at home people walk at context-dependent speeds (kitchen, errands) and never push to max effort, so median cadence reflects habit, not capacity. Sharpens the cross-setting framing: **clinic = cadence-driven (constrained max-effort 6MWT), home = duration/intensity-driven (free-living context)**. Script `temp_exps/cadence_only_home.py` retained as companion to the clinic parsimonious baseline.
