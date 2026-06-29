# Mediolateral WalkSway Analysis — Clinic 6MWT (POMS vs Healthy)

Analysis of **mediolateral (ML) trunk-sway features** from the clinic 6-minute walk
test (6MWT), comparing pediatric-onset MS (POMS) with healthy controls, and relating
them to 6-minute walk distance (6MWD). This folder is **exploratory** and
**self-contained** — it is not part of the main `home/`–`clinic/`–`analysis/`
prediction pipeline.

## Key facts

- **Cohort: n = 120** (50 POMS + 70 Healthy) — the **full clinic cohort**, NOT the
  n=101 prediction set. It includes 19 subjects that have a 6MWT recording but no 6MWD
  ground-truth label in `feats/target_6mwd.csv` (valid for group-difference and
  within-cohort analyses, which need only cohort labels and the filename-encoded 6MWD).
- **2 subjects unavailable:** M23 and M44 lack preprocessed AP/ML/VT files because their
  raw 6MWT recordings are far too short (~17 s and ~10 s). The clinic preprocessor
  (`clinic/reproduce_c2.py:145`) skips any file with ≤ 1000 samples. So 122 raw → 120 usable.
- **Source data:** `csv_preprocessed2/*.csv` (AP/ML/VT, gravity-removed, axis-aligned,
  30 Hz). Cohort from the filename prefix (`M*`=POMS, `C*`=Healthy). **6MWD is encoded in
  the filename** (`KEY_YEAR_6MWD.csv`) and was verified identical to `feats/target_6mwd.csv`
  on all 101 labelled subjects (0 mismatches); reported in **meters** (×0.3048).
- **Demographics:** `SwayDemographics.xlsx` (`M-01`/`C-01` IDs). Sex==1 male, Sex==2 female
  (per `analysis/generate_paper_tables.py:230`); `Female = (Sex==2)`. All 120 have Age/Sex/BMI.

## The four significant WalkSway features

All are **intensity-invariant directional ratios** that divide out overall movement
magnitude, isolating the *direction* of sway (forward→lateral redistribution). All are
**dimensionless** and **higher in POMS**. `ap_range_norm` is also BH-significant but is
**excluded** (anti-intuitive "rigid-gait" feature, like the main paper).

| Feature (display) | Formula | Type | d | p_BH | Source |
|---|---|---|---|---|---|
| `ML_Over_ENMO` (`ml_over_enmo`) | RMS(ML) / ENMO | intensity-normalized ratio | +0.74 | 0.0031 ** | original |
| `ML_Over_VT` (`ml_over_vt`) | RMS(ML) / RMS(VT) | directional ratio | +0.53 | 0.0323 * | original |
| `ML_Energy_Frac` (`ml_energy_frac`) | var(ML) / (var(AP)+var(ML)+var(VT)) | tri-axial energy partition (time) | +0.51 | 0.0323 * | **new (this work)** |
| `ML_Spec_Frac` (`ml_spec_horiz_frac`) | P_ML / (P_ML + P_AP), 0.3–10 Hz | horizontal-plane spectral partition (freq) | +0.51 | 0.0323 * | **new (this work)** |

The two **new** features are *inspired by* (not copied from) the MS sway literature in
`sway_reference/`. Literature-standard *absolute* measures (sample entropy, frequency
dispersion, step regularity, log-dimensionless jerk, index of harmonicity) were all
non-significant — the intensity confound washes them out — which motivated the
intensity-invariant ratio designs.

## Statistical methods used

| Question | Test / model | Where |
|---|---|---|
| Cohort difference per feature | **Mann–Whitney U** (two-sided), **BH-FDR** over 14-feature family | violin figure, stats tables |
| Normality check (test justification) | **Shapiro–Wilk** per cohort | `mannwhitney_justification.txt` |
| Group difference adjusted for demographics | **OLS** (one value/subject): `feature ~ MS + Female + BMI_c + Age_c` | group-difference tables |
| Genuine repeated-measures version | **LME** (random intercept), epochs as repeats | Table-5 LME, trajectory |
| Within-walk trajectory | **LME**: `feature ~ MS + Time + Time² + BMI cat + Age + Female + MS×Time(+²)` | Table-3 analog + figure |
| Predicting 6MWD | **OLS**: `6MWD_m ~ ML_Over_ENMO + MS + Female + BMI + Age` | 6MWD regression |
| Feature ↔ 6MWD association | **Spearman ρ** | scatter figure |

**Test choice (`mannwhitney_justification.txt`):** Shapiro–Wilk shows `ML_Over_ENMO`
strongly non-normal in both cohorts (p≈0.001, right-skewed); the others are normal
within-cohort but non-normal pooled and are bounded ratios. A single rank-based
(Mann–Whitney) test across all features is the principled, robust choice; medians +
95% bootstrap CIs are reported (consistent with a rank test).

## Folder layout

```
sway/
├── README.md                          this file
├── mannwhitney_justification.txt       Shapiro–Wilk values + test justification
├── codes/                             reproducible scripts (run from project root)
│   ├── make_walksway_tables.py          14-feature MW+BH table -> table/walksway_*.csv
│   ├── make_walksway_excel.py           3-tab Excel (defs/results/refs)
│   ├── make_walksway_violins.py         3-panel violin (BH p) -> figures/
│   ├── make_walksway_6mwd_scatter.py    3-panel feature-vs-6MWD scatter (Spearman) -> figures/
│   ├── make_walksway_lme_table.py       OLS group-diff Table 5 (4 features) -> csv + tex
│   ├── make_walksway_table5_lme.py      genuine-LME Table 5 (epochs) -> csv
│   ├── make_walksway_trajectory.py      Table-3 + Figure-2 analog (ML_Over_ENMO) -> csv + figure
│   ├── make_6mwd_regression.py          OLS 6MWD ~ ML_Over_ENMO + demo -> csv + tex
│   ├── make_ml_over_enmo_ols.py         OLS ML_Over_ENMO ~ MS+demo (Table 5) -> csv + tex
│   └── make_mannwhitney_justification.py Shapiro–Wilk -> justification txt
├── table/
│   ├── walksway_significant_features.csv          4 significant features (median, CI, d, p_BH)
│   ├── walksway_all14_ms_vs_healthy.csv           full 14-feature MW+BH results
│   ├── walksway_significant_feature_definitions.xlsx  3 tabs: defs / results / references
│   ├── walksway_lme_group_differences.csv         OLS group-diff, 4 features (MS/Female/BMI/Age)
│   ├── walksway_group_differences.tex             ^ Brenton Table-5-style LaTeX
│   ├── walksway_table5_lme_group_differences.csv  genuine-LME version (epoch repeats)
│   ├── walksway_ML_Over_ENMO_trajectory_LME.csv   Table-3 analog (Time/Time²/MS×Time …)
│   ├── ml_over_enmo_ols.csv / .tex                 OLS: ML_Over_ENMO ~ MS+Female+BMI+Age
│   └── sixmwd_on_ml_over_enmo.csv / .tex           OLS: 6MWD ~ ML_Over_ENMO + demo
├── figures/
│   ├── walksway_significant_violins.png            3-panel violin, blue Healthy/orange POMS, BH p
│   ├── walksway_6mwd_scatter.png                   3-panel scatter vs 6MWD (m), Spearman ρ/p
│   └── walksway_ML_Over_ENMO_trajectory.png        Figure-2 analog: modeled trajectory by BMI cat
└── sway_reference/                    supporting papers + references.md
    ├── Huisinga2013_AnnBiomedEng_gait_variability_MS_accelerometry.pdf
    ├── Solomon2015_JNER_postural_sway_inertial_MS.pdf
    ├── Alkathiry2025_Heliyon_walking_sway_accelerometry_measures.pdf
    ├── brenton-et-al-2022-six-minute-walk-...-pediatric-onset.pdf   (Table 3/5 template)
    └── references.md
```

## Tables (what each shows)

1. **`walksway_significant_features.csv`** — the 4 significant features: POMS/Healthy
   median [95% bootstrap CI], Cohen's d, raw p, p_BH. (Source for the violin figure.)
2. **`walksway_all14_ms_vs_healthy.csv`** — same stats for all 14 WalkSway features.
3. **`walksway_significant_feature_definitions.xlsx`** — 3 tabs: *Feature definitions*
   (formulas/definitions/refs), *Results (120 subj)* (medians/CI/d/p), *References*
   (citations + clickable DOI/URL).
4. **`walksway_lme_group_differences.csv` / `walksway_group_differences.tex`** —
   Brenton Table-5-style **OLS** group differences for all 4 features:
   `feature ~ MS + Female + BMI_c + Age_c`. Headline: after adjustment, **MS is
   significant only for `ML_Over_ENMO`** (p=0.032); the other three are driven by
   **Female** → the raw difference is substantially **sex-confounded**.
5. **`walksway_table5_lme_group_differences.csv`** — the same as (4) but a **genuine LME**
   (random intercept; 6 within-walk epochs as repeated measures). Confirms the
   sex-confounding conclusion.
6. **`ml_over_enmo_ols.csv` / .tex** — single-outcome Table 5: `ML_Over_ENMO ~ MS +
   Female + BMI + Age`. **MS = +0.130, p=0.032** → POMS sway more per effort,
   independent of sex/BMI/age; BMI also significant (+0.010, p=0.042).
7. **`sixmwd_on_ml_over_enmo.csv` / .tex** — `6MWD_m ~ ML_Over_ENMO + MS + Female + BMI +
   Age` (R²=0.470). `ML_Over_ENMO` trends negative (−40.6 m/unit) but **not significant**
   (p=0.10) once demographics are in; MS (−51.7 m), Female (−64.3 m), BMI (−3.8 m), Age
   (+9.7 m/yr) all significant.
8. **`walksway_ML_Over_ENMO_trajectory_LME.csv`** — Brenton **Table 3 analog**: LME of the
   minute-by-minute (6-epoch) `ML_Over_ENMO` trajectory. Strong **U-shape** (Time p<0.001,
   Time² p<0.001) in controls; significant **MS×Time** (p=0.029) → POMS have a **flatter,
   less-modulated** sway trajectory.

## Figures

1. **`walksway_significant_violins.png`** — 3-panel violin (ML_Over_ENMO, ML_Over_VT,
   ML_Spec_Frac), blue Healthy / orange POMS, black median bar, BH-corrected p annotated.
2. **`walksway_6mwd_scatter.png`** — 3-panel scatter of each feature vs **6MWD (m)**,
   colored by cohort, linear trend, Spearman ρ/p. All negative; `ML_Over_ENMO` strongest
   (ρ=−0.42, p<0.001), `ML_Over_VT` (ρ=−0.22, p=0.019), `ML_Spec_Frac` (ρ=−0.18, p=0.045).
3. **`walksway_ML_Over_ENMO_trajectory.png`** — Brenton **Figure 2 analog**: modeled
   `ML_Over_ENMO` trajectory across the 6MWT by BMI category, Control vs POMS panels.

## Headline interpretation (for writing the paper)

- **Cross-sectionally**, POMS show greater mediolateral sway per unit effort
  (`ML_Over_ENMO`, BH p=0.003, d=0.74); this survives adjustment for sex/BMI/age
  (OLS MS p=0.032). The other ratios largely reflect **sex** once adjusted.
- **Within the walk**, the disease signature is a **flatter sway trajectory** (MS×Time
  p=0.029): controls dip-and-recover (U-shape), POMS stay elevated and flat — the sway
  analog of Brenton's "POMS fail to re-accelerate at the end of the 6MW."
- **Sway does not independently predict 6MWD** once demographics are included (p=0.10),
  so `ML_Over_ENMO` is best framed as an **MS-discriminating gait-quality marker**, not a
  6MWD surrogate.

## Reproduce

Run from the project root (`/mnt/sdb/arafat/6mw`). Independent except where noted:

```bash
python3 sway/codes/make_walksway_tables.py        # -> table/walksway_{significant,all14}*.csv
python3 sway/codes/make_walksway_excel.py         # needs walksway_significant_features.csv
python3 sway/codes/make_walksway_violins.py       # -> figures/walksway_significant_violins.png
python3 sway/codes/make_walksway_6mwd_scatter.py  # -> figures/walksway_6mwd_scatter.png
python3 sway/codes/make_walksway_lme_table.py     # -> table/walksway_group_differences.{csv,tex}
python3 sway/codes/make_walksway_table5_lme.py    # -> table/walksway_table5_lme_group_differences.csv
python3 sway/codes/make_walksway_trajectory.py    # -> table + figures (trajectory)
python3 sway/codes/make_6mwd_regression.py        # -> table/sixmwd_on_ml_over_enmo.{csv,tex}
python3 sway/codes/make_ml_over_enmo_ols.py       # -> table/ml_over_enmo_ols.{csv,tex}
python3 sway/codes/make_mannwhitney_justification.py  # -> mannwhitney_justification.txt
```

Dependencies: numpy, pandas, scipy, seaborn, matplotlib, statsmodels, openpyxl. Scripts
read `csv_preprocessed2/` + `SwayDemographics.xlsx` and reuse clinic feature functions
(`clinic/reproduce_c2.py`, `clinic/extract_walking_sway.py`). LaTeX tables need
`\usepackage{booktabs}` (the 4-feature one also `graphicx` for `\resizebox`). Close the
Excel in LibreOffice before re-running `make_walksway_excel.py` (look for a `.~lock.*#`).

## Caveats (read before putting in the paper)

- **n = 120 ≠ n = 101.** Derived on the full clinic cohort; recompute on n=101 for
  consistency with the main prediction paper.
- **Post-hoc feature search.** ~14 candidates were tried; BH corrects over the final
  14-feature family, not the full search space — report the complete candidate list.
- **Sex confounding.** Three of four features' raw differences are largely sex-driven;
  only `ML_Over_ENMO` carries an adjusted MS effect.
- **Redundancy.** The four features encode one construct (lateral dominance) and are
  collinear; treat as a robustness set, not independent findings.
- **Epochs are equal sextiles** (~57 s), not strict clock-minutes (the walk is edge-trimmed);
  the trajectory figure uses a male, mean-age reference like Brenton's.
