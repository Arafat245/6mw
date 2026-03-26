#!/usr/bin/env python3
"""
Reproduce best results for A and C series (n=102).
Uses reproduce_c2.py for clinic features (not features13.csv which has 101).
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
ids = pd.read_csv(BASE / 'feats/target_6mwd.csv')
valid = np.ones(len(ids), dtype=bool)
valid[ids[(ids['cohort']=='M') & (ids['subj_id']==22)].index] = False
ids102 = ids[valid].reset_index(drop=True)
y = ids102['sixmwd'].values.astype(float)
groups = ids102['cohort'].values
n = len(y)

print(f"Total subjects (excl M22): {n} (MS={sum(groups=='M')}, Control={sum(groups=='C')})")
print("Note: M44 has too-short clinic recording (601 samples), excluded from clinic features")
print("  C series: n=102 (uses all with valid preprocessed files)")
print("  A series: n=101 (overlap of home + clinic features for PLS)")

# ══════════════════════════════════════════════════════════════════
# DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════
demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
p = ids102.merge(demo, on=['cohort', 'subj_id'], how='left')
p['cohort_M'] = (p['cohort'] == 'M').astype(int)
for c in ['Age', 'Sex', 'Height']:
    p[c] = pd.to_numeric(p[c], errors='coerce')
X_demo_4 = p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float)
X_demo_3 = p[['cohort_M', 'Age', 'Sex']].values.astype(float)
for X in [X_demo_4, X_demo_3]:
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        if m.any(): X[m, j] = np.nanmedian(X[:, j])

# ══════════════════════════════════════════════════════════════════
# CLINIC FEATURES: Extract Gait13 + CWT from reproduce_c2.py pipeline
# ══════════════════════════════════════════════════════════════════
print("\nExtracting clinic features from csv_preprocessed2 and csv_raw2...")
import sys
sys.path.insert(0, str(BASE))
from reproduce_c2 import (
    extract_gait10, compute_vt_rms, add_sway_ratios, extract_cwt,
    _filename_re, PreprocConfig
)
import re

PREPROC2 = BASE / 'csv_preprocessed2'
RAW = BASE / 'csv_raw2'

# Clinic Gait13
clinic_gait_rows = []
clinic_valid = []
for i, (_, r) in enumerate(ids102.iterrows()):
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    pp = PREPROC2 / fn
    if pp.exists():
        clinic_gait_rows.append(extract_gait10(pd.read_csv(pp)))
        clinic_valid.append(True)
    else:
        clinic_valid.append(False)

clinic_valid = np.array(clinic_valid)
print(f"  Clinic Gait10: {sum(clinic_valid)}/{n}")

# VT RMS
vt_rms_df = compute_vt_rms(PREPROC2)
gait_df = pd.DataFrame(clinic_gait_rows)
gait_with_meta = pd.concat([ids102[clinic_valid].reset_index(drop=True), gait_df], axis=1)
merged_sway = gait_with_meta.merge(vt_rms_df, on=['cohort', 'subj_id', 'sixmwd'], how='left')
sway_df = add_sway_ratios(merged_sway)

sway_cols = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
             'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
             'cadence_slope_per_min', 'vt_rms_g', 'ml_over_enmo', 'ml_over_vt']
X_gait13_c = sway_df[sway_cols].values.astype(float)
for j in range(X_gait13_c.shape[1]):
    m = np.isnan(X_gait13_c[:, j])
    if m.any(): X_gait13_c[m, j] = np.nanmedian(X_gait13_c[:, j])

# Clinic CWT
print("  Extracting CWT from csv_raw2...")
cwt_rows = []
for i, (_, r) in enumerate(ids102[clinic_valid].iterrows()):
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    raw = pd.read_csv(RAW / fn, usecols=['X', 'Y', 'Z']).values.astype(np.float32)
    cwt_rows.append(extract_cwt(raw))
cwt_df = pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan)
for c in cwt_df.columns:
    if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(cwt_df[c].median())
X_cwt_c = cwt_df.values.astype(float)
print(f"  Clinic Gait13: {X_gait13_c.shape}, CWT: {X_cwt_c.shape}")

# Adjust to only valid clinic subjects
y_c = y[clinic_valid]
groups_c = groups[clinic_valid]
X_demo_4_c = X_demo_4[clinic_valid]
n_c = len(y_c)

# ══════════════════════════════════════════════════════════════════
# HOME FEATURES: from saved npz
# ══════════════════════════════════════════════════════════════════
d = np.load(BASE / 'feats/home_hybrid_v2_features.npz', allow_pickle=True)
X_gait_home = d['X_gait']
X_clinic_10_for_pls = X_gait13_c[:, :10]  # first 10 features for PLS target

# Match home subjects to clinic valid subjects
home_idx = np.where(clinic_valid)[0]
X_gait_home_matched = X_gait_home[home_idx]
X_demo_3_matched = X_demo_3[home_idx]

# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════

def report(name, y_true, y_pred, grp):
    ms = grp == 'M'; ctrl = grp == 'C'
    r2 = r2_score(y_true, y_pred); mae = mean_absolute_error(y_true, y_pred)
    r2_ms = r2_score(y_true[ms], y_pred[ms]); mae_ms = mean_absolute_error(y_true[ms], y_pred[ms])
    r2_ctrl = r2_score(y_true[ctrl], y_pred[ctrl]); mae_ctrl = mean_absolute_error(y_true[ctrl], y_pred[ctrl])
    print(f"\n{name}:")
    print(f"  All:     R²={r2:.4f}  MAE={mae:.0f}  (n={len(y_true)})")
    print(f"  MS:      R²={r2_ms:.4f}  MAE={mae_ms:.0f}  (n={ms.sum()})")
    print(f"  Control: R²={r2_ctrl:.4f}  MAE={mae_ctrl:.0f}  (n={ctrl.sum()})")

# ── C1: Gait13+CWT, Ridge(10) ──
X_c1 = np.column_stack([X_gait13_c, X_cwt_c])
pr = np.zeros(n_c)
for tr, te in LeaveOneOut().split(X_c1):
    sc = StandardScaler(); m = Ridge(alpha=10)
    m.fit(sc.fit_transform(X_c1[tr]), y_c[tr]); pr[te] = m.predict(sc.transform(X_c1[te]))
report("C1: Gait13+CWT (41f), Ridge(10)", y_c, pr, groups_c)

# ── C2: Gait13+CWT+Demo(4), Ridge(10) ──
X_c2 = np.column_stack([X_gait13_c, X_cwt_c, X_demo_4_c])
pr = np.zeros(n_c)
for tr, te in LeaveOneOut().split(X_c2):
    sc = StandardScaler(); m = Ridge(alpha=10)
    m.fit(sc.fit_transform(X_c2[tr]), y_c[tr]); pr[te] = m.predict(sc.transform(X_c2[te]))
report("C2: Gait13+CWT+Demo (45f), Ridge(10)", y_c, pr, groups_c)

# ── A1: Home Gait13 top8, RF ──
corrs = [abs(spearmanr(X_gait_home_matched[:, j], y_c)[0]) for j in range(13)]
top8 = np.argsort(corrs)[::-1][:8]
X_a1 = X_gait_home_matched[:, top8]
pr = np.zeros(n_c)
for tr, te in LeaveOneOut().split(X_a1):
    sc = StandardScaler()
    m = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
    m.fit(sc.fit_transform(X_a1[tr]), y_c[tr]); pr[te] = m.predict(sc.transform(X_a1[te]))
report("A1: Gait13 top8 (8f), RF", y_c, pr, groups_c)

# ── A2: PLS(2) home13→clinic + Demo(3), Ridge(20) ──
pr = np.zeros(n_c)
for te in range(n_c):
    tr = np.ones(n_c, dtype=bool); tr[te] = False
    sh = StandardScaler(); sc = StandardScaler(); sd = StandardScaler()
    Xht = sh.fit_transform(X_gait_home_matched[tr]); Xhe = sh.transform(X_gait_home_matched[te:te+1])
    Xct = sc.fit_transform(X_clinic_10_for_pls[tr])
    Xdt = sd.fit_transform(X_demo_3_matched[tr]); Xde = sd.transform(X_demo_3_matched[te:te+1])
    pls = PLSRegression(n_components=2, scale=False); pls.fit(Xht, Xct)
    Xhm = pls.transform(Xht); Xhem = pls.transform(Xhe)
    Xf = np.column_stack([Xhm, Xdt]); Xfe = np.column_stack([Xhem, Xde])
    m = Ridge(alpha=20); m.fit(Xf, y_c[tr]); pr[te] = m.predict(Xfe)[0]
report("A2: PLS(2)+Demo (5f), Ridge(20)", y_c, pr, groups_c)
