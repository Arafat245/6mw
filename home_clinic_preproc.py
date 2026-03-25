#!/usr/bin/env python3
"""
Re-extract ALL home features using clinic-style preprocessing
(Rodrigues rotation + bandpass filter) and compare with original.
"""
import numpy as np, pandas as pd, warnings, sys
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from reproduce_c2 import (extract_gait10, compute_vt_rms, add_sway_ratios,
                           extract_cwt, align_to_ap_ml_vt, PreprocConfig)
from extract_walking_sway import extract_walking_sway
from preprocess_raw import detect_walking_bouts

BASE = Path('.')
FS = 30.0
HOME = BASE / 'csv_home_daytime'
PREPROC2 = BASE / 'csv_preprocessed2'
RAW = BASE / 'csv_raw2'
cfg = PreprocConfig()

ids = pd.read_csv('feats/target_6mwd.csv')
valid = np.ones(len(ids), dtype=bool)
valid[ids[(ids['cohort']=='M') & (ids['subj_id']==22)].index] = False
ids102 = ids[valid].reset_index(drop=True)

clinic_valid = []
for _, r in ids102.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    clinic_valid.append((PREPROC2/fn).exists())
clinic_valid = np.array(clinic_valid)
cidx = np.where(clinic_valid)[0]
subj = ids102[clinic_valid].reset_index(drop=True)
y = subj['sixmwd'].values.astype(float)
n = len(y)

# ══════════════════════════════════════════════════════════════════
# STEP 1: Detect walking bouts & apply clinic preprocessing
# ══════════════════════════════════════════════════════════════════
BOUT_CACHE = BASE / 'feats' / 'home_walking_bout_indices.npz'

# Step 0: Detect walking bouts (or load cached)
if BOUT_CACHE.exists():
    print("Loading cached walking bout boundaries...")
    bout_data = np.load(BOUT_CACHE, allow_pickle=True)
    all_bouts = bout_data['bouts'].item()  # dict: filename -> list of (start, end)
else:
    print(f"Detecting walking bouts for {n} subjects (will cache for future)...")
    all_bouts = {}
    for idx, (_, r) in enumerate(subj.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        hp = HOME / fn
        if not hp.exists():
            all_bouts[fn] = []
            continue
        xyz = pd.read_csv(hp, usecols=['X','Y','Z']).values.astype(np.float64)
        bouts = detect_walking_bouts(xyz, FS)
        all_bouts[fn] = bouts
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{n} ({len(bouts)} bouts)", flush=True)
    print(f"  {n}/{n} done")
    np.savez(BOUT_CACHE, bouts=all_bouts)
    print(f"  Cached bout boundaries to {BOUT_CACHE}")

print(f"\nStep 1: Applying clinic preprocessing to walking bouts (n={n})...")

home_gait_rows = []
home_ws_rows = []
home_cwt_rows = []

for idx, (_, r) in enumerate(subj.iterrows()):
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    hp = HOME / fn
    bouts = all_bouts.get(fn, [])

    if not hp.exists() or not bouts:
        home_gait_rows.append(None)
        home_ws_rows.append(None)
        home_cwt_rows.append(None)
        continue

    xyz = pd.read_csv(hp, usecols=['X','Y','Z']).values.astype(np.float64)

    bout_gait = []
    bout_ws = []
    bout_cwt = []

    for s, e in bouts[:20]:
        seg = xyz[s:e]
        if len(seg) < int(5 * FS):
            continue

        # Clinic preprocessing
        apmlvt_dyn, g_est = align_to_ap_ml_vt(seg, fs=FS, cfg=cfg)
        lo, hi = cfg.step_band_hz
        b, a = butter(N=cfg.filter_order, Wn=[lo, hi], btype='bandpass', fs=FS)
        apmlvt_bp = np.column_stack([filtfilt(b, a, apmlvt_dyn[:, j]) for j in range(3)])
        vm_raw = np.linalg.norm(seg, axis=1)
        enmo = np.maximum(vm_raw - 1.0, 0.0)
        vm_dyn = np.linalg.norm(apmlvt_dyn, axis=1)

        bout_df = pd.DataFrame({
            'AP': apmlvt_dyn[:, 0], 'ML': apmlvt_dyn[:, 1], 'VT': apmlvt_dyn[:, 2],
            'AP_bp': apmlvt_bp[:, 0], 'ML_bp': apmlvt_bp[:, 1], 'VT_bp': apmlvt_bp[:, 2],
            'VM_dyn': vm_dyn, 'VM_raw': vm_raw, 'ENMO': enmo, 'fs': FS
        })

        try:
            bout_gait.append(extract_gait10(bout_df))
        except:
            pass

        try:
            bout_ws.append(extract_walking_sway(apmlvt_dyn[:, 0], apmlvt_dyn[:, 1], apmlvt_dyn[:, 2], FS))
        except:
            pass

        try:
            bout_cwt.append(extract_cwt(seg.astype(np.float32), fs=FS))
        except:
            pass

    home_gait_rows.append(pd.DataFrame(bout_gait).median().to_dict() if bout_gait else None)
    home_ws_rows.append(pd.DataFrame(bout_ws).median().to_dict() if bout_ws else None)
    home_cwt_rows.append(pd.DataFrame(bout_cwt).median().to_dict() if bout_cwt else None)

    if (idx + 1) % 20 == 0:
        print(f"  {idx+1}/{n} (bouts: {len(bouts)}, used gait: {len(bout_gait)})", flush=True)

print(f"  {n}/{n} done")

# ══════════════════════════════════════════════════════════════════
# STEP 2: Build feature matrices
# ══════════════════════════════════════════════════════════════════
gait_cols = ['cadence_hz','step_time_cv_pct','acf_step_regularity','hr_ap','hr_vt',
             'ml_rms_g','ml_spectral_entropy','jerk_mean_abs_gps','enmo_mean_g',
             'cadence_slope_per_min']

# New home gait (10 features) + vt_rms placeholder
def build_matrix(rows, cols):
    all_r = []
    for row in rows:
        if row is None:
            all_r.append({k: np.nan for k in cols})
        else:
            all_r.append({k: row.get(k, np.nan) for k in cols})
    df = pd.DataFrame(all_r)
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df.values.astype(float)

X_home_gait_new = build_matrix(home_gait_rows, gait_cols)

# WalkSway
ws_cols = list(pd.DataFrame([r for r in home_ws_rows if r is not None]).columns)
X_home_ws_new = build_matrix(home_ws_rows, ws_cols)

# CWT
cwt_cols_all = list(pd.DataFrame([r for r in home_cwt_rows if r is not None]).columns)
X_home_cwt_new = build_matrix(home_cwt_rows, cwt_cols_all)

print(f"New Home: Gait={X_home_gait_new.shape}, WalkSway={X_home_ws_new.shape}, CWT={X_home_cwt_new.shape}")

# ══════════════════════════════════════════════════════════════════
# STEP 3: Load clinic features + old home features for comparison
# ══════════════════════════════════════════════════════════════════

# Clinic Gait10
gait_rows_c = []
for _, r in subj.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    gait_rows_c.append(extract_gait10(pd.read_csv(PREPROC2/fn)))
X_clinic_gait10 = pd.DataFrame(gait_rows_c)[gait_cols].values.astype(float)
for j in range(X_clinic_gait10.shape[1]):
    m = np.isnan(X_clinic_gait10[:, j])
    if m.any(): X_clinic_gait10[m, j] = np.nanmedian(X_clinic_gait10[:, j])

# Clinic CWT
cwt_rows_c = []
for _, r in subj.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    raw = pd.read_csv(RAW/fn, usecols=['X','Y','Z']).values.astype(np.float32)
    cwt_rows_c.append(extract_cwt(raw))
cwt_df_c = pd.DataFrame(cwt_rows_c).replace([np.inf,-np.inf], np.nan)
for c in cwt_df_c.columns:
    if cwt_df_c[c].isna().any(): cwt_df_c[c] = cwt_df_c[c].fillna(cwt_df_c[c].median())
X_clinic_cwt = cwt_df_c.values.astype(float)

# Clinic WalkSway
ws_rows_c = []
for _, r in subj.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    df = pd.read_csv(PREPROC2/fn)
    ws_rows_c.append(extract_walking_sway(df['AP'].values, df['ML'].values, df['VT'].values))
X_clinic_ws = pd.DataFrame(ws_rows_c).values.astype(float)
for j in range(X_clinic_ws.shape[1]):
    m = np.isnan(X_clinic_ws[:, j])
    if m.any(): X_clinic_ws[m, j] = np.nanmedian(X_clinic_ws[:, j])

# Old home features
d = np.load('feats/home_hybrid_v2_features.npz', allow_pickle=True)
X_home_gait_old = d['X_gait'][cidx, :10]

home_cwt_df = pd.read_csv('feats/home_cwt_hybrid.csv')
hcwt_cols = [c for c in home_cwt_df.columns if c not in ['cohort','subj_id','year','sixmwd']]
home_cwt_merged = subj.merge(home_cwt_df, on=['cohort','subj_id','sixmwd'], how='inner')
X_home_cwt_old = home_cwt_merged[hcwt_cols].values.astype(float)
for j in range(X_home_cwt_old.shape[1]):
    m = np.isnan(X_home_cwt_old[:, j])
    if m.any(): X_home_cwt_old[m, j] = np.nanmedian(X_home_cwt_old[:, j])

WALK_SEG = BASE / 'results_raw_pipeline' / 'walking_segments'
ws_home_old_rows = []
for _, r in subj.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    wp = WALK_SEG / fn
    if wp.exists():
        df2 = pd.read_csv(wp)
        ws_home_old_rows.append(extract_walking_sway(df2['AP'].values, df2['ML'].values, df2['VT'].values))
    else:
        ws_home_old_rows.append(None)
ws_old_cols = list(pd.DataFrame([r for r in ws_home_old_rows if r is not None]).columns)
home_ws_old_all = []
for row in ws_home_old_rows:
    if row is None: home_ws_old_all.append({k: np.nan for k in ws_old_cols})
    else: home_ws_old_all.append(row)
X_home_ws_old = pd.DataFrame(home_ws_old_all).values.astype(float)
for j in range(X_home_ws_old.shape[1]):
    m = np.isnan(X_home_ws_old[:, j])
    if m.any(): X_home_ws_old[m, j] = np.nanmedian(X_home_ws_old[:, j])

# Demo
demo = pd.read_excel('SwayDemographics.xlsx')
demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
p = subj.merge(demo, on=['cohort','subj_id'], how='left')
p['cohort_M'] = (p['cohort']=='M').astype(int)
for c in ['Age','Sex']: p[c] = pd.to_numeric(p[c], errors='coerce')
X_demo_3 = p[['cohort_M','Age','Sex']].values.astype(float)
for j in range(3):
    m = np.isnan(X_demo_3[:, j])
    if m.any(): X_demo_3[m, j] = np.nanmedian(X_demo_3[:, j])

# ══════════════════════════════════════════════════════════════════
# STEP 4: Run all configs — OLD vs NEW
# ══════════════════════════════════════════════════════════════════

def loo(X, y):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=10)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return round(r2_score(y, pr), 4)

def loo_pls(X_home, X_clinic, X_demo, y, nc=2):
    n = len(y); pr = np.zeros(n)
    has_demo = X_demo.shape[1] > 0 if len(X_demo.shape) > 1 else False
    for te in range(n):
        tr = np.ones(n, dtype=bool); tr[te] = False
        sh=StandardScaler(); sc=StandardScaler()
        Xht=sh.fit_transform(X_home[tr]); Xhe=sh.transform(X_home[te:te+1])
        Xct=sc.fit_transform(X_clinic[tr])
        pls=PLSRegression(n_components=nc, scale=False); pls.fit(Xht, Xct)
        Xhm=pls.transform(Xht); Xhem=pls.transform(Xhe)
        if has_demo:
            sd=StandardScaler()
            Xdt=sd.fit_transform(X_demo[tr]); Xde=sd.transform(X_demo[te:te+1])
            Xf=np.column_stack([Xhm,Xdt]); Xfe=np.column_stack([Xhem,Xde])
        else:
            Xf=Xhm; Xfe=Xhem
        m=Ridge(alpha=10); m.fit(Xf, y[tr]); pr[te]=m.predict(Xfe)[0]
    return round(r2_score(y, pr), 4)

print("\n" + "="*80)
print(f"{'Config':35s} {'Old Preproc':>12s} {'Clinic Preproc':>14s} {'Diff':>8s}")
print("="*80)

configs = [
    ("Gait", X_home_gait_old, X_home_gait_new),
    ("CWT", X_home_cwt_old, X_home_cwt_new),
    ("WalkSway", X_home_ws_old, X_home_ws_new),
    ("Gait+CWT+WalkSway+Demo",
     np.column_stack([X_home_gait_old, X_home_cwt_old, X_home_ws_old, X_demo_3]),
     np.column_stack([X_home_gait_new, X_home_cwt_new, X_home_ws_new, X_demo_3])),
]

for name, X_old, X_new in configs:
    r2_old = loo(X_old, y)
    r2_new = loo(X_new, y)
    diff = r2_new - r2_old
    print(f"  {name:33s} {r2_old:>12.4f} {r2_new:>14.4f} {diff:>+8.4f}")

# PLS configs
print()
pls_configs = [
    ("PLS(Gait)", X_home_gait_old, X_home_gait_new, X_clinic_gait10),
    ("PLS(Gait)+Demo", X_home_gait_old, X_home_gait_new, X_clinic_gait10),
]

for name, X_old, X_new, X_clinic in pls_configs:
    if "Demo" in name:
        r2_old = loo_pls(X_old, X_clinic, X_demo_3, y)
        r2_new = loo_pls(X_new, X_clinic, X_demo_3, y)
    else:
        r2_old = loo_pls(X_old, X_clinic, np.zeros((n,0)), y)
        r2_new = loo_pls(X_new, X_clinic, np.zeros((n,0)), y)
    diff = r2_new - r2_old
    print(f"  {name:33s} {r2_old:>12.4f} {r2_new:>14.4f} {diff:>+8.4f}")

print("="*80)
