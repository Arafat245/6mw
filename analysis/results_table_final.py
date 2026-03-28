#!/usr/bin/env python3
"""
Final results table.
Clinic: Gait(11)+CWT(28)+WalkSway(12)+Demo(4) from 6MWT, Ridge LOO CV.
Home: First 6 min of longest clinic-free bout + PerBout-Top20, Ridge LOO CV.
Home uses Demo(5) with BMI, Clinic uses Demo(4) without BMI.
n=101 (M22 and M44 excluded).
"""
import numpy as np, pandas as pd, warnings, sys
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════
# LOO with best alpha search
# ══════════════════════════════════════════════════════════════════
def loo_eval(X, y, alphas=[5, 10, 20, 50, 100]):
    """LOO CV with alpha search. Returns (R², MAE, Spearman ρ, best_alpha)."""
    best = (-999, 0, 0, 10)
    for a in alphas:
        pr = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=a)
            m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
        r2 = r2_score(y, pr)
        if r2 > best[0]:
            mae = mean_absolute_error(y, pr)
            rho = spearmanr(y, pr)[0]
            best = (r2, mae, rho, a)
    return best


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
ids101 = ids[~excl].reset_index(drop=True)

# Clinic-valid subjects
PREPROC2 = BASE / 'csv_preprocessed2'
RAW = BASE / 'csv_raw2'
clinic_valid = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                         for _, r in ids101.iterrows()])
ids_v = ids101[clinic_valid].reset_index(drop=True)
cidx = np.where(clinic_valid)[0]
y = ids_v['sixmwd'].values.astype(float)
n = len(y)

# ── Clinic features ──
from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios, extract_cwt
from clinic.extract_walking_sway import extract_walking_sway

# Clinic Gait (11)
gait_rows = []
for _, r in ids_v.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    gait_rows.append(extract_gait10(pd.read_csv(PREPROC2 / fn)))
vt_rms_df = compute_vt_rms(PREPROC2)
gdf = pd.DataFrame(gait_rows)
gm = pd.concat([ids_v.reset_index(drop=True), gdf], axis=1)
sway = add_sway_ratios(gm.merge(vt_rms_df, on=['cohort', 'subj_id', 'sixmwd'], how='left'))
gait_cols = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
             'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
             'cadence_slope_per_min', 'vt_rms_g']
X_clinic_gait = impute(sway[gait_cols].values.astype(float))

# Clinic CWT (28)
cwt_rows = []
for _, r in ids_v.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    raw = pd.read_csv(RAW / fn, usecols=['X', 'Y', 'Z']).values.astype(np.float32)
    cwt_rows.append(extract_cwt(raw))
X_clinic_cwt = impute(pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan).values.astype(float))

# Clinic WalkSway (12 = 10 normalized + 2 sway ratios)
ws_rows = []
for _, r in ids_v.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    df = pd.read_csv(PREPROC2 / fn)
    ws_rows.append(extract_walking_sway(df['AP'].values, df['ML'].values, df['VT'].values))
X_clinic_ws10 = impute(pd.DataFrame(ws_rows).values.astype(float))
clinic_sway_ratios = impute(sway[['ml_over_enmo', 'ml_over_vt']].values.astype(float))
X_clinic_ws = np.column_stack([X_clinic_ws10, clinic_sway_ratios])

# ── Home features from first 6 min of longest bout (cached from exp_longest_bout_segments.py) ──
home_cache = BASE / 'temporary_experiments' / 'clinic_free' / 'longest_bout_proper_cache.npz'
hc = np.load(home_cache, allow_pickle=True)
h_cfg = hc['home_configs'].item()['first6']
h_valid = np.array(h_cfg['valid'], dtype=bool)

# Build home Gait/CWT/WalkSway from cache
def build_from_cache(feat_list, valid_mask):
    df = pd.DataFrame([f for f, v in zip(feat_list, valid_mask) if v])
    return impute(df.replace([np.inf, -np.inf], np.nan).values.astype(float))

X_home_gait = build_from_cache(h_cfg['gait'], h_valid)
X_home_cwt = build_from_cache(h_cfg['cwt'], h_valid)
X_home_ws = build_from_cache(h_cfg['ws'], h_valid)

# ── PerBout-Top20 (clinic-free) ──
cf_npz = np.load(BASE / 'feats' / 'home_clinicfree_top20.npz')
X_perbout_all = cf_npz['X']  # (101, 20)
# Map to ids_v[h_valid] subjects
pb_idx = []
for idx in np.where(h_valid)[0]:
    r = ids_v.iloc[idx]
    match = ids101[(ids101['cohort'] == r['cohort']) & (ids101['subj_id'] == r['subj_id']) & (ids101['sixmwd'] == r['sixmwd'])]
    if len(match) > 0: pb_idx.append(match.index[0])
X_perbout = impute(X_perbout_all[pb_idx])

# ── Demographics ──
demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
p['cohort_M'] = (p['cohort'] == 'M').astype(int)
for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
X_demo4 = impute(p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float))
X_demo5 = impute(p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float))

# Align to subjects valid in BOTH home and clinic
# Clinic features use all ids_v, home features use h_valid subset
# Need common indexing
X_clinic_gait_v = X_clinic_gait[h_valid]
X_clinic_cwt_v = X_clinic_cwt[h_valid]
X_clinic_ws_v = X_clinic_ws[h_valid]
X_demo4_v = X_demo4[h_valid]
X_demo5_v = X_demo5[h_valid]
y_v = y[h_valid]
n_v = len(y_v)

print(f"n={n_v}, LOO CV, Ridge with alpha search [5,10,20,50,100]\n")
print(f"Home: First 6 min of longest clinic-free bout + PerBout-Top20, Demo(5)")
print(f"Clinic: Full 6MWT, Demo(4)\n")

# ══════════════════════════════════════════════════════════════════
# BUILD TABLE
# ══════════════════════════════════════════════════════════════════
results = []

def add(name, nf_h, nf_c, X_h, X_c):
    """Evaluate home and clinic, add row to results."""
    h_r2, h_mae, h_rho, h_a = loo_eval(X_h, y_v)
    if X_c is not None:
        c_r2, c_mae, c_rho, c_a = loo_eval(X_c, y_v)
    else:
        c_r2, c_mae, c_rho = np.nan, np.nan, np.nan

    results.append({
        'Feature Set': name, '#f': f"H:{nf_h}, C:{nf_c}" if nf_c else f"{nf_h}",
        'Home MAE (ft)': round(h_mae, 1), 'Home ρ': round(h_rho, 3), 'Home R²': round(h_r2, 4),
        'Clinic MAE (ft)': round(c_mae, 1) if np.isfinite(c_mae) else '—',
        'Clinic ρ': round(c_rho, 3) if np.isfinite(c_rho) else '—',
        'Clinic R²': round(c_r2, 4) if np.isfinite(c_r2) else '—',
    })

    c_str = f"R²={c_r2:.4f} MAE={c_mae:.0f}" if np.isfinite(c_r2) else "—"
    print(f"  {name:<35s}  Home: R²={h_r2:.4f} MAE={h_mae:.0f} ρ={h_rho:.3f}  |  Clinic: {c_str}", flush=True)

# Row 1: Gait
add('Gait', X_home_gait.shape[1], X_clinic_gait_v.shape[1],
    X_home_gait, X_clinic_gait_v)

# Row 2: CWT
add('CWT', X_home_cwt.shape[1], X_clinic_cwt_v.shape[1],
    X_home_cwt, X_clinic_cwt_v)

# Row 3: WalkSway
add('WalkSway', X_home_ws.shape[1], X_clinic_ws_v.shape[1],
    X_home_ws, X_clinic_ws_v)

# Row 4: PerBout-Top20 (home only)
add('PerBout-Top20', X_perbout.shape[1], 0,
    X_perbout, None)

# Row 5: Demo
add('Demo', 5, 4,
    X_demo5_v, X_demo4_v)

# Row 6: PerBout-Top20 + Demo (best home model)
add('PerBout-Top20+Demo', X_perbout.shape[1] + 5, 4,
    np.column_stack([X_perbout, X_demo5_v]),
    X_demo4_v)

# Row 7: Gait+CWT+WalkSway+Demo (best clinic model)
nf_h_all = X_home_gait.shape[1] + X_home_cwt.shape[1] + X_home_ws.shape[1] + 5
nf_c_all = X_clinic_gait_v.shape[1] + X_clinic_cwt_v.shape[1] + X_clinic_ws_v.shape[1] + 4
add('Gait+CWT+WalkSway+Demo', nf_h_all, nf_c_all,
    np.column_stack([X_home_gait, X_home_cwt, X_home_ws, X_demo5_v]),
    np.column_stack([X_clinic_gait_v, X_clinic_cwt_v, X_clinic_ws_v, X_demo4_v]))

# Save
rdf = pd.DataFrame(results)
rdf.to_csv(BASE / 'feats' / 'results_table_final.csv', index=False)
print(f"\n{'='*90}")
print(f"Saved feats/results_table_final.csv ({len(results)} rows)")
