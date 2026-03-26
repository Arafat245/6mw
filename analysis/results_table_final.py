#!/usr/bin/env python3
"""
Final results table for all feature set combinations.
Gait: 11 features, CWT: 28, WalkSway: 12 (ENMO-normalized + sway ratios), Demo: 3/4
No feature selection — Ridge(α=10) LOO CV, n=101.
"""
import numpy as np, pandas as pd, warnings, sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent  # project root
ids = pd.read_csv('feats/target_6mwd.csv')
valid = np.ones(len(ids), dtype=bool)
valid[ids[(ids['cohort']=='M') & (ids['subj_id']==22)].index] = False
ids102 = ids[valid].reset_index(drop=True)

from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios, extract_cwt
from clinic.extract_walking_sway import extract_walking_sway

PREPROC2 = BASE / 'csv_preprocessed2'
RAW = BASE / 'csv_raw2'
WALK_SEG = BASE / 'results_raw_pipeline' / 'walking_segments'

# ══════════════════════════════════════════════════════════════════
# LOAD ALL FEATURES
# ══════════════════════════════════════════════════════════════════

# Clinic Gait (11 features)
gait_rows = []; clinic_valid = []
for _, r in ids102.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    if (PREPROC2/fn).exists():
        gait_rows.append(extract_gait10(pd.read_csv(PREPROC2/fn))); clinic_valid.append(True)
    else: clinic_valid.append(False)
clinic_valid = np.array(clinic_valid)
vt_rms_df = compute_vt_rms(PREPROC2)
gdf = pd.DataFrame(gait_rows)
gm = pd.concat([ids102[clinic_valid].reset_index(drop=True), gdf], axis=1)
sway = add_sway_ratios(gm.merge(vt_rms_df, on=['cohort','subj_id','sixmwd'], how='left'))
gait_cols = ['cadence_hz','step_time_cv_pct','acf_step_regularity','hr_ap','hr_vt',
             'ml_rms_g','ml_spectral_entropy','jerk_mean_abs_gps','enmo_mean_g',
             'cadence_slope_per_min','vt_rms_g']
X_clinic_gait = sway[gait_cols].values.astype(float)
for j in range(X_clinic_gait.shape[1]):
    m = np.isnan(X_clinic_gait[:,j])
    if m.any(): X_clinic_gait[m,j] = np.nanmedian(X_clinic_gait[:,j])
clinic_sway_ratios = sway[['ml_over_enmo','ml_over_vt']].values.astype(float)
for j in range(2):
    m = np.isnan(clinic_sway_ratios[:,j])
    if m.any(): clinic_sway_ratios[m,j] = np.nanmedian(clinic_sway_ratios[:,j])

# Clinic CWT (28 features)
cwt_rows = []
for _, r in ids102[clinic_valid].iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    raw = pd.read_csv(RAW/fn, usecols=['X','Y','Z']).values.astype(np.float32)
    cwt_rows.append(extract_cwt(raw))
cwt_df = pd.DataFrame(cwt_rows).replace([np.inf,-np.inf], np.nan)
for c in cwt_df.columns:
    if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(cwt_df[c].median())
X_clinic_cwt = cwt_df.values.astype(float)

# Clinic WalkSway (10 normalized + 2 sway ratios = 12 features)
ws_clinic_rows = []
for _, r in ids102[clinic_valid].iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    df = pd.read_csv(PREPROC2/fn)
    ws_clinic_rows.append(extract_walking_sway(df['AP'].values, df['ML'].values, df['VT'].values))
X_clinic_ws10 = pd.DataFrame(ws_clinic_rows).values.astype(float)
for j in range(X_clinic_ws10.shape[1]):
    m = np.isnan(X_clinic_ws10[:,j])
    if m.any(): X_clinic_ws10[m,j] = np.nanmedian(X_clinic_ws10[:,j])
X_clinic_ws = np.column_stack([X_clinic_ws10, clinic_sway_ratios])

# Home Gait (11 features) + sway ratios
cidx = np.where(clinic_valid)[0]
d = np.load('feats/home_hybrid_v2_features.npz', allow_pickle=True)
X_home_gait = d['X_gait'][cidx, :11]
home_sway_ratios = d['X_gait'][cidx, 11:13]

# Home CWT (28 features)
home_cwt_df = pd.read_csv('feats/home_cwt_hybrid.csv')
hcwt_cols = [c for c in home_cwt_df.columns if c not in ['cohort','subj_id','year','sixmwd']]
home_cwt_merged = ids102[clinic_valid].reset_index(drop=True).merge(
    home_cwt_df, on=['cohort','subj_id','sixmwd'], how='inner')
X_home_cwt = home_cwt_merged[hcwt_cols].values.astype(float)
for j in range(X_home_cwt.shape[1]):
    m = np.isnan(X_home_cwt[:,j])
    if m.any(): X_home_cwt[m,j] = np.nanmedian(X_home_cwt[:,j])

# Home WalkSway (10 normalized + 2 sway ratios = 12 features)
ws_home_rows = []
for _, r in ids102.iterrows():
    fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
    wp = WALK_SEG / fn
    if wp.exists():
        df = pd.read_csv(wp)
        ws_home_rows.append(extract_walking_sway(df['AP'].values, df['ML'].values, df['VT'].values))
    else:
        ws_home_rows.append(None)
ws_cols = list(pd.DataFrame(ws_clinic_rows).columns)
home_ws_all = []
for row in ws_home_rows:
    if row is None: home_ws_all.append({k: np.nan for k in ws_cols})
    else: home_ws_all.append(row)
X_home_ws10 = pd.DataFrame(home_ws_all).iloc[cidx].reset_index(drop=True).values.astype(float)
for j in range(X_home_ws10.shape[1]):
    m = np.isnan(X_home_ws10[:,j])
    if m.any(): X_home_ws10[m,j] = np.nanmedian(X_home_ws10[:,j])
X_home_ws = np.column_stack([X_home_ws10, home_sway_ratios])

# Demo
demo = pd.read_excel('SwayDemographics.xlsx')
demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
p = ids102[clinic_valid].reset_index(drop=True).merge(demo, on=['cohort','subj_id'], how='left')
p['cohort_M'] = (p['cohort']=='M').astype(int)
for c in ['Age','Sex','Height']: p[c] = pd.to_numeric(p[c], errors='coerce')
X_demo_4 = p[['cohort_M','Age','Sex','Height']].values.astype(float)
X_demo_3 = p[['cohort_M','Age','Sex']].values.astype(float)
for X in [X_demo_4, X_demo_3]:
    for j in range(X.shape[1]):
        m = np.isnan(X[:,j])
        if m.any(): X[m,j] = np.nanmedian(X[:,j])

# MOMENT
Ec_pca50 = PCA(n_components=50).fit_transform(np.load('feats/moment_clinic_raw.npy')[valid][cidx])
Eh_pca50 = PCA(n_components=50).fit_transform(np.load('feats/moment_home_raw.npy')[valid][cidx])

# LimuBERT
Ec_limu = np.load('results_raw_pipeline/emb_limubert_clinic.npy')[valid][cidx]
Eh_limu = np.load('results_raw_pipeline/emb_limubert_home.npy')[valid][cidx]

y = ids102[clinic_valid].reset_index(drop=True)['sixmwd'].values.astype(float)
n = len(y)


# ══════════════════════════════════════════════════════════════════
# LOO FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def loo(X, y):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=10)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return round(r2_score(y, pr), 4)

def loo_rf(X, y):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        m = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return round(r2_score(y, pr), 4)

def loo_pls(X_home, X_clinic, X_demo, y, nc=2, alpha=10):
    n = len(y); pr = np.zeros(n)
    has_demo = X_demo.shape[1] > 0 if len(X_demo.shape) > 1 else False
    for te in range(n):
        tr = np.ones(n, dtype=bool); tr[te] = False
        sh = StandardScaler(); sc = StandardScaler()
        Xht = sh.fit_transform(X_home[tr]); Xhe = sh.transform(X_home[te:te+1])
        Xct = sc.fit_transform(X_clinic[tr])
        pls = PLSRegression(n_components=nc, scale=False); pls.fit(Xht, Xct)
        Xhm = pls.transform(Xht); Xhem = pls.transform(Xhe)
        if has_demo:
            sd = StandardScaler()
            Xdt = sd.fit_transform(X_demo[tr]); Xde = sd.transform(X_demo[te:te+1])
            Xf = np.column_stack([Xhm, Xdt]); Xfe = np.column_stack([Xhem, Xde])
        else:
            Xf = Xhm; Xfe = Xhem
        m = Ridge(alpha=alpha); m.fit(Xf, y[tr]); pr[te] = m.predict(Xfe)[0]
    return round(r2_score(y, pr), 4)


# ══════════════════════════════════════════════════════════════════
# BUILD TABLE
# ══════════════════════════════════════════════════════════════════

results = []
def add(name, nf, home, clinic):
    results.append({'Feature Set': name, '# Features': nf, 'Home R²': home, 'Clinic R²': clinic})
    print(f"  {name:30s} {nf:>55s}  Home={str(home):>8s}  Clinic={str(clinic):>8s}", flush=True)

print(f"n={n}\n")
print("Computing results...\n")

# Individual
h_ridge = loo(X_home_gait, y); h_rf = loo_rf(X_home_gait, y)
c_ridge = loo(X_clinic_gait, y); c_rf = loo_rf(X_clinic_gait, y)
add('Gait', '11', max(h_ridge, h_rf), max(c_ridge, c_rf))

add('CWT', '28', loo(X_home_cwt, y), loo(X_clinic_cwt, y))
add('WalkSway', '12', loo(X_home_ws, y), loo(X_clinic_ws, y))
add('Demo', 'Home: 3, Clinic: 4', loo(X_demo_3, y), loo(X_demo_4, y))

# Gait+Demo
add('Gait+Demo', 'Home: 11+3=14, Clinic: 11+4=15',
    loo(np.column_stack([X_home_gait, X_demo_3]), y),
    loo(np.column_stack([X_clinic_gait, X_demo_4]), y))

# All four
add('Gait+CWT+WalkSway+Demo', 'Home: 11+28+12+3=54, Clinic: 11+28+12+4=55',
    loo(np.column_stack([X_home_gait, X_home_cwt, X_home_ws, X_demo_3]), y),
    loo(np.column_stack([X_clinic_gait, X_clinic_cwt, X_clinic_ws, X_demo_4]), y))

# PLS (home only)
add('PLS(Gait)', '2 PLS components',
    loo_pls(X_home_gait, X_clinic_gait, np.zeros((n,0)), y, nc=2), '—')
add('PLS(Gait)+Demo', '2 PLS + 3 Demo = 5',
    loo_pls(X_home_gait, X_clinic_gait, X_demo_3, y, nc=2), '—')
add('PLS(Gait+WalkSway)', '2 PLS components',
    loo_pls(np.column_stack([X_home_gait, X_home_ws]),
            np.column_stack([X_clinic_gait, X_clinic_ws]),
            np.zeros((n,0)), y, nc=2), '—')
add('PLS(Gait+WalkSway)+Demo', '2 PLS + 3 Demo = 5',
    loo_pls(np.column_stack([X_home_gait, X_home_ws]),
            np.column_stack([X_clinic_gait, X_clinic_ws]),
            X_demo_3, y, nc=2), '—')

# Foundation models
add('MOMENT PCA50', '50', loo(Eh_pca50, y), loo(Ec_pca50, y))
add('MOMENT PCA50+Demo', 'Home: 50+3=53, Clinic: 50+4=54',
    loo(np.column_stack([Eh_pca50, X_demo_3]), y),
    loo(np.column_stack([Ec_pca50, X_demo_4]), y))
nl = Ec_limu.shape[1]
add('LimuBERT', str(nl), loo(Eh_limu, y), loo(Ec_limu, y))
add('LimuBERT+Demo', f'Home: {nl}+3={nl+3}, Clinic: {nl}+4={nl+4}',
    loo(np.column_stack([Eh_limu, X_demo_3]), y),
    loo(np.column_stack([Ec_limu, X_demo_4]), y))

# Save
rdf = pd.DataFrame(results)
rdf.to_csv('feats/results_table_final.csv', index=False)
print(f"\n{'='*90}")
print(f"n={n}, LOO CV, Ridge(α=10)")
print(f"Saved feats/results_table_final.csv")
