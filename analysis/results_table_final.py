#!/usr/bin/env python3
"""
Final results table.
Clinic: Gait(11)+CWT(28)+WalkSway(12)+Demo(4) from 6MWT, Ridge LOO CV.
Home: PerBout-Top20+Demo(4) from full recording, Ridge LOO CV.
n=101 (M22 and M44 excluded).

Matches files by subject key (cohort+subj_id), not full filename,
to handle year mismatches between target_6mwd.csv and csv filenames.
"""
import numpy as np, pandas as pd, warnings, sys
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent


def loo_eval(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 10)
    for a in alphas:
        pr = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=a)
            m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
        r2 = r2_score(y, pr)
        if r2 > best[0]:
            best = (r2, mean_absolute_error(y, pr), spearmanr(y, pr)[0], a)
    return best


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def find_file(directory, cohort, subj_id):
    """Find CSV by subject key (handles year mismatches)."""
    key = f'{cohort}{int(subj_id):02d}'
    for f in directory.glob(f'{key}_*.csv'):
        return f
    return None


# ══════════════════════════════════════════════════════════════════
# LOAD DATA — all 101 subjects
# ══════════════════════════════════════════════════════════════════
ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
ids101 = ids[~excl].reset_index(drop=True)
y = ids101['sixmwd'].values.astype(float)
n = len(y)

PREPROC2 = BASE / 'csv_preprocessed2'
RAW = BASE / 'csv_raw2'

# ── Clinic features ──
from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios, extract_cwt
from clinic.extract_walking_sway import extract_walking_sway

# Clinic Gait (11)
gait_rows = []
for _, r in ids101.iterrows():
    fp = find_file(PREPROC2, r['cohort'], r['subj_id'])
    gait_rows.append(extract_gait10(pd.read_csv(fp)))
vt_rms_df = compute_vt_rms(PREPROC2)
gdf = pd.DataFrame(gait_rows)
gm = pd.concat([ids101.reset_index(drop=True), gdf], axis=1)
sway = add_sway_ratios(gm.merge(vt_rms_df, on=['cohort', 'subj_id', 'sixmwd'], how='left'))
gait_cols = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
             'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
             'cadence_slope_per_min', 'vt_rms_g']
X_clinic_gait = impute(sway[gait_cols].values.astype(float))

# Clinic CWT (28)
cwt_rows = []
for _, r in ids101.iterrows():
    fp = find_file(RAW, r['cohort'], r['subj_id'])
    raw = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float32)
    cwt_rows.append(extract_cwt(raw))
X_clinic_cwt = impute(pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan).values.astype(float))

# Clinic WalkSway (12 = 10 normalized + 2 sway ratios)
ws_rows = []
for _, r in ids101.iterrows():
    fp = find_file(PREPROC2, r['cohort'], r['subj_id'])
    df = pd.read_csv(fp)
    ws_rows.append(extract_walking_sway(df['AP'].values, df['ML'].values, df['VT'].values))
X_clinic_ws10 = impute(pd.DataFrame(ws_rows).values.astype(float))
clinic_sway_ratios = impute(sway[['ml_over_enmo', 'ml_over_vt']].values.astype(float))
X_clinic_ws = np.column_stack([X_clinic_ws10, clinic_sway_ratios])

# ── Home features (from cached home_clinicfree_features.csv) ──
home_feat_df = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
home_accel_cols = [c for c in home_feat_df.columns if c != 'key']
X_home_accel = impute(home_feat_df[home_accel_cols].values.astype(float))

# ── Demographics ──
demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
p = ids101.merge(demo, on=['cohort', 'subj_id'], how='left')
p['cohort_M'] = (p['cohort'] == 'M').astype(int)
for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
X_demo4_clinic = impute(p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float))  # clinic: no BMI
X_demo4_home = impute(p[['cohort_M', 'Age', 'Sex', 'BMI']].values.astype(float))  # home: with BMI

# ── Home PerBout-Top20 with Spearman inside LOO ──
def home_spearman_loo(X_accel, X_demo, y, K=20, alpha=20):
    n_accel = X_accel.shape[1]
    n_demo = X_demo.shape[1]
    X_all = np.column_stack([X_accel, X_demo])
    demo_idx = list(range(n_accel, n_accel + n_demo))
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X_all):
        corrs = [abs(spearmanr(X_all[tr, j], y[tr])[0]) if np.std(X_all[tr, j]) > 0 else 0
                 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K]
        selected = top_k + demo_idx
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_all[tr][:, selected]), y[tr])
        preds[te] = m.predict(sc.transform(X_all[te][:, selected]))
    r2 = r2_score(y, preds)
    return r2, mean_absolute_error(y, preds), spearmanr(y, preds)[0]


# ══════════════════════════════════════════════════════════════════
# BUILD TABLE
# ══════════════════════════════════════════════════════════════════
print(f"n={n}, LOO CV, Ridge\n")

results = []

def add_clinic(name, X, nf):
    r2, mae, rho, a = loo_eval(X, y)
    results.append({'Setting': 'Clinic', 'Features': f'{name} ({nf}f)',
                    'R²': round(r2, 4), 'MAE (ft)': round(mae), 'ρ': round(rho, 3)})
    print(f"  Clinic {name:25s} ({nf:2d}f)  R²={r2:.4f}  MAE={mae:.0f}  ρ={rho:.3f}")

add_clinic('Gait', X_clinic_gait, 11)
add_clinic('CWT', X_clinic_cwt, 28)
add_clinic('WalkSway', X_clinic_ws, 12)
add_clinic('Gait+CWT+WS+Demo', np.column_stack([X_clinic_gait, X_clinic_cwt, X_clinic_ws, X_demo4_clinic]), 55)

print()

# Home: Spearman Top-20 inside LOO + Demo(4)
r2, mae, rho = home_spearman_loo(X_home_accel, X_demo4_home, y)
results.append({'Setting': 'Home', 'Features': 'PerBout-Top20+Demo(4) (24f, Spearman inside LOO)',
                'R²': round(r2, 4), 'MAE (ft)': round(mae), 'ρ': round(rho, 3)})
print(f"  Home  PerBout-Top20+Demo(4)   (24f)  R²={r2:.4f}  MAE={mae:.0f}  ρ={rho:.3f}")

rdf = pd.DataFrame(results)
rdf.to_csv(BASE / 'feats' / 'results_table_final.csv', index=False)
print(f"\nSaved feats/results_table_final.csv ({len(results)} rows)")
