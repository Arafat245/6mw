#!/usr/bin/env python3
"""
Generate all paper tables from cached features and data.

Output: results/paper_tables/*.csv  (9 tables)
Run:    python analysis/generate_paper_tables.py  (~1 min)
"""
import warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
FEATS = BASE / 'feats'
OUT = BASE / 'results' / 'paper_tables'
OUT.mkdir(parents=True, exist_ok=True)
FT2M = 0.3048


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_ridge_preds(X, y, alpha):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pr[te] = m.predict(sc.transform(X[te]))
    return pr


def loo_spearman_preds(X_accel, X_demo, y, K=20, alpha=20):
    n_accel = X_accel.shape[1]
    pr = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), dtype=bool); tr[i] = False
        corrs = [abs(spearmanr(X_accel[tr, j], y[tr])[0])
                 if np.std(X_accel[tr, j]) > 0 else 0 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:min(K, n_accel)]
        if X_demo is not None:
            X_tr = np.column_stack([X_accel[tr][:, top_k], X_demo[tr]])
            X_te = np.column_stack([X_accel[i:i+1][:, top_k], X_demo[i:i+1]])
        else:
            X_tr = X_accel[tr][:, top_k]
            X_te = X_accel[i:i+1][:, top_k]
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_tr), y[tr])
        pr[i] = m.predict(sc.transform(X_te))[0]
    return pr


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''


def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s = np.sqrt(((n1 - 1) * np.var(x1, ddof=1) + (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / s if s > 0 else 0


def bh_correct(pvals):
    n = len(pvals)
    idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[idx]
    adj = np.zeros(n)
    adj[n - 1] = sorted_p[n - 1]
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i + 1], sorted_p[i] * n / (i + 1))
    result = np.zeros(n)
    result[idx] = np.minimum(adj, 1.0)
    return result


def fmt(vals):
    v = pd.to_numeric(vals, errors='coerce').dropna()
    return f"{v.mean():.1f} \u00b1 {v.std():.1f}" if len(v) > 0 else '---'


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
t0 = time.time()
print("Loading data...")

subj_df = pd.read_csv(BASE / 'home_full_recording_npz' / '_subjects.csv')
y_ft = subj_df['sixmwd'].values.astype(float)
n = len(subj_df)
cohorts = subj_df['cohort'].values
is_poms = cohorts == 'M'
is_healthy = cohorts == 'C'

# Demographics
demo_xl = pd.read_excel(BASE / 'SwayDemographics.xlsx')
demo_xl['cohort'] = demo_xl['ID'].str.extract(r'^([A-Z])')[0]
demo_xl['subj_id'] = demo_xl['ID'].str.extract(r'(\d+)')[0].astype(int)
p = subj_df.merge(demo_xl, on=['cohort', 'subj_id'], how='left')
for c in ['Age', 'Sex', 'Height', 'Weight', 'BMI', 'EDSS Total', 'MS Dur',
          'MFIS Total', 'MFIS Phys', 'MFIS Cog', 'MFIS Psych', 'BDI Raw Score']:
    p[c] = pd.to_numeric(p[c], errors='coerce')
p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)

X_demo_bmi = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))
X_demo_height = impute(p[['cohort_POMS', 'Age', 'Sex', 'Height']].values.astype(float))

# Clinic features
c_gait_df = pd.read_csv(FEATS / 'clinic_gait_features.csv')
c_cwt_df = pd.read_csv(FEATS / 'clinic_cwt_features.csv')
c_ws_df = pd.read_csv(FEATS / 'clinic_walksway_features.csv')
c_gait = impute(c_gait_df.drop(columns='key').values.astype(float))
c_cwt = impute(c_cwt_df.drop(columns='key').values.astype(float))
c_ws = impute(c_ws_df.drop(columns='key').values.astype(float))
c_gait_cols = list(c_gait_df.columns[1:])
c_cwt_cols = list(c_cwt_df.columns[1:])
c_ws_cols = list(c_ws_df.columns[1:])

# Clinic perbout features
c_pb = impute(pd.read_csv(FEATS / 'clinic_perbout_features.csv').drop(columns='key').values.astype(float))

# Home features
h_pb = impute(pd.read_csv(FEATS / 'home_perbout_features.csv').drop(columns='key').values.astype(float))
h_gait_df = pd.read_csv(FEATS / 'home_gait_features.csv')
h_cwt_df = pd.read_csv(FEATS / 'home_cwt_features.csv')
h_ws_df = pd.read_csv(FEATS / 'home_walksway_features.csv')
h_gait = impute(h_gait_df.drop(columns='key').values.astype(float))
h_cwt = impute(h_cwt_df.drop(columns='key').values.astype(float))
h_ws = impute(h_ws_df.drop(columns='key').values.astype(float))

# Home median columns for feature comparison with clinic
h_gait_med_cols = [c for c in h_gait_df.columns if c.endswith('_med')]
h_cwt_med_cols = [c for c in h_cwt_df.columns if c.endswith('_med')]
h_ws_med_cols = [c for c in h_ws_df.columns if c.endswith('_med')]
h_gait_med = impute(h_gait_df[h_gait_med_cols].values.astype(float))
h_cwt_med = impute(h_cwt_df[h_cwt_med_cols].values.astype(float))
h_ws_med = impute(h_ws_df[h_ws_med_cols].values.astype(float))


# ══════════════════════════════════════════════════════════════════
# COMPUTE LOO PREDICTIONS (shared by tables 3, 4, 8)
# ══════════════════════════════════════════════════════════════════
print("Computing LOO predictions...")

# Clinic best: Gait+CWT+WS+Demo(Height), Ridge alpha=5
X_clinic_all = np.column_stack([c_gait, c_cwt, c_ws, X_demo_height])
pred_clinic = loo_ridge_preds(X_clinic_all, y_ft, alpha=5)

# Home best: PerBout-Top20+Demo(BMI), Spearman inside LOO, Ridge alpha=20
pred_home = loo_spearman_preds(h_pb, X_demo_bmi, y_ft, K=20, alpha=20)

print(f"  Clinic R2={r2_score(y_ft, pred_clinic):.3f}  Home R2={r2_score(y_ft, pred_home):.3f}")


# ══════════════════════════════════════════════════════════════════
# TABLE 1: Demographics
# ══════════════════════════════════════════════════════════════════
print("\n1. demographics_table.csv")
poms_p = p[p['cohort'] == 'M']
healthy_p = p[p['cohort'] == 'C']

rows = []
# Continuous variables
for var, col in [('Age (years)', 'Age'), ('Height (cm)', 'Height'),
                 ('Weight (kg)', 'Weight'), ('BMI (kg/m2)', 'BMI')]:
    pv = poms_p[col].dropna(); hv = healthy_p[col].dropna()
    _, pval = mannwhitneyu(pv, hv, alternative='two-sided')
    rows.append({'Variable': var, f'All (n={n})': fmt(p[col]),
                 f'POMS (n={is_poms.sum()})': fmt(poms_p[col]),
                 f'Healthy (n={is_healthy.sum()})': fmt(healthy_p[col]),
                 'p-value': f'{pval:.4f}{sig_stars(pval)}'})

# Sex
sex_pm = int((poms_p['Sex'] == 1).sum()); sex_pf = int((poms_p['Sex'] == 2).sum())
sex_hm = int((healthy_p['Sex'] == 1).sum()); sex_hf = int((healthy_p['Sex'] == 2).sum())
_, pval_sex, _, _ = chi2_contingency([[sex_pm, sex_pf], [sex_hm, sex_hf]])
rows.insert(1, {'Variable': 'Sex (M/F)', f'All (n={n})': f'{sex_pm + sex_hm}/{sex_pf + sex_hf}',
                 f'POMS (n={is_poms.sum()})': f'{sex_pm}/{sex_pf}',
                 f'Healthy (n={is_healthy.sum()})': f'{sex_hm}/{sex_hf}',
                 'p-value': f'{pval_sex:.4f}'})

# 6MWD in meters
y_m = y_ft * FT2M
_, pval_6mwd = mannwhitneyu(y_m[is_poms], y_m[is_healthy], alternative='two-sided')
rows.append({'Variable': '6MWD (m)', f'All (n={n})': f'{y_m.mean():.1f} \u00b1 {y_m.std():.1f}',
             f'POMS (n={is_poms.sum()})': f'{y_m[is_poms].mean():.1f} \u00b1 {y_m[is_poms].std():.1f}',
             f'Healthy (n={is_healthy.sum()})': f'{y_m[is_healthy].mean():.1f} \u00b1 {y_m[is_healthy].std():.1f}',
             'p-value': f'{pval_6mwd:.4f}{sig_stars(pval_6mwd)}'})

# Clinical scores (POMS-only: EDSS, MS Dur)
for var, col in [('EDSS', 'EDSS Total'), ('Disease Duration (yrs)', 'MS Dur')]:
    rows.append({'Variable': var, f'All (n={n})': '---',
                 f'POMS (n={is_poms.sum()})': fmt(poms_p[col]),
                 f'Healthy (n={is_healthy.sum()})': '---', 'p-value': '---'})

# Clinical scores (both groups)
for var, col in [('MFIS Total', 'MFIS Total'), ('MFIS Physical', 'MFIS Phys'),
                 ('MFIS Cognitive', 'MFIS Cog'), ('MFIS Psychosocial', 'MFIS Psych'),
                 ('BDI', 'BDI Raw Score')]:
    pv = poms_p[col].dropna(); hv = healthy_p[col].dropna()
    if len(pv) > 0 and len(hv) > 0:
        _, pval = mannwhitneyu(pv, hv, alternative='two-sided')
        pstr = f'{pval:.4f}{sig_stars(pval)}'
    else:
        pstr = '---'
    rows.append({'Variable': var, f'All (n={n})': fmt(p[col]),
                 f'POMS (n={is_poms.sum()})': fmt(poms_p[col]),
                 f'Healthy (n={is_healthy.sum()})': fmt(healthy_p[col]),
                 'p-value': pstr})

pd.DataFrame(rows).to_csv(OUT / 'demographics_table.csv', index=False)
print(f"  Saved demographics_table.csv")


# ══════════════════════════════════════════════════════════════════
# TABLE 2: Feature descriptions (static)
# ══════════════════════════════════════════════════════════════════
print("\n2. feature_descriptions.csv")
pd.DataFrame([
    {'Category': 'Gait', 'Count': 11, 'Features': ', '.join(c_gait_cols)},
    {'Category': 'CWT', 'Count': 28, 'Features': ', '.join(c_cwt_cols)},
    {'Category': 'WalkSway', 'Count': 12, 'Features': ', '.join(c_ws_cols)},
    {'Category': 'Demo', 'Count': '3-4',
     'Features': 'cohort (POMS/Healthy), age, sex, height (clinic) or BMI (home)'},
]).to_csv(OUT / 'feature_descriptions.csv', index=False)
print(f"  Saved feature_descriptions.csv")


# ══════════════════════════════════════════════════════════════════
# TABLE 3: Best predictions
# ══════════════════════════════════════════════════════════════════
print("\n3. best_predictions.csv")
pd.DataFrame({
    'key': subj_df['key'].values, 'cohort': cohorts,
    'subj_id': subj_df['subj_id'].values,
    'sixmwd_actual_m': np.round(y_ft * FT2M, 1),
    'sixmwd_pred_home_m': np.round(pred_home * FT2M, 1),
    'sixmwd_pred_clinic_m': np.round(pred_clinic * FT2M, 1),
}).to_csv(OUT / 'best_predictions.csv', index=False)
print(f"  Saved best_predictions.csv")


# ══════════════════════════════════════════════════════════════════
# TABLE 4: Error analysis by cohort
# ══════════════════════════════════════════════════════════════════
print("\n4. error_analysis_by_cohort.csv")

err_rows = []
for model_name, pred in [('Home: PerBout-Top20+Demo(4)', pred_home),
                          ('Clinic: Gait+CWT+WalkSway+Demo', pred_clinic)]:
    for cohort_name, mask in [(f'All (n={n})', np.ones(n, dtype=bool)),
                               (f'POMS (n={is_poms.sum()})', is_poms),
                               (f'Healthy (n={is_healthy.sum()})', is_healthy)]:
        yt = y_ft[mask] * FT2M; yp = pred[mask] * FT2M
        rho, prho = spearmanr(yt, yp)
        err_rows.append({
            'Model': model_name, 'Cohort': cohort_name,
            'R2': round(r2_score(yt, yp), 3),
            'MAE (m)': round(mean_absolute_error(yt, yp), 1),
            'RMSE (m)': round(np.sqrt(np.mean((yt - yp) ** 2)), 1),
            'Spearman rho': f"{rho:.3f}{sig_stars(prho)}"
        })

pd.DataFrame(err_rows).to_csv(OUT / 'error_analysis_by_cohort.csv', index=False)
print(f"  Saved error_analysis_by_cohort.csv")


# ══════════════════════════════════════════════════════════════════
# TABLE 5: Feature correlations with 6MWD
# ══════════════════════════════════════════════════════════════════
print("\n5. feature_correlations.csv")

# Map home median column names to base clinic names
h_gait_base = [c.rsplit('_med', 1)[0] for c in h_gait_med_cols]
h_cwt_base = [c.rsplit('_med', 1)[0] for c in h_cwt_med_cols]
h_ws_base = [c.rsplit('_med', 1)[0] for c in h_ws_med_cols]

# All clinic features
all_clinic_X = np.column_stack([c_gait, c_cwt, c_ws])
all_clinic_names = c_gait_cols + c_cwt_cols + c_ws_cols
all_clinic_cats = ['Gait'] * len(c_gait_cols) + ['CWT'] * len(c_cwt_cols) + ['WalkSway'] * len(c_ws_cols)

# All home median features
all_home_X = np.column_stack([h_gait_med, h_cwt_med, h_ws_med])
all_home_names = h_gait_base + h_cwt_base + h_ws_base
all_home_cats = ['Gait'] * len(h_gait_base) + ['CWT'] * len(h_cwt_base) + ['WalkSway'] * len(h_ws_base)

# Find common features
common = sorted(set(all_clinic_names) & set(all_home_names))

columns = ['Home (All)', 'Clinic (All)', 'Home (POMS)', 'Home (Healthy)',
           'Clinic (POMS)', 'Clinic (Healthy)']
corr_rows = []
for feat in common:
    c_idx = all_clinic_names.index(feat)
    h_idx = all_home_names.index(feat)
    cat = all_clinic_cats[c_idx]
    row = {'Feature': f"{cat}: {feat.replace('_', ' ')}"}
    rhos = []
    for label, X, idx, mask in [
        ('Home (All)', all_home_X, h_idx, np.ones(n, bool)),
        ('Clinic (All)', all_clinic_X, c_idx, np.ones(n, bool)),
        ('Home (POMS)', all_home_X, h_idx, is_poms),
        ('Home (Healthy)', all_home_X, h_idx, is_healthy),
        ('Clinic (POMS)', all_clinic_X, c_idx, is_poms),
        ('Clinic (Healthy)', all_clinic_X, c_idx, is_healthy),
    ]:
        rho, pval = spearmanr(X[mask, idx], y_ft[mask])
        row[label] = f"{rho:+.2f}{sig_stars(pval)}"
        rhos.append(abs(rho))
    row['_max_rho'] = max(rhos)
    corr_rows.append(row)

# Add Demo: Age
row = {'Feature': 'Demo: Age'}
age = p['Age'].values.astype(float)
rhos = []
for label, mask in [('Home (All)', np.ones(n, bool)), ('Clinic (All)', np.ones(n, bool)),
                     ('Home (POMS)', is_poms), ('Home (Healthy)', is_healthy),
                     ('Clinic (POMS)', is_poms), ('Clinic (Healthy)', is_healthy)]:
    rho, pval = spearmanr(age[mask], y_ft[mask])
    row[label] = f"{rho:+.2f}{sig_stars(pval)}"
    rhos.append(abs(rho))
row['_max_rho'] = max(rhos)
corr_rows.append(row)

# Filter |rho| > 0.3, sort by max |rho|
corr_rows = [r for r in corr_rows if r['_max_rho'] > 0.3]
corr_rows.sort(key=lambda r: r['_max_rho'], reverse=True)
for r in corr_rows:
    del r['_max_rho']

pd.DataFrame(corr_rows).to_csv(OUT / 'feature_correlations.csv', index=False)
print(f"  Saved feature_correlations.csv ({len(corr_rows)} features)")


# ══════════════════════════════════════════════════════════════════
# TABLE 6: MS vs Healthy group differences
# ══════════════════════════════════════════════════════════════════
print("\n6. ms_vs_healthy_features.csv")

diff_rows = []
all_pvals = []

for setting, X, names, cats in [
    ('Clinic', all_clinic_X, all_clinic_names, all_clinic_cats),
    ('Home', all_home_X, all_home_names, all_home_cats)]:
    for j, (feat, cat) in enumerate(zip(names, cats)):
        poms_vals = X[is_poms, j]; healthy_vals = X[is_healthy, j]
        d = cohens_d(poms_vals, healthy_vals)
        _, pval = mannwhitneyu(poms_vals, healthy_vals, alternative='two-sided')
        all_pvals.append(pval)
        diff_rows.append({
            'Setting': setting, 'Category': cat, 'Feature': feat,
            'Cohen_d': round(d, 3), 'p_raw': pval
        })

# BH correction across all tests
p_adj = bh_correct(np.array(all_pvals))
for i, row in enumerate(diff_rows):
    row['p_adj'] = p_adj[i]

# Filter: p_adj < 0.05 and |d| > 0.5, sort by |d|
diff_rows = [r for r in diff_rows if r['p_adj'] < 0.05 and abs(r['Cohen_d']) > 0.5]
diff_rows.sort(key=lambda r: (r['Setting'], -abs(r['Cohen_d'])))
for r in diff_rows:
    r['p_adj'] = f"{r['p_adj']:.4f}{sig_stars(r['p_adj'])}"
    del r['p_raw']

pd.DataFrame(diff_rows).to_csv(OUT / 'ms_vs_healthy_features.csv', index=False)
print(f"  Saved ms_vs_healthy_features.csv ({len(diff_rows)} features)")


# ══════════════════════════════════════════════════════════════════
# TABLE 7 & 8: Clinical score correlations (POMS only)
# ══════════════════════════════════════════════════════════════════
print("\n7. clinical_corr_ms_only.csv (clinic features)")
print("8. clinical_corr_ms_home.csv (home features)")

clinical_scores = {
    'EDSS': p.loc[is_poms, 'EDSS Total'].values.astype(float),
    'MFIS Total': p.loc[is_poms, 'MFIS Total'].values.astype(float),
    'MFIS Phys': p.loc[is_poms, 'MFIS Phys'].values.astype(float),
    'MFIS Cog': p.loc[is_poms, 'MFIS Cog'].values.astype(float),
    'MFIS Psych': p.loc[is_poms, 'MFIS Psych'].values.astype(float),
    'BDI': p.loc[is_poms, 'BDI Raw Score'].values.astype(float),
    'POMS Dur': p.loc[is_poms, 'MS Dur'].values.astype(float),
}
score_names_clinic = ['EDSS', 'MFIS Total', 'MFIS Phys', 'MFIS Cog', 'MFIS Psych', 'BDI', 'POMS Dur']
score_names_home = ['EDSS', 'MFIS Total', 'MFIS Phys', 'MFIS Cog', 'MFIS Psych', 'BDI']


def clinical_corr_table(X, feat_names, feat_cats, score_names, mask):
    """Compute feature-clinical score correlations for MS only."""
    X_ms = X[mask]
    rows = []
    for j, (feat, cat) in enumerate(zip(feat_names, feat_cats)):
        row = {'Feature': f"{cat}: {feat.replace('_', ' ')}"}
        max_sig = 1.0
        for sname in score_names:
            scores = clinical_scores[sname]
            valid = ~(np.isnan(X_ms[:, j]) | np.isnan(scores))
            if valid.sum() < 5:
                row[sname] = '---'
                continue
            rho, pval = spearmanr(X_ms[valid, j], scores[valid])
            row[sname] = f"{rho:+.2f}{sig_stars(pval)}"
            max_sig = min(max_sig, pval)
        row['_min_p'] = max_sig
        rows.append(row)
    # Filter: at least one p < 0.05
    rows = [r for r in rows if r['_min_p'] < 0.05]
    rows.sort(key=lambda r: r['_min_p'])
    for r in rows:
        del r['_min_p']
    return rows


clinic_corr = clinical_corr_table(all_clinic_X, all_clinic_names, all_clinic_cats,
                                   score_names_clinic, is_poms)
pd.DataFrame(clinic_corr).to_csv(OUT / 'clinical_corr_ms_only.csv', index=False)
print(f"  Saved clinical_corr_ms_only.csv ({len(clinic_corr)} features)")

home_corr = clinical_corr_table(all_home_X, all_home_names, all_home_cats,
                                 score_names_home, is_poms)
pd.DataFrame(home_corr).to_csv(OUT / 'clinical_corr_ms_home.csv', index=False)
print(f"  Saved clinical_corr_ms_home.csv ({len(home_corr)} features)")


# ══════════════════════════════════════════════════════════════════
# TABLE 9: Results table (all feature sets, clinic + home)
# ══════════════════════════════════════════════════════════════════
print("\n9. results_table_final.csv")

RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)


def loo_ridge_metrics(X, y_ft, alpha):
    pr = loo_ridge_preds(X, y_ft, alpha)
    ym = y_ft * FT2M; pm = pr * FT2M
    return r2_score(ym, pm), mean_absolute_error(ym, pm), spearmanr(y_ft, pr)[0]


def loo_spearman_metrics(X_accel, X_demo, y_ft, K=20, alpha=20):
    pr = loo_spearman_preds(X_accel, X_demo, y_ft, K, alpha)
    ym = y_ft * FT2M; pm = pr * FT2M
    return r2_score(ym, pm), mean_absolute_error(ym, pm), spearmanr(y_ft, pr)[0]


res_rows = []

# Row 1: Gait
cr2, cmae, crho = loo_ridge_metrics(c_gait, y_ft, alpha=5)
hr2, hmae, hrho = loo_spearman_metrics(h_gait, None, y_ft, K=11, alpha=5)
res_rows.append({'Feature Set': 'Gait', '#f': 11,
    'Clinic R2': round(cr2, 3), 'Clinic MAE (m)': round(cmae, 1), 'Clinic rho': round(crho, 3),
    'Home R2': round(hr2, 3), 'Home MAE (m)': round(hmae, 1), 'Home rho': round(hrho, 3)})
print(f'  Gait:              C R2={cr2:.3f}  H R2={hr2:.3f}')

# Row 2: CWT
cr2, cmae, crho = loo_ridge_metrics(c_cwt, y_ft, alpha=20)
hr2, hmae, hrho = loo_spearman_metrics(h_cwt, None, y_ft, K=11, alpha=20)
res_rows.append({'Feature Set': 'CWT', '#f': 28,
    'Clinic R2': round(cr2, 3), 'Clinic MAE (m)': round(cmae, 1), 'Clinic rho': round(crho, 3),
    'Home R2': round(hr2, 3), 'Home MAE (m)': round(hmae, 1), 'Home rho': round(hrho, 3)})
print(f'  CWT:               C R2={cr2:.3f}  H R2={hr2:.3f}')

# Row 3: WalkSway
cr2, cmae, crho = loo_ridge_metrics(c_ws, y_ft, alpha=5)
hr2, hmae, hrho = loo_spearman_metrics(h_ws, None, y_ft, K=11, alpha=5)
res_rows.append({'Feature Set': 'WalkSway', '#f': 12,
    'Clinic R2': round(cr2, 3), 'Clinic MAE (m)': round(cmae, 1), 'Clinic rho': round(crho, 3),
    'Home R2': round(hr2, 3), 'Home MAE (m)': round(hmae, 1), 'Home rho': round(hrho, 3)})
print(f'  WalkSway:          C R2={cr2:.3f}  H R2={hr2:.3f}')

# Row 4: Demo
cr2, cmae, crho = loo_ridge_metrics(X_demo_bmi, y_ft, alpha=20)
res_rows.append({'Feature Set': 'Demo', '#f': 4,
    'Clinic R2': round(cr2, 3), 'Clinic MAE (m)': round(cmae, 1), 'Clinic rho': round(crho, 3),
    'Home R2': round(cr2, 3), 'Home MAE (m)': round(cmae, 1), 'Home rho': round(crho, 3)})
print(f'  Demo:              C R2={cr2:.3f}  H R2={cr2:.3f}')

# Row 5: PerBout-Top20
cr2, cmae, crho = loo_spearman_metrics(c_pb, None, y_ft, K=20, alpha=5)
hr2, hmae, hrho = loo_spearman_metrics(h_pb, None, y_ft, K=20, alpha=20)
res_rows.append({'Feature Set': 'PerBout-Top20', '#f': 20,
    'Clinic R2': round(cr2, 3), 'Clinic MAE (m)': round(cmae, 1), 'Clinic rho': round(crho, 3),
    'Home R2': round(hr2, 3), 'Home MAE (m)': round(hmae, 1), 'Home rho': round(hrho, 3)})
print(f'  PerBout-Top20:     C R2={cr2:.3f}  H R2={hr2:.3f}')

# Row 6: PerBout-Top20+Demo
cr2, cmae, crho = loo_spearman_metrics(c_pb, X_demo_bmi, y_ft, K=20, alpha=20)
hr2, hmae, hrho = loo_spearman_metrics(h_pb, X_demo_bmi, y_ft, K=20, alpha=20)
res_rows.append({'Feature Set': 'PerBout-Top20+Demo', '#f': 24,
    'Clinic R2': round(cr2, 3), 'Clinic MAE (m)': round(cmae, 1), 'Clinic rho': round(crho, 3),
    'Home R2': round(hr2, 3), 'Home MAE (m)': round(hmae, 1), 'Home rho': round(hrho, 3)})
print(f'  PerBout-Top20+Demo: C R2={cr2:.3f}  H R2={hr2:.3f}')

# Row 7: Gait+CWT+WS+Demo
cr2, cmae, crho = loo_ridge_metrics(X_clinic_all, y_ft, alpha=5)
X_home_gcw = np.column_stack([h_gait, h_cwt, h_ws])
hr2, hmae, hrho = loo_spearman_metrics(X_home_gcw, X_demo_bmi, y_ft, K=20, alpha=20)
res_rows.append({'Feature Set': 'Gait+CWT+WS+Demo', '#f': 55,
    'Clinic R2': round(cr2, 3), 'Clinic MAE (m)': round(cmae, 1), 'Clinic rho': round(crho, 3),
    'Home R2': round(hr2, 3), 'Home MAE (m)': round(hmae, 1), 'Home rho': round(hrho, 3)})
print(f'  Gait+CWT+WS+Demo: C R2={cr2:.3f}  H R2={hr2:.3f}')

res_df = pd.DataFrame(res_rows)
res_df.to_csv(RESULTS / 'results_table_final.csv', index=False)
print(f"  Saved results/results_table_final.csv ({len(res_rows)} rows)")

print(f"\nAll 9 tables saved to {OUT}/ + results/")
print(f"Done in {time.time() - t0:.0f}s")
