#!/usr/bin/env python3
"""
Model-comparison table: Ridge (best), Lasso, ElasticNet, KNN, SVR, RF, XGBoost
on the best feature set for each setting, with 95% bootstrap CIs on (R², MAE, r).

Feature sets / protocols (identical to the headline pipelines):
  - Clinic: Gait(11) + CWT(28) + WalkSway(12) + Demo(4=Height) = 55 features,
            no feature selection, Ridge α=5 / others best hyperparameters.
  - Home:   152-feature Bout+Act pool + Demo(4=BMI), Spearman Top-20 inside
            each LOO fold + Demo(4) = 24 features fed to the model,
            Ridge α=20 / others best hyperparameters.

Inputs:
  feats/clinic_{gait,cwt,walksway}_features.csv, feats/home_perbout_features.csv,
  SwayDemographics.xlsx, home_full_recording_npz/_subjects.csv
Outputs:
  results/paper_tables/model_comparison.csv  (auto-mirrored to POMS/tables/)

Run: python analysis/model_comparison_table.py
"""
import warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent.parent
FEATS = BASE / 'feats'
NPZ_DIR = BASE / 'home_full_recording_npz'
OUT = BASE / 'results' / 'paper_tables'; OUT.mkdir(parents=True, exist_ok=True)
POMS_TABLES = BASE / 'POMS' / 'tables'; POMS_TABLES.mkdir(parents=True, exist_ok=True)
FT2M = 0.3048
N_BOOT = 2000
RNG = np.random.default_rng(42)


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


# Hyperparameters copied from clinic/predict_all_models.py and home/step3_predict_all_models.py
CLINIC_MODELS = {
    'Ridge (best)': lambda: Ridge(alpha=5),
    'Lasso':        lambda: Lasso(alpha=1, max_iter=5000),
    'ElasticNet':   lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
    'KNN':          lambda: KNeighborsRegressor(n_neighbors=7),
    'SVR':          lambda: SVR(kernel='rbf', C=500, gamma=0.01),
    'RF':           lambda: RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    'XGBoost':      lambda: GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
}
HOME_MODELS = {
    'Ridge (best)': lambda: Ridge(alpha=20),
    'Lasso':        lambda: Lasso(alpha=5, max_iter=5000),
    'ElasticNet':   lambda: ElasticNet(alpha=1, l1_ratio=0.5, max_iter=5000),
    'KNN':          lambda: KNeighborsRegressor(n_neighbors=7),
    'SVR':          lambda: SVR(kernel='rbf', C=500, gamma=0.05),
    'RF':           lambda: RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
    'XGBoost':      lambda: GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42),
}


def loo_clinic(X, y, model_fn):
    """LOO with no feature selection (clinic headline protocol)."""
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pr[te] = m.predict(sc.transform(X[te]))
    return pr


def loo_home_spearman_top20(X_accel, X_demo, y, model_fn, K=20):
    """LOO with Spearman Top-K accel + Demo (home headline protocol)."""
    n_accel = X_accel.shape[1]
    pr = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), dtype=bool); tr[i] = False
        corrs = [abs(spearmanr(X_accel[tr, j], y[tr])[0])
                 if np.std(X_accel[tr, j]) > 0 else 0 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K]
        X_tr = np.column_stack([X_accel[tr][:, top_k], X_demo[tr]])
        X_te = np.column_stack([X_accel[i:i + 1][:, top_k], X_demo[i:i + 1]])
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X_tr), y[tr])
        pr[i] = m.predict(sc.transform(X_te))[0]
    return pr


def metrics_with_ci(y_ft, pr_ft, n_boot=N_BOOT, rng=RNG):
    """Point estimates + 95% bootstrap percentile CIs for R², MAE (m), Pearson r."""
    y_m = y_ft * FT2M
    pr_m = pr_ft * FT2M
    r2 = r2_score(y_ft, pr_ft)
    mae = mean_absolute_error(y_m, pr_m)
    r_v = pearsonr(y_ft, pr_ft)[0]
    n = len(y_ft)
    r2_b, mae_b, r_b = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_ft[idx]; pb = pr_ft[idx]
        try:
            r2_b[b] = r2_score(yb, pb)
        except Exception:
            r2_b[b] = np.nan
        mae_b[b] = mean_absolute_error(yb * FT2M, pb * FT2M)
        if np.std(yb) > 0 and np.std(pb) > 0:
            r_b[b] = pearsonr(yb, pb)[0]
        else:
            r_b[b] = np.nan
    ci = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))
    return r2, ci(r2_b), mae, ci(mae_b), r_v, ci(r_b)


def fmt(val, lo, hi, nd=2):
    return f"{val:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_mae(val, lo, hi):
    return f"{val:.1f} [{lo:.1f}, {hi:.1f}]"


def main():
    t0 = time.time()
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo_h = impute(p[['cohort_POMS', 'Age', 'Sex', 'Height']].values.astype(float))
    X_demo_b = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))

    # Clinic inputs (55f, no selection)
    c_gait = impute(pd.read_csv(FEATS / 'clinic_gait_features.csv').drop(columns='key').values.astype(float))
    c_cwt = impute(pd.read_csv(FEATS / 'clinic_cwt_features.csv').drop(columns='key').values.astype(float))
    c_ws = impute(pd.read_csv(FEATS / 'clinic_walksway_features.csv').drop(columns='key').values.astype(float))
    X_clinic = impute(np.column_stack([c_gait, c_cwt, c_ws, X_demo_h]))

    # Home inputs (151f pool + Demo, Spearman Top-20 inside LOO)
    X_home = impute(pd.read_csv(FEATS / 'home_perbout_features.csv')
                    .drop(columns='key').values.astype(float))

    rows = []
    print(f"Model comparison  |  n={n}  |  {N_BOOT} bootstrap resamples  |  LOO CV\n")

    for model_name in CLINIC_MODELS.keys():
        # ── Clinic ──
        pr_c = loo_clinic(X_clinic, y, CLINIC_MODELS[model_name])
        r2c, r2c_ci, maec, maec_ci, rc, rc_ci = metrics_with_ci(y, pr_c)
        # ── Home ──
        pr_h = loo_home_spearman_top20(X_home, X_demo_b, y, HOME_MODELS[model_name])
        r2h, r2h_ci, maeh, maeh_ci, rh, rh_ci = metrics_with_ci(y, pr_h)

        rows.append({
            'Model': model_name,
            'Clinic R² [95% CI]':    fmt(r2c, *r2c_ci),
            'Clinic MAE (m) [95% CI]': fmt_mae(maec, *maec_ci),
            'Clinic r [95% CI]':     fmt(rc, *rc_ci),
            'Home R² [95% CI]':      fmt(r2h, *r2h_ci),
            'Home MAE (m) [95% CI]': fmt_mae(maeh, *maeh_ci),
            'Home r [95% CI]':       fmt(rh, *rh_ci),
            '_sort_r2_mean': (r2c + r2h) / 2,  # sort key, dropped before save
        })
        print(f"  {model_name:<13s}  clinic R²={r2c:.2f} / home R²={r2h:.2f}  (done)")

    # Sort worst → best by mean of (clinic R², home R²) so best model sits at the bottom
    df = pd.DataFrame(rows).sort_values('_sort_r2_mean', ascending=True).drop(columns='_sort_r2_mean').reset_index(drop=True)
    df.to_csv(OUT / 'model_comparison.csv', index=False)
    df.to_csv(POMS_TABLES / 'model_comparison.csv', index=False)

    print('\nSorted table (worst → best by mean R²):\n')
    print(df.to_string(index=False))
    print(f"\nSaved results/paper_tables/model_comparison.csv (+ POMS/tables/)")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
