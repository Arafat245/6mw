#!/usr/bin/env python3
"""
Step 3 (all models): LOO CV with Spearman Top-20 + Demo(4).
Best config per model from hyperparameter search.

Input:  feats/home_perbout_features.csv + SwayDemographics.xlsx
        home_full_recording_npz/_subjects.csv
Output: Prints R², MAE, ρ for each model + voting ensembles

Run:  python home/step3_predict_all_models.py
"""
import warnings
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
NPZ_DIR = BASE / 'home_full_recording_npz'
FT2M = 0.3048


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


MODELS = {
    'Ridge':      lambda: Ridge(alpha=20),
    'Lasso':      lambda: Lasso(alpha=5, max_iter=5000),
    'ElasticNet': lambda: ElasticNet(alpha=1, l1_ratio=0.5, max_iter=5000),
    'KNN':        lambda: KNeighborsRegressor(n_neighbors=7),
    'SVR':        lambda: SVR(kernel='rbf', C=500, gamma=0.05),
    'RF':         lambda: RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
    'XGBoost':    lambda: GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42),
}


if __name__ == '__main__':
    FEATS_DIR = BASE / 'feats'
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)

    # Load 151 accel features
    feat_df = pd.read_csv(FEATS_DIR / 'home_perbout_features.csv')
    accel_cols = [c for c in feat_df.columns if c != 'key']
    X_accel = impute(feat_df[accel_cols].values.astype(float))
    n_accel = X_accel.shape[1]

    # Demo(4): cohort_POMS, Age, Sex, BMI
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))
    n_demo = 4

    X_all = np.column_stack([X_accel, X_demo])
    demo_idx = list(range(n_accel, n_accel + n_demo))
    K = 20

    # LOO CV: Spearman Top-20 inside each fold + Demo(4)
    all_preds = {name: np.zeros(n) for name in MODELS}

    for tr, te in LeaveOneOut().split(X_all):
        corrs = [abs(spearmanr(X_all[tr, j], y[tr])[0]) if np.std(X_all[tr, j]) > 0 else 0
                 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K]
        selected = top_k + demo_idx
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_all[tr][:, selected])
        X_te = sc.transform(X_all[te][:, selected])
        for name, model_fn in MODELS.items():
            m = model_fn()
            m.fit(X_tr, y[tr])
            all_preds[name][te] = m.predict(X_te)

    # Report individual models
    print(f"Home Results — All Models (n={n}, Spearman Top-{K} + Demo({n_demo}), LOO CV)")
    print(f"{'='*80}")
    print(f"  {'Model':<15s}  {'R²':>6s}  {'MAE (m)':>8s}  {'ρ':>6s}  {'r':>6s}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")
    for name, preds in all_preds.items():
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds) * FT2M
        rho = spearmanr(y, preds)[0]
        r_val = pearsonr(y, preds)[0]
        print(f"  {name:<15s}  {r2:6.3f}  {mae:8.1f}  {rho:6.3f}  {r_val:6.3f}")

    # Voting ensembles
    ensembles = {
        'Vote(Ri+La+SVR)': ['Ridge', 'Lasso', 'SVR'],
        'Vote(Ri+SVR)':    ['Ridge', 'SVR'],
        'Vote(Ri+La+SVR+XGB)': ['Ridge', 'Lasso', 'SVR', 'XGBoost'],
    }
    print(f"  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")
    for ens_name, members in ensembles.items():
        blend = np.mean([all_preds[m] for m in members], axis=0)
        r2 = r2_score(y, blend)
        mae = mean_absolute_error(y, blend) * FT2M
        rho = spearmanr(y, blend)[0]
        r_val = pearsonr(y, blend)[0]
        print(f"  {ens_name:<15s}  {r2:6.3f}  {mae:8.1f}  {rho:6.3f}  {r_val:6.3f}")
