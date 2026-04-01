#!/usr/bin/env python3
"""
Clinic all models: Gait+CWT+WalkSway+Demo(Height), 55 features, no selection, LOO CV.
Best config per model from hyperparameter search.

Input:  feats/clinic_{gait,cwt,walksway}_features.csv + SwayDemographics.xlsx
        home_full_recording_npz/_subjects.csv
Output: Prints R², MAE, ρ for each model

Run:  python clinic/predict_all_models.py
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
FT2M = 0.3048


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


MODELS = {
    'Ridge':      lambda: Ridge(alpha=5),
    'Lasso':      lambda: Lasso(alpha=1, max_iter=5000),
    'ElasticNet': lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
    'KNN':        lambda: KNeighborsRegressor(n_neighbors=7),
    'SVR':        lambda: SVR(kernel='rbf', C=500, gamma=0.01),
    'RF':         lambda: RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    'XGBoost':    lambda: GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
}


if __name__ == '__main__':
    FEATS_DIR = BASE / 'feats'
    subj_df = pd.read_csv(BASE / 'home_full_recording_npz' / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)

    c_gait = pd.read_csv(FEATS_DIR / 'clinic_gait_features.csv').drop(columns='key').values.astype(float)
    c_cwt = pd.read_csv(FEATS_DIR / 'clinic_cwt_features.csv').drop(columns='key').values.astype(float)
    c_ws = pd.read_csv(FEATS_DIR / 'clinic_walksway_features.csv').drop(columns='key').values.astype(float)

    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'Height']].values.astype(float))

    X = impute(np.column_stack([c_gait, c_cwt, c_ws, X_demo]))
    n_feat = X.shape[1]

    print(f"Clinic Results — All Models (n={n}, {n_feat}f Gait+CWT+WS+Demo, no selection, LOO CV)")
    print(f"{'='*80}")
    print(f"  {'Model':<15s}  {'R²':>6s}  {'MAE (m)':>8s}  {'ρ':>6s}  {'r':>6s}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")

    all_preds = {}
    for name, model_fn in MODELS.items():
        preds = np.zeros(n)
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler()
            m = model_fn()
            m.fit(sc.fit_transform(X[tr]), y[tr])
            preds[te] = m.predict(sc.transform(X[te]))
        all_preds[name] = preds
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y * FT2M, preds * FT2M)
        rho = spearmanr(y, preds)[0]
        r_val = pearsonr(y, preds)[0]
        print(f"  {name:<15s}  {r2:6.3f}  {mae:8.1f}  {rho:6.3f}  {r_val:6.3f}")
