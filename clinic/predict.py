#!/usr/bin/env python3
"""
Clinic best result: Gait+CWT+WalkSway+Demo(Height), Ridge α=5, LOO CV.

Input:  feats/clinic_gait_features.csv
        feats/clinic_cwt_features.csv
        feats/clinic_walksway_features.csv
        SwayDemographics.xlsx
        home_full_recording_npz/_subjects.csv
Output: Prints R², MAE, Pearson r (Spearman ρ also shown) → R²=0.806

Run:  python clinic/predict.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

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


if __name__ == '__main__':
    FEATS_DIR = BASE / 'feats'
    subj_df = pd.read_csv(BASE / 'home_full_recording_npz' / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)

    # Load cached clinic features
    c_gait_df = pd.read_csv(FEATS_DIR / 'clinic_gait_features.csv')
    c_cwt_df = pd.read_csv(FEATS_DIR / 'clinic_cwt_features.csv')
    c_ws_df = pd.read_csv(FEATS_DIR / 'clinic_walksway_features.csv')
    for df_name, df in [('gait', c_gait_df), ('cwt', c_cwt_df), ('ws', c_ws_df)]:
        assert list(subj_df['key']) == list(df['key']), f"Key mismatch in clinic_{df_name}_features.csv"
    c_gait = c_gait_df.drop(columns='key').values.astype(float)
    c_cwt = c_cwt_df.drop(columns='key').values.astype(float)
    c_ws = c_ws_df.drop(columns='key').values.astype(float)

    # Demo(4): cohort_POMS, Age, Sex, Height
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'Height']].values.astype(float))

    # Combine: Gait(11) + CWT(28) + WalkSway(12) + Demo(4) = 55 features
    X = impute(np.column_stack([c_gait, c_cwt, c_ws, X_demo]))
    n_feat = X.shape[1]

    # LOO CV: no feature selection, Ridge α=5
    alpha = 5
    preds = np.zeros(n)
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))

    y_m = y * FT2M
    pr_m = preds * FT2M
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y_m, pr_m)
    rho = spearmanr(y, preds)[0]
    r_val = pearsonr(y, preds)[0]

    print(f"Clinic Result (Gait+CWT+WalkSway+Demo)")
    print(f"  n={n}, no feature selection, Ridge α={alpha}")
    print(f"  R²={r2:.3f}  MAE={mae:.1f} m  r={r_val:.3f}  ρ={rho:.3f}")
    print(f"  Features: Gait({c_gait.shape[1]}) + CWT({c_cwt.shape[1]}) + WS({c_ws.shape[1]}) + Demo(4) = {n_feat}f")
