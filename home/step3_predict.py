#!/usr/bin/env python3
"""
Step 3: LOO CV prediction with Spearman Top-20 + Demo(4), Ridge alpha=20.

Input:  feats/home_perbout_features.csv + SwayDemographics.xlsx
        home_full_recording_npz/_subjects.csv
Output: Prints R², MAE, Pearson r (Spearman ρ also shown for reference)

Run:  python home/step3_predict.py
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
NPZ_DIR = BASE / 'home_full_recording_npz'


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


if __name__ == '__main__':
    FEATS_DIR = BASE / 'feats'
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)

    # Load 151 accel features
    feat_df = pd.read_csv(FEATS_DIR / 'home_perbout_features.csv')
    assert list(subj_df['key']) == list(feat_df['key']), "Key mismatch between _subjects.csv and features"
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

    # LOO CV: Spearman Top-20 inside each fold + Demo(4), Ridge α=20
    K = 20
    alpha = 20
    preds = np.zeros(n)
    for tr, te in LeaveOneOut().split(X_all):
        corrs = [abs(spearmanr(X_all[tr, j], y[tr])[0]) if np.std(X_all[tr, j]) > 0 else 0
                 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K]
        selected = top_k + demo_idx
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_all[tr][:, selected]), y[tr])
        preds[te] = m.predict(sc.transform(X_all[te][:, selected]))

    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rho = spearmanr(y, preds)[0]
    r_val = pearsonr(y, preds)[0]
    print(f"Home Result (clinic-free, no data leakage)")
    print(f"  n={n}, Spearman Top-{K} inside LOO, Ridge α={alpha}")
    print(f"  R²={r2:.4f}  MAE={mae:.0f} ft  r={r_val:.3f}  ρ={rho:.3f}")
    print(f"  Features: {K} PerBout + {n_demo} Demo = {K+n_demo}f")
