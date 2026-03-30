#!/usr/bin/env python3
"""
Reproduce best home result: R²=0.462, MAE=187, ρ=0.661
Spearman Top-20 selected inside LOO (no data leakage) + Demo(4), Ridge α=20.

Input:
  feats/home_perbout_features.csv  — 153 per-bout + activity features
  feats/target_6mwd.csv              — subject list with 6MWD targets
  SwayDemographics.xlsx              — demographics (Age, Sex, BMI)

Output:
  Prints R², MAE, ρ to stdout
"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent.parent


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


if __name__ == '__main__':
    # Load subjects (exclude M22, M44)
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)

    # Match to clinic-valid subjects
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                             for _, r in ids101.iterrows()])
    ids_v = ids101[clinic_valid].reset_index(drop=True)
    y = ids_v['sixmwd'].values.astype(float)
    n = len(y)

    # Load 153 PerBout + Activity features
    cf_csv = pd.read_csv(BASE / 'feats' / 'home_perbout_features.csv')
    pb_idx = []
    for _, r in ids_v.iterrows():
        match = ids101[(ids101['cohort'] == r['cohort']) & (ids101['subj_id'] == r['subj_id']) & (ids101['sixmwd'] == r['sixmwd'])]
        if len(match) > 0: pb_idx.append(match.index[0])
    cf_sub = cf_csv.iloc[pb_idx].reset_index(drop=True)
    pb_cols = [c for c in cf_sub.columns if c.startswith('g_') or c.startswith('act_')]
    X_pb = impute(cf_sub[pb_cols].values.astype(float))

    # Demo(4): cohort_POMS, Age, Sex, BMI
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo4 = impute(p[['cohort_M', 'Age', 'Sex', 'BMI']].values.astype(float))

    # Nested LOO: Spearman Top-20 inside each fold, Ridge α=20
    K = 20
    alpha = 20
    pr = np.zeros(n)

    for i in range(n):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[i] = False
        X_train, y_train = X_pb[tr_mask], y[tr_mask]
        X_test = X_pb[i:i+1]

        # Feature selection on TRAINING data only
        corrs = []
        for j in range(X_train.shape[1]):
            if np.std(X_train[:, j]) < 1e-12:
                corrs.append(0)
            else:
                corrs.append(abs(spearmanr(X_train[:, j], y_train)[0]))
        sel_idx = np.argsort(corrs)[::-1][:K]

        # Combine selected features + Demo(4)
        X_tr = np.column_stack([X_train[:, sel_idx], X_demo4[tr_mask]])
        X_te = np.column_stack([X_test[:, sel_idx], X_demo4[i:i+1]])

        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_tr), y_train)
        pr[i] = m.predict(sc.transform(X_te))[0]

    r2 = r2_score(y, pr)
    mae = mean_absolute_error(y, pr)
    rho = spearmanr(y, pr)[0]
    rv = pearsonr(y, pr)[0]

    print(f"Home Result (clinic-free, no data leakage)")
    print(f"  n={n}, Spearman Top-{K} inside LOO, Ridge α={alpha}")
    print(f"  R²={r2:.4f}  MAE={mae:.1f} ft  r={rv:.3f}  ρ={rho:.3f}")
    print(f"  Features: {K} PerBout (selected per fold) + 4 Demo = {K+4}f")
