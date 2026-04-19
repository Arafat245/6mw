#!/usr/bin/env python3
"""
Results table with best models per setting:
- Clinic: Ridge(α=5)
- Home: Ridge(α=20)

Both settings use Ridge regression — Vote ensemble (R²=0.478) offered
only marginal improvement over Ridge (R²=0.454), not worth the complexity.

Input:  feats/*.csv + SwayDemographics.xlsx
Output: results/results_table_best_models.csv

Run:  python analysis/reproduce_results_table_best_models.py
"""
import warnings, time
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
FT2M = 0.3048


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_ridge(X, y, alpha):
    """Ridge LOO, no feature selection."""
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pr[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, pr), mean_absolute_error(y * FT2M, pr * FT2M), pearsonr(y, pr)[0]


def loo_spearman_ridge(X_accel, X_demo, y, K=20, alpha=20):
    """Spearman Top-K + Demo, Ridge."""
    n_accel = X_accel.shape[1]
    pr = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), dtype=bool); tr[i] = False
        K_use = min(K, n_accel)
        corrs = [abs(spearmanr(X_accel[tr, j], y[tr])[0])
                 if np.std(X_accel[tr, j]) > 0 else 0 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K_use]
        if X_demo is not None:
            X_tr = np.column_stack([X_accel[tr][:, top_k], X_demo[tr]])
            X_te = np.column_stack([X_accel[i:i + 1][:, top_k], X_demo[i:i + 1]])
        else:
            X_tr = X_accel[tr][:, top_k]
            X_te = X_accel[i:i + 1][:, top_k]
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_tr), y[tr])
        pr[i] = m.predict(sc.transform(X_te))[0]
    return r2_score(y, pr), mean_absolute_error(y * FT2M, pr * FT2M), pearsonr(y, pr)[0]


if __name__ == '__main__':
    t0 = time.time()
    FEATS = BASE / 'feats'

    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)

    # Demographics
    demo_xl = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo_xl['cohort'] = demo_xl['ID'].str.extract(r'^([A-Z])')[0]
    demo_xl['subj_id'] = demo_xl['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo_xl, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')

    X_demo_bmi = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))
    X_demo_height = impute(p[['cohort_POMS', 'Age', 'Sex', 'Height']].values.astype(float))

    # Clinic features
    c_gait = impute(pd.read_csv(FEATS / 'clinic_gait_features.csv').drop(columns='key').values.astype(float))
    c_cwt = impute(pd.read_csv(FEATS / 'clinic_cwt_features.csv').drop(columns='key').values.astype(float))
    c_ws = impute(pd.read_csv(FEATS / 'clinic_walksway_features.csv').drop(columns='key').values.astype(float))
    c_pb = impute(pd.read_csv(FEATS / 'clinic_perbout_features.csv').drop(columns='key').values.astype(float))

    # Home features
    h_pb = impute(pd.read_csv(FEATS / 'home_perbout_features.csv').drop(columns='key').values.astype(float))
    h_gait = impute(pd.read_csv(FEATS / 'home_gait_features.csv').drop(columns='key').values.astype(float))
    h_cwt = impute(pd.read_csv(FEATS / 'home_cwt_features.csv').drop(columns='key').values.astype(float))
    h_ws = impute(pd.read_csv(FEATS / 'home_walksway_features.csv').drop(columns='key').values.astype(float))

    print(f'\nComputing results (Clinic=Ridge(α=5), Home=Ridge(α=20))...')
    rows = []

    # ── Row 1: Gait ──
    cr2, cmae, cr_p =loo_ridge(c_gait, y, alpha=5)
    hr2, hmae, hr_p =loo_spearman_ridge(h_gait, None, y, K=11, alpha=5)
    rows.append({'Feature Set': 'Gait', '#f': 11,
                 'Clinic R²': round(cr2, 2), 'Clinic MAE (m)': round(cmae, 1), 'Clinic r': round(cr_p, 2),
                 'Home R²': round(hr2, 2), 'Home MAE (m)': round(hmae, 1), 'Home r': round(hr_p, 2)})
    print(f'  Gait:              C R²={cr2:.3f}  H R²={hr2:.3f}')

    # ── Row 2: CWT ──
    cr2, cmae, cr_p =loo_ridge(c_cwt, y, alpha=20)
    hr2, hmae, hr_p =loo_spearman_ridge(h_cwt, None, y, K=11, alpha=20)
    rows.append({'Feature Set': 'CWT', '#f': 28,
                 'Clinic R²': round(cr2, 2), 'Clinic MAE (m)': round(cmae, 1), 'Clinic r': round(cr_p, 2),
                 'Home R²': round(hr2, 2), 'Home MAE (m)': round(hmae, 1), 'Home r': round(hr_p, 2)})
    print(f'  CWT:               C R²={cr2:.3f}  H R²={hr2:.3f}')

    # ── Row 3: WalkSway ──
    cr2, cmae, cr_p =loo_ridge(c_ws, y, alpha=5)
    hr2, hmae, hr_p =loo_spearman_ridge(h_ws, None, y, K=11, alpha=5)
    rows.append({'Feature Set': 'WalkSway', '#f': 12,
                 'Clinic R²': round(cr2, 2), 'Clinic MAE (m)': round(cmae, 1), 'Clinic r': round(cr_p, 2),
                 'Home R²': round(hr2, 2), 'Home MAE (m)': round(hmae, 1), 'Home r': round(hr_p, 2)})
    print(f'  WalkSway:          C R²={cr2:.3f}  H R²={hr2:.3f}')

    # ── Row 4: Demo ──
    cr2, cmae, cr_p =loo_ridge(X_demo_bmi, y, alpha=20)
    hr2, hmae, hr_p = cr2, cmae, cr_p  # identical
    rows.append({'Feature Set': 'Demo', '#f': 4,
                 'Clinic R²': round(cr2, 2), 'Clinic MAE (m)': round(cmae, 1), 'Clinic r': round(cr_p, 2),
                 'Home R²': round(hr2, 2), 'Home MAE (m)': round(hmae, 1), 'Home r': round(hr_p, 2)})
    print(f'  Demo:              C R²={cr2:.3f}  H R²={hr2:.3f}')

    # ── Row 5: Bout+Act-Top20 ──
    cr2, cmae, cr_p =loo_spearman_ridge(c_pb, None, y, K=20, alpha=5)
    hr2, hmae, hr_p =loo_spearman_ridge(h_pb, None, y, K=20, alpha=20)
    rows.append({'Feature Set': 'Bout+Act-Top20', '#f': 20,
                 'Clinic R²': round(cr2, 2), 'Clinic MAE (m)': round(cmae, 1), 'Clinic r': round(cr_p, 2),
                 'Home R²': round(hr2, 2), 'Home MAE (m)': round(hmae, 1), 'Home r': round(hr_p, 2)})
    print(f'  Bout+Act-Top20:     C R²={cr2:.3f}  H R²={hr2:.3f}')

    # ── Row 6: Bout+Act-Top20+Demo ──
    cr2, cmae, cr_p =loo_spearman_ridge(c_pb, X_demo_bmi, y, K=20, alpha=20)
    hr2, hmae, hr_p =loo_spearman_ridge(h_pb, X_demo_bmi, y, K=20, alpha=20)
    rows.append({'Feature Set': 'Bout+Act-Top20+Demo', '#f': 24,
                 'Clinic R²': round(cr2, 2), 'Clinic MAE (m)': round(cmae, 1), 'Clinic r': round(cr_p, 2),
                 'Home R²': round(hr2, 2), 'Home MAE (m)': round(hmae, 1), 'Home r': round(hr_p, 2)})
    print(f'  Bout+Act-Top20+Demo: C R²={cr2:.3f}  H R²={hr2:.3f}')

    # ── Row 7: Gait+CWT+WS+Demo ──
    X_clinic_all = np.column_stack([c_gait, c_cwt, c_ws, X_demo_height])
    cr2, cmae, cr_p =loo_ridge(X_clinic_all, y, alpha=5)
    X_home_gcw = np.column_stack([h_gait, h_cwt, h_ws])
    hr2, hmae, hr_p =loo_spearman_ridge(X_home_gcw, X_demo_bmi, y, K=20, alpha=20)
    rows.append({'Feature Set': 'Gait+CWT+WS+Demo', '#f': 55,
                 'Clinic R²': round(cr2, 2), 'Clinic MAE (m)': round(cmae, 1), 'Clinic r': round(cr_p, 2),
                 'Home R²': round(hr2, 2), 'Home MAE (m)': round(hmae, 1), 'Home r': round(hr_p, 2)})
    print(f'  Gait+CWT+WS+Demo: C R²={cr2:.3f}  H R²={hr2:.3f}')

    # Save
    df = pd.DataFrame(rows)
    RESULTS = BASE / 'results'
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / 'results_table_best_models.csv', index=False)

    print(f'\n{df.to_string(index=False)}')
    print(f'\nClinic model: Ridge(α=5)')
    print(f'Home model:   Ridge(α=20)')
    print(f'\nSaved results/results_table_best_models.csv')
    print(f'Done in {time.time() - t0:.0f}s')
