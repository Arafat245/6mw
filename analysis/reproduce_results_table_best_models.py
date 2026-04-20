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
N_BOOT = 2000
RNG = np.random.default_rng(42)


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_ridge(X, y, alpha):
    """Ridge LOO, no feature selection. Returns LOO predictions (feet)."""
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pr[te] = m.predict(sc.transform(X[te]))
    return pr


def loo_spearman_ridge(X_accel, X_demo, y, K=20, alpha=20):
    """Spearman Top-K + Demo, Ridge. Returns LOO predictions (feet)."""
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
    return pr


def metrics_with_ci(y_ft, pr_ft, n_boot=N_BOOT, rng=RNG):
    """Point estimate + 95% percentile bootstrap CI for R², MAE (m), Pearson r."""
    y_m = y_ft * FT2M; pr_m = pr_ft * FT2M
    r2 = r2_score(y_ft, pr_ft)
    mae = mean_absolute_error(y_m, pr_m)
    r_v = pearsonr(y_ft, pr_ft)[0]
    n = len(y_ft)
    r2b, maeb, rb = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y_ft[idx], pr_ft[idx]
        try:
            r2b[b] = r2_score(yb, pb)
        except Exception:
            r2b[b] = np.nan
        maeb[b] = mean_absolute_error(yb * FT2M, pb * FT2M)
        rb[b] = pearsonr(yb, pb)[0] if np.std(yb) > 0 and np.std(pb) > 0 else np.nan
    pct = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))
    return r2, pct(r2b), mae, pct(maeb), r_v, pct(rb)


def fmt2(v, lo, hi): return f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
def fmt1(v, lo, hi): return f"{v:.1f} [{lo:.1f}, {hi:.1f}]"


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

    print(f'\nComputing LOO predictions (Clinic=Ridge(α=5), Home=Ridge(α=20))...')
    configs = [
        ('Gait',                 11, lambda: loo_ridge(c_gait, y, alpha=5),
                                     lambda: loo_spearman_ridge(h_gait, None, y, K=11, alpha=5)),
        ('CWT',                  28, lambda: loo_ridge(c_cwt, y, alpha=20),
                                     lambda: loo_spearman_ridge(h_cwt, None, y, K=11, alpha=20)),
        ('WalkSway',             12, lambda: loo_ridge(c_ws, y, alpha=5),
                                     lambda: loo_spearman_ridge(h_ws, None, y, K=11, alpha=5)),
        ('Demo',                  4, lambda: loo_ridge(X_demo_bmi, y, alpha=20),
                                     None),  # home == clinic for demo-only
        ('Bout+Act-Top20',       20, lambda: loo_spearman_ridge(c_pb, None, y, K=20, alpha=5),
                                     lambda: loo_spearman_ridge(h_pb, None, y, K=20, alpha=20)),
        ('Bout+Act-Top20+Demo',  24, lambda: loo_spearman_ridge(c_pb, X_demo_bmi, y, K=20, alpha=20),
                                     lambda: loo_spearman_ridge(h_pb, X_demo_bmi, y, K=20, alpha=20)),
        ('Gait+CWT+WS+Demo',     55, lambda: loo_ridge(np.column_stack([c_gait, c_cwt, c_ws, X_demo_height]), y, alpha=5),
                                     lambda: loo_spearman_ridge(np.column_stack([h_gait, h_cwt, h_ws]), X_demo_bmi, y, K=20, alpha=20)),
    ]

    rows = []
    for name, nf, clinic_fn, home_fn in configs:
        pr_c = clinic_fn()
        cr2, cci, cmae, mcci, cr_p, rcci = metrics_with_ci(y, pr_c)
        if home_fn is None:  # demo-only: home prediction identical to clinic
            pr_h = pr_c
            hr2, hci, hmae, mhci, hr_p, rhci = cr2, cci, cmae, mcci, cr_p, rcci
        else:
            pr_h = home_fn()
            hr2, hci, hmae, mhci, hr_p, rhci = metrics_with_ci(y, pr_h)
        rows.append({
            'Feature Set': name, '#f': nf,
            'Clinic R² [95% CI]':       fmt2(cr2, *cci),
            'Clinic MAE (m) [95% CI]':  fmt1(cmae, *mcci),
            'Clinic r [95% CI]':        fmt2(cr_p, *rcci),
            'Home R² [95% CI]':         fmt2(hr2, *hci),
            'Home MAE (m) [95% CI]':    fmt1(hmae, *mhci),
            'Home r [95% CI]':          fmt2(hr_p, *rhci),
        })
        print(f'  {name:<22s}  C R²={cr2:.2f} [{cci[0]:.2f},{cci[1]:.2f}]  H R²={hr2:.2f} [{hci[0]:.2f},{hci[1]:.2f}]')

    # Save to results/ (canonical) and POMS/tables/ (paper-side copy)
    df = pd.DataFrame(rows)
    RESULTS = BASE / 'results'; RESULTS.mkdir(exist_ok=True)
    POMS_TABLES = BASE / 'POMS' / 'tables'; POMS_TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / 'results_table_best_models.csv', index=False)
    df.to_csv(POMS_TABLES / 'results_table_final.csv', index=False)  # replaces legacy file

    print(f'\n{df.to_string(index=False)}')
    print(f'\nClinic model: Ridge(α=5)    Home model: Ridge(α=20)')
    print(f'Bootstrap resamples: {N_BOOT}  |  seed: 42  |  percentile CI (2.5%, 97.5%)')
    print(f'Saved: results/results_table_best_models.csv')
    print(f'       POMS/tables/results_table_final.csv')
    print(f'Done in {time.time() - t0:.0f}s')
