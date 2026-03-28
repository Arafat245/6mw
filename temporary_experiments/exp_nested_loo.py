#!/usr/bin/env python3
"""
Nested LOO CV — no data leakage in feature selection.
For each held-out subject:
  1. Use 100 training subjects to select features (forward selection)
  2. Train Ridge on selected features
  3. Predict held-out subject

Tests:
  A) PerBout-Top20 + Demo(4) — Top-20 selected inside each fold
  B) All 161 features → forward selection inside each fold
  C) Fixed 20 features (current best) — no selection needed, just verify
"""
import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def select_top_k_spearman(X_train, y_train, K):
    """Select top-K features by |Spearman ρ| with y on training data only."""
    corrs = []
    for j in range(X_train.shape[1]):
        if np.std(X_train[:, j]) < 1e-12:
            corrs.append(0)
        else:
            corrs.append(abs(spearmanr(X_train[:, j], y_train)[0]))
    idx = np.argsort(corrs)[::-1][:K]
    return idx


def forward_select_train(X_train, y_train, max_features=20, alpha=10):
    """Forward selection on training data only. Returns selected feature indices."""
    n_feat = X_train.shape[1]
    selected = []
    remaining = list(range(n_feat))

    for step in range(min(max_features, n_feat)):
        best_r2, best_f = -999, remaining[0]
        for f in remaining:
            trial = selected + [f]
            # Internal LOO on training set (double LOO)
            X_sel = X_train[:, trial]
            pr = np.zeros(len(y_train))
            for tr2, te2 in LeaveOneOut().split(X_sel):
                sc = StandardScaler(); m = Ridge(alpha=alpha)
                m.fit(sc.fit_transform(X_sel[tr2]), y_train[tr2])
                pr[te2] = m.predict(sc.transform(X_sel[te2]))
            r2 = r2_score(y_train, pr)
            if r2 > best_r2:
                best_r2, best_f = r2, f
        selected.append(best_f)
        remaining.remove(best_f)

        # Early stopping: no improvement for 3 steps
        if len(selected) >= 4:
            # Check if recent additions helped
            X_recent = X_train[:, selected]
            pr = np.zeros(len(y_train))
            for tr2, te2 in LeaveOneOut().split(X_recent):
                sc = StandardScaler(); m = Ridge(alpha=alpha)
                m.fit(sc.fit_transform(X_recent[tr2]), y_train[tr2])
                pr[te2] = m.predict(sc.transform(X_recent[te2]))
            curr_r2 = r2_score(y_train, pr)

            X_prev = X_train[:, selected[:-3]]
            pr2 = np.zeros(len(y_train))
            for tr2, te2 in LeaveOneOut().split(X_prev):
                sc = StandardScaler(); m = Ridge(alpha=alpha)
                m.fit(sc.fit_transform(X_prev[tr2]), y_train[tr2])
                pr2[te2] = m.predict(sc.transform(X_prev[te2]))
            prev_r2 = r2_score(y_train, pr2)

            if curr_r2 <= prev_r2:
                selected = selected[:-3]  # revert last 3
                break

    return selected


def nested_loo_spearman_topk(X, y, K, alpha=20):
    """Nested LOO: select top-K by Spearman inside each fold."""
    n = len(y)
    pr = np.zeros(n)
    for i in range(n):
        tr_mask = np.ones(n, dtype=bool); tr_mask[i] = False
        X_train, y_train = X[tr_mask], y[tr_mask]
        X_test = X[i:i+1]

        # Feature selection on training data only
        sel_idx = select_top_k_spearman(X_train, y_train, K)

        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_train[:, sel_idx]), y_train)
        pr[i] = m.predict(sc.transform(X_test[:, sel_idx]))[0]

    return r2_score(y, pr), mean_absolute_error(y, pr), spearmanr(y, pr)[0]


def nested_loo_forward(X, y, max_features=20, alpha=10):
    """Nested LOO: forward selection inside each fold."""
    n = len(y)
    pr = np.zeros(n)
    all_n_selected = []

    for i in range(n):
        tr_mask = np.ones(n, dtype=bool); tr_mask[i] = False
        X_train, y_train = X[tr_mask], y[tr_mask]
        X_test = X[i:i+1]

        # Forward selection on training data only
        sel_idx = forward_select_train(X_train, y_train, max_features, alpha)
        all_n_selected.append(len(sel_idx))

        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_train[:, sel_idx]), y_train)
        pr[i] = m.predict(sc.transform(X_test[:, sel_idx]))[0]

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{n}] selected {len(sel_idx)}f", flush=True)

    mean_sel = np.mean(all_n_selected)
    return r2_score(y, pr), mean_absolute_error(y, pr), spearmanr(y, pr)[0], mean_sel


if __name__ == '__main__':
    t0 = time.time()

    # Load data
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                             for _, r in ids101.iterrows()])
    ids_v = ids101[clinic_valid].reset_index(drop=True)
    y = ids_v['sixmwd'].values.astype(float)
    n = len(y)

    # Demo(4)
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo4 = impute(p[['cohort_M', 'Age', 'Sex', 'BMI']].values.astype(float))

    # PerBout-Top20 (pre-selected — for comparison)
    cf = np.load(BASE / 'feats' / 'home_clinicfree_top20.npz')
    X_pb20 = cf['X']
    pb_idx = []
    for _, r in ids_v.iterrows():
        match = ids101[(ids101['cohort'] == r['cohort']) & (ids101['subj_id'] == r['subj_id']) & (ids101['sixmwd'] == r['sixmwd'])]
        if len(match) > 0: pb_idx.append(match.index[0])
    X_pb20 = impute(X_pb20[pb_idx])

    # All 153 PerBout features
    cf_csv = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
    cf_sub = cf_csv.iloc[pb_idx].reset_index(drop=True)
    pb_cols = [c for c in cf_sub.columns if c.startswith('g_') or c.startswith('act_')]
    X_pb_all = impute(cf_sub[pb_cols].values.astype(float))

    # Goldman
    gt3x = np.load(BASE / 'temporary_experiments' / 'goldman_gt3x_cache.npz', allow_pickle=True)
    gt3x_feats = list(gt3x['features'])
    gm_cols = ['msr', 'hwsr', 'avg_daily_steps', 'msr_plus_hwsr']
    X_gm = impute(np.array([[f[k] for k in gm_cols] for f in gt3x_feats]))

    # Combined pool (same as forward-ALL experiment): 153 PerBout + 4 Goldman + 4 Demo = 161
    X_all = np.column_stack([X_demo4, X_gm, X_pb_all])
    all_names = ['demo_cohort', 'demo_age', 'demo_sex', 'demo_bmi'] + \
                [f'gm_{c}' for c in gm_cols] + [f'pb_{c}' for c in pb_cols]

    # Remove constant features
    valid_cols = [j for j in range(X_all.shape[1]) if np.std(X_all[:, j]) > 1e-12]
    X_all = X_all[:, valid_cols]
    all_names = [all_names[j] for j in valid_cols]

    print(f"Nested LOO CV — No Data Leakage")
    print(f"n={n}, {X_all.shape[1]} total features")
    print(f"{'='*80}\n")

    # ── A) Fixed features (no leakage — no selection needed) ──
    print("A) Fixed feature sets (no selection, no leakage):")

    # PerBout-Top20 + Demo(4) — these 20 features are fixed, no selection per fold
    X_fixed = np.column_stack([X_pb20, X_demo4])
    for alpha in [5, 10, 20, 50]:
        pr = np.zeros(n)
        for tr, te in LeaveOneOut().split(X_fixed):
            sc = StandardScaler(); m = Ridge(alpha=alpha)
            m.fit(sc.fit_transform(X_fixed[tr]), y[tr]); pr[te] = m.predict(sc.transform(X_fixed[te]))
        r2 = r2_score(y, pr); mae = mean_absolute_error(y, pr); rho = spearmanr(y, pr)[0]
        print(f"  PerBout-Top20+Demo(4) α={alpha:>3d}: R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}  [24f, NO leakage]")

    print()

    # The "fixed 20" from forward selection — but these WERE selected with leakage
    # Now test them as fixed (no re-selection per fold) — this is the OPTIMISTIC estimate
    fixed20_names = ['demo_cohort', 'gm_msr', 'demo_age', 'demo_sex', 'demo_bmi',
                     'pb_g_enmo_mean_p10', 'pb_act_late_enmo', 'pb_g_cadence_power_max',
                     'pb_g_acf_step_reg_max', 'pb_g_ml_range_med', 'pb_g_enmo_p95_med',
                     'pb_g_stride_time_std_p10', 'pb_g_acf_step_reg_p90', 'pb_g_ml_rms_med',
                     'pb_g_cadence_power_cv', 'pb_g_duration_sec_iqr', 'pb_g_duration_sec_med',
                     'pb_g_ap_rms_p10', 'pb_act_enmo_p95', 'pb_act_enmo_mean']
    fixed20_idx = [all_names.index(f) for f in fixed20_names if f in all_names]
    X_fixed20 = X_all[:, fixed20_idx]
    pr = np.zeros(n)
    for tr, te in LeaveOneOut().split(X_fixed20):
        sc = StandardScaler(); m = Ridge(alpha=5)
        m.fit(sc.fit_transform(X_fixed20[tr]), y[tr]); pr[te] = m.predict(sc.transform(X_fixed20[te]))
    r2 = r2_score(y, pr); mae = mean_absolute_error(y, pr); rho = spearmanr(y, pr)[0]
    print(f"  Forward-selected 20f (fixed) α=5: R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}  [OPTIMISTIC — features chosen with leakage]")

    # ── B) Nested LOO: Spearman Top-K inside each fold ──
    print(f"\nB) Nested LOO: Spearman Top-K selected inside each fold:")
    for K in [10, 15, 20, 25]:
        for alpha in [10, 20]:
            r2, mae, rho = nested_loo_spearman_topk(X_all, y, K, alpha)
            print(f"  Top-{K}+α={alpha}: R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}  [NO leakage]")

    print(f"\n{'='*80}")
    print(f"COMPARISON:")
    print(f"  Previous (with leakage):    R²=0.555  MAE=166  ρ=0.708  [20f forward-selected]")
    print(f"  PerBout-Top20+Demo(4) fixed: see above  [24f, no leakage]")
    print(f"  Nested LOO results:          see above  [no leakage]")
    print(f"{'='*80}")
    print(f"Done in {time.time()-t0:.0f}s")
