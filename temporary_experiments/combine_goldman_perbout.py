#!/usr/bin/env python3
"""
Combine Goldman PAM features (MSR, HWSR, etc.) with PerBout-Top20 features.
Use feature selection to find the best combination, not naive concatenation.

Methods:
1. Correlation-based: rank all combined features by |ρ| with 6MWD, pick top-K
2. MRMR: minimum redundancy maximum relevance
3. Forward selection: greedily add features that improve LOO R²
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


def eval_loo(X, y, alpha=20):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, pr), mean_absolute_error(y, pr), spearmanr(y, pr)[0]

def best_alpha(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 10)
    for a in alphas:
        r2, mae, rho = eval_loo(X, y, a)
        if r2 > best[0]: best = (r2, mae, rho, a)
    return best

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def mrmr_select(X, y, names, K):
    """Minimum Redundancy Maximum Relevance forward selection."""
    n_feat = X.shape[1]
    relevance = np.array([abs(spearmanr(X[:, j], y)[0]) if np.std(X[:, j]) > 1e-12 else 0
                          for j in range(n_feat)])
    selected = []
    remaining = list(range(n_feat))

    for _ in range(min(K, n_feat)):
        best_score, best_f = -999, remaining[0]
        for f in remaining:
            rel = relevance[f]
            red = np.mean([abs(spearmanr(X[:, f], X[:, s])[0]) for s in selected]) if selected else 0
            score = rel - red
            if score > best_score:
                best_score, best_f = score, f
        selected.append(best_f)
        remaining.remove(best_f)

    return selected, [names[i] for i in selected]


def forward_selection(X, y, names, max_features=20, alpha=20):
    """Greedy forward selection: add feature that gives best LOO R² improvement."""
    n_feat = X.shape[1]
    selected = []
    remaining = list(range(n_feat))
    history = []

    for step in range(min(max_features, n_feat)):
        best_r2, best_f = -999, remaining[0]
        for f in remaining:
            trial = selected + [f]
            r2, _, _ = eval_loo(X[:, trial], y, alpha)
            if r2 > best_r2:
                best_r2, best_f = r2, f
        selected.append(best_f)
        remaining.remove(best_f)
        history.append((len(selected), names[best_f], best_r2))

        # Stop if no improvement for 3 steps
        if len(history) >= 4 and all(history[-i][2] <= history[-4][2] for i in range(1, 4)):
            break

    return selected, [names[i] for i in selected], history


if __name__ == '__main__':
    t0 = time.time()

    # Load subjects
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                             for _, r in ids101.iterrows()])
    ids_v = ids101[clinic_valid].reset_index(drop=True)
    y = ids_v['sixmwd'].values.astype(float)
    n = len(y)

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo5 = impute(p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float))

    # ── Load PerBout-Top20 features ──
    cf_npz = np.load(BASE / 'feats' / 'home_clinicfree_top20.npz')
    X_perbout = cf_npz['X']  # (101, 20)
    perbout_names = list(cf_npz['feature_names'])
    # Align to ids_v
    pb_idx = []
    for _, r in ids_v.iterrows():
        match = ids101[(ids101['cohort'] == r['cohort']) & (ids101['subj_id'] == r['subj_id']) & (ids101['sixmwd'] == r['sixmwd'])]
        if len(match) > 0: pb_idx.append(match.index[0])
    X_perbout = impute(X_perbout[pb_idx])

    # ── Load Goldman features (try GT3X first, fall back to AGD) ──
    gt3x_cache = BASE / 'temporary_experiments' / 'goldman_gt3x_cache.npz'
    agd_csv = BASE / 'temporary_experiments' / 'goldman_pam_features.csv'

    if gt3x_cache.exists():
        print("Using GT3X-based Goldman features")
        d = np.load(gt3x_cache, allow_pickle=True)
        gt3x_features = list(d['features'])
        gt3x_valid = np.array(d['valid'], dtype=bool)
        source = 'GT3X'
    else:
        print("GT3X cache not ready, using AGD-based Goldman features")
        gt3x_features = None
        gt3x_valid = None
        source = 'AGD'

    if source == 'AGD':
        agd_df = pd.read_csv(agd_csv)
        goldman_names = ['msr', 'hwsr', 'avg_daily_steps', 'mvpa_min_per_day', 'msr_plus_hwsr']
        X_goldman = np.full((n, len(goldman_names)), np.nan)
        for i, (_, r) in enumerate(ids_v.iterrows()):
            match = agd_df[(agd_df['cohort'] == r['cohort']) & (agd_df['subj_id'] == r['subj_id'])]
            if len(match) > 0:
                for j, gn in enumerate(goldman_names):
                    if gn in match.columns:
                        X_goldman[i, j] = match[gn].values[0]
        X_goldman = impute(X_goldman)
        goldman_valid = ~np.isnan(X_goldman[:, 0])
    else:
        goldman_names_all = sorted(gt3x_features[gt3x_valid.tolist().index(True)].keys())
        goldman_names = [g for g in goldman_names_all if g != 'n_valid_days']
        X_goldman = np.full((n, len(goldman_names)), np.nan)
        for i in range(n):
            if gt3x_valid[i]:
                for j, gn in enumerate(goldman_names):
                    X_goldman[i, j] = gt3x_features[i].get(gn, np.nan)
        X_goldman = impute(X_goldman)
        goldman_valid = gt3x_valid

    # Find subjects valid in both
    both_valid = np.ones(n, dtype=bool)  # PerBout has all 101
    if source == 'AGD':
        both_valid = ~np.all(np.isnan(X_goldman) | (X_goldman == 0), axis=1)

    nv = both_valid.sum()
    X_pb = X_perbout[both_valid]
    X_gm = X_goldman[both_valid]
    X_d5 = X_demo5[both_valid]
    y_v = y[both_valid]

    print(f"\nn={nv} subjects with both PerBout + Goldman features")
    print(f"PerBout: {X_pb.shape[1]} features, Goldman: {X_gm.shape[1]} features, Demo: 5")

    # ── Baselines ──
    print(f"\n{'='*90}")
    print(f"BASELINES")
    print(f"{'='*90}")

    r2, mae, rho, a = best_alpha(np.column_stack([X_pb, X_d5]), y_v)
    print(f"  PerBout-Top20 + Demo(5):    {25}f  α={a}  R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}")

    r2, mae, rho, a = best_alpha(np.column_stack([X_gm, X_d5]), y_v)
    print(f"  Goldman + Demo(5):          {X_gm.shape[1]+5}f  α={a}  R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}")

    # ── Naive combination ──
    print(f"\n{'='*90}")
    print(f"NAIVE COMBINATIONS")
    print(f"{'='*90}")

    X_all = np.column_stack([X_pb, X_gm, X_d5])
    all_names = [f'pb_{n}' for n in perbout_names] + [f'gm_{n}' for n in goldman_names] + ['demo_cohort', 'demo_age', 'demo_sex', 'demo_height', 'demo_bmi']
    nf_all = X_all.shape[1]

    r2, mae, rho, a = best_alpha(X_all, y_v)
    print(f"  All combined ({nf_all}f):        α={a}  R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}")

    # ── Method 1: Correlation-based top-K ──
    print(f"\n{'='*90}")
    print(f"METHOD 1: Correlation-based top-K (from combined pool)")
    print(f"{'='*90}")

    corrs = []
    for j in range(nf_all):
        rho_val = abs(spearmanr(X_all[:, j], y_v)[0]) if np.std(X_all[:, j]) > 1e-12 else 0
        corrs.append((j, rho_val, all_names[j]))
    corrs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top 15 features by |ρ|:")
    for idx, rho_val, name in corrs[:15]:
        print(f"    {name:35s}  |ρ|={rho_val:.3f}")

    for K in [5, 8, 10, 12, 15, 20]:
        top_idx = [c[0] for c in corrs[:K]]
        r2, mae, rho, a = best_alpha(X_all[:, top_idx], y_v)
        n_pb = sum(1 for c in corrs[:K] if c[2].startswith('pb_'))
        n_gm = sum(1 for c in corrs[:K] if c[2].startswith('gm_'))
        n_dm = sum(1 for c in corrs[:K] if c[2].startswith('demo_'))
        print(f"  Top-{K:>2d} (pb:{n_pb},gm:{n_gm},dm:{n_dm}):  α={a}  R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}")

    # ── Method 2: MRMR ──
    print(f"\n{'='*90}")
    print(f"METHOD 2: MRMR selection")
    print(f"{'='*90}")

    for K in [5, 8, 10, 12, 15, 20]:
        sel_idx, sel_names = mrmr_select(X_all, y_v, all_names, K)
        r2, mae, rho, a = best_alpha(X_all[:, sel_idx], y_v)
        n_pb = sum(1 for n in sel_names if n.startswith('pb_'))
        n_gm = sum(1 for n in sel_names if n.startswith('gm_'))
        n_dm = sum(1 for n in sel_names if n.startswith('demo_'))
        print(f"  MRMR-{K:>2d} (pb:{n_pb},gm:{n_gm},dm:{n_dm}):  α={a}  R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}")

    # ── Method 3: Forward selection ──
    print(f"\n{'='*90}")
    print(f"METHOD 3: Forward selection (greedy LOO R² improvement)")
    print(f"{'='*90}")

    sel_idx, sel_names, history = forward_selection(X_all, y_v, all_names, max_features=20, alpha=20)
    print(f"  Forward selection path:")
    for step, name, r2_val in history:
        src = 'PB' if name.startswith('pb_') else ('GM' if name.startswith('gm_') else 'DM')
        print(f"    Step {step:>2d}: +{name:35s} [{src}]  R²={r2_val:.4f}")

    # Best from forward selection
    best_step = max(history, key=lambda x: x[2])
    best_k = best_step[0]
    best_idx = sel_idx[:best_k]
    r2, mae, rho, a = best_alpha(X_all[:, best_idx], y_v)
    print(f"\n  Best forward selection: {best_k}f  α={a}  R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}")
    print(f"  Selected features: {[all_names[i] for i in best_idx]}")

    # ── Summary ──
    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    print(f"  PerBout-Top20+Demo(5):      R²=0.441  MAE=191  (baseline)")
    print(f"  Goldman+Demo(5):            see above")
    print(f"  Best combination:           see above")

    print(f"\nDone in {time.time()-t0:.0f}s")
