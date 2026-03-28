#!/usr/bin/env python3
"""
Compare feature screening methods on all 153 PerBout + 4 Goldman + 5 Demo features.
Each method screens to top-30, then forward selection finds the best combination.

Methods:
1. Spearman top-30 → forward selection
2. MRMR top-30 → forward selection
3. Mutual Information top-30 → forward selection
4. Random Forest importance top-30 → forward selection
5. Spearman top-40 → forward selection
6. Forward selection on ALL 158 features directly (expensive but thorough)
"""
import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


def eval_loo(X, y, alpha=10):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, pr), mean_absolute_error(y, pr), spearmanr(y, pr)[0]


def forward_selection(X, y, names, max_features=20):
    """Greedy forward selection with alpha search at each step."""
    n_feat = X.shape[1]
    selected = []
    remaining = list(range(n_feat))
    history = []

    for step in range(min(max_features, n_feat)):
        best_r2, best_f = -999, remaining[0]
        for f in remaining:
            trial = selected + [f]
            # Try multiple alphas
            for alpha in [5, 10, 20, 50]:
                r2, _, _ = eval_loo(X[:, trial], y, alpha)
                if r2 > best_r2:
                    best_r2, best_f = r2, f
        selected.append(best_f)
        remaining.remove(best_f)
        history.append((len(selected), names[best_f], best_r2))

        # Stop if no improvement for 3 steps
        if len(history) >= 4 and all(history[-i][2] <= history[-4][2] for i in range(1, 4)):
            break

    # Find peak
    best_step = max(history, key=lambda x: x[2])
    return selected[:best_step[0]], history


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


# ══════════════════════════════════════════════════════════════════
# SCREENING METHODS
# ══════════════════════════════════════════════════════════════════

def screen_spearman(X, y, names, K):
    scores = [abs(spearmanr(X[:, j], y)[0]) if np.std(X[:, j]) > 1e-12 else 0
              for j in range(X.shape[1])]
    idx = np.argsort(scores)[::-1][:K]
    return idx.tolist()


def screen_mrmr(X, y, names, K):
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
    return selected


def screen_mi(X, y, names, K):
    scores = mutual_info_regression(X, y, random_state=42)
    idx = np.argsort(scores)[::-1][:K]
    return idx.tolist()


def screen_rf(X, y, names, K):
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
    rf.fit(X, y)
    idx = np.argsort(rf.feature_importances_)[::-1][:K]
    return idx.tolist()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

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

    # Demo
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float))
    demo_names = ['demo_cohort', 'demo_age', 'demo_sex', 'demo_height', 'demo_bmi']

    # All 153 PerBout features
    cf_csv = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
    pb_idx = []
    for _, r in ids_v.iterrows():
        match = ids101[(ids101['cohort'] == r['cohort']) & (ids101['subj_id'] == r['subj_id']) & (ids101['sixmwd'] == r['sixmwd'])]
        if len(match) > 0: pb_idx.append(match.index[0])
    cf_sub = cf_csv.iloc[pb_idx].reset_index(drop=True)
    pb_cols = [c for c in cf_sub.columns if c.startswith('g_') or c.startswith('act_') or c.startswith('sec_')]
    X_pb_all = impute(cf_sub[pb_cols].values.astype(float))
    pb_names = [f'pb_{c}' for c in pb_cols]

    # Goldman features
    gt3x = np.load(BASE / 'temporary_experiments' / 'goldman_gt3x_cache.npz', allow_pickle=True)
    gt3x_feats = list(gt3x['features'])
    gm_cols = ['msr', 'hwsr', 'avg_daily_steps', 'msr_plus_hwsr']
    X_gm = impute(np.array([[f[k] for k in gm_cols] for f in gt3x_feats]))
    gm_names = [f'gm_{c}' for c in gm_cols]

    # Combined pool
    X_all = np.column_stack([X_demo, X_gm, X_pb_all])
    all_names = demo_names + gm_names + pb_names
    nf_total = X_all.shape[1]

    # Remove constant features
    valid_cols = [j for j in range(nf_total) if np.std(X_all[:, j]) > 1e-12]
    X_all = X_all[:, valid_cols]
    all_names = [all_names[j] for j in valid_cols]
    nf_total = len(all_names)

    print(f"n={n}, Total features: {nf_total} (Demo:{len(demo_names)}, Goldman:{len(gm_cols)}, PerBout:{len(pb_cols)})")
    print(f"{'='*90}\n")

    # ── Run each screening method ──
    methods = [
        ('Spearman-30', lambda X, y, n: screen_spearman(X, y, n, 30)),
        ('MRMR-30', lambda X, y, n: screen_mrmr(X, y, n, 30)),
        ('MI-30', lambda X, y, n: screen_mi(X, y, n, 30)),
        ('RF-30', lambda X, y, n: screen_rf(X, y, n, 30)),
        ('Spearman-40', lambda X, y, n: screen_spearman(X, y, n, 40)),
    ]

    results = []

    for method_name, screen_fn in methods:
        print(f"{'─'*90}")
        print(f"{method_name}: Screening → forward selection")
        print(f"{'─'*90}")

        t1 = time.time()
        screened_idx = screen_fn(X_all, y, all_names)
        screened_names = [all_names[i] for i in screened_idx]

        # Show what was selected
        n_dm = sum(1 for n in screened_names if n.startswith('demo_'))
        n_gm = sum(1 for n in screened_names if n.startswith('gm_'))
        n_pb = sum(1 for n in screened_names if n.startswith('pb_'))
        print(f"  Screened: {len(screened_idx)} features (Demo:{n_dm}, Goldman:{n_gm}, PerBout:{n_pb})")

        # Forward selection on screened features
        X_screened = X_all[:, screened_idx]
        sel_idx, history = forward_selection(X_screened, y, screened_names, max_features=20)

        best_step = max(history, key=lambda x: x[2])
        best_k = best_step[0]
        best_r2 = best_step[2]

        # Get final features
        final_idx = sel_idx[:best_k]
        final_names = [screened_names[i] for i in final_idx]
        X_final = X_screened[:, final_idx]
        r2, mae, rho = eval_loo(X_final, y, alpha=10)

        n_dm_f = sum(1 for n in final_names if n.startswith('demo_'))
        n_gm_f = sum(1 for n in final_names if n.startswith('gm_'))
        n_pb_f = sum(1 for n in final_names if n.startswith('pb_'))

        print(f"  Forward selection path:")
        for step, name, r2_val in history[:best_k + 2]:
            src = 'DM' if name.startswith('demo_') else ('GM' if name.startswith('gm_') else 'PB')
            marker = ' ←' if step == best_k else ''
            print(f"    Step {step:>2d}: +{name:40s} [{src}]  R²={r2_val:.4f}{marker}")

        print(f"  Best: {best_k}f (DM:{n_dm_f},GM:{n_gm_f},PB:{n_pb_f})  R²={best_r2:.4f}  ({time.time()-t1:.0f}s)")

        results.append({
            'Method': method_name, '#screened': len(screened_idx),
            'Best #f': best_k, 'R²': round(best_r2, 4),
            'Demo': n_dm_f, 'Goldman': n_gm_f, 'PerBout': n_pb_f,
            'Features': final_names,
        })

    # Method 6: Forward selection on ALL features (no screening)
    print(f"\n{'─'*90}")
    print(f"Forward-ALL: Forward selection on all {nf_total} features (no screening)")
    print(f"{'─'*90}")
    t1 = time.time()
    sel_idx, history = forward_selection(X_all, y, all_names, max_features=20)
    best_step = max(history, key=lambda x: x[2])
    best_k = best_step[0]
    final_idx = sel_idx[:best_k]
    final_names = [all_names[i] for i in final_idx]
    n_dm_f = sum(1 for n in final_names if n.startswith('demo_'))
    n_gm_f = sum(1 for n in final_names if n.startswith('gm_'))
    n_pb_f = sum(1 for n in final_names if n.startswith('pb_'))

    print(f"  Forward selection path:")
    for step, name, r2_val in history[:best_k + 2]:
        src = 'DM' if name.startswith('demo_') else ('GM' if name.startswith('gm_') else 'PB')
        marker = ' ←' if step == best_k else ''
        print(f"    Step {step:>2d}: +{name:40s} [{src}]  R²={r2_val:.4f}{marker}")

    print(f"  Best: {best_k}f (DM:{n_dm_f},GM:{n_gm_f},PB:{n_pb_f})  R²={best_step[2]:.4f}  ({time.time()-t1:.0f}s)")

    results.append({
        'Method': 'Forward-ALL', '#screened': nf_total,
        'Best #f': best_k, 'R²': round(best_step[2], 4),
        'Demo': n_dm_f, 'Goldman': n_gm_f, 'PerBout': n_pb_f,
        'Features': final_names,
    })

    # ── Summary ──
    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    print(f"{'Method':<20s} {'#screened':>9s} {'#final':>6s} {'R²':>8s}  {'DM':>3s} {'GM':>3s} {'PB':>3s}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['R²'], reverse=True):
        print(f"  {r['Method']:<18s} {r['#screened']:>9d} {r['Best #f']:>6d} {r['R²']:>8.4f}  {r['Demo']:>3d} {r['Goldman']:>3d} {r['PerBout']:>3d}")

    print(f"\n  Previous best (Spearman-20 → forward): R²=0.502, 13 features")
    print(f"\nDone in {time.time()-t0:.0f}s")

    # Save
    pd.DataFrame(results).to_csv(BASE / 'temporary_experiments' / 'screening_comparison.csv', index=False)
