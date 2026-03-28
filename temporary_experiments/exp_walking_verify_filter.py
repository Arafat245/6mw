#!/usr/bin/env python3
"""
Test if filtering walking bouts with walking_verify.py improves PerBout features.
Compare: A) Current (no filter) vs B) walking_verify filtered bouts.
Same feature extraction and prediction pipeline, only bout filtering differs.
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

from home.extract_clinicfree_features import (detect_walking_bouts, extract_bout_features,
                                               extract_activity_features)
from notebooks.walking_verify import verify_walking_segment_df

HOME_DIR = BASE / 'csv_home_daytime'
FS = 30


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def verify_bout(xyz, fs=30):
    """Check if bout is walking using walking_verify."""
    if len(xyz) < fs * 5:
        return False
    ts = np.arange(len(xyz)) / fs
    df = pd.DataFrame({'Timestamp': ts, 'X': xyz[:, 0], 'Y': xyz[:, 1], 'Z': xyz[:, 2]})
    result = verify_walking_segment_df(df)
    metrics = dict(zip(result['metric'], result['value']))
    return bool(metrics.get('is_walking', False))


def aggregate_bout_features(bout_feats, gait_feat_names):
    """Aggregate per-bout features into subject-level features."""
    if not bout_feats:
        return None
    arr = np.array([[bf.get(k, np.nan) for k in gait_feat_names] for bf in bout_feats])
    row = {}
    for j, name in enumerate(gait_feat_names):
        col = arr[:, j]; valid = col[np.isfinite(col)]
        if len(valid) < 2: continue
        row[f'g_{name}_med'] = np.median(valid)
        row[f'g_{name}_iqr'] = np.percentile(valid, 75) - np.percentile(valid, 25)
        row[f'g_{name}_p10'] = np.percentile(valid, 10)
        row[f'g_{name}_p90'] = np.percentile(valid, 90)
        row[f'g_{name}_max'] = np.max(valid)
        row[f'g_{name}_cv'] = np.std(valid) / (np.mean(valid) + 1e-12)
    row['g_n_valid_bouts'] = len(bout_feats)
    row['g_total_walk_sec'] = sum(bf.get('duration_sec', 0) for bf in bout_feats)
    durs = [bf.get('duration_sec', 0) for bf in bout_feats]
    row['g_mean_bout_dur'] = np.mean(durs)
    row['g_bout_dur_cv'] = np.std(durs) / (np.mean(durs) + 1e-12) if np.mean(durs) > 0 else 0
    return row


def nested_loo_spearman(X, y, X_demo, K=20, alpha=20):
    """Nested LOO with Spearman inside each fold."""
    n = len(y)
    pr = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, dtype=bool); tr[i] = False
        corrs = [abs(spearmanr(X[tr, j], y[tr])[0]) if np.std(X[tr, j]) > 1e-12 else 0
                 for j in range(X.shape[1])]
        sel = np.argsort(corrs)[::-1][:K]
        X_tr = np.column_stack([X[tr][:, sel], X_demo[tr]])
        X_te = np.column_stack([X[i:i+1][:, sel], X_demo[i:i+1]])
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_tr), y[tr])
        pr[i] = m.predict(sc.transform(X_te))[0]
    return r2_score(y, pr), mean_absolute_error(y, pr), spearmanr(y, pr)[0]


if __name__ == '__main__':
    t0 = time.time()

    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                             for _, r in ids101.iterrows()])
    ids_v = ids101[clinic_valid].reset_index(drop=True)
    y = ids_v['sixmwd'].values.astype(float)
    n = len(y)

    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo4 = impute(p[['cohort_M', 'Age', 'Sex', 'BMI']].values.astype(float))

    print(f"Walking Verify Filter Experiment (n={n})")
    print(f"{'='*80}\n")

    # Extract features with both strategies
    gait_feat_names = None
    results_A = []  # no filter
    results_B = []  # walking_verify filter
    bout_stats = []

    for i, (_, r) in enumerate(ids_v.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        fp = HOME_DIR / fn

        if not fp.exists():
            results_A.append(None); results_B.append(None)
            bout_stats.append({'n_all': 0, 'n_verified': 0})
            continue

        xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
        bouts = detect_walking_bouts(xyz, FS, min_bout_sec=10, merge_gap_sec=5)

        # Strategy A: All bouts (current pipeline)
        feats_A = []
        for s, e in bouts:
            bf = extract_bout_features(xyz[s:e], FS)
            if bf is not None:
                feats_A.append(bf)

        # Strategy B: Only verified bouts
        feats_B = []
        n_verified = 0
        for s, e in bouts:
            if verify_bout(xyz[s:e], FS):
                n_verified += 1
                bf = extract_bout_features(xyz[s:e], FS)
                if bf is not None:
                    feats_B.append(bf)

        bout_stats.append({'n_all': len(feats_A), 'n_verified': len(feats_B)})

        if gait_feat_names is None and feats_A:
            gait_feat_names = sorted(feats_A[0].keys())

        # Aggregate A
        row_A = aggregate_bout_features(feats_A, gait_feat_names) if feats_A else None
        act = extract_activity_features(xyz, FS)
        if row_A and act:
            row_A.update(act)
        results_A.append(row_A)

        # Aggregate B
        row_B = aggregate_bout_features(feats_B, gait_feat_names) if feats_B else None
        if row_B and act:
            row_B.update(act)
        results_B.append(row_B)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{n}] {fn}: all={len(feats_A)}, verified={len(feats_B)}", flush=True)

    # Bout statistics
    bs = pd.DataFrame(bout_stats)
    print(f"\n{'='*80}")
    print(f"BOUT STATISTICS")
    print(f"{'='*80}")
    print(f"  All bouts:      mean={bs['n_all'].mean():.0f}, median={bs['n_all'].median():.0f}, total={bs['n_all'].sum()}")
    print(f"  Verified bouts: mean={bs['n_verified'].mean():.0f}, median={bs['n_verified'].median():.0f}, total={bs['n_verified'].sum()}")
    print(f"  Retention:      {100*bs['n_verified'].sum()/bs['n_all'].sum():.1f}%")

    # Build feature matrices
    valid_A = np.array([r is not None for r in results_A])
    valid_B = np.array([r is not None for r in results_B])
    both = valid_A & valid_B
    nv = both.sum()

    pb_cols = sorted([c for c in results_A[valid_A.tolist().index(True)].keys()
                      if c.startswith('g_') or c.startswith('act_')])

    X_A = impute(pd.DataFrame([r for r, v in zip(results_A, both) if v])[pb_cols].values)
    X_B = impute(pd.DataFrame([r for r, v in zip(results_B, both) if v])[pb_cols].values)
    y_v = y[both]
    D_v = X_demo4[both]

    print(f"\n  Valid subjects (both strategies): {nv}")
    print(f"  Features: {len(pb_cols)}")

    # Compare correlations
    print(f"\n{'='*80}")
    print(f"TOP FEATURE CORRELATIONS (n={nv})")
    print(f"{'='*80}")
    print(f"{'Feature':<40s}  {'A (all) ρ':>10s}  {'B (verified) ρ':>14s}  {'Diff':>6s}")
    print('-' * 75)

    corr_A = [(j, abs(spearmanr(X_A[:, j], y_v)[0]), spearmanr(X_A[:, j], y_v)[0]) for j in range(len(pb_cols))]
    corr_B = [(j, abs(spearmanr(X_B[:, j], y_v)[0]), spearmanr(X_B[:, j], y_v)[0]) for j in range(len(pb_cols))]
    corr_A.sort(key=lambda x: x[1], reverse=True)

    improved = 0
    for j, abs_rho_a, rho_a in corr_A[:20]:
        rho_b = corr_B[j][2]
        abs_b = abs(rho_b)
        diff = abs_b - abs_rho_a
        marker = '↑' if diff > 0.01 else ('↓' if diff < -0.01 else ' ')
        if diff > 0: improved += 1
        if abs_rho_a > 0.25:
            print(f"  {pb_cols[j]:<38s}  {rho_a:>+10.3f}  {rho_b:>+14.3f}  {diff:>+6.3f} {marker}")

    print(f"\n  Features with improved |ρ|: {improved}/20 top features")

    # Prediction
    print(f"\n{'='*80}")
    print(f"PREDICTION (Spearman inside LOO, n={nv})")
    print(f"{'='*80}")

    for K in [15, 20, 25]:
        for alpha in [10, 20]:
            r2_a, mae_a, rho_a = nested_loo_spearman(X_A, y_v, D_v, K, alpha)
            r2_b, mae_b, rho_b = nested_loo_spearman(X_B, y_v, D_v, K, alpha)
            diff = r2_b - r2_a
            marker = '↑' if diff > 0.005 else ('↓' if diff < -0.005 else '=')
            print(f"  K={K} α={alpha:>2d}: A(all) R²={r2_a:.4f} MAE={mae_a:.0f} ρ={rho_a:.3f}  |  "
                  f"B(verified) R²={r2_b:.4f} MAE={mae_b:.0f} ρ={rho_b:.3f}  {marker}")

    print(f"\nDone in {time.time()-t0:.0f}s")
