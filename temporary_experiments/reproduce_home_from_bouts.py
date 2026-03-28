#!/usr/bin/env python3
"""
Reproduce best home result starting from saved walking bouts.
Loads bouts from walking_bouts/{subj_id}/bout_*.csv instead of re-detecting.
Activity features loaded from cached CSV (they come from full daytime recording).

Target: R²=0.462, MAE=187, ρ=0.661
Pipeline: bout features (20/bout) → aggregate (124 gait + 4 meta) + activity (29) = 153
         → Spearman Top-20 inside LOO + Demo(4) = 24f, Ridge α=20
"""
import sys, time
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'home'))
from extract_clinicfree_features import extract_bout_features, FS

BOUT_DIR = BASE / 'walking_bouts'


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all():
            X[:, j] = 0
        elif m.any():
            X[m, j] = np.nanmedian(X[~m, j])
    return X


def load_bouts_and_extract(subj_id):
    """Load saved walking bouts for a subject, extract per-bout features, aggregate."""
    subj_dir = BOUT_DIR / subj_id
    if not subj_dir.exists():
        return {}

    bout_files = sorted(subj_dir.glob('bout_*.csv'))
    if not bout_files:
        return {}

    bout_feats = []
    for bf in bout_files:
        xyz = pd.read_csv(bf, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
        feats = extract_bout_features(xyz, FS)
        if feats is not None:
            bout_feats.append(feats)

    if not bout_feats:
        return {}

    # Aggregate across bouts: 6 stats × 20 features = 120 + 4 meta
    gait_feat_names = sorted(bout_feats[0].keys())
    arr = np.array([[bf.get(k, np.nan) for k in gait_feat_names] for bf in bout_feats])

    row = {}
    for j, name in enumerate(gait_feat_names):
        col = arr[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) < 2:
            continue
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


if __name__ == '__main__':
    t0 = time.time()

    # Load subjects, exclude M22/M44
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = (ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44]))
    ids101 = ids[~excl].reset_index(drop=True)
    y = ids101['sixmwd'].values.astype(float)
    n = len(y)
    print(f"Subjects: {n}")

    # Step 1: Extract gait features from saved walking bouts
    print("Extracting per-bout gait features from walking_bouts/...")
    gait_rows = []
    for i, (_, r) in enumerate(ids101.iterrows()):
        subj_id = f"{r['cohort']}{int(r['subj_id']):02d}"
        row = load_bouts_and_extract(subj_id)
        gait_rows.append(row)
        print(f"  [{i+1}/{n}] {subj_id}: {row.get('g_n_valid_bouts', 0)} valid bouts", flush=True)

    gait_df = pd.DataFrame(gait_rows)
    gait_cols = [c for c in gait_df.columns if c.startswith('g_')]
    print(f"Gait features from bouts: {len(gait_cols)} columns")
    print(f"Gait extraction took {time.time() - t0:.1f}s")

    # Step 2: Load activity features from cached CSV (from full daytime recording)
    cf_csv = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
    act_cols = [c for c in cf_csv.columns if c.startswith('act_')]

    # Match rows by subject order (same ordering as ids101)
    act_rows = []
    for _, r in ids101.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        match = cf_csv[cf_csv['fn'] == fn]
        if len(match) > 0:
            act_rows.append(match[act_cols].iloc[0].to_dict())
        else:
            act_rows.append({})
    act_df = pd.DataFrame(act_rows)
    print(f"Activity features from cache: {len(act_cols)} columns")

    # Combine gait + activity
    df = pd.concat([gait_df, act_df], axis=1)
    pb_cols = [c for c in df.columns if c.startswith('g_') or c.startswith('act_')]
    X_pb = impute(df[pb_cols].values.astype(float))
    print(f"Combined feature matrix: {X_pb.shape[0]} subjects × {X_pb.shape[1]} features")

    # Step 2: Demographics (4 features)
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids101.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo4 = impute(p[['cohort_M', 'Age', 'Sex', 'BMI']].values.astype(float))

    # Step 3: LOO with nested Spearman Top-20 + Demo(4), Ridge α=20
    K = 20
    alpha = 20
    pr = np.zeros(n)

    print(f"\nRunning LOO CV (n={n}, Top-{K} Spearman inside each fold, Ridge α={alpha})...")
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        X_train, y_train = X_pb[tr], y[tr]
        X_test = X_pb[i:i+1]

        # Feature selection on TRAINING data only
        corrs = []
        for j in range(X_train.shape[1]):
            if np.std(X_train[:, j]) < 1e-12:
                corrs.append(0)
            else:
                corrs.append(abs(spearmanr(X_train[:, j], y_train)[0]))
        sel_idx = np.argsort(corrs)[::-1][:K]

        X_tr = np.column_stack([X_train[:, sel_idx], X_demo4[tr]])
        X_te = np.column_stack([X_test[:, sel_idx], X_demo4[i:i+1]])

        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_tr), y_train)
        pr[i] = m.predict(sc.transform(X_te))[0]

    r2 = r2_score(y, pr)
    mae = mean_absolute_error(y, pr)
    rho = spearmanr(y, pr)[0]
    rv = pearsonr(y, pr)[0]

    print(f"\n{'='*60}")
    print(f"Home Result (clinic-free, from saved walking bouts)")
    print(f"  n={n}, Spearman Top-{K} inside LOO, Ridge α={alpha}")
    print(f"  R²={r2:.4f}  MAE={mae:.1f} ft  r={rv:.3f}  ρ={rho:.3f}")
    print(f"  Features: {K} PerBout (selected per fold) + 4 Demo = {K+4}f")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print(f"{'='*60}")
