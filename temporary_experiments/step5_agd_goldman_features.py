#!/usr/bin/env python3
"""
Step 5: Extract Goldman features from AGD files (ActiLife step counts).
Compare with per-bout features, with and without feature selection, no demo.
"""
import sqlite3, os, re, pickle, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
NPZ_DIR = BASE / 'csv_home_daytime_npz'
FS = 30


def load_agd(path):
    """Load AGD file (SQLite) and return per-minute DataFrame."""
    conn = sqlite3.connect(str(path))
    df = pd.read_sql('SELECT * FROM data', conn)
    conn.close()
    net_epoch = datetime(1, 1, 1)
    df['datetime'] = df['dataTimestamp'].apply(lambda t: net_epoch + timedelta(microseconds=t / 10))
    df['hour'] = df['datetime'].apply(lambda d: d.hour)
    df['date'] = df['datetime'].apply(lambda d: d.date())
    return df


def extract_goldman_from_agd(df):
    """Extract Goldman features from per-minute AGD data."""
    f = {}
    steps = df['steps'].values

    # Waking hours (7AM-10PM)
    waking = df[(df['hour'] >= 7) & (df['hour'] < 22)]
    waking_steps = waking['steps'].values

    if len(waking_steps) < 60:
        return f

    # MSR: Maximum Step Rate (max per-minute steps)
    f['agd_msr'] = float(np.max(waking_steps))

    # Walking minutes: steps >= 30 (Goldman used HMM, we approximate)
    walking = waking_steps[waking_steps >= 30]
    if len(walking) >= 5:
        f['agd_hwsr'] = float(np.mean(walking))
        f['agd_hwsr_median'] = float(np.median(walking))
        f['agd_hwsr_std'] = float(np.std(walking))
    else:
        # Fallback
        active = waking_steps[waking_steps > 0]
        if len(active) >= 3:
            f['agd_hwsr'] = float(np.mean(active))
            f['agd_hwsr_median'] = float(np.median(active))
            f['agd_hwsr_std'] = float(np.std(active))
        else:
            f['agd_hwsr'] = 0
            f['agd_hwsr_median'] = 0
            f['agd_hwsr_std'] = 0

    # MSR + HWSR
    f['agd_msr_plus_hwsr'] = f['agd_msr'] + f['agd_hwsr']

    # Average daily steps
    daily_steps = waking.groupby('date')['steps'].sum()
    # Valid days: >= 10h wear (>= 600 waking minutes with any data)
    daily_wear = waking.groupby('date').size()
    valid_days = daily_wear[daily_wear >= 600].index
    if len(valid_days) >= 1:
        valid_daily_steps = daily_steps[daily_steps.index.isin(valid_days)]
        f['agd_avg_daily_steps'] = float(valid_daily_steps.mean())
        f['agd_n_valid_days'] = len(valid_days)
    else:
        f['agd_avg_daily_steps'] = float(daily_steps.mean())
        f['agd_n_valid_days'] = len(daily_steps)

    # Activity levels (from step counts)
    f['agd_pct_sedentary'] = float(np.mean(waking_steps == 0))
    f['agd_pct_light'] = float(np.mean((waking_steps > 0) & (waking_steps < 30)))
    f['agd_pct_walking'] = float(np.mean(waking_steps >= 30))
    f['agd_pct_vigorous'] = float(np.mean(waking_steps >= 100))

    # Total MVPA minutes
    total_waking_hours = len(waking_steps) / 60
    f['agd_mvpa_min_per_hr'] = float(np.sum(waking_steps >= 30) / (total_waking_hours + 1e-12))

    # Step rate percentiles (active minutes only)
    active = waking_steps[waking_steps > 0]
    if len(active) >= 5:
        f['agd_step_rate_p90'] = float(np.percentile(active, 90))
        f['agd_step_rate_p75'] = float(np.percentile(active, 75))
        f['agd_step_rate_p50'] = float(np.median(active))
        f['agd_step_rate_cv'] = float(np.std(active) / (np.mean(active) + 1e-12))

    # Walking bout analysis (consecutive walking minutes)
    walk_mask = waking_steps >= 30
    bout_durs = []
    in_b, bs = False, 0
    for i in range(len(walk_mask)):
        if walk_mask[i] and not in_b:
            bs = i; in_b = True
        elif not walk_mask[i] and in_b:
            bout_durs.append(i - bs)
            in_b = False
    if in_b:
        bout_durs.append(len(walk_mask) - bs)

    f['agd_n_walk_bouts'] = len(bout_durs)
    if bout_durs:
        f['agd_walk_bout_mean'] = float(np.mean(bout_durs))
        f['agd_walk_bout_max'] = float(np.max(bout_durs))
        f['agd_walk_bout_cv'] = float(np.std(bout_durs) / (np.mean(bout_durs) + 1e-12))
        f['agd_total_walk_min'] = float(np.sum(bout_durs))
    else:
        f['agd_walk_bout_mean'] = 0
        f['agd_walk_bout_max'] = 0
        f['agd_walk_bout_cv'] = 0
        f['agd_total_walk_min'] = 0

    # Day-to-day variability
    if len(valid_days) >= 2:
        valid_daily = daily_steps[daily_steps.index.isin(valid_days)]
        f['agd_daily_steps_cv'] = float(valid_daily.std() / (valid_daily.mean() + 1e-12))
        # MSR per day
        daily_msr = waking[waking['date'].isin(valid_days)].groupby('date')['steps'].max()
        f['agd_daily_msr_mean'] = float(daily_msr.mean())
        f['agd_daily_msr_cv'] = float(daily_msr.std() / (daily_msr.mean() + 1e-12))

    return f


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_ridge_best(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 10)
    for a in alphas:
        preds = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=a)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            preds[te] = m.predict(sc.transform(X[te]))
        r2 = r2_score(y, preds)
        if r2 > best[0]:
            best = (r2, mean_absolute_error(y, preds), spearmanr(y, preds)[0], a)
    return best


def loo_spearman_select(X, y, K, alpha=20):
    n_feat = X.shape[1]
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        corrs = [abs(spearmanr(X[tr, j], y[tr])[0]) if np.std(X[tr, j]) > 0 else 0 for j in range(n_feat)]
        top_k = sorted(range(n_feat), key=lambda j: corrs[j], reverse=True)[:K]
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr][:, top_k]), y[tr])
        preds[te] = m.predict(sc.transform(X[te][:, top_k]))
    return r2_score(y, preds), mean_absolute_error(y, preds), spearmanr(y, preds)[0]


if __name__ == '__main__':
    t0 = time.time()

    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(y)

    # Build AGD path map
    accel = BASE / 'Accel files'
    agd_map = {}
    for d in accel.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r'^([CM])(\d+)', d.name)
        if not m:
            continue
        cohort, sid = m.group(1), int(m.group(2))
        key = f'{cohort}{sid:02d}' if sid < 100 else f'{cohort}{sid}'
        agd_files = list(d.glob('*.agd'))
        if agd_files:
            agd_map[key] = str(agd_files[0])

    # Extract Goldman features from AGD
    print(f"=== Extracting Goldman features from AGD files ===", flush=True)
    agd_rows = []
    for i, (_, r) in enumerate(subj_df.iterrows()):
        key = r['key']
        if key in agd_map:
            try:
                df_agd = load_agd(agd_map[key])
                gf = extract_goldman_from_agd(df_agd)
            except Exception as ex:
                gf = {}
                print(f"  WARNING: {key} failed: {ex}")
        else:
            gf = {}
        agd_rows.append(gf)
        if (i + 1) % 20 == 0:
            msr = gf.get('agd_msr', 0)
            hwsr = gf.get('agd_hwsr', 0)
            print(f"  [{i+1}/{n}] {key}: MSR={msr:.0f} HWSR={hwsr:.1f}", flush=True)

    agd_df = pd.DataFrame(agd_rows)
    agd_cols = list(agd_df.columns)
    X_agd = impute(agd_df.values.astype(float))
    print(f"  AGD Goldman features: {len(agd_cols)}", flush=True)

    # Correlations
    print(f"\n  Correlations with 6MWD:", flush=True)
    for j, name in enumerate(agd_cols):
        rho_s = spearmanr(X_agd[:, j], y)[0]
        rho_p = pearsonr(X_agd[:, j], y)[0]
        if abs(rho_s) > 0.15:
            print(f"    {name:30s}  Spearman={rho_s:+.3f}  Pearson={rho_p:+.3f}")

    # Load per-bout features
    orig_df = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
    orig_cols = [c for c in orig_df.columns if c != 'key']
    X_orig = impute(orig_df[orig_cols].values.astype(float))

    # Demo(4)
    demo_data = pd.read_excel(BASE / 'Accel files' / 'PedMSWalkStudy_Demographic.xlsx')
    demo_data['cohort'] = demo_data['ID'].str.extract(r'^([A-Z])')[0]
    demo_data['subj_id'] = demo_data['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo_data, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    demo_cols = ['cohort_POMS', 'Age', 'Sex', 'BMI']
    X_demo = impute(p[demo_cols].values.astype(float))

    # === WITHOUT FEATURE SELECTION, NO DEMO ===
    print(f"\n{'='*70}", flush=True)
    print(f"WITHOUT FEATURE SELECTION, NO DEMO", flush=True)
    print(f"{'='*70}", flush=True)
    for name, X_test in [
        ('AGD Goldman', X_agd),
        ('Per-bout (153f)', X_orig),
        ('AGD + Per-bout', np.column_stack([X_agd, X_orig])),
    ]:
        r2, mae, rho, a = loo_ridge_best(X_test, y)
        print(f"  {name:30s} {X_test.shape[1]:3d}f  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}  a={a}", flush=True)

    # === WITH FEATURE SELECTION, NO DEMO ===
    print(f"\n{'='*70}", flush=True)
    print(f"WITH SPEARMAN SELECTION (inside LOO), NO DEMO", flush=True)
    print(f"{'='*70}", flush=True)
    for K in [5, 10, 15, 20]:
        for name, X_test in [
            ('AGD Goldman', X_agd),
            ('Per-bout', X_orig),
            ('AGD + Per-bout', np.column_stack([X_agd, X_orig])),
        ]:
            if K > X_test.shape[1]:
                continue
            r2, mae, rho = loo_spearman_select(X_test, y, K, alpha=20)
            print(f"  K={K:2d}  {name:20s} {X_test.shape[1]:3d}f  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}", flush=True)
        print()

    # === WITH DEMO(4) ===
    print(f"{'='*70}", flush=True)
    print(f"WITH SPEARMAN SELECTION (inside LOO) + DEMO(4)", flush=True)
    print(f"{'='*70}", flush=True)
    n_demo = len(demo_cols)
    for K in [5, 10, 15, 20]:
        for name, X_accel in [
            ('AGD Goldman', X_agd),
            ('Per-bout', X_orig),
            ('AGD + Per-bout', np.column_stack([X_agd, X_orig])),
        ]:
            if K > X_accel.shape[1]:
                continue
            X_all = np.column_stack([X_accel, X_demo])
            n_accel = X_accel.shape[1]
            demo_idx = list(range(n_accel, n_accel + n_demo))
            preds = np.zeros(n)
            for tr, te in LeaveOneOut().split(X_all):
                corrs = [abs(spearmanr(X_all[tr, j], y[tr])[0]) if np.std(X_all[tr, j]) > 0 else 0 for j in range(n_accel)]
                top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K]
                selected = top_k + demo_idx
                sc = StandardScaler(); m = Ridge(alpha=20)
                m.fit(sc.fit_transform(X_all[tr][:, selected]), y[tr])
                preds[te] = m.predict(sc.transform(X_all[te][:, selected]))
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rho = spearmanr(y, preds)[0]
            print(f"  K={K:2d}+Demo4  {name:20s}  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}", flush=True)
        print()

    # Save
    agd_df.insert(0, 'key', subj_df['key'].values)
    agd_df.to_csv(BASE / 'feats' / 'agd_goldman_features.csv', index=False)
    print(f"\n  Saved feats/agd_goldman_features.csv ({agd_df.shape})", flush=True)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)
