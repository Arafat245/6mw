#!/usr/bin/env python3
"""
Extract clinic Bout+Act features:
  - Split 6MWT into 60s windows, extract 20 per-bout features per window,
    aggregate across windows (120 gait + 2 meta = 122 bout-aggregated features).
  - Extract 29 activity features from the full trimmed 6MWT recording.

Input:  csv_raw2/*.csv
Output: feats/clinic_perbout_features.csv (151 features x 101 subjects)

Run:  python clinic/extract_perbout_features.py
"""
import math, time, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent
RAW = BASE / 'csv_raw2'
FS = 30
WIN_SEC = 60

from home.step2_extract_features import preprocess_segment, extract_bout_features, extract_activity_features


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def find_file(directory, cohort, subj_id):
    key = f'{cohort}{int(subj_id):02d}'
    for f in directory.glob(f'{key}_*.csv'):
        return f
    return None


def aggregate_bout_feats(bout_feats):
    row = {}
    if not bout_feats: return row
    gfn = sorted(bout_feats[0].keys())
    arr = np.array([[bf.get(k, np.nan) for k in gfn] for bf in bout_feats])
    for j, name in enumerate(gfn):
        col = arr[:, j]; valid = col[np.isfinite(col)]
        if len(valid) < 2: continue
        row[f'g_{name}_med'] = np.median(valid)
        row[f'g_{name}_iqr'] = np.percentile(valid, 75) - np.percentile(valid, 25)
        row[f'g_{name}_p10'] = np.percentile(valid, 10)
        row[f'g_{name}_p90'] = np.percentile(valid, 90)
        row[f'g_{name}_max'] = np.max(valid)
        row[f'g_{name}_cv'] = np.std(valid) / (np.mean(valid) + 1e-12)
    row['g_total_walk_sec'] = sum(bf.get('duration_sec', 0) for bf in bout_feats)
    durs = [bf.get('duration_sec', 0) for bf in bout_feats]
    row['g_mean_bout_dur'] = np.mean(durs)
    # g_bout_dur_cv removed — duplicate of g_duration_sec_cv (VIF = 10^12).
    return row


if __name__ == '__main__':
    t0 = time.time()
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = (ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44]))
    ids101 = ids[~excl].reset_index(drop=True)
    n = len(ids101)
    print(f"Extracting clinic PerBout features ({WIN_SEC}s windows) for {n} subjects...")

    all_rows = []
    for i, (_, r) in enumerate(ids101.iterrows()):
        fp = find_file(RAW, r['cohort'], r['subj_id'])
        raw_df = pd.read_csv(fp)
        if 'Timestamp' in raw_df.columns:
            dt = np.diff(raw_df['Timestamp'].values[:1000])
            dt = dt[dt > 0]
            fs = round(1.0 / np.median(dt)) if len(dt) > 0 else 30
        else:
            fs = 30
        xyz = raw_df[['X', 'Y', 'Z']].values.astype(np.float64)

        trim = int(10 * fs)
        if 2 * trim < len(xyz):
            xyz = xyz[trim:len(xyz) - trim]

        win_samples = int(WIN_SEC * fs)
        bout_feats = []
        for start in range(0, len(xyz) - win_samples + 1, win_samples):
            seg = xyz[start:start + win_samples]
            feats = extract_bout_features(seg, fs)
            if feats is not None:
                bout_feats.append(feats)

        row = aggregate_bout_feats(bout_feats)
        act = extract_activity_features(xyz, fs)
        if act:
            row.update(act)
        all_rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}]", flush=True)

    feat_df = pd.DataFrame(all_rows)
    key_col = ids101.apply(lambda r: f"{r['cohort']}{int(r['subj_id']):02d}", axis=1)
    feat_df.insert(0, 'key', key_col.values)
    feat_df.to_csv(BASE / 'feats' / 'clinic_perbout_features.csv', index=False)
    print(f"\nSaved feats/clinic_perbout_features.csv ({feat_df.shape})")
    print(f"Done in {time.time()-t0:.0f}s")
