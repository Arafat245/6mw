#!/usr/bin/env python3
"""
Extract home Gait(11), CWT(28), WalkSway(12) features using VM-based extraction.
Top-10 clean bouts ≥60s (drift≤0.5g, orient≤10°), per-bout aggregation.

Input:  walking_bouts/{subject_id}/bout_*.csv
Output: feats/home_gait_features.csv (66 features x 101 subjects)
        feats/home_cwt_features.csv (168 features x 101 subjects)
        feats/home_walksway_features.csv (72 features x 101 subjects)

Run:  python home/extract_gait_cwt_ws_features.py
"""
import math, time, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent
BOUT_DIR = BASE / 'walking_bouts'
NPZ_DIR = BASE / 'home_full_recording_npz'
FS = 30

from clinic.reproduce_c2 import extract_gait10, extract_cwt
from clinic.extract_walking_sway import extract_walking_sway


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def vm_to_clinic_df(vm, fs):
    """Create clinic-like DataFrame from raw VM signal (no mean subtraction)."""
    b, a = butter(4, [0.25, 2.5], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b, a, vm)
    enmo = np.maximum(vm - 1.0, 0.0)
    return pd.DataFrame({
        'AP': vm, 'ML': vm, 'VT': vm,
        'AP_bp': vm_bp, 'ML_bp': vm_bp, 'VT_bp': vm_bp,
        'VM_dyn': vm, 'VM_raw': vm,
        'ENMO': enmo, 'fs': fs,
    })


def bout_quality(xyz, fs=30):
    first = xyz[:fs].mean(axis=0); last = xyz[-fs:].mean(axis=0)
    drift = np.linalg.norm(last - first)
    orient = 0.0
    if len(xyz) >= 60:
        half = len(xyz) // 2
        g1 = xyz[:half].mean(axis=0); g2 = xyz[half:].mean(axis=0)
        cos_a = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-12)
        orient = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    return drift, orient


def get_clean_topN(subj_key, N=10, min_sec=60, max_drift=0.5, max_orient=10):
    subj_dir = BOUT_DIR / subj_key
    if not subj_dir.exists(): return []
    candidates = []
    for bf in sorted(subj_dir.glob('bout_*.csv')):
        try:
            dur = float(bf.stem.split('_')[-1].replace('s', ''))
            if dur < min_sec: continue
            df = pd.read_csv(bf)
            xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
            drift, orient = bout_quality(xyz)
            if drift <= max_drift and orient <= max_orient:
                candidates.append((dur, bf))
        except: continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [bf for _, bf in candidates[:N]]


def aggregate_feature_dicts(feat_list):
    if not feat_list: return {}
    names = sorted(feat_list[0].keys())
    arr = np.array([[f.get(k, np.nan) for k in names] for f in feat_list])
    row = {}
    for j, name in enumerate(names):
        col = arr[:, j]; valid = col[np.isfinite(col)]
        if len(valid) < 2: continue
        row[f'{name}_med'] = np.median(valid)
        row[f'{name}_iqr'] = np.percentile(valid, 75) - np.percentile(valid, 25)
        row[f'{name}_p10'] = np.percentile(valid, 10)
        row[f'{name}_p90'] = np.percentile(valid, 90)
        row[f'{name}_max'] = np.max(valid)
        row[f'{name}_cv'] = np.std(valid) / (np.mean(valid) + 1e-12)
    return row


if __name__ == '__main__':
    t0 = time.time()
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    n = len(subj_df)
    print(f"Extracting home Gait/CWT/WalkSway (VM-based) for {n} subjects...")
    print(f"Top-10 clean bouts ≥60s, drift≤0.5g, orient≤10°\n")

    gait_rows, cwt_rows, ws_rows = [], [], []

    for i, (_, r) in enumerate(subj_df.iterrows()):
        top_bouts = get_clean_topN(r['key'], N=10, min_sec=60,
                                    max_drift=0.5, max_orient=10)
        gait_per, cwt_per, ws_per = [], [], []

        for bf_path in top_bouts:
            try:
                df = pd.read_csv(bf_path)
                xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
                if len(xyz) < int(10 * FS): continue

                vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)

                # Gait from raw VM
                df_vm = vm_to_clinic_df(vm, FS)
                gf = extract_gait10(df_vm)
                gf['vt_rms_g'] = float(np.sqrt(np.mean(vm**2)))
                gait_per.append(gf)

                # CWT from raw XYZ
                cwt_per.append(extract_cwt(xyz.astype(np.float32)))

                # WalkSway from raw VM
                ws = extract_walking_sway(vm, vm, vm)
                ws['ml_over_enmo'] = (gf.get('ml_rms_g', 0) /
                    gf.get('enmo_mean_g', 1e-12) if gf.get('enmo_mean_g', 0) > 0 else np.nan)
                ws['ml_over_vt'] = 1.0
                ws_per.append(ws)
            except: continue

        gait_rows.append(aggregate_feature_dicts(gait_per))
        cwt_rows.append(aggregate_feature_dicts(cwt_per))
        ws_rows.append(aggregate_feature_dicts(ws_per))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] {r['key']}: {len(top_bouts)} clean bouts", flush=True)

    # Save each feature set
    keys = subj_df['key'].values
    FEATS = BASE / 'feats'

    gait_df = pd.DataFrame(gait_rows)
    gait_df.insert(0, 'key', keys)
    gait_df.to_csv(FEATS / 'home_gait_features.csv', index=False)
    print(f"\nSaved feats/home_gait_features.csv ({gait_df.shape})")

    cwt_df = pd.DataFrame(cwt_rows)
    cwt_df.insert(0, 'key', keys)
    cwt_df.to_csv(FEATS / 'home_cwt_features.csv', index=False)
    print(f"Saved feats/home_cwt_features.csv ({cwt_df.shape})")

    ws_df = pd.DataFrame(ws_rows)
    ws_df.insert(0, 'key', keys)
    ws_df.to_csv(FEATS / 'home_walksway_features.csv', index=False)
    print(f"Saved feats/home_walksway_features.csv ({ws_df.shape})")

    print(f"Done in {time.time()-t0:.0f}s")
