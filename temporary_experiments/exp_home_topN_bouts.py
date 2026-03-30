#!/usr/bin/env python3
"""
Experiment: Extract Gait(11), CWT(28), WalkSway(12) from Top-N longest bouts (≥60s).
Selects the N longest walking bouts per subject, extracts clinic features per bout,
aggregates across bouts (median, IQR, p10, p90, max, CV).

Tests N=5 and N=10.

Run:  python temporary_experiments/exp_home_topN_bouts.py
"""
import math, time, warnings
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent
BOUT_DIR = BASE / 'walking_bouts'
NPZ_DIR = BASE / 'home_full_recording_npz'
FS = 30
FT2M = 0.3048

from clinic.reproduce_c2 import extract_gait10, extract_cwt
from clinic.extract_walking_sway import extract_walking_sway


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING (raw XYZ → clinic-like DataFrame)
# ══════════════════════════════════════════════════════════════════

def preprocess_to_clinic_df(xyz, fs):
    arr = xyz.copy()
    b, a = butter(4, 0.25, btype='lowpass', fs=fs)
    g_est = np.column_stack([filtfilt(b, a, arr[:, j]) for j in range(3)])
    arr_dyn = arr - g_est
    g_mean = g_est.mean(axis=0); zhat = np.array([0., 0., 1.])
    gvec = g_mean / (np.linalg.norm(g_mean) + 1e-12)
    angle = math.acos(np.clip(float(zhat @ gvec), -1, 1))
    if angle > 1e-4:
        axis = np.cross(gvec, zhat)
        if np.linalg.norm(axis) < 1e-8: axis = np.array([1., 0., 0.])
        ax = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        arr_v = arr_dyn @ R.T
    else:
        arr_v = arr_dyn.copy()
    XY = arr_v[:, :2]; C = np.cov(XY, rowvar=False)
    vals, vecs = np.linalg.eigh(C); ap_dir = vecs[:, np.argmax(vals)]
    theta = math.atan2(float(ap_dir[1]), float(ap_dir[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]); apmlvt = arr_v @ Rz.T
    b, a = butter(4, [0.25, 2.5], btype='bandpass', fs=fs)
    apmlvt_bp = np.column_stack([filtfilt(b, a, apmlvt[:, j]) for j in range(3)])
    vm_raw = np.linalg.norm(arr, axis=1)
    enmo = np.maximum(vm_raw - 1.0, 0.0)
    return pd.DataFrame({
        'AP': apmlvt[:, 0], 'ML': apmlvt[:, 1], 'VT': apmlvt[:, 2],
        'AP_bp': apmlvt_bp[:, 0], 'ML_bp': apmlvt_bp[:, 1], 'VT_bp': apmlvt_bp[:, 2],
        'VM_dyn': np.linalg.norm(apmlvt, axis=1), 'VM_raw': vm_raw,
        'ENMO': enmo, 'fs': fs,
    })


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_eval(X, y, alpha):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    r2 = r2_score(y, pr)
    mae = mean_absolute_error(y * FT2M, pr * FT2M)
    rho = spearmanr(y, pr)[0]
    return r2, mae, rho


def aggregate_feature_dicts(feat_list):
    """Aggregate list of feature dicts into one row with med/iqr/p10/p90/max/cv."""
    if not feat_list:
        return {}
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


def get_topN_bouts(subj_key, N, min_sec=60):
    """Get top-N longest bouts ≥ min_sec for a subject."""
    subj_dir = BOUT_DIR / subj_key
    if not subj_dir.exists():
        return []
    bout_files = sorted(subj_dir.glob('bout_*.csv'))
    # Parse duration from filename: bout_0001_28s.csv
    bouts = []
    for bf in bout_files:
        try:
            dur_str = bf.stem.split('_')[-1].replace('s', '')
            dur = float(dur_str)
            if dur >= min_sec:
                bouts.append((dur, bf))
        except:
            continue
    # Sort by duration descending, take top N
    bouts.sort(key=lambda x: x[0], reverse=True)
    return [bf for _, bf in bouts[:N]]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(y)

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))

    for N in [5, 10]:
        print(f"\n{'='*70}")
        print(f"Top-{N} longest bouts ≥ 60s")
        print(f"{'='*70}")

        gait_rows, cwt_rows, ws_rows = [], [], []
        bout_counts = []

        for i, (_, r) in enumerate(subj_df.iterrows()):
            top_bouts = get_topN_bouts(r['key'], N, min_sec=60)
            bout_counts.append(len(top_bouts))

            gait_per_bout, cwt_per_bout, ws_per_bout = [], [], []

            for bf_path in top_bouts:
                try:
                    df = pd.read_csv(bf_path)
                    xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
                    if len(xyz) < int(10 * FS):
                        continue

                    df_proc = preprocess_to_clinic_df(xyz, FS)

                    # Gait (11)
                    gf = extract_gait10(df_proc)
                    gf['vt_rms_g'] = float(np.sqrt(np.mean(df_proc['VT'].values ** 2)))
                    gait_per_bout.append(gf)

                    # CWT (28)
                    cwt_per_bout.append(extract_cwt(xyz.astype(np.float32)))

                    # WalkSway (12)
                    ws = extract_walking_sway(
                        df_proc['AP'].values, df_proc['ML'].values, df_proc['VT'].values)
                    enmo_mean = gf.get('enmo_mean_g', 0)
                    vt_rms = gf.get('vt_rms_g', 0)
                    ml_rms = gf.get('ml_rms_g', 0)
                    ws['ml_over_enmo'] = ml_rms / enmo_mean if enmo_mean > 0 else np.nan
                    ws['ml_over_vt'] = ml_rms / vt_rms if vt_rms > 0 else np.nan
                    ws_per_bout.append(ws)
                except:
                    continue

            # Aggregate across bouts
            gait_rows.append(aggregate_feature_dicts(gait_per_bout))
            cwt_rows.append(aggregate_feature_dicts(cwt_per_bout))
            ws_rows.append(aggregate_feature_dicts(ws_per_bout))

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n}] {r['key']}: {len(top_bouts)} bouts ≥60s, "
                      f"valid: G={len(gait_per_bout)} C={len(cwt_per_bout)} W={len(ws_per_bout)}",
                      flush=True)

        bc = np.array(bout_counts)
        print(f"\n  Bout stats: median={np.median(bc):.0f}, mean={np.mean(bc):.1f}, "
              f"min={bc.min()}, max={bc.max()}, zero={np.sum(bc==0)}")

        X_gait = impute(pd.DataFrame(gait_rows).replace([np.inf, -np.inf], np.nan).values.astype(float))
        X_cwt = impute(pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan).values.astype(float))
        X_ws = impute(pd.DataFrame(ws_rows).replace([np.inf, -np.inf], np.nan).values.astype(float))

        nf_g = X_gait.shape[1]; nf_c = X_cwt.shape[1]; nf_w = X_ws.shape[1]

        print(f"\n  Features: Gait={nf_g}, CWT={nf_c}, WalkSway={nf_w}")
        print(f"\n  Results (LOO CV, Ridge, fixed alpha):")

        for name, X, alpha in [
            (f'Gait', X_gait, 5),
            (f'CWT', X_cwt, 20),
            (f'WalkSway', X_ws, 5),
            (f'Gait+CWT+WS+Demo', np.column_stack([X_gait, X_cwt, X_ws, X_demo]), 5),
        ]:
            r2, mae, rho = loo_eval(X, y, alpha=alpha)
            nf = X.shape[1]
            print(f"    {name:25s} ({nf:3d}f)  R²={r2:.4f}  MAE={mae:.1f}m  ρ={rho:.3f}")

    print(f"\nDone in {time.time()-t0:.0f}s")
