#!/usr/bin/env python3
"""
Clinic PerBout: Split 6MWT into 60s windows, extract 20 per-bout features,
aggregate across windows, evaluate with Spearman Top-20 inside LOO.

Input:  csv_raw2/*.csv (clinic 6MWT raw data)
Output: Prints R² for PerBout-Top20 and PerBout-Top20+Demo(4)

Run:  python temporary_experiments/exp_clinic_perbout.py
"""
import math, time, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent
FS = 30
FT2M = 0.3048
WIN_SEC = 60


# ── Preprocessing (same as home pipeline) ──

def _rodrigues(axis, theta):
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def preprocess_segment(xyz, fs):
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
        arr_v = arr_dyn @ _rodrigues(axis, angle).T
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
    return apmlvt, apmlvt_bp, enmo, np.linalg.norm(apmlvt, axis=1)


# ── Per-bout features (same 20 as home) ──

def extract_bout_features(xyz, fs):
    if len(xyz) < int(10 * fs): return None
    try:
        apmlvt, apmlvt_bp, enmo, vm_dyn = preprocess_segment(xyz, fs)
    except: return None
    ap, ml, vt = apmlvt[:, 0], apmlvt[:, 1], apmlvt[:, 2]
    ap_bp, ml_bp, vt_bp = apmlvt_bp[:, 0], apmlvt_bp[:, 1], apmlvt_bp[:, 2]
    nperseg = min(len(vt_bp), int(fs * 4))
    if nperseg < int(fs): return None
    freqs, Pxx = welch(vt_bp, fs=fs, nperseg=max(nperseg, 256), noverlap=nperseg // 2, detrend="constant")
    band = (freqs >= 0.5) & (freqs <= 3.5)
    if not np.any(band): return None
    cad = float(freqs[band][np.argmax(Pxx[band])])
    if cad < 1.0: return None
    f = {}
    f['cadence_hz'] = cad; f['cadence_power'] = float(Pxx[band].max())
    lag = max(1, min(int(round(fs / cad)), len(vt_bp) - 1))
    x = vt_bp - vt_bp.mean(); d = np.dot(x, x)
    f['acf_step_reg'] = float(np.dot(x[:len(x) - lag], x[lag:]) / (d + 1e-12)) if d > 0 else 0
    def _hr(sig, cad_f):
        x = sig - sig.mean()
        if len(x) < 2: return np.nan
        X = np.fft.rfft(x); fr = np.fft.rfftfreq(len(x), d=1.0 / fs); mags = np.abs(X)
        ev, od = 0.0, 0.0
        for k in range(1, 11):
            fk = k * cad_f
            if fk >= fr[-1]: break
            idx = int(np.argmin(np.abs(fr - fk)))
            if k % 2 == 0: ev += mags[idx]
            else: od += mags[idx]
        return float(ev / od) if od > 0 else np.nan
    f['hr_ap'] = _hr(ap_bp, cad); f['hr_vt'] = _hr(vt_bp, cad); f['hr_ml'] = _hr(ml_bp, cad)
    min_dist = max(1, int(round(0.5 * fs / cad)))
    prom = 0.5 * np.std(vt_bp) if np.std(vt_bp) > 0 else 0
    peaks, _ = find_peaks(vt_bp, distance=min_dist, prominence=prom)
    if peaks.size >= 3:
        si = np.diff(peaks) / fs
        f['stride_time_mean'] = float(np.mean(si)); f['stride_time_std'] = float(np.std(si, ddof=1))
        f['stride_time_cv'] = float(np.std(si, ddof=1) / np.mean(si)) if np.mean(si) > 0 else np.nan
    else:
        f['stride_time_mean'] = np.nan; f['stride_time_std'] = np.nan; f['stride_time_cv'] = np.nan
    f['ml_rms'] = float(np.sqrt(np.mean(ml**2))); f['vt_rms'] = float(np.sqrt(np.mean(vt**2)))
    f['ap_rms'] = float(np.sqrt(np.mean(ap**2)))
    f['enmo_mean'] = float(np.mean(enmo)); f['enmo_p95'] = float(np.percentile(enmo, 95))
    f['vm_std'] = float(np.std(vm_dyn)); f['vt_range'] = float(np.ptp(vt)); f['ml_range'] = float(np.ptp(ml))
    f['jerk_mean'] = float(np.mean(np.abs(np.diff(vm_dyn) * fs)))
    f['signal_energy'] = float(np.mean(vm_dyn**2)); f['duration_sec'] = len(xyz) / fs
    return f


# ── Helpers ──

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
    row['g_n_valid_bouts'] = len(bout_feats)
    row['g_total_walk_sec'] = sum(bf.get('duration_sec', 0) for bf in bout_feats)
    durs = [bf.get('duration_sec', 0) for bf in bout_feats]
    row['g_mean_bout_dur'] = np.mean(durs)
    row['g_bout_dur_cv'] = np.std(durs) / (np.mean(durs) + 1e-12) if np.mean(durs) > 0 else 0
    return row


def spearman_loo(X_accel, X_demo, y, K=20, alpha=20):
    n_accel = X_accel.shape[1]
    X_all = np.column_stack([X_accel, X_demo]) if X_demo is not None else X_accel
    n_demo = X_demo.shape[1] if X_demo is not None else 0
    demo_idx = list(range(n_accel, n_accel + n_demo)) if n_demo > 0 else []
    K_use = min(K, n_accel)
    preds = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), dtype=bool); tr[i] = False
        corrs = [abs(spearmanr(X_all[tr, j], y[tr])[0]) if np.std(X_all[tr, j]) > 0 else 0
                 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K_use]
        selected = top_k + demo_idx
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_all[tr][:, selected]), y[tr])
        preds[i] = m.predict(sc.transform(X_all[i:i+1][:, selected]))[0]
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y * FT2M, preds * FT2M)
    rho = spearmanr(y, preds)[0]
    return r2, mae, rho


if __name__ == '__main__':
    t0 = time.time()
    RAW = BASE / 'csv_raw2'

    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = (ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44]))
    ids101 = ids[~excl].reset_index(drop=True)
    y = ids101['sixmwd'].values.astype(float)
    n = len(y)

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids101.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))

    print(f"n={n}, Clinic PerBout ({WIN_SEC}s windows)")

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

        # Trim first/last 10s
        trim = int(10 * fs)
        if 2 * trim < len(xyz):
            xyz = xyz[trim:len(xyz) - trim]

        # Split into 60s windows
        win_samples = int(WIN_SEC * fs)
        bout_feats = []
        for start in range(0, len(xyz) - win_samples + 1, win_samples):
            seg = xyz[start:start + win_samples]
            feats = extract_bout_features(seg, fs)
            if feats is not None:
                bout_feats.append(feats)

        all_rows.append(aggregate_bout_feats(bout_feats))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}]", flush=True)

    feat_df = pd.DataFrame(all_rows)
    X_accel = impute(feat_df.values.astype(float))

    # PerBout-Top20
    r2, mae, rho = spearman_loo(X_accel, None, y, K=20, alpha=20)
    print(f"\n  PerBout-Top20           ({X_accel.shape[1]}f→20)  R²={r2:.4f}  MAE={mae:.1f}m  ρ={rho:.3f}")

    # PerBout-Top20 + Demo(4)
    r2, mae, rho = spearman_loo(X_accel, X_demo, y, K=20, alpha=20)
    print(f"  PerBout-Top20+Demo(4)   ({X_accel.shape[1]}f→24)  R²={r2:.4f}  MAE={mae:.1f}m  ρ={rho:.3f}")

    print(f"\nDone in {time.time()-t0:.0f}s")
