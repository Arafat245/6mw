#!/usr/bin/env python3
"""
Reproduce home result from saved walking bout CSVs.
Loads pre-detected bouts from walking_bouts/ (or custom folder),
extracts gait + activity features, runs LOO CV prediction.

Input:  walking_bouts/{subject_id}/bout_*.csv (Timestamp, X, Y, Z)
        home_full_recording_npz/*.npz (for activity features)
        SwayDemographics.xlsx
Output: Prints R², MAE, ρ

Run:  python home/reproduce_from_bouts.py [--bout-dir walking_bouts]
"""
import argparse, math, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import spearmanr, pearsonr, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent.parent
NPZ_DIR = BASE / 'home_full_recording_npz'
FS = 30


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def _rodrigues(axis, theta):
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def preprocess_segment(xyz, fs=30.0):
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


# ══════════════════════════════════════════════════════════════════
# PER-BOUT GAIT FEATURES (20 features)
# ══════════════════════════════════════════════════════════════════

def extract_bout_features(xyz, fs=30.0):
    if len(xyz) < int(10 * fs):
        return None
    try:
        apmlvt, apmlvt_bp, enmo, vm_dyn = preprocess_segment(xyz, fs)
    except:
        return None
    ap, ml, vt = apmlvt[:, 0], apmlvt[:, 1], apmlvt[:, 2]
    ap_bp, ml_bp, vt_bp = apmlvt_bp[:, 0], apmlvt_bp[:, 1], apmlvt_bp[:, 2]
    nperseg = min(len(vt_bp), max(int(fs * 4), 256))
    if nperseg < int(fs): return None
    freqs, Pxx = welch(vt_bp, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    band = (freqs >= 0.5) & (freqs <= 2.5)
    if not np.any(band): return None
    cad = float(freqs[band][np.argmax(Pxx[band])])
    if cad < 1.0: return None

    f = {}
    f['cadence_hz'] = cad
    f['cadence_power'] = float(Pxx[band].max())
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
    f['hr_ap'] = _hr(ap_bp, cad)
    f['hr_vt'] = _hr(vt_bp, cad)
    f['hr_ml'] = _hr(ml_bp, cad)
    min_dist = max(1, int(round(0.5 * fs / cad)))
    prom = 0.5 * np.std(vt_bp) if np.std(vt_bp) > 0 else 0
    peaks, _ = find_peaks(vt_bp, distance=min_dist, prominence=prom)
    if peaks.size >= 3:
        si = np.diff(peaks) / fs
        f['stride_time_mean'] = float(np.mean(si))
        f['stride_time_std'] = float(np.std(si, ddof=1))
        f['stride_time_cv'] = float(np.std(si, ddof=1) / np.mean(si)) if np.mean(si) > 0 else np.nan
    else:
        f['stride_time_mean'] = np.nan; f['stride_time_std'] = np.nan; f['stride_time_cv'] = np.nan
    f['ml_rms'] = float(np.sqrt(np.mean(ml**2)))
    f['vt_rms'] = float(np.sqrt(np.mean(vt**2)))
    f['ap_rms'] = float(np.sqrt(np.mean(ap**2)))
    f['enmo_mean'] = float(np.mean(enmo))
    f['enmo_p95'] = float(np.percentile(enmo, 95))
    f['vm_std'] = float(np.std(vm_dyn))
    f['vt_range'] = float(np.ptp(vt))
    f['ml_range'] = float(np.ptp(ml))
    f['jerk_mean'] = float(np.mean(np.abs(np.diff(vm_dyn) * fs)))
    f['signal_energy'] = float(np.mean(vm_dyn**2))
    f['duration_sec'] = len(xyz) / fs
    return f


# ══════════════════════════════════════════════════════════════════
# ACTIVITY FEATURES (29 features, whole recording)
# ══════════════════════════════════════════════════════════════════

def extract_activity_features(xyz, fs):
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs); n_secs = len(enmo) // sec
    if n_secs < 60: return None
    enmo_sec = enmo[:n_secs * sec].reshape(n_secs, sec).mean(axis=1)
    f = {}
    f['act_enmo_mean'] = np.mean(enmo_sec)
    f['act_enmo_std'] = np.std(enmo_sec)
    f['act_enmo_median'] = np.median(enmo_sec)
    f['act_enmo_p5'] = np.percentile(enmo_sec, 5)
    f['act_enmo_p25'] = np.percentile(enmo_sec, 25)
    f['act_enmo_p75'] = np.percentile(enmo_sec, 75)
    f['act_enmo_p95'] = np.percentile(enmo_sec, 95)
    f['act_enmo_iqr'] = f['act_enmo_p75'] - f['act_enmo_p25']
    f['act_enmo_skew'] = float(skew(enmo_sec))
    f['act_enmo_kurtosis'] = float(kurtosis(enmo_sec))
    hist, _ = np.histogram(enmo_sec, bins=20, density=True)
    hist = hist[hist > 0]; hist = hist / hist.sum()
    f['act_enmo_entropy'] = -np.sum(hist * np.log2(hist + 1e-12))
    f['act_pct_sedentary'] = np.mean(enmo_sec < 0.02)
    f['act_pct_light'] = np.mean((enmo_sec >= 0.02) & (enmo_sec < 0.06))
    f['act_pct_moderate'] = np.mean((enmo_sec >= 0.06) & (enmo_sec < 0.1))
    f['act_pct_vigorous'] = np.mean(enmo_sec >= 0.1)
    total_hours = n_secs / 3600
    f['act_mvpa_min_per_hr'] = (np.sum(enmo_sec >= 0.06) / 60) / (total_hours + 1e-12)
    active = enmo_sec >= 0.02
    bout_durs = []
    in_b, bs = False, 0
    for j in range(len(active)):
        if active[j] and not in_b: bs = j; in_b = True
        elif not active[j] and in_b:
            if j - bs >= 5: bout_durs.append(j - bs)
            in_b = False
    if in_b and len(active) - bs >= 5: bout_durs.append(len(active) - bs)
    f['act_n_bouts'] = len(bout_durs)
    f['act_bouts_per_hr'] = len(bout_durs) / (total_hours + 1e-12)
    f['act_bout_mean_dur'] = np.mean(bout_durs) if bout_durs else 0
    f['act_bout_dur_cv'] = np.std(bout_durs) / (np.mean(bout_durs) + 1e-12) if bout_durs else 0
    f['act_longest_bout'] = max(bout_durs) if bout_durs else 0
    tas, tsa, ac, sc = 0, 0, 0, 0
    for j in range(len(active) - 1):
        if active[j]: ac += 1; tas += (not active[j + 1])
        else: sc += 1; tsa += active[j + 1]
    f['act_astp'] = tas / (ac + 1e-12)
    f['act_satp'] = tsa / (sc + 1e-12)
    f['act_fragmentation'] = f['act_astp'] + f['act_satp']
    third = n_secs // 3
    if third > 60:
        f['act_early_enmo'] = np.mean(enmo_sec[:third])
        f['act_mid_enmo'] = np.mean(enmo_sec[third:2 * third])
        f['act_late_enmo'] = np.mean(enmo_sec[2 * third:])
        f['act_early_late_ratio'] = f['act_early_enmo'] / (f['act_late_enmo'] + 1e-12)
    day_len = 24 * 3600
    n_days = max(1, n_secs // day_len)
    if n_days >= 2:
        daily_means = [np.mean(enmo_sec[d * day_len:(d + 1) * day_len])
                       for d in range(n_days) if (d + 1) * day_len <= n_secs]
        f['act_daily_cv'] = np.std(daily_means) / (np.mean(daily_means) + 1e-12) if len(daily_means) >= 2 else 0
    else:
        f['act_daily_cv'] = 0
    return f


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


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bout-dir', default='walking_bouts', help='Folder with walking bout CSVs')
    args = parser.parse_args()

    t0 = time.time()
    BOUT_DIR = BASE / args.bout_dir
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)
    print(f"n={n} subjects, loading bouts from {args.bout_dir}/")

    # ── Extract features from saved bout CSVs + full NPZ ──
    all_rows = []
    for i, (_, r) in enumerate(subj_df.iterrows()):
        row = {}
        subj_bout_dir = BOUT_DIR / r['key']

        # Gait features from saved bout CSVs
        bout_feats = []
        if subj_bout_dir.exists():
            bout_files = sorted(subj_bout_dir.glob('bout_*.csv'))
            for bf_path in bout_files:
                df = pd.read_csv(bf_path)
                xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
                feats = extract_bout_features(xyz, FS)
                if feats is not None:
                    bout_feats.append(feats)

        if bout_feats:
            gait_feat_names = sorted(bout_feats[0].keys())
            arr = np.array([[bf.get(k, np.nan) for k in gait_feat_names] for bf in bout_feats])
            for j, name in enumerate(gait_feat_names):
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
            # g_bout_dur_cv removed — duplicate of g_duration_sec_cv

        # Activity features from full NPZ
        npz_path = NPZ_DIR / f"{r['key']}.npz"
        if npz_path.exists():
            xyz_full = np.load(npz_path)['xyz'].astype(np.float64)
            act = extract_activity_features(xyz_full, FS)
            if act:
                row.update(act)

        all_rows.append(row)
        nb = len(bout_feats)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] {r['key']}: {nb} valid bouts", flush=True)

    feat_df = pd.DataFrame(all_rows)
    accel_cols = [c for c in feat_df.columns]
    X_accel = impute(feat_df.values.astype(float))
    n_accel = X_accel.shape[1]

    # ── Demographics ──
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))
    n_demo = 4

    X_all = np.column_stack([X_accel, X_demo])
    demo_idx = list(range(n_accel, n_accel + n_demo))

    # ── LOO CV ──
    K = 20; alpha = 20
    preds = np.zeros(n)
    for tr, te in LeaveOneOut().split(X_all):
        corrs = [abs(spearmanr(X_all[tr, j], y[tr])[0]) if np.std(X_all[tr, j]) > 0 else 0
                 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K]
        selected = top_k + demo_idx
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_all[tr][:, selected]), y[tr])
        preds[te] = m.predict(sc.transform(X_all[te][:, selected]))

    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rho = spearmanr(y, preds)[0]
    rv = pearsonr(y, preds)[0]
    print(f"\nHome Result (clinic-free, no data leakage)")
    print(f"  n={n}, Spearman Top-{K} inside LOO, Ridge α={alpha}")
    print(f"  R²={r2:.4f}  MAE={mae:.0f} ft  r={rv:.3f}  ρ={rho:.3f}")
    print(f"  Features: {K} PerBout + {n_demo} Demo = {K+n_demo}f")
    print(f"  Bouts from: {args.bout_dir}/")
    print(f"  Done in {time.time()-t0:.0f}s")
