#!/usr/bin/env python3
"""
Step 1: Load daytime NPZ -> clinic-free features -> Top-20 + Demo(5) -> Ridge LOO CV.
Run AFTER step0_gt3x_to_npz.py finishes.

Requires: pygt3x-env (or any env with numpy, pandas, scipy, sklearn, openpyxl)
Run:  python temporary_experiments/step1_extract_and_predict.py
"""
import os, re, math, time, warnings, pickle
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
NPZ_DIR = BASE / 'csv_home_daytime_npz'
FS = 30

TOP20_FEATURES = [
    'g_duration_sec_max', 'g_bout_dur_cv', 'g_duration_sec_cv',
    'act_enmo_p95', 'act_pct_vigorous',
    'g_acf_step_reg_max', 'g_enmo_mean_p10', 'g_ml_range_med',
    'g_ml_rms_cv', 'g_ap_rms_cv', 'g_ap_rms_med',
    'g_acf_step_reg_p90', 'g_jerk_mean_med', 'g_mean_bout_dur',
    'g_signal_energy_med', 'g_ml_rms_med', 'g_vm_std_med',
    'g_enmo_mean_med', 'g_enmo_p95_med', 'g_acf_step_reg_med',
]


# ── Walking detection ──

def detect_walking_bouts(xyz, fs, min_bout_sec=10, merge_gap_sec=5):
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs); n_secs = len(enmo) // sec
    if n_secs < min_bout_sec:
        return []
    enmo_sec = enmo[:n_secs * sec].reshape(n_secs, sec).mean(axis=1)
    active = enmo_sec >= 0.015
    raw_bouts = []
    in_b, bs = False, 0
    for s in range(n_secs):
        if active[s] and not in_b: bs = s; in_b = True
        elif not active[s] and in_b:
            if s - bs >= min_bout_sec: raw_bouts.append((bs * sec, s * sec))
            in_b = False
    if in_b and n_secs - bs >= min_bout_sec:
        raw_bouts.append((bs * sec, n_secs * sec))
    if not raw_bouts:
        return []
    b_filt, a_filt = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b_filt, a_filt, vm - vm.mean())
    win = int(10 * fs); step = int(10 * fs)
    fft_freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band = (fft_freqs >= 0.8) & (fft_freqs <= 3.5)
    refined = []
    for bout_s, bout_e in raw_bouts:
        walking_wins = []
        for wi in range(bout_s, bout_e - win, step):
            seg = vm_bp[wi:wi + win]
            X = np.fft.rfft(seg); mags = np.abs(X)
            if not np.any(band): continue
            cadence = fft_freqs[band][np.argmax(mags[band])]
            even, odd = 0.0, 0.0
            for k in range(1, 11):
                fk = k * cadence
                if fk >= fft_freqs[-1]: break
                idx = int(np.argmin(np.abs(fft_freqs - fk)))
                if k % 2 == 0: even += mags[idx]
                else: odd += mags[idx]
            hr = even / (odd + 1e-12) if odd > 0 else 0
            if hr >= 0.2: walking_wins.append((wi, wi + win))
        if walking_wins:
            cs, ce = walking_wins[0]
            for ws, we in walking_wins[1:]:
                if ws <= ce + step: ce = max(ce, we)
                else:
                    if ce - cs >= min_bout_sec * fs: refined.append((cs, ce))
                    cs, ce = ws, we
            if ce - cs >= min_bout_sec * fs: refined.append((cs, ce))
        else:
            if (bout_e - bout_s) >= min_bout_sec * fs:
                refined.append((bout_s, bout_e))
    if not refined:
        return []
    merged = [refined[0]]
    for s, e in refined[1:]:
        prev_s, prev_e = merged[-1]
        if (s - prev_e) / fs <= merge_gap_sec:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


# ── Preprocessing ──

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


# ── Per-bout features ──

def extract_bout_features(xyz, fs=30.0):
    if len(xyz) < int(10 * fs):
        return None
    try:
        apmlvt, apmlvt_bp, enmo, vm_dyn = preprocess_segment(xyz, fs)
    except:
        return None
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


# ── Activity features ──

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
    day_len = 15 * 3600
    n_days = max(1, n_secs // day_len)
    if n_days >= 2:
        daily_means = [np.mean(enmo_sec[d * day_len:(d + 1) * day_len])
                       for d in range(n_days) if (d + 1) * day_len <= n_secs]
        f['act_daily_cv'] = np.std(daily_means) / (np.mean(daily_means) + 1e-12) if len(daily_means) >= 2 else 0
    else:
        f['act_daily_cv'] = 0
    return f


# ── Aggregation ──

def extract_subject_features(xyz_daytime, fs, return_bouts=False):
    row = {}
    bouts = detect_walking_bouts(xyz_daytime, fs, min_bout_sec=10, merge_gap_sec=5)
    bout_feats = []
    for s, e in bouts:
        feats = extract_bout_features(xyz_daytime[s:e], fs)
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
        row['g_n_valid_bouts'] = len(bout_feats)
        row['g_total_walk_sec'] = sum(bf.get('duration_sec', 0) for bf in bout_feats)
        durs = [bf.get('duration_sec', 0) for bf in bout_feats]
        row['g_mean_bout_dur'] = np.mean(durs)
        row['g_bout_dur_cv'] = np.std(durs) / (np.mean(durs) + 1e-12) if np.mean(durs) > 0 else 0
    act = extract_activity_features(xyz_daytime, fs)
    if act:
        row.update(act)
    if return_bouts:
        return row, bouts, bout_feats
    return row


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
    t0 = time.time()

    # Load subject list
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)
    print(f"n={n} subjects (M={sum(subj_df.cohort=='M')}, C={sum(subj_df.cohort=='C')})")

    # Extract features from cached NPZ
    print(f"\nExtracting clinic-free features from NPZ...")
    all_rows = []
    all_bouts = {}
    all_bout_feats = {}
    for i, (_, r) in enumerate(subj_df.iterrows()):
        npz_path = NPZ_DIR / f"{r['key']}.npz"
        if not npz_path.exists():
            print(f"  WARNING: {r['key']} NPZ missing")
            all_rows.append({})
            continue
        xyz = np.load(npz_path)['xyz'].astype(np.float64)
        row, bouts, bout_feats = extract_subject_features(xyz, FS, return_bouts=True)
        all_rows.append(row)
        all_bouts[r['key']] = bouts
        all_bout_feats[r['key']] = bout_feats
        nb = row.get('g_n_valid_bouts', 0)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] {r['key']}: {nb} bouts", flush=True)

    feat_df = pd.DataFrame(all_rows)

    # Select Top-20
    X_top20 = np.full((n, len(TOP20_FEATURES)), np.nan)
    for j, feat in enumerate(TOP20_FEATURES):
        if feat in feat_df.columns:
            X_top20[:, j] = feat_df[feat].values
    X_top20 = impute(X_top20)

    # Demographics
    print("\nLoading demographics...")
    demo = pd.read_excel(BASE / 'Accel files' / 'PedMSWalkStudy_Demographic.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float))

    # Combine + LOO Ridge CV
    X = np.column_stack([X_top20, X_demo])
    print(f"\nLOO Ridge CV: X={X.shape}, n={n}")
    print(f"{'alpha':>7s}  {'R2':>8s}  {'MAE':>8s}  {'r':>8s}  {'rho':>8s}")
    print("-" * 48)
    for alpha in [5, 10, 20, 50, 100]:
        preds = np.zeros(n)
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=alpha)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            preds[te] = m.predict(sc.transform(X[te]))
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        r_val = pearsonr(y, preds)[0]
        rho = spearmanr(y, preds)[0]
        marker = "  <<<" if alpha == 20 else ""
        print(f"  {alpha:>5d}  {r2:>8.4f}  {mae:>7.1f}  {r_val:>8.3f}  {rho:>7.3f}{marker}")

    # Verify against cached npz
    npz_ref = np.load(BASE / 'feats' / 'home_clinicfree_top20.npz')
    y_ref = npz_ref['y']
    if np.array_equal(y.astype(int), y_ref.astype(int)):
        corr = np.corrcoef(X_top20.flatten(), npz_ref['X'].flatten())[0, 1]
        print(f"\n  Subject order matches cached. Feature corr: r={corr:.4f}")

    # Save features, bouts, and Top-20 NPZ for later use
    FEATS_DIR = BASE / 'feats'

    feat_df.insert(0, 'key', subj_df['key'].values)
    feat_df.to_csv(FEATS_DIR / 'home_clinicfree_features.csv', index=False)
    print(f"\n  Saved feats/home_clinicfree_features.csv ({feat_df.shape})")

    np.savez(FEATS_DIR / 'home_clinicfree_top20_reproduced.npz',
             X=X_top20, feature_names=TOP20_FEATURES, y=y)
    print(f"  Saved feats/home_clinicfree_top20_reproduced.npz")

    import pickle
    with open(FEATS_DIR / 'home_walking_bouts.pkl', 'wb') as f:
        pickle.dump({'bouts': all_bouts, 'bout_feats': all_bout_feats}, f)
    print(f"  Saved feats/home_walking_bouts.pkl ({len(all_bouts)} subjects)")

    # Also save target_6mwd.csv for future use
    subj_df[['cohort', 'subj_id', 'year', 'sixmwd']].to_csv(
        FEATS_DIR / 'target_6mwd.csv', index=False)
    print(f"  Saved feats/target_6mwd.csv")

    print(f"\nDone in {time.time()-t0:.0f}s")
