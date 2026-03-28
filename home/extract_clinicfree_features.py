#!/usr/bin/env python3
"""
Clinic-free home feature extraction.
No clinic data used anywhere — no PLS, no cosine similarity to clinic signature.

Walking detection: ENMO threshold → HR refinement → merge adjacent bouts (gap <5s)
  - Min bout: 10s, cadence filter: reject <1.0 Hz
Feature extraction: per-bout gait features + whole-recording activity features
  - Aggregate per-bout features with robust stats (median, IQR, p10, p90, max, CV)
Output: feats/home_clinicfree_features.npz

Best result: Top-20 correlated features + Demo(5), Ridge α=20 → R²=0.441, MAE=191ft
"""
import numpy as np
import pandas as pd
import math
import warnings
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
HOME_DIR = BASE / 'csv_home_daytime'
FS = 30

# Top-20 features by Spearman |ρ| with 6MWD (all clinic-free)
TOP20_FEATURES = [
    'g_duration_sec_max',       # ρ=+0.415  longest walking bout
    'g_bout_dur_cv',            # ρ=+0.406  bout duration variability
    'g_duration_sec_cv',        # ρ=+0.406  (same metric, different path)
    'act_enmo_p95',             # ρ=+0.396  peak daily activity intensity
    'act_pct_vigorous',         # ρ=+0.390  % time in vigorous activity
    'g_acf_step_reg_max',       # ρ=+0.378  best step regularity across bouts
    'g_enmo_mean_p10',          # ρ=+0.356  10th %ile ENMO across bouts
    'g_ml_range_med',           # ρ=+0.355  median ML sway range
    'g_ml_rms_cv',              # ρ=-0.354  CV of ML RMS across bouts
    'g_ap_rms_cv',              # ρ=-0.354  CV of AP RMS across bouts
    'g_ap_rms_med',             # ρ=+0.351  median AP amplitude
    'g_acf_step_reg_p90',       # ρ=+0.347  90th %ile step regularity
    'g_jerk_mean_med',          # ρ=+0.343  median jerk
    'g_mean_bout_dur',          # ρ=+0.338  mean bout duration
    'g_signal_energy_med',      # ρ=+0.333  median signal energy
    'g_ml_rms_med',             # ρ=+0.328  median ML RMS
    'g_vm_std_med',             # ρ=+0.328  median VM variability
    'g_enmo_mean_med',          # ρ=+0.327  median ENMO across bouts
    'g_enmo_p95_med',           # ρ=+0.324  median of per-bout 95th %ile ENMO
    'g_acf_step_reg_med',       # ρ=+0.324  median step regularity
]


# ══════════════════════════════════════════════════════════════════
# WALKING DETECTION (clinic-free)
# ══════════════════════════════════════════════════════════════════

def detect_walking_bouts(xyz, fs, min_bout_sec=10, merge_gap_sec=5):
    """
    Detect walking bouts without clinic data.
    Stage 1: ENMO ≥ 0.015 per second, min 10s bouts
    Stage 2: Harmonic ratio ≥ 0.2 in 10s windows (confirms periodic walking)
    Stage 3: Merge adjacent bouts within 5s gap
    """
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs); n_secs = len(enmo) // sec
    if n_secs < min_bout_sec:
        return []
    enmo_sec = enmo[:n_secs * sec].reshape(n_secs, sec).mean(axis=1)
    active = enmo_sec >= 0.015

    # Stage 1: Find active bouts
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

    # Stage 2: HR refinement
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

    # Stage 3: Merge adjacent bouts (gap < merge_gap_sec)
    merged = [refined[0]]
    for s, e in refined[1:]:
        prev_s, prev_e = merged[-1]
        if (s - prev_e) / fs <= merge_gap_sec:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


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
# PER-BOUT GAIT FEATURES
# ══════════════════════════════════════════════════════════════════

def extract_bout_features(xyz, fs=30.0):
    """Extract gait features from a single walking bout. Returns dict or None."""
    if len(xyz) < int(10 * fs):
        return None
    try:
        apmlvt, apmlvt_bp, enmo, vm_dyn = preprocess_segment(xyz, fs)
    except:
        return None

    ap, ml, vt = apmlvt[:, 0], apmlvt[:, 1], apmlvt[:, 2]
    ap_bp, ml_bp, vt_bp = apmlvt_bp[:, 0], apmlvt_bp[:, 1], apmlvt_bp[:, 2]

    # Cadence
    nperseg = min(len(vt_bp), int(fs * 4))
    if nperseg < int(fs): return None
    freqs, Pxx = welch(vt_bp, fs=fs, nperseg=max(nperseg, 256), noverlap=nperseg // 2, detrend="constant")
    band = (freqs >= 0.5) & (freqs <= 3.5)
    if not np.any(band): return None
    cad = float(freqs[band][np.argmax(Pxx[band])])
    if cad < 1.0:  # cadence filter
        return None

    f = {}
    f['cadence_hz'] = cad
    f['cadence_power'] = float(Pxx[band].max())

    # Step regularity
    lag = max(1, min(int(round(fs / cad)), len(vt_bp) - 1))
    x = vt_bp - vt_bp.mean(); d = np.dot(x, x)
    f['acf_step_reg'] = float(np.dot(x[:len(x) - lag], x[lag:]) / (d + 1e-12)) if d > 0 else 0

    # Harmonic ratios
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

    # Step timing
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

    # Amplitude
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
# WHOLE-RECORDING ACTIVITY FEATURES
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

    day_len = 15 * 3600
    n_days = max(1, n_secs // day_len)
    if n_days >= 2:
        daily_means = [np.mean(enmo_sec[d * day_len:(d + 1) * day_len])
                       for d in range(n_days) if (d + 1) * day_len <= n_secs]
        f['act_daily_cv'] = np.std(daily_means) / (np.mean(daily_means) + 1e-12) if len(daily_means) >= 2 else 0
    else:
        f['act_daily_cv'] = 0
    return f


# ══════════════════════════════════════════════════════════════════
# MAIN: Extract + cache features
# ══════════════════════════════════════════════════════════════════

def extract_all_features(ids_df, save_bouts=True):
    """Extract clinic-free features for all subjects. Returns DataFrame.
    If save_bouts=True, saves each walking bout as CSV in walking_bouts/ folder."""
    gait_feat_names = None
    results = []
    BOUT_DIR = BASE / 'walking_bouts'

    for i, (_, r) in enumerate(ids_df.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        fp = HOME_DIR / fn
        row = {'fn': fn}

        if not fp.exists():
            results.append(row)
            continue

        home_df = pd.read_csv(fp)
        has_ts = 'Timestamp' in home_df.columns
        xyz = home_df[['X', 'Y', 'Z']].values.astype(np.float64)
        ts = home_df['Timestamp'].values if has_ts else None

        # Per-bout gait features
        bouts = detect_walking_bouts(xyz, FS, min_bout_sec=10, merge_gap_sec=5)

        # Save walking bouts
        if save_bouts and bouts:
            subj_id = f"{r['cohort']}{int(r['subj_id']):02d}"
            subj_bout_dir = BOUT_DIR / subj_id
            subj_bout_dir.mkdir(parents=True, exist_ok=True)
            for bout_idx, (s, e) in enumerate(bouts):
                dur_sec = (e - s) / FS
                bout_data = {
                    'Timestamp': ts[s:e] if ts is not None else np.arange(e - s) / FS,
                    'X': xyz[s:e, 0],
                    'Y': xyz[s:e, 1],
                    'Z': xyz[s:e, 2],
                }
                bout_df = pd.DataFrame(bout_data)
                bout_df.to_csv(subj_bout_dir / f"bout_{bout_idx+1:04d}_{dur_sec:.0f}s.csv", index=False)

        bout_feats = []
        for s, e in bouts:
            feats = extract_bout_features(xyz[s:e], FS)
            if feats is not None:
                bout_feats.append(feats)

        if bout_feats:
            if gait_feat_names is None:
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

        # Activity features
        act = extract_activity_features(xyz, FS)
        if act:
            row.update(act)

        results.append(row)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(ids_df)}] {fn}: {len(bout_feats)} valid bouts", flush=True)

    return pd.DataFrame(results)


if __name__ == '__main__':
    import time
    t0 = time.time()

    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)

    print(f"Extracting clinic-free features for {len(ids101)} subjects...")
    df = extract_all_features(ids101)
    df.to_csv(BASE / 'feats' / 'home_clinicfree_features.csv', index=False)

    # Also save Top-20 as NPZ for quick loading
    top20 = TOP20_FEATURES
    X = np.full((len(df), len(top20)), np.nan)
    for j, feat in enumerate(top20):
        if feat in df.columns:
            X[:, j] = df[feat].values
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        if m.any(): X[m, j] = np.nanmedian(X[:, j])

    np.savez(BASE / 'feats' / 'home_clinicfree_top20.npz',
             X=X, feature_names=top20, y=ids101['sixmwd'].values)

    print(f"Saved feats/home_clinicfree_features.csv ({len(df)} rows, {len(df.columns)} cols)")
    print(f"Saved feats/home_clinicfree_top20.npz ({X.shape})")
    print(f"Done in {time.time() - t0:.0f}s")
