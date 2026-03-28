#!/usr/bin/env python3
"""
Step 6: find_walking pipeline — exact same data flow as step1, but replacing
ENMO+HR walking detection with find_walking (Straczkiewicz et al. 2023).

Data flow:
  NPZ (daytime X,Y,Z at 30Hz)
  --> find_walking(vm, fs) --> walking bouts (intensity + periodicity + duration)
  --> extract_bout_features() on each bout (same gait features as step1)
  --> aggregate with 6 stats (median, IQR, p10, p90, max, CV)
  --> Spearman inside LOO K=20 + Demo(4) --> R²

No new features added — same 20+ raw gait features, same aggregation, only
the walking detection algorithm is different.
"""
import os, re, math, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.signal.windows import tukey
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent.parent
NPZ_DIR = BASE / 'csv_home_daytime_npz'
FS = 30


# ══════════════════════════════════════════════════════════════════
# FIND_WALKING: CWT-based walking detection (Straczkiewicz 2023)
# ══════════════════════════════════════════════════════════════════

def find_walking(vm, fs, min_amp=0.3, T=3, delta=2, alpha=0.6, beta=2.5,
                 step_freq=(1.4, 2.3)):
    """Detect walking using intensity + periodicity + duration criteria."""
    sec = int(fs)
    n_secs = len(vm) // sec
    if n_secs < 5:
        return []

    vm_mat = vm[:n_secs * sec].reshape(n_secs, sec)
    p2p = vm_mat.max(axis=1) - vm_mat.min(axis=1)
    valid = p2p >= min_amp

    freq_grid = np.arange(0.5, 4.05, 0.05)
    n_freqs = len(freq_grid)
    loc1 = np.searchsorted(freq_grid, step_freq[0])
    loc2 = np.searchsorted(freq_grid, step_freq[1])

    win_len = 3 * sec
    fft_freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
    f_interp_indices = np.searchsorted(fft_freqs, freq_grid)
    f_interp_indices = np.clip(f_interp_indices, 0, len(fft_freqs) - 1)

    peak_matrix = np.zeros((n_freqs, n_secs), dtype=bool)
    tukey_win = tukey(win_len, alpha=0.02)

    for s in range(n_secs):
        if not valid[s]:
            continue
        s_start = max(0, s - 1)
        s_end = min(n_secs, s + 2)
        if s_end - s_start < 3:
            s_start = max(0, s_end - 3)
        seg = vm[s_start * sec:s_end * sec]
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len - len(seg)), mode='edge')
        seg = seg[:win_len]
        seg_win = (seg - seg.mean()) * tukey_win
        fft_vals = np.fft.rfft(seg_win)
        power = np.abs(fft_vals[f_interp_indices]) ** 2
        if np.max(power) < 1e-10:
            continue
        peaks_idx, _ = find_peaks(power, height=0.1 * np.max(power))
        if len(peaks_idx) == 0:
            continue
        peak_vals = power[peaks_idx]
        order = np.argsort(peak_vals)[::-1]
        peaks_idx = peaks_idx[order]
        peak_vals = peak_vals[order]
        in_range = [(p, power[p]) for p in peaks_idx if loc1 <= p <= loc2]
        if not in_range:
            continue
        best_idx, best_val = in_range[0]
        highest_idx, highest_val = peaks_idx[0], peak_vals[0]
        accept = False
        if highest_idx == best_idx:
            accept = True
        elif highest_idx > loc2:
            accept = (highest_val / (best_val + 1e-12)) < beta
        elif highest_idx < loc1:
            accept = (highest_val / (best_val + 1e-12)) < alpha
        if accept:
            peak_matrix[best_idx, s] = True

    # Continuity check
    final_walk = np.zeros(n_secs, dtype=bool)
    for s in range(n_secs - T + 1):
        window = peak_matrix[:, s:s + T]
        freqs_in_window = []
        for t in range(T):
            col_peaks = np.where(window[:, t])[0]
            if len(col_peaks) == 0:
                break
            freqs_in_window.append(col_peaks)
        else:
            continuous = True
            for t in range(T - 1):
                min_diff = min(abs(f1 - f2) for f1 in freqs_in_window[t]
                              for f2 in freqs_in_window[t + 1])
                if min_diff > delta:
                    continuous = False
                    break
            if continuous:
                for t in range(T):
                    final_walk[s + t] = True

    # Convert to bout sample indices (min 10s)
    bouts = []
    in_b, bs = False, 0
    for s in range(n_secs):
        if final_walk[s] and not in_b:
            bs = s; in_b = True
        elif not final_walk[s] and in_b:
            if s - bs >= 10:
                bouts.append((bs * sec, s * sec))
            in_b = False
    if in_b and n_secs - bs >= 10:
        bouts.append((bs * sec, n_secs * sec))

    return bouts


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING + PER-BOUT FEATURES (same as step1)
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


# ══════════════════════════════════════════════════════════════════
# ACTIVITY FEATURES (same as step1 — whole recording, not bouts)
# ══════════════════════════════════════════════════════════════════

def extract_activity_features(xyz, fs):
    from scipy.stats import skew, kurtosis
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
# AGGREGATION
# ══════════════════════════════════════════════════════════════════

def aggregate_and_extract(xyz, fs, bouts):
    """Extract per-bout features from given bouts, aggregate, add activity features."""
    row = {}
    bout_feats = []
    for s, e in bouts:
        if e > len(xyz): e = len(xyz)
        bf = extract_bout_features(xyz[s:e], fs)
        if bf is not None:
            bout_feats.append(bf)

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

    act = extract_activity_features(xyz, fs)
    if act:
        row.update(act)

    return row, len(bouts), len(bout_feats)


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

    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(y)
    print(f"n={n} subjects", flush=True)

    # ── Extract features using find_walking bout detection ──
    print(f"\n=== find_walking pipeline ===", flush=True)
    print(f"NPZ -> find_walking(vm) -> extract_bout_features -> aggregate", flush=True)
    fw_rows = []
    all_fw_bouts = {}
    for i, (_, r) in enumerate(subj_df.iterrows()):
        npz_path = NPZ_DIR / f"{r['key']}.npz"
        xyz = np.load(npz_path)['xyz'].astype(np.float64)
        vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)

        bouts = find_walking(vm, FS)
        all_fw_bouts[r['key']] = bouts
        row, n_bouts, n_valid = aggregate_and_extract(xyz, FS, bouts)
        fw_rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] {r['key']}: {n_bouts} bouts, {n_valid} valid", flush=True)

    fw_df = pd.DataFrame(fw_rows)
    fw_cols = [c for c in fw_df.columns]
    gait_cols = [c for c in fw_cols if c.startswith('g_')]
    act_cols = [c for c in fw_cols if c.startswith('act_')]
    print(f"  Features: {len(fw_cols)} total ({len(gait_cols)} gait + {len(act_cols)} activity)", flush=True)

    # ── Also load original ENMO+HR features for comparison ──
    orig_df = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
    orig_cols_all = [c for c in orig_df.columns if c != 'key']

    # ── Demo(4) ──
    demo_data = pd.read_excel(BASE / 'Accel files' / 'PedMSWalkStudy_Demographic.xlsx')
    demo_data['cohort'] = demo_data['ID'].str.extract(r'^([A-Z])')[0]
    demo_data['subj_id'] = demo_data['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo_data, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    demo_cols = ['cohort_POMS', 'Age', 'Sex', 'BMI']
    X_demo = impute(p[demo_cols].values.astype(float))

    X_fw = impute(fw_df.values.astype(float))
    X_orig = impute(orig_df[orig_cols_all].values.astype(float))
    n_fw = len(fw_cols)
    n_orig = len(orig_cols_all)
    n_demo = len(demo_cols)

    # ── Evaluate: Spearman inside LOO ──
    print(f"\n{'='*70}", flush=True)
    print(f"COMPARISON: Spearman inside LOO + Demo(4), Ridge a=20", flush=True)
    print(f"{'='*70}", flush=True)

    for K in [10, 15, 20, 25]:
        for name, X_accel, accel_cols in [
            ('find_walking', X_fw, fw_cols),
            ('ENMO+HR (original)', X_orig, orig_cols_all),
        ]:
            X_all = np.column_stack([X_accel, X_demo])
            n_accel = X_accel.shape[1]
            demo_idx = list(range(n_accel, n_accel + n_demo))

            preds = np.zeros(n)
            for tr, te in LeaveOneOut().split(X_all):
                corrs = [abs(spearmanr(X_all[tr, j], y[tr])[0]) if np.std(X_all[tr, j]) > 0 else 0
                         for j in range(n_accel)]
                top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K]
                selected = top_k + demo_idx
                sc = StandardScaler(); m = Ridge(alpha=20)
                m.fit(sc.fit_transform(X_all[tr][:, selected]), y[tr])
                preds[te] = m.predict(sc.transform(X_all[te][:, selected]))

            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rho = spearmanr(y, preds)[0]
            print(f"  K={K:2d}+Demo4  {name:20s}  {X_accel.shape[1]:3d}f  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}", flush=True)
        print()

    # ── Save ──
    fw_df.insert(0, 'key', subj_df['key'].values)
    fw_df.to_csv(BASE / 'feats' / 'findwalking_perbout_features.csv', index=False)

    with open(BASE / 'feats' / 'findwalking_bouts.pkl', 'wb') as f:
        pickle.dump(all_fw_bouts, f)

    print(f"\n  Saved feats/findwalking_perbout_features.csv ({fw_df.shape})", flush=True)
    print(f"  Saved feats/findwalking_bouts.pkl ({len(all_fw_bouts)} subjects)", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
