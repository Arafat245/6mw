#!/usr/bin/env python3
"""
Home Walking Detection + C2 Pipeline
=====================================
1. Extract raw accelerometer data from GT3X files (Accel files/)
2. Detect walking bouts using harmonic ratio on sliding windows
3. Select ~6 minutes of home walking most similar to clinic 6MW acceleration
4. Apply C2 preprocessing pipeline (gravity removal, alignment, bandpass)
5. Extract Gait13 + CWT features
6. Predict 6MWD with Ridge LOO

Walking detection: Harmonic ratio on sliding windows (Urbanek et al., 2018)
Segment selection: Rank walking bouts by similarity to subject's own clinic
  6MW accelerometer signature (VM distribution, cadence, energy) and pick
  the top bouts until ~6 min is reached.
"""

import numpy as np
import pandas as pd
import math
import re
from pathlib import Path
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import linregress
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import pywt
from pygt3x.reader import FileReader

BASE = Path(__file__).parent
ACCEL_DIR = BASE / "Accel files"
HOME_PREPROC_DIR = BASE / "csv_home_walking_preprocessed"
HOME_PREPROC_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# GT3X EXTRACTION
# ══════════════════════════════════════════════════════════════════

def find_gt3x_file(cohort, subj_id, year):
    """Find the GT3X file for a given subject."""
    # Try different folder naming patterns
    patterns = [
        f"{cohort}{subj_id:02d}_OPT",
        f"{cohort}{subj_id:02d}_OPT-{year}",
        f"{cohort}{subj_id:02d}_OPT {year}",
        f"{cohort}{subj_id:02d}_MW-{year}",
    ]
    for pat in patterns:
        folder = ACCEL_DIR / pat
        if folder.exists():
            gt3x_files = list(folder.glob("*.gt3x"))
            if gt3x_files:
                return gt3x_files[0]
    return None


def read_gt3x(gt3x_path, hour_start=8, hour_end=23):
    """Read GT3X file, filter 8am–11pm local, calibrate. Return (N,3), fs."""
    with FileReader(str(gt3x_path)) as reader:
        fs = reader.info.sample_rate
        accel_raw = reader.acceleration  # (N, 5): [timestamp, x_raw, y_raw, z_raw, idle]

        # Parse timezone offset
        tz_str = str(reader.info.timezone)
        tz_offset_sec = 0
        try:
            parts = tz_str.strip().split(':')
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            tz_offset_sec = h * 3600 + (m if h >= 0 else -m) * 60
        except (ValueError, IndexError):
            pass

        # Filter to daytime BEFORE calibration (saves time)
        unix_ts = accel_raw[:, 0]
        local_hours = ((unix_ts + tz_offset_sec) % 86400) / 3600
        mask = (local_hours >= hour_start) & (local_hours < hour_end)
        accel_day = accel_raw[mask]

        if len(accel_day) == 0:
            return np.empty((0, 3)), int(fs)

        # Calibrate only daytime data
        accel_cal = reader.calibrate_acceleration(accel_day)

    xyz = accel_cal[:, 1:4].astype(np.float64)
    return xyz, int(fs)


# ══════════════════════════════════════════════════════════════════
# WALKING DETECTION (Harmonic Ratio based)
# ══════════════════════════════════════════════════════════════════

def detect_walking_bouts(xyz, fs,
                         win_sec=10, step_sec=10,
                         hr_threshold=0.3,
                         min_cadence=0.8, max_cadence=3.5,
                         min_bout_sec=30):
    """
    Fast two-stage walking detection:
    Stage 1: Per-second ENMO → find active regions (vectorized)
    Stage 2: Bandpass filter whole signal once, then HR check on active windows only
    """
    n = len(xyz)
    if n < int(win_sec * fs * 2):
        return []

    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)

    # Stage 1: Per-second ENMO → active seconds (vectorized)
    sec_samples = int(fs)
    n_secs = n // sec_samples
    enmo_sec = enmo[:n_secs * sec_samples].reshape(n_secs, sec_samples).mean(axis=1)
    active_secs = (enmo_sec >= 0.015) & (enmo_sec <= 0.5)

    # Find contiguous active regions ≥ win_sec seconds
    regions = []
    in_r, rs = False, 0
    for s in range(n_secs):
        if active_secs[s] and not in_r:
            rs = s; in_r = True
        elif not active_secs[s] and in_r:
            if s - rs >= win_sec:
                regions.append((rs * sec_samples, s * sec_samples))
            in_r = False
    if in_r and n_secs - rs >= win_sec:
        regions.append((rs * sec_samples, n_secs * sec_samples))

    if not regions:
        return []

    # Stage 2: Bandpass filter ONCE on whole VM
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b, a, vm - vm.mean())

    win_samples = int(win_sec * fs)
    step_samples = int(step_sec * fs)
    fft_freqs = np.fft.rfftfreq(win_samples, d=1.0 / fs)
    band = (fft_freqs >= min_cadence) & (fft_freqs <= max_cadence)

    walking_windows = []

    for reg_s, reg_e in regions:
        for wi_start in range(reg_s, reg_e - win_samples, step_samples):
            seg = vm_bp[wi_start:wi_start + win_samples]

            X = np.fft.rfft(seg)
            mags = np.abs(X)
            cadence = fft_freqs[band][np.argmax(mags[band])]

            # Harmonic ratio
            even_sum, odd_sum = 0.0, 0.0
            for k in range(1, 11):
                fk = k * cadence
                if fk >= fft_freqs[-1]: break
                idx = int(np.argmin(np.abs(fft_freqs - fk)))
                if k % 2 == 0: even_sum += mags[idx]
                else: odd_sum += mags[idx]
            hr = even_sum / (odd_sum + 1e-12) if odd_sum > 0 else 0

            # Step regularity
            lag = int(round(fs / cadence)) if cadence > 0 else int(fs)
            lag = max(1, min(lag, len(seg) - 1))
            seg_dm = seg - seg.mean()
            denom = np.dot(seg_dm, seg_dm)
            if denom < 1e-12: continue
            acf_val = np.dot(seg_dm[:len(seg) - lag], seg_dm[lag:]) / denom

            if hr >= hr_threshold and acf_val > 0.3:
                walking_windows.append((wi_start, wi_start + win_samples))

    if not walking_windows:
        return []

    # Merge adjacent windows
    walking_windows.sort()
    bouts = []
    cs, ce = walking_windows[0]
    for ws, we in walking_windows[1:]:
        if ws <= ce + step_samples:
            ce = max(ce, we)
        else:
            if (ce - cs) >= min_bout_sec * fs:
                bouts.append((cs, ce))
            cs, ce = ws, we
    if (ce - cs) >= min_bout_sec * fs:
        bouts.append((cs, ce))

    return bouts


def compute_walking_signature(xyz_seg, fs):
    """
    Compute a compact signature of a walking segment for similarity comparison.
    Returns a feature vector: [vm_mean, vm_std, cadence, step_reg, energy, vm_p25, vm_p75]
    """
    vm = np.sqrt(xyz_seg[:, 0]**2 + xyz_seg[:, 1]**2 + xyz_seg[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)

    # VM stats
    vm_mean = np.mean(enmo)
    vm_std = np.std(enmo)
    vm_p25 = np.percentile(enmo, 25)
    vm_p75 = np.percentile(enmo, 75)

    # Cadence from PSD
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    if len(vm) > 3 * max(len(b), len(a)):
        vm_bp = filtfilt(b, a, vm - np.mean(vm))
        nperseg = min(len(vm_bp), int(fs * 4))
        if nperseg >= int(fs):
            freqs, pxx = welch(vm_bp, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
            band = (freqs >= 0.5) & (freqs <= 3.5)
            cadence = freqs[band][np.argmax(pxx[band])] if np.any(band) else 1.5
        else:
            cadence = 1.5
        # Step regularity via autocorrelation
        lag = int(round(fs / cadence)) if cadence > 0 else int(fs)
        lag = max(1, min(lag, len(vm_bp) - 1))
        x = vm_bp - np.mean(vm_bp)
        denom = np.dot(x, x)
        step_reg = np.dot(x[:len(x)-lag], x[lag:]) / (denom + 1e-12) if denom > 0 else 0
    else:
        cadence = 1.5
        step_reg = 0.0

    # Energy
    energy = np.mean(enmo**2)

    return np.array([vm_mean, vm_std, cadence, step_reg, energy, vm_p25, vm_p75])


def select_walking_segment(xyz, fs, bouts, target_sec=360, clinic_xyz=None, clinic_fs=None,
                           min_sim=0.85):
    """
    Select walking data most similar to clinic 6MW data.

    Strategy:
      1. Score each bout by cosine similarity to clinic signature
      2. Keep only bouts above min_sim threshold
      3. Collect top bouts up to ~target_sec (but accept less if quality is high)
      4. Return None if no bouts pass quality threshold

    Returns: (walking_xyz, best_similarity) or (None, 0)
    """
    if not bouts:
        return None, 0.0

    bout_info = [(s, e, (e - s) / fs) for s, e in bouts]

    # Filter out very short bouts (< 10 seconds)
    bout_info = [(s, e, d) for s, e, d in bout_info if d >= 10]
    if not bout_info:
        return None, 0.0

    if clinic_xyz is not None and clinic_fs is not None:
        # Compute clinic signature
        clinic_sig = compute_walking_signature(clinic_xyz, clinic_fs)

        # Score each bout by similarity to clinic
        scored = []
        for s, e, dur in bout_info:
            bout_sig = compute_walking_signature(xyz[s:e], fs)
            dot = np.dot(clinic_sig, bout_sig)
            norm = (np.linalg.norm(clinic_sig) * np.linalg.norm(bout_sig) + 1e-12)
            sim = dot / norm
            scored.append((s, e, dur, sim))

        # Sort by similarity (highest first)
        scored.sort(key=lambda x: x[3], reverse=True)

        # Keep only bouts above quality threshold
        good_bouts = [(s, e, dur, sim) for s, e, dur, sim in scored if sim >= min_sim]

        if not good_bouts:
            # Subject has no clinic-like walking — discard
            return None, scored[0][3] if scored else 0.0
    else:
        # No clinic data — sort by duration
        good_bouts = [(s, e, dur, 1.0) for s, e, dur in bout_info]
        good_bouts.sort(key=lambda x: x[2], reverse=True)

    # Collect bouts up to target duration
    collected = []
    total_samples = 0
    target_samples = int(target_sec * fs)
    best_sim = good_bouts[0][3]

    for s, e, dur, sim in good_bouts:
        need = target_samples - total_samples
        take = min(e - s, need)
        collected.append(xyz[s:s + take])
        total_samples += take
        if total_samples >= target_samples:
            break

    # Accept any amount if quality is high (min 30s for feature extraction)
    if total_samples < 30 * fs:
        return None, best_sim

    return np.concatenate(collected, axis=0), best_sim


# ══════════════════════════════════════════════════════════════════
# C2 PREPROCESSING (from reproduce_c2.py)
# ══════════════════════════════════════════════════════════════════

def butter_lowpass(cut_hz, fs, order=4):
    b, a = butter(N=order, Wn=cut_hz, btype="lowpass", fs=fs)
    return b, a

def butter_bandpass(lo_hz, hi_hz, fs, order=4):
    b, a = butter(N=order, Wn=[lo_hz, hi_hz], btype="bandpass", fs=fs)
    return b, a

def zero_phase_filter(x, b, a):
    y = np.empty_like(x)
    for j in range(x.shape[1]):
        y[:, j] = filtfilt(b, a, x[:, j], axis=0)
    return y

def _rodrigues(axis, theta):
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    return I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

def resample_uniform(arr, src_fs, dst_fs):
    n_src, d = arr.shape
    dur = (n_src - 1) / src_fs
    n_dst = int(round(dur * dst_fs)) + 1
    oldt = np.linspace(0.0, dur, num=n_src, endpoint=True)
    newt = np.linspace(0.0, dur, num=n_dst, endpoint=True)
    out = np.empty((n_dst, d), dtype=np.float64)
    for j in range(d):
        out[:, j] = np.interp(newt, oldt, arr[:, j])
    return out

def preprocess_walking_segment(walking_xyz, fs_orig, target_fs=30.0):
    """
    Apply C2 preprocessing to walking segment.
    Same as reproduce_c2.py but no trimming (already selected walking).
    """
    # Resample to target_fs if needed
    if fs_orig != target_fs:
        arr = resample_uniform(walking_xyz, fs_orig, target_fs)
    else:
        arr = walking_xyz.copy()
    fs = target_fs

    # Gravity removal
    b, a = butter_lowpass(0.25, fs, 4)
    g_est = zero_phase_filter(arr, b, a)
    arr_dyn = arr - g_est

    # Rodrigues rotation (align gravity to Z)
    g_mean = g_est.mean(axis=0)
    zhat = np.array([0.0, 0.0, 1.0])
    gvec = g_mean / (np.linalg.norm(g_mean) + 1e-12)
    dot = np.clip(float(zhat @ gvec), -1.0, 1.0)
    angle = math.acos(dot)
    if angle > 1e-4:
        axis = np.cross(gvec, zhat)
        if np.linalg.norm(axis) < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        R = _rodrigues(axis, angle)
        arr_v = arr_dyn @ R.T
    else:
        arr_v = arr_dyn.copy()

    # PCA yaw alignment
    XY = arr_v[:, :2]
    C = np.cov(XY, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    ap_dir = vecs[:, np.argmax(vals)]
    theta = math.atan2(float(ap_dir[1]), float(ap_dir[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    apmlvt = arr_v @ Rz.T

    # Bandpass filter
    b, a = butter_bandpass(0.25, 2.5, fs, order=4)
    apmlvt_bp = zero_phase_filter(apmlvt, b, a)

    # Derived signals
    vm_dyn = np.linalg.norm(apmlvt, axis=1)
    vm_raw = np.linalg.norm(arr, axis=1)
    enmo = np.maximum(vm_raw - 1.0, 0.0)

    out = pd.DataFrame({
        "AP": apmlvt[:, 0], "ML": apmlvt[:, 1], "VT": apmlvt[:, 2],
        "AP_bp": apmlvt_bp[:, 0], "ML_bp": apmlvt_bp[:, 1], "VT_bp": apmlvt_bp[:, 2],
        "VM_dyn": vm_dyn, "VM_raw": vm_raw, "ENMO": enmo,
        "fs": fs,
    })
    return out


# ══════════════════════════════════════════════════════════════════
# GAIT13 FEATURE EXTRACTION (from reproduce_c2.py)
# ══════════════════════════════════════════════════════════════════

def _psd_peak_freq(x, fs, fmin=0.5, fmax=3.5):
    if len(x) < int(fs): return float("nan")
    nperseg = int(max(fs * 4, 256))
    freqs, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band): return float("nan")
    return float(freqs[band][np.argmax(Pxx[band])])

def _acf(x, max_lag):
    x = np.asarray(x, float); x = x - np.mean(x); n = len(x)
    if n <= 1: return np.zeros(max_lag + 1)
    denom = np.dot(x, x)
    ac = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        ac[k] = np.dot(x[:n-k], x[k:]) / (denom if denom > 0 else 1.0)
    return ac

def _harmonic_ratio(signal, fs, cadence_hz, n_harm=10):
    if not np.isfinite(cadence_hz) or cadence_hz <= 0: return float("nan")
    x = signal - np.mean(signal); n = len(x)
    if n < 2: return float("nan")
    X = np.fft.rfft(x); freqs = np.fft.rfftfreq(n, d=1.0/fs); mags = np.abs(X)
    ev, od = 0.0, 0.0
    for k in range(1, n_harm + 1):
        fk = k * cadence_hz
        if fk >= freqs[-1]: break
        idx = int(np.argmin(np.abs(freqs - fk)))
        if k % 2 == 0: ev += mags[idx]
        else: od += mags[idx]
    return float(ev / od) if od > 0 else float("nan")

def extract_gait10(df):
    fs = float(df["fs"].iloc[0])
    vt_bp = df["VT_bp"].to_numpy(float)
    ap_bp = df["AP_bp"].to_numpy(float)
    ml_bp = df["ML_bp"].to_numpy(float)
    ml_dyn = df["ML"].to_numpy(float)
    ap_dyn = df["AP"].to_numpy(float)
    vt_dyn = df["VT"].to_numpy(float)
    enmo = df["ENMO"].to_numpy(float)

    cad = _psd_peak_freq(vt_bp, fs)
    f = {"cadence_hz": cad}

    if np.isfinite(cad) and cad > 0:
        min_dist = max(1, int(round(0.5 * fs / cad)))
        prom = 0.5 * np.std(vt_bp) if np.std(vt_bp) > 0 else 0.0
        peaks, _ = find_peaks(vt_bp, distance=min_dist, prominence=prom)
        if peaks.size >= 3:
            si = np.diff(peaks) / fs
            f["step_time_cv_pct"] = 100 * np.std(si, ddof=1) / np.mean(si) if np.mean(si) > 0 else float("nan")
        else: f["step_time_cv_pct"] = float("nan")
        lag1 = int(np.clip(round(fs / cad), 1, 1e7))
        ac = _acf(vt_bp, lag1 * 3)
        f["acf_step_regularity"] = float(ac[lag1]) if lag1 < ac.size else float("nan")
    else:
        f["step_time_cv_pct"] = float("nan")
        f["acf_step_regularity"] = float("nan")

    f["hr_ap"] = _harmonic_ratio(ap_bp, fs, cad)
    f["hr_vt"] = _harmonic_ratio(vt_bp, fs, cad)
    f["ml_rms_g"] = float(np.sqrt(np.mean(np.square(ml_dyn))))

    if np.isfinite(cad) and cad > 0:
        lo = max(0.25, 0.5 * cad); hi = min(3.5, 3.0 * cad)
        nperseg = int(max(fs * 4, 256))
        freqs, Pxx = welch(ml_bp, fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend="constant")
        band = (freqs >= lo) & (freqs <= hi)
        if np.any(band):
            p = Pxx[band]; s = p.sum()
            if s > 0: p = p/s; ent = -(p * np.log(p + 1e-12)).sum(); f["ml_spectral_entropy"] = ent / np.log(len(p))
            else: f["ml_spectral_entropy"] = float("nan")
        else: f["ml_spectral_entropy"] = float("nan")
    else: f["ml_spectral_entropy"] = float("nan")

    vm = np.linalg.norm(np.c_[ap_dyn, ml_dyn, vt_dyn], axis=1)
    f["jerk_mean_abs_gps"] = float(np.mean(np.abs(np.diff(vm) * fs)))
    f["enmo_mean_g"] = float(np.mean(enmo))

    per_min = int(round(60 * fs)); m = min(6, max(1, len(vt_bp) // per_min))
    cads = [_psd_peak_freq(vt_bp[i*per_min:(i+1)*per_min], fs) for i in range(m)
            if len(vt_bp[i*per_min:(i+1)*per_min]) >= per_min // 2]
    cads = np.array([c for c in cads if np.isfinite(c)], dtype=float)
    if len(cads) >= 3:
        slope, _ = np.polyfit(np.arange(len(cads)), cads, 1)
        f["cadence_slope_per_min"] = float(slope)
    else: f["cadence_slope_per_min"] = float("nan")

    return f


# ══════════════════════════════════════════════════════════════════
# CWT EXTRACTION (from reproduce_c2.py)
# ══════════════════════════════════════════════════════════════════

def extract_cwt(raw_sig, fs=30.0, n_seg=6):
    vm = np.sqrt(raw_sig[:,0]**2 + raw_sig[:,1]**2 + raw_sig[:,2]**2)
    vm_c = vm - vm.mean()
    freqs = np.linspace(0.5, 12, 50)
    def cwt_seg(seg_raw):
        s = seg_raw / (np.max(np.abs(seg_raw)) + 1e-12)
        scales = fs / (freqs + 1e-12)
        coeffs, _ = pywt.cwt(s, scales, 'morl', sampling_period=1.0/fs)
        pw = np.abs(coeffs)**2; mp = pw.mean(1)
        f = {}
        f['mean_energy'] = np.mean(pw)
        hm = freqs >= 3.5; f['high_freq_energy'] = np.mean(pw[hm]) if hm.any() else 0
        f['dominant_freq'] = freqs[np.argmax(mp)]
        gm = (freqs >= 0.5) & (freqs <= 3.5); gp = mp.copy(); gp[~gm] = 0
        f['estimated_cadence'] = freqs[np.argmax(gp)] * 60
        f['max_power_freq'] = freqs[np.unravel_index(np.argmax(pw), pw.shape)[0]]
        nw = max(1, pw.shape[1] // int(fs))
        dl = [freqs[np.argmax(pw[:,w*int(fs):min((w+1)*int(fs),pw.shape[1])].mean(1))] for w in range(nw)]
        f['freq_variability'] = np.std(dl); f['freq_cv'] = np.std(dl) / (np.mean(dl) + 1e-12)
        pn = mp / (mp.sum() + 1e-12); pnz = pn[pn > 0]
        f['wavelet_entropy'] = -np.sum(pnz * np.log2(pnz + 1e-12))
        from scipy.fft import rfft, rfftfreq
        fv = np.abs(rfft(s)); ff = rfftfreq(len(s), 1/fs); gb = (ff >= 0.5) & (ff <= 3.5)
        if gb.any():
            f0 = ff[gb][np.argmax(fv[gb])]; f['fundamental_freq'] = f0
            if f0 > 0:
                ep, op = 0, 0
                for h in range(1, 11):
                    idx = np.argmin(np.abs(ff - h*f0))
                    if h % 2 == 0: ep += fv[idx]**2
                    else: op += fv[idx]**2
                f['harmonic_ratio'] = ep / (op + 1e-12)
            else: f['harmonic_ratio'] = 0
        else: f['fundamental_freq'] = 0; f['harmonic_ratio'] = 0
        return f
    T = len(vm_c); sl = T // n_seg; sfs = []
    for i in range(n_seg):
        s, e = i*sl, min((i+1)*sl, T)
        if e - s < int(2*fs): continue
        sfs.append(cwt_seg(vm_c[s:e]))
    if not sfs: sfs = [cwt_seg(vm_c)]
    df = pd.DataFrame(sfs)
    f = {f"cwt_{k}_mean": df[k].mean() for k in df.columns}
    f.update({f"cwt_{k}_std": df[k].std() for k in df.columns})
    for key in ["mean_energy", "high_freq_energy", "freq_variability", "wavelet_entropy"]:
        if key in df.columns and len(df) >= 3:
            sl2, _, rv, _, _ = linregress(np.arange(len(df)), df[key].values)
            f[f"cwt_{key}_slope"] = sl2; f[f"cwt_{key}_slope_r"] = rv
        else: f[f"cwt_{key}_slope"] = 0; f[f"cwt_{key}_slope_r"] = 0
    return f


# ══════════════════════════════════════════════════════════════════
# LOO PREDICTION
# ══════════════════════════════════════════════════════════════════

def loo(X, y):
    pred = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=10)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return round(r2_score(y, pred), 4)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def get_fs_from_timestamps(timestamps):
    """Get actual sampling rate from timestamp column."""
    diffs = np.diff(timestamps)
    diffs_pos = diffs[diffs > 0]
    if len(diffs_pos) == 0:
        return 30.0
    median_dt = np.median(diffs_pos)
    return round(1.0 / median_dt)


def load_clinic_raw(cohort, subj_id, year, sixmwd):
    """Load raw clinic 6MW data from csv_raw2 for this subject."""
    RAW = BASE / "csv_raw2"
    fn = f"{cohort}{subj_id:02d}_{year}_{sixmwd}.csv"
    p = RAW / fn
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    timestamps = df["Timestamp"].values
    fs = get_fs_from_timestamps(timestamps)
    xyz = df[["X", "Y", "Z"]].values.astype(np.float64)
    # Trim 10s from each end (same as C2 pipeline)
    n_trim = int(10 * fs)
    if 2 * n_trim >= len(xyz):
        return xyz, int(fs)
    xyz = xyz[n_trim:len(xyz) - n_trim]
    return xyz, int(fs)


if __name__ == "__main__":
    ids = pd.read_csv(BASE / "feats" / "target_6mwd.csv")
    y_all = ids["sixmwd"].values.astype(float)

    print("=" * 70)
    print("HOME WALKING DETECTION + C2 PIPELINE")
    print("=" * 70)
    print("Segment selection: ranked by similarity to clinic 6MW acceleration")

    # ── Step 1: Extract GT3X and detect walking ──
    print("\nStep 1: GT3X → walking detection → clinic-matched selection → preprocess...")

    gait_rows = []
    cwt_rows = []
    valid_mask = []
    walk_stats = []

    for i, (_, r) in enumerate(ids.iterrows()):
        cohort = r["cohort"]
        subj_id = int(r["subj_id"])
        year = int(r["year"])
        sixmwd = int(r["sixmwd"])
        label = f"{cohort}{subj_id:02d}"

        gt3x = find_gt3x_file(cohort, subj_id, year)
        if gt3x is None:
            print(f"  [{i+1:3d}] {label}: GT3X not found — SKIPPED")
            valid_mask.append(False)
            continue

        try:
            # Read GT3X
            xyz_raw, fs_orig = read_gt3x(gt3x)
            total_hours = len(xyz_raw) / fs_orig / 3600

            # Load clinic 6MW data for this subject (template)
            clinic_xyz, clinic_fs = load_clinic_raw(cohort, subj_id, year, sixmwd)

            # Detect walking bouts
            bouts = detect_walking_bouts(xyz_raw, fs_orig)
            total_walk_sec = sum((e - s) / fs_orig for s, e in bouts)
            n_bouts = len(bouts)

            # Select walking segments ranked by similarity to clinic data
            walking_seg, best_sim = select_walking_segment(
                xyz_raw, fs_orig, bouts, target_sec=360,
                clinic_xyz=clinic_xyz, clinic_fs=clinic_fs,
                min_sim=0.85
            )

            has_clinic = clinic_xyz is not None

            if walking_seg is None:
                reason = f"low similarity ({best_sim:.3f})" if n_bouts > 0 and has_clinic else f"{n_bouts} bouts ({total_walk_sec:.0f}s)"
                print(f"  [{i+1:3d}] {label}: {total_hours:.1f}h, {reason} — DISCARDED")
                valid_mask.append(False)
                continue

            seg_duration = len(walking_seg) / fs_orig
            walk_stats.append({
                'subject': label, 'recording_hours': total_hours,
                'n_bouts': n_bouts, 'total_walk_sec': total_walk_sec,
                'selected_sec': seg_duration, 'clinic_matched': has_clinic,
                'best_sim': best_sim
            })

            # Preprocess (C2 pipeline)
            preproc_df = preprocess_walking_segment(walking_seg, fs_orig, target_fs=30.0)

            # Save preprocessed file
            out_fn = f"{cohort}{subj_id:02d}_{year}_{sixmwd}.csv"
            preproc_df.to_csv(HOME_PREPROC_DIR / out_fn, index=False)

            # Extract Gait13
            gait10 = extract_gait10(preproc_df)
            vt_rms = float(np.sqrt(np.mean(np.square(preproc_df["VT"].values))))
            gait10["vt_rms_g"] = vt_rms
            ml_rms = gait10["ml_rms_g"]
            enmo_mean = gait10["enmo_mean_g"]
            gait10["ml_over_enmo"] = ml_rms / enmo_mean if enmo_mean > 0 else np.nan
            gait10["ml_over_vt"] = ml_rms / vt_rms if vt_rms > 0 else np.nan
            gait_rows.append(gait10)

            # Extract CWT from walking segment (resample to 30Hz first for consistency)
            if fs_orig != 30:
                walking_30 = resample_uniform(walking_seg, fs_orig, 30.0)
            else:
                walking_30 = walking_seg
            cwt_feats = extract_cwt(walking_30, fs=30.0)
            cwt_rows.append(cwt_feats)

            valid_mask.append(True)
            matched_str = f"sim={best_sim:.3f}" if has_clinic else "no-clinic"
            print(f"  [{i+1:3d}] {label}: {seg_duration:.0f}s selected ({matched_str}), "
                  f"{n_bouts} bouts, {total_walk_sec:.0f}s total", flush=True)

        except Exception as e:
            print(f"  [{i+1:3d}] {label}: ERROR — {e}")
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    n_valid = sum(valid_mask)
    print(f"\n  Processed: {n_valid}/{len(ids)} subjects")

    # Walking stats
    if walk_stats:
        ws = pd.DataFrame(walk_stats)
        n_matched = ws['clinic_matched'].sum()
        print(f"\n  Walking detection stats ({n_valid} valid, {n_matched} clinic-matched):")
        print(f"    Bouts per subject: mean={ws['n_bouts'].mean():.1f}, "
              f"min={ws['n_bouts'].min()}, max={ws['n_bouts'].max()}")
        print(f"    Total walking (sec): mean={ws['total_walk_sec'].mean():.0f}, "
              f"min={ws['total_walk_sec'].min():.0f}, max={ws['total_walk_sec'].max():.0f}")
        print(f"    Selected segment (sec): mean={ws['selected_sec'].mean():.0f}, "
              f"min={ws['selected_sec'].min():.0f}, max={ws['selected_sec'].max():.0f}")
        if 'best_sim' in ws.columns:
            print(f"    Clinic similarity: mean={ws['best_sim'].mean():.3f}, "
                  f"min={ws['best_sim'].min():.3f}, max={ws['best_sim'].max():.3f}")

    # ── Step 2: Build feature matrices ──
    sway_cols = ["cadence_hz", "step_time_cv_pct", "acf_step_regularity", "hr_ap", "hr_vt",
                 "ml_rms_g", "ml_spectral_entropy", "jerk_mean_abs_gps", "enmo_mean_g",
                 "cadence_slope_per_min", "vt_rms_g", "ml_over_enmo", "ml_over_vt"]

    gait_df = pd.DataFrame(gait_rows)
    X_gait = gait_df[sway_cols].values.astype(float)
    for j in range(X_gait.shape[1]):
        m = np.isnan(X_gait[:, j])
        if m.any(): X_gait[m, j] = np.nanmedian(X_gait[:, j])

    cwt_df = pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan)
    for c in cwt_df.columns:
        if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(cwt_df[c].median())
    X_cwt = cwt_df.values.astype(float)

    # Demographics
    demo = pd.read_excel(BASE / "SwayDemographics.xlsx")
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    p = ids[valid_mask].reset_index(drop=True).merge(demo, on=["cohort", "subj_id"], how="left")
    p["cohort_M"] = (p["cohort"] == "M").astype(int)
    for c in ["Age", "Sex", "Height"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    X_demo = p[["cohort_M", "Age", "Sex", "Height"]].values.astype(float)
    for j in range(X_demo.shape[1]):
        m = np.isnan(X_demo[:, j])
        if m.any(): X_demo[m, j] = np.nanmedian(X_demo[:, j])

    y = y_all[valid_mask]

    # ── Step 3: Predict ──
    print(f"\n{'='*60}")
    print(f"RESULTS — Home Walking C2 Pipeline (n={len(y)})")
    print(f"{'='*60}")

    print(f"\n  A1 (no demographics):")
    r2_g13 = loo(X_gait, y)
    r2_cwt = loo(X_cwt, y)
    r2_g13cwt = loo(np.column_stack([X_gait, X_cwt]), y)
    print(f"    Gait13 only        (13f):  R² = {r2_g13}")
    print(f"    CWT only           (28f):  R² = {r2_cwt}")
    print(f"    Gait13 + CWT       (41f):  R² = {r2_g13cwt}")

    print(f"\n  A2 (with demographics):")
    r2_demo = loo(X_demo, y)
    r2_g13d = loo(np.column_stack([X_gait, X_demo]), y)
    r2_cwtd = loo(np.column_stack([X_cwt, X_demo]), y)
    r2_all = loo(np.column_stack([X_gait, X_cwt, X_demo]), y)
    print(f"    Demo only           (4f):  R² = {r2_demo}")
    print(f"    Gait13 + Demo      (17f):  R² = {r2_g13d}")
    print(f"    CWT + Demo         (32f):  R² = {r2_cwtd}")
    print(f"    Gait13+CWT+Demo    (45f):  R² = {r2_all}")

    print(f"\n  C2 reference (clinic):        R² = 0.791")
