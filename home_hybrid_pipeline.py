#!/usr/bin/env python3
"""
Hybrid Home Pipeline — Best of Both Approaches
================================================
From new pipeline:  GT3X reading, 8am-11pm, C2 preprocessing, Gait13+CWT, clinic similarity
From old pipeline:  Keep ALL subjects, permissive ENMO detection, longer segments
Added:              Activity Profile features (best previous A1=0.146)

Key design decisions:
  1. ENMO-based activity detection (permissive) — gets more walking data
  2. Clinic similarity for RANKING, not filtering — no subjects discarded
  3. Fallback to longest bouts if no clinic data or low similarity
  4. Minimum 30s for gait features, use full daytime for activity features
  5. Combined feature set: Gait13 + CWT + Activity Profile + Circadian
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
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
HOME_HYBRID_DIR = BASE / "csv_home_hybrid_preprocessed"
HOME_HYBRID_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# GT3X READING (from new pipeline)
# ══════════════════════════════════════════════════════════════════

def find_gt3x_file(cohort, subj_id, year):
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


def read_gt3x_daytime(gt3x_path, hour_start=8, hour_end=23):
    """Read GT3X, return daytime (8am-11pm) data and full data."""
    with FileReader(str(gt3x_path)) as reader:
        fs = int(reader.info.sample_rate)
        accel_raw = reader.acceleration

        tz_str = str(reader.info.timezone)
        tz_offset_sec = 0
        try:
            parts = tz_str.strip().split(':')
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            tz_offset_sec = h * 3600 + (m if h >= 0 else -m) * 60
        except (ValueError, IndexError):
            pass

        # Filter daytime BEFORE calibration
        unix_ts = accel_raw[:, 0]
        local_hours = ((unix_ts + tz_offset_sec) % 86400) / 3600
        day_mask = (local_hours >= hour_start) & (local_hours < hour_end)
        accel_day_raw = accel_raw[day_mask]

        if len(accel_day_raw) == 0:
            return np.empty((0, 3)), fs

        accel_cal = reader.calibrate_acceleration(accel_day_raw)

    xyz = accel_cal[:, 1:4].astype(np.float64)
    return xyz, fs


# ══════════════════════════════════════════════════════════════════
# WALKING DETECTION — Permissive ENMO + optional HR refinement
# ══════════════════════════════════════════════════════════════════

def detect_active_bouts(xyz, fs, min_bout_sec=30):
    """
    Stage 1: ENMO-based activity detection (permissive, like old pipeline).
    Returns all active bouts >= min_bout_sec.
    """
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)

    sec = int(fs)
    n_secs = len(enmo) // sec
    if n_secs < min_bout_sec:
        return []

    enmo_sec = enmo[:n_secs * sec].reshape(n_secs, sec).mean(axis=1)

    # Permissive threshold: any movement above sedentary
    active = enmo_sec >= 0.015

    # Find contiguous active regions
    bouts = []
    in_b, bs = False, 0
    for s in range(n_secs):
        if active[s] and not in_b:
            bs = s; in_b = True
        elif not active[s] and in_b:
            if s - bs >= min_bout_sec:
                bouts.append((bs * sec, s * sec))
            in_b = False
    if in_b and n_secs - bs >= min_bout_sec:
        bouts.append((bs * sec, n_secs * sec))

    return bouts


def refine_with_hr(xyz, fs, bouts, hr_threshold=0.2):
    """
    Stage 2: Within each active bout, identify 10s windows with
    walking-like harmonic structure. Return sub-bouts that are walking.
    If no windows pass HR, return original bout (permissive fallback).
    """
    if not bouts:
        return []

    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    # Bandpass filter once
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b, a, vm - vm.mean())

    win = int(10 * fs)
    step = int(10 * fs)
    fft_freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band = (fft_freqs >= 0.8) & (fft_freqs <= 3.5)

    refined_bouts = []

    for bout_s, bout_e in bouts:
        walking_wins = []
        for wi in range(bout_s, bout_e - win, step):
            seg = vm_bp[wi:wi + win]

            X = np.fft.rfft(seg)
            mags = np.abs(X)
            if not np.any(band):
                continue
            cadence = fft_freqs[band][np.argmax(mags[band])]

            # Harmonic ratio
            even, odd = 0.0, 0.0
            for k in range(1, 11):
                fk = k * cadence
                if fk >= fft_freqs[-1]: break
                idx = int(np.argmin(np.abs(fft_freqs - fk)))
                if k % 2 == 0: even += mags[idx]
                else: odd += mags[idx]
            hr = even / (odd + 1e-12) if odd > 0 else 0

            if hr >= hr_threshold:
                walking_wins.append((wi, wi + win))

        if walking_wins:
            # Merge adjacent walking windows
            cs, ce = walking_wins[0]
            for ws, we in walking_wins[1:]:
                if ws <= ce + step:
                    ce = max(ce, we)
                else:
                    if ce - cs >= 30 * fs:
                        refined_bouts.append((cs, ce))
                    cs, ce = ws, we
            if ce - cs >= 30 * fs:
                refined_bouts.append((cs, ce))
        else:
            # Fallback: keep original bout even without HR confirmation
            refined_bouts.append((bout_s, bout_e))

    return refined_bouts


# ══════════════════════════════════════════════════════════════════
# CLINIC SIMILARITY SCORING (from new pipeline)
# ══════════════════════════════════════════════════════════════════

def compute_walking_signature(xyz_seg, fs):
    vm = np.sqrt(xyz_seg[:, 0]**2 + xyz_seg[:, 1]**2 + xyz_seg[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    vm_mean = np.mean(enmo)
    vm_std = np.std(enmo)
    vm_p25 = np.percentile(enmo, 25)
    vm_p75 = np.percentile(enmo, 75)

    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    cadence, step_reg, energy = 1.5, 0.0, np.mean(enmo**2)
    if len(vm) > 3 * max(len(b), len(a)):
        vm_bp = filtfilt(b, a, vm - np.mean(vm))
        nperseg = min(len(vm_bp), int(fs * 4))
        if nperseg >= int(fs):
            freqs, pxx = welch(vm_bp, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            band = (freqs >= 0.5) & (freqs <= 3.5)
            if np.any(band):
                cadence = freqs[band][np.argmax(pxx[band])]
        lag = int(round(fs / cadence)) if cadence > 0 else int(fs)
        lag = max(1, min(lag, len(vm_bp) - 1))
        x = vm_bp - np.mean(vm_bp)
        denom = np.dot(x, x)
        step_reg = np.dot(x[:len(x) - lag], x[lag:]) / (denom + 1e-12) if denom > 0 else 0

    return np.array([vm_mean, vm_std, cadence, step_reg, energy, vm_p25, vm_p75])


def select_walking_segment(xyz, fs, bouts, target_sec=360, clinic_xyz=None, clinic_fs=None):
    """
    Select walking bouts ranked by clinic similarity.
    NO DISCARDING — always returns something if there are any bouts.
    """
    if not bouts:
        return None

    bout_info = [(s, e, (e - s) / fs) for s, e in bouts]

    if clinic_xyz is not None and clinic_fs is not None and len(clinic_xyz) > 100:
        clinic_sig = compute_walking_signature(clinic_xyz, clinic_fs)
        scored = []
        for s, e, dur in bout_info:
            bout_sig = compute_walking_signature(xyz[s:e], fs)
            dot = np.dot(clinic_sig, bout_sig)
            norm = (np.linalg.norm(clinic_sig) * np.linalg.norm(bout_sig) + 1e-12)
            sim = dot / norm
            scored.append((s, e, dur, sim))
        scored.sort(key=lambda x: x[3], reverse=True)
    else:
        scored = [(s, e, dur, 0.0) for s, e, dur in bout_info]
        scored.sort(key=lambda x: x[2], reverse=True)

    collected = []
    total_samples = 0
    target_samples = int(target_sec * fs)
    best_sim = scored[0][3] if scored else 0.0

    for s, e, dur, sim in scored:
        need = target_samples - total_samples
        take = min(e - s, need)
        collected.append(xyz[s:s + take])
        total_samples += take
        if total_samples >= target_samples:
            break

    if total_samples < 30 * fs:
        return None, 0.0

    return np.concatenate(collected, axis=0), best_sim


# ══════════════════════════════════════════════════════════════════
# CLINIC DATA LOADING
# ══════════════════════════════════════════════════════════════════

def get_fs_from_timestamps(timestamps):
    diffs = np.diff(timestamps)
    diffs_pos = diffs[diffs > 0]
    if len(diffs_pos) == 0: return 30.0
    return round(1.0 / np.median(diffs_pos))

def load_clinic_raw(cohort, subj_id, year, sixmwd):
    RAW = BASE / "csv_raw2"
    fn = f"{cohort}{subj_id:02d}_{year}_{sixmwd}.csv"
    p = RAW / fn
    if not p.exists(): return None, None
    df = pd.read_csv(p)
    fs = get_fs_from_timestamps(df["Timestamp"].values)
    xyz = df[["X", "Y", "Z"]].values.astype(np.float64)
    n_trim = int(10 * fs)
    if 2 * n_trim >= len(xyz): return xyz, int(fs)
    return xyz[n_trim:len(xyz) - n_trim], int(fs)


# ══════════════════════════════════════════════════════════════════
# C2 PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def butter_lowpass(cut_hz, fs, order=4):
    return butter(N=order, Wn=cut_hz, btype="lowpass", fs=fs)

def butter_bandpass(lo_hz, hi_hz, fs, order=4):
    return butter(N=order, Wn=[lo_hz, hi_hz], btype="bandpass", fs=fs)

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
    if fs_orig != target_fs:
        arr = resample_uniform(walking_xyz, fs_orig, target_fs)
    else:
        arr = walking_xyz.copy()
    fs = target_fs

    b, a = butter_lowpass(0.25, fs, 4)
    g_est = zero_phase_filter(arr, b, a)
    arr_dyn = arr - g_est

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

    XY = arr_v[:, :2]
    C = np.cov(XY, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    ap_dir = vecs[:, np.argmax(vals)]
    theta = math.atan2(float(ap_dir[1]), float(ap_dir[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    apmlvt = arr_v @ Rz.T

    b, a = butter_bandpass(0.25, 2.5, fs, order=4)
    apmlvt_bp = zero_phase_filter(apmlvt, b, a)

    vm_dyn = np.linalg.norm(apmlvt, axis=1)
    vm_raw = np.linalg.norm(arr, axis=1)
    enmo = np.maximum(vm_raw - 1.0, 0.0)

    return pd.DataFrame({
        "AP": apmlvt[:, 0], "ML": apmlvt[:, 1], "VT": apmlvt[:, 2],
        "AP_bp": apmlvt_bp[:, 0], "ML_bp": apmlvt_bp[:, 1], "VT_bp": apmlvt_bp[:, 2],
        "VM_dyn": vm_dyn, "VM_raw": vm_raw, "ENMO": enmo, "fs": fs,
    })


# ══════════════════════════════════════════════════════════════════
# GAIT13 FEATURES (from reproduce_c2.py)
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
        ac[k] = np.dot(x[:n - k], x[k:]) / (denom if denom > 0 else 1.0)
    return ac

def _harmonic_ratio(signal, fs, cadence_hz, n_harm=10):
    if not np.isfinite(cadence_hz) or cadence_hz <= 0: return float("nan")
    x = signal - np.mean(signal); n = len(x)
    if n < 2: return float("nan")
    X = np.fft.rfft(x); freqs = np.fft.rfftfreq(n, d=1.0 / fs); mags = np.abs(X)
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
        freqs, Pxx = welch(ml_bp, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
        band = (freqs >= lo) & (freqs <= hi)
        if np.any(band):
            p = Pxx[band]; s = p.sum()
            if s > 0: p = p / s; ent = -(p * np.log(p + 1e-12)).sum(); f["ml_spectral_entropy"] = ent / np.log(len(p))
            else: f["ml_spectral_entropy"] = float("nan")
        else: f["ml_spectral_entropy"] = float("nan")
    else: f["ml_spectral_entropy"] = float("nan")

    vm = np.linalg.norm(np.c_[ap_dyn, ml_dyn, vt_dyn], axis=1)
    f["jerk_mean_abs_gps"] = float(np.mean(np.abs(np.diff(vm) * fs)))
    f["enmo_mean_g"] = float(np.mean(enmo))

    per_min = int(round(60 * fs)); m = min(6, max(1, len(vt_bp) // per_min))
    cads = [_psd_peak_freq(vt_bp[i * per_min:(i + 1) * per_min], fs) for i in range(m)
            if len(vt_bp[i * per_min:(i + 1) * per_min]) >= per_min // 2]
    cads = np.array([c for c in cads if np.isfinite(c)], dtype=float)
    if len(cads) >= 3:
        slope, _ = np.polyfit(np.arange(len(cads)), cads, 1)
        f["cadence_slope_per_min"] = float(slope)
    else: f["cadence_slope_per_min"] = float("nan")

    return f


# ══════════════════════════════════════════════════════════════════
# CWT FEATURES (from reproduce_c2.py)
# ══════════════════════════════════════════════════════════════════

def extract_cwt(raw_sig, fs=30.0, n_seg=6):
    vm = np.sqrt(raw_sig[:, 0]**2 + raw_sig[:, 1]**2 + raw_sig[:, 2]**2)
    vm_c = vm - vm.mean()
    freqs = np.linspace(0.5, 12, 50)
    def cwt_seg(seg_raw):
        s = seg_raw / (np.max(np.abs(seg_raw)) + 1e-12)
        scales = fs / (freqs + 1e-12)
        coeffs, _ = pywt.cwt(s, scales, 'morl', sampling_period=1.0 / fs)
        pw = np.abs(coeffs)**2; mp = pw.mean(1)
        f = {}
        f['mean_energy'] = np.mean(pw)
        hm = freqs >= 3.5; f['high_freq_energy'] = np.mean(pw[hm]) if hm.any() else 0
        f['dominant_freq'] = freqs[np.argmax(mp)]
        gm = (freqs >= 0.5) & (freqs <= 3.5); gp = mp.copy(); gp[~gm] = 0
        f['estimated_cadence'] = freqs[np.argmax(gp)] * 60
        f['max_power_freq'] = freqs[np.unravel_index(np.argmax(pw), pw.shape)[0]]
        nw = max(1, pw.shape[1] // int(fs))
        dl = [freqs[np.argmax(pw[:, w * int(fs):min((w + 1) * int(fs), pw.shape[1])].mean(1))] for w in range(nw)]
        f['freq_variability'] = np.std(dl); f['freq_cv'] = np.std(dl) / (np.mean(dl) + 1e-12)
        pn = mp / (mp.sum() + 1e-12); pnz = pn[pn > 0]
        f['wavelet_entropy'] = -np.sum(pnz * np.log2(pnz + 1e-12))
        from scipy.fft import rfft, rfftfreq
        fv = np.abs(rfft(s)); ff = rfftfreq(len(s), 1 / fs); gb = (ff >= 0.5) & (ff <= 3.5)
        if gb.any():
            f0 = ff[gb][np.argmax(fv[gb])]; f['fundamental_freq'] = f0
            if f0 > 0:
                ep, op = 0, 0
                for h in range(1, 11):
                    idx = np.argmin(np.abs(ff - h * f0))
                    if h % 2 == 0: ep += fv[idx]**2
                    else: op += fv[idx]**2
                f['harmonic_ratio'] = ep / (op + 1e-12)
            else: f['harmonic_ratio'] = 0
        else: f['fundamental_freq'] = 0; f['harmonic_ratio'] = 0
        return f
    T = len(vm_c); sl = T // n_seg; sfs = []
    for i in range(n_seg):
        s, e = i * sl, min((i + 1) * sl, T)
        if e - s < int(2 * fs): continue
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
# ACTIVITY PROFILE FEATURES (from extract_home_features_v2.py)
# ══════════════════════════════════════════════════════════════════

def extract_activity_features(xyz_daytime, fs, sed_thresh, mvpa_thresh):
    """Extract activity profile from full daytime data (not just walking)."""
    vm = np.sqrt(xyz_daytime[:, 0]**2 + xyz_daytime[:, 1]**2 + xyz_daytime[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)

    bs = int(fs)
    n_bins = len(enmo) // bs
    if n_bins < 10:
        return None
    enmo_sec = enmo[:n_bins * bs].reshape(n_bins, bs).mean(axis=1)
    total_hours = n_bins / 3600

    af = {}
    af['enmo_mean'] = np.mean(enmo_sec)
    af['enmo_std'] = np.std(enmo_sec)
    af['enmo_iqr'] = np.percentile(enmo_sec, 75) - np.percentile(enmo_sec, 25)
    af['enmo_median'] = np.median(enmo_sec)
    af['enmo_p95'] = np.percentile(enmo_sec, 95)

    hist, _ = np.histogram(enmo_sec, bins=20, density=True)
    hist = hist[hist > 0]; hist = hist / hist.sum()
    af['enmo_entropy'] = -np.sum(hist * np.log2(hist + 1e-12))

    af['pct_sedentary'] = np.mean(enmo_sec < sed_thresh)
    af['pct_lipa'] = np.mean((enmo_sec >= sed_thresh) & (enmo_sec < mvpa_thresh))
    af['pct_mvpa'] = np.mean(enmo_sec >= mvpa_thresh)
    af['mvpa_min_per_hour'] = (np.sum(enmo_sec >= mvpa_thresh) / 60) / (total_hours + 1e-12)

    # Bouts
    active = enmo_sec >= sed_thresh
    bout_durs = []
    in_b, bstart = False, 0
    for j in range(len(active)):
        if active[j] and not in_b: bstart = j; in_b = True
        elif not active[j] and in_b:
            if j - bstart >= 5: bout_durs.append(j - bstart)
            in_b = False
    if in_b and len(active) - bstart >= 5: bout_durs.append(len(active) - bstart)

    af['bouts_per_hour'] = len(bout_durs) / (total_hours + 1e-12)
    if bout_durs:
        af['bout_mean_dur'] = np.mean(bout_durs)
        af['bout_dur_cv'] = np.std(bout_durs) / (np.mean(bout_durs) + 1e-12)
    else:
        af['bout_mean_dur'] = 0; af['bout_dur_cv'] = 0

    # Transition probabilities
    transitions_as, transitions_sa = 0, 0
    active_count, sed_count = 0, 0
    for j in range(len(active) - 1):
        if active[j]:
            active_count += 1
            if not active[j + 1]: transitions_as += 1
        else:
            sed_count += 1
            if active[j + 1]: transitions_sa += 1
    af['astp'] = transitions_as / (active_count + 1e-12)
    af['satp'] = transitions_sa / (sed_count + 1e-12)

    return af


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

if __name__ == "__main__":
    ids = pd.read_csv(BASE / "feats" / "target_6mwd.csv")
    y_all = ids["sixmwd"].values.astype(float)

    print("=" * 70)
    print("HYBRID HOME PIPELINE")
    print("=" * 70)

    # ── Pass 1: Read all GT3X, compute global ENMO thresholds ──
    print("\nPass 1: Reading GT3X files, computing global ENMO thresholds...")
    all_daytime = {}  # {idx: (xyz, fs)}
    all_enmo_secs = []

    for i, (_, r) in enumerate(ids.iterrows()):
        cohort, subj_id, year = r["cohort"], int(r["subj_id"]), int(r["year"])
        label = f"{cohort}{subj_id:02d}"

        gt3x = find_gt3x_file(cohort, subj_id, year)
        if gt3x is None:
            print(f"  [{i+1:3d}] {label}: GT3X not found", flush=True)
            continue

        try:
            xyz, fs = read_gt3x_daytime(gt3x)
            if len(xyz) < 1800 * fs:  # less than 30 min daytime
                print(f"  [{i+1:3d}] {label}: too short ({len(xyz)/fs/3600:.1f}h)", flush=True)
                continue
            all_daytime[i] = (xyz, fs)

            # Per-second ENMO for global thresholds
            vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
            enmo = np.maximum(vm - 1.0, 0.0)
            sec = int(fs)
            n_secs = len(enmo) // sec
            enmo_sec = enmo[:n_secs * sec].reshape(n_secs, sec).mean(axis=1)
            all_enmo_secs.append(enmo_sec)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1:3d}] {label}: {len(xyz)/fs/3600:.1f}h OK", flush=True)
        except Exception as e:
            print(f"  [{i+1:3d}] {label}: ERROR — {e}", flush=True)

    print(f"  Loaded: {len(all_daytime)}/{len(ids)}")

    # Global activity thresholds
    pooled = np.concatenate(all_enmo_secs)
    sed_thresh = np.percentile(pooled, 50)
    mvpa_thresh = np.percentile(pooled, 85)
    print(f"  Global thresholds: sedentary < {sed_thresh:.4f}, MVPA > {mvpa_thresh:.4f}")

    # ── Pass 2: Walking detection, feature extraction ──
    print("\nPass 2: Walking detection + feature extraction...")

    gait_rows, cwt_rows, activity_rows = [], [], []
    valid_mask = []
    walk_stats = []

    for i, (_, r) in enumerate(ids.iterrows()):
        cohort, subj_id, year, sixmwd = r["cohort"], int(r["subj_id"]), int(r["year"]), int(r["sixmwd"])
        label = f"{cohort}{subj_id:02d}"

        if i not in all_daytime:
            valid_mask.append(False)
            continue

        xyz_day, fs = all_daytime[i]

        try:
            # Activity features from FULL daytime (like old pipeline)
            act_feats = extract_activity_features(xyz_day, fs, sed_thresh, mvpa_thresh)
            if act_feats is None:
                print(f"  [{i+1:3d}] {label}: too short for activity features — SKIP", flush=True)
                valid_mask.append(False)
                continue

            # Detect walking bouts (permissive ENMO)
            bouts = detect_active_bouts(xyz_day, fs, min_bout_sec=30)

            # Refine with HR (but keep original as fallback)
            if bouts:
                bouts = refine_with_hr(xyz_day, fs, bouts, hr_threshold=0.2)

            total_walk_sec = sum((e - s) / fs for s, e in bouts)
            n_bouts = len(bouts)

            # Load clinic data for similarity ranking
            clinic_xyz, clinic_fs = load_clinic_raw(cohort, subj_id, year, sixmwd)

            # Select walking segment
            if bouts:
                result = select_walking_segment(
                    xyz_day, fs, bouts, target_sec=360,
                    clinic_xyz=clinic_xyz, clinic_fs=clinic_fs
                )
                walking_seg, best_sim = result if result is not None else (None, 0.0)
            else:
                walking_seg, best_sim = None, 0.0

            # Gait + CWT features from walking segment
            if walking_seg is not None and len(walking_seg) >= 30 * fs:
                seg_duration = len(walking_seg) / fs
                preproc_df = preprocess_walking_segment(walking_seg, fs, target_fs=30.0)

                # Gait13
                gait10 = extract_gait10(preproc_df)
                vt_rms = float(np.sqrt(np.mean(np.square(preproc_df["VT"].values))))
                gait10["vt_rms_g"] = vt_rms
                ml_rms = gait10["ml_rms_g"]
                enmo_mean = gait10["enmo_mean_g"]
                gait10["ml_over_enmo"] = ml_rms / enmo_mean if enmo_mean > 0 else np.nan
                gait10["ml_over_vt"] = ml_rms / vt_rms if vt_rms > 0 else np.nan
                gait_rows.append(gait10)

                # CWT
                if fs != 30:
                    walking_30 = resample_uniform(walking_seg, fs, 30.0)
                else:
                    walking_30 = walking_seg
                cwt_rows.append(extract_cwt(walking_30, fs=30.0))

                has_gait = True
            else:
                # No walking found — fill gait/CWT with NaN
                gait_rows.append({k: np.nan for k in [
                    "cadence_hz", "step_time_cv_pct", "acf_step_regularity",
                    "hr_ap", "hr_vt", "ml_rms_g", "ml_spectral_entropy",
                    "jerk_mean_abs_gps", "enmo_mean_g", "cadence_slope_per_min",
                    "vt_rms_g", "ml_over_enmo", "ml_over_vt"]})
                cwt_rows.append({})  # will be filled with NaN below
                seg_duration = 0
                best_sim = 0
                has_gait = False

            activity_rows.append(act_feats)
            valid_mask.append(True)

            walk_stats.append({
                'subject': label, 'n_bouts': n_bouts,
                'total_walk_sec': total_walk_sec,
                'selected_sec': seg_duration,
                'best_sim': best_sim, 'has_gait': has_gait
            })

            if (i + 1) % 10 == 0:
                status = f"sim={best_sim:.3f}" if has_gait else "no-walk"
                print(f"  [{i+1:3d}] {label}: {seg_duration:.0f}s walk ({status}), "
                      f"activity OK", flush=True)

        except Exception as e:
            print(f"  [{i+1:3d}] {label}: ERROR — {e}", flush=True)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    n_valid = sum(valid_mask)
    print(f"\n  Valid: {n_valid}/{len(ids)}")

    if walk_stats:
        ws = pd.DataFrame(walk_stats)
        n_gait = ws['has_gait'].sum()
        print(f"  With walking segments: {n_gait}/{n_valid}")
        if n_gait > 0:
            ws_g = ws[ws['has_gait']]
            print(f"  Walking stats (n={n_gait}):")
            print(f"    Selected duration: mean={ws_g['selected_sec'].mean():.0f}s, "
                  f"min={ws_g['selected_sec'].min():.0f}s, max={ws_g['selected_sec'].max():.0f}s")
            print(f"    Clinic similarity: mean={ws_g['best_sim'].mean():.3f}")

    # ── Build feature matrices ──
    sway_cols = ["cadence_hz", "step_time_cv_pct", "acf_step_regularity", "hr_ap", "hr_vt",
                 "ml_rms_g", "ml_spectral_entropy", "jerk_mean_abs_gps", "enmo_mean_g",
                 "cadence_slope_per_min", "vt_rms_g", "ml_over_enmo", "ml_over_vt"]

    gait_df = pd.DataFrame(gait_rows)
    for c in sway_cols:
        if c not in gait_df.columns:
            gait_df[c] = np.nan
    X_gait = gait_df[sway_cols].values.astype(float)
    for j in range(X_gait.shape[1]):
        m = np.isnan(X_gait[:, j])
        if m.any(): X_gait[m, j] = np.nanmedian(X_gait[:, j])

    cwt_df = pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan)
    for c in cwt_df.columns:
        if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(cwt_df[c].median())
    # Fill completely NaN rows (subjects with no walking)
    for c in cwt_df.columns:
        if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(0)
    X_cwt = cwt_df.values.astype(float)

    act_df = pd.DataFrame(activity_rows).replace([np.inf, -np.inf], np.nan)
    for c in act_df.columns:
        if act_df[c].isna().any(): act_df[c] = act_df[c].fillna(act_df[c].median())
    X_act = act_df.values.astype(float)

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

    # ── Results ──
    print(f"\n{'='*70}")
    print(f"RESULTS — Hybrid Home Pipeline (n={len(y)})")
    print(f"{'='*70}")

    n_act = X_act.shape[1]
    n_gait = X_gait.shape[1]
    n_cwt = X_cwt.shape[1]
    n_demo = X_demo.shape[1]

    print(f"\n  A1 (no demographics):")
    print(f"    Activity only      ({n_act}f):  R² = {loo(X_act, y)}")
    print(f"    Gait13 only        ({n_gait}f):  R² = {loo(X_gait, y)}")
    print(f"    CWT only           ({n_cwt}f):  R² = {loo(X_cwt, y)}")
    print(f"    Activity+Gait13    ({n_act+n_gait}f):  R² = {loo(np.column_stack([X_act, X_gait]), y)}")
    print(f"    Activity+CWT       ({n_act+n_cwt}f):  R² = {loo(np.column_stack([X_act, X_cwt]), y)}")
    print(f"    Gait13+CWT         ({n_gait+n_cwt}f):  R² = {loo(np.column_stack([X_gait, X_cwt]), y)}")
    X_all_a1 = np.column_stack([X_act, X_gait, X_cwt])
    print(f"    All A1             ({X_all_a1.shape[1]}f):  R² = {loo(X_all_a1, y)}")

    print(f"\n  A2 (with demographics):")
    print(f"    Demo only          ({n_demo}f):  R² = {loo(X_demo, y)}")
    print(f"    Activity+Demo      ({n_act+n_demo}f):  R² = {loo(np.column_stack([X_act, X_demo]), y)}")
    print(f"    Gait13+Demo        ({n_gait+n_demo}f):  R² = {loo(np.column_stack([X_gait, X_demo]), y)}")
    print(f"    CWT+Demo           ({n_cwt+n_demo}f):  R² = {loo(np.column_stack([X_cwt, X_demo]), y)}")
    print(f"    Activity+Gait+Demo ({n_act+n_gait+n_demo}f):  R² = {loo(np.column_stack([X_act, X_gait, X_demo]), y)}")
    X_all_a2 = np.column_stack([X_act, X_gait, X_cwt, X_demo])
    print(f"    All A2             ({X_all_a2.shape[1]}f):  R² = {loo(X_all_a2, y)}")

    print(f"\n  Previous best: A1=0.146 (Activity), A2=0.431 (Activity+Demo XGB)")
    print(f"  C2 reference:  R² = 0.791")

    # Free memory
    del all_daytime
