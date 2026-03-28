#!/usr/bin/env python3
"""
Hybrid Home Pipeline v2 — with 6M95c features from Joy et al. (2026)
=====================================================================
NEW: Six-Minute Activity Centile features (6M25c, 6M50c, 6M75c, 6M95c)
  - Split daily wear into 6-min segments
  - Compute VM sum per segment
  - Take 25th, 50th, 75th, 95th centiles per day
  - Exclude first/last day
  - Average across days

This mirrors the 6MWT concept: peak 6-minute physical activity from free-living data.
"""

import numpy as np
import pandas as pd
import math
import warnings
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import linregress, spearmanr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import pywt

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent  # project root
CACHE_DIR = BASE / "csv_home_daytime"


# ══════════════════════════════════════════════════════════════════
# 6-MINUTE ACTIVITY CENTILE FEATURES (from Joy et al. 2026)
# ══════════════════════════════════════════════════════════════════

def compute_6min_centiles(xyz, fs):
    """
    Compute six-minute activity centiles following Joy et al.:
    1. Compute VM for entire recording
    2. Split into 6-minute (360s) non-overlapping segments
    3. Sum VM per segment (= total activity count per 6 min)
    4. Group segments by day (using sample index / samples_per_day)
    5. Per day: compute 25th, 50th, 75th, 95th centile of segment sums
    6. Exclude first and last day
    7. Average centiles across remaining days → 6M25c, 6M50c, 6M75c, 6M95c

    Also compute: VMs per minute, and centile ratios.
    """
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    n = len(vm)
    seg_len = int(6 * 60 * fs)  # 6 minutes in samples

    if n < seg_len * 3:  # need at least 3 segments
        return None

    # Split into 6-minute segments and sum VM
    n_segs = n // seg_len
    vm_sums = np.array([np.sum(vm[i*seg_len:(i+1)*seg_len]) for i in range(n_segs)])

    # Group by day: daytime is 15 hours (8am-11pm), so ~150 segments per day
    segs_per_day = int(15 * 60 * 60 * fs / seg_len)  # ~150 at 30Hz
    n_days = max(1, n_segs // segs_per_day + (1 if n_segs % segs_per_day > 0 else 0))

    # Compute centiles per day
    daily_centiles = {25: [], 50: [], 75: [], 95: []}
    daily_vms_per_min = []

    for d in range(n_days):
        start_seg = d * segs_per_day
        end_seg = min((d + 1) * segs_per_day, n_segs)
        day_sums = vm_sums[start_seg:end_seg]

        if len(day_sums) < 5:  # need enough segments for meaningful centiles
            continue

        for pct in [25, 50, 75, 95]:
            daily_centiles[pct].append(np.percentile(day_sums, pct))

        # VMs per minute for the day
        daily_vms_per_min.append(np.mean(day_sums) / 6.0)

    # Exclude first and last day if we have ≥ 3 days
    for pct in daily_centiles:
        vals = daily_centiles[pct]
        if len(vals) >= 3:
            daily_centiles[pct] = vals[1:-1]  # exclude first/last

    if len(daily_vms_per_min) >= 3:
        daily_vms_per_min = daily_vms_per_min[1:-1]

    if not daily_centiles[95]:
        return None

    f = {}
    # Core centile features (log-transformed as in paper)
    for pct in [25, 50, 75, 95]:
        vals = daily_centiles[pct]
        mean_val = np.mean(vals)
        f[f'6M{pct}c'] = mean_val
        f[f'6M{pct}c_log'] = np.log(mean_val + 1)
        if len(vals) >= 2:
            f[f'6M{pct}c_cv'] = np.std(vals) / (np.mean(vals) + 1e-12)
        else:
            f[f'6M{pct}c_cv'] = 0

    # VMs per minute
    f['vms_per_min'] = np.mean(daily_vms_per_min)
    f['vms_per_min_log'] = np.log(np.mean(daily_vms_per_min) + 1)

    # Centile ratios (shape of activity distribution)
    f['centile_ratio_95_50'] = f['6M95c'] / (f['6M50c'] + 1e-12)
    f['centile_ratio_75_25'] = f['6M75c'] / (f['6M25c'] + 1e-12)
    f['centile_iqr'] = f['6M75c'] - f['6M25c']
    f['centile_range'] = f['6M95c'] - f['6M25c']

    return f


# ══════════════════════════════════════════════════════════════════
# LOADING + WALKING DETECTION + PREPROCESSING (same as v1)
# ══════════════════════════════════════════════════════════════════

def load_cached_daytime(cohort, subj_id, year, sixmwd):
    fn = f"{cohort}{subj_id:02d}_{year}_{sixmwd}.csv"
    p = CACHE_DIR / fn
    if not p.exists(): return None, 30
    df = pd.read_csv(p)
    return df[["X", "Y", "Z"]].values.astype(np.float64), 30

def load_clinic_raw(cohort, subj_id, year, sixmwd):
    RAW = BASE / "csv_raw2"
    fn = f"{cohort}{subj_id:02d}_{year}_{sixmwd}.csv"
    p = RAW / fn
    if not p.exists(): return None, None
    df = pd.read_csv(p)
    diffs = np.diff(df["Timestamp"].values); diffs_pos = diffs[diffs > 0]
    fs = round(1.0 / np.median(diffs_pos)) if len(diffs_pos) > 0 else 30
    xyz = df[["X", "Y", "Z"]].values.astype(np.float64)
    n_trim = int(10 * fs)
    if 2 * n_trim < len(xyz): xyz = xyz[n_trim:len(xyz)-n_trim]
    return xyz, int(fs)

def detect_active_bouts(xyz, fs, min_bout_sec=30):
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs); n_secs = len(enmo) // sec
    if n_secs < min_bout_sec: return []
    enmo_sec = enmo[:n_secs*sec].reshape(n_secs, sec).mean(axis=1)
    active = enmo_sec >= 0.015
    bouts, in_b, bs = [], False, 0
    for s in range(n_secs):
        if active[s] and not in_b: bs=s; in_b=True
        elif not active[s] and in_b:
            if s-bs >= min_bout_sec: bouts.append((bs*sec, s*sec))
            in_b = False
    if in_b and n_secs-bs >= min_bout_sec: bouts.append((bs*sec, n_secs*sec))
    return bouts

def refine_with_hr(xyz, fs, bouts, hr_threshold=0.2):
    if not bouts: return []
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b, a, vm - vm.mean())
    win = int(10*fs); step = int(10*fs)
    fft_freqs = np.fft.rfftfreq(win, d=1.0/fs)
    band = (fft_freqs >= 0.8) & (fft_freqs <= 3.5)
    refined = []
    for bout_s, bout_e in bouts:
        walking_wins = []
        for wi in range(bout_s, bout_e - win, step):
            seg = vm_bp[wi:wi+win]
            X = np.fft.rfft(seg); mags = np.abs(X)
            if not np.any(band): continue
            cadence = fft_freqs[band][np.argmax(mags[band])]
            even, odd = 0.0, 0.0
            for k in range(1, 11):
                fk = k*cadence
                if fk >= fft_freqs[-1]: break
                idx = int(np.argmin(np.abs(fft_freqs - fk)))
                if k%2==0: even += mags[idx]
                else: odd += mags[idx]
            hr = even / (odd+1e-12) if odd > 0 else 0
            if hr >= hr_threshold: walking_wins.append((wi, wi+win))
        if walking_wins:
            cs, ce = walking_wins[0]
            for ws, we in walking_wins[1:]:
                if ws <= ce+step: ce = max(ce, we)
                else:
                    if ce-cs >= 30*fs: refined.append((cs, ce))
                    cs, ce = ws, we
            if ce-cs >= 30*fs: refined.append((cs, ce))
        else:
            refined.append((bout_s, bout_e))
    return refined

def compute_walking_signature(xyz_seg, fs):
    vm = np.sqrt(xyz_seg[:,0]**2 + xyz_seg[:,1]**2 + xyz_seg[:,2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    cadence, step_reg = 1.5, 0.0
    if len(vm) > 3 * max(len(b), len(a)):
        vm_bp = filtfilt(b, a, vm - np.mean(vm))
        nperseg = min(len(vm_bp), int(fs*4))
        if nperseg >= int(fs):
            freqs, pxx = welch(vm_bp, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
            bd = (freqs >= 0.5) & (freqs <= 3.5)
            if np.any(bd): cadence = freqs[bd][np.argmax(pxx[bd])]
        lag = max(1, min(int(round(fs/cadence)), len(vm_bp)-1))
        x = vm_bp - vm_bp.mean(); d = np.dot(x, x)
        step_reg = np.dot(x[:len(x)-lag], x[lag:]) / (d+1e-12) if d > 0 else 0
    return np.array([np.mean(enmo), np.std(enmo), cadence, step_reg, np.mean(enmo**2),
                     np.percentile(enmo, 25), np.percentile(enmo, 75)])

def select_walking_segment(xyz, fs, bouts, target_sec=360, clinic_xyz=None, clinic_fs=None):
    if not bouts: return None, 0.0
    bout_info = [(s, e, (e-s)/fs) for s, e in bouts]
    if clinic_xyz is not None and clinic_fs is not None and len(clinic_xyz) > 100:
        clinic_sig = compute_walking_signature(clinic_xyz, clinic_fs)
        scored = []
        for s, e, dur in bout_info:
            bout_sig = compute_walking_signature(xyz[s:e], fs)
            sim = np.dot(clinic_sig, bout_sig) / (np.linalg.norm(clinic_sig)*np.linalg.norm(bout_sig)+1e-12)
            scored.append((s, e, dur, sim))
        scored.sort(key=lambda x: x[3], reverse=True)
    else:
        scored = [(s, e, dur, 0.0) for s, e, dur in bout_info]
        scored.sort(key=lambda x: x[2], reverse=True)
    collected, total = [], 0; target_samples = int(target_sec*fs)
    for s, e, dur, sim in scored:
        take = min(e-s, target_samples-total)
        collected.append(xyz[s:s+take]); total += take
        if total >= target_samples: break
    if total < 30*fs: return None, 0.0
    return np.concatenate(collected, axis=0), scored[0][3]

def _rodrigues(axis, theta):
    ax = axis / (np.linalg.norm(axis)+1e-12)
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    I = np.eye(3); return I + math.sin(theta)*K + (1-math.cos(theta))*(K@K)

def preprocess_walking(walking_xyz, fs_orig, target_fs=30.0):
    if fs_orig != target_fs:
        n_src, d = walking_xyz.shape; dur = (n_src-1)/fs_orig
        n_dst = int(round(dur*target_fs))+1
        oldt = np.linspace(0, dur, n_src); newt = np.linspace(0, dur, n_dst)
        arr = np.column_stack([np.interp(newt, oldt, walking_xyz[:,j]) for j in range(d)])
    else: arr = walking_xyz.copy()
    fs = target_fs
    b, a = butter(4, 0.25, btype='lowpass', fs=fs)
    g_est = np.column_stack([filtfilt(b, a, arr[:,j]) for j in range(3)])
    arr_dyn = arr - g_est
    g_mean = g_est.mean(axis=0); zhat = np.array([0.,0.,1.])
    gvec = g_mean / (np.linalg.norm(g_mean)+1e-12)
    angle = math.acos(np.clip(float(zhat@gvec), -1, 1))
    if angle > 1e-4:
        axis = np.cross(gvec, zhat)
        if np.linalg.norm(axis) < 1e-8: axis = np.array([1.,0.,0.])
        arr_v = arr_dyn @ _rodrigues(axis, angle).T
    else: arr_v = arr_dyn.copy()
    XY = arr_v[:,:2]; C = np.cov(XY, rowvar=False)
    vals, vecs = np.linalg.eigh(C); ap_dir = vecs[:, np.argmax(vals)]
    theta = math.atan2(float(ap_dir[1]), float(ap_dir[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[c,-s,0],[s,c,0],[0,0,1.]]); apmlvt = arr_v @ Rz.T
    b, a = butter(4, [0.25, 2.5], btype='bandpass', fs=fs)
    apmlvt_bp = np.column_stack([filtfilt(b, a, apmlvt[:,j]) for j in range(3)])
    vm_raw = np.linalg.norm(arr, axis=1); enmo = np.maximum(vm_raw - 1.0, 0.0)
    return pd.DataFrame({
        "AP": apmlvt[:,0], "ML": apmlvt[:,1], "VT": apmlvt[:,2],
        "AP_bp": apmlvt_bp[:,0], "ML_bp": apmlvt_bp[:,1], "VT_bp": apmlvt_bp[:,2],
        "VM_dyn": np.linalg.norm(apmlvt, axis=1), "VM_raw": vm_raw, "ENMO": enmo, "fs": fs,
    })


# ══════════════════════════════════════════════════════════════════
# GAIT13 + ACTIVITY + CWT (same as v1, compact)
# ══════════════════════════════════════════════════════════════════

def _psd_peak(x, fs, fmin=0.5, fmax=3.5):
    if len(x) < int(fs): return float("nan")
    nperseg = int(max(fs*4, 256))
    freqs, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend="constant")
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band): return float("nan")
    return float(freqs[band][np.argmax(Pxx[band])])

def _acf(x, max_lag):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    if n <= 1: return np.zeros(max_lag+1)
    d = np.dot(x, x)
    return np.array([np.dot(x[:n-k], x[k:]) / (d if d > 0 else 1.0) for k in range(max_lag+1)])

def _hr(sig, fs, cad, n_harm=10):
    if not np.isfinite(cad) or cad <= 0: return float("nan")
    x = sig - sig.mean()
    if len(x) < 2: return float("nan")
    X = np.fft.rfft(x); freqs = np.fft.rfftfreq(len(x), d=1.0/fs); mags = np.abs(X)
    ev, od = 0.0, 0.0
    for k in range(1, n_harm+1):
        fk = k*cad
        if fk >= freqs[-1]: break
        idx = int(np.argmin(np.abs(freqs - fk)))
        if k%2==0: ev += mags[idx]
        else: od += mags[idx]
    return float(ev/od) if od > 0 else float("nan")

def extract_gait13(df):
    fs = float(df["fs"].iloc[0])
    vt_bp = df["VT_bp"].values; ap_bp = df["AP_bp"].values; ml_bp = df["ML_bp"].values
    ml = df["ML"].values; ap = df["AP"].values; vt = df["VT"].values; enmo = df["ENMO"].values
    cad = _psd_peak(vt_bp, fs); f = {"cadence_hz": cad}
    if np.isfinite(cad) and cad > 0:
        min_dist = max(1, int(round(0.5*fs/cad)))
        prom = 0.5*np.std(vt_bp) if np.std(vt_bp) > 0 else 0
        peaks, _ = find_peaks(vt_bp, distance=min_dist, prominence=prom)
        if peaks.size >= 3:
            si = np.diff(peaks)/fs
            f["step_time_cv_pct"] = 100*np.std(si, ddof=1)/np.mean(si) if np.mean(si)>0 else np.nan
        else: f["step_time_cv_pct"] = np.nan
        lag1 = int(np.clip(round(fs/cad), 1, 1e7))
        ac = _acf(vt_bp, lag1*3)
        f["acf_step_regularity"] = float(ac[lag1]) if lag1 < ac.size else np.nan
    else: f["step_time_cv_pct"] = np.nan; f["acf_step_regularity"] = np.nan
    f["hr_ap"] = _hr(ap_bp, fs, cad); f["hr_vt"] = _hr(vt_bp, fs, cad)
    f["ml_rms_g"] = float(np.sqrt(np.mean(ml**2)))
    if np.isfinite(cad) and cad > 0:
        lo = max(0.25, 0.5*cad); hi = min(3.5, 3.0*cad); nperseg = int(max(fs*4, 256))
        freqs, Pxx = welch(ml_bp, fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend="constant")
        band = (freqs >= lo) & (freqs <= hi)
        if np.any(band):
            p = Pxx[band]; s = p.sum()
            if s > 0: p = p/s; f["ml_spectral_entropy"] = -(p*np.log(p+1e-12)).sum()/np.log(len(p))
            else: f["ml_spectral_entropy"] = np.nan
        else: f["ml_spectral_entropy"] = np.nan
    else: f["ml_spectral_entropy"] = np.nan
    vm = np.linalg.norm(np.c_[ap, ml, vt], axis=1)
    f["jerk_mean_abs_gps"] = float(np.mean(np.abs(np.diff(vm)*fs)))
    f["enmo_mean_g"] = float(np.mean(enmo))
    per_min = int(round(60*fs)); m = min(6, max(1, len(vt_bp)//per_min))
    cads = [_psd_peak(vt_bp[j*per_min:(j+1)*per_min], fs) for j in range(m)
            if len(vt_bp[j*per_min:(j+1)*per_min]) >= per_min//2]
    cads = np.array([c for c in cads if np.isfinite(c)])
    f["cadence_slope_per_min"] = float(np.polyfit(np.arange(len(cads)), cads, 1)[0]) if len(cads) >= 3 else np.nan
    vt_rms = float(np.sqrt(np.mean(vt**2))); f["vt_rms_g"] = vt_rms
    f["ml_over_enmo"] = f["ml_rms_g"]/f["enmo_mean_g"] if f["enmo_mean_g"] > 0 else np.nan
    f["ml_over_vt"] = f["ml_rms_g"]/vt_rms if vt_rms > 0 else np.nan
    return f

def extract_activity(xyz, fs, sed_thresh, mvpa_thresh):
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs); n_bins = len(enmo)//sec
    if n_bins < 10: return None
    enmo_sec = enmo[:n_bins*sec].reshape(n_bins, sec).mean(axis=1)
    total_hours = n_bins/3600
    af = {}
    af['enmo_mean'] = np.mean(enmo_sec); af['enmo_std'] = np.std(enmo_sec)
    af['enmo_iqr'] = np.percentile(enmo_sec, 75) - np.percentile(enmo_sec, 25)
    af['enmo_median'] = np.median(enmo_sec); af['enmo_p95'] = np.percentile(enmo_sec, 95)
    hist, _ = np.histogram(enmo_sec, bins=20, density=True)
    hist = hist[hist>0]; hist = hist/hist.sum()
    af['enmo_entropy'] = -np.sum(hist * np.log2(hist+1e-12))
    af['pct_sedentary'] = np.mean(enmo_sec < sed_thresh)
    af['pct_lipa'] = np.mean((enmo_sec >= sed_thresh) & (enmo_sec < mvpa_thresh))
    af['pct_mvpa'] = np.mean(enmo_sec >= mvpa_thresh)
    af['mvpa_min_per_hour'] = (np.sum(enmo_sec >= mvpa_thresh)/60) / (total_hours+1e-12)
    active = enmo_sec >= sed_thresh; bds = []; in_b, bs = False, 0
    for j in range(len(active)):
        if active[j] and not in_b: bs=j; in_b=True
        elif not active[j] and in_b:
            if j-bs >= 5: bds.append(j-bs)
            in_b = False
    if in_b and len(active)-bs >= 5: bds.append(len(active)-bs)
    af['bouts_per_hour'] = len(bds)/(total_hours+1e-12)
    af['bout_mean_dur'] = np.mean(bds) if bds else 0
    af['bout_dur_cv'] = np.std(bds)/(np.mean(bds)+1e-12) if bds else 0
    tas, tsa, ac2, sc2 = 0, 0, 0, 0
    for j in range(len(active)-1):
        if active[j]: ac2 += 1; tas += (not active[j+1])
        else: sc2 += 1; tsa += active[j+1]
    af['astp'] = tas/(ac2+1e-12); af['satp'] = tsa/(sc2+1e-12)
    return af


# ══════════════════════════════════════════════════════════════════
# MULTI-MODEL LOO
# ══════════════════════════════════════════════════════════════════

def loo_multi(X, y):
    models = {
        'Ridge': lambda: Ridge(alpha=10),
        'ElasNet': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        'RF': lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42),
        'XGB': lambda: __import__('xgboost').XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1, reg_lambda=5, random_state=42, verbosity=0),
        'SVR': lambda: SVR(kernel='rbf', C=10, epsilon=0.1),
    }
    results = {}
    for name, model_fn in models.items():
        pred = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr]); X_te = sc.transform(X[te])
            m = model_fn(); m.fit(X_tr, y[tr])
            pred[te] = m.predict(X_te)
        results[name] = round(r2_score(y, pred), 4)
    return results

def select_features_by_corr(X, y, feature_names, top_k=10):
    corrs = []
    for j in range(X.shape[1]):
        mask = ~np.isnan(X[:,j]) & ~np.isnan(y)
        if mask.sum() > 10:
            rho, _ = spearmanr(X[mask,j], y[mask]); corrs.append(abs(rho))
        else: corrs.append(0)
    idx = np.argsort(corrs)[::-1][:top_k]
    return X[:, idx], [feature_names[i] for i in idx], idx


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ids = pd.read_csv(BASE / "feats" / "target_6mwd.csv")
    y_all = ids["sixmwd"].values.astype(float)

    print("=" * 70)
    print("HYBRID HOME v2 — with 6-Minute Activity Centiles")
    print("=" * 70)

    # Pass 1: Load + thresholds
    print("\nPass 1: Loading cached data...")
    all_data = {}; all_enmo = []
    for i, (_, r) in enumerate(ids.iterrows()):
        cohort, subj_id, year, sixmwd = r["cohort"], int(r["subj_id"]), int(r["year"]), int(r["sixmwd"])
        xyz, fs = load_cached_daytime(cohort, subj_id, year, sixmwd)
        if xyz is None or len(xyz) < 1800*fs: continue
        all_data[i] = (xyz, fs)
        vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
        enmo = np.maximum(vm - 1.0, 0.0)
        sec = int(fs); n_s = len(enmo)//sec
        all_enmo.append(enmo[:n_s*sec].reshape(n_s, sec).mean(axis=1))
    pooled = np.concatenate(all_enmo)
    sed_thresh = np.percentile(pooled, 50); mvpa_thresh = np.percentile(pooled, 85)
    print(f"  Loaded: {len(all_data)}/{len(ids)}")

    # Pass 2: Extract all features
    print("\nPass 2: Extracting features...")
    gait_rows, act_rows, centile_rows = [], [], []
    valid_mask = []
    sway_cols = ["cadence_hz","step_time_cv_pct","acf_step_regularity","hr_ap","hr_vt",
                 "ml_rms_g","ml_spectral_entropy","jerk_mean_abs_gps","enmo_mean_g",
                 "cadence_slope_per_min","vt_rms_g","ml_over_enmo","ml_over_vt"]

    for i, (_, r) in enumerate(ids.iterrows()):
        if i not in all_data: valid_mask.append(False); continue
        cohort, subj_id, year, sixmwd = r["cohort"], int(r["subj_id"]), int(r["year"]), int(r["sixmwd"])
        xyz_day, fs = all_data[i]
        try:
            # Activity profile
            act = extract_activity(xyz_day, fs, sed_thresh, mvpa_thresh)
            if act is None: valid_mask.append(False); continue

            # 6-minute centiles (NEW)
            centiles = compute_6min_centiles(xyz_day, fs)
            if centiles is None: valid_mask.append(False); continue

            # Walking detection + gait
            bouts = detect_active_bouts(xyz_day, fs, min_bout_sec=30)
            if bouts: bouts = refine_with_hr(xyz_day, fs, bouts, hr_threshold=0.2)
            clinic_xyz, clinic_fs = load_clinic_raw(cohort, subj_id, year, sixmwd)
            walking_seg, sim = (None, 0.0)
            if bouts:
                result = select_walking_segment(xyz_day, fs, bouts, 360, clinic_xyz, clinic_fs)
                if result[0] is not None: walking_seg, sim = result

            if walking_seg is not None and len(walking_seg) >= 30*fs:
                preproc = preprocess_walking(walking_seg, fs)
                gait_rows.append(extract_gait13(preproc))
            else:
                gait_rows.append({k: np.nan for k in sway_cols})

            act_rows.append(act)
            centile_rows.append(centiles)
            valid_mask.append(True)
            if (i+1) % 20 == 0: print(f"  [{i+1:3d}] done", flush=True)
        except Exception as e:
            print(f"  [{i+1:3d}] ERROR: {e}", flush=True)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    n_valid = sum(valid_mask)
    y = y_all[valid_mask]
    print(f"  Valid: {n_valid}/{len(ids)}")

    # Build feature matrices
    gait_df = pd.DataFrame(gait_rows)
    for c in sway_cols:
        if c not in gait_df.columns: gait_df[c] = np.nan
    X_gait = gait_df[sway_cols].values.astype(float)
    for j in range(X_gait.shape[1]):
        m = np.isnan(X_gait[:,j])
        if m.any(): X_gait[m,j] = np.nanmedian(X_gait[:,j])

    act_df = pd.DataFrame(act_rows).replace([np.inf, -np.inf], np.nan)
    for c in act_df.columns:
        if act_df[c].isna().any(): act_df[c] = act_df[c].fillna(act_df[c].median())
    X_act = act_df.values.astype(float)

    cent_df = pd.DataFrame(centile_rows).replace([np.inf, -np.inf], np.nan)
    for c in cent_df.columns:
        if cent_df[c].isna().any(): cent_df[c] = cent_df[c].fillna(cent_df[c].median())
    X_cent = cent_df.values.astype(float)

    # Demographics
    demo = pd.read_excel(BASE / "SwayDemographics.xlsx")
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    p = ids[valid_mask].reset_index(drop=True).merge(demo, on=["cohort","subj_id"], how="left")
    p["cohort_M"] = (p["cohort"]=="M").astype(int)
    for c in ["Age","Sex","Height"]: p[c] = pd.to_numeric(p[c], errors="coerce")
    X_demo = p[["cohort_M","Age","Sex","Height"]].values.astype(float)
    for j in range(X_demo.shape[1]):
        m = np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j] = np.nanmedian(X_demo[:,j])

    # Show 6M95c correlations
    print(f"\n  6-Minute Centile correlations with 6MWD:")
    for c in cent_df.columns:
        vals = cent_df[c].values.astype(float)
        mask = ~np.isnan(vals) & ~np.isnan(y)
        if mask.sum() > 10:
            rho, pval = spearmanr(vals[mask], y[mask])
            sig = '*' if pval < 0.05 else ' '
            print(f"    {c:25s}: rho={rho:+.4f}{sig}")

    # ── Results ──
    print(f"\n{'='*70}")
    print(f"A1 RESULTS — No Demographics (n={len(y)})")
    print(f"{'='*70}")
    hdr = f"{'Feature Set':30s} {'Ridge':>8s} {'ElasNet':>8s} {'RF':>8s} {'XGB':>8s} {'SVR':>8s}"
    print(hdr); print("-"*len(hdr))

    a1_sets = {
        'Activity': X_act,
        'Gait13': X_gait,
        '6MinCentiles': X_cent,
        'Activity+Gait13': np.column_stack([X_act, X_gait]),
        'Activity+6MC': np.column_stack([X_act, X_cent]),
        'Gait13+6MC': np.column_stack([X_gait, X_cent]),
        'Act+Gait+6MC': np.column_stack([X_act, X_gait, X_cent]),
    }

    best_a1, best_a1_name = -999, ""
    for name, X in a1_sets.items():
        results = loo_multi(X, y)
        row = f"{name:30s}"
        for mn in ['Ridge','ElasNet','RF','XGB','SVR']:
            r2 = results[mn]; row += f" {r2:8.4f}"
            if r2 > best_a1: best_a1 = r2; best_a1_name = f"{name}/{mn}"
        print(row)

    # Feature selected
    all_a1 = np.column_stack([X_act, X_gait, X_cent])
    all_a1_names = list(act_df.columns) + sway_cols + list(cent_df.columns)
    for k in [8, 10, 15]:
        X_sel, sel_names, _ = select_features_by_corr(all_a1, y, all_a1_names, top_k=k)
        results = loo_multi(X_sel, y)
        row = f"All_A1(top{k}):".ljust(30)
        for mn in ['Ridge','ElasNet','RF','XGB','SVR']:
            r2 = results[mn]; row += f" {r2:8.4f}"
            if r2 > best_a1: best_a1 = r2; best_a1_name = f"All_A1(top{k})/{mn}"
        print(row)
        if k == 10: print(f"    Selected: {sel_names}")

    print(f"\n  Best A1: {best_a1:.4f} ({best_a1_name})")
    print(f"  Previous best A1: 0.146 → v1 best: 0.288")

    # A2
    print(f"\n{'='*70}")
    print(f"A2 RESULTS — With Demographics (n={len(y)})")
    print(f"{'='*70}")
    print(hdr); print("-"*len(hdr))

    a2_sets = {
        'Demo': X_demo,
        'Activity+Demo': np.column_stack([X_act, X_demo]),
        'Gait13+Demo': np.column_stack([X_gait, X_demo]),
        '6MC+Demo': np.column_stack([X_cent, X_demo]),
        'Act+Gait+Demo': np.column_stack([X_act, X_gait, X_demo]),
        'Act+6MC+Demo': np.column_stack([X_act, X_cent, X_demo]),
        'Gait+6MC+Demo': np.column_stack([X_gait, X_cent, X_demo]),
        'Act+Gait+6MC+Demo': np.column_stack([X_act, X_gait, X_cent, X_demo]),
    }

    best_a2, best_a2_name = -999, ""
    for name, X in a2_sets.items():
        results = loo_multi(X, y)
        row = f"{name:30s}"
        for mn in ['Ridge','ElasNet','RF','XGB','SVR']:
            r2 = results[mn]; row += f" {r2:8.4f}"
            if r2 > best_a2: best_a2 = r2; best_a2_name = f"{name}/{mn}"
        print(row)

    # Feature selected A2
    all_a2 = np.column_stack([X_act, X_gait, X_cent, X_demo])
    all_a2_names = list(act_df.columns) + sway_cols + list(cent_df.columns) + ["cohort_M","Age","Sex","Height"]
    for k in [10, 15, 20]:
        X_sel, sel_names, _ = select_features_by_corr(all_a2, y, all_a2_names, top_k=k)
        results = loo_multi(X_sel, y)
        row = f"All(top{k}):".ljust(30)
        for mn in ['Ridge','ElasNet','RF','XGB','SVR']:
            r2 = results[mn]; row += f" {r2:8.4f}"
            if r2 > best_a2: best_a2 = r2; best_a2_name = f"All(top{k})/{mn}"
        print(row)
        if k == 10: print(f"    Selected: {sel_names}")

    print(f"\n  Best A2: {best_a2:.4f} ({best_a2_name})")
    print(f"  Previous best A2: 0.431")
