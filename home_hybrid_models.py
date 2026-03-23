#!/usr/bin/env python3
"""
Hybrid Home Pipeline — Multi-model evaluation
Try to surpass A1=0.146, A2=0.431 with Ridge, ElasticNet, RF, XGBoost, SVR
Also try feature selection and combined feature sets.
Uses cached CSV from csv_home_daytime/ for speed.
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

BASE = Path(__file__).parent
CACHE_DIR = BASE / "csv_home_daytime"

# ══════════════════════════════════════════════════════════════════
# DATA LOADING (from cache)
# ══════════════════════════════════════════════════════════════════

def load_cached_daytime(cohort, subj_id, year, sixmwd):
    fn = f"{cohort}{subj_id:02d}_{year}_{sixmwd}.csv"
    p = CACHE_DIR / fn
    if not p.exists():
        return None, 30
    df = pd.read_csv(p)
    return df[["X", "Y", "Z"]].values.astype(np.float64), 30

def load_clinic_raw(cohort, subj_id, year, sixmwd):
    RAW = BASE / "csv_raw2"
    fn = f"{cohort}{subj_id:02d}_{year}_{sixmwd}.csv"
    p = RAW / fn
    if not p.exists(): return None, None
    df = pd.read_csv(p)
    diffs = np.diff(df["Timestamp"].values)
    diffs_pos = diffs[diffs > 0]
    fs = round(1.0 / np.median(diffs_pos)) if len(diffs_pos) > 0 else 30
    xyz = df[["X", "Y", "Z"]].values.astype(np.float64)
    n_trim = int(10 * fs)
    if 2 * n_trim < len(xyz):
        xyz = xyz[n_trim:len(xyz) - n_trim]
    return xyz, int(fs)


# ══════════════════════════════════════════════════════════════════
# WALKING DETECTION (permissive ENMO + HR refinement)
# ══════════════════════════════════════════════════════════════════

def detect_active_bouts(xyz, fs, min_bout_sec=30):
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs)
    n_secs = len(enmo) // sec
    if n_secs < min_bout_sec: return []
    enmo_sec = enmo[:n_secs * sec].reshape(n_secs, sec).mean(axis=1)
    active = enmo_sec >= 0.015
    bouts = []
    in_b, bs = False, 0
    for s in range(n_secs):
        if active[s] and not in_b: bs = s; in_b = True
        elif not active[s] and in_b:
            if s - bs >= min_bout_sec: bouts.append((bs * sec, s * sec))
            in_b = False
    if in_b and n_secs - bs >= min_bout_sec: bouts.append((bs * sec, n_secs * sec))
    return bouts

def refine_with_hr(xyz, fs, bouts, hr_threshold=0.2):
    if not bouts: return []
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b, a, vm - vm.mean())
    win = int(10 * fs); step = int(10 * fs)
    fft_freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band = (fft_freqs >= 0.8) & (fft_freqs <= 3.5)
    refined = []
    for bout_s, bout_e in bouts:
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
            if hr >= hr_threshold: walking_wins.append((wi, wi + win))
        if walking_wins:
            cs, ce = walking_wins[0]
            for ws, we in walking_wins[1:]:
                if ws <= ce + step: ce = max(ce, we)
                else:
                    if ce - cs >= 30 * fs: refined.append((cs, ce))
                    cs, ce = ws, we
            if ce - cs >= 30 * fs: refined.append((cs, ce))
        else:
            refined.append((bout_s, bout_e))
    return refined

def compute_walking_signature(xyz_seg, fs):
    vm = np.sqrt(xyz_seg[:, 0]**2 + xyz_seg[:, 1]**2 + xyz_seg[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    cadence, step_reg = 1.5, 0.0
    if len(vm) > 3 * max(len(b), len(a)):
        vm_bp = filtfilt(b, a, vm - np.mean(vm))
        nperseg = min(len(vm_bp), int(fs * 4))
        if nperseg >= int(fs):
            freqs, pxx = welch(vm_bp, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            bd = (freqs >= 0.5) & (freqs <= 3.5)
            if np.any(bd): cadence = freqs[bd][np.argmax(pxx[bd])]
        lag = max(1, min(int(round(fs / cadence)), len(vm_bp) - 1))
        x = vm_bp - vm_bp.mean(); d = np.dot(x, x)
        step_reg = np.dot(x[:len(x)-lag], x[lag:]) / (d + 1e-12) if d > 0 else 0
    return np.array([np.mean(enmo), np.std(enmo), cadence, step_reg,
                     np.mean(enmo**2), np.percentile(enmo, 25), np.percentile(enmo, 75)])

def select_walking_segment(xyz, fs, bouts, target_sec=360, clinic_xyz=None, clinic_fs=None):
    if not bouts: return None, 0.0
    bout_info = [(s, e, (e-s)/fs) for s, e in bouts]
    if clinic_xyz is not None and clinic_fs is not None and len(clinic_xyz) > 100:
        clinic_sig = compute_walking_signature(clinic_xyz, clinic_fs)
        scored = []
        for s, e, dur in bout_info:
            bout_sig = compute_walking_signature(xyz[s:e], fs)
            sim = np.dot(clinic_sig, bout_sig) / (np.linalg.norm(clinic_sig) * np.linalg.norm(bout_sig) + 1e-12)
            scored.append((s, e, dur, sim))
        scored.sort(key=lambda x: x[3], reverse=True)
    else:
        scored = [(s, e, dur, 0.0) for s, e, dur in bout_info]
        scored.sort(key=lambda x: x[2], reverse=True)
    collected, total = [], 0
    target_samples = int(target_sec * fs)
    for s, e, dur, sim in scored:
        take = min(e - s, target_samples - total)
        collected.append(xyz[s:s+take]); total += take
        if total >= target_samples: break
    if total < 30 * fs: return None, 0.0
    return np.concatenate(collected, axis=0), scored[0][3]


# ══════════════════════════════════════════════════════════════════
# C2 PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def _rodrigues(axis, theta):
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    I = np.eye(3); return I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

def preprocess_walking(walking_xyz, fs_orig, target_fs=30.0):
    from scipy.signal import butter, filtfilt
    if fs_orig != target_fs:
        n_src, d = walking_xyz.shape; dur = (n_src-1)/fs_orig
        n_dst = int(round(dur*target_fs))+1
        oldt = np.linspace(0, dur, n_src); newt = np.linspace(0, dur, n_dst)
        arr = np.column_stack([np.interp(newt, oldt, walking_xyz[:,j]) for j in range(d)])
    else:
        arr = walking_xyz.copy()
    fs = target_fs
    b, a = butter(4, 0.25, btype='lowpass', fs=fs)
    g_est = np.column_stack([filtfilt(b, a, arr[:,j]) for j in range(3)])
    arr_dyn = arr - g_est
    g_mean = g_est.mean(axis=0); zhat = np.array([0.,0.,1.])
    gvec = g_mean / (np.linalg.norm(g_mean)+1e-12)
    angle = math.acos(np.clip(float(zhat @ gvec), -1, 1))
    if angle > 1e-4:
        axis = np.cross(gvec, zhat)
        if np.linalg.norm(axis) < 1e-8: axis = np.array([1.,0.,0.])
        arr_v = arr_dyn @ _rodrigues(axis, angle).T
    else: arr_v = arr_dyn.copy()
    XY = arr_v[:,:2]; C = np.cov(XY, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    ap_dir = vecs[:, np.argmax(vals)]
    theta = math.atan2(float(ap_dir[1]), float(ap_dir[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    apmlvt = arr_v @ Rz.T
    b, a = butter(4, [0.25, 2.5], btype='bandpass', fs=fs)
    apmlvt_bp = np.column_stack([filtfilt(b, a, apmlvt[:,j]) for j in range(3)])
    vm_raw = np.linalg.norm(arr, axis=1)
    enmo = np.maximum(vm_raw - 1.0, 0.0)
    return pd.DataFrame({
        "AP": apmlvt[:,0], "ML": apmlvt[:,1], "VT": apmlvt[:,2],
        "AP_bp": apmlvt_bp[:,0], "ML_bp": apmlvt_bp[:,1], "VT_bp": apmlvt_bp[:,2],
        "VM_dyn": np.linalg.norm(apmlvt, axis=1), "VM_raw": vm_raw, "ENMO": enmo, "fs": fs,
    })


# ══════════════════════════════════════════════════════════════════
# GAIT13 FEATURES
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
    ml = df["ML"].values; ap = df["AP"].values; vt = df["VT"].values
    enmo = df["ENMO"].values

    cad = _psd_peak(vt_bp, fs)
    f = {"cadence_hz": cad}
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
    else:
        f["step_time_cv_pct"] = np.nan; f["acf_step_regularity"] = np.nan

    f["hr_ap"] = _hr(ap_bp, fs, cad); f["hr_vt"] = _hr(vt_bp, fs, cad)
    f["ml_rms_g"] = float(np.sqrt(np.mean(ml**2)))

    if np.isfinite(cad) and cad > 0:
        lo = max(0.25, 0.5*cad); hi = min(3.5, 3.0*cad)
        nperseg = int(max(fs*4, 256))
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

    # Sway ratios
    vt_rms = float(np.sqrt(np.mean(vt**2)))
    f["vt_rms_g"] = vt_rms
    f["ml_over_enmo"] = f["ml_rms_g"] / f["enmo_mean_g"] if f["enmo_mean_g"] > 0 else np.nan
    f["ml_over_vt"] = f["ml_rms_g"] / vt_rms if vt_rms > 0 else np.nan
    return f


# ══════════════════════════════════════════════════════════════════
# CWT FEATURES
# ══════════════════════════════════════════════════════════════════

def extract_cwt(raw_sig, fs=30.0, n_seg=6):
    vm = np.sqrt(raw_sig[:,0]**2 + raw_sig[:,1]**2 + raw_sig[:,2]**2)
    vm_c = vm - vm.mean()
    freqs = np.linspace(0.5, 12, 50)
    def cwt_seg(sr):
        s = sr / (np.max(np.abs(sr)) + 1e-12)
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
        f['freq_variability'] = np.std(dl); f['freq_cv'] = np.std(dl)/(np.mean(dl)+1e-12)
        pn = mp/(mp.sum()+1e-12); pnz = pn[pn>0]
        f['wavelet_entropy'] = -np.sum(pnz*np.log2(pnz+1e-12))
        from scipy.fft import rfft, rfftfreq
        fv = np.abs(rfft(s)); ff = rfftfreq(len(s), 1/fs); gb = (ff>=0.5)&(ff<=3.5)
        if gb.any():
            f0 = ff[gb][np.argmax(fv[gb])]; f['fundamental_freq'] = f0
            if f0 > 0:
                ep, op = 0, 0
                for h in range(1, 11):
                    idx = np.argmin(np.abs(ff-h*f0))
                    if h%2==0: ep += fv[idx]**2
                    else: op += fv[idx]**2
                f['harmonic_ratio'] = ep/(op+1e-12)
            else: f['harmonic_ratio'] = 0
        else: f['fundamental_freq'] = 0; f['harmonic_ratio'] = 0
        return f
    T = len(vm_c); sl = T//n_seg; sfs = []
    for i in range(n_seg):
        s, e = i*sl, min((i+1)*sl, T)
        if e-s < int(2*fs): continue
        sfs.append(cwt_seg(vm_c[s:e]))
    if not sfs: sfs = [cwt_seg(vm_c)]
    df = pd.DataFrame(sfs)
    f = {f"cwt_{k}_mean": df[k].mean() for k in df.columns}
    f.update({f"cwt_{k}_std": df[k].std() for k in df.columns})
    for key in ["mean_energy","high_freq_energy","freq_variability","wavelet_entropy"]:
        if key in df.columns and len(df)>=3:
            sl2,_,rv,_,_ = linregress(np.arange(len(df)), df[key].values)
            f[f"cwt_{key}_slope"] = sl2; f[f"cwt_{key}_slope_r"] = rv
        else: f[f"cwt_{key}_slope"] = 0; f[f"cwt_{key}_slope_r"] = 0
    return f


# ══════════════════════════════════════════════════════════════════
# ACTIVITY PROFILE FEATURES
# ══════════════════════════════════════════════════════════════════

def extract_activity(xyz, fs, sed_thresh, mvpa_thresh):
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs); n_bins = len(enmo) // sec
    if n_bins < 10: return None
    enmo_sec = enmo[:n_bins*sec].reshape(n_bins, sec).mean(axis=1)
    total_hours = n_bins / 3600
    af = {}
    af['enmo_mean'] = np.mean(enmo_sec)
    af['enmo_std'] = np.std(enmo_sec)
    af['enmo_iqr'] = np.percentile(enmo_sec, 75) - np.percentile(enmo_sec, 25)
    af['enmo_median'] = np.median(enmo_sec)
    af['enmo_p95'] = np.percentile(enmo_sec, 95)
    hist, _ = np.histogram(enmo_sec, bins=20, density=True)
    hist = hist[hist>0]; hist = hist/hist.sum()
    af['enmo_entropy'] = -np.sum(hist * np.log2(hist+1e-12))
    af['pct_sedentary'] = np.mean(enmo_sec < sed_thresh)
    af['pct_lipa'] = np.mean((enmo_sec >= sed_thresh) & (enmo_sec < mvpa_thresh))
    af['pct_mvpa'] = np.mean(enmo_sec >= mvpa_thresh)
    af['mvpa_min_per_hour'] = (np.sum(enmo_sec >= mvpa_thresh)/60) / (total_hours+1e-12)
    active = enmo_sec >= sed_thresh
    bds = []; in_b, bs = False, 0
    for j in range(len(active)):
        if active[j] and not in_b: bs=j; in_b=True
        elif not active[j] and in_b:
            if j-bs >= 5: bds.append(j-bs)
            in_b = False
    if in_b and len(active)-bs >= 5: bds.append(len(active)-bs)
    af['bouts_per_hour'] = len(bds) / (total_hours+1e-12)
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

def loo_multi(X, y, models=None):
    if models is None:
        models = {
            'Ridge': lambda: Ridge(alpha=10),
            'ElasticNet': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            'RF': lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42),
            'XGBoost': lambda: __import__('xgboost').XGBRegressor(
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


# ══════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════

def select_features_by_corr(X, y, feature_names, top_k=10):
    """Select top_k features by absolute Spearman correlation with y."""
    corrs = []
    for j in range(X.shape[1]):
        mask = ~np.isnan(X[:, j]) & ~np.isnan(y)
        if mask.sum() > 10:
            rho, _ = spearmanr(X[mask, j], y[mask])
            corrs.append(abs(rho))
        else:
            corrs.append(0)
    idx = np.argsort(corrs)[::-1][:top_k]
    return X[:, idx], [feature_names[i] for i in idx], idx


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ids = pd.read_csv(BASE / "feats" / "target_6mwd.csv")
    y_all = ids["sixmwd"].values.astype(float)

    print("=" * 70)
    print("HYBRID HOME — MULTI-MODEL EVALUATION")
    print("=" * 70)

    # ── Pass 1: Load cached data + global thresholds ──
    print("\nPass 1: Loading cached daytime data...")
    all_data = {}
    all_enmo = []

    for i, (_, r) in enumerate(ids.iterrows()):
        cohort, subj_id, year, sixmwd = r["cohort"], int(r["subj_id"]), int(r["year"]), int(r["sixmwd"])
        xyz, fs = load_cached_daytime(cohort, subj_id, year, sixmwd)
        if xyz is None or len(xyz) < 1800 * fs:
            continue
        all_data[i] = (xyz, fs)
        vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
        enmo = np.maximum(vm - 1.0, 0.0)
        sec = int(fs); n_s = len(enmo) // sec
        all_enmo.append(enmo[:n_s*sec].reshape(n_s, sec).mean(axis=1))

    pooled = np.concatenate(all_enmo)
    sed_thresh = np.percentile(pooled, 50)
    mvpa_thresh = np.percentile(pooled, 85)
    print(f"  Loaded: {len(all_data)}/{len(ids)}")
    print(f"  Thresholds: sed < {sed_thresh:.4f}, MVPA > {mvpa_thresh:.4f}")

    # ── Pass 2: Extract all features ──
    print("\nPass 2: Extracting features...")
    gait_rows, cwt_rows, act_rows = [], [], []
    valid_mask = []

    for i, (_, r) in enumerate(ids.iterrows()):
        if i not in all_data:
            valid_mask.append(False); continue
        cohort, subj_id, year, sixmwd = r["cohort"], int(r["subj_id"]), int(r["year"]), int(r["sixmwd"])
        xyz_day, fs = all_data[i]

        try:
            # Activity features from full daytime
            act = extract_activity(xyz_day, fs, sed_thresh, mvpa_thresh)
            if act is None: valid_mask.append(False); continue

            # Walking detection
            bouts = detect_active_bouts(xyz_day, fs, min_bout_sec=30)
            if bouts: bouts = refine_with_hr(xyz_day, fs, bouts, hr_threshold=0.2)

            # Clinic-similarity ranked selection
            clinic_xyz, clinic_fs = load_clinic_raw(cohort, subj_id, year, sixmwd)
            if bouts:
                walking_seg, sim = select_walking_segment(xyz_day, fs, bouts, 360, clinic_xyz, clinic_fs)
            else:
                walking_seg, sim = None, 0.0

            if walking_seg is not None and len(walking_seg) >= 30 * fs:
                preproc = preprocess_walking(walking_seg, fs)
                gait_rows.append(extract_gait13(preproc))
                cwt_rows.append(extract_cwt(walking_seg, fs=30.0))
            else:
                gait_rows.append({k: np.nan for k in [
                    "cadence_hz","step_time_cv_pct","acf_step_regularity","hr_ap","hr_vt",
                    "ml_rms_g","ml_spectral_entropy","jerk_mean_abs_gps","enmo_mean_g",
                    "cadence_slope_per_min","vt_rms_g","ml_over_enmo","ml_over_vt"]})
                cwt_rows.append({})

            act_rows.append(act)
            valid_mask.append(True)
            if (i+1) % 20 == 0: print(f"  [{i+1:3d}] done", flush=True)
        except Exception as e:
            print(f"  [{i+1:3d}] ERROR: {e}", flush=True)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    n_valid = sum(valid_mask)
    print(f"  Valid: {n_valid}/{len(ids)}")

    # ── Build matrices ──
    sway_cols = ["cadence_hz","step_time_cv_pct","acf_step_regularity","hr_ap","hr_vt",
                 "ml_rms_g","ml_spectral_entropy","jerk_mean_abs_gps","enmo_mean_g",
                 "cadence_slope_per_min","vt_rms_g","ml_over_enmo","ml_over_vt"]

    gait_df = pd.DataFrame(gait_rows)
    for c in sway_cols:
        if c not in gait_df.columns: gait_df[c] = np.nan
    X_gait = gait_df[sway_cols].values.astype(float)
    for j in range(X_gait.shape[1]):
        m = np.isnan(X_gait[:,j])
        if m.any(): X_gait[m,j] = np.nanmedian(X_gait[:,j])

    cwt_df = pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan)
    for c in cwt_df.columns:
        if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(cwt_df[c].median())
    for c in cwt_df.columns:
        if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(0)
    X_cwt = cwt_df.values.astype(float)

    act_df = pd.DataFrame(act_rows).replace([np.inf, -np.inf], np.nan)
    for c in act_df.columns:
        if act_df[c].isna().any(): act_df[c] = act_df[c].fillna(act_df[c].median())
    X_act = act_df.values.astype(float)

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

    y = y_all[valid_mask]

    # ── Feature combinations ──
    feature_sets = {
        'Activity': (X_act, list(act_df.columns)),
        'Gait13': (X_gait, sway_cols),
        'CWT': (X_cwt, list(cwt_df.columns)),
        'Activity+Gait13': (np.column_stack([X_act, X_gait]), list(act_df.columns) + sway_cols),
    }

    # ── A1: No demographics ──
    print(f"\n{'='*70}")
    print(f"A1 RESULTS — No Demographics (n={len(y)})")
    print(f"{'='*70}")
    print(f"{'Feature Set':25s} {'Ridge':>8s} {'ElasNet':>8s} {'RF':>8s} {'XGB':>8s} {'SVR':>8s}")
    print("-" * 70)

    best_a1, best_a1_name = -999, ""
    for name, (X, fnames) in feature_sets.items():
        results = loo_multi(X, y)
        row = f"{name:25s}"
        for mname in ['Ridge','ElasticNet','RF','XGBoost','SVR']:
            r2 = results[mname]
            row += f" {r2:8.4f}"
            if r2 > best_a1: best_a1 = r2; best_a1_name = f"{name}/{mname}"
        print(row)

    # Feature-selected versions
    for name, (X, fnames) in feature_sets.items():
        if X.shape[1] > 10:
            X_sel, sel_names, _ = select_features_by_corr(X, y, fnames, top_k=8)
            results = loo_multi(X_sel, y)
            row = f"{name}(top8):".ljust(25)
            for mname in ['Ridge','ElasticNet','RF','XGBoost','SVR']:
                r2 = results[mname]
                row += f" {r2:8.4f}"
                if r2 > best_a1: best_a1 = r2; best_a1_name = f"{name}(top8)/{mname}"
            print(row)

    print(f"\n  Best A1: {best_a1:.4f} ({best_a1_name})")
    print(f"  Previous best A1: 0.146")

    # ── A2: With demographics ──
    print(f"\n{'='*70}")
    print(f"A2 RESULTS — With Demographics (n={len(y)})")
    print(f"{'='*70}")
    print(f"{'Feature Set':25s} {'Ridge':>8s} {'ElasNet':>8s} {'RF':>8s} {'XGB':>8s} {'SVR':>8s}")
    print("-" * 70)

    best_a2, best_a2_name = -999, ""
    a2_sets = {
        'Demo': X_demo,
        'Activity+Demo': np.column_stack([X_act, X_demo]),
        'Gait13+Demo': np.column_stack([X_gait, X_demo]),
        'CWT+Demo': np.column_stack([X_cwt, X_demo]),
        'Act+Gait+Demo': np.column_stack([X_act, X_gait, X_demo]),
        'Act+CWT+Demo': np.column_stack([X_act, X_cwt, X_demo]),
        'Gait+CWT+Demo': np.column_stack([X_gait, X_cwt, X_demo]),
        'All+Demo': np.column_stack([X_act, X_gait, X_cwt, X_demo]),
    }

    for name, X in a2_sets.items():
        results = loo_multi(X, y)
        row = f"{name:25s}"
        for mname in ['Ridge','ElasticNet','RF','XGBoost','SVR']:
            r2 = results[mname]
            row += f" {r2:8.4f}"
            if r2 > best_a2: best_a2 = r2; best_a2_name = f"{name}/{mname}"
        print(row)

    # Feature-selected A2
    all_X = np.column_stack([X_act, X_gait, X_cwt, X_demo])
    all_names = list(act_df.columns) + sway_cols + list(cwt_df.columns) + ["cohort_M","Age","Sex","Height"]
    for top_k in [10, 15, 20]:
        X_sel, sel_names, _ = select_features_by_corr(all_X, y, all_names, top_k=top_k)
        results = loo_multi(X_sel, y)
        row = f"All(top{top_k})+Demo:".ljust(25)
        for mname in ['Ridge','ElasticNet','RF','XGBoost','SVR']:
            r2 = results[mname]
            row += f" {r2:8.4f}"
            if r2 > best_a2: best_a2 = r2; best_a2_name = f"All(top{top_k})/{mname}"
        print(row)

    print(f"\n  Best A2: {best_a2:.4f} ({best_a2_name})")
    print(f"  Previous best A2: 0.431")
