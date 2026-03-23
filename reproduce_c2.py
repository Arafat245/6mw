#!/usr/bin/env python3
"""
Reproduce C2 R²=0.792 from scratch.

Pipeline:
  Step 1: csv_raw2/ → preprocess (get fs from Timestamps, trim, resample,
          gravity removal, Rodrigues rotation, PCA yaw, bandpass) → csv_preprocessed2/
  Step 2: csv_preprocessed2/ → extract 10 gait features (features_top10)
  Step 3: csv_preprocessed2/ → compute vt_rms_g
  Step 4: Add ml_over_enmo, ml_over_vt → 13 sway features (features13)
  Step 5: csv_raw2/ → CWT extraction → 28 features
  Step 6: SwayDemographics.xlsx → Demo(H) → 4 features
  Step 7: Combine → Ridge LOO → R²

Code copied from predicting_6mwd2.py — NO feature values copied.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import linregress
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import pywt

BASE = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════
# STEP 1: PREPROCESSING (from predicting_6mwd2.py)
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PreprocConfig:
    target_fs: float = 30.0
    trim_seconds: float = 10.0
    gravity_lp_hz: float = 0.25
    step_band_hz: tuple[float, float] = (0.25, 2.5)
    filter_order: int = 4
    out_dir: Path = Path("csv_preprocessed2")

_filename_re = re.compile(
    r"^(?P<cohort>[CM])(?P<id>\d+)_?(?P<year>\d{4})?_(?P<sixmwd>\d+)\.csv$", re.IGNORECASE
)

def parse_filename(p: Path) -> dict:
    m = _filename_re.match(p.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {p.name}")
    return {
        "cohort": m.group("cohort").upper(),
        "subj_id": int(m.group("id")),
        "year": int(m.group("year")) if m.group("year") else None,
        "sixmwd": int(m.group("sixmwd")),
    }

def get_fs_from_timestamps(timestamps):
    """Get actual sampling rate from timestamp column."""
    diffs = np.diff(timestamps)
    diffs_pos = diffs[diffs > 0]
    if len(diffs_pos) == 0:
        return 30.0  # fallback
    median_dt = np.median(diffs_pos)
    return round(1.0 / median_dt)

def trim_edges(arr, fs, trim_seconds):
    n_trim = int(round(trim_seconds * fs))
    if 2 * n_trim >= len(arr):
        raise ValueError("Trim length too large for signal length.")
    return arr[n_trim:len(arr) - n_trim, :]

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

def align_to_ap_ml_vt(arr_raw_g, fs, cfg):
    b, a = butter_lowpass(cfg.gravity_lp_hz, fs, cfg.filter_order)
    g_est = zero_phase_filter(arr_raw_g, b, a)
    arr_dyn = arr_raw_g - g_est
    g_mean = g_est.mean(axis=0)
    # Rodrigues rotation
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
    return apmlvt, g_est

def preprocess_file(path, cfg):
    meta = parse_filename(path)
    df_all = pd.read_csv(path, usecols=["Timestamp", "X", "Y", "Z"])
    timestamps = df_all["Timestamp"].values
    df = df_all[["X", "Y", "Z"]].astype(np.float64)
    arr_raw = df.to_numpy()

    if arr_raw.shape[0] <= 1000:
        return None  # skip very short files

    fs0 = get_fs_from_timestamps(timestamps)
    arr_trim = trim_edges(arr_raw, fs=fs0, trim_seconds=cfg.trim_seconds)
    arr_rs = resample_uniform(arr_trim, src_fs=fs0, dst_fs=cfg.target_fs)
    fs = cfg.target_fs
    arr_rs_raw = arr_rs.copy()
    apmlvt_dyn, g_est = align_to_ap_ml_vt(arr_rs, fs=fs, cfg=cfg)
    lo, hi = cfg.step_band_hz
    b, a = butter_bandpass(lo, hi, fs, order=cfg.filter_order)
    apmlvt_bp = zero_phase_filter(apmlvt_dyn, b, a)
    vm_dyn = np.linalg.norm(apmlvt_dyn, axis=1)
    vm_raw = np.linalg.norm(arr_rs_raw, axis=1)
    enmo = np.maximum(vm_raw - 1.0, 0.0)
    out = pd.DataFrame({
        "AP": apmlvt_dyn[:, 0], "ML": apmlvt_dyn[:, 1], "VT": apmlvt_dyn[:, 2],
        "AP_bp": apmlvt_bp[:, 0], "ML_bp": apmlvt_bp[:, 1], "VT_bp": apmlvt_bp[:, 2],
        "VM_dyn": vm_dyn, "VM_raw": vm_raw, "ENMO": enmo,
        "cohort": meta["cohort"], "subj_id": meta["subj_id"],
        "year": meta["year"], "sixmwd": meta["sixmwd"], "fs": fs,
        "trim_s": cfg.trim_seconds, "lp_hz": cfg.gravity_lp_hz,
        "bp_lo_hz": cfg.step_band_hz[0], "bp_hi_hz": cfg.step_band_hz[1],
    })
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / path.name
    out.to_csv(out_path, index=False)
    return out_path


# ══════════════════════════════════════════════════════════════════
# STEP 2: GAIT10 FEATURE EXTRACTION (from predicting_6mwd2.py)
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
# STEP 3-4: VT_RMS + SWAY RATIOS (from predicting_6mwd2.py)
# ══════════════════════════════════════════════════════════════════

def compute_vt_rms(preproc_dir):
    rows = []
    for p in sorted(preproc_dir.glob("*.csv")):
        meta_m = _filename_re.match(p.name)
        if not meta_m: continue
        dfp = pd.read_csv(p, usecols=["VT"], dtype={"VT": "float64"})
        vt_rms = float(np.sqrt(np.mean(np.square(dfp["VT"].to_numpy(float)))))
        rows.append({
            "cohort": meta_m.group("cohort").upper(),
            "subj_id": int(meta_m.group("id")),
            "sixmwd": int(meta_m.group("sixmwd")),
            "vt_rms_g": vt_rms,
        })
    return pd.DataFrame(rows)

def add_sway_ratios(feats):
    f = feats.copy()
    f["ml_over_enmo"] = f["ml_rms_g"] / f["enmo_mean_g"].replace(0.0, np.nan)
    f["ml_over_vt"] = f["ml_rms_g"] / f["vt_rms_g"].replace(0.0, np.nan) if "vt_rms_g" in f.columns else np.nan
    return f


# ══════════════════════════════════════════════════════════════════
# STEP 5: CWT EXTRACTION (from raw csv_raw2)
# ══════════════════════════════════════════════════════════════════

def extract_cwt(raw_sig, fs=30.0, n_seg=6):
    vm = np.sqrt(raw_sig[:,0]**2 + raw_sig[:,1]**2 + raw_sig[:,2]**2)
    vm_c = vm - vm.mean()  # mean-subtract full signal, but normalize PER SEGMENT
    freqs = np.linspace(0.5, 12, 50)
    def cwt_seg(seg_raw):
        # Normalize each segment independently
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
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PREPROC2 = BASE / "csv_preprocessed2"
    RAW = BASE / "csv_raw2"
    ids = pd.read_csv(BASE / "feats" / "target_6mwd.csv")
    y_all = ids["sixmwd"].values.astype(float)

    # Step 1: Preprocess (skip if csv_preprocessed2 already exists)
    if len(list(PREPROC2.glob("*.csv"))) < 120:
        print("Step 1: Preprocessing csv_raw2 → csv_preprocessed2...")
        cfg = PreprocConfig(target_fs=30.0, trim_seconds=10.0)
        for p in sorted(RAW.glob("*.csv")):
            result = preprocess_file(p, cfg)
            if result is None:
                print(f"  Skipped {p.name} (too short)")
        print(f"  Done: {len(list(PREPROC2.glob('*.csv')))} files")
    else:
        print(f"Step 1: Using existing csv_preprocessed2 ({len(list(PREPROC2.glob('*.csv')))} files)")

    # Step 2-4: Extract gait13 features
    print("\nStep 2-4: Extracting gait13 features...")
    gait_rows, valid = [], []
    for i, (_, r) in enumerate(ids.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        pp = PREPROC2 / fn
        if pp.exists():
            gait_rows.append(extract_gait10(pd.read_csv(pp)))
            valid.append(True)
        else:
            valid.append(False)
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(ids)}")
    print(f"  {len(ids)}/{len(ids)} (valid: {sum(valid)})")
    valid = np.array(valid)
    gait_df = pd.DataFrame(gait_rows)

    # VT RMS + sway ratios
    vt_rms_df = compute_vt_rms(PREPROC2)
    gait_with_meta = pd.concat([ids[valid].reset_index(drop=True), gait_df], axis=1)
    merged = gait_with_meta.merge(vt_rms_df, on=["cohort", "subj_id", "sixmwd"], how="left")
    sway_df = add_sway_ratios(merged)

    sway_cols = ["cadence_hz", "step_time_cv_pct", "acf_step_regularity", "hr_ap", "hr_vt",
                 "ml_rms_g", "ml_spectral_entropy", "jerk_mean_abs_gps", "enmo_mean_g",
                 "cadence_slope_per_min", "vt_rms_g", "ml_over_enmo", "ml_over_vt"]
    X_sway = sway_df[sway_cols].values.astype(float)
    for j in range(X_sway.shape[1]):
        m = np.isnan(X_sway[:, j])
        if m.any(): X_sway[m, j] = np.nanmedian(X_sway[:, j])

    # Step 5: CWT from raw
    print("\nStep 5: Extracting CWT from csv_raw2...")
    cwt_rows = []
    for i, (_, r) in enumerate(ids[valid].iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        raw = pd.read_csv(RAW / fn, usecols=["X", "Y", "Z"]).values.astype(np.float32)
        cwt_rows.append(extract_cwt(raw))
        if (i+1) % 50 == 0: print(f"  {i+1}/{sum(valid)}")
    print(f"  {sum(valid)}/{sum(valid)}")
    cwt_df = pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan)
    for c in cwt_df.columns:
        if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(cwt_df[c].median())
    X_cwt = cwt_df.values.astype(float)

    # Step 6: Demographics
    print("\nStep 6: Loading demographics...")
    demo = pd.read_excel(BASE / "SwayDemographics.xlsx")
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    p = ids[valid].reset_index(drop=True).merge(demo, on=["cohort", "subj_id"], how="left")
    p["cohort_M"] = (p["cohort"] == "M").astype(int)
    for c in ["Age", "Sex", "Height"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    X_demo = p[["cohort_M", "Age", "Sex", "Height"]].values.astype(float)
    for j in range(X_demo.shape[1]):
        m = np.isnan(X_demo[:, j])
        if m.any(): X_demo[m, j] = np.nanmedian(X_demo[:, j])

    y = y_all[valid]

    # Step 7: Predict
    def loo(X, y):
        pred = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=10)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))
        return round(r2_score(y, pred), 4)

    print(f"\n{'='*60}")
    print(f"RESULTS (n={len(y)})")
    print(f"{'='*60}")
    print(f"\n  Individual:")
    print(f"    Gait13 only        (13f):  R² = {loo(X_sway, y)}")
    print(f"    CWT only           (28f):  R² = {loo(X_cwt, y)}")
    print(f"    Demo(H) only        (4f):  R² = {loo(X_demo, y)}")
    print(f"\n  Pairwise:")
    print(f"    Gait13 + CWT       (41f):  R² = {loo(np.column_stack([X_sway, X_cwt]), y)}")
    print(f"    Gait13 + Demo(H)   (17f):  R² = {loo(np.column_stack([X_sway, X_demo]), y)}")
    print(f"    CWT + Demo(H)      (32f):  R² = {loo(np.column_stack([X_cwt, X_demo]), y)}")
    r2 = loo(np.column_stack([X_sway, X_cwt, X_demo]), y)
    print(f"\n  All Combined:")
    print(f"    Gait13 + CWT + Demo(H) (45f):  R² = {r2}")
    print(f"\n  Target: 0.792")
