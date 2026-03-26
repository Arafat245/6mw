#!/usr/bin/env python3
"""
Experiments to improve home 6MWD prediction based on investigation findings.
Baseline: Gait(11)+Demo(5), Ridge α=50, R²=0.488, MAE=175ft

Exp 1: Feature selection — drop unreliable features, keep intensity-robust ones
Exp 2: Add activity/centile features from existing NPZ
Exp 3: Per-bout median aggregation (robust to outlier bouts)
Exp 4: Cadence quality filter (reject bouts with cadence < 1.0 Hz)
Exp 5: Longer walking window (600s, 900s, 1200s)
Exp 6: Combinations of best ideas
"""
import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import pearsonr, spearmanr
import math

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

# ══════════════════════════════════════════════════════════════════
# COMMON
# ══════════════════════════════════════════════════════════════════
def eval_loo(X, y, alpha=50):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    r2 = r2_score(y, pr); mae = mean_absolute_error(y, pr)
    return r2, mae, pearsonr(y, pr)[0], spearmanr(y, pr)[0]


def best_alpha(X, y, alphas=[5, 10, 20, 50, 100, 200]):
    best_r2, best_a = -999, 50
    for a in alphas:
        r2, mae, rv, rho = eval_loo(X, y, a)
        if r2 > best_r2:
            best_r2, best_mae, best_rv, best_rho, best_a = r2, mae, rv, rho, a
    return best_r2, best_mae, best_rv, best_rho, best_a


def report(name, nf, r2, mae, rv, rho, alpha):
    print(f"  {name:45s} {nf:>3d}f  α={alpha:>3d}  R²={r2:.4f}  MAE={mae:.0f}ft  r={rv:.3f}  ρ={rho:.3f}")


def load_data():
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    valid = ~((ids['cohort'] == 'M') & (ids['subj_id'] == 22))
    ids102 = ids[valid].reset_index(drop=True)

    d = np.load(BASE / 'feats' / 'home_hybrid_v2_features.npz', allow_pickle=True)

    # Find clinic-valid subjects (same as results_table_final.py)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = []
    for _, r in ids102.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        clinic_valid.append((PREPROC2 / fn).exists())
    clinic_valid = np.array(clinic_valid)
    cidx = np.where(clinic_valid)[0]

    X_gait = d['X_gait'][cidx]     # (n, 13): 11 gait + 2 sway ratios
    X_act = d['X_act'][cidx]       # (n, 15)
    X_cent = d['X_cent'][cidx]     # (n, 18)
    X_cwt = d['X_cwt'][cidx]       # (n, 28)

    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids102[clinic_valid].reset_index(drop=True).merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo_5 = p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float)
    X_demo_3 = p[['cohort_M', 'Age', 'Sex']].values.astype(float)

    for X in [X_gait, X_act, X_cent, X_cwt, X_demo_5, X_demo_3]:
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any(): X[m, j] = np.nanmedian(X[:, j])

    y = ids102[clinic_valid].reset_index(drop=True)['sixmwd'].values.astype(float)
    ids_valid = ids102[clinic_valid].reset_index(drop=True)

    return X_gait, X_act, X_cent, X_cwt, X_demo_5, X_demo_3, y, ids_valid


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION HELPERS (for per-bout experiments)
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
    return np.array([np.dot(x[:n-k], x[k:])/(d if d>0 else 1.0) for k in range(max_lag+1)])

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

def _rodrigues(axis, theta):
    ax = axis / (np.linalg.norm(axis)+1e-12)
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*(K@K)

def preprocess_walking(walking_xyz, fs=30.0):
    arr = walking_xyz.copy()
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
        "ENMO": enmo, "fs": fs})

def extract_gait11(df):
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
    per_min = int(round(60*fs)); m_cnt = min(6, max(1, len(vt_bp)//per_min))
    cads = [_psd_peak(vt_bp[j*per_min:(j+1)*per_min], fs) for j in range(m_cnt)
            if len(vt_bp[j*per_min:(j+1)*per_min]) >= per_min//2]
    cads = np.array([c for c in cads if np.isfinite(c)])
    f["cadence_slope_per_min"] = float(np.polyfit(np.arange(len(cads)), cads, 1)[0]) if len(cads) >= 3 else np.nan
    f["vt_rms_g"] = float(np.sqrt(np.mean(vt**2)))
    return f


# ══════════════════════════════════════════════════════════════════
# MAIN EXPERIMENTS
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    X_gait, X_act, X_cent, X_cwt, X_demo_5, X_demo_3, y, ids_valid = load_data()
    n = len(y)

    # Gait feature column names (from home_hybrid_v2)
    gait_names = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
                  'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
                  'cadence_slope_per_min', 'vt_rms_g', 'ml_over_enmo', 'ml_over_vt']

    X_gait11 = X_gait[:, :11]  # first 11 = gait features
    X_sway2 = X_gait[:, 11:13]  # last 2 = sway ratios

    print(f"n={n} subjects")
    print(f"{'='*100}")

    # ── BASELINE ──
    print("\nBASELINE:")
    X_base = np.column_stack([X_gait11, X_demo_5])
    r2, mae, rv, rho = eval_loo(X_base, y, 50)
    report("Gait(11)+Demo(5) [BASELINE]", 16, r2, mae, rv, rho, 50)

    # ══════════════════════════════════════════════════════════════
    # EXP 1: Feature selection — keep only reliable features
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print("EXP 1: Feature selection (drop unreliable features)")
    print(f"{'─'*100}")

    # Reliability tiers from investigation:
    # Good (ρ>0.5): enmo_mean(0.74), jerk(0.63), ml_rms(0.56), cadence(0.53)
    # OK (ρ 0.2-0.5): acf_step_reg(0.25), hr_ap(0.33)
    # Bad (ρ<0.2): step_time_cv(0.09), ml_spec_entropy(0.13), cadence_slope(0.16), hr_vt(0.18)

    # Top 4 reliable features
    top4_idx = [gait_names.index(f) for f in ['enmo_mean_g', 'jerk_mean_abs_gps', 'ml_rms_g', 'cadence_hz']]
    X_top4 = X_gait[:, top4_idx]
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_top4, X_demo_5]), y)
    report("Top4(enmo,jerk,ml_rms,cad)+Demo(5)", 9, r2, mae, rv, rho, a)

    # Top 6 (add acf + hr_ap)
    top6_idx = top4_idx + [gait_names.index(f) for f in ['acf_step_regularity', 'hr_ap']]
    X_top6 = X_gait[:, top6_idx]
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_top6, X_demo_5]), y)
    report("Top6(+acf,hr_ap)+Demo(5)", 11, r2, mae, rv, rho, a)

    # Top 6 + vt_rms (ρ not tested but physically meaningful)
    top7_idx = top6_idx + [gait_names.index('vt_rms_g')]
    X_top7 = X_gait[:, top7_idx]
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_top7, X_demo_5]), y)
    report("Top7(+vt_rms)+Demo(5)", 12, r2, mae, rv, rho, a)

    # Drop only the worst 3 (step_time_cv, cadence_slope, ml_spectral_entropy)
    drop3_idx = [i for i in range(11) if gait_names[i] not in
                 ['step_time_cv_pct', 'cadence_slope_per_min', 'ml_spectral_entropy']]
    X_drop3 = X_gait[:, drop3_idx]
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_drop3, X_demo_5]), y)
    report("Gait(8, drop cv/slope/entropy)+Demo(5)", 13, r2, mae, rv, rho, a)

    # Drop worst 4 (+ hr_vt)
    drop4_idx = [i for i in range(11) if gait_names[i] not in
                 ['step_time_cv_pct', 'cadence_slope_per_min', 'ml_spectral_entropy', 'hr_vt']]
    X_drop4 = X_gait[:, drop4_idx]
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_drop4, X_demo_5]), y)
    report("Gait(7, drop cv/slope/entropy/hr_vt)+Demo(5)", 12, r2, mae, rv, rho, a)

    # ══════════════════════════════════════════════════════════════
    # EXP 2: Add activity and centile features
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print("EXP 2: Add activity/centile features from NPZ")
    print(f"{'─'*100}")

    # Activity alone
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_act, X_demo_5]), y)
    report("Activity(15)+Demo(5)", 20, r2, mae, rv, rho, a)

    # Centile alone
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_cent, X_demo_5]), y)
    report("Centile(18)+Demo(5)", 23, r2, mae, rv, rho, a)

    # Gait + Activity
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait11, X_act, X_demo_5]), y)
    report("Gait(11)+Activity(15)+Demo(5)", 31, r2, mae, rv, rho, a)

    # Gait + Centile
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait11, X_cent, X_demo_5]), y)
    report("Gait(11)+Centile(18)+Demo(5)", 34, r2, mae, rv, rho, a)

    # Gait + Activity + Centile
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait11, X_act, X_cent, X_demo_5]), y)
    report("Gait(11)+Act(15)+Cent(18)+Demo(5)", 49, r2, mae, rv, rho, a)

    # Top7 gait + Activity
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_top7, X_act, X_demo_5]), y)
    report("Top7+Activity(15)+Demo(5)", 27, r2, mae, rv, rho, a)

    # Top7 gait + Centile
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_top7, X_cent, X_demo_5]), y)
    report("Top7+Centile(18)+Demo(5)", 30, r2, mae, rv, rho, a)

    # Top7 + Act + Cent
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_top7, X_act, X_cent, X_demo_5]), y)
    report("Top7+Act(15)+Cent(18)+Demo(5)", 45, r2, mae, rv, rho, a)

    # Home CWT
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait11, X_cwt, X_demo_5]), y)
    report("Gait(11)+CWT(28)+Demo(5)", 44, r2, mae, rv, rho, a)

    # All home features
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait11, X_act, X_cent, X_cwt, X_demo_5]), y)
    report("Gait(11)+Act(15)+Cent(18)+CWT(28)+Demo(5)", 77, r2, mae, rv, rho, a)

    # Sway ratios too
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait, X_act, X_cent, X_cwt, X_demo_5]), y)
    report("Gait(13)+Act(15)+Cent(18)+CWT(28)+Demo(5)", 79, r2, mae, rv, rho, a)

    # ══════════════════════════════════════════════════════════════
    # EXP 3: Per-bout median aggregation
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print("EXP 3: Per-bout median aggregation")
    print(f"{'─'*100}")

    from home.home_hybrid_models_v2 import (load_cached_daytime, load_clinic_raw,
        detect_active_bouts, refine_with_hr, compute_walking_signature, select_walking_segment)

    cache_path = OUT / 'perbout_features_cache.npz'
    if cache_path.exists():
        print("  Loading per-bout features from cache...", flush=True)
        cached = np.load(cache_path, allow_pickle=True)
        X_median = cached['X_median']
        X_iqr = cached['X_iqr']
        perbout_valid = cached['valid']
    else:
        print("  Extracting per-bout features (slow)...", flush=True)
        gait_keys = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
                     'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
                     'cadence_slope_per_min', 'vt_rms_g']

        all_medians, all_iqrs, perbout_valid = [], [], []
        for i, (_, r) in enumerate(ids_valid.iterrows()):
            cohort, sid = r['cohort'], int(r['subj_id'])
            year, sixmwd = int(r['year']), int(r['sixmwd'])
            xyz_home, fs = load_cached_daytime(cohort, sid, year, sixmwd)
            xyz_clinic, fs_c = load_clinic_raw(cohort, sid, year, sixmwd)

            if xyz_home is None:
                all_medians.append(np.full(11, np.nan))
                all_iqrs.append(np.full(11, np.nan))
                perbout_valid.append(False)
                continue

            bouts = detect_active_bouts(xyz_home, fs)
            refined = refine_with_hr(xyz_home, fs, bouts)

            if not refined or len(refined) < 3:
                all_medians.append(np.full(11, np.nan))
                all_iqrs.append(np.full(11, np.nan))
                perbout_valid.append(False)
                continue

            # Rank bouts by clinic similarity (same as pipeline)
            if xyz_clinic is not None and len(xyz_clinic) > 100:
                clinic_sig = compute_walking_signature(xyz_clinic, fs_c)
                scored = []
                for s, e in refined:
                    bout_sig = compute_walking_signature(xyz_home[s:e], fs)
                    sim = float(np.dot(clinic_sig, bout_sig) /
                                (np.linalg.norm(clinic_sig)*np.linalg.norm(bout_sig)+1e-12))
                    scored.append((s, e, sim))
                scored.sort(key=lambda x: x[2], reverse=True)
            else:
                scored = [(s, e, 0) for s, e in refined]
                scored.sort(key=lambda x: x[1]-x[0], reverse=True)

            # Extract features per bout (top 20 bouts, min 30s each)
            bout_features = []
            for s, e, sim in scored[:20]:
                dur = (e - s) / fs
                if dur < 30:
                    continue
                try:
                    pp = preprocess_walking(xyz_home[s:e], fs)
                    feats = extract_gait11(pp)
                    vals = [feats.get(k, np.nan) for k in gait_keys]
                    # Quality check: skip if cadence < 1.0
                    if np.isfinite(vals[0]) and vals[0] >= 1.0:
                        bout_features.append(vals)
                except:
                    pass

            if len(bout_features) < 2:
                all_medians.append(np.full(11, np.nan))
                all_iqrs.append(np.full(11, np.nan))
                perbout_valid.append(False)
                continue

            arr = np.array(bout_features)
            medians = np.nanmedian(arr, axis=0)
            iqrs = np.nanpercentile(arr, 75, axis=0) - np.nanpercentile(arr, 25, axis=0)
            all_medians.append(medians)
            all_iqrs.append(iqrs)
            perbout_valid.append(True)

            if (i+1) % 20 == 0:
                print(f"    [{i+1}/{n}] {len(bout_features)} valid bouts", flush=True)

        X_median = np.array(all_medians)
        X_iqr = np.array(all_iqrs)
        perbout_valid = np.array(perbout_valid)
        np.savez(cache_path, X_median=X_median, X_iqr=X_iqr, valid=perbout_valid)
        print(f"  Cached. Valid: {perbout_valid.sum()}/{n}", flush=True)

    # Evaluate per-bout features
    valid_mask = perbout_valid.astype(bool)
    n_valid = valid_mask.sum()
    print(f"  Per-bout valid: {n_valid}/{n}")

    if n_valid >= 80:
        X_med_v = X_median[valid_mask]
        y_v = y[valid_mask]
        D5_v = X_demo_5[valid_mask]

        # Impute NaN
        for j in range(X_med_v.shape[1]):
            m = np.isnan(X_med_v[:, j])
            if m.any(): X_med_v[m, j] = np.nanmedian(X_med_v[:, j])

        r2, mae, rv, rho, a = best_alpha(np.column_stack([X_med_v, D5_v]), y_v)
        report(f"Median-per-bout Gait(11)+Demo(5) [n={n_valid}]", 16, r2, mae, rv, rho, a)

        # Compare to baseline on same subjects
        X_base_v = np.column_stack([X_gait11[valid_mask], D5_v])
        r2b, maeb, rvb, rhob = eval_loo(X_base_v, y_v, 50)
        report(f"BASELINE on same n={n_valid}", 16, r2b, maeb, rvb, rhob, 50)

        # Median + IQR features
        X_iqr_v = X_iqr[valid_mask]
        for j in range(X_iqr_v.shape[1]):
            m = np.isnan(X_iqr_v[:, j])
            if m.any(): X_iqr_v[m, j] = np.nanmedian(X_iqr_v[:, j])
        r2, mae, rv, rho, a = best_alpha(np.column_stack([X_med_v, X_iqr_v, D5_v]), y_v)
        report(f"Median+IQR Gait(22)+Demo(5) [n={n_valid}]", 27, r2, mae, rv, rho, a)

        # Median top4 reliable only
        top4_rel = [gait_keys.index(f) for f in ['enmo_mean_g', 'jerk_mean_abs_gps', 'ml_rms_g', 'cadence_hz']]
        X_med_top4 = X_med_v[:, top4_rel]
        r2, mae, rv, rho, a = best_alpha(np.column_stack([X_med_top4, D5_v]), y_v)
        report(f"Median Top4+Demo(5) [n={n_valid}]", 9, r2, mae, rv, rho, a)

        # Median + Activity
        X_act_v = X_act[valid_mask]
        r2, mae, rv, rho, a = best_alpha(np.column_stack([X_med_v, X_act_v, D5_v]), y_v)
        report(f"Median Gait(11)+Act(15)+Demo(5) [n={n_valid}]", 31, r2, mae, rv, rho, a)

    # ══════════════════════════════════════════════════════════════
    # EXP 4: Longer walking windows
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print("EXP 4: Longer walking windows (600s, 900s, 1200s)")
    print(f"{'─'*100}")

    cache_long = OUT / 'longer_window_features_cache.npz'
    if cache_long.exists():
        print("  Loading from cache...", flush=True)
        cl = np.load(cache_long, allow_pickle=True)
        X_600 = cl['X_600']; X_900 = cl['X_900']; X_1200 = cl['X_1200']
        long_valid = cl['valid']
    else:
        print("  Extracting features with longer windows...", flush=True)
        gait_keys = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
                     'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
                     'cadence_slope_per_min', 'vt_rms_g']
        all_feats = {dur: [] for dur in [600, 900, 1200]}
        long_valid = []

        for i, (_, r) in enumerate(ids_valid.iterrows()):
            cohort, sid = r['cohort'], int(r['subj_id'])
            year, sixmwd = int(r['year']), int(r['sixmwd'])
            xyz_home, fs = load_cached_daytime(cohort, sid, year, sixmwd)
            xyz_clinic, fs_c = load_clinic_raw(cohort, sid, year, sixmwd)

            if xyz_home is None:
                for dur in [600, 900, 1200]:
                    all_feats[dur].append(np.full(11, np.nan))
                long_valid.append(False)
                continue

            bouts = detect_active_bouts(xyz_home, fs)
            refined = refine_with_hr(xyz_home, fs, bouts)

            ok = True
            for dur in [600, 900, 1200]:
                walking, _ = select_walking_segment(xyz_home, fs, refined, dur, xyz_clinic, fs_c)
                if walking is None or len(walking) < 30*fs:
                    all_feats[dur].append(np.full(11, np.nan))
                    ok = False
                    continue
                pp = preprocess_walking(walking, fs)
                feats = extract_gait11(pp)
                all_feats[dur].append([feats.get(k, np.nan) for k in gait_keys])

            long_valid.append(ok)
            if (i+1) % 20 == 0:
                print(f"    [{i+1}/{n}]", flush=True)

        X_600 = np.array(all_feats[600])
        X_900 = np.array(all_feats[900])
        X_1200 = np.array(all_feats[1200])
        long_valid = np.array(long_valid)
        np.savez(cache_long, X_600=X_600, X_900=X_900, X_1200=X_1200, valid=long_valid)
        print(f"  Cached. Valid: {long_valid.sum()}/{n}", flush=True)

    for label, X_dur in [('600s', X_600), ('900s', X_900), ('1200s', X_1200)]:
        mask = ~np.isnan(X_dur).any(axis=1)
        nv = mask.sum()
        if nv < 80:
            print(f"  {label}: only {nv} valid — skipping")
            continue
        Xv = X_dur[mask]; yv = y[mask]; Dv = X_demo_5[mask]
        for j in range(Xv.shape[1]):
            m = np.isnan(Xv[:, j])
            if m.any(): Xv[m, j] = np.nanmedian(Xv[:, j])
        r2, mae, rv, rho, a = best_alpha(np.column_stack([Xv, Dv]), yv)
        report(f"Window={label} Gait(11)+Demo(5) [n={nv}]", 16, r2, mae, rv, rho, a)
        # Baseline on same subjects
        Xb = np.column_stack([X_gait11[mask], Dv])
        r2b, maeb, rvb, rhob = eval_loo(Xb, yv, 50)
        report(f"  BASELINE (360s) on same n={nv}", 16, r2b, maeb, rvb, rhob, 50)

    # ══════════════════════════════════════════════════════════════
    # EXP 5: Best combinations
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print("EXP 5: Best combinations")
    print(f"{'─'*100}")

    # Top7 + Centile (best from Exp1 + Exp2 insights)
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_top7, X_cent, X_demo_5]), y)
    report("Top7+Centile(18)+Demo(5)", 30, r2, mae, rv, rho, a)

    # Drop3 + Activity
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_drop3, X_act, X_demo_5]), y)
    report("Gait(8)+Activity(15)+Demo(5)", 28, r2, mae, rv, rho, a)

    # Drop3 + Centile
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_drop3, X_cent, X_demo_5]), y)
    report("Gait(8)+Centile(18)+Demo(5)", 31, r2, mae, rv, rho, a)

    # Drop3 + Act + Cent
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_drop3, X_act, X_cent, X_demo_5]), y)
    report("Gait(8)+Act(15)+Cent(18)+Demo(5)", 46, r2, mae, rv, rho, a)

    # Gait + Sway ratios + Activity
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait, X_act, X_demo_5]), y)
    report("Gait(13)+Activity(15)+Demo(5)", 33, r2, mae, rv, rho, a)

    # Demo(3) instead of Demo(5)
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait11, X_demo_3]), y)
    report("Gait(11)+Demo(3) [fewer demo]", 14, r2, mae, rv, rho, a)

    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"Done in {elapsed:.0f}s")
    print(f"BASELINE: Gait(11)+Demo(5), α=50 → R²=0.488, MAE=175ft")
    print(f"{'='*100}")
