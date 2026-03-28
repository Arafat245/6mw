#!/usr/bin/env python3
"""
Enhanced WalkSway features based on literature:
- Alkathiry 2025: mean referencing, 3.5Hz lowpass, NPL, MAA, MFA, P2P
- Lee 2014: raw unnormalized features (avoid U-shaped normalization artifacts)
- Meyer 2024: per-bout distribution aggregation for home data
- Mancini 2012: FD, ApEn features

Same preprocessing for home and clinic. Home uses first 6 min of longest bout.
Per-bout aggregation tested for home only.
"""
import sys, warnings, time, math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import spearmanr, pearsonr, chi2

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

from clinic.reproduce_c2 import (PreprocConfig, trim_edges, resample_uniform,
                                  align_to_ap_ml_vt, butter_bandpass, zero_phase_filter,
                                  get_fs_from_timestamps)
from clinic.extract_walking_sway import extract_walking_sway as extract_original_ws
from temporary_experiments.clinic_free.exp_clinic_free_v2 import detect_walking_bouts_clinicfree

CFG = PreprocConfig()
HOME_DIR = BASE / 'csv_home_daytime'


def eval_loo(X, y, alpha=10):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, pr), mean_absolute_error(y, pr), spearmanr(y, pr)[0]

def best_alpha(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 10)
    for a in alphas:
        r2, mae, rho = eval_loo(X, y, a)
        if r2 > best[0]: best = (r2, mae, rho, a)
    return best

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


# ══════════════════════════════════════════════════════════════════
# ENHANCED WALKSWAY FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _psd_peak_freq(x, fs, fmin=0.5, fmax=3.5):
    if len(x) < int(fs): return float('nan')
    nperseg = int(max(fs * 4, 256))
    freqs, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend='constant')
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band): return float('nan')
    return float(freqs[band][np.argmax(Pxx[band])])


def _harmonic_ratio_ml(signal, fs, cadence_hz, n_harm=10):
    if not np.isfinite(cadence_hz) or cadence_hz <= 0: return float('nan')
    x = signal - np.mean(signal)
    if len(x) < 2: return float('nan')
    X = np.fft.rfft(x); freqs = np.fft.rfftfreq(len(x), d=1.0/fs); mags = np.abs(X)
    ev, od = 0.0, 0.0
    for k in range(1, n_harm + 1):
        fk = k * cadence_hz
        if fk >= freqs[-1]: break
        idx = int(np.argmin(np.abs(freqs - fk)))
        if k % 2 == 0: ev += mags[idx]
        else: od += mags[idx]
    return float(od / (ev + 1e-12)) if ev > 0 else float('nan')


def _approx_entropy(x, m=2, r_factor=0.2):
    """Approximate entropy — measures signal regularity/complexity."""
    N = len(x)
    if N < 50: return float('nan')
    r = r_factor * np.std(x)
    if r < 1e-12: return float('nan')

    def _phi(m_val):
        patterns = np.array([x[i:i + m_val] for i in range(N - m_val + 1)])
        C = np.zeros(len(patterns))
        for i in range(len(patterns)):
            diffs = np.max(np.abs(patterns - patterns[i]), axis=1)
            C[i] = np.sum(diffs <= r) / len(patterns)
        return np.mean(np.log(C + 1e-12))

    return abs(_phi(m) - _phi(m + 1))


def extract_enhanced_walksway(ap, ml, vt, fs=30.0):
    """
    Enhanced walking sway features combining:
    - Original ENMO-normalized features (10)
    - Alkathiry 2025: mean-referenced + 3.5Hz lowpass features
    - Raw unnormalized features
    - Mancini: Approximate Entropy, Frequency Dispersion
    """
    f = {}

    # ── 3.5 Hz lowpass filtering (Alkathiry 2025) ──
    b_lp, a_lp = butter(2, 3.5, btype='lowpass', fs=fs)
    if len(ml) > 3 * max(len(b_lp), len(a_lp)):
        ml_lp = filtfilt(b_lp, a_lp, ml)
        ap_lp = filtfilt(b_lp, a_lp, ap)
        vt_lp = filtfilt(b_lp, a_lp, vt)
    else:
        ml_lp, ap_lp, vt_lp = ml.copy(), ap.copy(), vt.copy()

    # Mean referencing (Alkathiry 2025)
    ml_mr = ml_lp - np.mean(ml_lp)
    ap_mr = ap_lp - np.mean(ap_lp)
    combined_2d = np.sqrt(ap_mr**2 + ml_mr**2)

    duration = len(ml) / fs
    if duration < 1: duration = 1

    # ── Original ENMO-normalized features (current 10) ──
    orig = extract_original_ws(ap, ml, vt, fs)
    for k, v in orig.items():
        f[k] = v

    # ── Mean-referenced features (Alkathiry 2025) ──
    # RMS (mean-referenced)
    f['ml_rms_mr'] = float(np.sqrt(np.mean(ml_mr**2)))
    f['ap_rms_mr'] = float(np.sqrt(np.mean(ap_mr**2)))
    f['2d_rms_mr'] = float(np.sqrt(np.mean(combined_2d**2)))

    # NPL - Normalized Path Length = sum|diff| / duration (Alkathiry Eq.4)
    f['ml_npl'] = float(np.sum(np.abs(np.diff(ml_mr)))) / duration
    f['ap_npl'] = float(np.sum(np.abs(np.diff(ap_mr)))) / duration
    f['2d_npl'] = float(np.sum(np.abs(np.diff(combined_2d)))) / duration

    # MAA - Mean Absolute Acceleration (Alkathiry Eq.5)
    f['ml_maa'] = float(np.mean(np.abs(ml_mr)))
    f['ap_maa'] = float(np.mean(np.abs(ap_mr)))
    f['2d_maa'] = float(np.mean(np.abs(combined_2d)))

    # MFA - Mean Frequency of Acceleration (Alkathiry Eq.6)
    ml_path = np.sum(np.abs(np.diff(ml_mr)))
    ap_path = np.sum(np.abs(np.diff(ap_mr)))
    ml_mean_disp = np.mean(np.abs(ml_mr))
    ap_mean_disp = np.mean(np.abs(ap_mr))
    f['ml_mfa'] = float(ml_path / (4 * np.sqrt(2) * ml_mean_disp * duration)) if ml_mean_disp > 1e-12 else 0
    f['ap_mfa'] = float(ap_path / (4 * np.sqrt(2) * ap_mean_disp * duration)) if ap_mean_disp > 1e-12 else 0

    # P2P - Peak to Peak (Alkathiry)
    f['ml_p2p'] = float(np.ptp(ml_mr))
    f['ap_p2p'] = float(np.ptp(ap_mr))

    # ── Raw unnormalized features ──
    f['ml_range_raw'] = float(np.ptp(ml))
    f['ml_path_raw'] = float(np.sum(np.abs(np.diff(ml))))
    f['ap_rms_raw'] = float(np.sqrt(np.mean(ap**2)))
    if len(ml) > 2 and len(ap) > 2:
        cov = np.cov(ml, ap)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0)
        f['sway_area_raw'] = float(math.pi * chi2.ppf(0.95, 2) *
                                    np.sqrt(eigenvalues[0]) * np.sqrt(eigenvalues[1]))
    else:
        f['sway_area_raw'] = 0.0

    # ── Frequency domain features (Mancini/Meyer) ──
    # Frequency Dispersion (FD) — strongest predictor in Meyer 2024
    if len(ml_mr) > 64:
        freqs_ml, psd_ml = welch(ml_mr, fs=fs, nperseg=min(len(ml_mr), 256))
        total_power = np.sum(psd_ml)
        if total_power > 1e-12:
            psd_norm = psd_ml / total_power
            cf = float(np.sum(freqs_ml * psd_norm))  # centroidal frequency
            f['ml_cf'] = cf
            # FD = sqrt(1 - (sum(f*P)^2 / (sum(f²*P) * sum(P))))
            f2p = np.sum(freqs_ml**2 * psd_norm)
            fd_sq = 1 - (cf**2 / (f2p + 1e-12))
            f['ml_fd'] = float(np.sqrt(max(fd_sq, 0)))
        else:
            f['ml_cf'] = 0; f['ml_fd'] = 0
    else:
        f['ml_cf'] = 0; f['ml_fd'] = 0

    if len(ap_mr) > 64:
        freqs_ap, psd_ap = welch(ap_mr, fs=fs, nperseg=min(len(ap_mr), 256))
        total_power = np.sum(psd_ap)
        if total_power > 1e-12:
            psd_norm = psd_ap / total_power
            cf = float(np.sum(freqs_ap * psd_norm))
            f['ap_cf'] = cf
            f2p = np.sum(freqs_ap**2 * psd_norm)
            fd_sq = 1 - (cf**2 / (f2p + 1e-12))
            f['ap_fd'] = float(np.sqrt(max(fd_sq, 0)))
        else:
            f['ap_cf'] = 0; f['ap_fd'] = 0
    else:
        f['ap_cf'] = 0; f['ap_fd'] = 0

    # Approximate Entropy (Mancini)
    f['ml_apen'] = _approx_entropy(ml_mr[:min(len(ml_mr), 1000)])
    f['ap_apen'] = _approx_entropy(ap_mr[:min(len(ap_mr), 1000)])

    # Jerk in all 3 axes (Meyer 2023 — strongest correlations)
    f['ml_jerk_rms_raw'] = float(np.sqrt(np.mean((np.diff(ml_mr) * fs)**2))) if len(ml_mr) > 1 else 0
    f['ap_jerk_rms_raw'] = float(np.sqrt(np.mean((np.diff(ap_mr) * fs)**2))) if len(ap_mr) > 1 else 0
    f['vt_jerk_rms_raw'] = float(np.sqrt(np.mean((np.diff(vt_lp - np.mean(vt_lp)) * fs)**2))) if len(vt_lp) > 1 else 0

    return f


def preprocess_raw_to_apmlvt(raw_xyz, fs_orig):
    """Same preprocessing as clinic pipeline → returns AP, ML, VT arrays."""
    arr_trim = trim_edges(raw_xyz, fs=fs_orig, trim_seconds=CFG.trim_seconds)
    arr_rs = resample_uniform(arr_trim, src_fs=fs_orig, dst_fs=CFG.target_fs)
    fs = CFG.target_fs
    apmlvt, _ = align_to_ap_ml_vt(arr_rs, fs=fs, cfg=CFG)
    return apmlvt[:, 0], apmlvt[:, 1], apmlvt[:, 2], fs


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
    RAW = BASE / 'csv_raw2'
    clinic_valid = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                             for _, r in ids101.iterrows()])
    ids_v = ids101[clinic_valid].reset_index(drop=True)
    y = ids_v['sixmwd'].values.astype(float)
    n = len(y)

    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo5 = impute(p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float))

    print(f"Enhanced WalkSway Features (n={n})")
    print(f"{'='*90}\n")

    # ── Extract clinic features ──
    print("Extracting CLINIC enhanced WalkSway...")
    clinic_ws = []
    for i, (_, r) in enumerate(ids_v.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        df = pd.read_csv(PREPROC2 / fn)
        ws = extract_enhanced_walksway(df['AP'].values, df['ML'].values, df['VT'].values, 30.0)
        clinic_ws.append(ws)
    ws_cols = sorted(clinic_ws[0].keys())
    X_ws_c = impute(pd.DataFrame(clinic_ws)[ws_cols].values.astype(float))
    print(f"  {X_ws_c.shape[1]} features extracted")

    # ── Extract home features (first 6 min of longest bout) ──
    print("Extracting HOME enhanced WalkSway (first 6 min of longest bout)...")
    home_ws = []
    home_valid = []
    FS = 30
    six_min = 6 * 60 * FS

    for i, (_, r) in enumerate(ids_v.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        fp = HOME_DIR / fn
        if not fp.exists():
            home_ws.append(None); home_valid.append(False); continue
        xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
        bouts = detect_walking_bouts_clinicfree(xyz, FS, min_bout_sec=10, merge_gap_sec=5)
        if not bouts:
            home_ws.append(None); home_valid.append(False); continue
        longest = max(bouts, key=lambda b: b[1] - b[0])
        bout_xyz = xyz[longest[0]:longest[1]]
        first6 = bout_xyz[:min(six_min, len(bout_xyz))]
        try:
            ap, ml, vt, fs = preprocess_raw_to_apmlvt(first6, FS)
            ws = extract_enhanced_walksway(ap, ml, vt, fs)
            home_ws.append(ws); home_valid.append(True)
        except:
            home_ws.append(None); home_valid.append(False)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}]", flush=True)

    home_valid = np.array(home_valid)
    both = home_valid
    nv = both.sum()
    X_ws_h = impute(pd.DataFrame([w for w, v in zip(home_ws, both) if v])[ws_cols].values.astype(float))
    X_ws_c_v = X_ws_c[both]
    X_d5_v = X_demo5[both]
    y_v = y[both]

    print(f"  {nv} valid subjects, {X_ws_h.shape[1]} features")

    # ── Correlations ──
    print(f"\n{'='*90}")
    print(f"FEATURE CORRELATIONS WITH 6MWD (n={nv})")
    print(f"{'='*90}")
    print(f"{'Feature':<30s}  {'Home ρ':>8s}  {'Clinic ρ':>8s}  {'Same dir?':>9s}")
    print('-' * 65)
    n_same, n_diff = 0, 0
    for j, col in enumerate(ws_cols):
        rho_h = spearmanr(X_ws_h[:, j], y_v)[0]
        rho_c = spearmanr(X_ws_c_v[:, j], y_v)[0]
        same = 'YES' if (rho_h * rho_c > 0) or (abs(rho_h) < 0.05 or abs(rho_c) < 0.05) else 'NO'
        if same == 'YES': n_same += 1
        else: n_diff += 1
        if abs(rho_h) > 0.15 or abs(rho_c) > 0.15:
            print(f'{col:<30s}  {rho_h:>+8.3f}  {rho_c:>+8.3f}  {same:>9s}')
    print(f"\nDirection agreement: {n_same}/{n_same+n_diff} features ({100*n_same/(n_same+n_diff):.0f}%)")

    # ── Prediction: original vs enhanced ──
    print(f"\n{'='*90}")
    print(f"PREDICTION (n={nv})")
    print(f"{'='*90}")

    # Original 10 WalkSway features
    orig_cols = ['ml_range_norm', 'ml_path_length_norm', 'ml_jerk_rms_norm', 'ap_rms_norm',
                 'ap_range_norm', 'sway_ellipse_norm', 'ml_velocity_rms_norm', 'stride_ml_cv',
                 'ml_ap_ratio', 'hr_ml']
    orig_idx = [ws_cols.index(c) for c in orig_cols if c in ws_cols]
    new_idx = [j for j in range(len(ws_cols)) if j not in orig_idx]

    def report(name, nf, h, c):
        print(f"  {name:<40s} {nf:>3d}f  Home R²={h[0]:.4f} MAE={h[1]:.0f}  Clinic R²={c[0]:.4f} MAE={c[1]:.0f}")

    # Original WalkSway
    h = best_alpha(X_ws_h[:, orig_idx], y_v)
    c = best_alpha(X_ws_c_v[:, orig_idx], y_v)
    report('Original WalkSway (10)', len(orig_idx), h, c)

    # Enhanced WalkSway (all)
    h = best_alpha(X_ws_h, y_v)
    c = best_alpha(X_ws_c_v, y_v)
    report(f'Enhanced WalkSway ({len(ws_cols)})', len(ws_cols), h, c)

    # New features only
    h = best_alpha(X_ws_h[:, new_idx], y_v)
    c = best_alpha(X_ws_c_v[:, new_idx], y_v)
    report(f'New features only ({len(new_idx)})', len(new_idx), h, c)

    # + Demo
    h = best_alpha(np.column_stack([X_ws_h[:, orig_idx], X_d5_v]), y_v)
    c = best_alpha(np.column_stack([X_ws_c_v[:, orig_idx], X_d5_v]), y_v)
    report('Original WalkSway + Demo(5)', len(orig_idx) + 5, h, c)

    h = best_alpha(np.column_stack([X_ws_h, X_d5_v]), y_v)
    c = best_alpha(np.column_stack([X_ws_c_v, X_d5_v]), y_v)
    report(f'Enhanced WalkSway + Demo(5)', len(ws_cols) + 5, h, c)

    # ── Per-bout aggregation for home ──
    print(f"\n{'='*90}")
    print(f"PER-BOUT AGGREGATED WalkSway (home only, n={nv})")
    print(f"{'='*90}")

    print("Extracting per-bout WalkSway...")
    perbout_ws = []
    perbout_valid = []

    for i, (_, r) in enumerate(ids_v.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        fp = HOME_DIR / fn
        if not fp.exists():
            perbout_ws.append(None); perbout_valid.append(False); continue
        xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
        bouts = detect_walking_bouts_clinicfree(xyz, FS, min_bout_sec=10, merge_gap_sec=5)

        bout_feats = []
        for s, e in bouts:
            if (e - s) / FS < 15: continue  # need ≥ 15s for sway
            try:
                ap_b, ml_b, vt_b, fs_b = preprocess_raw_to_apmlvt(xyz[s:e], FS)
                ws = extract_enhanced_walksway(ap_b, ml_b, vt_b, fs_b)
                bout_feats.append(ws)
            except:
                pass
            if len(bout_feats) >= 20: break  # cap at 20 bouts

        if len(bout_feats) < 3:
            perbout_ws.append(None); perbout_valid.append(False); continue

        # Aggregate with percentiles
        arr = pd.DataFrame(bout_feats)[ws_cols].values.astype(float)
        agg = {}
        for j, col in enumerate(ws_cols):
            vals = arr[:, j]; valid = vals[np.isfinite(vals)]
            if len(valid) < 2: continue
            agg[f'{col}_med'] = np.median(valid)
            agg[f'{col}_p5'] = np.percentile(valid, 5)
            agg[f'{col}_p95'] = np.percentile(valid, 95)
            agg[f'{col}_std'] = np.std(valid)
        perbout_ws.append(agg); perbout_valid.append(True)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}] {len(bout_feats)} bouts", flush=True)

    perbout_valid = np.array(perbout_valid)
    npb = perbout_valid.sum()
    print(f"  {npb} valid subjects")

    if npb >= 80:
        pb_cols = sorted(perbout_ws[perbout_valid.tolist().index(True)].keys())
        X_pb_ws = impute(pd.DataFrame([w for w, v in zip(perbout_ws, perbout_valid) if v])[pb_cols].values.astype(float))
        y_pb = y[perbout_valid]
        D_pb = X_demo5[perbout_valid]

        h = best_alpha(X_pb_ws, y_pb)
        print(f"  PerBout WalkSway ({len(pb_cols)}f):       R²={h[0]:.4f}  MAE={h[1]:.0f}ft  ρ={h[2]:.3f}")
        h = best_alpha(np.column_stack([X_pb_ws, D_pb]), y_pb)
        print(f"  PerBout WalkSway + Demo(5) ({len(pb_cols)+5}f): R²={h[0]:.4f}  MAE={h[1]:.0f}ft  ρ={h[2]:.3f}")

        # Correlation-selected top features
        corrs = [(j, abs(spearmanr(X_pb_ws[:, j], y_pb)[0])) for j in range(len(pb_cols))]
        corrs.sort(key=lambda x: x[1], reverse=True)
        for K in [10, 15, 20]:
            top_idx = [c[0] for c in corrs[:K]]
            h = best_alpha(np.column_stack([X_pb_ws[:, top_idx], D_pb]), y_pb)
            print(f"  PerBout top-{K} + Demo(5):           R²={h[0]:.4f}  MAE={h[1]:.0f}ft  ρ={h[2]:.3f}")

    print(f"\nDone in {time.time()-t0:.0f}s")
