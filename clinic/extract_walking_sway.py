#!/usr/bin/env python3
"""
Extract WALKING sway features from clinic (6MWT) and home (walking_segments/).
These are gait-related lateral/anterior control features, NOT standing sway.

Walking sway features (10):
  1. ml_range_g        — peak-to-peak ML range (lateral excursion extremes)
  2. ml_path_length    — cumulative |diff(ML)| (total lateral displacement)
  3. ml_jerk_rms       — RMS of ML jerk (smoothness of lateral control)
  4. ap_rms_g          — AP RMS (forward-back dynamic component)
  5. ap_range_g        — peak-to-peak AP range
  6. sway_ellipse_area — 95% confidence ellipse in ML-AP plane
  7. ml_velocity_rms   — RMS of ML velocity (rate of lateral displacement)
  8. stride_ml_cv      — stride-to-stride variability of ML peak amplitude
  9. ml_ap_ratio       — ML_RMS / AP_RMS (lateral vs forward dominance)
 10. hr_ml             — harmonic ratio in ML (gait symmetry in lateral plane)
"""
import numpy as np, pandas as pd, warnings, math
from pathlib import Path
from scipy.signal import welch, find_peaks
from scipy.stats import chi2
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent  # project root
FS = 30.0


def _psd_peak_freq(x, fs, fmin=0.5, fmax=3.5):
    if len(x) < int(fs):
        return float('nan')
    nperseg = int(max(fs * 4, 256))
    freqs, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend='constant')
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return float('nan')
    return float(freqs[band][np.argmax(Pxx[band])])


def _harmonic_ratio_ml(signal, fs, cadence_hz, n_harm=10):
    """For ML: odd harmonics dominate (left-right alternation), ratio = odd/even."""
    if not np.isfinite(cadence_hz) or cadence_hz <= 0:
        return float('nan')
    x = signal - np.mean(signal)
    n = len(x)
    if n < 2:
        return float('nan')
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mags = np.abs(X)
    ev, od = 0.0, 0.0
    for k in range(1, n_harm + 1):
        fk = k * cadence_hz
        if fk >= freqs[-1]:
            break
        idx = int(np.argmin(np.abs(freqs - fk)))
        if k % 2 == 0:
            ev += mags[idx]
        else:
            od += mags[idx]
    return float(od / (ev + 1e-12)) if ev > 0 else float('nan')


def extract_walking_sway(ap, ml, vt, fs=FS):
    """Extract ENMO-normalized walking sway features.

    Raw features are divided by ENMO to remove walking speed confound.
    Higher normalized sway = more instability = worse.

    Returns 10 features:
      5 ENMO-normalized: ml_range, ml_path_length, ml_jerk_rms, ap_rms, ml_velocity_rms
      3 ratio features: sway_ellipse_area/enmo², ml_ap_ratio, stride_ml_cv
      2 spectral: hr_ml, ap_range/enmo
    """
    f = {}

    # Compute ENMO for normalization
    vm = np.sqrt(ap**2 + ml**2 + vt**2)
    enmo = float(np.mean(np.maximum(vm - 1.0, 0.0))) if len(vm) > 0 else 1e-12
    # For preprocessed data (gravity removed), ENMO ≈ mean(VM)
    # Use VM mean as fallback if ENMO is tiny (already gravity-subtracted)
    vm_mean = float(np.mean(vm))
    norm = max(enmo, vm_mean, 1e-12)

    # ENMO-normalized features (higher = more sway per unit effort = worse)
    f['ml_range_norm'] = float(np.ptp(ml)) / norm
    f['ml_path_length_norm'] = float(np.sum(np.abs(np.diff(ml)))) / norm
    ml_vel = np.diff(ml) * fs
    ml_jerk = np.diff(ml_vel) * fs
    f['ml_jerk_rms_norm'] = (float(np.sqrt(np.mean(ml_jerk**2))) / norm) if len(ml_jerk) > 0 else 0.0
    f['ap_rms_norm'] = float(np.sqrt(np.mean(ap**2))) / norm
    f['ap_range_norm'] = float(np.ptp(ap)) / norm

    # Sway ellipse area normalized by norm² (area scales quadratically)
    if len(ml) > 2 and len(ap) > 2:
        cov = np.cov(ml, ap)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0)
        raw_area = float(math.pi * chi2.ppf(0.95, 2) *
                         np.sqrt(eigenvalues[0]) * np.sqrt(eigenvalues[1]))
        f['sway_ellipse_norm'] = raw_area / (norm**2)
    else:
        f['sway_ellipse_norm'] = 0.0

    f['ml_velocity_rms_norm'] = (float(np.sqrt(np.mean(ml_vel**2))) / norm) if len(ml_vel) > 0 else 0.0

    # Ratio features (already intensity-independent)
    cad_hz = _psd_peak_freq(vt, fs)
    if np.isfinite(cad_hz) and cad_hz > 0:
        min_dist = max(1, int(round(0.4 * fs / cad_hz)))
        prom = 0.3 * np.std(ml) if np.std(ml) > 0 else 0.0
        peaks, _ = find_peaks(np.abs(ml), distance=min_dist, prominence=prom)
        if len(peaks) >= 3:
            peak_vals = np.abs(ml[peaks])
            f['stride_ml_cv'] = float(np.std(peak_vals, ddof=1) / (np.mean(peak_vals) + 1e-12))
        else:
            f['stride_ml_cv'] = 0.0
    else:
        f['stride_ml_cv'] = 0.0

    ap_rms = float(np.sqrt(np.mean(ap**2)))
    ml_rms = float(np.sqrt(np.mean(ml**2)))
    f['ml_ap_ratio'] = float(ml_rms / (ap_rms + 1e-12))

    f['hr_ml'] = _harmonic_ratio_ml(ml, fs, cad_hz)

    return f


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score
    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import PLSRegression
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios, extract_cwt

    ids = pd.read_csv('feats/target_6mwd.csv')
    valid = np.ones(len(ids), dtype=bool)
    valid[ids[(ids['cohort']=='M') & (ids['subj_id']==22)].index] = False
    ids102 = ids[valid].reset_index(drop=True)

    PREPROC2 = BASE / 'csv_preprocessed2'
    WALK_SEG = BASE / 'results_raw_pipeline' / 'walking_segments'

    # ── CLINIC: extract from 6MWT preprocessed data ──
    print("Extracting CLINIC walking sway features...")
    clinic_rows = []
    clinic_valid = []
    for _, r in ids102.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        pp = PREPROC2 / fn
        if pp.exists():
            df = pd.read_csv(pp)
            clinic_rows.append(extract_walking_sway(
                df['AP'].values, df['ML'].values, df['VT'].values, FS))
            clinic_valid.append(True)
        else:
            clinic_valid.append(False)
    clinic_valid = np.array(clinic_valid)
    X_clinic_wsway = pd.DataFrame(clinic_rows)
    print(f"  Clinic: {len(clinic_rows)} subjects, {X_clinic_wsway.shape[1]} features")

    # ── HOME: extract from existing walking_segments/ ──
    print("\nExtracting HOME walking sway from results_raw_pipeline/walking_segments/...")
    home_rows = []
    for _, r in ids102.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        wp = WALK_SEG / fn
        if wp.exists():
            df = pd.read_csv(wp)
            home_rows.append(extract_walking_sway(
                df['AP'].values, df['ML'].values, df['VT'].values, FS))
        else:
            home_rows.append(None)

    # Build full home matrix, match to clinic valid
    home_all = []
    for row in home_rows:
        if row is None:
            home_all.append({k: np.nan for k in X_clinic_wsway.columns})
        else:
            home_all.append(row)
    X_home_wsway_full = pd.DataFrame(home_all)

    cidx = np.where(clinic_valid)[0]
    X_home_wsway = X_home_wsway_full.iloc[cidx].reset_index(drop=True)

    # Impute NaN with median
    for c in X_home_wsway.columns:
        if X_home_wsway[c].isna().any():
            X_home_wsway[c] = X_home_wsway[c].fillna(X_home_wsway[c].median())
    for c in X_clinic_wsway.columns:
        if X_clinic_wsway[c].isna().any():
            X_clinic_wsway[c] = X_clinic_wsway[c].fillna(X_clinic_wsway[c].median())

    n_home_valid = X_home_wsway_full.notna().all(axis=1).sum()
    print(f"  Home: {n_home_valid}/{len(ids102)} subjects with walking segments")

    X_cw = X_clinic_wsway.values.astype(float)
    X_hw = X_home_wsway.values.astype(float)
    y = ids102[clinic_valid].reset_index(drop=True)['sixmwd'].values.astype(float)
    groups = ids102[clinic_valid].reset_index(drop=True)['cohort'].values
    n = len(y)
    print(f"\nn={n}")

    # Save
    X_clinic_wsway.to_csv('feats/clinic_walking_sway.csv', index=False)
    X_home_wsway.to_csv('feats/home_walking_sway.csv', index=False)
    print("Saved feats/clinic_walking_sway.csv and feats/home_walking_sway.csv")

    # ── Load existing features ──
    gait_rows = []
    for _, r in ids102[clinic_valid].iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        gait_rows.append(extract_gait10(pd.read_csv(PREPROC2 / fn)))
    vt_rms_df = compute_vt_rms(PREPROC2)
    gdf = pd.DataFrame(gait_rows)
    gm = pd.concat([ids102[clinic_valid].reset_index(drop=True), gdf], axis=1)
    sway = add_sway_ratios(gm.merge(vt_rms_df, on=['cohort','subj_id','sixmwd'], how='left'))
    sway_cols = ['cadence_hz','step_time_cv_pct','acf_step_regularity','hr_ap','hr_vt',
                 'ml_rms_g','ml_spectral_entropy','jerk_mean_abs_gps','enmo_mean_g',
                 'cadence_slope_per_min','vt_rms_g','ml_over_enmo','ml_over_vt']
    X_gait13_c = sway[sway_cols].values.astype(float)
    for j in range(13):
        m = np.isnan(X_gait13_c[:, j])
        if m.any(): X_gait13_c[m, j] = np.nanmedian(X_gait13_c[:, j])

    d = np.load('feats/home_hybrid_v2_features.npz', allow_pickle=True)
    X_gait13_h = d['X_gait'][cidx]

    demo = pd.read_excel('SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids102[clinic_valid].reset_index(drop=True).merge(demo, on=['cohort','subj_id'], how='left')
    p['cohort_M'] = (p['cohort']=='M').astype(int)
    for c in ['Age','Sex','Height']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo_4 = p[['cohort_M','Age','Sex','Height']].values.astype(float)
    X_demo_3 = p[['cohort_M','Age','Sex']].values.astype(float)
    for X in [X_demo_4, X_demo_3]:
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any(): X[m, j] = np.nanmedian(X[:, j])

    RAW = BASE / 'csv_raw2'
    cwt_rows = []
    for _, r in ids102[clinic_valid].iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        raw = pd.read_csv(RAW/fn, usecols=['X','Y','Z']).values.astype(np.float32)
        cwt_rows.append(extract_cwt(raw))
    cwt_df = pd.DataFrame(cwt_rows).replace([np.inf,-np.inf], np.nan)
    for c in cwt_df.columns:
        if cwt_df[c].isna().any(): cwt_df[c] = cwt_df[c].fillna(cwt_df[c].median())
    X_cwt_c = cwt_df.values.astype(float)

    home_cwt_df = pd.read_csv('feats/home_cwt_hybrid.csv')
    hcwt_cols = [c for c in home_cwt_df.columns if c not in ['cohort','subj_id','year','sixmwd']]
    home_cwt_merged = ids102[clinic_valid].reset_index(drop=True).merge(
        home_cwt_df, on=['cohort','subj_id','sixmwd'], how='inner')
    X_cwt_h = home_cwt_merged[hcwt_cols].values.astype(float)
    for j in range(X_cwt_h.shape[1]):
        m = np.isnan(X_cwt_h[:, j])
        if m.any(): X_cwt_h[m, j] = np.nanmedian(X_cwt_h[:, j])

    # ── LOO EVALUATION ──
    def loo(X, y):
        pr = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=10)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            pr[te] = m.predict(sc.transform(X[te]))
        return pr

    def report(name, y_true, y_pred, grp):
        ms = grp == 'M'; ctrl = grp == 'C'
        r2 = r2_score(y_true, y_pred)
        r2_ms = r2_score(y_true[ms], y_pred[ms])
        r2_ctrl = r2_score(y_true[ctrl], y_pred[ctrl])
        print(f"  {name:50s}  All={r2:.4f}  MS={r2_ms:.4f}  Ctrl={r2_ctrl:.4f}")

    print("\n" + "="*95)
    print("WALKING SWAY FEATURES — LOO Results (Ridge α=10)")
    print("="*95)

    # Walking sway alone
    pr = loo(X_cw, y); report("Clinic: WalkSway (10f)", y, pr, groups)
    pr = loo(X_hw, y); report("Home: WalkSway (10f)", y, pr, groups)

    print("\n--- Gait13 + WalkSway ---")
    pr = loo(np.column_stack([X_gait13_c, X_cw]), y)
    report("Clinic: Gait13+WalkSway (23f)", y, pr, groups)
    pr = loo(np.column_stack([X_gait13_h, X_hw]), y)
    report("Home: Gait13+WalkSway (23f)", y, pr, groups)

    print("\n--- Gait13 + WalkSway + Demo ---")
    pr = loo(np.column_stack([X_gait13_c, X_cw, X_demo_4]), y)
    report("Clinic: Gait13+WalkSway+Demo (27f)", y, pr, groups)
    pr = loo(np.column_stack([X_gait13_h, X_hw, X_demo_3]), y)
    report("Home: Gait13+WalkSway+Demo (26f)", y, pr, groups)

    print("\n--- Gait13 + CWT + WalkSway + Demo ---")
    pr = loo(np.column_stack([X_gait13_c, X_cwt_c, X_cw, X_demo_4]), y)
    report("Clinic: Gait13+CWT+WalkSway+Demo (55f)", y, pr, groups)
    pr = loo(np.column_stack([X_gait13_h, X_cwt_h, X_hw, X_demo_3]), y)
    report("Home: Gait13+CWT+WalkSway+Demo (54f)", y, pr, groups)

    # PLS with walking sway
    print("\n--- PLS: Home → Clinic + Demo ---")
    for label, Xh, Xc in [
        ("WalkSway", X_hw, X_cw),
        ("Gait13+WalkSway", np.column_stack([X_gait13_h, X_hw]),
                             np.column_stack([X_gait13_c, X_cw])),
    ]:
        for nc in [2, 3]:
            pr = np.zeros(n)
            for te in range(n):
                tr = np.ones(n, dtype=bool); tr[te] = False
                sh = StandardScaler(); sc = StandardScaler(); sd = StandardScaler()
                Xht = sh.fit_transform(Xh[tr]); Xhe = sh.transform(Xh[te:te+1])
                Xct = sc.fit_transform(Xc[tr])
                pls = PLSRegression(n_components=nc, scale=False); pls.fit(Xht, Xct)
                Xhm = pls.transform(Xht); Xhem = pls.transform(Xhe)
                Xdt = sd.fit_transform(X_demo_3[tr]); Xde = sd.transform(X_demo_3[te:te+1])
                Xf = np.column_stack([Xhm, Xdt]); Xfe = np.column_stack([Xhem, Xde])
                m = Ridge(alpha=10); m.fit(Xf, y[tr]); pr[te] = m.predict(Xfe)[0]
            report(f"Home PLS({nc})({label}→same)+Demo ({nc+3}f)", y, pr, groups)

    print("\n--- Reference baselines ---")
    pr = loo(np.column_stack([X_gait13_c, X_cwt_c, X_demo_4]), y)
    report("Clinic: Gait13+CWT+Demo (45f) [BEST=0.7923]", y, pr, groups)

    pr = np.zeros(n)
    for te in range(n):
        tr = np.ones(n, dtype=bool); tr[te] = False
        sh = StandardScaler(); sc = StandardScaler(); sd = StandardScaler()
        Xht = sh.fit_transform(X_gait13_h[tr]); Xhe = sh.transform(X_gait13_h[te:te+1])
        Xct = sc.fit_transform(X_gait13_c[tr])
        pls = PLSRegression(n_components=2, scale=False); pls.fit(Xht, Xct)
        Xhm = pls.transform(Xht); Xhem = pls.transform(Xhe)
        Xdt = sd.fit_transform(X_demo_3[tr]); Xde = sd.transform(X_demo_3[te:te+1])
        Xf = np.column_stack([Xhm, Xdt]); Xfe = np.column_stack([Xhem, Xde])
        m = Ridge(alpha=10); m.fit(Xf, y[tr]); pr[te] = m.predict(Xfe)[0]
    report("Home: PLS(2)(Gait13→Gait13)+Demo (5f) [BEST=0.483]", y, pr, groups)
