#!/usr/bin/env python3
"""
Clinic-free home 6MWD prediction.
No clinic data used anywhere — no PLS, no cosine similarity to clinic signature.
All outputs in temporary_experiments/clinic_free/.

Step 0: Feature extraction + correlation screen
Exp 1-7: Model building with correlation-selected features
"""
import sys, warnings, time, math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import pearsonr, spearmanr, skew, kurtosis

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

FS = 30
HOME_DIR = BASE / 'csv_home_daytime'
WALK_SEG_DIR = BASE / 'results_raw_pipeline' / 'walking_segments'


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def eval_loo(X, y, alpha=20):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, pr), mean_absolute_error(y, pr), pearsonr(y, pr)[0], spearmanr(y, pr)[0]

def best_alpha(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 0, 20)
    for a in alphas:
        r2, mae, rv, rho = eval_loo(X, y, a)
        if r2 > best[0]: best = (r2, mae, rv, rho, a)
    return best

def report(name, nf, r2, mae, rv, rho, alpha):
    print(f"  {name:50s} {nf:>3d}f  α={alpha:>3d}  R²={r2:.4f}  MAE={mae:.0f}ft  r={rv:.3f}  ρ={rho:.3f}")

def _rodrigues(axis, theta):
    ax = axis / (np.linalg.norm(axis)+1e-12)
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*(K@K)

def preprocess_segment(xyz, fs=30.0):
    """Preprocess a walking segment: gravity removal, Rodrigues, PCA, bandpass."""
    arr = xyz.copy()
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
    vm_raw = np.linalg.norm(arr, axis=1)
    enmo = np.maximum(vm_raw - 1.0, 0.0)
    return apmlvt, apmlvt_bp, enmo, vm_raw


def extract_segment_features(xyz, fs=30.0):
    """Extract rich gait features from a single walking segment. Returns dict or None."""
    if len(xyz) < int(10 * fs):  # need at least 10s
        return None
    try:
        apmlvt, apmlvt_bp, enmo, vm_raw = preprocess_segment(xyz, fs)
    except:
        return None

    ap, ml, vt = apmlvt[:,0], apmlvt[:,1], apmlvt[:,2]
    ap_bp, ml_bp, vt_bp = apmlvt_bp[:,0], apmlvt_bp[:,1], apmlvt_bp[:,2]
    vm_dyn = np.linalg.norm(apmlvt, axis=1)

    # Cadence
    nperseg = min(len(vt_bp), int(fs*4))
    if nperseg < int(fs): return None
    freqs, Pxx = welch(vt_bp, fs=fs, nperseg=max(nperseg, 256), noverlap=nperseg//2, detrend="constant")
    band = (freqs >= 0.5) & (freqs <= 3.5)
    if not np.any(band): return None
    cad = float(freqs[band][np.argmax(Pxx[band])])
    if cad < 1.0:  # cadence filter — reject non-walking
        return None
    cad_power = float(Pxx[band].max())  # strength of cadence peak

    f = {}
    f['cadence_hz'] = cad
    f['cadence_power'] = cad_power

    # Step regularity via ACF
    lag = max(1, min(int(round(fs/cad)), len(vt_bp)-1))
    x = vt_bp - vt_bp.mean(); d = np.dot(x, x)
    f['acf_step_reg'] = float(np.dot(x[:len(x)-lag], x[lag:]) / (d+1e-12)) if d > 0 else 0

    # Harmonic ratios
    def _hr(sig, cad_f):
        x = sig - sig.mean()
        if len(x) < 2: return np.nan
        X = np.fft.rfft(x); fr = np.fft.rfftfreq(len(x), d=1.0/fs); mags = np.abs(X)
        ev, od = 0.0, 0.0
        for k in range(1, 11):
            fk = k*cad_f
            if fk >= fr[-1]: break
            idx = int(np.argmin(np.abs(fr - fk)))
            if k%2==0: ev += mags[idx]
            else: od += mags[idx]
        return float(ev/od) if od > 0 else np.nan
    f['hr_ap'] = _hr(ap_bp, cad)
    f['hr_vt'] = _hr(vt_bp, cad)
    f['hr_ml'] = _hr(ml_bp, cad)

    # Step timing
    min_dist = max(1, int(round(0.5*fs/cad)))
    prom = 0.5*np.std(vt_bp) if np.std(vt_bp) > 0 else 0
    peaks, _ = find_peaks(vt_bp, distance=min_dist, prominence=prom)
    if peaks.size >= 3:
        si = np.diff(peaks)/fs
        f['stride_time_mean'] = float(np.mean(si))
        f['stride_time_std'] = float(np.std(si, ddof=1))
        f['stride_time_cv'] = float(np.std(si, ddof=1)/np.mean(si)) if np.mean(si) > 0 else np.nan
    else:
        f['stride_time_mean'] = np.nan; f['stride_time_std'] = np.nan; f['stride_time_cv'] = np.nan

    # Amplitude features
    f['ml_rms'] = float(np.sqrt(np.mean(ml**2)))
    f['vt_rms'] = float(np.sqrt(np.mean(vt**2)))
    f['ap_rms'] = float(np.sqrt(np.mean(ap**2)))
    f['enmo_mean'] = float(np.mean(enmo))
    f['enmo_p95'] = float(np.percentile(enmo, 95))
    f['vm_std'] = float(np.std(vm_dyn))
    f['vt_range'] = float(np.ptp(vt))
    f['ml_range'] = float(np.ptp(ml))

    # Jerk
    f['jerk_mean'] = float(np.mean(np.abs(np.diff(vm_dyn)*fs)))

    # Signal energy
    f['signal_energy'] = float(np.mean(vm_dyn**2))

    # Bout duration
    f['duration_sec'] = len(xyz) / fs

    return f


def detect_walking_bouts_clinicfree(xyz, fs, min_bout_sec=10, merge_gap_sec=5):
    """Detect walking bouts without clinic data. Merge adjacent bouts within gap threshold."""
    # Stage 1: ENMO thresholding
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs); n_secs = len(enmo) // sec
    if n_secs < min_bout_sec: return []
    enmo_sec = enmo[:n_secs*sec].reshape(n_secs, sec).mean(axis=1)
    active = enmo_sec >= 0.015

    # Find raw bouts (relaxed: min 10s)
    raw_bouts = []
    in_b, bs = False, 0
    for s in range(n_secs):
        if active[s] and not in_b: bs = s; in_b = True
        elif not active[s] and in_b:
            if s - bs >= min_bout_sec: raw_bouts.append((bs*sec, s*sec))
            in_b = False
    if in_b and n_secs - bs >= min_bout_sec: raw_bouts.append((bs*sec, n_secs*sec))

    if not raw_bouts: return []

    # Stage 2: HR refinement (relaxed min bout to 10s)
    b_filt, a_filt = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b_filt, a_filt, vm - vm.mean())
    win = int(10*fs); step = int(10*fs)
    fft_freqs = np.fft.rfftfreq(win, d=1.0/fs)
    band = (fft_freqs >= 0.8) & (fft_freqs <= 3.5)

    refined = []
    for bout_s, bout_e in raw_bouts:
        walking_wins = []
        for wi in range(bout_s, bout_e - win, step):
            seg = vm_bp[wi:wi+win]
            X = np.fft.rfft(seg); mags = np.abs(X)
            if not np.any(band): continue
            cadence = fft_freqs[band][np.argmax(mags[band])]
            even, odd = 0.0, 0.0
            for k in range(1, 11):
                fk = k * cadence
                if fk >= fft_freqs[-1]: break
                idx = int(np.argmin(np.abs(fft_freqs - fk)))
                if k%2==0: even += mags[idx]
                else: odd += mags[idx]
            hr = even / (odd+1e-12) if odd > 0 else 0
            if hr >= 0.2: walking_wins.append((wi, wi+win))

        if walking_wins:
            cs, ce = walking_wins[0]
            for ws, we in walking_wins[1:]:
                if ws <= ce + step: ce = max(ce, we)
                else:
                    if ce - cs >= min_bout_sec*fs: refined.append((cs, ce))
                    cs, ce = ws, we
            if ce - cs >= min_bout_sec*fs: refined.append((cs, ce))
        else:
            if (bout_e - bout_s) >= min_bout_sec*fs:
                refined.append((bout_s, bout_e))

    if not refined: return []

    # Stage 3: Merge adjacent bouts (gap < merge_gap_sec seconds)
    merged = [refined[0]]
    for s, e in refined[1:]:
        prev_s, prev_e = merged[-1]
        gap_sec = (s - prev_e) / fs
        if gap_sec <= merge_gap_sec:
            merged[-1] = (prev_s, e)  # merge
        else:
            merged.append((s, e))

    return merged


def extract_activity_features(xyz, fs):
    """Extract whole-recording activity features (no walking detection needed)."""
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    n = len(enmo)
    sec = int(fs); n_secs = n // sec
    if n_secs < 60: return None
    enmo_sec = enmo[:n_secs*sec].reshape(n_secs, sec).mean(axis=1)

    f = {}
    # ENMO distribution
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
    # Entropy
    hist, _ = np.histogram(enmo_sec, bins=20, density=True)
    hist = hist[hist > 0]; hist = hist / hist.sum()
    f['act_enmo_entropy'] = -np.sum(hist * np.log2(hist + 1e-12))

    # Intensity zones
    f['act_pct_sedentary'] = np.mean(enmo_sec < 0.02)
    f['act_pct_light'] = np.mean((enmo_sec >= 0.02) & (enmo_sec < 0.06))
    f['act_pct_moderate'] = np.mean((enmo_sec >= 0.06) & (enmo_sec < 0.1))
    f['act_pct_vigorous'] = np.mean(enmo_sec >= 0.1)
    total_hours = n_secs / 3600
    f['act_mvpa_min_per_hr'] = (np.sum(enmo_sec >= 0.06) / 60) / (total_hours + 1e-12)

    # Active bout statistics (any activity, not just walking)
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

    # Transition probabilities
    tas, tsa, ac, sc = 0, 0, 0, 0
    for j in range(len(active)-1):
        if active[j]: ac += 1; tas += (not active[j+1])
        else: sc += 1; tsa += active[j+1]
    f['act_astp'] = tas / (ac + 1e-12)
    f['act_satp'] = tsa / (sc + 1e-12)
    f['act_fragmentation'] = f['act_astp'] + f['act_satp']

    # Temporal patterns (approximate: split recording into thirds)
    third = n_secs // 3
    if third > 60:
        f['act_early_enmo'] = np.mean(enmo_sec[:third])
        f['act_mid_enmo'] = np.mean(enmo_sec[third:2*third])
        f['act_late_enmo'] = np.mean(enmo_sec[2*third:])
        f['act_early_late_ratio'] = f['act_early_enmo'] / (f['act_late_enmo'] + 1e-12)

    # Daily consistency (approximate: split by ~15h chunks = 54000 seconds)
    day_len = 15 * 3600  # assume ~15h daytime
    n_days = max(1, n_secs // day_len)
    if n_days >= 2:
        daily_means = [np.mean(enmo_sec[d*day_len:(d+1)*day_len])
                       for d in range(n_days) if (d+1)*day_len <= n_secs]
        if len(daily_means) >= 2:
            f['act_daily_cv'] = np.std(daily_means) / (np.mean(daily_means) + 1e-12)
        else:
            f['act_daily_cv'] = 0
    else:
        f['act_daily_cv'] = 0

    return f


# ══════════════════════════════════════════════════════════════════
# STEP 0: FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════
def step0_extract_all(ids101, y):
    cache = OUT / 'feature_cache.npz'
    if cache.exists():
        print("Step 0: Loading cached features...", flush=True)
        d = np.load(cache, allow_pickle=True)
        return d['gait_features'].item(), d['activity_features'].item(), d['secondary_features'].item()

    print("Step 0: Extracting all features (clinic-free)...", flush=True)
    t0 = time.time()

    all_gait = {}      # per-subject: list of per-bout feature dicts
    all_activity = {}   # per-subject: single feature dict
    all_secondary = {}  # per-subject: features from secondary pipeline

    for i, (_, r) in enumerate(ids101.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        subj_key = fn

        # ── 0A: Per-bout gait features ──
        fp = HOME_DIR / fn
        if fp.exists():
            xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
            bouts = detect_walking_bouts_clinicfree(xyz, FS, min_bout_sec=10, merge_gap_sec=5)
            bout_feats = []
            for s, e in bouts:
                seg_xyz = xyz[s:e]
                feats = extract_segment_features(seg_xyz, FS)
                if feats is not None:
                    bout_feats.append(feats)
            all_gait[subj_key] = bout_feats
        else:
            all_gait[subj_key] = []

        # ── 0B: Whole-recording activity features ──
        if fp.exists():
            if 'xyz' not in dir() or len(xyz) == 0:
                xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
            all_activity[subj_key] = extract_activity_features(xyz, FS)
        else:
            all_activity[subj_key] = None

        # ── 0C: Secondary pipeline features ──
        wp = WALK_SEG_DIR / fn
        if wp.exists():
            wdf = pd.read_csv(wp)
            if 'AP' in wdf.columns and 'ML' in wdf.columns and 'VT' in wdf.columns:
                seg_xyz = wdf[['AP', 'ML', 'VT']].values.astype(np.float64)
                # This is already preprocessed — extract features directly
                # Split into segments by gaps in the data (>1s gap = new segment)
                # For simplicity, extract features from the whole concatenated segment
                vm = np.linalg.norm(seg_xyz, axis=1)
                enmo_proxy = np.maximum(vm, 0)
                sec_feats = {}
                sec_feats['sec_ml_rms'] = float(np.sqrt(np.mean(seg_xyz[:,1]**2)))
                sec_feats['sec_vt_rms'] = float(np.sqrt(np.mean(seg_xyz[:,2]**2)))
                sec_feats['sec_ap_rms'] = float(np.sqrt(np.mean(seg_xyz[:,0]**2)))
                sec_feats['sec_vm_mean'] = float(np.mean(vm))
                sec_feats['sec_vm_std'] = float(np.std(vm))
                sec_feats['sec_jerk'] = float(np.mean(np.abs(np.diff(vm)*FS)))
                sec_feats['sec_duration'] = len(seg_xyz) / FS
                # Cadence from VT
                if len(seg_xyz) > 256:
                    freqs, Pxx = welch(seg_xyz[:,2], fs=FS, nperseg=min(len(seg_xyz), 256))
                    bd = (freqs >= 0.5) & (freqs <= 3.5)
                    if np.any(bd):
                        sec_feats['sec_cadence'] = float(freqs[bd][np.argmax(Pxx[bd])])
                all_secondary[subj_key] = sec_feats
            else:
                all_secondary[subj_key] = None
        else:
            all_secondary[subj_key] = None

        if (i+1) % 20 == 0:
            n_bouts = len(all_gait[subj_key])
            print(f"  [{i+1}/{len(ids101)}] {fn}: {n_bouts} valid walking segments", flush=True)

    np.savez(cache, gait_features=all_gait, activity_features=all_activity,
             secondary_features=all_secondary, allow_pickle=True)
    print(f"  Cached in {time.time()-t0:.0f}s", flush=True)
    return all_gait, all_activity, all_secondary


def step0d_correlation_screen(ids101, y, all_gait, all_activity, all_secondary):
    """Build feature matrix from all sources, compute correlations with 6MWD."""
    print("\nStep 0D: Correlation screen...", flush=True)

    # ── Aggregate per-bout gait features ──
    gait_feat_names = None
    gait_agg_rows = []
    for _, r in ids101.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        bout_feats = all_gait.get(fn, [])
        if not bout_feats:
            gait_agg_rows.append(None)
            continue
        if gait_feat_names is None:
            gait_feat_names = sorted(bout_feats[0].keys())

        arr = np.array([[bf.get(k, np.nan) for k in gait_feat_names] for bf in bout_feats])
        agg = {}
        for j, name in enumerate(gait_feat_names):
            col = arr[:, j]
            valid = col[np.isfinite(col)]
            if len(valid) < 2:
                for stat in ['_med', '_iqr', '_p10', '_p90', '_max', '_cv']:
                    agg[f'g_{name}{stat}'] = np.nan
                continue
            agg[f'g_{name}_med'] = np.median(valid)
            agg[f'g_{name}_iqr'] = np.percentile(valid, 75) - np.percentile(valid, 25)
            agg[f'g_{name}_p10'] = np.percentile(valid, 10)
            agg[f'g_{name}_p90'] = np.percentile(valid, 90)
            agg[f'g_{name}_max'] = np.max(valid)
            agg[f'g_{name}_cv'] = np.std(valid) / (np.mean(valid) + 1e-12)

        # Meta-features
        agg['g_n_valid_bouts'] = len(bout_feats)
        agg['g_total_walk_sec'] = sum(bf.get('duration_sec', 0) for bf in bout_feats)
        durs = [bf.get('duration_sec', 0) for bf in bout_feats]
        agg['g_mean_bout_dur'] = np.mean(durs)
        agg['g_bout_dur_cv'] = np.std(durs) / (np.mean(durs) + 1e-12) if np.mean(durs) > 0 else 0
        gait_agg_rows.append(agg)

    # ── Activity features ──
    act_rows = []
    for _, r in ids101.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        act_rows.append(all_activity.get(fn))

    # ── Secondary features ──
    sec_rows = []
    for _, r in ids101.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        sec_rows.append(all_secondary.get(fn))

    # ── Build unified feature matrix ──
    all_feat_dicts = []
    for i in range(len(ids101)):
        combined = {}
        if gait_agg_rows[i]: combined.update(gait_agg_rows[i])
        if act_rows[i]: combined.update(act_rows[i])
        if sec_rows[i]: combined.update(sec_rows[i])
        all_feat_dicts.append(combined)

    # Get all feature names
    all_names = set()
    for d in all_feat_dicts:
        all_names.update(d.keys())
    all_names = sorted(all_names)

    # Build matrix
    X_all = np.full((len(ids101), len(all_names)), np.nan)
    for i, d in enumerate(all_feat_dicts):
        for j, name in enumerate(all_names):
            if name in d and d[name] is not None:
                X_all[i, j] = d[name]

    # Impute NaN with median
    for j in range(X_all.shape[1]):
        col = X_all[:, j]
        m = np.isnan(col)
        if m.all():
            X_all[:, j] = 0
        elif m.any():
            X_all[m, j] = np.nanmedian(col)

    # ── Correlation screen ──
    corrs = []
    for j, name in enumerate(all_names):
        col = X_all[:, j]
        if np.std(col) < 1e-12: continue
        rho, p = spearmanr(col, y)
        if np.isfinite(rho):
            corrs.append({'feature': name, 'rho': rho, 'abs_rho': abs(rho), 'p': p})

    corr_df = pd.DataFrame(corrs).sort_values('abs_rho', ascending=False)
    corr_df.to_csv(OUT / 'feature_correlations.csv', index=False)

    print(f"\n  Total features extracted: {len(all_names)}")
    print(f"  Features with |ρ| > 0.2: {(corr_df['abs_rho'] > 0.2).sum()}")
    print(f"  Features with |ρ| > 0.3: {(corr_df['abs_rho'] > 0.3).sum()}")
    print(f"  Features with |ρ| > 0.4: {(corr_df['abs_rho'] > 0.4).sum()}")
    print(f"\n  Top 30 features by |ρ| with 6MWD:")
    for _, row in corr_df.head(30).iterrows():
        print(f"    {row['feature']:45s}  ρ={row['rho']:+.3f}  p={row['p']:.4f}")

    return X_all, all_names, corr_df


# ══════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════
def run_experiments(X_all, all_names, corr_df, y, X_demo_5, X_demo_3):
    n = len(y)
    log_lines = []

    def log(line):
        log_lines.append(line)
        print(line)

    log(f"\n{'='*100}")
    log(f"CLINIC-FREE EXPERIMENTS (n={n})")
    log(f"{'='*100}")

    # Baseline
    log(f"\nBASELINES:")
    log(f"  Clinic-informed Gait(8)+Demo(5), α=20:  R²=0.507  MAE=174ft")
    log(f"  Previous clinic-free best (ENMO):        R²=0.381  MAE=199ft")

    # ── Exp 1: Top-K correlated features ──
    log(f"\n{'─'*100}")
    log(f"EXP 1: Top-K correlated features + Demo(5)")
    log(f"{'─'*100}")
    sig_feats = corr_df[corr_df['abs_rho'] > 0.2]['feature'].tolist()

    for K in [5, 8, 10, 15, 20]:
        top_k = corr_df.head(K)['feature'].tolist()
        idx = [all_names.index(f) for f in top_k if f in all_names]
        if not idx: continue
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"Top-{K} features + Demo(5)", K+5, r2, mae, rv, rho, a)
        log(f"  Top-{K}+Demo(5): {K+5}f α={a} R²={r2:.4f} MAE={mae:.0f}")

    # ── Exp 2: Activity-only ──
    log(f"\n{'─'*100}")
    log(f"EXP 2: Activity-only features (no walking detection)")
    log(f"{'─'*100}")
    act_feats = [f for f in all_names if f.startswith('act_')]
    act_sig = [f for f in act_feats if f in sig_feats]
    if act_sig:
        idx = [all_names.index(f) for f in act_sig]
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"Activity(sig={len(act_sig)})+Demo(5)", len(act_sig)+5, r2, mae, rv, rho, a)
        log(f"  Activity({len(act_sig)})+Demo(5): R²={r2:.4f} MAE={mae:.0f}")

    # All activity features
    if act_feats:
        idx = [all_names.index(f) for f in act_feats]
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"Activity(all={len(act_feats)})+Demo(5)", len(act_feats)+5, r2, mae, rv, rho, a)
        log(f"  Activity(all={len(act_feats)})+Demo(5): R²={r2:.4f} MAE={mae:.0f}")

    # ── Exp 3: Per-bout gait aggregated ──
    log(f"\n{'─'*100}")
    log(f"EXP 3: Per-bout aggregated gait features")
    log(f"{'─'*100}")
    gait_feats = [f for f in all_names if f.startswith('g_')]
    gait_sig = [f for f in gait_feats if f in sig_feats]
    if gait_sig:
        idx = [all_names.index(f) for f in gait_sig]
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"Gait-agg(sig={len(gait_sig)})+Demo(5)", len(gait_sig)+5, r2, mae, rv, rho, a)
        log(f"  Gait-agg({len(gait_sig)})+Demo(5): R²={r2:.4f} MAE={mae:.0f}")

    # Top gait features by correlation
    gait_corr = corr_df[corr_df['feature'].isin(gait_feats)].head(10)
    if len(gait_corr) > 0:
        idx = [all_names.index(f) for f in gait_corr['feature']]
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"Gait-top10+Demo(5)", len(idx)+5, r2, mae, rv, rho, a)
        log(f"  Gait-top10+Demo(5): R²={r2:.4f} MAE={mae:.0f}")

    # ── Exp 4: Secondary pipeline ──
    log(f"\n{'─'*100}")
    log(f"EXP 4: Secondary pipeline features")
    log(f"{'─'*100}")
    sec_feats = [f for f in all_names if f.startswith('sec_')]
    sec_sig = [f for f in sec_feats if f in sig_feats]
    if sec_sig:
        idx = [all_names.index(f) for f in sec_sig]
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"Secondary(sig={len(sec_sig)})+Demo(5)", len(sec_sig)+5, r2, mae, rv, rho, a)
        log(f"  Secondary({len(sec_sig)})+Demo(5): R²={r2:.4f} MAE={mae:.0f}")

    # ── Exp 5: Hybrid — best of each category ──
    log(f"\n{'─'*100}")
    log(f"EXP 5: Hybrid combinations (correlation-selected)")
    log(f"{'─'*100}")

    # Gait + Activity (both sig)
    hybrid1 = [f for f in sig_feats if f.startswith('g_') or f.startswith('act_')]
    if hybrid1:
        idx = [all_names.index(f) for f in hybrid1]
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"Gait+Activity(sig={len(hybrid1)})+Demo(5)", len(hybrid1)+5, r2, mae, rv, rho, a)
        log(f"  Gait+Activity({len(hybrid1)})+Demo(5): R²={r2:.4f} MAE={mae:.0f}")

    # All significant features
    if sig_feats:
        idx = [all_names.index(f) for f in sig_feats]
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"All-sig({len(sig_feats)})+Demo(5)", len(sig_feats)+5, r2, mae, rv, rho, a)
        log(f"  All-sig({len(sig_feats)})+Demo(5): R²={r2:.4f} MAE={mae:.0f}")

    # Top features capped at reasonable count
    for K in [8, 12, 15]:
        top = corr_df.head(K)['feature'].tolist()
        idx = [all_names.index(f) for f in top]
        X = np.column_stack([X_all[:, idx], X_demo_5])
        r2, mae, rv, rho, a = best_alpha(X, y)
        report(f"Top-{K}+Demo(5) [hybrid]", K+5, r2, mae, rv, rho, a)
        log(f"  Top-{K}+Demo(5): R²={r2:.4f} MAE={mae:.0f}")

    # ── Exp 6: Demo(3) instead of Demo(5) ──
    log(f"\n{'─'*100}")
    log(f"EXP 6: Best configs with Demo(3) vs Demo(5)")
    log(f"{'─'*100}")
    for K in [8, 10, 12]:
        top = corr_df.head(K)['feature'].tolist()
        idx = [all_names.index(f) for f in top]
        X3 = np.column_stack([X_all[:, idx], X_demo_3])
        X5 = np.column_stack([X_all[:, idx], X_demo_5])
        r2_3, mae_3, rv_3, rho_3, a_3 = best_alpha(X3, y)
        r2_5, mae_5, rv_5, rho_5, a_5 = best_alpha(X5, y)
        report(f"Top-{K}+Demo(3)", K+3, r2_3, mae_3, rv_3, rho_3, a_3)
        report(f"Top-{K}+Demo(5)", K+5, r2_5, mae_5, rv_5, rho_5, a_5)
        log(f"  Top-{K}+Demo(3): R²={r2_3:.4f} | Top-{K}+Demo(5): R²={r2_5:.4f}")

    # Save experiment log
    with open(OUT / 'experiment_log.txt', 'w') as f:
        f.write("CLINIC-FREE EXPERIMENT LOG\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("="*80 + "\n\n")
        for line in log_lines:
            f.write(line + '\n')
    print(f"\nSaved experiment_log.txt")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()

    # Load subject list
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)

    # Load targets (match to clinic-valid as in main pipeline)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = []
    for _, r in ids101.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        clinic_valid.append((PREPROC2 / fn).exists())
    clinic_valid = np.array(clinic_valid)
    ids101 = ids101[clinic_valid].reset_index(drop=True)
    y = ids101['sixmwd'].values.astype(float)

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids101.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo_5 = p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float)
    X_demo_3 = p[['cohort_M', 'Age', 'Sex']].values.astype(float)
    for X in [X_demo_5, X_demo_3]:
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any(): X[m, j] = np.nanmedian(X[:, j])

    # Step 0: Extract + correlate
    all_gait, all_activity, all_secondary = step0_extract_all(ids101, y)
    X_all, all_names, corr_df = step0d_correlation_screen(ids101, y, all_gait, all_activity, all_secondary)

    # Experiments
    run_experiments(X_all, all_names, corr_df, y, X_demo_5, X_demo_3)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
