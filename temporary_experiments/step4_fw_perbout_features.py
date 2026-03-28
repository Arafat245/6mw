#!/usr/bin/env python3
"""
Step 4: Use find_walking CWT bouts to extract per-bout gait features,
then combine ALL feature sets and forward-select.

1. find_walking -> CWT walking bout boundaries
2. Extract same per-bout gait features as step1 from CWT bouts
3. Aggregate with 6 stats (median, IQR, p10, p90, max, CV)
4. Combine: original per-bout (ENMO+HR) + CWT per-bout + fw summary + demo
5. Forward selection
6. Save CWT walking bouts for later use
"""
import os, re, math, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.signal.windows import tukey
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent.parent
NPZ_DIR = BASE / 'csv_home_daytime_npz'
FS = 30


# ══════════════════════════════════════════════════════════════════
# FIND_WALKING (from step3, copied here for self-containedness)
# ══════════════════════════════════════════════════════════════════

def find_walking(vm, fs, min_amp=0.3, T=3, delta=2, alpha=0.6, beta=2.5,
                 step_freq=(1.4, 2.3)):
    n_samples = len(vm)
    sec = int(fs)
    n_secs = n_samples // sec
    if n_secs < 5:
        return np.zeros(n_secs), 0, np.zeros(n_secs)

    vm_mat = vm[:n_secs * sec].reshape(n_secs, sec)
    p2p = vm_mat.max(axis=1) - vm_mat.min(axis=1)
    valid = p2p >= min_amp

    freq_grid = np.arange(0.5, 4.05, 0.05)
    n_freqs = len(freq_grid)
    loc1 = np.searchsorted(freq_grid, step_freq[0])
    loc2 = np.searchsorted(freq_grid, step_freq[1])

    fft_freqs = np.fft.rfftfreq(3 * sec, d=1.0 / fs)
    f_interp_indices = np.searchsorted(fft_freqs, freq_grid)
    f_interp_indices = np.clip(f_interp_indices, 0, len(fft_freqs) - 1)

    detected_freq = np.full(n_secs, np.nan)
    peak_matrix = np.zeros((n_freqs, n_secs), dtype=bool)
    tukey_win = tukey(3 * sec, alpha=0.02)

    for s in range(n_secs):
        if not valid[s]:
            continue
        s_start = max(0, s - 1)
        s_end = min(n_secs, s + 2)
        if s_end - s_start < 3:
            s_start = max(0, s_end - 3)
        seg = vm[s_start * sec:s_end * sec]
        win_len = 3 * sec
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len - len(seg)), mode='edge')
        seg = seg[:win_len]
        seg_win = (seg - seg.mean()) * tukey_win
        fft_vals = np.fft.rfft(seg_win)
        power_full = np.abs(fft_vals) ** 2
        power = power_full[f_interp_indices]
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
            detected_freq[s] = freq_grid[best_idx]
            peak_matrix[best_idx, s] = True

    final_freq = np.full(n_secs, np.nan)
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
                    for f_idx in freqs_in_window[t]:
                        final_freq[s + t] = freq_grid[f_idx]

    wi = ~np.isnan(final_freq)
    cad = np.where(wi, final_freq, 0)
    steps = np.sum(cad)
    return wi.astype(float), steps, cad


def fw_get_bouts(wi, fs, min_bout_sec=10):
    """Convert per-second walking indication to bout sample indices, min 10s."""
    sec = int(fs)
    bouts = []
    in_b, bs = False, 0
    for s in range(len(wi)):
        if wi[s] and not in_b:
            bs = s; in_b = True
        elif not wi[s] and in_b:
            if s - bs >= min_bout_sec:
                bouts.append((bs * sec, s * sec))
            in_b = False
    if in_b and len(wi) - bs >= min_bout_sec:
        bouts.append((bs * sec, len(wi) * sec))
    return bouts


# ══════════════════════════════════════════════════════════════════
# PER-BOUT GAIT FEATURES (same as step1, imported logic)
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


def aggregate_bout_features(bout_feats, prefix='cg_'):
    """Aggregate per-bout features with 6 robust stats. prefix distinguishes from original."""
    row = {}
    if not bout_feats:
        return row
    gait_feat_names = sorted(bout_feats[0].keys())
    arr = np.array([[bf.get(k, np.nan) for k in gait_feat_names] for bf in bout_feats])
    for j, name in enumerate(gait_feat_names):
        col = arr[:, j]; valid = col[np.isfinite(col)]
        if len(valid) < 2: continue
        row[f'{prefix}{name}_med'] = np.median(valid)
        row[f'{prefix}{name}_iqr'] = np.percentile(valid, 75) - np.percentile(valid, 25)
        row[f'{prefix}{name}_p10'] = np.percentile(valid, 10)
        row[f'{prefix}{name}_p90'] = np.percentile(valid, 90)
        row[f'{prefix}{name}_max'] = np.max(valid)
        row[f'{prefix}{name}_cv'] = np.std(valid) / (np.mean(valid) + 1e-12)
    row[f'{prefix}n_valid_bouts'] = len(bout_feats)
    row[f'{prefix}total_walk_sec'] = sum(bf.get('duration_sec', 0) for bf in bout_feats)
    durs = [bf.get('duration_sec', 0) for bf in bout_feats]
    row[f'{prefix}mean_bout_dur'] = np.mean(durs)
    row[f'{prefix}bout_dur_cv'] = np.std(durs) / (np.mean(durs) + 1e-12) if np.mean(durs) > 0 else 0
    return row


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_ridge(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 10)
    for a in alphas:
        preds = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=a)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            preds[te] = m.predict(sc.transform(X[te]))
        r2 = r2_score(y, preds)
        if r2 > best[0]:
            mae = mean_absolute_error(y, preds)
            rho = spearmanr(y, preds)[0]
            best = (r2, mae, rho, a)
    return best


def forward_selection(X, y, feature_names, max_features=30):
    n_feat = X.shape[1]
    selected = []
    remaining = list(range(n_feat))
    best_scores = []
    for step in range(min(max_features, n_feat)):
        best_r2 = -999
        best_idx = -1
        for idx in remaining:
            trial = selected + [idx]
            r2, mae, rho, alpha = loo_ridge(X[:, trial], y)
            if r2 > best_r2:
                best_r2 = r2; best_idx = idx
                best_mae = mae; best_rho = rho; best_alpha = alpha
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        best_scores.append((best_r2, best_mae, best_rho, best_alpha))
        fname = feature_names[best_idx]
        if step < 10 or (step + 1) % 5 == 0:
            print(f"  Step {step+1:2d}: +{fname:40s} R2={best_r2:.4f} MAE={best_mae:.0f} rho={best_rho:.3f} a={best_alpha}", flush=True)
        if len(best_scores) > 5:
            recent = [s[0] for s in best_scores[-5:]]
            if max(recent) - min(recent) < 0.002:
                print(f"  Early stop at step {step+1} (plateau)", flush=True)
                break
    return selected, best_scores


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)
    print(f"n={n} subjects", flush=True)

    # ── Extract CWT per-bout gait features + save bouts ──
    print("\n=== Extracting CWT per-bout gait features ===", flush=True)
    cwt_perbout_rows = []
    all_cwt_bouts = {}       # key -> list of (start, end) sample indices
    all_cwt_bout_feats = {}  # key -> list of feature dicts

    for i, (_, r) in enumerate(subj_df.iterrows()):
        npz_path = NPZ_DIR / f"{r['key']}.npz"
        xyz = np.load(npz_path)['xyz'].astype(np.float64)
        vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)

        # find_walking detection
        wi, steps, cad = find_walking(vm, FS)

        # Convert to bout sample indices (min 10s)
        cwt_bouts = fw_get_bouts(wi, FS, min_bout_sec=10)
        all_cwt_bouts[r['key']] = cwt_bouts

        # Extract per-bout gait features from CWT bouts
        bout_feats = []
        for s, e in cwt_bouts:
            if e > len(xyz): e = len(xyz)
            bf = extract_bout_features(xyz[s:e], FS)
            if bf is not None:
                bout_feats.append(bf)
        all_cwt_bout_feats[r['key']] = bout_feats

        # Aggregate with 'cg_' prefix (cwt-gait)
        row = aggregate_bout_features(bout_feats, prefix='cg_')
        cwt_perbout_rows.append(row)

        if (i + 1) % 10 == 0:
            nb = len(bout_feats)
            nc = len(cwt_bouts)
            print(f"  [{i+1}/{n}] {r['key']}: {nc} CWT bouts, {nb} valid, {nb} feats extracted", flush=True)

    cwt_df = pd.DataFrame(cwt_perbout_rows)
    cwt_cols = list(cwt_df.columns)
    print(f"  CWT per-bout features: {len(cwt_cols)}", flush=True)

    # ── Save CWT walking bouts ──
    with open(BASE / 'feats' / 'cwt_walking_bouts.pkl', 'wb') as f:
        pickle.dump({'bouts': all_cwt_bouts, 'bout_feats': all_cwt_bout_feats}, f)
    print(f"  Saved feats/cwt_walking_bouts.pkl ({len(all_cwt_bouts)} subjects)", flush=True)

    # ── Load existing features ──
    # Original per-bout (ENMO+HR detection)
    orig_df = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
    orig_cols = [c for c in orig_df.columns if c != 'key']
    X_orig = impute(orig_df[orig_cols].values.astype(float))

    # find_walking summary features (from step3)
    fw_path = BASE / 'feats' / 'find_walking_features.csv'
    if fw_path.exists():
        fw_df = pd.read_csv(fw_path)
        fw_cols = [c for c in fw_df.columns if c != 'key']
        X_fw = impute(fw_df[fw_cols].values.astype(float))
    else:
        fw_cols = []
        X_fw = np.zeros((n, 0))

    # CWT per-bout
    X_cwt = impute(cwt_df.values.astype(float))

    # Demographics
    demo = pd.read_excel(BASE / 'Accel files' / 'PedMSWalkStudy_Demographic.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    demo_cols = ['cohort_POMS', 'Age', 'Sex', 'Height', 'BMI']
    X_demo = impute(p[demo_cols].values.astype(float))

    # ── Quick evaluations ──
    print("\n" + "=" * 70, flush=True)
    print("QUICK EVALUATIONS", flush=True)
    print("=" * 70, flush=True)

    for name, X_test in [
        ('CWT per-bout only', X_cwt),
        ('CWT per-bout + Demo(5)', np.column_stack([X_cwt, X_demo])),
        ('Original per-bout + Demo(5)', np.column_stack([X_orig, X_demo])),
        ('Original + CWT per-bout + Demo(5)', np.column_stack([X_orig, X_cwt, X_demo])),
        ('All (orig + cwt + fw + demo)', np.column_stack([X_orig, X_cwt, X_fw, X_demo])),
    ]:
        r2, mae, rho, alpha = loo_ridge(X_test, y)
        print(f"  {name:45s} {X_test.shape[1]:3d}f  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}  a={alpha}", flush=True)

    # ── Forward selection from ALL features ──
    print("\n" + "=" * 70, flush=True)
    X_all = np.column_stack([X_orig, X_cwt, X_fw, X_demo])
    all_names = orig_cols + cwt_cols + fw_cols + demo_cols
    print(f"FORWARD SELECTION: {X_all.shape[1]} features ({len(orig_cols)} orig + {len(cwt_cols)} cwt + {len(fw_cols)} fw + {len(demo_cols)} demo)", flush=True)
    print("=" * 70, flush=True)

    selected_idx, scores = forward_selection(X_all, y, all_names, max_features=30)

    best_step = np.argmax([s[0] for s in scores])
    best_r2, best_mae, best_rho, best_alpha = scores[best_step]
    best_feats = [all_names[i] for i in selected_idx[:best_step + 1]]
    print(f"\n  Best: {best_step+1} features, R2={best_r2:.4f}, MAE={best_mae:.0f}, rho={best_rho:.3f}, a={best_alpha}", flush=True)
    print(f"  Features: {best_feats}", flush=True)

    # Count by source
    orig_sel = [f for f in best_feats if f.startswith('g_') or f.startswith('act_')]
    cwt_sel = [f for f in best_feats if f.startswith('cg_')]
    fw_sel = [f for f in best_feats if f.startswith('fw_')]
    demo_sel = [f for f in best_feats if f in demo_cols]
    print(f"  Original: {len(orig_sel)}, CWT per-bout: {len(cwt_sel)}, fw summary: {len(fw_sel)}, demo: {len(demo_sel)}", flush=True)

    # ── Save ──
    cwt_df.insert(0, 'key', subj_df['key'].values)
    cwt_df.to_csv(BASE / 'feats' / 'cwt_perbout_features.csv', index=False)

    results = {
        'forward_selected_features': best_feats,
        'forward_selected_indices': selected_idx[:best_step + 1],
        'forward_selection_scores': scores,
        'all_feature_names': all_names,
    }
    with open(BASE / 'feats' / 'step4_experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\n  Saved feats/cwt_perbout_features.csv ({cwt_df.shape})", flush=True)
    print(f"  Saved feats/cwt_walking_bouts.pkl", flush=True)
    print(f"  Saved feats/step4_experiment_results.pkl", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
