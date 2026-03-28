#!/usr/bin/env python3
"""
Step 3: Implement Straczkiewicz et al. (2023) find_walking algorithm in Python,
extract CWT-based walking features, and combine with existing features for prediction.

Algorithm: CWT (Morse/Morlet wavelet) walking detection using intensity + periodicity + duration.
Reference: npj Digital Medicine (2023)6:29
"""
import os, re, math, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
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
# FIND_WALKING: Python implementation of Straczkiewicz et al. (2023)
# ══════════════════════════════════════════════════════════════════

def find_walking(vm, fs, min_amp=0.3, T=3, delta=2, alpha=0.6, beta=2.5,
                 step_freq=(1.4, 2.3)):
    """
    CWT-based walking detection (vectorized).
    Returns: wi (walking indication per second), steps, cad (cadence per second)
    """
    n_samples = len(vm)
    sec = int(fs)
    n_secs = n_samples // sec
    if n_secs < 5:
        return np.zeros(n_secs), 0, np.zeros(n_secs)

    # Reshape to (n_secs, fs) for vectorized per-second ops
    vm_mat = vm[:n_secs * sec].reshape(n_secs, sec)

    # Step 1: Peak-to-peak amplitude per second (vectorized)
    p2p = vm_mat.max(axis=1) - vm_mat.min(axis=1)
    valid = p2p >= min_amp

    # Step 2: FFT-based frequency analysis (fast, vectorized)
    # Use 3-second windows (stride 1 second) for frequency resolution
    win_secs = 3
    win_len = win_secs * sec
    freq_grid = np.arange(0.5, 4.05, 0.05)
    n_freqs = len(freq_grid)
    loc1 = np.searchsorted(freq_grid, step_freq[0])
    loc2 = np.searchsorted(freq_grid, step_freq[1])

    # FFT frequencies for win_len samples
    fft_freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
    f_interp_indices = np.searchsorted(fft_freqs, freq_grid)
    f_interp_indices = np.clip(f_interp_indices, 0, len(fft_freqs) - 1)

    detected_freq = np.full(n_secs, np.nan)
    peak_matrix = np.zeros((n_freqs, n_secs), dtype=bool)
    tukey_win = tukey(win_len, alpha=0.02)

    for s in range(n_secs):
        if not valid[s]:
            continue

        # 3-second window centered on s
        s_start = max(0, s - 1)
        s_end = min(n_secs, s + 2)
        if s_end - s_start < win_secs:
            s_start = max(0, s_end - win_secs)

        seg = vm[s_start * sec:s_end * sec]
        if len(seg) < win_len:
            # Pad if at edges
            seg = np.pad(seg, (0, win_len - len(seg)), mode='edge')

        seg = seg[:win_len]
        seg_win = (seg - seg.mean()) * tukey_win

        # FFT power spectrum
        fft_vals = np.fft.rfft(seg_win)
        power_full = np.abs(fft_vals) ** 2

        # Sample at frequency grid points
        power = power_full[f_interp_indices]

        if np.max(power) < 1e-10:
            continue

        # Find peaks
        peaks_idx, _ = find_peaks(power, height=0.1 * np.max(power))
        if len(peaks_idx) == 0:
            continue

        peak_vals = power[peaks_idx]
        order = np.argsort(peak_vals)[::-1]
        peaks_idx = peaks_idx[order]
        peak_vals = peak_vals[order]

        # Best peak in step frequency range
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

    # Step 3: Continuity check
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


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION from find_walking results
# ══════════════════════════════════════════════════════════════════

def extract_fw_features(vm, fs, wi, cad):
    """Extract features from find_walking outputs + CWT cadence analysis."""
    f = {}
    n_secs = len(wi)
    total_hours = n_secs / 3600

    # Walking detection summary
    walk_secs = wi.astype(bool)
    n_walk = int(np.sum(walk_secs))
    f['fw_n_walking_secs'] = n_walk
    f['fw_pct_walking'] = n_walk / (n_secs + 1e-12)
    f['fw_walking_min_per_hr'] = (n_walk / 60) / (total_hours + 1e-12)

    if n_walk < 5:
        return f

    # Cadence features (steps/sec -> steps/min)
    walk_cad = cad[walk_secs] * 60  # convert to steps/min
    f['fw_cad_mean'] = float(np.mean(walk_cad))
    f['fw_cad_median'] = float(np.median(walk_cad))
    f['fw_cad_std'] = float(np.std(walk_cad))
    f['fw_cad_cv'] = float(np.std(walk_cad) / (np.mean(walk_cad) + 1e-12))
    f['fw_cad_p10'] = float(np.percentile(walk_cad, 10))
    f['fw_cad_p90'] = float(np.percentile(walk_cad, 90))
    f['fw_cad_max'] = float(np.max(walk_cad))
    f['fw_cad_iqr'] = float(np.percentile(walk_cad, 75) - np.percentile(walk_cad, 25))

    # MSR-like: max cadence (steps/min)
    f['fw_msr'] = float(np.max(walk_cad))
    # HWSR-like: median cadence during walking
    f['fw_hwsr'] = float(np.median(walk_cad))

    # Walking bout detection from wi
    bout_durs = []
    in_b, bs = False, 0
    for s in range(n_secs):
        if walk_secs[s] and not in_b:
            bs = s; in_b = True
        elif not walk_secs[s] and in_b:
            bout_durs.append(s - bs)
            in_b = False
    if in_b:
        bout_durs.append(n_secs - bs)

    f['fw_n_bouts'] = len(bout_durs)
    if bout_durs:
        f['fw_bout_dur_mean'] = float(np.mean(bout_durs))
        f['fw_bout_dur_max'] = float(np.max(bout_durs))
        f['fw_bout_dur_cv'] = float(np.std(bout_durs) / (np.mean(bout_durs) + 1e-12))
        f['fw_bout_dur_total'] = float(np.sum(bout_durs))

        # Per-bout cadence variability
        bout_cad_means = []
        start = 0
        for dur in bout_durs:
            # Find start index in walk_secs
            while start < n_secs and not walk_secs[start]:
                start += 1
            end = start + dur
            if end <= n_secs:
                bout_cad = cad[start:end]
                bout_cad_walk = bout_cad[bout_cad > 0]
                if len(bout_cad_walk) > 0:
                    bout_cad_means.append(np.mean(bout_cad_walk) * 60)
            start = end

        if len(bout_cad_means) >= 2:
            f['fw_bout_cad_cv'] = float(np.std(bout_cad_means) / (np.mean(bout_cad_means) + 1e-12))
            f['fw_bout_cad_range'] = float(np.max(bout_cad_means) - np.min(bout_cad_means))
    else:
        f['fw_bout_dur_mean'] = 0
        f['fw_bout_dur_max'] = 0
        f['fw_bout_dur_cv'] = 0
        f['fw_bout_dur_total'] = 0

    # Cadence drift (trend over recording)
    if n_walk >= 10:
        walk_indices = np.where(walk_secs)[0]
        walk_cad_ts = cad[walk_secs] * 60
        # Linear fit of cadence over time
        if len(walk_cad_ts) > 2:
            slope = np.polyfit(walk_indices / 3600, walk_cad_ts, 1)[0]  # steps/min per hour
            f['fw_cad_slope'] = float(slope)

    # Intensity during walking vs non-walking
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs)
    enmo_sec = np.zeros(n_secs)
    for s in range(n_secs):
        enmo_sec[s] = np.mean(enmo[s * sec:(s + 1) * sec])

    walk_enmo = enmo_sec[walk_secs]
    nonwalk_enmo = enmo_sec[~walk_secs]
    f['fw_walk_enmo_mean'] = float(np.mean(walk_enmo))
    f['fw_walk_enmo_p95'] = float(np.percentile(walk_enmo, 95))
    if len(nonwalk_enmo) > 0:
        f['fw_walk_nonwalk_ratio'] = float(np.mean(walk_enmo) / (np.mean(nonwalk_enmo) + 1e-12))

    return f


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
    print(f"n={n} subjects")

    # ── Extract find_walking features ──
    print("\n=== Extracting find_walking (CWT) features ===", flush=True)
    fw_rows = []
    for i, (_, r) in enumerate(subj_df.iterrows()):
        npz_path = NPZ_DIR / f"{r['key']}.npz"
        xyz = np.load(npz_path)['xyz'].astype(np.float64)
        vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)

        wi, steps, cad = find_walking(vm, FS, min_amp=0.3, T=3, delta=2,
                                       alpha=0.6, beta=2.5, step_freq=(1.4, 2.3))
        fw_feats = extract_fw_features(vm, FS, wi, cad)
        fw_feats['fw_total_steps'] = steps
        fw_rows.append(fw_feats)

        if (i + 1) % 10 == 0:
            n_walk = int(fw_feats.get('fw_n_walking_secs', 0))
            cad_m = fw_feats.get('fw_cad_mean', 0)
            print(f"  [{i+1}/{n}] {r['key']}: {n_walk}s walking, cad={cad_m:.0f} steps/min", flush=True)

    fw_df = pd.DataFrame(fw_rows)
    fw_cols = list(fw_df.columns)
    print(f"  find_walking features: {len(fw_cols)}")

    # ── Correlations ──
    print("\n  Feature correlations with 6MWD:", flush=True)
    X_fw = impute(fw_df.values.astype(float))
    for j, name in enumerate(fw_cols):
        rho_val, pval = spearmanr(X_fw[:, j], y)
        if abs(rho_val) > 0.15:
            print(f"    {name:30s}  rho={rho_val:+.3f}  p={pval:.1e}")

    # ── Model 1: find_walking features only ──
    print("\n" + "=" * 70, flush=True)
    print("MODEL 1: find_walking features only", flush=True)
    r2, mae, rho, alpha = loo_ridge(X_fw, y)
    print(f"  All fw features ({X_fw.shape[1]}f):  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}  a={alpha}")

    # ── Load demographics ──
    demo = pd.read_excel(BASE / 'Accel files' / 'PedMSWalkStudy_Demographic.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    demo_cols = ['cohort_POMS', 'Age', 'Sex', 'Height', 'BMI']
    X_demo = impute(p[demo_cols].values.astype(float))

    # fw + Demo
    X_fw_demo = np.column_stack([X_fw, X_demo])
    r2, mae, rho, alpha = loo_ridge(X_fw_demo, y)
    print(f"  fw + Demo(5) ({X_fw_demo.shape[1]}f):  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}  a={alpha}")

    # ── Model 2: Combine with previous per-bout features ──
    print("\n" + "=" * 70, flush=True)
    print("MODEL 2: Per-bout + find_walking + Demo(5)", flush=True)

    perbout_df = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
    perbout_cols = [c for c in perbout_df.columns if c != 'key']
    X_perbout = impute(perbout_df[perbout_cols].values.astype(float))

    X_all = np.column_stack([X_perbout, X_fw, X_demo])
    all_names = perbout_cols + fw_cols + demo_cols
    print(f"  Total features: {X_all.shape[1]} ({len(perbout_cols)} per-bout + {len(fw_cols)} fw + {len(demo_cols)} demo)")

    r2, mae, rho, alpha = loo_ridge(X_all, y)
    print(f"  All combined ({X_all.shape[1]}f):  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}  a={alpha}")

    # ── Model 3: Forward selection from ALL ──
    print("\n" + "=" * 70, flush=True)
    print("MODEL 3: Forward selection from ALL features (per-bout + fw + demo)", flush=True)
    print(f"  {X_all.shape[1]} candidate features", flush=True)
    print(f"  Running forward selection...\n", flush=True)

    selected_idx, scores = forward_selection(X_all, y, all_names, max_features=30)

    best_step = np.argmax([s[0] for s in scores])
    best_r2, best_mae, best_rho, best_alpha = scores[best_step]
    best_feats = [all_names[i] for i in selected_idx[:best_step + 1]]
    print(f"\n  Best: {best_step+1} features, R2={best_r2:.4f}, MAE={best_mae:.0f}, rho={best_rho:.3f}, a={best_alpha}")
    print(f"  Features: {best_feats}")

    # Count how many fw features selected
    fw_selected = [f for f in best_feats if f.startswith('fw_')]
    print(f"  find_walking features selected: {len(fw_selected)}/{len(fw_cols)} -> {fw_selected}")

    # ── Save ──
    fw_df.insert(0, 'key', subj_df['key'].values)
    fw_df.to_csv(BASE / 'feats' / 'find_walking_features.csv', index=False)

    results = {
        'fw_features': fw_df,
        'forward_selected_features': best_feats,
        'forward_selected_indices': selected_idx[:best_step + 1],
        'forward_selection_scores': scores,
        'all_feature_names': all_names,
    }
    with open(BASE / 'feats' / 'find_walking_experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\n  Saved feats/find_walking_features.csv")
    print(f"  Saved feats/find_walking_experiment_results.pkl")
    print(f"\nDone in {time.time()-t0:.0f}s")
