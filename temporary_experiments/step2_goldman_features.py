#!/usr/bin/env python3
"""
Step 2: Extract Goldman et al. (2018) features (MSR, HWSR) from home accelerometry,
combine with per-bout features + demographics, and do feature selection.

MSR = Maximum Step Rate (steps/min) - highest minutewise step rate across recording
HWSR = Habitual Walking Step Rate (steps/min) - expected step rate during walking
"""
import os, re, math, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
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
# STEP DETECTION (per-minute step counts from raw accelerometry)
# ══════════════════════════════════════════════════════════════════

def compute_minutewise_steps(xyz, fs=30):
    """Compute step counts per minute from raw XYZ accelerometry."""
    # Bandpass filter for walking frequencies
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    vm_bp = filtfilt(b, a, vm - vm.mean())

    # Also compute per-minute ENMO for activity classification
    enmo = np.maximum(vm - 1.0, 0.0)

    min_samples = int(60 * fs)
    n_minutes = len(vm_bp) // min_samples
    if n_minutes < 1:
        return np.array([]), np.array([])

    step_rates = []
    enmo_per_min = []
    for m in range(n_minutes):
        seg = vm_bp[m * min_samples:(m + 1) * min_samples]
        enmo_seg = enmo[m * min_samples:(m + 1) * min_samples]

        # Check if worn (std > 0.01)
        if np.std(seg) < 0.005:
            step_rates.append(0)
            enmo_per_min.append(0)
            continue

        # Peak detection for steps
        # Minimum distance between steps: ~0.3s (200 steps/min max)
        min_dist = max(1, int(0.3 * fs))
        prominence = max(0.02, 0.3 * np.std(seg))
        peaks, _ = find_peaks(seg, distance=min_dist, prominence=prominence)
        step_rates.append(len(peaks))  # steps per minute
        enmo_per_min.append(np.mean(enmo_seg))

    return np.array(step_rates), np.array(enmo_per_min)


def extract_goldman_features(xyz, fs=30):
    """Extract MSR, HWSR, and related features."""
    step_rates, enmo_per_min = compute_minutewise_steps(xyz, fs)

    if len(step_rates) < 10:
        return {}

    f = {}

    # MSR: Maximum Step Rate (max across all minutes with steps > 0)
    active_steps = step_rates[step_rates > 0]
    if len(active_steps) == 0:
        return {}

    f['msr'] = float(np.max(step_rates))

    # Identify walking minutes using ENMO threshold + step rate
    # Walking: ENMO >= 0.015 and step_rate >= 30 (at least 0.5 steps/sec)
    walking = (enmo_per_min >= 0.015) & (step_rates >= 30)
    running = (enmo_per_min >= 0.1) & (step_rates >= 100)
    walking_only = walking & ~running

    # HWSR: Habitual Walking Step Rate (mean step rate during walking minutes)
    walk_steps = step_rates[walking_only]
    if len(walk_steps) >= 5:
        f['hwsr'] = float(np.mean(walk_steps))
        f['hwsr_median'] = float(np.median(walk_steps))
        f['hwsr_std'] = float(np.std(walk_steps))
    else:
        # Fallback: use all active minutes with reasonable step rates (30-180)
        reasonable = step_rates[(step_rates >= 30) & (step_rates <= 180)]
        if len(reasonable) >= 3:
            f['hwsr'] = float(np.mean(reasonable))
            f['hwsr_median'] = float(np.median(reasonable))
            f['hwsr_std'] = float(np.std(reasonable))
        else:
            f['hwsr'] = float(np.mean(active_steps))
            f['hwsr_median'] = float(np.median(active_steps))
            f['hwsr_std'] = float(np.std(active_steps))

    # MSR + HWSR (Goldman's combined feature)
    f['msr_plus_hwsr'] = f['msr'] + f['hwsr']

    # Additional step-rate features
    f['avg_daily_steps'] = float(np.sum(step_rates))  # total steps (single recording)
    f['pct_walking_minutes'] = float(np.mean(walking_only)) if len(walking_only) > 0 else 0
    f['pct_active_minutes'] = float(np.mean(step_rates > 0))
    f['step_rate_p90'] = float(np.percentile(active_steps, 90)) if len(active_steps) > 0 else 0
    f['step_rate_p75'] = float(np.percentile(active_steps, 75)) if len(active_steps) > 0 else 0
    f['n_walking_minutes'] = float(np.sum(walking_only))

    return f


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_ridge(X, y, alphas=[5, 10, 20, 50, 100]):
    """LOO CV with alpha search. Returns (R2, MAE, rho, best_alpha, preds)."""
    best = (-999, 0, 0, 10, None)
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
            best = (r2, mae, rho, a, preds)
    return best


def forward_selection(X, y, feature_names, max_features=30):
    """Forward feature selection using LOO Ridge R2."""
    n_feat = X.shape[1]
    selected = []
    remaining = list(range(n_feat))
    best_scores = []

    for step in range(min(max_features, n_feat)):
        best_r2 = -999
        best_idx = -1
        for idx in remaining:
            trial = selected + [idx]
            X_trial = X[:, trial]
            r2, mae, rho, alpha, _ = loo_ridge(X_trial, y)
            if r2 > best_r2:
                best_r2 = r2
                best_idx = idx
                best_mae = mae
                best_rho = rho
                best_alpha = alpha

        if best_idx < 0:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        best_scores.append((best_r2, best_mae, best_rho, best_alpha))
        fname = feature_names[best_idx] if best_idx < len(feature_names) else f'f{best_idx}'

        if step < 10 or (step + 1) % 5 == 0:
            print(f"  Step {step+1:2d}: +{fname:40s} R2={best_r2:.4f} MAE={best_mae:.0f} rho={best_rho:.3f} a={best_alpha}")

        # Early stopping: if no improvement for 5 steps
        if len(best_scores) > 5:
            recent = [s[0] for s in best_scores[-5:]]
            if max(recent) - min(recent) < 0.002:
                print(f"  Early stop at step {step+1} (plateau)")
                break

    return selected, best_scores


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    # Load subject list and targets
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)
    print(f"n={n} subjects")

    # ── Extract Goldman features ──
    print("\n=== Extracting Goldman features (MSR, HWSR) ===")
    goldman_rows = []
    for i, (_, r) in enumerate(subj_df.iterrows()):
        npz_path = NPZ_DIR / f"{r['key']}.npz"
        xyz = np.load(npz_path)['xyz'].astype(np.float64)
        gf = extract_goldman_features(xyz, FS)
        goldman_rows.append(gf)
        if (i + 1) % 20 == 0:
            msr = gf.get('msr', 0)
            hwsr = gf.get('hwsr', 0)
            print(f"  [{i+1}/{n}] {r['key']}: MSR={msr:.0f} HWSR={hwsr:.0f}", flush=True)

    goldman_df = pd.DataFrame(goldman_rows)
    print(f"  Goldman features: {list(goldman_df.columns)}")

    # ── Load per-bout features ──
    perbout_df = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
    perbout_cols = [c for c in perbout_df.columns if c != 'key']

    # ── Demographics ──
    demo = pd.read_excel(BASE / 'Accel files' / 'PedMSWalkStudy_Demographic.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    demo_cols = ['cohort_POMS', 'Age', 'Sex', 'Height', 'BMI']
    X_demo = impute(p[demo_cols].values.astype(float))

    # ══════════════════════════════════════════════════════════════
    # MODEL 1: Goldman features only (MSR, HWSR, MSR+HWSR)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL 1: Goldman features only")
    print("=" * 70)

    goldman_feat_names = list(goldman_df.columns)
    X_goldman = impute(goldman_df.values.astype(float))

    # Correlations
    print("\n  Feature correlations with 6MWD:")
    for j, name in enumerate(goldman_feat_names):
        rho, pval = spearmanr(X_goldman[:, j], y)
        print(f"    {name:25s}  rho={rho:+.3f}  p={pval:.1e}")

    # MSR + HWSR only
    msr_idx = goldman_feat_names.index('msr')
    hwsr_idx = goldman_feat_names.index('hwsr')
    msr_hwsr_idx = goldman_feat_names.index('msr_plus_hwsr')

    for name, cols in [
        ('MSR only', [msr_idx]),
        ('HWSR only', [hwsr_idx]),
        ('MSR+HWSR (sum)', [msr_hwsr_idx]),
        ('MSR, HWSR (2f)', [msr_idx, hwsr_idx]),
        ('All Goldman', list(range(len(goldman_feat_names)))),
        ('All Goldman + Demo(5)', None),
    ]:
        if cols is None:
            X_test = np.column_stack([X_goldman, X_demo])
        else:
            X_test = X_goldman[:, cols]
        r2, mae, rho, alpha, _ = loo_ridge(X_test, y)
        nf = X_test.shape[1]
        print(f"  {name:30s}  {nf:2d}f  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}  a={alpha}")

    # ══════════════════════════════════════════════════════════════
    # MODEL 2: Goldman + PerBout-Top20 + Demo(5) (baseline comparison)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL 2: Goldman + Previous Top-20 + Demo(5)")
    print("=" * 70)

    # Load Top-20
    TOP20 = [
        'g_duration_sec_max', 'g_bout_dur_cv', 'g_duration_sec_cv',
        'act_enmo_p95', 'act_pct_vigorous',
        'g_acf_step_reg_max', 'g_enmo_mean_p10', 'g_ml_range_med',
        'g_ml_rms_cv', 'g_ap_rms_cv', 'g_ap_rms_med',
        'g_acf_step_reg_p90', 'g_jerk_mean_med', 'g_mean_bout_dur',
        'g_signal_energy_med', 'g_ml_rms_med', 'g_vm_std_med',
        'g_enmo_mean_med', 'g_enmo_p95_med', 'g_acf_step_reg_med',
    ]
    X_top20 = np.full((n, len(TOP20)), np.nan)
    for j, feat in enumerate(TOP20):
        if feat in perbout_df.columns:
            X_top20[:, j] = perbout_df[feat].values
    X_top20 = impute(X_top20)

    # Top20 + Demo (baseline)
    X_baseline = np.column_stack([X_top20, X_demo])
    r2, mae, rho, alpha, _ = loo_ridge(X_baseline, y)
    print(f"  Baseline (Top20+Demo5):        25f  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}  a={alpha}")

    # Top20 + Goldman + Demo
    X_combined = np.column_stack([X_top20, X_goldman, X_demo])
    combined_names = TOP20 + goldman_feat_names + demo_cols
    r2, mae, rho, alpha, _ = loo_ridge(X_combined, y)
    print(f"  Top20+Goldman+Demo5:           {X_combined.shape[1]:2d}f  R2={r2:.4f}  MAE={mae:.0f}  rho={rho:.3f}  a={alpha}")

    # ══════════════════════════════════════════════════════════════
    # MODEL 3: Forward selection from ALL features (153 PerBout + Goldman + Demo)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL 3: Forward selection from ALL features")
    print("=" * 70)

    X_perbout = impute(perbout_df[perbout_cols].values.astype(float))
    X_all = np.column_stack([X_perbout, X_goldman, X_demo])
    all_names = perbout_cols + goldman_feat_names + demo_cols

    print(f"  Total candidate features: {X_all.shape[1]}")
    print(f"  Running forward selection (this takes a few minutes)...\n")

    selected_idx, scores = forward_selection(X_all, y, all_names, max_features=30)

    # Report best
    best_step = np.argmax([s[0] for s in scores])
    best_r2, best_mae, best_rho, best_alpha = scores[best_step]
    best_feats = [all_names[i] for i in selected_idx[:best_step + 1]]
    print(f"\n  Best: {best_step+1} features, R2={best_r2:.4f}, MAE={best_mae:.0f}, rho={best_rho:.3f}, a={best_alpha}")
    print(f"  Features: {best_feats}")

    # Save results
    results = {
        'goldman_features': goldman_df,
        'forward_selected_features': best_feats,
        'forward_selected_indices': selected_idx[:best_step + 1],
        'forward_selection_scores': scores,
        'all_feature_names': all_names,
    }
    with open(BASE / 'feats' / 'goldman_experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    goldman_df.insert(0, 'key', subj_df['key'].values)
    goldman_df.to_csv(BASE / 'feats' / 'goldman_features.csv', index=False)
    print(f"\n  Saved feats/goldman_features.csv")
    print(f"  Saved feats/goldman_experiment_results.pkl")

    print(f"\nDone in {time.time()-t0:.0f}s")
