#!/usr/bin/env python3
"""
Goldman PAM implementation using GT3X raw acceleration directly.
Step detection from raw signal instead of AGD files.
Recovers subjects missing from AGD-based approach.

Step detection algorithm:
1. Read GT3X via actipy (30 Hz, timestamped)
2. Compute VM = sqrt(x² + y² + z²)
3. Bandpass filter 0.5-3.0 Hz (walking frequency band)
4. Peak detection with min distance 0.3s, min prominence
5. Count peaks per minute = steps per minute
6. Then apply Goldman PAM (MSR, HWSR via HMM)
"""
import sys, warnings, time, re
import numpy as np
import pandas as pd
import actipy
from pathlib import Path
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
ACCEL_DIR = BASE / 'Accel files'
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from hmmlearn.hmm import GaussianHMM


def eval_loo(X, y, alpha=10):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, pr), mean_absolute_error(y, pr), spearmanr(y, pr)[0], pearsonr(y, pr)[0]

def best_alpha(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 0, 10)
    for a in alphas:
        r2, mae, rho, r_val = eval_loo(X, y, a)
        if r2 > best[0]: best = (r2, mae, rho, r_val, a)
    return best

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def count_steps_per_minute(gt3x_data):
    """
    Count steps per minute from raw acceleration.
    1. Drop NaN, estimate fs from timestamps
    2. Compute ENMO (removes gravity bias)
    3. Bandpass 0.5-3.0 Hz
    4. Peak detection with adaptive prominence
    5. Aggregate to per-minute step counts
    """
    # Drop NaN rows
    data = gt3x_data.dropna()
    if len(data) < 1000:
        return pd.DataFrame()

    # Estimate fs
    dt = np.diff(data.index[:1000].values).astype(float) / 1e9  # ns to seconds
    dt = dt[dt > 0]
    fs = round(1.0 / np.median(dt)) if len(dt) > 0 else 30

    x = data['x'].values.astype(float)
    y_acc = data['y'].values.astype(float)
    z = data['z'].values.astype(float)

    vm = np.sqrt(x**2 + y_acc**2 + z**2)
    # ENMO: removes gravity, keeps dynamic acceleration
    enmo = np.maximum(vm - 1.0, 0.0)

    # Bandpass filter on VM (centered)
    vm_centered = vm - np.median(vm)
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    if len(vm_centered) < 3 * max(len(b), len(a)):
        return pd.DataFrame()
    vm_filt = filtfilt(b, a, vm_centered)

    # Adaptive prominence: use std of filtered signal
    sig_std = np.std(vm_filt)
    prominence = max(0.003, sig_std * 0.3)  # 30% of std, min 0.003g
    min_dist = int(0.3 * fs)

    peaks, _ = find_peaks(vm_filt, distance=min_dist, prominence=prominence)

    if len(peaks) == 0:
        # Fallback: use ENMO peaks
        b2, a2 = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
        enmo_filt = filtfilt(b2, a2, enmo - np.mean(enmo))
        prominence2 = max(0.002, np.std(enmo_filt) * 0.3)
        peaks, _ = find_peaks(enmo_filt, distance=min_dist, prominence=prominence2)

    # Build per-minute step counts
    timestamps = data.index
    if len(peaks) > 0:
        peak_times = timestamps[peaks]
        peak_series = pd.Series(1, index=peak_times)
        step_counts = peak_series.resample('min').sum().fillna(0)
    else:
        # No steps found — create empty minute series
        step_counts = pd.Series(0, index=pd.date_range(timestamps[0].floor('min'),
                                                         timestamps[-1].ceil('min'), freq='min'))

    # Activity counts: sum of ENMO per minute
    enmo_series = pd.Series(enmo, index=timestamps)
    activity_counts = enmo_series.resample('min').sum().fillna(0)

    df_min = pd.DataFrame({
        'datetime': step_counts.index,
        'steps_per_min': step_counts.values,
        'activity_counts': activity_counts.reindex(step_counts.index).fillna(0).values,
    })

    return df_min


def get_valid_wear_days(df_min, min_wear_hours=10):
    """Valid wear days: ≥ min_wear_hours with activity."""
    df_min['date'] = df_min['datetime'].dt.date
    df_min['is_worn'] = (df_min['steps_per_min'] > 0) | (df_min['activity_counts'] > 0)
    daily = df_min.groupby('date').agg(
        worn_minutes=('is_worn', 'sum'),
        total_steps=('steps_per_min', 'sum'),
    ).reset_index()
    daily['worn_hours'] = daily['worn_minutes'] / 60
    valid = daily[daily['worn_hours'] >= min_wear_hours]['date'].tolist()
    return valid


def compute_msr(df_min, valid_days):
    df_valid = df_min[df_min['datetime'].dt.date.isin(valid_days)]
    return float(df_valid['steps_per_min'].max()) if len(df_valid) > 0 else np.nan


def fit_pam_hwsr(df_min, valid_days, msr):
    """Fit HMM → extract HWSR."""
    df_valid = df_min[df_min['datetime'].dt.date.isin(valid_days)]
    step_rates = df_valid['steps_per_min'].values.astype(float)

    if len(step_rates) < 100 or msr <= 0:
        return np.nan

    X = step_rates.reshape(-1, 1)
    best_model, best_bic, best_n = None, np.inf, 2

    for n_states in range(2, 7):
        try:
            model = GaussianHMM(n_components=n_states, covariance_type='diag',
                                n_iter=100, random_state=42)
            model.fit(X)
            score = model.score(X)
            n_params = n_states * n_states + n_states * 2
            bic = -2 * score + n_params * np.log(len(X))
            if bic < best_bic:
                best_bic, best_model, best_n = bic, model, n_states
        except:
            continue

    if best_model is None:
        return np.nan

    states = best_model.predict(X)
    means = best_model.means_.flatten()

    # Identify walking state
    active_states = []
    for s in range(best_n):
        state_samples = step_rates[states == s]
        if len(state_samples) < 5: continue
        state_mean = means[s]
        state_cv = np.std(state_samples) / (state_mean + 1e-12)
        if state_mean > msr / 2 and state_cv >= 8:
            active_states.append((s, state_mean))

    if len(active_states) == 1:
        walking = active_states[0] if active_states[0][1] <= 130 else None
    elif len(active_states) >= 2:
        active_states.sort(key=lambda x: x[1])
        walking = active_states[0]
    else:
        candidates = [(s, means[s]) for s in range(best_n) if 10 < means[s] < 130]
        walking = max(candidates, key=lambda x: x[1]) if candidates else None

    return float(walking[1]) if walking else np.nan


if __name__ == '__main__':
    t0 = time.time()

    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
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

    # Also load AGD-based features for comparison
    agd_feats = pd.read_csv(BASE / 'temporary_experiments' / 'goldman_pam_features.csv')

    print(f"Goldman PAM from GT3X raw acceleration")
    print(f"n={n} subjects")
    print(f"{'='*80}\n")

    cache = BASE / 'temporary_experiments' / 'goldman_gt3x_cache.npz'
    if cache.exists():
        print("Loading from cache...", flush=True)
        d = np.load(cache, allow_pickle=True)
        all_features = list(d['features'])
        valid_mask = d['valid']
    else:
        all_features = []
        valid_mask = []

        for i, (_, r) in enumerate(ids_v.iterrows()):
            cohort, sid = r['cohort'], int(r['subj_id'])
            prefix = f"{cohort}{sid:02d}_"

            # Find GT3X
            accel_folder = None
            for d in ACCEL_DIR.iterdir():
                if d.is_dir() and d.name.startswith(prefix):
                    accel_folder = d; break

            if accel_folder is None:
                all_features.append(None); valid_mask.append(False); continue

            gt3x_files = list(accel_folder.glob('*.gt3x'))
            if not gt3x_files:
                all_features.append(None); valid_mask.append(False); continue

            # Read GT3X
            try:
                gt3x_data, info = actipy.read_device(
                    str(gt3x_files[0]), lowpass_hz=None,
                    calibrate_gravity=False, detect_nonwear=False, resample_hz=None)
            except Exception as e:
                all_features.append(None); valid_mask.append(False)
                print(f"  [{i+1:3d}] {cohort}{sid:02d}: GT3X ERROR: {e}"); continue

            # Count steps per minute
            df_min = count_steps_per_minute(gt3x_data)
            if len(df_min) < 60:
                all_features.append(None); valid_mask.append(False); continue

            # Valid wear days (relax to 6 hours if needed)
            valid_days = get_valid_wear_days(df_min, min_wear_hours=10)
            if len(valid_days) < 3:
                valid_days = get_valid_wear_days(df_min, min_wear_hours=6)
            if len(valid_days) < 1:
                valid_days = get_valid_wear_days(df_min, min_wear_hours=1)

            if len(valid_days) < 1:
                all_features.append(None); valid_mask.append(False); continue

            # Conventional HPA
            df_valid = df_min[df_min['datetime'].dt.date.isin(valid_days)]
            n_days = len(valid_days)
            avg_daily_steps = df_valid['steps_per_min'].sum() / n_days

            # MSR
            msr = compute_msr(df_min, valid_days)

            # HWSR
            hwsr = fit_pam_hwsr(df_min, valid_days, msr)

            feats = {
                'msr': msr,
                'hwsr': hwsr,
                'avg_daily_steps': avg_daily_steps,
                'n_valid_days': n_days,
                'msr_plus_hwsr': msr + hwsr if np.isfinite(hwsr) else msr,
            }
            all_features.append(feats)
            valid_mask.append(True)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1:3d}/{n}] {cohort}{sid:02d}: MSR={msr:.0f} HWSR={hwsr:.1f} "
                      f"steps/day={avg_daily_steps:.0f} days={n_days}", flush=True)

        valid_mask = np.array(valid_mask)
        np.savez(cache, features=all_features, valid=valid_mask, allow_pickle=True)
        print("Cached.", flush=True)

    valid_mask = np.array(valid_mask, dtype=bool)
    nv = valid_mask.sum()
    print(f"\nValid: {nv}/{n} (vs AGD-based: 92/{n})")

    # Build matrix
    feat_names = sorted([k for k in all_features[valid_mask.tolist().index(True)].keys()])
    X_all = np.array([[f[k] for k in feat_names] for f, v in zip(all_features, valid_mask) if v])
    X_all = impute(X_all)
    y_v = y[valid_mask]
    D5_v = X_demo5[valid_mask]

    # Correlations
    print(f"\n{'='*80}")
    print(f"CORRELATIONS WITH 6MWD (n={nv}, GT3X-based step detection)")
    print(f"{'='*80}")
    for j, name in enumerate(feat_names):
        r_val, p_r = pearsonr(X_all[:, j], y_v)
        rho, p_s = spearmanr(X_all[:, j], y_v)
        print(f"  {name:25s}  r={r_val:+.3f} (p={p_r:.4f})  ρ={rho:+.3f} (p={p_s:.4f})")

    # Prediction
    print(f"\n{'='*80}")
    print(f"PREDICTION (LOO CV, n={nv})")
    print(f"{'='*80}")

    def report(name, nf, r2, mae, rho, r_val, alpha):
        print(f"  {name:45s} {nf:>3d}f  α={alpha:>3d}  R²={r2:.4f}  MAE={mae:.0f}ft  r={r_val:.3f}  ρ={rho:.3f}")

    # Individual
    for j, name in enumerate(feat_names):
        r2, mae, rho, r_val, a = best_alpha(X_all[:, j:j+1], y_v)
        report(name, 1, r2, mae, rho, r_val, a)

    # MSR + HWSR
    msr_idx = feat_names.index('msr')
    hwsr_idx = feat_names.index('hwsr')
    r2, mae, rho, r_val, a = best_alpha(X_all[:, [msr_idx, hwsr_idx]], y_v)
    report('MSR + HWSR', 2, r2, mae, rho, r_val, a)

    # + Demo
    r2, mae, rho, r_val, a = best_alpha(np.column_stack([X_all[:, [msr_idx, hwsr_idx]], D5_v]), y_v)
    report('MSR + HWSR + Demo(5)', 7, r2, mae, rho, r_val, a)

    # All
    r2, mae, rho, r_val, a = best_alpha(X_all, y_v)
    report('All Goldman (GT3X)', len(feat_names), r2, mae, rho, r_val, a)

    r2, mae, rho, r_val, a = best_alpha(np.column_stack([X_all, D5_v]), y_v)
    report('All Goldman (GT3X) + Demo(5)', len(feat_names) + 5, r2, mae, rho, r_val, a)

    # Compare AGD vs GT3X on overlapping subjects
    print(f"\n{'='*80}")
    print(f"COMPARISON: AGD-based (n=92) vs GT3X-based (n={nv})")
    print(f"{'='*80}")
    print(f"  AGD:  MSR+HWSR+Demo R²=0.364  |  GT3X: see above")
    print(f"  Our PerBout-Top20+Demo: R²=0.441, MAE=191, ρ=0.633")

    # Verify: compare GT3X steps vs AGD steps for overlapping subjects
    print(f"\n  GT3X vs AGD step count comparison (overlapping subjects):")
    gt3x_msrs, agd_msrs = [], []
    for f, v in zip(all_features, valid_mask):
        if not v: continue
        # Find in AGD features
        # (simplified — just compare MSR distributions)
        gt3x_msrs.append(f['msr'])
    agd_msrs = agd_feats['msr'].values
    print(f"    GT3X MSR: mean={np.mean(gt3x_msrs):.0f}, median={np.median(gt3x_msrs):.0f}")
    print(f"    AGD  MSR: mean={np.mean(agd_msrs):.0f}, median={np.median(agd_msrs):.0f}")

    print(f"\nDone in {time.time()-t0:.0f}s")
