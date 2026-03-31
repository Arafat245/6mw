#!/usr/bin/env python3
"""
Goldman et al. (2018) features from GT3X raw accelerometer data.
Steps estimated via peak detection on VM signal.
"""
import warnings, time, re
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

FT2M = 0.3048
BASE = Path(__file__).parent.parent

def estimate_steps_per_minute(xyz, fs=30):
    """
    Estimate steps per minute from raw XYZ accelerometer.
    Uses peak detection on bandpass-filtered VM signal.
    Returns array of steps per 60-second epoch.
    """
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    
    # Bandpass 0.5-3 Hz (walking frequency range)
    b, a = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm_filt = filtfilt(b, a, vm - np.mean(vm))
    
    # Split into 60-second epochs
    epoch_len = 60 * fs
    n_epochs = len(vm_filt) // epoch_len
    steps_per_min = np.zeros(n_epochs)
    
    for i in range(n_epochs):
        seg = vm_filt[i * epoch_len : (i + 1) * epoch_len]
        # Peak detection: min distance between steps ~0.3s, prominence > 0.05g
        peaks, props = find_peaks(seg, distance=int(0.3 * fs), prominence=0.05)
        steps_per_min[i] = len(peaks)
    
    return steps_per_min

def compute_enmo_per_minute(xyz, fs=30):
    """Compute ENMO per minute for MVPA classification."""
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    epoch_len = 60 * fs
    n_epochs = len(enmo) // epoch_len
    enmo_per_min = np.zeros(n_epochs)
    for i in range(n_epochs):
        enmo_per_min[i] = enmo[i * epoch_len : (i + 1) * epoch_len].mean()
    return enmo_per_min

def fit_pam(step_rates):
    """Fit HMM for MSR and HWSR."""
    sr = step_rates[step_rates >= 0].reshape(-1, 1).astype(float)
    if len(sr) < 100:
        return np.nan, np.nan, 0
    
    msr = float(sr.max())
    
    best_bic = np.inf
    best_model = None
    best_n = 0
    
    for n_states in range(2, min(7, len(sr) // 20 + 1)):
        try:
            model = GaussianHMM(n_components=n_states, covariance_type='diag',
                               n_iter=100, random_state=42, tol=0.01)
            model.fit(sr)
            log_like = model.score(sr)
            n_params = n_states * 3
            bic = -2 * log_like * len(sr) + n_params * np.log(len(sr))
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n_states
        except:
            continue
    
    if best_model is None:
        return msr, np.nan, 0
    
    means = best_model.means_.flatten()
    active_mask = means > (msr / 2)
    active_idx = np.where(active_mask)[0]
    
    if len(active_idx) == 0:
        nonzero = np.where(means > 5)[0]
        if len(nonzero) > 0:
            active_idx = np.array([nonzero[np.argmax(means[nonzero])]])
    
    if len(active_idx) == 0:
        return msr, np.nan, best_n
    
    if len(active_idx) == 1:
        hwsr = float(means[active_idx[0]])
    else:
        sorted_active = active_idx[np.argsort(means[active_idx])]
        hwsr = float(means[sorted_active[0]])
    
    return msr, hwsr, best_n

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X

def loo_ridge(X, y, alpha):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pr[te] = m.predict(sc.transform(X[te]))
    r2 = r2_score(y, pr)
    return r2, mean_absolute_error(y*FT2M, pr*FT2M), spearmanr(y, pr)[0]

if __name__ == '__main__':
    t0 = time.time()
    
    subj_df = pd.read_csv(BASE / 'home_full_recording_npz' / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)
    
    # Demographics
    demo_xl = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo_xl['cohort'] = demo_xl['ID'].str.extract(r'^([A-Z])')[0]
    demo_xl['subj_id'] = demo_xl['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo_xl, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))
    
    NPZ_DIR = BASE / 'home_full_recording_npz'
    
    print(f'Extracting Goldman features from GT3X/NPZ (n={n})...')
    features = []
    for i, (_, row) in enumerate(subj_df.iterrows()):
        key = row['key']
        feat = {}
        
        try:
            npz = np.load(NPZ_DIR / f'{key}.npz')
            xyz = npz['xyz']
            fs = 30  # hardcoded
            
            # Estimate steps per minute
            steps_per_min = estimate_steps_per_minute(xyz, fs)
            enmo_per_min = compute_enmo_per_minute(xyz, fs)
            
            n_epochs = len(steps_per_min)
            
            # Wear detection: epochs where steps > 0 or ENMO > 0.002
            worn = (steps_per_min > 0) | (enmo_per_min > 0.002)
            
            # Daily breakdown (assume continuous recording)
            epochs_per_day = 24 * 60  # 1440 minutes
            n_days = max(1, n_epochs // epochs_per_day)
            
            # Valid days: >=10h wear
            valid_day_steps = []
            valid_worn_steps = []
            for d in range(n_days):
                s = d * epochs_per_day
                e = min((d + 1) * epochs_per_day, n_epochs)
                day_worn = worn[s:e].sum() / 60  # hours
                if day_worn >= 10:
                    valid_day_steps.append(steps_per_min[s:e].sum())
                    valid_worn_steps.append(steps_per_min[s:e][worn[s:e]])
            
            if not valid_day_steps:
                # Fallback: use all data
                valid_day_steps = [steps_per_min.sum()]
                valid_worn_steps = [steps_per_min[worn]]
            
            feat['avg_daily_steps'] = np.mean(valid_day_steps)
            feat['n_valid_days'] = len(valid_day_steps)
            
            # MVPA: ENMO > 0.1g threshold (common threshold)
            all_worn_enmo = enmo_per_min[worn]
            feat['total_mvpa_min'] = (all_worn_enmo > 0.1).sum() / max(len(valid_day_steps), 1)
            feat['active_min_per_day'] = worn.sum() / max(len(valid_day_steps), 1)
            
            # MSR and HWSR from worn epochs
            all_worn_steps = np.concatenate(valid_worn_steps) if valid_worn_steps else steps_per_min[worn]
            if len(all_worn_steps) > 0:
                msr, hwsr, n_st = fit_pam(all_worn_steps)
                feat['msr'] = msr
                feat['hwsr'] = hwsr
                
                active_steps = all_worn_steps[all_worn_steps > 0]
                if len(active_steps) > 1:
                    feat['step_rate_cv'] = active_steps.std() / (active_steps.mean() + 1e-12)
                    feat['step_rate_median'] = np.median(active_steps)
                    feat['step_rate_p90'] = np.percentile(active_steps, 90)
            
        except Exception as e:
            print(f'  ERROR {key}: {e}')
        
        features.append(feat)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{n}] extracted ({elapsed:.0f}s)', flush=True)
    
    feat_df = pd.DataFrame(features)
    print(f'\nFeature matrix: {feat_df.shape}')
    print(f'NaN counts:\n{feat_df.isna().sum()}')
    print(f'\nFeature stats:')
    print(feat_df.describe().round(1))
    
    goldman_cols = ['avg_daily_steps', 'msr', 'hwsr', 'total_mvpa_min', 'active_min_per_day',
                    'step_rate_cv', 'step_rate_median', 'step_rate_p90']
    
    X_goldman = impute(feat_df[goldman_cols].values.astype(float))
    X_core = impute(feat_df[['msr', 'hwsr', 'avg_daily_steps']].values.astype(float))
    X_msr_hwsr = impute(feat_df[['msr', 'hwsr']].values.astype(float))
    
    print(f'\n{"="*80}')
    print(f'Goldman GT3X Features → 6MWD Prediction (LOO Ridge, n={n})')
    print(f'{"="*80}')
    
    for alpha in [5, 10, 20, 50]:
        print(f'\n--- α={alpha} ---')
        r2, mae, rho = loo_ridge(X_msr_hwsr, y, alpha)
        print(f'  MSR+HWSR (2f):           R²={r2:.3f}  MAE={mae:.1f}m  ρ={rho:.3f}')
        
        r2, mae, rho = loo_ridge(X_core, y, alpha)
        print(f'  MSR+HWSR+DailySteps (3f): R²={r2:.3f}  MAE={mae:.1f}m  ρ={rho:.3f}')
        
        r2, mae, rho = loo_ridge(X_goldman, y, alpha)
        print(f'  All Goldman (8f):         R²={r2:.3f}  MAE={mae:.1f}m  ρ={rho:.3f}')
        
        X_g_demo = np.column_stack([X_goldman, X_demo])
        r2, mae, rho = loo_ridge(X_g_demo, y, alpha)
        print(f'  All Goldman+Demo (12f):   R²={r2:.3f}  MAE={mae:.1f}m  ρ={rho:.3f}')
    
    # Individual correlations
    print(f'\n{"="*80}')
    print(f'Individual Spearman correlations with 6MWD:')
    for col in goldman_cols:
        vals = feat_df[col].values
        valid = ~np.isnan(vals)
        if valid.sum() > 10:
            r, p = spearmanr(vals[valid], y[valid])
            print(f'  {col:25s}  ρ={r:.3f}  p={p:.1e}  (n={valid.sum()})')
    
    print(f'\nDone in {time.time()-t0:.0f}s')
