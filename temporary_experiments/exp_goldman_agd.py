#!/usr/bin/env python3
"""
Goldman et al. (2018) features from AGD files.
Features: MSR, HWSR, Avg Daily Steps, MVPA minutes, total active minutes.
Two experiments: (1) Goldman features alone, (2) Goldman + Demo.
"""
import sqlite3, warnings, time, re
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

TICKS_PER_SEC = 10_000_000
EPOCH_DIFF = 62135596800
FT2M = 0.3048
BASE = Path(__file__).parent.parent

def read_agd(agd_path):
    conn = sqlite3.connect(str(agd_path))
    df = pd.read_sql('SELECT * FROM data', conn)
    conn.close()
    df['unix_ts'] = df['dataTimestamp'] / TICKS_PER_SEC - EPOCH_DIFF
    df['dt'] = pd.to_datetime(df['unix_ts'], unit='s')
    df['worn'] = (df['axis1'] > 0) | (df['axis2'] > 0) | (df['axis3'] > 0) | (df['steps'] > 0)
    df['date'] = df['dt'].dt.date
    return df

def get_valid_days(df, min_wear_hrs=10):
    daily_wear = df.groupby('date')['worn'].sum() / 60
    valid_dates = daily_wear[daily_wear >= min_wear_hrs].index
    return df[df['date'].isin(valid_dates)]

def fit_pam(step_rates):
    """Fit HMM to get MSR and HWSR (Goldman 2018)."""
    sr = step_rates.values.reshape(-1, 1).astype(float)
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
            n_params = n_states * 3  # mean + var + transition
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
    
    # Identify walking state (Goldman: active state criteria)
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
    
    # Find AGD files
    agd_dir = BASE / 'Accel files'
    agd_map = {}
    for agd_f in agd_dir.rglob('*.agd'):
        folder = agd_f.parent.name
        m = re.match(r'([CM]\d+)', folder)
        if m:
            key = m.group(1)
            if key not in agd_map:
                agd_map[key] = agd_f
    
    # Extract Goldman features for each subject
    print(f'Extracting Goldman features from AGD files (n={n})...')
    features = []
    for i, (_, row) in enumerate(subj_df.iterrows()):
        key = row['key']
        agd_path = agd_map.get(key)
        
        feat = {}
        if agd_path is None:
            print(f'  WARNING: No AGD for {key}')
            features.append(feat)
            continue
        
        try:
            df = read_agd(agd_path)
            
            # Use all valid days (>=10h wear) — Goldman used >=6 valid days
            # We'll be more lenient since our subjects may have fewer days
            vdf = get_valid_days(df, min_wear_hrs=10)
            n_valid_days = vdf['date'].nunique()
            
            # If too few valid days, try 8h threshold
            if n_valid_days < 3:
                vdf = get_valid_days(df, min_wear_hrs=8)
                n_valid_days = vdf['date'].nunique()
            
            if n_valid_days < 1:
                # Use all data as fallback
                vdf = df
                n_valid_days = vdf['date'].nunique()
            
            # Conventional HPA
            daily_steps = vdf.groupby('date')['steps'].sum()
            feat['avg_daily_steps'] = daily_steps.mean()
            
            # Activity counts thresholds (Freedson 1998): 
            # MPA: 1952-5724 counts/min, VPA: >=5725 counts/min
            worn_epochs = vdf[vdf['worn']]
            feat['total_mpa_min'] = ((worn_epochs['axis1'] >= 1952) & (worn_epochs['axis1'] < 5725)).sum() / max(n_valid_days, 1)
            feat['total_vpa_min'] = (worn_epochs['axis1'] >= 5725).sum() / max(n_valid_days, 1)
            feat['total_mvpa_min'] = (worn_epochs['axis1'] >= 1952).sum() / max(n_valid_days, 1)
            
            # Active minutes per day
            feat['active_min_per_day'] = (vdf.groupby('date').apply(lambda x: (x['steps'] > 0).sum())).mean()
            
            # New HPA: MSR and HWSR
            worn_steps = worn_epochs['steps']
            msr, hwsr, n_states = fit_pam(worn_steps)
            feat['msr'] = msr
            feat['hwsr'] = hwsr
            feat['n_valid_days'] = n_valid_days
            
            # Additional: step rate variability
            active_steps = worn_steps[worn_steps > 0]
            if len(active_steps) > 1:
                feat['step_rate_cv'] = active_steps.std() / (active_steps.mean() + 1e-12)
                feat['step_rate_median'] = active_steps.median()
                feat['step_rate_p90'] = np.percentile(active_steps, 90)
            
        except Exception as e:
            print(f'  ERROR {key}: {e}')
        
        features.append(feat)
        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{n}] extracted', flush=True)
    
    feat_df = pd.DataFrame(features)
    print(f'\nFeature matrix: {feat_df.shape}')
    print(f'NaN counts:\n{feat_df.isna().sum()}')
    print(f'\nFeature stats:')
    print(feat_df.describe().round(1))
    
    # Prepare feature matrices
    goldman_cols = ['avg_daily_steps', 'msr', 'hwsr', 'total_mvpa_min', 'active_min_per_day',
                    'step_rate_cv', 'step_rate_median', 'step_rate_p90']
    
    X_goldman = impute(feat_df[goldman_cols].values.astype(float))
    
    # Core Goldman (MSR + HWSR + daily steps)
    X_core = impute(feat_df[['msr', 'hwsr', 'avg_daily_steps']].values.astype(float))
    
    # MSR + HWSR only (paper's best pair)
    X_msr_hwsr = impute(feat_df[['msr', 'hwsr']].values.astype(float))
    
    print(f'\n{"="*80}')
    print(f'Goldman AGD Features → 6MWD Prediction (LOO Ridge, n={n})')
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
    
    # Individual feature correlations with 6MWD
    print(f'\n{"="*80}')
    print(f'Individual Spearman correlations with 6MWD:')
    for col in goldman_cols:
        vals = feat_df[col].values
        valid = ~np.isnan(vals)
        if valid.sum() > 10:
            r, p = spearmanr(vals[valid], y[valid])
            print(f'  {col:25s}  ρ={r:.3f}  p={p:.1e}  (n={valid.sum()})')
    
    print(f'\nDone in {time.time()-t0:.0f}s')
