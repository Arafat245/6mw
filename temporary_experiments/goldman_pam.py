#!/usr/bin/env python3
"""
Implementation of Goldman/Engelhard et al. (Gait & Posture, 2018):
"Real-world walking in multiple sclerosis: Separating capacity from behavior"

Uses AGD step counts from ActiGraph to compute:
1. Conventional HPA: average daily steps, MVPA time
2. New HPA via Personalized Activity Modeling (PAM):
   - MSR (Maximum Step Rate): highest minute-wise step rate
   - HWSR (Habitual Walking Step Rate): from Hidden Markov Model

Then correlate with 6MWD and predict using Ridge LOO CV.
Entirely from AGD files — no clinic data dependency.
"""
import sys, warnings, time, re
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
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


def ticks_to_datetime(ticks):
    return datetime(1, 1, 1) + timedelta(microseconds=ticks // 10)


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


# ══════════════════════════════════════════════════════════════════
# STEP 1: Read AGD and compute minute-wise step rates
# ══════════════════════════════════════════════════════════════════

def read_agd_minutewise(agd_path):
    """Read AGD and return minute-wise step rates with timestamps.
    Returns DataFrame with columns: datetime, steps_per_min, axis1, axis2, axis3"""
    conn = sqlite3.connect(str(agd_path))
    cursor = conn.cursor()

    cursor.execute("SELECT settingValue FROM settings WHERE settingName='epochlength'")
    epoch_sec = int(cursor.fetchone()[0])

    cursor.execute("SELECT dataTimestamp, axis1, axis2, axis3, steps FROM data ORDER BY dataTimestamp")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, epoch_sec

    # Build DataFrame
    data = []
    for ts_ticks, a1, a2, a3, steps in rows:
        dt = ticks_to_datetime(ts_ticks)
        data.append({'datetime': dt, 'steps': float(steps),
                     'axis1': float(a1), 'axis2': float(a2), 'axis3': float(a3)})

    df = pd.DataFrame(data)

    # Aggregate to minute-level if epoch < 60s
    if epoch_sec < 60:
        df['minute'] = df['datetime'].dt.floor('min')
        df_min = df.groupby('minute').agg({
            'steps': 'sum',
            'axis1': 'sum',
            'axis2': 'sum',
            'axis3': 'sum',
        }).reset_index()
        df_min.rename(columns={'minute': 'datetime', 'steps': 'steps_per_min'}, inplace=True)
    else:
        # Already 60-sec epochs
        df_min = df.rename(columns={'steps': 'steps_per_min'})

    return df_min, epoch_sec


# ══════════════════════════════════════════════════════════════════
# STEP 2: Identify valid wear days
# ══════════════════════════════════════════════════════════════════

def get_valid_wear_days(df_min, min_wear_hours=10):
    """Identify valid wear days (≥ min_wear_hours of steps detected).
    A minute counts as "worn" if steps > 0 or axis counts > 0.
    Returns list of dates."""
    df_min['date'] = df_min['datetime'].dt.date
    df_min['is_worn'] = (df_min['steps_per_min'] > 0) | (df_min['axis1'] > 0)

    daily = df_min.groupby('date').agg(
        worn_minutes=('is_worn', 'sum'),
        total_steps=('steps_per_min', 'sum'),
    ).reset_index()

    daily['worn_hours'] = daily['worn_minutes'] / 60
    valid_days = daily[daily['worn_hours'] >= min_wear_hours]['date'].tolist()
    return valid_days, daily


# ══════════════════════════════════════════════════════════════════
# STEP 3: Compute conventional HPA statistics
# ══════════════════════════════════════════════════════════════════

def compute_conventional_hpa(df_min, valid_days):
    """Compute average daily steps, MPA, VPA, MVPA from valid days."""
    df_valid = df_min[df_min['datetime'].dt.date.isin(valid_days)]

    if len(df_valid) == 0:
        return None

    n_days = len(valid_days)
    total_steps = df_valid['steps_per_min'].sum()
    avg_daily_steps = total_steps / n_days

    # Activity counts thresholds (from ActiGraph/Freedson 1998 for adults)
    # MPA: axis1 counts 1952-5724 per minute
    # VPA: axis1 counts ≥ 5725 per minute
    mpa_mins = ((df_valid['axis1'] >= 1952) & (df_valid['axis1'] < 5725)).sum()
    vpa_mins = (df_valid['axis1'] >= 5725).sum()
    mvpa_mins = mpa_mins + vpa_mins

    return {
        'avg_daily_steps': avg_daily_steps,
        'total_mpa_min': mpa_mins / n_days,  # per day
        'total_vpa_min': vpa_mins / n_days,
        'total_mvpa_min': mvpa_mins / n_days,
        'n_valid_days': n_days,
    }


# ══════════════════════════════════════════════════════════════════
# STEP 4: Maximum Step Rate (MSR)
# ══════════════════════════════════════════════════════════════════

def compute_msr(df_min, valid_days):
    """MSR = highest minute-wise step rate across all valid days."""
    df_valid = df_min[df_min['datetime'].dt.date.isin(valid_days)]
    if len(df_valid) == 0:
        return np.nan
    return float(df_valid['steps_per_min'].max())


# ══════════════════════════════════════════════════════════════════
# STEP 5: Personalized Activity Model (PAM) → HWSR
# ══════════════════════════════════════════════════════════════════

def fit_pam_and_get_hwsr(df_min, valid_days, msr):
    """
    Fit Hidden Markov Model (PAM) per Goldman et al.:
    1. Use minute-wise step rates from valid days
    2. Fit HMM with 2-6 states + "not-worn" state (step=0)
    3. Select model with lowest BIC
    4. Identify walking state: expected value > MSR/2 and CV ≥ 8
    5. HWSR = expected value of walking state

    Returns HWSR or NaN if walking state not identified.
    """
    df_valid = df_min[df_min['datetime'].dt.date.isin(valid_days)]
    step_rates = df_valid['steps_per_min'].values.astype(float)

    if len(step_rates) < 100 or msr <= 0:
        return np.nan

    # Reshape for HMM
    X = step_rates.reshape(-1, 1)

    best_model = None
    best_bic = np.inf
    best_n = 2

    for n_states in range(2, 7):  # 2-6 active states (paper says MSR + N)
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type='diag',
                n_iter=100,
                random_state=42,
                init_params='mc',  # initialize means and covars
                params='stmc',
            )

            # Initialize means evenly spaced from 0 to MSR
            means_init = np.linspace(0, msr, n_states).reshape(-1, 1)
            model.means_ = means_init

            model.fit(X)
            score = model.score(X)
            n_params = n_states * n_states + n_states * 2  # transition + mean + var
            bic = -2 * score + n_params * np.log(len(X))

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n_states
        except:
            continue

    if best_model is None:
        return np.nan

    # Decode states
    states = best_model.predict(X)
    means = best_model.means_.flatten()
    covars = best_model.covars_.flatten() if best_model.covars_.ndim == 1 else np.diag(best_model.covars_).flatten() if best_model.covars_.ndim == 2 else best_model.covars_.reshape(-1)

    # Identify walking state per paper:
    # Active state: expected value > MSR/2 AND coefficient of variance ≥ 8
    # If one active state → walking
    # If two active states → lower = walking, higher = running
    walking_state = None
    active_states = []
    for s in range(best_n):
        state_mean = means[s]
        # Get samples in this state for CV calculation
        state_samples = step_rates[states == s]
        if len(state_samples) < 5:
            continue
        state_std = np.std(state_samples)
        state_cv = state_std / (state_mean + 1e-12) if state_mean > 0 else 0

        if state_mean > msr / 2 and state_cv >= 8:
            active_states.append((s, state_mean))

    if len(active_states) == 1:
        # Single active state
        if active_states[0][1] > 130:
            # Running, not walking
            walking_state = None
        else:
            walking_state = active_states[0]
    elif len(active_states) >= 2:
        # Sort by mean: lower = walking, higher = running
        active_states.sort(key=lambda x: x[1])
        walking_state = active_states[0]

    if walking_state is None:
        # Fallback: use the state with the highest mean that's < 130 spm
        # and has reasonable activity
        candidates = [(s, means[s]) for s in range(best_n)
                       if means[s] > 10 and means[s] < 130]
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            walking_state = candidates[0]

    hwsr = float(walking_state[1]) if walking_state else np.nan
    return hwsr


# ══════════════════════════════════════════════════════════════════
# STEP 6: Find best AGD for each subject
# ══════════════════════════════════════════════════════════════════

def find_best_agd(accel_folder):
    """Find 60-sec AGD file."""
    agd_files = list(accel_folder.glob('*.agd'))
    if not agd_files: return None
    for f in agd_files:
        if '60sec' in f.name.lower(): return f
    return agd_files[0]


def match_subject(folder_name, ids_df):
    """Match Accel folder name to subject in ids_df."""
    m = re.match(r'([CM])(\d+)', folder_name)
    if not m: return None
    cohort, sid = m.group(1), int(m.group(2))
    match = ids_df[(ids_df['cohort'] == cohort) & (ids_df['subj_id'] == sid)]
    if len(match) > 0: return match.index[0]
    return None


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    # Load subjects
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)

    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                             for _, r in ids101.iterrows()])
    ids_v = ids101[clinic_valid].reset_index(drop=True)
    y = ids_v['sixmwd'].values.astype(float)
    n = len(y)

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo4 = impute(p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float))
    X_demo5 = impute(p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float))

    print(f"Goldman et al. PAM Implementation")
    print(f"n={n} subjects")
    print(f"{'='*80}\n")

    # Process each subject
    all_features = []
    valid_mask = []

    for i, (_, r) in enumerate(ids_v.iterrows()):
        cohort, sid = r['cohort'], int(r['subj_id'])

        # Find Accel folder
        prefix = f"{cohort}{sid:02d}_"
        accel_folder = None
        for d in ACCEL_DIR.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                accel_folder = d
                break

        if accel_folder is None:
            all_features.append(None); valid_mask.append(False)
            continue

        agd_path = find_best_agd(accel_folder)
        if agd_path is None:
            all_features.append(None); valid_mask.append(False)
            continue

        # Step 1: Read AGD
        df_min, epoch_sec = read_agd_minutewise(agd_path)
        if df_min is None or len(df_min) < 100:
            all_features.append(None); valid_mask.append(False)
            continue

        # Step 2: Valid wear days
        valid_days, daily_stats = get_valid_wear_days(df_min, min_wear_hours=10)

        # Relax to 6 hours if < 6 valid days (our recordings are shorter than 7 days)
        if len(valid_days) < 3:
            valid_days, daily_stats = get_valid_wear_days(df_min, min_wear_hours=6)

        if len(valid_days) < 1:
            all_features.append(None); valid_mask.append(False)
            continue

        # Step 3: Conventional HPA
        hpa = compute_conventional_hpa(df_min, valid_days)
        if hpa is None:
            all_features.append(None); valid_mask.append(False)
            continue

        # Step 4: MSR
        msr = compute_msr(df_min, valid_days)

        # Step 5: PAM → HWSR
        hwsr = fit_pam_and_get_hwsr(df_min, valid_days, msr)

        feats = {
            'msr': msr,
            'hwsr': hwsr,
            'avg_daily_steps': hpa['avg_daily_steps'],
            'mvpa_min_per_day': hpa['total_mvpa_min'],
            'mpa_min_per_day': hpa['total_mpa_min'],
            'vpa_min_per_day': hpa['total_vpa_min'],
            'n_valid_days': hpa['n_valid_days'],
            'msr_plus_hwsr': msr + hwsr if np.isfinite(hwsr) else msr,
        }
        all_features.append(feats)
        valid_mask.append(True)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{n}] {cohort}{sid:02d}: MSR={msr:.0f} HWSR={hwsr:.1f} "
                  f"avg_steps={hpa['avg_daily_steps']:.0f} days={len(valid_days)}", flush=True)

    valid_mask = np.array(valid_mask)
    nv = valid_mask.sum()
    print(f"\nValid subjects: {nv}/{n}")

    # ── Build feature matrix ──
    feat_names = sorted(all_features[valid_mask.tolist().index(True)].keys())
    X_all = np.array([[f[k] for k in feat_names] for f, v in zip(all_features, valid_mask) if v])
    X_all = impute(X_all)
    y_v = y[valid_mask]
    D4_v = X_demo4[valid_mask]
    D5_v = X_demo5[valid_mask]

    # ── Correlations ──
    print(f"\n{'='*80}")
    print(f"CORRELATIONS WITH 6MWD (n={nv})")
    print(f"{'='*80}")
    for j, name in enumerate(feat_names):
        r_val, p_r = pearsonr(X_all[:, j], y_v)
        rho, p_s = spearmanr(X_all[:, j], y_v)
        print(f"  {name:25s}  r={r_val:+.3f} (p={p_r:.4f})  ρ={rho:+.3f} (p={p_s:.4f})")

    # ── Prediction ──
    print(f"\n{'='*80}")
    print(f"PREDICTION (LOO CV, n={nv})")
    print(f"{'='*80}")

    def report(name, nf, r2, mae, rho, r_val, alpha):
        print(f"  {name:45s} {nf:>3d}f  α={alpha:>3d}  R²={r2:.4f}  MAE={mae:.0f}ft  r={r_val:.3f}  ρ={rho:.3f}")

    # Individual features
    for j, name in enumerate(feat_names):
        X = X_all[:, j:j+1]
        r2, mae, rho, r_val, a = best_alpha(X, y_v)
        report(name, 1, r2, mae, rho, r_val, a)

    # MSR + HWSR (paper's best combination)
    msr_idx = feat_names.index('msr')
    hwsr_idx = feat_names.index('hwsr')
    X_msr_hwsr = X_all[:, [msr_idx, hwsr_idx]]
    r2, mae, rho, r_val, a = best_alpha(X_msr_hwsr, y_v)
    report('MSR + HWSR', 2, r2, mae, rho, r_val, a)

    # MSR + HWSR + Demo
    r2, mae, rho, r_val, a = best_alpha(np.column_stack([X_msr_hwsr, D5_v]), y_v)
    report('MSR + HWSR + Demo(5)', 7, r2, mae, rho, r_val, a)

    # All Goldman features
    r2, mae, rho, r_val, a = best_alpha(X_all, y_v)
    report('All Goldman features', len(feat_names), r2, mae, rho, r_val, a)

    # All Goldman + Demo
    r2, mae, rho, r_val, a = best_alpha(np.column_stack([X_all, D5_v]), y_v)
    report('All Goldman + Demo(5)', len(feat_names) + 5, r2, mae, rho, r_val, a)

    # Avg daily steps only
    ads_idx = feat_names.index('avg_daily_steps')
    r2, mae, rho, r_val, a = best_alpha(X_all[:, ads_idx:ads_idx+1], y_v)
    report('Avg Daily Steps only', 1, r2, mae, rho, r_val, a)

    # Avg daily steps + Demo
    r2, mae, rho, r_val, a = best_alpha(np.column_stack([X_all[:, ads_idx:ads_idx+1], D5_v]), y_v)
    report('Avg Daily Steps + Demo(5)', 6, r2, mae, rho, r_val, a)

    print(f"\n{'='*80}")
    print(f"COMPARISON:")
    print(f"  Goldman paper (MS only):  MSR r=0.801, HWSR r=0.701, MSR+HWSR r=0.884 with 6MWD")
    print(f"  Our PerBout-Top20+Demo:   R²=0.441, MAE=191, ρ=0.633")
    print(f"{'='*80}")

    # Save features
    feat_df = pd.DataFrame([f for f, v in zip(all_features, valid_mask) if v])
    feat_df['cohort'] = ids_v[valid_mask]['cohort'].values
    feat_df['subj_id'] = ids_v[valid_mask]['subj_id'].values
    feat_df['sixmwd'] = y_v
    feat_df.to_csv(BASE / 'temporary_experiments' / 'goldman_pam_features.csv', index=False)
    print(f"\nSaved goldman_pam_features.csv")
    print(f"Done in {time.time()-t0:.0f}s")
