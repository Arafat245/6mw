#!/usr/bin/env python3
"""
Extract features from AGD files (60-sec epoch data) for home monitoring.
Features: Activity Profile (10), Inclinometer (6), Walking Bout Quality (6) = 22 total.
"""
import numpy as np, pandas as pd, sqlite3, os, warnings
from pathlib import Path
from scipy.stats import entropy
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent  # project root
ACCEL_DIR = BASE / 'Accel files'
FS_EPOCH = 1  # 1 epoch per minute


def find_agd_file(cohort, subj_id):
    """Find the AGD file for a given subject."""
    patterns = [
        f"{cohort}{subj_id:02d}_OPT",
        f"{cohort}{subj_id:02d}_OPT-*",
    ]
    for folder in ACCEL_DIR.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name
        sid = f"{cohort}{subj_id:02d}"
        if name.startswith(sid):
            for f in folder.iterdir():
                if f.suffix == '.agd':
                    return f
    return None


def load_agd(agd_path):
    """Load AGD data from SQLite database."""
    conn = sqlite3.connect(str(agd_path))
    data = pd.read_sql('SELECT * FROM data', conn)
    conn.close()
    return data


def extract_agd_features(data):
    """Extract 22 features from AGD epoch data."""
    f = {}

    axis1 = data['axis1'].values.astype(float)
    axis2 = data['axis2'].values.astype(float)
    axis3 = data['axis3'].values.astype(float)
    steps = data['steps'].values.astype(float)
    standing = data['inclineStanding'].values.astype(float)
    sitting = data['inclineSitting'].values.astype(float)
    lying = data['inclineLying'].values.astype(float)
    off = data['inclineOff'].values.astype(float)

    vm = np.sqrt(axis1**2 + axis2**2 + axis3**2)
    n_epochs = len(data)

    # Wear time: epochs where device is not "off" for full 60 seconds
    wear_mask = off < 55  # at least 5 seconds of wear in the epoch
    n_wear = wear_mask.sum()
    if n_wear < 60:  # less than 1 hour of wear
        return None

    # Estimate number of days
    n_days = max(1, n_wear / (14 * 60))  # ~14 hours of wear per day

    # ═══ Activity Profile Features (10) ═══

    # Daily steps
    total_steps = steps[wear_mask].sum()
    f['daily_steps_mean'] = total_steps / n_days
    f['daily_steps_median'] = np.median(steps[wear_mask & (steps > 0)]) if (steps[wear_mask] > 0).any() else 0

    # Steps per active hour
    active_mask = wear_mask & (axis1 > 100)
    active_hours = active_mask.sum() / 60
    f['steps_per_active_hour'] = total_steps / max(active_hours, 0.1)

    # Active and MVPA minutes per day
    f['active_minutes_per_day'] = active_mask.sum() / n_days
    mvpa_mask = wear_mask & (axis1 > 2020)  # Freedson MVPA cutpoint
    f['mvpa_minutes_per_day'] = mvpa_mask.sum() / n_days

    # Sedentary percentage
    sedentary_mask = wear_mask & (axis1 < 100)
    f['sedentary_pct'] = sedentary_mask.sum() / max(n_wear, 1)

    # Activity entropy
    counts_wear = axis1[wear_mask]
    hist, _ = np.histogram(counts_wear, bins=20, density=True)
    hist = hist[hist > 0]
    f['activity_entropy'] = entropy(hist) if len(hist) > 1 else 0

    # Activity IQR (during active periods)
    if active_mask.sum() > 5:
        f['activity_iqr'] = np.percentile(axis1[active_mask], 75) - np.percentile(axis1[active_mask], 25)
    else:
        f['activity_iqr'] = 0

    # Activity CV
    if counts_wear.mean() > 0:
        f['activity_cv'] = counts_wear.std() / (counts_wear.mean() + 1e-12)
    else:
        f['activity_cv'] = 0

    # Peak activity hour (rough estimate from epoch index)
    # Group by hour-of-day modulo 24
    epoch_hour = np.arange(n_epochs) % (24 * 60) // 60  # hour of day
    hourly_activity = pd.Series(axis1).groupby(epoch_hour).mean()
    f['peak_activity_hour'] = hourly_activity.idxmax() if len(hourly_activity) > 0 else 12

    # ═══ Inclinometer Features (6) ═══

    wear_time_sec = n_wear * 60  # total wear time in seconds

    # Standing percentage
    total_standing = standing[wear_mask].sum()
    f['standing_pct'] = total_standing / max(wear_time_sec, 1)

    # Standing bout mean duration
    is_standing = wear_mask & (standing > 30)  # epoch is mostly standing
    standing_bouts = []
    in_bout = False
    bout_len = 0
    for i in range(len(is_standing)):
        if is_standing[i]:
            if not in_bout:
                in_bout = True
                bout_len = 1
            else:
                bout_len += 1
        else:
            if in_bout:
                standing_bouts.append(bout_len)
                in_bout = False
    if in_bout:
        standing_bouts.append(bout_len)
    f['standing_bout_mean_dur'] = np.mean(standing_bouts) if standing_bouts else 0

    # Sit-to-stand transitions per day
    is_sitting_epoch = wear_mask & (sitting > 30)
    is_standing_epoch = wear_mask & (standing > 30)
    transitions = 0
    for i in range(1, len(data)):
        if is_sitting_epoch[i-1] and is_standing_epoch[i]:
            transitions += 1
    f['sit_to_stand_transitions'] = transitions / n_days

    # Lying percentage
    total_lying = lying[wear_mask].sum()
    f['lying_pct'] = total_lying / max(wear_time_sec, 1)

    # Standing to walking ratio
    walking_time_sec = (steps[wear_mask] > 0).sum() * 60
    f['standing_to_walking_ratio'] = total_standing / max(walking_time_sec, 1)

    # Upright percentage (standing + walking)
    upright_sec = total_standing + walking_time_sec
    f['upright_pct'] = upright_sec / max(wear_time_sec, 1)

    # ═══ Walking Bout Quality Features (6) ═══

    # Detect walking bouts from step counts
    is_walking_epoch = wear_mask & (steps > 0)
    walk_bouts = []
    in_wb = False
    wb_steps = 0
    wb_dur = 0
    for i in range(len(is_walking_epoch)):
        if is_walking_epoch[i]:
            if not in_wb:
                in_wb = True
                wb_steps = steps[i]
                wb_dur = 1
            else:
                wb_steps += steps[i]
                wb_dur += 1
        else:
            if in_wb:
                walk_bouts.append((wb_dur, wb_steps))
                in_wb = False
    if in_wb:
        walk_bouts.append((wb_dur, wb_steps))

    f['walking_bout_count_per_day'] = len(walk_bouts) / n_days if walk_bouts else 0

    if walk_bouts:
        durs = [b[0] for b in walk_bouts]
        bout_steps = [b[1] for b in walk_bouts]
        f['longest_walking_bout_steps'] = max(bout_steps)
        f['walking_bout_mean_steps'] = np.mean(bout_steps)
        f['walking_bout_duration_cv'] = np.std(durs) / (np.mean(durs) + 1e-12)
        # Steps in long bouts (>5 min)
        long_bout_steps = sum(s for d, s in walk_bouts if d >= 5)
        f['steps_in_long_bouts_pct'] = long_bout_steps / max(total_steps, 1)
        # Walking cadence from steps
        walking_epochs_with_steps = steps[is_walking_epoch]
        f['walking_cadence_from_steps'] = np.median(walking_epochs_with_steps) if len(walking_epochs_with_steps) > 0 else 0
    else:
        f['longest_walking_bout_steps'] = 0
        f['walking_bout_mean_steps'] = 0
        f['walking_bout_duration_cv'] = 0
        f['steps_in_long_bouts_pct'] = 0
        f['walking_cadence_from_steps'] = 0

    return f


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score
    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import PLSRegression

    ids = pd.read_csv('feats/target_6mwd.csv')
    valid = np.ones(len(ids), dtype=bool)
    valid[ids[(ids['cohort']=='M') & (ids['subj_id']==22)].index] = False
    ids102 = ids[valid].reset_index(drop=True)

    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = []
    for _, r in ids102.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        clinic_valid.append((PREPROC2/fn).exists())
    clinic_valid = np.array(clinic_valid)
    subj = ids102[clinic_valid].reset_index(drop=True)
    y = subj['sixmwd'].values.astype(float)
    n = len(y)

    # Extract AGD features for all subjects
    print(f"Extracting AGD features (n={n})...")
    agd_rows = []
    for idx, (_, r) in enumerate(subj.iterrows()):
        agd_path = find_agd_file(r['cohort'], int(r['subj_id']))
        if agd_path is not None:
            data = load_agd(agd_path)
            feats = extract_agd_features(data)
            agd_rows.append(feats)
        else:
            agd_rows.append(None)
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{n}", flush=True)
    print(f"  {n}/{n} done")

    # Build feature matrix
    valid_rows = [r for r in agd_rows if r is not None]
    all_cols = list(valid_rows[0].keys()) if valid_rows else []
    print(f"  Features: {len(all_cols)}")
    print(f"  Valid subjects: {len(valid_rows)}/{n}")

    all_r = []
    for row in agd_rows:
        if row is None:
            all_r.append({k: np.nan for k in all_cols})
        else:
            all_r.append(row)
    X_agd = pd.DataFrame(all_r)
    for c in X_agd.columns:
        if X_agd[c].isna().any():
            X_agd[c] = X_agd[c].fillna(X_agd[c].median())
    X_agd_arr = X_agd.values.astype(float)

    # Save
    X_agd.to_csv('feats/home_agd_features.csv', index=False)
    print(f"  Saved feats/home_agd_features.csv")

    # Load existing features
    cidx = np.where(clinic_valid)[0]
    d = np.load('feats/home_hybrid_v2_features.npz', allow_pickle=True)
    X_home_gait = d['X_gait'][cidx, :11]

    demo = pd.read_excel('SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj.merge(demo, on=['cohort','subj_id'], how='left')
    p['cohort_M'] = (p['cohort']=='M').astype(int)
    for c in ['Age','Sex']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo_3 = p[['cohort_M','Age','Sex']].values.astype(float)
    for j in range(3):
        m = np.isnan(X_demo_3[:, j])
        if m.any(): X_demo_3[m, j] = np.nanmedian(X_demo_3[:, j])

    # Clinic gait for PLS
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from clinic.reproduce_c2 import extract_gait10
    gait_cols = ['cadence_hz','step_time_cv_pct','acf_step_regularity','hr_ap','hr_vt',
                 'ml_rms_g','ml_spectral_entropy','jerk_mean_abs_gps','enmo_mean_g',
                 'cadence_slope_per_min']
    gait_rows_c = []
    for _, r in subj.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        gait_rows_c.append(extract_gait10(pd.read_csv(PREPROC2/fn)))
    X_clinic_gait = pd.DataFrame(gait_rows_c)[gait_cols].values.astype(float)
    for j in range(10):
        m = np.isnan(X_clinic_gait[:, j])
        if m.any(): X_clinic_gait[m, j] = np.nanmedian(X_clinic_gait[:, j])

    # LOO functions
    def loo(X, y):
        pr = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=10)
            m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
        return round(r2_score(y, pr), 4)

    def loo_pls(X_home, X_clinic, X_demo, y, nc=2):
        n = len(y); pr = np.zeros(n)
        has_demo = X_demo.shape[1] > 0 if len(X_demo.shape) > 1 else False
        for te in range(n):
            tr = np.ones(n, dtype=bool); tr[te] = False
            sh=StandardScaler(); sc=StandardScaler()
            Xht=sh.fit_transform(X_home[tr]); Xhe=sh.transform(X_home[te:te+1])
            Xct=sc.fit_transform(X_clinic[tr])
            pls=PLSRegression(n_components=nc, scale=False); pls.fit(Xht, Xct)
            Xhm=pls.transform(Xht); Xhem=pls.transform(Xhe)
            if has_demo:
                sd=StandardScaler()
                Xdt=sd.fit_transform(X_demo[tr]); Xde=sd.transform(X_demo[te:te+1])
                Xf=np.column_stack([Xhm,Xdt]); Xfe=np.column_stack([Xhem,Xde])
            else:
                Xf=Xhm; Xfe=Xhem
            m=Ridge(alpha=10); m.fit(Xf, y[tr]); pr[te]=m.predict(Xfe)[0]
        return round(r2_score(y, pr), 4)

    # Evaluate
    print("\n" + "="*75)
    print(f"{'Config':40s} {'R²':>8s}")
    print("="*75)

    # Baselines
    print("--- Baselines ---")
    r2 = loo(np.column_stack([X_home_gait, X_demo_3]), y)
    print(f"  {'Gait+Demo (current best no PLS)':38s} {r2:>8.4f}")
    r2 = loo_pls(X_home_gait, X_clinic_gait, X_demo_3, y)
    print(f"  {'PLS(Gait)+Demo (current best)':38s} {r2:>8.4f}")

    # AGD alone
    print("\n--- AGD Features ---")
    r2 = loo(X_agd_arr, y)
    print(f"  {'AGD (22f)':38s} {r2:>8.4f}")
    r2 = loo(np.column_stack([X_agd_arr, X_demo_3]), y)
    print(f"  {'AGD+Demo (25f)':38s} {r2:>8.4f}")

    # AGD subsets
    activity_cols = ['daily_steps_mean','daily_steps_median','steps_per_active_hour',
                     'active_minutes_per_day','mvpa_minutes_per_day','sedentary_pct',
                     'activity_entropy','activity_iqr','activity_cv','peak_activity_hour']
    incline_cols = ['standing_pct','standing_bout_mean_dur','sit_to_stand_transitions',
                    'lying_pct','standing_to_walking_ratio','upright_pct']
    walk_cols = ['walking_bout_count_per_day','longest_walking_bout_steps','walking_bout_mean_steps',
                 'walking_bout_duration_cv','steps_in_long_bouts_pct','walking_cadence_from_steps']

    X_activity = X_agd[activity_cols].values.astype(float)
    X_incline = X_agd[incline_cols].values.astype(float)
    X_walk = X_agd[walk_cols].values.astype(float)

    r2 = loo(np.column_stack([X_activity, X_demo_3]), y)
    print(f"  {'Activity+Demo (13f)':38s} {r2:>8.4f}")
    r2 = loo(np.column_stack([X_incline, X_demo_3]), y)
    print(f"  {'Inclinometer+Demo (9f)':38s} {r2:>8.4f}")
    r2 = loo(np.column_stack([X_walk, X_demo_3]), y)
    print(f"  {'WalkQuality+Demo (9f)':38s} {r2:>8.4f}")

    # Combined with Gait
    print("\n--- Combined with Gait ---")
    r2 = loo(np.column_stack([X_home_gait, X_agd_arr, X_demo_3]), y)
    print(f"  {'Gait+AGD+Demo (36f)':38s} {r2:>8.4f}")
    r2 = loo(np.column_stack([X_home_gait, X_activity, X_demo_3]), y)
    print(f"  {'Gait+Activity+Demo (24f)':38s} {r2:>8.4f}")
    r2 = loo(np.column_stack([X_home_gait, X_walk, X_demo_3]), y)
    print(f"  {'Gait+WalkQuality+Demo (20f)':38s} {r2:>8.4f}")

    # PLS with AGD
    print("\n--- PLS with AGD ---")
    r2 = loo_pls(np.column_stack([X_home_gait, X_agd_arr]),
                 X_clinic_gait, X_demo_3, y)
    print(f"  {'PLS(Gait+AGD)+Demo':38s} {r2:>8.4f}")
    r2 = loo_pls(np.column_stack([X_home_gait, X_activity]),
                 X_clinic_gait, X_demo_3, y)
    print(f"  {'PLS(Gait+Activity)+Demo':38s} {r2:>8.4f}")

    print("="*75)
