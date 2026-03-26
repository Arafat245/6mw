#!/usr/bin/env python3
"""
Fixed Home Feature Extraction v2
================================
Fixes:
1. Use ENMO (max(||VM||-1, 0)) instead of raw VM for Activity Index — removes gravity
2. Use global (population-level) thresholds for activity levels, not per-subject percentiles
3. Normalize time-dependent features by recording duration (per-hour rates)
4. Compute DARE from raw data before daytime trimming (if available)
5. Fix RA by using ENMO-based M10/L5 instead of gravity-dominated VM

First pass: compute global thresholds from all subjects.
Second pass: extract features using those thresholds.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, linregress
from scipy.signal import welch, find_peaks
from sklearn.decomposition import PCA

BASE = Path(__file__).parent
FEATS = BASE / 'feats'
OUT = BASE / 'results_raw_pipeline'
DAY_DIR = OUT / 'daytime_segments'
FS = 30.0

ids = pd.read_csv(FEATS / 'target_6mwd.csv')
y = ids['sixmwd'].values.astype(float)

def fname(r):
    return f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"


def compute_enmo(vm):
    """ENMO = max(||VM|| - 1, 0). Removes gravity, keeps only dynamic acceleration."""
    return np.maximum(vm - 1.0, 0.0)


# ═══════════════════════════════════════════════════════════════════
# PASS 1: Compute global thresholds from all subjects
# ═══════════════════════════════════════════════════════════════════
print("Pass 1: Computing global ENMO thresholds...")
all_enmo_seconds = []

for i, (_, r) in enumerate(ids.iterrows()):
    fn = fname(r)
    dp = DAY_DIR / fn
    if dp.exists():
        day = pd.read_csv(dp)[['X', 'Y', 'Z']].values.astype(np.float32)
    else:
        day = pd.read_csv(BASE / 'csv_processed_home' / fn)[['AP', 'ML', 'VT']].values.astype(np.float32)

    vm = np.sqrt(day[:, 0]**2 + day[:, 1]**2 + day[:, 2]**2)
    enmo = compute_enmo(vm)

    # Per-second ENMO
    bs = int(FS)
    n_bins = len(enmo) // bs
    enmo_sec = np.array([np.mean(enmo[j*bs:(j+1)*bs]) for j in range(n_bins)])
    all_enmo_seconds.append(enmo_sec)

# Pool all per-second ENMO values to get population thresholds
pooled = np.concatenate(all_enmo_seconds)
# Thresholds based on population distribution (not per-subject)
# Sedentary: ENMO < population 50th percentile (low movement)
# LIPA: between 50th and 85th percentile
# MVPA: above 85th percentile
sed_thresh = np.percentile(pooled, 50)
mvpa_thresh = np.percentile(pooled, 85)

print(f"  Global thresholds: sedentary < {sed_thresh:.4f}, MVPA > {mvpa_thresh:.4f}")
print(f"  Pooled ENMO stats: mean={pooled.mean():.4f}, median={np.median(pooled):.4f}, p85={mvpa_thresh:.4f}")


# ═══════════════════════════════════════════════════════════════════
# PASS 2: Extract features with fixed thresholds
# ═══════════════════════════════════════════════════════════════════
print("\nPass 2: Extracting features...")

activity_rows = []
circadian_rows = []

for i, (_, r) in enumerate(ids.iterrows()):
    fn = fname(r)
    dp = DAY_DIR / fn
    if dp.exists():
        day = pd.read_csv(dp)[['X', 'Y', 'Z']].values.astype(np.float32)
    else:
        day = pd.read_csv(BASE / 'csv_processed_home' / fn)[['AP', 'ML', 'VT']].values.astype(np.float32)

    vm = np.sqrt(day[:, 0]**2 + day[:, 1]**2 + day[:, 2]**2)
    enmo = compute_enmo(vm)

    # Per-second ENMO
    bs = int(FS)
    n_bins = len(enmo) // bs
    enmo_sec = np.array([np.mean(enmo[j*bs:(j+1)*bs]) for j in range(n_bins)])

    total_hours = n_bins / 3600

    # ════════════════════════════════════════
    # ACTIVITY PROFILE FEATURES (fixed)
    # ════════════════════════════════════════
    af = {}

    # ENMO-based intensity stats (not raw VM)
    af['enmo_mean'] = np.mean(enmo_sec)
    af['enmo_std'] = np.std(enmo_sec)
    af['enmo_iqr'] = np.percentile(enmo_sec, 75) - np.percentile(enmo_sec, 25)
    af['enmo_median'] = np.median(enmo_sec)
    af['enmo_p95'] = np.percentile(enmo_sec, 95)

    # Entropy of ENMO distribution
    hist, _ = np.histogram(enmo_sec, bins=20, density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    af['enmo_entropy'] = -np.sum(hist * np.log2(hist + 1e-12))

    # Activity levels using GLOBAL thresholds (fixed!)
    af['pct_sedentary'] = np.mean(enmo_sec < sed_thresh)
    af['pct_lipa'] = np.mean((enmo_sec >= sed_thresh) & (enmo_sec < mvpa_thresh))
    af['pct_mvpa'] = np.mean(enmo_sec >= mvpa_thresh)

    # MVPA minutes per hour (normalized by duration)
    af['mvpa_min_per_hour'] = (np.sum(enmo_sec >= mvpa_thresh) / 60) / (total_hours + 1e-12)

    # Activity bouts (using global threshold)
    active = enmo_sec >= sed_thresh
    bouts = []
    in_b, bstart = False, 0
    for j in range(len(active)):
        if active[j] and not in_b:
            bstart = j; in_b = True
        elif not active[j] and in_b:
            if (j - bstart) >= 5:
                bouts.append((bstart, j))
            in_b = False
    if in_b and (len(active) - bstart) >= 5:
        bouts.append((bstart, len(active)))

    bout_durs = [(e - s) for s, e in bouts]

    # Normalized bout features (per hour)
    af['bouts_per_hour'] = len(bouts) / (total_hours + 1e-12)
    if bouts:
        af['bout_mean_dur'] = np.mean(bout_durs)
        af['bout_dur_cv'] = np.std(bout_durs) / (np.mean(bout_durs) + 1e-12)
        af['bout_mean_enmo'] = np.mean([np.mean(enmo_sec[s:e]) for s, e in bouts])
    else:
        af['bout_mean_dur'] = 0
        af['bout_dur_cv'] = 0
        af['bout_mean_enmo'] = 0

    # Transition probabilities
    transitions_as, transitions_sa = 0, 0
    active_count, sed_count = 0, 0
    for j in range(len(active) - 1):
        if active[j]:
            active_count += 1
            if not active[j + 1]:
                transitions_as += 1
        else:
            sed_count += 1
            if active[j + 1]:
                transitions_sa += 1
    af['astp'] = transitions_as / (active_count + 1e-12)
    af['satp'] = transitions_sa / (sed_count + 1e-12)

    activity_rows.append(af)

    # ════════════════════════════════════════
    # CIRCADIAN FEATURES (fixed with ENMO)
    # ════════════════════════════════════════
    cf = {}

    n_hours = n_bins // 3600
    if n_hours >= 10:
        hourly_enmo = np.array([np.mean(enmo_sec[h*3600:(h+1)*3600]) for h in range(n_hours)])

        # M10: most active 10 consecutive hours (ENMO-based)
        best_m10 = 0
        for h in range(max(1, n_hours - 9)):
            m10_val = np.mean(hourly_enmo[h:h + 10])
            if m10_val > best_m10:
                best_m10 = m10_val
        cf['m10'] = best_m10

        # L5: least active 5 consecutive hours
        best_l5 = np.inf
        for h in range(max(1, n_hours - 4)):
            l5_val = np.mean(hourly_enmo[h:h + 5])
            if l5_val < best_l5:
                best_l5 = l5_val
        cf['l5'] = best_l5 if best_l5 != np.inf else 0

        # RA (now meaningful because ENMO removes gravity)
        cf['ra'] = (cf['m10'] - cf['l5']) / (cf['m10'] + cf['l5'] + 1e-12)

        # IV: intradaily variability (ENMO-based)
        diffs = np.diff(hourly_enmo)
        cf['iv'] = np.mean(diffs**2) / (np.var(hourly_enmo) + 1e-12)

        # MESOR and Amplitude (ENMO-based)
        cf['mesor'] = np.mean(hourly_enmo)
        cf['amplitude'] = (np.max(hourly_enmo) - np.min(hourly_enmo)) / 2

        # Activity slope (endurance/fatigue)
        if n_hours >= 3:
            slope, _, r_val, _, _ = linregress(np.arange(n_hours), hourly_enmo)
            cf['activity_slope'] = slope
            cf['activity_slope_r'] = r_val
        else:
            cf['activity_slope'] = 0
            cf['activity_slope_r'] = 0

        # Hourly CV
        cf['hourly_enmo_cv'] = np.std(hourly_enmo) / (np.mean(hourly_enmo) + 1e-12)

    else:
        cf['m10'] = np.percentile(enmo_sec, 90)
        cf['l5'] = np.percentile(enmo_sec, 10)
        cf['ra'] = (cf['m10'] - cf['l5']) / (cf['m10'] + cf['l5'] + 1e-12)
        cf['iv'] = 0
        cf['mesor'] = np.mean(enmo_sec)
        cf['amplitude'] = (np.percentile(enmo_sec, 95) - np.percentile(enmo_sec, 5)) / 2
        cf['activity_slope'] = 0
        cf['activity_slope_r'] = 0
        cf['hourly_enmo_cv'] = np.std(enmo_sec) / (np.mean(enmo_sec) + 1e-12)

    circadian_rows.append(cf)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(ids)}", flush=True)
print(f"  {len(ids)}/{len(ids)}")

# Build and save
act_df = pd.DataFrame(activity_rows).replace([np.inf, -np.inf], np.nan)
circ_df = pd.DataFrame(circadian_rows).replace([np.inf, -np.inf], np.nan)
for df in [act_df, circ_df]:
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

# Sanity checks
print("\n=== SANITY CHECKS ===")
print("Activity Profile:")
for c in act_df.columns:
    vals = act_df[c].values
    print(f"  {c:20s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, nunique={len(np.unique(np.round(vals,4)))}")

print("\nCircadian:")
for c in circ_df.columns:
    vals = circ_df[c].values
    print(f"  {c:20s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

# Check: no constant features
const = [c for c in act_df.columns if act_df[c].std() < 1e-10]
if const:
    print(f"\n  WARNING: Constant features: {const}")
else:
    print(f"\n  OK: No constant features")

# Save
act_out = pd.concat([ids.reset_index(drop=True), act_df.reset_index(drop=True)], axis=1)
act_out.to_csv(FEATS / 'home_activity_profile_v2.csv', index=False)
print(f"\nSaved feats/home_activity_profile_v2.csv ({act_df.shape[1]} features)")

circ_out = pd.concat([ids.reset_index(drop=True), circ_df.reset_index(drop=True)], axis=1)
circ_out.to_csv(FEATS / 'home_circadian_v2.csv', index=False)
print(f"Saved feats/home_circadian_v2.csv ({circ_df.shape[1]} features)")

# Correlations
print("\n=== CORRELATIONS WITH 6MWD ===")

# Load clinical scores
demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
p = ids.merge(demo, on=['cohort', 'subj_id'], how='left')
for c in ['MFIS Total', 'EDSS Total', 'BDI Raw Score']:
    p[c] = pd.to_numeric(p[c], errors='coerce')

targets = {
    '6MWD': y,
    'MFIS': p['MFIS Total'].values.astype(float),
    'EDSS': p['EDSS Total'].values.astype(float),
    'BDI': p['BDI Raw Score'].values.astype(float),
}

for df_name, df in [('Activity_v2', act_df), ('Circadian_v2', circ_df)]:
    print(f"\n{df_name}:")
    header = f"  {'Feature':20s}"
    for t in targets:
        header += f" {t:>8s}"
    print(header)
    print("  " + "-" * (20 + 9 * len(targets)))

    for c in df.columns:
        x = df[c].values.astype(float)
        row = f"  {c:20s}"
        for tname, yy in targets.items():
            mask = ~np.isnan(x) & ~np.isnan(yy)
            if mask.sum() > 10:
                rho, pval = spearmanr(x[mask], yy[mask])
                sig = '*' if pval < 0.05 else ' '
                row += f" {rho:+7.4f}{sig}"
            else:
                row += f"      ---"
        print(row)
