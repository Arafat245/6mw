#!/usr/bin/env python3
"""
Investigation: Why home accelerometer data predicts 6MWD poorly vs clinic.
Clinic R²=0.806 vs Home R²=0.488. All outputs to temporary_experiments/investigations/.
"""
import sys, warnings, time, os, re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.signal import butter, filtfilt, welch
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent  # 6mw/
OUT = Path(__file__).parent                 # investigations/
sys.path.insert(0, str(BASE))

# ══════════════════════════════════════════════════════════════════
# MODULE 0: DATA LOADING
# ══════════════════════════════════════════════════════════════════
def load_all_data():
    print("Module 0: Loading data...", flush=True)

    # Targets (103 rows, includes M22 and M44)
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'] == 22)) | \
           ((ids['cohort'] == 'M') & (ids['subj_id'] == 44))
    ids101 = ids[~excl].reset_index(drop=True)

    # Predictions (101 rows)
    preds = pd.read_csv(BASE / 'feats' / 'best_predictions.csv')
    df = ids101.merge(preds, on=['cohort', 'subj_id', 'year'], how='left', suffixes=('', '_p'))
    df['abs_err_home'] = np.abs(df['sixmwd_pred_home'] - df['sixmwd_actual'])
    df['abs_err_clinic'] = np.abs(df['sixmwd_pred_clinic'] - df['sixmwd_actual'])
    df['err_gap'] = df['abs_err_home'] - df['abs_err_clinic']  # positive = home worse
    df['err_home'] = df['sixmwd_pred_home'] - df['sixmwd_actual']
    df['err_clinic'] = df['sixmwd_pred_clinic'] - df['sixmwd_actual']

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        demo[c] = pd.to_numeric(demo[c], errors='coerce')
    df = df.merge(demo[['cohort', 'subj_id', 'Age', 'Sex', 'Height', 'BMI']],
                  on=['cohort', 'subj_id'], how='left')
    df['cohort_label'] = df['cohort'].map({'C': 'Healthy', 'M': 'POMS'})
    df['fn'] = df.apply(lambda r: f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv", axis=1)

    # Clinic gait features
    clinic_gait = pd.read_csv(BASE / 'feats' / 'clinic_gait10.csv')
    # Home gait features
    home_gait = pd.read_csv(BASE / 'feats' / 'home_gait13_hybrid.csv')

    # Bout indices
    bout_data = np.load(BASE / 'feats' / 'home_walking_bout_indices.npz', allow_pickle=True)
    bout_dict = bout_data['bouts'].item()

    print(f"  Loaded {len(df)} subjects, {len(clinic_gait)} clinic gait rows, "
          f"{len(home_gait)} home gait rows, {len(bout_dict)} bout entries", flush=True)
    return df, clinic_gait, home_gait, bout_dict


# ══════════════════════════════════════════════════════════════════
# MODULE 1: SUBJECT-LEVEL PREDICTION ERRORS
# ══════════════════════════════════════════════════════════════════
def module1_prediction_errors(df):
    print("\nModule 1: Subject-level prediction errors...", flush=True)
    findings = []

    # Basic stats
    mae_h = df['abs_err_home'].mean()
    mae_c = df['abs_err_clinic'].mean()
    findings.append(f"Mean abs error: Home={mae_h:.1f}ft, Clinic={mae_c:.1f}ft")

    # Quadrants
    both_good = ((df['abs_err_home'] < mae_h) & (df['abs_err_clinic'] < mae_c)).sum()
    clinic_good_home_bad = ((df['abs_err_home'] >= mae_h) & (df['abs_err_clinic'] < mae_c)).sum()
    clinic_bad_home_good = ((df['abs_err_home'] < mae_h) & (df['abs_err_clinic'] >= mae_c)).sum()
    both_bad = ((df['abs_err_home'] >= mae_h) & (df['abs_err_clinic'] >= mae_c)).sum()
    findings.append(f"Quadrants: both_good={both_good}, clinic_good_home_bad={clinic_good_home_bad}, "
                    f"clinic_bad_home_good={clinic_bad_home_good}, both_bad={both_bad}")

    # Worst home-minus-clinic subjects
    worst = df.nlargest(10, 'err_gap')[['cohort', 'subj_id', 'sixmwd', 'abs_err_home',
                                         'abs_err_clinic', 'err_gap', 'Age', 'cohort_label']]
    findings.append(f"Top 10 worst home-vs-clinic gap:\n{worst.to_string(index=False)}")

    # Correlations
    for col in ['Age', 'BMI']:
        mask = df[col].notna()
        if mask.sum() > 10:
            rho, p = spearmanr(df.loc[mask, col], df.loc[mask, 'abs_err_home'])
            findings.append(f"  {col} vs abs_err_home: rho={rho:.3f}, p={p:.3f}")

    # POMS vs Healthy error
    poms = df[df['cohort'] == 'M']
    healthy = df[df['cohort'] == 'C']
    findings.append(f"Mean abs_err_home: POMS={poms['abs_err_home'].mean():.1f}, Healthy={healthy['abs_err_home'].mean():.1f}")
    findings.append(f"Mean abs_err_clinic: POMS={poms['abs_err_clinic'].mean():.1f}, Healthy={healthy['abs_err_clinic'].mean():.1f}")

    # Figure: error scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for cohort, color, label in [('C', '#2196F3', 'Healthy'), ('M', '#F44336', 'POMS')]:
        sub = df[df['cohort'] == cohort]
        axes[0].scatter(sub['abs_err_clinic'], sub['abs_err_home'], c=color, alpha=0.7, label=label, s=40)
    axes[0].plot([0, 600], [0, 600], 'k--', alpha=0.3)
    axes[0].axhline(mae_h, color='gray', ls=':', alpha=0.5)
    axes[0].axvline(mae_c, color='gray', ls=':', alpha=0.5)
    axes[0].set_xlabel('Clinic |Error| (ft)'); axes[0].set_ylabel('Home |Error| (ft)')
    axes[0].set_title('Subject-Level Prediction Errors'); axes[0].legend()
    axes[0].set_xlim(0, 600); axes[0].set_ylim(0, 600)

    # Waterfall
    sorted_df = df.sort_values('err_gap')
    colors = ['#2196F3' if c == 'C' else '#F44336' for c in sorted_df['cohort']]
    axes[1].bar(range(len(sorted_df)), sorted_df['err_gap'], color=colors, alpha=0.7)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_xlabel('Subjects (sorted)'); axes[1].set_ylabel('Home Error - Clinic Error (ft)')
    axes[1].set_title('Per-Subject Error Gap (positive = home worse)')

    plt.tight_layout()
    fig.savefig(OUT / 'fig_error_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    return findings


# ══════════════════════════════════════════════════════════════════
# MODULE 2: RECORDING QUALITY
# ══════════════════════════════════════════════════════════════════
def module2_recording_quality(df, bout_dict):
    print("\nModule 2: Recording quality...", flush=True)
    findings = []

    rec_hours, walk_mins, n_bouts_list = [], [], []
    for _, r in df.iterrows():
        fn = r['fn']
        fp = BASE / 'csv_home_daytime' / fn

        # Recording hours from file size (each row ~25 bytes: "0.123,-0.456,0.789,30\n")
        if fp.exists():
            fsize = fp.stat().st_size
            approx_rows = fsize / 25  # rough estimate
            hours = approx_rows / (30 * 3600)
        else:
            hours = np.nan
        rec_hours.append(hours)

        # Walking from bout indices
        if fn in bout_dict:
            bouts = bout_dict[fn]
            total_walk_samples = sum(e - s for s, e in bouts)
            walk_mins.append(total_walk_samples / (30 * 60))
            n_bouts_list.append(len(bouts))
        else:
            walk_mins.append(np.nan)
            n_bouts_list.append(np.nan)

    df['rec_hours'] = rec_hours
    df['walk_min'] = walk_mins
    df['n_bouts'] = n_bouts_list
    df['walk_pct'] = df['walk_min'] / (df['rec_hours'] * 60) * 100

    findings.append(f"Recording hours: mean={df['rec_hours'].mean():.1f}, "
                    f"min={df['rec_hours'].min():.1f}, max={df['rec_hours'].max():.1f}")
    findings.append(f"Walking minutes: mean={df['walk_min'].mean():.1f}, "
                    f"min={df['walk_min'].min():.1f}, max={df['walk_min'].max():.1f}")
    findings.append(f"N bouts: mean={df['n_bouts'].mean():.0f}, "
                    f"min={df['n_bouts'].min():.0f}, max={df['n_bouts'].max():.0f}")

    # Correlations
    for col in ['rec_hours', 'walk_min', 'n_bouts', 'walk_pct']:
        mask = df[col].notna() & df['abs_err_home'].notna()
        if mask.sum() > 10:
            rho, p = spearmanr(df.loc[mask, col], df.loc[mask, 'abs_err_home'])
            findings.append(f"  {col} vs abs_err_home: rho={rho:.3f}, p={p:.4f}")

    # Short recordings
    short = df[df['rec_hours'] < 20]
    if len(short) > 0:
        findings.append(f"Subjects with <20h recording: {len(short)}, "
                        f"mean abs_err_home={short['abs_err_home'].mean():.1f} vs "
                        f"all={df['abs_err_home'].mean():.1f}")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, col, title in [(axes[0, 0], 'rec_hours', 'Recording Hours'),
                           (axes[0, 1], 'walk_min', 'Total Walking (min)'),
                           (axes[1, 0], 'n_bouts', 'Number of Bouts'),
                           (axes[1, 1], 'walk_pct', 'Walking %')]:
        mask = df[col].notna()
        for cohort, color, label in [('C', '#2196F3', 'Healthy'), ('M', '#F44336', 'POMS')]:
            sub = df[(df['cohort'] == cohort) & mask]
            ax.scatter(sub[col], sub['abs_err_home'], c=color, alpha=0.6, label=label, s=30)
        rho, p = spearmanr(df.loc[mask, col], df.loc[mask, 'abs_err_home'])
        ax.set_xlabel(title); ax.set_ylabel('Home |Error| (ft)')
        ax.set_title(f'{title} vs Error (rho={rho:.3f}, p={p:.3f})')
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / 'fig_recording_quality.png', dpi=150, bbox_inches='tight')
    plt.close()

    return findings


# ══════════════════════════════════════════════════════════════════
# MODULE 3: BOUT SELECTION QUALITY
# ══════════════════════════════════════════════════════════════════
def module3_bout_selection(df):
    print("\nModule 3: Bout selection quality...", flush=True)
    findings = []
    cache_path = OUT / 'bout_similarity_cache.npz'

    if cache_path.exists():
        print("  Loading from cache...", flush=True)
        cached = np.load(cache_path, allow_pickle=True)
        df['top_sim'] = cached['top_sim']
        df['mean_sim'] = cached['mean_sim']
        df['n_refined'] = cached['n_refined']
        df['selected_sec'] = cached['selected_sec']
    else:
        from home.home_hybrid_models_v2 import (load_cached_daytime, load_clinic_raw,
            detect_active_bouts, refine_with_hr, compute_walking_signature, select_walking_segment)

        top_sims, mean_sims, n_refined_list, selected_secs = [], [], [], []
        for i, (_, r) in enumerate(df.iterrows()):
            cohort, sid, year, sixmwd = r['cohort'], int(r['subj_id']), int(r['year']), int(r['sixmwd'])
            xyz_home, fs_h = load_cached_daytime(cohort, sid, year, sixmwd)
            xyz_clinic, fs_c = load_clinic_raw(cohort, sid, year, sixmwd)

            if xyz_home is None or xyz_clinic is None:
                top_sims.append(np.nan); mean_sims.append(np.nan)
                n_refined_list.append(0); selected_secs.append(0)
                continue

            bouts = detect_active_bouts(xyz_home, fs_h)
            refined = refine_with_hr(xyz_home, fs_h, bouts)
            n_refined_list.append(len(refined))

            if not refined:
                top_sims.append(np.nan); mean_sims.append(np.nan); selected_secs.append(0)
                continue

            clinic_sig = compute_walking_signature(xyz_clinic, fs_c)
            sims = []
            for s, e in refined:
                bout_sig = compute_walking_signature(xyz_home[s:e], fs_h)
                sim = float(np.dot(clinic_sig, bout_sig) /
                            (np.linalg.norm(clinic_sig) * np.linalg.norm(bout_sig) + 1e-12))
                sims.append(sim)

            top_sims.append(max(sims) if sims else np.nan)
            mean_sims.append(np.mean(sims) if sims else np.nan)

            _, top_s = select_walking_segment(xyz_home, fs_h, refined, 360, xyz_clinic, fs_c)
            selected_secs.append(360 if _ is not None else 0)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(df)}]", flush=True)

        df['top_sim'] = top_sims
        df['mean_sim'] = mean_sims
        df['n_refined'] = n_refined_list
        df['selected_sec'] = selected_secs

        np.savez(cache_path, top_sim=top_sims, mean_sim=mean_sims,
                 n_refined=n_refined_list, selected_sec=selected_secs)
        print("  Cached to bout_similarity_cache.npz", flush=True)

    # Analysis
    mask = df['top_sim'].notna()
    rho, p = spearmanr(df.loc[mask, 'top_sim'], df.loc[mask, 'abs_err_home'])
    findings.append(f"top_sim vs abs_err_home: rho={rho:.3f}, p={p:.4f}")
    rho2, p2 = spearmanr(df.loc[mask, 'mean_sim'], df.loc[mask, 'abs_err_home'])
    findings.append(f"mean_sim vs abs_err_home: rho={rho2:.3f}, p={p2:.4f}")
    findings.append(f"Top similarity: mean={df['top_sim'].mean():.3f}, std={df['top_sim'].std():.3f}")
    findings.append(f"N refined bouts: mean={df['n_refined'].mean():.0f}, min={df['n_refined'].min():.0f}")

    # Low similarity subjects
    low_sim = df[df['top_sim'] < df['top_sim'].quantile(0.25)]
    findings.append(f"Bottom 25% similarity: mean abs_err_home={low_sim['abs_err_home'].mean():.1f} "
                    f"vs top 75%={df[df['top_sim'] >= df['top_sim'].quantile(0.25)]['abs_err_home'].mean():.1f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for cohort, color, label in [('C', '#2196F3', 'Healthy'), ('M', '#F44336', 'POMS')]:
        sub = df[(df['cohort'] == cohort) & mask]
        axes[0].scatter(sub['top_sim'], sub['abs_err_home'], c=color, alpha=0.6, label=label, s=30)
        axes[1].scatter(sub['mean_sim'], sub['abs_err_home'], c=color, alpha=0.6, label=label, s=30)
    axes[0].set_xlabel('Top Bout Similarity'); axes[0].set_ylabel('Home |Error| (ft)')
    axes[0].set_title(f'Top Similarity vs Error (rho={rho:.3f})')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Mean Bout Similarity'); axes[1].set_ylabel('Home |Error| (ft)')
    axes[1].set_title(f'Mean Similarity vs Error (rho={rho2:.3f})')
    axes[1].legend(fontsize=8)

    axes[2].hist(df.loc[df['cohort'] == 'C', 'top_sim'].dropna(), bins=20, alpha=0.6,
                 color='#2196F3', label='Healthy')
    axes[2].hist(df.loc[df['cohort'] == 'M', 'top_sim'].dropna(), bins=20, alpha=0.6,
                 color='#F44336', label='POMS')
    axes[2].set_xlabel('Top Bout Similarity'); axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Top Similarity'); axes[2].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / 'fig_bout_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()

    return findings


# ══════════════════════════════════════════════════════════════════
# MODULE 4: FEATURE DRIFT
# ══════════════════════════════════════════════════════════════════
def module4_feature_drift(df, clinic_gait, home_gait):
    print("\nModule 4: Feature drift...", flush=True)
    findings = []

    shared_feats = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
                    'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
                    'cadence_slope_per_min']

    merged = clinic_gait.merge(home_gait, on=['cohort', 'subj_id', 'sixmwd'],
                               suffixes=('_clinic', '_home'))
    # Exclude M22 and M44
    merged = merged[~((merged['cohort'] == 'M') & (merged['subj_id'].isin([22, 44])))]

    # Per-feature drift
    drift_cols = []
    for feat in shared_feats:
        cc = f'{feat}_clinic'
        hc = f'{feat}_home'
        if cc in merged.columns and hc in merged.columns:
            merged[f'drift_{feat}'] = np.abs(merged[hc] - merged[cc]) / (np.abs(merged[cc]) + 1e-8)
            drift_cols.append(f'drift_{feat}')
            rho_ch, _ = spearmanr(merged[cc].fillna(0), merged[hc].fillna(0))
            findings.append(f"  {feat}: clinic-home rho={rho_ch:.3f}, "
                            f"mean drift={merged[f'drift_{feat}'].mean():.3f}")

    # Composite drift score
    for dc in drift_cols:
        merged[dc + '_z'] = (merged[dc] - merged[dc].mean()) / (merged[dc].std() + 1e-8)
    merged['composite_drift'] = merged[[dc + '_z' for dc in drift_cols]].mean(axis=1)

    # Merge drift into main df
    df_drift = merged[['cohort', 'subj_id', 'sixmwd', 'composite_drift'] + drift_cols]
    df2 = df.merge(df_drift, on=['cohort', 'subj_id', 'sixmwd'], how='left')

    mask = df2['composite_drift'].notna() & df2['abs_err_home'].notna()
    rho, p = spearmanr(df2.loc[mask, 'composite_drift'], df2.loc[mask, 'abs_err_home'])
    findings.append(f"Composite drift vs abs_err_home: rho={rho:.3f}, p={p:.4f}")

    # Flag bad cadence
    cad_col = 'cadence_hz_home' if 'cadence_hz_home' in merged.columns else None
    if cad_col:
        low_cad = merged[merged[cad_col] < 1.0]
        findings.append(f"Subjects with home cadence < 1.0 Hz: {len(low_cad)}/{len(merged)}")
        if len(low_cad) > 0:
            low_ids = low_cad[['cohort', 'subj_id', cad_col, 'cadence_hz_clinic']].head(10)
            findings.append(f"  Examples:\n{low_ids.to_string(index=False)}")

    # Copy composite_drift back to df
    for col in ['composite_drift'] + drift_cols:
        if col in df2.columns:
            df[col] = df2[col].values

    # Figure: scatter grid
    n_feats = len(shared_feats)
    ncols = 5; nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()
    for i, feat in enumerate(shared_feats):
        cc = f'{feat}_clinic'; hc = f'{feat}_home'
        if cc not in merged.columns or hc not in merged.columns:
            continue
        ax = axes[i]
        for cohort, color, label in [('C', '#2196F3', 'Healthy'), ('M', '#F44336', 'POMS')]:
            sub = merged[merged['cohort'] == cohort]
            ax.scatter(sub[cc], sub[hc], c=color, alpha=0.5, s=20, label=label)
        # Identity line
        lo = min(merged[cc].min(), merged[hc].min())
        hi = max(merged[cc].max(), merged[hc].max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3)
        rho_f, _ = spearmanr(merged[cc].fillna(0), merged[hc].fillna(0))
        ax.set_title(f'{feat}\nrho={rho_f:.3f}', fontsize=9)
        ax.set_xlabel('Clinic', fontsize=8); ax.set_ylabel('Home', fontsize=8)
        if i == 0: ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Feature Drift: Clinic vs Home (per subject)', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / 'fig_feature_drift.png', dpi=150, bbox_inches='tight')
    plt.close()

    return findings


# ══════════════════════════════════════════════════════════════════
# MODULE 5: TEMPORAL GAP
# ══════════════════════════════════════════════════════════════════
def module5_temporal_gap(df):
    print("\nModule 5: Temporal gap...", flush=True)
    findings = []

    # Parse clinic dates from csv_raw2 timestamps
    clinic_dates = {}
    for fn in (BASE / 'csv_raw2').glob('*.csv'):
        try:
            first = pd.read_csv(fn, nrows=1)
            if 'Timestamp' in first.columns:
                ts = first['Timestamp'].iloc[0]
                clinic_dates[fn.name] = datetime.fromtimestamp(ts)
        except:
            pass

    # Parse home dates from GT3X filenames
    home_dates = {}
    accel_dir = BASE / 'Accel files'
    if accel_dir.exists():
        for subdir in accel_dir.iterdir():
            if not subdir.is_dir():
                continue
            # Extract subject ID from dir name (e.g., C01_OPT)
            m = re.match(r'([CM])(\d+)', subdir.name)
            if not m:
                continue
            cohort, sid = m.group(1), int(m.group(2))
            for f in subdir.glob('*.gt3x'):
                dm = re.search(r'\((\d{4}-\d{2}-\d{2})\)', f.name)
                if dm:
                    home_dates[(cohort, sid)] = datetime.strptime(dm.group(1), '%Y-%m-%d')
                    break

    # Compute gaps
    gaps = []
    for _, r in df.iterrows():
        fn = r['fn']
        cohort, sid = r['cohort'], int(r['subj_id'])
        c_date = clinic_dates.get(fn)
        h_date = home_dates.get((cohort, sid))
        if c_date and h_date:
            gaps.append(abs((h_date - c_date).days))
        else:
            gaps.append(np.nan)
    df['date_gap_days'] = gaps

    valid = df['date_gap_days'].notna()
    n_valid = valid.sum()
    findings.append(f"Date gaps computed for {n_valid}/{len(df)} subjects")

    if n_valid > 10:
        findings.append(f"Gap stats: mean={df['date_gap_days'].mean():.1f}d, "
                        f"median={df['date_gap_days'].median():.1f}d, "
                        f"max={df['date_gap_days'].max():.0f}d")
        within3 = (df['date_gap_days'] <= 3).sum()
        over30 = (df['date_gap_days'] > 30).sum()
        findings.append(f"  Within 3 days: {within3}/{n_valid} ({100*within3/n_valid:.0f}%)")
        findings.append(f"  Over 30 days: {over30}/{n_valid}")

        rho, p = spearmanr(df.loc[valid, 'date_gap_days'], df.loc[valid, 'abs_err_home'])
        findings.append(f"  date_gap vs abs_err_home: rho={rho:.3f}, p={p:.4f}")

        # Gap categories
        for label, lo, hi in [('0-3d', 0, 3), ('4-30d', 4, 30), ('>30d', 31, 9999)]:
            sub = df[(df['date_gap_days'] >= lo) & (df['date_gap_days'] <= hi)]
            if len(sub) > 0:
                findings.append(f"  Gap {label}: n={len(sub)}, "
                                f"mean_err_home={sub['abs_err_home'].mean():.1f}, "
                                f"mean_err_clinic={sub['abs_err_clinic'].mean():.1f}")

        # Drift correlation
        if 'composite_drift' in df.columns:
            mask2 = valid & df['composite_drift'].notna()
            if mask2.sum() > 10:
                rho2, p2 = spearmanr(df.loc[mask2, 'date_gap_days'], df.loc[mask2, 'composite_drift'])
                findings.append(f"  date_gap vs composite_drift: rho={rho2:.3f}, p={p2:.4f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if n_valid > 10:
        for cohort, color, label in [('C', '#2196F3', 'Healthy'), ('M', '#F44336', 'POMS')]:
            sub = df[(df['cohort'] == cohort) & valid]
            axes[0].scatter(sub['date_gap_days'], sub['abs_err_home'], c=color, alpha=0.6, label=label, s=30)
        axes[0].set_xlabel('Date Gap (days)'); axes[0].set_ylabel('Home |Error| (ft)')
        axes[0].set_title('Temporal Gap vs Error'); axes[0].legend(fontsize=8)

        # By category
        cats = [('0-3', 0, 3), ('4-30', 4, 30), ('>30', 31, 9999)]
        means_h, means_c, labels = [], [], []
        for label, lo, hi in cats:
            sub = df[(df['date_gap_days'] >= lo) & (df['date_gap_days'] <= hi)]
            if len(sub) > 0:
                means_h.append(sub['abs_err_home'].mean())
                means_c.append(sub['abs_err_clinic'].mean())
                labels.append(f'{label}\n(n={len(sub)})')
        x = np.arange(len(labels))
        axes[1].bar(x - 0.15, means_h, 0.3, label='Home', color='#FF9800')
        axes[1].bar(x + 0.15, means_c, 0.3, label='Clinic', color='#4CAF50')
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel('Mean |Error| (ft)'); axes[1].set_title('Error by Gap Category')
        axes[1].legend(fontsize=8)

        if 'composite_drift' in df.columns:
            mask2 = valid & df['composite_drift'].notna()
            for cohort, color, label in [('C', '#2196F3', 'Healthy'), ('M', '#F44336', 'POMS')]:
                sub = df[(df['cohort'] == cohort) & mask2]
                axes[2].scatter(sub['date_gap_days'], sub['composite_drift'], c=color, alpha=0.6, label=label, s=30)
            axes[2].set_xlabel('Date Gap (days)'); axes[2].set_ylabel('Composite Drift Score')
            axes[2].set_title('Temporal Gap vs Feature Drift'); axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / 'fig_temporal_gap.png', dpi=150, bbox_inches='tight')
    plt.close()

    return findings


# ══════════════════════════════════════════════════════════════════
# MODULE 6: SIGNAL CHARACTERISTICS (sample of subjects)
# ══════════════════════════════════════════════════════════════════
def module6_signal_chars(df):
    print("\nModule 6: Signal characteristics (sampled subjects)...", flush=True)
    findings = []

    # Pick representative subjects
    sorted_df = df.sort_values('abs_err_home')
    n = len(sorted_df)
    best5 = sorted_df.head(5)
    worst5 = sorted_df.tail(5)
    mid5 = sorted_df.iloc[n//2 - 2:n//2 + 3]
    sample = pd.concat([best5, mid5, worst5])

    clinic_stats, home_stats = [], []
    for _, r in sample.iterrows():
        fn = r['fn']
        # Clinic preprocessed
        cp = BASE / 'csv_preprocessed2' / fn
        if cp.exists():
            cdf = pd.read_csv(cp)
            clinic_stats.append({
                'fn': fn, 'cohort': r['cohort'], 'subj_id': r['subj_id'],
                'abs_err_home': r['abs_err_home'],
                'enmo_mean': cdf['ENMO'].mean(), 'enmo_std': cdf['ENMO'].std(),
                'vt_bp_std': cdf['VT_bp'].std() if 'VT_bp' in cdf else np.nan,
                'ml_rms': np.sqrt((cdf['ML']**2).mean()) if 'ML' in cdf else np.nan,
                'setting': 'clinic'
            })

        # Home: use raw data, compute ENMO directly
        hp = BASE / 'csv_home_daytime' / fn
        if hp.exists():
            # Just load first 360s * 30 = 10800 samples for quick comparison
            hdf = pd.read_csv(hp, nrows=10800, usecols=['X', 'Y', 'Z'])
            xyz = hdf.values.astype(float)
            vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
            enmo = np.maximum(vm - 1.0, 0.0)
            home_stats.append({
                'fn': fn, 'cohort': r['cohort'], 'subj_id': r['subj_id'],
                'abs_err_home': r['abs_err_home'],
                'enmo_mean': enmo.mean(), 'enmo_std': enmo.std(),
                'vt_bp_std': np.nan,  # would need preprocessing
                'ml_rms': np.nan,
                'setting': 'home_raw_first6min'
            })

    all_stats = pd.DataFrame(clinic_stats + home_stats)
    findings.append(f"Sampled {len(sample)} subjects (5 best, 5 median, 5 worst home predictions)")

    clinic_s = pd.DataFrame(clinic_stats)
    home_s = pd.DataFrame(home_stats)
    if len(clinic_s) > 0 and len(home_s) > 0:
        findings.append(f"Clinic ENMO: mean={clinic_s['enmo_mean'].mean():.4f}, "
                        f"std={clinic_s['enmo_std'].mean():.4f}")
        findings.append(f"Home raw ENMO (first 6min): mean={home_s['enmo_mean'].mean():.4f}, "
                        f"std={home_s['enmo_std'].mean():.4f}")

    # Figure: ENMO comparison for best/worst
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    examples = [('Best', best5.iloc[0]), ('Median', mid5.iloc[0]), ('Worst', worst5.iloc[0])]
    for col, (label, subj) in enumerate(examples):
        fn = subj['fn']
        # Clinic
        cp = BASE / 'csv_preprocessed2' / fn
        if cp.exists():
            cdf = pd.read_csv(cp)
            t_c = np.arange(len(cdf)) / 30
            axes[0, col].plot(t_c, cdf['ENMO'].values, color='#4CAF50', alpha=0.5, lw=0.3)
            axes[0, col].set_title(f'{label}: {fn}\nClinic ENMO (err={subj["abs_err_clinic"]:.0f}ft)', fontsize=9)
            axes[0, col].set_ylim(0, 0.5)

        # Home (first 6 min)
        hp = BASE / 'csv_home_daytime' / fn
        if hp.exists():
            hdf = pd.read_csv(hp, nrows=10800, usecols=['X', 'Y', 'Z'])
            xyz = hdf.values.astype(float)
            vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
            enmo = np.maximum(vm - 1.0, 0.0)
            t_h = np.arange(len(enmo)) / 30
            axes[1, col].plot(t_h, enmo, color='#FF9800', alpha=0.5, lw=0.3)
            axes[1, col].set_title(f'Home raw first 6min (err={subj["abs_err_home"]:.0f}ft)', fontsize=9)
            axes[1, col].set_ylim(0, 0.5)

    for ax in axes.flatten():
        ax.set_xlabel('Time (s)', fontsize=8); ax.set_ylabel('ENMO (g)', fontsize=8)
    plt.suptitle('Signal Comparison: Clinic 6MWT vs Home Raw (first 6min)', y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / 'fig_signal_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    return findings


# ══════════════════════════════════════════════════════════════════
# MODULE 7: COHORT BREAKDOWN
# ══════════════════════════════════════════════════════════════════
def module7_cohort_breakdown(df):
    print("\nModule 7: Cohort breakdown...", flush=True)
    findings = []

    for cohort, label in [('C', 'Healthy'), ('M', 'POMS')]:
        sub = df[df['cohort'] == cohort]
        n = len(sub)
        r2_h = r2_score(sub['sixmwd_actual'], sub['sixmwd_pred_home'])
        r2_c = r2_score(sub['sixmwd_actual'], sub['sixmwd_pred_clinic'])
        mae_h = sub['abs_err_home'].mean()
        mae_c = sub['abs_err_clinic'].mean()
        findings.append(f"\n{label} (n={n}):")
        findings.append(f"  Home:   R²={r2_h:.4f}, MAE={mae_h:.1f}ft")
        findings.append(f"  Clinic: R²={r2_c:.4f}, MAE={mae_c:.1f}ft")
        findings.append(f"  Gap:    R²_diff={r2_c - r2_h:.4f}, MAE_diff={mae_h - mae_c:.1f}ft")

        for col in ['composite_drift', 'top_sim', 'date_gap_days', 'rec_hours', 'walk_min']:
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    findings.append(f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}")

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = [('abs_err_home', 'Home |Error| (ft)'), ('abs_err_clinic', 'Clinic |Error| (ft)'),
               ('err_gap', 'Error Gap (home-clinic)')]
    if 'composite_drift' in df.columns:
        metrics += [('composite_drift', 'Feature Drift Score')]
    if 'top_sim' in df.columns:
        metrics += [('top_sim', 'Top Bout Similarity')]
    if 'date_gap_days' in df.columns:
        metrics += [('date_gap_days', 'Date Gap (days)')]

    for i, (col, title) in enumerate(metrics[:6]):
        ax = axes.flatten()[i]
        data = [df.loc[df['cohort'] == 'C', col].dropna(), df.loc[df['cohort'] == 'M', col].dropna()]
        bp = ax.boxplot(data, labels=['Healthy', 'POMS'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2196F3'); bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('#F44336'); bp['boxes'][1].set_alpha(0.5)
        ax.set_title(title, fontsize=10)
        # Mann-Whitney test
        if len(data[0]) > 5 and len(data[1]) > 5:
            u, p = mannwhitneyu(data[0], data[1])
            ax.text(0.95, 0.95, f'p={p:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=8)

    for j in range(len(metrics), 6):
        axes.flatten()[j].set_visible(False)

    plt.suptitle('Cohort Breakdown: Healthy vs POMS', fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT / 'fig_cohort_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()

    return findings


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print("=" * 70)
    print("INVESTIGATION: Why home data predicts 6MWD poorly")
    print("=" * 70)

    df, clinic_gait, home_gait, bout_dict = load_all_data()

    all_findings = []

    f1 = module1_prediction_errors(df)
    all_findings += ["=== MODULE 1: PREDICTION ERRORS ==="] + f1

    f2 = module2_recording_quality(df, bout_dict)
    all_findings += ["", "=== MODULE 2: RECORDING QUALITY ==="] + f2

    f3 = module3_bout_selection(df)
    all_findings += ["", "=== MODULE 3: BOUT SELECTION QUALITY ==="] + f3

    f4 = module4_feature_drift(df, clinic_gait, home_gait)
    all_findings += ["", "=== MODULE 4: FEATURE DRIFT ==="] + f4

    f5 = module5_temporal_gap(df)
    all_findings += ["", "=== MODULE 5: TEMPORAL GAP ==="] + f5

    f6 = module6_signal_chars(df)
    all_findings += ["", "=== MODULE 6: SIGNAL CHARACTERISTICS ==="] + f6

    f7 = module7_cohort_breakdown(df)
    all_findings += ["", "=== MODULE 7: COHORT BREAKDOWN ==="] + f7

    # Save master CSV
    export_cols = ['cohort', 'subj_id', 'year', 'sixmwd', 'cohort_label',
                   'sixmwd_actual', 'sixmwd_pred_home', 'sixmwd_pred_clinic',
                   'abs_err_home', 'abs_err_clinic', 'err_gap',
                   'Age', 'Sex', 'BMI',
                   'rec_hours', 'walk_min', 'n_bouts', 'walk_pct']
    for c in ['top_sim', 'mean_sim', 'n_refined', 'selected_sec',
              'composite_drift', 'date_gap_days']:
        if c in df.columns:
            export_cols.append(c)
    df[export_cols].to_csv(OUT / 'investigation_summary.csv', index=False)
    print(f"\nSaved investigation_summary.csv ({len(df)} rows)")

    # Save findings
    with open(OUT / 'investigation_findings.txt', 'w') as f:
        for line in all_findings:
            f.write(line + '\n')
            print(line)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Done in {elapsed:.0f}s. All outputs in {OUT}")
    print(f"{'='*70}")
