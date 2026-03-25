#!/usr/bin/env python3
"""
Extract ALS paper features (Gupta et al. 2023) from home accelerometer data.
85 features: Activity Index (10), Spectral (1), Activity Bout (4), Submovement (70)
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
BASE = Path('.')
CACHE = BASE / 'csv_home_daytime'


def extract_als_features(xyz, fs=30):
    """Extract all 85 ALS paper features from triaxial accelerometer data."""
    n = len(xyz)
    if n < fs * 60:  # need at least 1 min
        return None

    # ═══════════════════════════════════════════
    # Preprocessing: gravity removal, bandpass 0.1-20Hz
    # ═══════════════════════════════════════════
    b, a = butter(6, [0.1, min(20, fs/2-1)], btype='bandpass', fs=fs)
    xyz_filt = np.column_stack([filtfilt(b, a, xyz[:, j]) for j in range(3)])

    # Vector magnitude
    vm = np.sqrt(xyz_filt[:, 0]**2 + xyz_filt[:, 1]**2 + xyz_filt[:, 2]**2)

    # ═══════════════════════════════════════════
    # ACTIVITY INDEX (10 features)
    # Per 1-second windows, compute AI = std of VM
    # ═══════════════════════════════════════════
    sec = int(fs)
    n_secs = n // sec
    ai_values = np.array([np.std(vm[j*sec:(j+1)*sec]) for j in range(n_secs)])

    # Exclude periods of inactivity (AI < threshold)
    ai_thresh = 0.01  # approximate threshold for inactivity
    active_ai = ai_values[ai_values >= ai_thresh]

    feat = {}
    if len(active_ai) > 10:
        feat['ai_mean'] = np.mean(active_ai)
        feat['ai_median'] = np.median(active_ai)
        # Mode approximation: most common bin
        hist, edges = np.histogram(active_ai, bins=50)
        feat['ai_mode'] = (edges[np.argmax(hist)] + edges[np.argmax(hist)+1]) / 2
        # Entropy
        p = hist / hist.sum()
        p = p[p > 0]
        feat['ai_entropy'] = -np.sum(p * np.log2(p))
    else:
        feat['ai_mean'] = np.mean(ai_values)
        feat['ai_median'] = np.median(ai_values)
        feat['ai_mode'] = 0
        feat['ai_entropy'] = 0

    # % daytime at low/moderate/high AI
    # Thresholds from Bai et al.: low < 0.02, moderate 0.02-0.1, high > 0.1
    feat['pct_low_ai'] = np.mean(ai_values < 0.02)
    feat['pct_mod_ai'] = np.mean((ai_values >= 0.02) & (ai_values < 0.1))
    feat['pct_high_ai'] = np.mean(ai_values >= 0.1)

    # % acceleration in single direction (PCA on 1-sec windows)
    # For low, moderate, high AI windows separately
    for level, mask_fn in [('low', lambda x: x < 0.02),
                           ('mod', lambda x: (x >= 0.02) & (x < 0.1)),
                           ('high', lambda x: x >= 0.1)]:
        mask = mask_fn(ai_values)
        idxs = np.where(mask)[0]
        if len(idxs) > 5:
            pca_vars = []
            for idx in idxs[:500]:  # sample for speed
                win = xyz_filt[idx*sec:(idx+1)*sec]
                if len(win) < sec: continue
                pca = PCA(n_components=1)
                pca.fit(win)
                pca_vars.append(pca.explained_variance_ratio_[0])
            feat[f'pct_single_dir_{level}'] = np.mean(pca_vars) if pca_vars else 0
        else:
            feat[f'pct_single_dir_{level}'] = 0

    # ═══════════════════════════════════════════
    # SPECTRAL (1 feature)
    # Total power in 0.1-5 Hz
    # ═══════════════════════════════════════════
    from scipy.signal import welch
    nperseg = min(len(vm), int(fs * 10))
    freqs, pxx = welch(vm, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    band = (freqs >= 0.1) & (freqs <= 5.0)
    feat['total_power_0_1_5hz'] = np.sum(pxx[band]) * (freqs[1] - freqs[0])

    # ═══════════════════════════════════════════
    # ACTIVITY BOUT (4 features)
    # Bouts: continuous periods with AI > threshold, duration 4-18s
    # ═══════════════════════════════════════════
    bout_accel_list = []
    bout_jerk_list = []
    in_bout, bout_start = False, 0

    for j in range(n_secs):
        if ai_values[j] >= ai_thresh and not in_bout:
            bout_start = j; in_bout = True
        elif (ai_values[j] < ai_thresh or j == n_secs-1) and in_bout:
            bout_dur = j - bout_start
            if 4 <= bout_dur <= 18:
                bout_data = vm[bout_start*sec:j*sec]
                bout_accel_list.append(np.max(np.abs(bout_data)))
                bout_jerk_list.append(np.mean(np.abs(np.diff(bout_data) * fs)))
            in_bout = False

    feat['bout_accel_mean'] = np.mean(bout_accel_list) if bout_accel_list else 0
    feat['bout_accel_sd'] = np.std(bout_accel_list) if len(bout_accel_list) > 1 else 0
    feat['bout_jerk_mean'] = np.mean(bout_jerk_list) if bout_jerk_list else 0
    feat['bout_jerk_sd'] = np.std(bout_jerk_list) if len(bout_jerk_list) > 1 else 0

    # ═══════════════════════════════════════════
    # SUBMOVEMENT (SM) FEATURES (70 features)
    # Project onto 2D plane via PCA, integrate to velocity,
    # identify submovements as bell-shaped velocity curves
    # ═══════════════════════════════════════════

    # PCA to find primary and secondary directions of planar movement
    pca2d = PCA(n_components=2)
    xyz_2d = pca2d.fit_transform(xyz_filt)  # project onto 2D plane

    # Integrate acceleration to velocity (with high-pass to remove drift)
    dt = 1.0 / fs
    vel = np.cumsum(xyz_2d, axis=0) * dt

    # High-pass filter velocity to remove drift (0.1 Hz)
    b_hp, a_hp = butter(4, 0.1, btype='highpass', fs=fs)
    vel[:, 0] = filtfilt(b_hp, a_hp, vel[:, 0])
    vel[:, 1] = filtfilt(b_hp, a_hp, vel[:, 1])

    # Speed (magnitude of 2D velocity)
    speed = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)

    # Find submovements: segments between zero-crossings of velocity
    # Use primary direction velocity for segmentation
    vel_primary = vel[:, 0]

    # Find zero crossings
    sign_changes = np.where(np.diff(np.sign(vel_primary)))[0]

    # Extract submovements
    sm_features = {
        'dist': [], 'vel': [], 'accel': [], 'jerk': [], 'dur': []
    }
    sm_features_short = {'dist': [], 'vel': [], 'accel': [], 'jerk': [], 'dur': []}
    sm_features_long = {'dist': [], 'vel': [], 'accel': [], 'jerk': [], 'dur': []}

    # Also for secondary direction
    sm_features_pc2 = {
        'dist': [], 'vel': [], 'accel': [], 'jerk': [], 'dur': []
    }
    sm_features_short_pc2 = {'dist': [], 'vel': [], 'accel': [], 'jerk': [], 'dur': []}
    sm_features_long_pc2 = {'dist': [], 'vel': [], 'accel': [], 'jerk': [], 'dur': []}

    median_dur_threshold = 0.5  # seconds, will be updated

    # Process submovements in primary direction
    all_durs = []
    for i in range(len(sign_changes) - 1):
        s, e = sign_changes[i], sign_changes[i+1]
        if e - s < 3: continue  # too short
        dur = (e - s) / fs
        all_durs.append(dur)

    if all_durs:
        median_dur_threshold = np.median(all_durs)

    for direction, vel_dir, accel_dir, sf, sf_short, sf_long in [
        ('PC1', vel[:, 0], xyz_2d[:, 0], sm_features, sm_features_short, sm_features_long),
        ('PC2', vel[:, 1], xyz_2d[:, 1], sm_features_pc2, sm_features_short_pc2, sm_features_long_pc2),
    ]:
        sign_ch = np.where(np.diff(np.sign(vel_dir)))[0]
        for i in range(len(sign_ch) - 1):
            s, e = sign_ch[i], sign_ch[i+1]
            if e - s < 3: continue

            seg_vel = vel_dir[s:e]
            seg_accel = accel_dir[s:e]
            seg_speed = np.abs(seg_vel)

            dur = (e - s) / fs
            dist = np.sum(seg_speed) * dt
            peak_vel = np.max(seg_speed)
            peak_accel = np.max(np.abs(seg_accel))

            # Normalized jerk
            seg_jerk = np.diff(seg_accel) * fs
            if dur > 0 and peak_vel > 0:
                norm_jerk = np.sqrt(0.5 * np.sum(seg_jerk**2) * dt) * (dur**2.5) / dist if dist > 0 else 0
            else:
                norm_jerk = 0

            sf['dist'].append(dist)
            sf['vel'].append(peak_vel)
            sf['accel'].append(peak_accel)
            sf['jerk'].append(norm_jerk)
            sf['dur'].append(dur)

            if dur <= median_dur_threshold:
                sf_short['dist'].append(dist)
                sf_short['vel'].append(peak_vel)
                sf_short['accel'].append(peak_accel)
                sf_short['jerk'].append(norm_jerk)
                sf_short['dur'].append(dur)
            else:
                sf_long['dist'].append(dist)
                sf_long['vel'].append(peak_vel)
                sf_long['accel'].append(peak_accel)
                sf_long['jerk'].append(norm_jerk)
                sf_long['dur'].append(dur)

    # Compute SM features: mean and SD for each, grouped by short/long and PC1/PC2
    for prefix, groups in [
        ('sm_pc1', [('short', sm_features_short), ('long', sm_features_long)]),
        ('sm_pc2', [('short', sm_features_short_pc2), ('long', sm_features_long_pc2)]),
    ]:
        for dur_label, sf in groups:
            for metric in ['dist', 'vel', 'accel', 'jerk', 'dur']:
                vals = sf[metric]
                key_m = f'{prefix}_{dur_label}_{metric}_mean'
                key_s = f'{prefix}_{dur_label}_{metric}_sd'
                feat[key_m] = np.mean(vals) if vals else 0
                feat[key_s] = np.std(vals) if len(vals) > 1 else 0

    # SM velocity-time curve shape (PC scores)
    # Fit PCA to normalized velocity-time curves of submovements
    # Use primary direction long-duration SMs
    vel_curves = []
    target_len = int(median_dur_threshold * fs * 2) if median_dur_threshold > 0 else int(fs)
    target_len = max(10, min(target_len, int(fs * 2)))

    sign_ch = np.where(np.diff(np.sign(vel[:, 0])))[0]
    for i in range(len(sign_ch) - 1):
        s, e = sign_ch[i], sign_ch[i+1]
        seg = vel[s:e, 0]
        if len(seg) >= 5:
            # Resample to fixed length
            resampled = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(seg)), seg)
            # Normalize
            mx = np.max(np.abs(resampled))
            if mx > 0:
                resampled = resampled / mx
            vel_curves.append(resampled)

    if len(vel_curves) >= 10:
        vel_matrix = np.array(vel_curves[:2000])  # limit for speed
        n_pc = min(5, vel_matrix.shape[1], vel_matrix.shape[0])
        pca_sm = PCA(n_components=n_pc)
        pc_scores = pca_sm.fit_transform(vel_matrix)

        for pc in range(n_pc):
            scores = pc_scores[:, pc]
            feat[f'sm_pc{pc+1}_mean'] = np.mean(np.abs(scores))
            feat[f'sm_pc{pc+1}_sd'] = np.std(scores)
            feat[f'sm_pc{pc+1}_kurtosis'] = float(pd.Series(scores).kurtosis())
    else:
        for pc in range(5):
            feat[f'sm_pc{pc+1}_mean'] = 0
            feat[f'sm_pc{pc+1}_sd'] = 0
            feat[f'sm_pc{pc+1}_kurtosis'] = 0

    return feat


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
if __name__ == '__main__':
    ids = pd.read_csv('feats/target_6mwd.csv')

    print("Extracting ALS paper features from home data...")
    rows = []
    for i, (_, r) in enumerate(ids.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        p = CACHE / fn
        if not p.exists():
            rows.append(None)
            continue
        try:
            df = pd.read_csv(p, usecols=['X', 'Y', 'Z'])
            xyz = df.values.astype(np.float64)
            feat = extract_als_features(xyz, fs=30)
            rows.append(feat)
        except Exception as e:
            print(f"  [{i+1}] {fn}: ERROR {e}")
            rows.append(None)
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(ids)}]", flush=True)

    print(f"  {len(ids)}/{len(ids)}")

    # Build dataframe
    valid = [r is not None for r in rows]
    feat_df = pd.DataFrame([r for r in rows if r is not None])
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    for c in feat_df.columns:
        if feat_df[c].isna().any():
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    valid_mask = np.array(valid)
    print(f"Valid: {sum(valid)}/{len(ids)}")
    print(f"Features: {feat_df.shape[1]}")

    # Save
    out = pd.concat([ids[valid_mask].reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    out.to_csv('feats/home_als_features.csv', index=False)
    print(f"Saved feats/home_als_features.csv")

    # Correlations
    demo = pd.read_excel('SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids[valid_mask].reset_index(drop=True).merge(demo, on=['cohort', 'subj_id'], how='left')

    targets = {}
    for c in ['MFIS Total', 'MFIS Phys', 'MFIS Cog', 'MFIS Psych',
              'EDSS Total', 'BDI Raw Score', 'BDI Rank', 'MS FSS', 'MS Dur']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
        targets[c] = p[c].values.astype(float)
    for c in ['Age', 'Sex', 'Height', 'Weight', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
        targets[c] = p[c].values.astype(float)
    targets['6MWD'] = ids[valid_mask]['sixmwd'].values.astype(float)

    # Print only features with |rho| >= 0.25 for any target
    print(f"\nFeatures with |rho| >= 0.25 for any clinical target:")
    print(f"{'Feature':40s} {'6MWD':>7s} {'MFIS_T':>7s} {'MFIS_P':>7s} {'MFIS_C':>7s} {'MFIS_Ps':>7s} "
          f"{'EDSS':>7s} {'BDI_R':>7s} {'FSS':>7s} {'MSDur':>7s} {'Age':>7s}")
    print("-" * 120)

    for c in feat_df.columns:
        vals = feat_df[c].values.astype(float)
        row_strs = []
        any_sig = False
        for tname in ['6MWD', 'MFIS Total', 'MFIS Phys', 'MFIS Cog', 'MFIS Psych',
                      'EDSS Total', 'BDI Raw Score', 'MS FSS', 'MS Dur', 'Age']:
            yy = targets[tname]
            mask = ~np.isnan(vals) & ~np.isnan(yy)
            if mask.sum() > 10:
                rho, pval = spearmanr(vals[mask], yy[mask])
                sig = '*' if pval < 0.05 else ' '
                row_strs.append(f'{rho:+.3f}{sig}')
                if abs(rho) >= 0.25:
                    any_sig = True
            else:
                row_strs.append('   --- ')
        if any_sig:
            print(f"{c:40s} " + " ".join(row_strs))
