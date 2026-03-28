#!/usr/bin/env python3
"""
Extract clinic-style gait features from AGD-detected walking bouts (walking_bouts/).
Goal: approach R²=0.682 (clinic Gait(11) performance) using home walking bouts.

Multiple bout selection strategies tested:
1. Longest bout — most data, closest to 6MWT duration
2. Top-5 longest bouts — aggregated with median
3. Top-5 most intense (ENMO) bouts — most vigorous walking
4. All bouts ≥ 60s — only sustained walking
5. All bouts ≥ 30s — moderate filter
6. Best 6-min continuous segment from longest bout — closest to 6MWT
7. Weighted aggregation: weight by bout duration × ENMO (longer + more vigorous = higher weight)

Features: same Gait(11) as clinic (extract_gait10 + vt_rms_g)
Preprocessing: same as clinic (trim 10s, resample 30 Hz, Rodrigues, PCA, bandpass)
"""
import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from glob import glob

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
BOUTS_DIR = BASE / 'walking_bouts'
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

from clinic.reproduce_c2 import (extract_gait10, PreprocConfig, trim_edges,
                                  resample_uniform, align_to_ap_ml_vt,
                                  butter_bandpass, zero_phase_filter,
                                  get_fs_from_timestamps)
from notebooks.walking_verify import verify_walking_segment_df

CFG = PreprocConfig()


def eval_loo(X, y, alpha=10):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    r2 = r2_score(y, pr); mae = mean_absolute_error(y, pr); rho = spearmanr(y, pr)[0]
    return r2, mae, rho

def best_alpha(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 10)
    for a in alphas:
        r2, mae, rho = eval_loo(X, y, a)
        if r2 > best[0]: best = (r2, mae, rho, a)
    return best

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def load_bout(bout_csv):
    """Load a walking bout CSV (Timestamp, X, Y, Z). Returns xyz array and fs."""
    df = pd.read_csv(bout_csv)
    xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
    # Get fs from timestamps
    if 'Timestamp' in df.columns and len(df) > 1:
        diffs = np.diff(df['Timestamp'].values)
        diffs_pos = diffs[diffs > 0]
        fs = round(1.0 / np.median(diffs_pos)) if len(diffs_pos) > 0 else 30
    else:
        fs = 30
    return xyz, int(fs)


def preprocess_and_extract_gait(xyz, fs):
    """Same preprocessing as clinic + extract Gait(11). Returns dict or None."""
    if len(xyz) < int(30 * fs):  # need at least 30s
        return None
    try:
        arr_trim = trim_edges(xyz, fs=fs, trim_seconds=CFG.trim_seconds)
        arr_rs = resample_uniform(arr_trim, src_fs=fs, dst_fs=CFG.target_fs)
        fs30 = CFG.target_fs
        apmlvt, g_est = align_to_ap_ml_vt(arr_rs, fs=fs30, cfg=CFG)
        lo, hi = CFG.step_band_hz
        b, a = butter_bandpass(lo, hi, fs30, order=CFG.filter_order)
        apmlvt_bp = zero_phase_filter(apmlvt, b, a)
        vm_raw = np.linalg.norm(arr_rs, axis=1)
        enmo = np.maximum(vm_raw - 1.0, 0.0)

        pp = pd.DataFrame({
            'AP': apmlvt[:, 0], 'ML': apmlvt[:, 1], 'VT': apmlvt[:, 2],
            'AP_bp': apmlvt_bp[:, 0], 'ML_bp': apmlvt_bp[:, 1], 'VT_bp': apmlvt_bp[:, 2],
            'VM_dyn': np.linalg.norm(apmlvt, axis=1), 'VM_raw': vm_raw, 'ENMO': enmo,
            'cohort': 'X', 'subj_id': 0, 'year': 0, 'sixmwd': 0,
            'fs': fs30, 'trim_s': 10, 'lp_hz': 0.25, 'bp_lo_hz': 0.25, 'bp_hi_hz': 2.5
        })
        gait = extract_gait10(pp)
        gait['vt_rms_g'] = float(np.sqrt(np.mean(pp['VT'].values ** 2)))
        return gait
    except:
        return None


def is_valid_walking(bout_csv):
    """Use walking_verify to check if bout is truly walking. Returns (is_walking, dur_sec, enmo_mean, cadence_spm)."""
    df = pd.read_csv(bout_csv)
    xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo_mean = float(np.mean(np.maximum(vm - 1.0, 0.0)))
    dur_sec = len(df) / 30

    # Run walking verifier
    verify_df = verify_walking_segment_df(df)
    metrics = dict(zip(verify_df['metric'], verify_df['value']))
    is_walking = bool(metrics.get('is_walking', False))
    cadence = float(metrics.get('cadence_steps_per_min', 0))
    band_power = float(metrics.get('band_power_ratio', 0))

    return is_walking, dur_sec, enmo_mean, cadence, band_power


def get_bout_info(bout_csv):
    """Get bout duration and mean ENMO without full preprocessing."""
    df = pd.read_csv(bout_csv)
    xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo_mean = float(np.mean(np.maximum(vm - 1.0, 0.0)))
    dur_sec = len(df) / 30  # approximate
    return dur_sec, enmo_mean


def aggregate_gait_features(feat_list, weights=None):
    """Aggregate multiple gait feature dicts. Weighted median if weights provided."""
    if not feat_list:
        return None
    keys = sorted(feat_list[0].keys())
    arr = np.array([[f.get(k, np.nan) for k in keys] for f in feat_list])

    if weights is not None:
        # Weighted average
        w = np.array(weights)
        w = w / w.sum()
        agg = {}
        for j, k in enumerate(keys):
            col = arr[:, j]
            valid = np.isfinite(col)
            if valid.any():
                agg[k] = float(np.average(col[valid], weights=w[valid]))
            else:
                agg[k] = np.nan
    else:
        # Simple median
        agg = {}
        for j, k in enumerate(keys):
            col = arr[:, j]
            valid = col[np.isfinite(col)]
            agg[k] = float(np.median(valid)) if len(valid) > 0 else np.nan
    return agg


GAIT_COLS = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
             'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
             'cadence_slope_per_min', 'vt_rms_g']


if __name__ == '__main__':
    t0 = time.time()

    # Load target subjects
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
    for c in ['Age', 'Sex', 'Height']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float))

    # Clinic Gait baseline (from preprocessed)
    from clinic.reproduce_c2 import compute_vt_rms, add_sway_ratios
    gait_rows_c = []
    for _, r in ids_v.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        gait_rows_c.append(extract_gait10(pd.read_csv(PREPROC2 / fn)))
    vt_rms_df = compute_vt_rms(PREPROC2)
    gdf_c = pd.DataFrame(gait_rows_c)
    gm_c = pd.concat([ids_v.reset_index(drop=True), gdf_c], axis=1)
    sway_c = add_sway_ratios(gm_c.merge(vt_rms_df, on=['cohort', 'subj_id', 'sixmwd'], how='left'))
    X_clinic_gait = impute(sway_c[GAIT_COLS].values.astype(float))

    # ── Find walking bout folders for each subject ──
    def find_bout_folder(cohort, subj_id):
        """Find the walking_bouts subfolder for a subject."""
        prefix = f"{cohort}{subj_id:02d}_"
        for d in BOUTS_DIR.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                return d
        return None

    print(f"n={n} subjects")
    print(f"Loading walking bouts and extracting features...\n")

    # ── Strategy results storage ──
    strategies = {
        'longest': [],
        'first6_of_longest': [],
        'top5_longest_median': [],
        'top5_enmo_median': [],
        'all_60s_median': [],
        'all_30s_median': [],
        'dur_x_enmo_weighted': [],
        'verified_longest': [],
        'verified_first6': [],
        'verified_top5_median': [],
        'verified_top5_enmo': [],
        'verified_60s_median': [],
    }
    valid_mask = []

    for i, (_, r) in enumerate(ids_v.iterrows()):
        cohort, sid = r['cohort'], int(r['subj_id'])
        bout_folder = find_bout_folder(cohort, sid)

        if bout_folder is None or not list(bout_folder.glob('bout_*.csv')):
            for s in strategies: strategies[s].append(None)
            valid_mask.append(False)
            continue

        # Load all bout info (fast)
        bout_files = sorted(bout_folder.glob('bout_*.csv'))
        bout_infos = []
        for bf in bout_files:
            dur, enmo = get_bout_info(bf)
            bout_infos.append({'file': bf, 'dur': dur, 'enmo': enmo})

        bout_infos.sort(key=lambda x: x['dur'], reverse=True)

        # Strategy 1: Longest bout
        longest = bout_infos[0]
        xyz, fs = load_bout(longest['file'])
        strategies['longest'].append(preprocess_and_extract_gait(xyz, fs))

        # Strategy 2: First 6 min of longest bout
        six_min = int(6 * 60 * fs)
        first6_xyz = xyz[:min(six_min, len(xyz))]
        strategies['first6_of_longest'].append(preprocess_and_extract_gait(first6_xyz, fs))

        # Strategy 3: Top-5 longest bouts, median
        top5_long = bout_infos[:5]
        feats_list = []
        for bi in top5_long:
            bxyz, bfs = load_bout(bi['file'])
            g = preprocess_and_extract_gait(bxyz, bfs)
            if g: feats_list.append(g)
        strategies['top5_longest_median'].append(aggregate_gait_features(feats_list))

        # Strategy 4: Top-5 most intense (ENMO) bouts
        by_enmo = sorted(bout_infos, key=lambda x: x['enmo'], reverse=True)[:5]
        feats_list = []
        for bi in by_enmo:
            bxyz, bfs = load_bout(bi['file'])
            g = preprocess_and_extract_gait(bxyz, bfs)
            if g: feats_list.append(g)
        strategies['top5_enmo_median'].append(aggregate_gait_features(feats_list))

        # Strategy 5: All bouts ≥ 60s, median
        long_bouts = [bi for bi in bout_infos if bi['dur'] >= 60]
        feats_list = []
        for bi in long_bouts[:20]:  # cap at 20
            bxyz, bfs = load_bout(bi['file'])
            g = preprocess_and_extract_gait(bxyz, bfs)
            if g: feats_list.append(g)
        strategies['all_60s_median'].append(aggregate_gait_features(feats_list) if feats_list else None)

        # Strategy 6: All bouts ≥ 30s, median
        med_bouts = [bi for bi in bout_infos if bi['dur'] >= 30]
        feats_list = []
        for bi in med_bouts[:20]:
            bxyz, bfs = load_bout(bi['file'])
            g = preprocess_and_extract_gait(bxyz, bfs)
            if g: feats_list.append(g)
        strategies['all_30s_median'].append(aggregate_gait_features(feats_list) if feats_list else None)

        # Strategy 7: Duration × ENMO weighted aggregation (top 10)
        top10 = bout_infos[:10]
        feats_list, weights = [], []
        for bi in top10:
            bxyz, bfs = load_bout(bi['file'])
            g = preprocess_and_extract_gait(bxyz, bfs)
            if g:
                feats_list.append(g)
                weights.append(bi['dur'] * bi['enmo'])
        strategies['dur_x_enmo_weighted'].append(
            aggregate_gait_features(feats_list, weights) if feats_list else None)

        # ── Verified walking strategies (using walking_verify filter) ──
        # Verify all bouts (cache results)
        verified_bouts = []
        for bi in bout_infos:
            is_walk, dur, enmo, cadence, bp = is_valid_walking(bi['file'])
            if is_walk:
                verified_bouts.append(bi)

        # Strategy 8: Verified longest bout
        if verified_bouts:
            vb_sorted = sorted(verified_bouts, key=lambda x: x['dur'], reverse=True)
            xyz_v, fs_v = load_bout(vb_sorted[0]['file'])
            strategies['verified_longest'].append(preprocess_and_extract_gait(xyz_v, fs_v))
        else:
            strategies['verified_longest'].append(None)

        # Strategy 9: First 6 min of verified longest
        if verified_bouts:
            vb_sorted = sorted(verified_bouts, key=lambda x: x['dur'], reverse=True)
            xyz_v, fs_v = load_bout(vb_sorted[0]['file'])
            six_min_v = int(6 * 60 * fs_v)
            strategies['verified_first6'].append(
                preprocess_and_extract_gait(xyz_v[:min(six_min_v, len(xyz_v))], fs_v))
        else:
            strategies['verified_first6'].append(None)

        # Strategy 10: Top-5 verified longest, median
        if len(verified_bouts) >= 2:
            vb_sorted = sorted(verified_bouts, key=lambda x: x['dur'], reverse=True)[:5]
            feats_list = []
            for bi in vb_sorted:
                bxyz, bfs = load_bout(bi['file'])
                g = preprocess_and_extract_gait(bxyz, bfs)
                if g: feats_list.append(g)
            strategies['verified_top5_median'].append(
                aggregate_gait_features(feats_list) if feats_list else None)
        else:
            strategies['verified_top5_median'].append(None)

        # Strategy 11: Top-5 verified by ENMO, median
        if len(verified_bouts) >= 2:
            vb_enmo = sorted(verified_bouts, key=lambda x: x['enmo'], reverse=True)[:5]
            feats_list = []
            for bi in vb_enmo:
                bxyz, bfs = load_bout(bi['file'])
                g = preprocess_and_extract_gait(bxyz, bfs)
                if g: feats_list.append(g)
            strategies['verified_top5_enmo'].append(
                aggregate_gait_features(feats_list) if feats_list else None)
        else:
            strategies['verified_top5_enmo'].append(None)

        # Strategy 12: All verified ≥ 60s, median
        vb_60 = [bi for bi in verified_bouts if bi['dur'] >= 60]
        if len(vb_60) >= 2:
            feats_list = []
            for bi in sorted(vb_60, key=lambda x: x['dur'], reverse=True)[:20]:
                bxyz, bfs = load_bout(bi['file'])
                g = preprocess_and_extract_gait(bxyz, bfs)
                if g: feats_list.append(g)
            strategies['verified_60s_median'].append(
                aggregate_gait_features(feats_list) if feats_list else None)
        else:
            strategies['verified_60s_median'].append(None)

        valid_mask.append(True)

        if (i + 1) % 20 == 0 or i == 0:
            n_bouts = len(bout_infos)
            n_verified = len(verified_bouts)
            longest_dur = bout_infos[0]['dur']
            print(f"  [{i+1:3d}/{n}] {cohort}{sid:02d}: {n_bouts} bouts, {n_verified} verified, longest={longest_dur:.0f}s", flush=True)

    valid_mask = np.array(valid_mask)
    nv = valid_mask.sum()
    print(f"\nValid subjects: {nv}/{n}")

    # ── Evaluate all strategies ──
    print(f"\n{'='*90}")
    print(f"RESULTS: Gait(11) from home walking bouts (n varies by strategy)")
    print(f"{'='*90}")
    print(f"{'Strategy':<35s} {'n':>3s}  {'Gait R²':>8s} {'Gait MAE':>9s} {'Gait ρ':>7s}  {'G+Demo R²':>9s} {'G+D MAE':>8s} {'G+D ρ':>6s}")
    print("-" * 95)

    # Clinic baseline
    r2, mae, rho, a = best_alpha(X_clinic_gait, y)
    r2d, maed, rhod, ad = best_alpha(np.column_stack([X_clinic_gait, X_demo]), y)
    print(f"  {'CLINIC BASELINE':<33s} {n:>3d}  {r2:>8.4f} {mae:>8.0f}ft {rho:>7.3f}  {r2d:>9.4f} {maed:>7.0f}ft {rhod:>6.3f}")
    print("-" * 95)

    for strat_name, feat_list in strategies.items():
        # Build matrix for valid subjects
        strat_valid = np.array([f is not None for f in feat_list])
        both = valid_mask & strat_valid
        nv_s = both.sum()

        if nv_s < 50:
            print(f"  {strat_name:<33s} {nv_s:>3d}  — too few subjects")
            continue

        X_h = np.array([[f.get(c, np.nan) for c in GAIT_COLS] for f, v in zip(feat_list, both) if v])
        X_h = impute(X_h)
        y_s = y[both]
        D_s = X_demo[both]
        X_c_s = X_clinic_gait[both]

        # Gait only
        r2, mae, rho, a = best_alpha(X_h, y_s)
        # Gait + Demo
        r2d, maed, rhod, ad = best_alpha(np.column_stack([X_h, D_s]), y_s)

        print(f"  {strat_name:<33s} {nv_s:>3d}  {r2:>8.4f} {mae:>8.0f}ft {rho:>7.3f}  {r2d:>9.4f} {maed:>7.0f}ft {rhod:>6.3f}")

    print(f"\nDone in {time.time()-t0:.0f}s")
