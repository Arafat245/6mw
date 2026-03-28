#!/usr/bin/env python3
"""
Compare 3 walking detection strategies and their impact on features + prediction.
A) Current ENMO+HR
B) find_walking (CWT-based, Straczkiewicz 2023)
C) Current + walking_verify filter

For each: extract Gait(11), CWT(28), WalkSway(12) from first 6 min of longest bout,
and PerBout aggregated features. Evaluate home R² for all.
"""
import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

from clinic.reproduce_c2 import (extract_gait10, extract_cwt, PreprocConfig, trim_edges,
                                  resample_uniform, align_to_ap_ml_vt, butter_bandpass,
                                  zero_phase_filter, get_fs_from_timestamps)
from clinic.extract_walking_sway import extract_walking_sway
from home.extract_clinicfree_features import (detect_walking_bouts, extract_bout_features,
                                               extract_activity_features)
from temporary_experiments.find_walking_py import detect_walking_bouts_findwalking
from notebooks.walking_verify import verify_walking_segment_df

CFG = PreprocConfig()
HOME_DIR = BASE / 'csv_home_daytime'
FS = 30


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X

def best_alpha(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 10)
    for a in alphas:
        pr = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler(); m = Ridge(alpha=a)
            m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
        r2 = r2_score(y, pr)
        if r2 > best[0]:
            best = (r2, mean_absolute_error(y, pr), spearmanr(y, pr)[0], a)
    return best


def preprocess_and_extract_all(raw_xyz, fs_orig):
    """Preprocess raw XYZ → Gait(11) + CWT(28) + WalkSway(12). Returns dicts or Nones."""
    if len(raw_xyz) < int(30 * fs_orig):
        return None, None, None
    try:
        arr_trim = trim_edges(raw_xyz, fs=fs_orig, trim_seconds=CFG.trim_seconds)
        arr_rs = resample_uniform(arr_trim, src_fs=fs_orig, dst_fs=CFG.target_fs)
        fs = CFG.target_fs
        apmlvt, _ = align_to_ap_ml_vt(arr_rs, fs=fs, cfg=CFG)
        lo, hi = CFG.step_band_hz
        b, a = butter_bandpass(lo, hi, fs, order=CFG.filter_order)
        apmlvt_bp = zero_phase_filter(apmlvt, b, a)
        vm_raw = np.linalg.norm(arr_rs, axis=1)
        enmo = np.maximum(vm_raw - 1.0, 0.0)

        pp = pd.DataFrame({
            'AP': apmlvt[:, 0], 'ML': apmlvt[:, 1], 'VT': apmlvt[:, 2],
            'AP_bp': apmlvt_bp[:, 0], 'ML_bp': apmlvt_bp[:, 1], 'VT_bp': apmlvt_bp[:, 2],
            'VM_dyn': np.linalg.norm(apmlvt, axis=1), 'VM_raw': vm_raw, 'ENMO': enmo,
            'cohort': 'X', 'subj_id': 0, 'year': 0, 'sixmwd': 0,
            'fs': fs, 'trim_s': 10, 'lp_hz': 0.25, 'bp_lo_hz': 0.25, 'bp_hi_hz': 2.5
        })

        gait = extract_gait10(pp)
        gait['vt_rms_g'] = float(np.sqrt(np.mean(pp['VT'].values ** 2)))
        cwt_f = extract_cwt(raw_xyz.astype(np.float32))
        ws = extract_walking_sway(pp['AP'].values, pp['ML'].values, pp['VT'].values)
        ml_rms = float(np.sqrt(np.mean(pp['ML'].values**2)))
        enmo_mean = gait.get('enmo_mean_g', float(np.mean(enmo)))
        vt_rms = gait['vt_rms_g']
        ws['ml_over_enmo'] = ml_rms / enmo_mean if enmo_mean > 0 else np.nan
        ws['ml_over_vt'] = ml_rms / vt_rms if vt_rms > 0 else np.nan

        return gait, cwt_f, ws
    except:
        return None, None, None


def verify_bout(xyz_bout, fs=30):
    """Use walking_verify to check if bout is walking."""
    if len(xyz_bout) < fs * 5:
        return False
    vm = np.sqrt(xyz_bout[:, 0]**2 + xyz_bout[:, 1]**2 + xyz_bout[:, 2]**2)
    ts = np.arange(len(xyz_bout)) / fs
    df = pd.DataFrame({'Timestamp': ts, 'X': xyz_bout[:, 0], 'Y': xyz_bout[:, 1], 'Z': xyz_bout[:, 2]})
    result = verify_walking_segment_df(df)
    metrics = dict(zip(result['metric'], result['value']))
    return bool(metrics.get('is_walking', False))


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
    X_demo4 = impute(p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float))

    print(f"Enhanced Detection Comparison (n={n})")
    print(f"{'='*90}\n")

    # Storage for each strategy
    strategies = {
        'A_current': {'gait': [], 'cwt': [], 'ws': [], 'n_bouts': [], 'walk_min': [], 'valid': []},
        'B_findwalk': {'gait': [], 'cwt': [], 'ws': [], 'n_bouts': [], 'walk_min': [], 'valid': []},
        'C_verified': {'gait': [], 'cwt': [], 'ws': [], 'n_bouts': [], 'walk_min': [], 'valid': []},
    }
    # Also collect per-bout features for PerBout aggregation
    perbout = {k: [] for k in strategies}

    six_min = 6 * 60 * FS

    for i, (_, r) in enumerate(ids_v.iterrows()):
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        fp = HOME_DIR / fn

        if not fp.exists():
            for s in strategies.values():
                s['gait'].append(None); s['cwt'].append(None); s['ws'].append(None)
                s['n_bouts'].append(0); s['walk_min'].append(0); s['valid'].append(False)
            for k in perbout: perbout[k].append([])
            continue

        xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)

        # ── Strategy A: Current ENMO+HR ──
        bouts_a = detect_walking_bouts(xyz, FS, min_bout_sec=10, merge_gap_sec=5)
        strategies['A_current']['n_bouts'].append(len(bouts_a))
        strategies['A_current']['walk_min'].append(sum((e-s)/FS for s, e in bouts_a) / 60)

        if bouts_a:
            longest = max(bouts_a, key=lambda b: b[1]-b[0])
            bout_xyz = xyz[longest[0]:longest[1]][:six_min]
            g, c, w = preprocess_and_extract_all(bout_xyz, FS)
            strategies['A_current']['gait'].append(g)
            strategies['A_current']['cwt'].append(c)
            strategies['A_current']['ws'].append(w)
            strategies['A_current']['valid'].append(g is not None)
        else:
            strategies['A_current']['gait'].append(None)
            strategies['A_current']['cwt'].append(None)
            strategies['A_current']['ws'].append(None)
            strategies['A_current']['valid'].append(False)

        # Per-bout features for A
        bout_feats_a = []
        for s, e in bouts_a:
            bf = extract_bout_features(xyz[s:e], FS)
            if bf is not None: bout_feats_a.append(bf)
        perbout['A_current'].append(bout_feats_a)

        # ── Strategy B: find_walking (CWT) ──
        bouts_b = detect_walking_bouts_findwalking(xyz, FS, min_amp=0.3, T=3,
                                                    min_bout_sec=10, merge_gap_sec=5)
        strategies['B_findwalk']['n_bouts'].append(len(bouts_b))
        strategies['B_findwalk']['walk_min'].append(sum((e-s)/FS for s, e in bouts_b) / 60)

        if bouts_b:
            longest = max(bouts_b, key=lambda b: b[1]-b[0])
            bout_xyz = xyz[longest[0]:longest[1]][:six_min]
            g, c, w = preprocess_and_extract_all(bout_xyz, FS)
            strategies['B_findwalk']['gait'].append(g)
            strategies['B_findwalk']['cwt'].append(c)
            strategies['B_findwalk']['ws'].append(w)
            strategies['B_findwalk']['valid'].append(g is not None)
        else:
            strategies['B_findwalk']['gait'].append(None)
            strategies['B_findwalk']['cwt'].append(None)
            strategies['B_findwalk']['ws'].append(None)
            strategies['B_findwalk']['valid'].append(False)

        bout_feats_b = []
        for s, e in bouts_b:
            bf = extract_bout_features(xyz[s:e], FS)
            if bf is not None: bout_feats_b.append(bf)
        perbout['B_findwalk'].append(bout_feats_b)

        # ── Strategy C: Current + walking_verify filter ──
        bouts_c = []
        for s, e in bouts_a:
            if verify_bout(xyz[s:e], FS):
                bouts_c.append((s, e))
        strategies['C_verified']['n_bouts'].append(len(bouts_c))
        strategies['C_verified']['walk_min'].append(sum((e-s)/FS for s, e in bouts_c) / 60)

        if bouts_c:
            longest = max(bouts_c, key=lambda b: b[1]-b[0])
            bout_xyz = xyz[longest[0]:longest[1]][:six_min]
            g, c, w = preprocess_and_extract_all(bout_xyz, FS)
            strategies['C_verified']['gait'].append(g)
            strategies['C_verified']['cwt'].append(c)
            strategies['C_verified']['ws'].append(w)
            strategies['C_verified']['valid'].append(g is not None)
        else:
            strategies['C_verified']['gait'].append(None)
            strategies['C_verified']['cwt'].append(None)
            strategies['C_verified']['ws'].append(None)
            strategies['C_verified']['valid'].append(False)

        bout_feats_c = []
        for s, e in bouts_c:
            bf = extract_bout_features(xyz[s:e], FS)
            if bf is not None: bout_feats_c.append(bf)
        perbout['C_verified'].append(bout_feats_c)

        if (i + 1) % 10 == 0 or i == 0:
            na = len(bouts_a); nb = len(bouts_b); nc = len(bouts_c)
            print(f"  [{i+1:3d}/{n}] {fn}: A={na} B={nb} C={nc} bouts", flush=True)

    # ── Bout statistics ──
    print(f"\n{'='*90}")
    print(f"BOUT STATISTICS")
    print(f"{'='*90}")
    for sname, sdata in strategies.items():
        nb = np.array(sdata['n_bouts'])
        wm = np.array(sdata['walk_min'])
        nv = sum(sdata['valid'])
        print(f"  {sname:15s}: valid={nv}/{n}, bouts/subj={nb.mean():.0f} (med={np.median(nb):.0f}), "
              f"walk={wm.mean():.0f}min (med={np.median(wm):.0f}min)")

    # ── Feature extraction + prediction ──
    print(f"\n{'='*90}")
    print(f"PREDICTION: Gait, CWT, WalkSway from first 6 min of longest bout")
    print(f"{'='*90}")

    GAIT_COLS = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
                 'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
                 'cadence_slope_per_min', 'vt_rms_g']

    for sname, sdata in strategies.items():
        vmask = np.array(sdata['valid'])
        nv = vmask.sum()
        if nv < 50:
            print(f"\n  {sname}: only {nv} valid — skipping"); continue

        X_g = impute(pd.DataFrame([g for g, v in zip(sdata['gait'], vmask) if v])[GAIT_COLS].values)
        X_c = impute(pd.DataFrame([c for c, v in zip(sdata['cwt'], vmask) if v]).replace([np.inf, -np.inf], np.nan).values)
        ws_df = pd.DataFrame([w for w, v in zip(sdata['ws'], vmask) if v])
        X_w = impute(ws_df.values)
        D = X_demo4[vmask]
        yv = y[vmask]

        print(f"\n  {sname} (n={nv}):")
        for label, X in [('Gait(11)', X_g), ('CWT(28)', X_c), ('WalkSway(12)', X_w),
                          ('Gait+Demo', np.column_stack([X_g, D])),
                          ('Gait+CWT+WS+Demo', np.column_stack([X_g, X_c, X_w, D]))]:
            r2, mae, rho, a = best_alpha(X, yv)
            print(f"    {label:25s} {X.shape[1]:>3d}f  R²={r2:.4f}  MAE={mae:.0f}ft  ρ={rho:.3f}")

    # ── PerBout aggregated features ──
    print(f"\n{'='*90}")
    print(f"PERBOUT AGGREGATED FEATURES")
    print(f"{'='*90}")

    for sname in strategies:
        feat_names = None
        rows = []
        pb_valid = []

        for subj_feats in perbout[sname]:
            if len(subj_feats) < 2:
                rows.append(None); pb_valid.append(False); continue
            if feat_names is None:
                feat_names = sorted(subj_feats[0].keys())
            arr = np.array([[bf.get(k, np.nan) for k in feat_names] for bf in subj_feats])
            agg = {}
            for j, name in enumerate(feat_names):
                col = arr[:, j]; valid = col[np.isfinite(col)]
                if len(valid) < 2: continue
                agg[f'{name}_med'] = np.median(valid)
                agg[f'{name}_max'] = np.max(valid)
                agg[f'{name}_cv'] = np.std(valid) / (np.mean(valid) + 1e-12)
            agg['n_bouts'] = len(subj_feats)
            agg['total_walk_sec'] = sum(bf.get('duration_sec', 0) for bf in subj_feats)
            rows.append(agg); pb_valid.append(True)

        pb_valid = np.array(pb_valid)
        npb = pb_valid.sum()
        if npb < 50:
            print(f"  {sname}: only {npb} valid — skipping"); continue

        pb_cols = sorted(rows[pb_valid.tolist().index(True)].keys())
        X_pb = impute(pd.DataFrame([r for r, v in zip(rows, pb_valid) if v])[pb_cols].values)
        D_pb = X_demo4[pb_valid]
        y_pb = y[pb_valid]

        # Correlation-selected top-20
        corrs = [(j, abs(spearmanr(X_pb[:, j], y_pb)[0])) for j in range(len(pb_cols))]
        corrs.sort(key=lambda x: x[1], reverse=True)
        top20_idx = [c[0] for c in corrs[:20]]
        X_top = X_pb[:, top20_idx]

        r2_all, mae_all, rho_all, _ = best_alpha(np.column_stack([X_pb, D_pb]), y_pb)
        r2_top, mae_top, rho_top, _ = best_alpha(np.column_stack([X_top, D_pb]), y_pb)

        print(f"  {sname} (n={npb}): All({len(pb_cols)}f)+Demo R²={r2_all:.4f} | "
              f"Top20+Demo R²={r2_top:.4f} MAE={mae_top:.0f} ρ={rho_top:.3f}")

    print(f"\nDone in {time.time()-t0:.0f}s")
