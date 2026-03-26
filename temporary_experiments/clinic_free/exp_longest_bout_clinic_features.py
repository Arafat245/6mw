#!/usr/bin/env python3
"""
Extract clinic-style features from the longest walking bout per subject.
Same preprocessing and feature extraction as the clinic pipeline.
Then check correlations with 6MWD and run prediction.
"""
import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

from clinic.reproduce_c2 import extract_gait10, extract_cwt
from clinic.extract_walking_sway import extract_walking_sway
from temporary_experiments.clinic_free.exp_clinic_free_v2 import detect_walking_bouts_clinicfree

FS = 30
HOME_DIR = BASE / 'csv_home_daytime'


def eval_loo(X, y, alpha=10):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, pr), mean_absolute_error(y, pr), pearsonr(y, pr)[0], spearmanr(y, pr)[0]

def best_alpha(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 0, 10)
    for a in alphas:
        r2, mae, rv, rho = eval_loo(X, y, a)
        if r2 > best[0]: best = (r2, mae, rv, rho, a)
    return best

def report(name, nf, r2, mae, rv, rho, alpha):
    print(f"  {name:55s} {nf:>3d}f  α={alpha:>3d}  R²={r2:.4f}  MAE={mae:.0f}ft  r={rv:.3f}  ρ={rho:.3f}")


def preprocess_like_clinic(raw_xyz, fs):
    """Same preprocessing as clinic/reproduce_c2.py: preprocess_file()
    Gravity removal, Rodrigues rotation, PCA yaw, bandpass, ENMO."""
    import math
    from scipy.signal import butter, filtfilt

    arr = raw_xyz.copy()
    # Trim first/last 10s
    n_trim = int(10 * fs)
    if 2 * n_trim < len(arr):
        arr = arr[n_trim:len(arr) - n_trim]

    # Resample to uniform 30 Hz (already 30 Hz from home data)
    # Gravity removal
    b, a = butter(4, 0.25, btype='lowpass', fs=fs)
    g_est = np.column_stack([filtfilt(b, a, arr[:, j]) for j in range(3)])
    arr_dyn = arr - g_est

    # Rodrigues rotation
    g_mean = g_est.mean(axis=0)
    zhat = np.array([0., 0., 1.])
    gvec = g_mean / (np.linalg.norm(g_mean) + 1e-12)
    angle = math.acos(np.clip(float(zhat @ gvec), -1, 1))
    if angle > 1e-4:
        axis = np.cross(gvec, zhat)
        if np.linalg.norm(axis) < 1e-8:
            axis = np.array([1., 0., 0.])
        ax = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        arr_v = arr_dyn @ R.T
    else:
        arr_v = arr_dyn.copy()

    # PCA yaw alignment
    XY = arr_v[:, :2]
    C = np.cov(XY, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    ap_dir = vecs[:, np.argmax(vals)]
    theta = math.atan2(float(ap_dir[1]), float(ap_dir[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    apmlvt = arr_v @ Rz.T

    # Bandpass
    b, a = butter(4, [0.25, 2.5], btype='bandpass', fs=fs)
    apmlvt_bp = np.column_stack([filtfilt(b, a, apmlvt[:, j]) for j in range(3)])

    vm_raw = np.linalg.norm(arr, axis=1)
    vm_dyn = np.linalg.norm(apmlvt, axis=1)
    enmo = np.maximum(vm_raw - 1.0, 0.0)

    df = pd.DataFrame({
        'AP': apmlvt[:, 0], 'ML': apmlvt[:, 1], 'VT': apmlvt[:, 2],
        'AP_bp': apmlvt_bp[:, 0], 'ML_bp': apmlvt_bp[:, 1], 'VT_bp': apmlvt_bp[:, 2],
        'VM_dyn': vm_dyn, 'VM_raw': vm_raw, 'ENMO': enmo,
        'cohort': 'X', 'subj_id': 0, 'year': 0, 'sixmwd': 0,
        'fs': fs, 'trim_s': 10, 'lp_hz': 0.25, 'bp_lo_hz': 0.25, 'bp_hi_hz': 2.5
    })
    return df, arr  # return raw trimmed too for CWT


if __name__ == '__main__':
    t0 = time.time()

    # Load subjects
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = [((PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists())
                    for _, r in ids101.iterrows()]
    clinic_valid = np.array(clinic_valid)
    ids101 = ids101[clinic_valid].reset_index(drop=True)
    y = ids101['sixmwd'].values.astype(float)
    n = len(y)

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        demo[c] = pd.to_numeric(demo[c], errors='coerce')
    p = ids101.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    X_demo_4 = p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float)
    X_demo_5 = p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float)
    for X in [X_demo_4, X_demo_5]:
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any(): X[m, j] = np.nanmedian(X[:, j])

    # ── Extract clinic-style features from longest home bout ──
    print(f"Extracting clinic-style features from longest bout (n={n})...", flush=True)

    cache = OUT / 'longest_bout_clinic_features_cache.npz'
    if cache.exists():
        print("  Loading from cache...", flush=True)
        d = np.load(cache, allow_pickle=True)
        gait_rows = list(d['gait_rows'])
        cwt_rows = list(d['cwt_rows'])
        ws_rows = list(d['ws_rows'])
        bout_durs = list(d['bout_durs'])
        valid = d['valid']
    else:
        gait_rows, cwt_rows, ws_rows, bout_durs, valid = [], [], [], [], []

        for i, (_, r) in enumerate(ids101.iterrows()):
            fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
            fp = HOME_DIR / fn

            if not fp.exists():
                gait_rows.append(None); cwt_rows.append(None); ws_rows.append(None)
                bout_durs.append(0); valid.append(False)
                continue

            xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
            bouts = detect_walking_bouts_clinicfree(xyz, FS, min_bout_sec=10, merge_gap_sec=5)

            if not bouts:
                gait_rows.append(None); cwt_rows.append(None); ws_rows.append(None)
                bout_durs.append(0); valid.append(False)
                continue

            # Find longest bout
            longest = max(bouts, key=lambda b: b[1] - b[0])
            s, e = longest
            bout_dur = (e - s) / FS
            bout_durs.append(bout_dur)
            bout_xyz = xyz[s:e]

            try:
                # Preprocess like clinic
                preprocessed_df, trimmed_raw = preprocess_like_clinic(bout_xyz, FS)

                # Gait features (same as clinic extract_gait10)
                gait = extract_gait10(preprocessed_df)
                gait_rows.append(gait)

                # CWT features (uses raw XYZ, same as clinic)
                # extract_cwt expects raw XYZ array
                raw_for_cwt = bout_xyz.astype(np.float32)
                cwt_feats = extract_cwt(raw_for_cwt)
                cwt_rows.append(cwt_feats)

                # WalkSway features
                ws = extract_walking_sway(
                    preprocessed_df['AP'].values,
                    preprocessed_df['ML'].values,
                    preprocessed_df['VT'].values
                )
                ws_rows.append(ws)
                valid.append(True)

            except Exception as ex:
                gait_rows.append(None); cwt_rows.append(None); ws_rows.append(None)
                valid.append(False)
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{n}] FAILED: {ex}", flush=True)
                continue

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n}] {fn}: longest={bout_dur:.0f}s", flush=True)

        valid = np.array(valid)
        np.savez(cache, gait_rows=gait_rows, cwt_rows=cwt_rows, ws_rows=ws_rows,
                 bout_durs=bout_durs, valid=valid, allow_pickle=True)
        print(f"  Cached. Valid: {valid.sum()}/{n}", flush=True)

    vmask = valid.astype(bool)
    nv = vmask.sum()
    print(f"\nValid subjects: {nv}/{n}")
    durs = np.array(bout_durs)
    print(f"Longest bout durations: mean={durs[vmask].mean():.0f}s, median={np.median(durs[vmask]):.0f}s, "
          f"min={durs[vmask].min():.0f}s, max={durs[vmask].max():.0f}s")

    # ── Build feature matrices ──
    # Gait
    gait_df = pd.DataFrame([g for g, v in zip(gait_rows, vmask) if v])
    gait_cols = list(gait_df.columns)
    X_gait = gait_df.values.astype(float)

    # CWT
    cwt_df = pd.DataFrame([c for c, v in zip(cwt_rows, vmask) if v]).replace([np.inf, -np.inf], np.nan)
    cwt_cols = list(cwt_df.columns)
    X_cwt = cwt_df.values.astype(float)

    # WalkSway
    ws_df = pd.DataFrame([w for w, v in zip(ws_rows, vmask) if v])
    ws_cols = list(ws_df.columns)
    X_ws = ws_df.values.astype(float)

    # Impute NaN
    for X in [X_gait, X_cwt, X_ws]:
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any(): X[m, j] = np.nanmedian(X[:, j])

    yv = y[vmask]
    D4v = X_demo_4[vmask]
    D5v = X_demo_5[vmask]

    # ── Correlation screen ──
    print(f"\n{'='*90}")
    print(f"FEATURE CORRELATIONS WITH 6MWD (n={nv})")
    print(f"{'='*90}")

    all_corrs = []
    for label, cols, X in [('Gait', gait_cols, X_gait),
                            ('CWT', cwt_cols, X_cwt),
                            ('WalkSway', ws_cols, X_ws)]:
        print(f"\n{label} ({X.shape[1]} features):")
        for j, col in enumerate(cols):
            rho_val, p_val = spearmanr(X[:, j], yv)
            if np.isfinite(rho_val):
                all_corrs.append({'category': label, 'feature': col, 'rho': rho_val, 'abs_rho': abs(rho_val), 'p': p_val})
                if abs(rho_val) >= 0.2:
                    print(f"  {col:40s}  ρ={rho_val:+.3f}  p={p_val:.4f}")

    corr_df = pd.DataFrame(all_corrs).sort_values('abs_rho', ascending=False)
    corr_df.to_csv(OUT / 'longest_bout_feature_correlations.csv', index=False)
    print(f"\nFeatures with |ρ| > 0.3: {(corr_df['abs_rho'] > 0.3).sum()}")
    print(f"Features with |ρ| > 0.4: {(corr_df['abs_rho'] > 0.4).sum()}")
    print(f"Features with |ρ| > 0.5: {(corr_df['abs_rho'] > 0.5).sum()}")

    # ── Prediction experiments ──
    print(f"\n{'='*90}")
    print(f"PREDICTION (LOO CV, n={nv})")
    print(f"{'='*90}")

    # Same combos as clinic
    # Gait only
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait, D4v]), yv)
    report("Gait(10)+Demo(4) [clinic-style]", X_gait.shape[1]+4, r2, mae, rv, rho, a)

    # CWT only
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_cwt, D4v]), yv)
    report("CWT(28)+Demo(4)", X_cwt.shape[1]+4, r2, mae, rv, rho, a)

    # WalkSway only
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_ws, D4v]), yv)
    report("WalkSway(10)+Demo(4)", X_ws.shape[1]+4, r2, mae, rv, rho, a)

    # Gait + CWT
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait, X_cwt, D4v]), yv)
    report("Gait+CWT+Demo(4)", X_gait.shape[1]+X_cwt.shape[1]+4, r2, mae, rv, rho, a)

    # Gait + WalkSway
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait, X_ws, D4v]), yv)
    report("Gait+WalkSway+Demo(4)", X_gait.shape[1]+X_ws.shape[1]+4, r2, mae, rv, rho, a)

    # ALL: Gait + CWT + WalkSway + Demo (same as clinic best)
    X_all = np.column_stack([X_gait, X_cwt, X_ws, D4v])
    r2, mae, rv, rho, a = best_alpha(X_all, yv)
    report("Gait+CWT+WalkSway+Demo(4) [full clinic-style]", X_all.shape[1], r2, mae, rv, rho, a)

    # With Demo(5)
    X_all5 = np.column_stack([X_gait, X_cwt, X_ws, D5v])
    r2, mae, rv, rho, a = best_alpha(X_all5, yv)
    report("Gait+CWT+WalkSway+Demo(5)", X_all5.shape[1], r2, mae, rv, rho, a)

    # Feature-selected: top features by correlation
    for K in [10, 15, 20, 25]:
        top = corr_df.head(K)
        top_feats = top['feature'].tolist()
        # Collect indices from each matrix
        idx_list = []
        X_combined = np.column_stack([X_gait, X_cwt, X_ws])
        combined_cols = gait_cols + cwt_cols + ws_cols
        for f in top_feats:
            if f in combined_cols:
                idx_list.append(combined_cols.index(f))
        if idx_list:
            X_sel = X_combined[:, idx_list]
            r2, mae, rv, rho, a = best_alpha(np.column_stack([X_sel, D5v]), yv)
            report(f"Top-{K} correlated + Demo(5)", len(idx_list)+5, r2, mae, rv, rho, a)

    # Gait only with Demo(5) — direct comparison to home baseline
    r2, mae, rv, rho, a = best_alpha(np.column_stack([X_gait, D5v]), yv)
    report("Gait(10)+Demo(5)", X_gait.shape[1]+5, r2, mae, rv, rho, a)

    print(f"\n{'─'*90}")
    print(f"BASELINES (for comparison):")
    print(f"  Clinic actual:        Gait+CWT+WS+Demo(4), 55f, α=10  →  R²=0.806  MAE=100ft")
    print(f"  Home clinic-informed: Gait(8)+Demo(5), 13f, α=20       →  R²=0.507  MAE=174ft")
    print(f"  Home clinic-free v2:  Top-20+Demo(5), 25f, α=20        →  R²=0.441  MAE=191ft")
    print(f"{'─'*90}")

    print(f"\nDone in {time.time()-t0:.0f}s")
