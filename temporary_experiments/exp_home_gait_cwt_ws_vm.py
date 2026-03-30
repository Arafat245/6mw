#!/usr/bin/env python3
"""
Home Gait(11), CWT(28), WalkSway(12) using VM-based extraction.
Uses magnitude signal (no axis alignment needed) from Top-10 clean bouts ≥60s.
Quality filter: drift ≤ 0.5g, orientation change ≤ 10°.
Spearman Top-11 inside LOO, Ridge.

Result: Gait R²=0.145, CWT R²=0.150, WalkSway R²=0.061

Run:  python temporary_experiments/exp_home_gait_cwt_ws_vm.py
"""
import math, time, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent
BOUT_DIR = BASE / 'walking_bouts'
NPZ_DIR = BASE / 'home_full_recording_npz'
FS = 30
FT2M = 0.3048

from clinic.reproduce_c2 import extract_gait10, extract_cwt
from clinic.extract_walking_sway import extract_walking_sway


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def spearman_loo(X_accel, X_demo, y, K=11, alpha=20):
    n_accel = X_accel.shape[1]
    X_all = np.column_stack([X_accel, X_demo]) if X_demo is not None else X_accel
    n_demo = X_demo.shape[1] if X_demo is not None else 0
    demo_idx = list(range(n_accel, n_accel + n_demo)) if n_demo > 0 else []
    K_use = min(K, n_accel)
    preds = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), dtype=bool); tr[i] = False
        corrs = [abs(spearmanr(X_all[tr, j], y[tr])[0]) if np.std(X_all[tr, j]) > 0 else 0
                 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K_use]
        selected = top_k + demo_idx
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_all[tr][:, selected]), y[tr])
        preds[i] = m.predict(sc.transform(X_all[i:i+1][:, selected]))[0]
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y * FT2M, preds * FT2M)
    rho = spearmanr(y, preds)[0]
    return r2, mae, rho


def aggregate_feature_dicts(feat_list):
    """Aggregate list of feature dicts into one row with med/iqr/p10/p90/max/cv."""
    if not feat_list:
        return {}
    names = sorted(feat_list[0].keys())
    arr = np.array([[f.get(k, np.nan) for k in names] for f in feat_list])
    row = {}
    for j, name in enumerate(names):
        col = arr[:, j]; valid = col[np.isfinite(col)]
        if len(valid) < 2: continue
        row[f'{name}_med'] = np.median(valid)
        row[f'{name}_iqr'] = np.percentile(valid, 75) - np.percentile(valid, 25)
        row[f'{name}_p10'] = np.percentile(valid, 10)
        row[f'{name}_p90'] = np.percentile(valid, 90)
        row[f'{name}_max'] = np.max(valid)
        row[f'{name}_cv'] = np.std(valid) / (np.mean(valid) + 1e-12)
    return row


def vm_to_clinic_df(vm, fs):
    """Create clinic-like DataFrame from raw VM signal (no mean subtraction)."""
    b, a = butter(4, [0.25, 2.5], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b, a, vm)
    enmo = np.maximum(vm - 1.0, 0.0)
    return pd.DataFrame({
        'AP': vm, 'ML': vm, 'VT': vm,
        'AP_bp': vm_bp, 'ML_bp': vm_bp, 'VT_bp': vm_bp,
        'VM_dyn': vm, 'VM_raw': vm,
        'ENMO': enmo, 'fs': fs,
    })


def bout_quality(xyz, fs=30):
    """Return (drift_g, orient_deg) for quality filtering."""
    first = xyz[:fs].mean(axis=0)
    last = xyz[-fs:].mean(axis=0)
    drift = np.linalg.norm(last - first)
    orient = 0.0
    if len(xyz) >= 60:
        half = len(xyz) // 2
        g1 = xyz[:half].mean(axis=0)
        g2 = xyz[half:].mean(axis=0)
        cos_a = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-12)
        orient = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    return drift, orient


def get_clean_topN(subj_key, N=10, min_sec=60, max_drift=0.5, max_orient=10):
    """Get top-N longest clean bouts ≥ min_sec."""
    subj_dir = BOUT_DIR / subj_key
    if not subj_dir.exists():
        return []
    candidates = []
    for bf in sorted(subj_dir.glob('bout_*.csv')):
        try:
            dur = float(bf.stem.split('_')[-1].replace('s', ''))
            if dur < min_sec:
                continue
            df = pd.read_csv(bf)
            xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
            drift, orient = bout_quality(xyz)
            if drift <= max_drift and orient <= max_orient:
                candidates.append((dur, bf))
        except:
            continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [bf for _, bf in candidates[:N]]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(y)

    # Demographics
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))

    print(f"n={n}, VM-based Gait/CWT/WalkSway from Top-10 clean bouts ≥60s")
    print(f"Quality filter: drift ≤ 0.5g, orientation change ≤ 10°\n")

    gait_rows, cwt_rows, ws_rows = [], [], []
    bout_counts = []

    for i, (_, r) in enumerate(subj_df.iterrows()):
        top_bouts = get_clean_topN(r['key'], N=10, min_sec=60,
                                    max_drift=0.5, max_orient=10)
        bout_counts.append(len(top_bouts))
        gait_per, cwt_per, ws_per = [], [], []

        for bf_path in top_bouts:
            try:
                df = pd.read_csv(bf_path)
                xyz = df[['X', 'Y', 'Z']].values.astype(np.float64)
                if len(xyz) < int(10 * FS):
                    continue

                vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)

                # Gait from raw VM
                df_vm = vm_to_clinic_df(vm, FS)
                gf = extract_gait10(df_vm)
                gf['vt_rms_g'] = float(np.sqrt(np.mean(vm**2)))
                gait_per.append(gf)

                # CWT from raw XYZ (uses VM internally)
                cwt_per.append(extract_cwt(xyz.astype(np.float32)))

                # WalkSway from raw VM (no mean subtraction)
                ws = extract_walking_sway(vm, vm, vm)
                ws['ml_over_enmo'] = (gf.get('ml_rms_g', 0) /
                    gf.get('enmo_mean_g', 1e-12) if gf.get('enmo_mean_g', 0) > 0 else np.nan)
                ws['ml_over_vt'] = 1.0
                ws_per.append(ws)
            except:
                continue

        gait_rows.append(aggregate_feature_dicts(gait_per))
        cwt_rows.append(aggregate_feature_dicts(cwt_per))
        ws_rows.append(aggregate_feature_dicts(ws_per))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] {r['key']}: {len(top_bouts)} clean bouts", flush=True)

    bc = np.array(bout_counts)
    X_gait = impute(pd.DataFrame(gait_rows).replace([np.inf, -np.inf], np.nan).values.astype(float))
    X_cwt = impute(pd.DataFrame(cwt_rows).replace([np.inf, -np.inf], np.nan).values.astype(float))
    X_ws = impute(pd.DataFrame(ws_rows).replace([np.inf, -np.inf], np.nan).values.astype(float))

    print(f"\nBout stats: median={np.median(bc):.0f}, mean={np.mean(bc):.1f}, zero={np.sum(bc==0)}")
    print(f"Features: Gait={X_gait.shape[1]}, CWT={X_cwt.shape[1]}, WalkSway={X_ws.shape[1]}")

    print(f"\n{'='*70}")
    print(f"Results (Spearman Top-K inside LOO, Ridge):")
    print(f"{'='*70}")

    for name, X, K, alpha in [
        ('Gait', X_gait, 11, 5),
        ('CWT', X_cwt, 11, 20),
        ('WalkSway', X_ws, 11, 5),
        ('Gait+CWT+WS+Demo', np.column_stack([X_gait, X_cwt, X_ws]), 20, 5),
    ]:
        d = X_demo if 'Demo' in name else None
        r2, mae, rho = spearman_loo(X, d, y, K=K, alpha=alpha)
        print(f"  {name:25s} ({X.shape[1]:3d}f→{K:2d})  R²={r2:.4f}  MAE={mae:.1f}m  ρ={rho:.3f}")

    print(f"\nDone in {time.time()-t0:.0f}s")
