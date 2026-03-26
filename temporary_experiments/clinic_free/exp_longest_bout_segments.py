#!/usr/bin/env python3
"""
Extract clinic-identical features from longest home bout in 10 configurations.
SAME preprocessing for home and clinic (from clinic/reproduce_c2.py).
Same number of features in each row.

Exp 1: Full longest bout (home) vs full 6MWT (clinic)
Exp 2: First 6 min
Exp 3: Last 6 min
Exp 4: Minute-by-minute median
Exp 5-10: Minute 1 through Minute 6 individually
"""
import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# Import the EXACT clinic preprocessing functions
from clinic.reproduce_c2 import (PreprocConfig, get_fs_from_timestamps, trim_edges,
                                  resample_uniform, align_to_ap_ml_vt,
                                  butter_bandpass, zero_phase_filter,
                                  extract_gait10, extract_cwt)
from clinic.extract_walking_sway import extract_walking_sway
from temporary_experiments.clinic_free.exp_clinic_free_v2 import detect_walking_bouts_clinicfree

FS = 30
HOME_DIR = BASE / 'csv_home_daytime'
RAW = BASE / 'csv_raw2'
CFG = PreprocConfig()


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


def trim_and_resample(raw_xyz, fs_orig):
    """Step 1-2: Trim 10s edges + resample to 30 Hz. Done ONCE per recording."""
    arr_trim = trim_edges(raw_xyz, fs=fs_orig, trim_seconds=CFG.trim_seconds)
    arr_rs = resample_uniform(arr_trim, src_fs=fs_orig, dst_fs=CFG.target_fs)
    return arr_rs


def preprocess_trimmed(arr_30hz):
    """Steps 3-7 on already-trimmed-and-resampled data at 30 Hz.
    Gravity removal, Rodrigues, PCA, bandpass, VM, ENMO."""
    fs = CFG.target_fs
    apmlvt_dyn, g_est = align_to_ap_ml_vt(arr_30hz, fs=fs, cfg=CFG)
    lo, hi = CFG.step_band_hz
    b, a = butter_bandpass(lo, hi, fs, order=CFG.filter_order)
    apmlvt_bp = zero_phase_filter(apmlvt_dyn, b, a)
    vm_raw = np.linalg.norm(arr_30hz, axis=1)
    enmo = np.maximum(vm_raw - 1.0, 0.0)

    df = pd.DataFrame({
        'AP': apmlvt_dyn[:, 0], 'ML': apmlvt_dyn[:, 1], 'VT': apmlvt_dyn[:, 2],
        'AP_bp': apmlvt_bp[:, 0], 'ML_bp': apmlvt_bp[:, 1], 'VT_bp': apmlvt_bp[:, 2],
        'VM_dyn': np.linalg.norm(apmlvt_dyn, axis=1), 'VM_raw': vm_raw, 'ENMO': enmo,
        'cohort': 'X', 'subj_id': 0, 'year': 0, 'sixmwd': 0,
        'fs': fs, 'trim_s': 0, 'lp_hz': CFG.gravity_lp_hz,
        'bp_lo_hz': lo, 'bp_hi_hz': hi
    })
    return df


def safe_extract(arr_trimmed_30hz, arr_raw_for_cwt):
    """Extract Gait(11)+CWT(28)+WalkSway(12).
    Gait/WalkSway: from trimmed+resampled 30Hz data.
    CWT: from raw (untrimmed) data — matches original pipeline."""
    if len(arr_trimmed_30hz) < int(30 * CFG.target_fs):
        return None, None, None, False
    try:
        pp = preprocess_trimmed(arr_trimmed_30hz)
        # Gait(11): gait10 + vt_rms_g
        gait = extract_gait10(pp)
        gait['vt_rms_g'] = float(np.sqrt(np.mean(pp['VT'].values ** 2)))
        # CWT(28): from RAW data (not trimmed), matching original pipeline
        cwt = extract_cwt(arr_raw_for_cwt.astype(np.float32))
        # WalkSway(12): 10 normalized + ml_over_enmo + ml_over_vt
        ws = extract_walking_sway(pp['AP'].values, pp['ML'].values, pp['VT'].values)
        enmo_mean = gait.get('enmo_mean_g', float(np.mean(pp['ENMO'].values)))
        vt_rms = gait['vt_rms_g']
        ml_rms = gait.get('ml_rms_g', float(np.sqrt(np.mean(pp['ML'].values ** 2))))
        ws['ml_over_enmo'] = ml_rms / enmo_mean if enmo_mean > 0 else np.nan
        ws['ml_over_vt'] = ml_rms / vt_rms if vt_rms > 0 else np.nan
        return gait, cwt, ws, True
    except:
        return None, None, None, False


def build_matrices(gait_list, cwt_list, ws_list, valid):
    vmask = np.array(valid, dtype=bool)
    gait_df = pd.DataFrame([g for g, v in zip(gait_list, vmask) if v])
    X_gait = gait_df.values.astype(float)
    cwt_df = pd.DataFrame([c for c, v in zip(cwt_list, vmask) if v]).replace([np.inf, -np.inf], np.nan)
    X_cwt = cwt_df.values.astype(float)
    ws_df = pd.DataFrame([w for w, v in zip(ws_list, vmask) if v])
    X_ws = ws_df.values.astype(float)
    for X in [X_gait, X_cwt, X_ws]:
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j]) | np.isinf(X[:, j])
            if m.all(): X[:, j] = 0
            elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X_gait, X_cwt, X_ws, vmask


def run_table(label, X_gait_h, X_cwt_h, X_ws_h, X_demo_v,
              X_gait_c, X_cwt_c, X_ws_c,
              Ec_pca50, Eh_pca50, Ec_limu, Eh_limu, y_v):
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    print(f"  {'Feature Set':<40s} {'#f':>3s}  {'Home R²':>10s}  {'Clinic R²':>10s}")
    print(f"  {'-'*70}")

    rows = []
    nd = X_demo_v.shape[1]

    def add(name, nf, h_r2, c_r2):
        rows.append({'Feature Set': name, '#f': nf, 'Home R²': round(h_r2, 4), 'Clinic R²': round(c_r2, 4)})
        print(f"  {name:<40s} {nf:>3d}   {h_r2:>10.4f}   {c_r2:>10.4f}")

    ng = X_gait_h.shape[1]  # 11 (gait10 + vt_rms_g)
    nc = X_cwt_h.shape[1]   # 28
    nw = X_ws_h.shape[1]    # 12 (10 normalized + ml_over_enmo + ml_over_vt)
    add('Gait', ng, best_alpha(X_gait_h, y_v)[0], best_alpha(X_gait_c, y_v)[0])
    add('CWT', nc, best_alpha(X_cwt_h, y_v)[0], best_alpha(X_cwt_c, y_v)[0])
    add('WalkSway', nw, best_alpha(X_ws_h, y_v)[0], best_alpha(X_ws_c, y_v)[0])
    add('Gait+Demo', ng+nd, best_alpha(np.column_stack([X_gait_h, X_demo_v]), y_v)[0],
        best_alpha(np.column_stack([X_gait_c, X_demo_v]), y_v)[0])
    add('Gait+CWT+WalkSway+Demo', ng+nc+nw+nd,
        best_alpha(np.column_stack([X_gait_h, X_cwt_h, X_ws_h, X_demo_v]), y_v)[0],
        best_alpha(np.column_stack([X_gait_c, X_cwt_c, X_ws_c, X_demo_v]), y_v)[0])

    nl = Ec_limu.shape[1]
    add('MOMENT PCA50', 50, best_alpha(Eh_pca50, y_v)[0], best_alpha(Ec_pca50, y_v)[0])
    add('MOMENT PCA50+Demo', 50+nd, best_alpha(np.column_stack([Eh_pca50, X_demo_v]), y_v)[0],
        best_alpha(np.column_stack([Ec_pca50, X_demo_v]), y_v)[0])
    add('LimuBERT', nl, best_alpha(Eh_limu, y_v)[0], best_alpha(Ec_limu, y_v)[0])
    add('LimuBERT+Demo', nl+nd, best_alpha(np.column_stack([Eh_limu, X_demo_v]), y_v)[0],
        best_alpha(np.column_stack([Ec_limu, X_demo_v]), y_v)[0])

    return pd.DataFrame(rows)


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

    # Demographics (Demo(4) same as clinic: cohort_M, Age, Sex, Height)
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float)
    for j in range(X_demo.shape[1]):
        m = np.isnan(X_demo[:, j])
        if m.any(): X_demo[m, j] = np.nanmedian(X_demo[:, j])

    # Foundation model embeddings
    valid_102 = ~((ids['cohort'] == 'M') & (ids['subj_id'] == 22))
    cidx = np.where(clinic_valid)[0]
    Ec_pca50 = PCA(n_components=50).fit_transform(np.load(str(BASE / 'feats/moment_clinic_raw.npy'))[valid_102.values][cidx])
    Eh_pca50 = PCA(n_components=50).fit_transform(np.load(str(BASE / 'feats/moment_home_raw.npy'))[valid_102.values][cidx])
    Ec_limu = np.load(str(BASE / 'results_raw_pipeline/emb_limubert_clinic.npy'))[valid_102.values][cidx]
    Eh_limu = np.load(str(BASE / 'results_raw_pipeline/emb_limubert_home.npy'))[valid_102.values][cidx]

    # ── Extract features ──
    cache = OUT / 'longest_bout_proper_cache.npz'
    if cache.exists():
        print("Loading from cache...", flush=True)
        d = np.load(cache, allow_pickle=True)
        home_configs = d['home_configs'].item()
        clinic_configs = d['clinic_configs'].item()
    else:
        print(f"n={n}. Extracting with EXACT clinic preprocessing...", flush=True)
        config_names = ['full', 'first6', 'last6', 'minbymin',
                        'min1', 'min2', 'min3', 'min4', 'min5', 'min6']
        home_configs = {k: {'gait': [], 'cwt': [], 'ws': [], 'valid': []} for k in config_names}
        clinic_configs = {k: {'gait': [], 'cwt': [], 'ws': [], 'valid': []} for k in config_names}

        one_min = int(60 * FS)
        six_min = int(6 * 60 * FS)

        def append_result(cfg, g, c, w, v):
            cfg['gait'].append(g); cfg['cwt'].append(c); cfg['ws'].append(w); cfg['valid'].append(v)
        def append_none(cfg):
            append_result(cfg, None, None, None, False)

        for i, (_, r) in enumerate(ids_v.iterrows()):
            fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"

            # ── HOME: longest bout ──
            fp = HOME_DIR / fn
            if not fp.exists():
                for k in config_names: append_none(home_configs[k])
            else:
                xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
                bouts = detect_walking_bouts_clinicfree(xyz, FS, min_bout_sec=10, merge_gap_sec=5)
                if not bouts:
                    for k in config_names: append_none(home_configs[k])
                else:
                    longest = max(bouts, key=lambda b: b[1] - b[0])
                    bout_xyz = xyz[longest[0]:longest[1]]

                    fs30 = int(CFG.target_fs)
                    trim_samples = int(10 * FS)  # 10s in original fs
                    six_min_raw = 6 * 60 * FS
                    one_min_raw = 60 * FS
                    six_min_30 = 6 * 60 * fs30
                    one_min_30 = 60 * fs30

                    # Trim 10s from each end + resample ONCE for full/first6/last6
                    try:
                        h_trimmed = trim_and_resample(bout_xyz, FS)
                    except:
                        for k in config_names: append_none(home_configs[k])
                        continue

                    # full: trimmed for Gait/WS, raw for CWT
                    g, c, w, v = safe_extract(h_trimmed, bout_xyz)
                    append_result(home_configs['full'], g, c, w, v)

                    # first6: from trimmed data, raw slice for CWT
                    h_first6 = h_trimmed[:min(six_min_30, len(h_trimmed))]
                    raw_first6 = bout_xyz[:min(six_min_raw, len(bout_xyz))]
                    g, c, w, v = safe_extract(h_first6, raw_first6)
                    append_result(home_configs['first6'], g, c, w, v)

                    # last6
                    h_last6 = h_trimmed[max(0, len(h_trimmed)-six_min_30):]
                    raw_last6 = bout_xyz[max(0, len(bout_xyz)-six_min_raw):]
                    g, c, w, v = safe_extract(h_last6, raw_last6)
                    append_result(home_configs['last6'], g, c, w, v)

                    # Per-minute: trim 10s from edge of first minute and last minute only
                    # Interior minutes are NOT trimmed
                    n_mins_raw = len(bout_xyz) // one_min_raw
                    per_min = []
                    for mi in range(min(n_mins_raw, 6)):
                        raw_min = bout_xyz[mi*one_min_raw:(mi+1)*one_min_raw]
                        if mi == 0:
                            # First minute: trim 10s from start only
                            trimmed_min = raw_min[trim_samples:]
                        elif mi == min(n_mins_raw, 6) - 1:
                            # Last minute: trim 10s from end only
                            trimmed_min = raw_min[:len(raw_min)-trim_samples]
                        else:
                            # Interior minutes: no trimming
                            trimmed_min = raw_min
                        # Resample trimmed minute to 30 Hz
                        if len(trimmed_min) < 10 * FS:
                            per_min.append((None, None, None, False))
                            continue
                        rs_min = resample_uniform(trimmed_min, src_fs=FS, dst_fs=CFG.target_fs)
                        per_min.append(safe_extract(rs_min, raw_min))

                    # minbymin median
                    valid_mins = [(g, c, w) for g, c, w, v in per_min if v]
                    if valid_mins:
                        gdf = pd.DataFrame([x[0] for x in valid_mins])
                        cdf = pd.DataFrame([x[1] for x in valid_mins])
                        wdf = pd.DataFrame([x[2] for x in valid_mins])
                        append_result(home_configs['minbymin'],
                                      gdf.median().to_dict(), cdf.median().to_dict(),
                                      wdf.median().to_dict(), True)
                    else:
                        append_none(home_configs['minbymin'])

                    for mi in range(6):
                        if mi < len(per_min):
                            g, c, w, v = per_min[mi]
                            append_result(home_configs[f'min{mi+1}'], g, c, w, v)
                        else:
                            append_none(home_configs[f'min{mi+1}'])

            # ── CLINIC: 6MWT with SAME preprocessing ──
            clinic_path = RAW / fn
            if not clinic_path.exists():
                for k in config_names: append_none(clinic_configs[k])
            else:
                cdf_raw = pd.read_csv(clinic_path)
                fs_clinic = get_fs_from_timestamps(cdf_raw['Timestamp'].values)
                clinic_xyz = cdf_raw[['X', 'Y', 'Z']].values.astype(np.float64)

                fs30 = int(CFG.target_fs)
                c_trim_samples = int(10 * fs_clinic)
                c_six_min = int(6 * 60 * fs_clinic)
                c_one_min = int(60 * fs_clinic)
                six_min_30 = 6 * 60 * fs30
                one_min_30 = 60 * fs30

                # Trim 10s edges + resample ONCE
                try:
                    c_trimmed = trim_and_resample(clinic_xyz, fs_clinic)
                except:
                    for k in config_names: append_none(clinic_configs[k])
                    continue

                # Clinic: entire recording IS the 6MWT (~6 min)
                # Always use full raw for CWT, trimmed for Gait/WS
                # full/first6/last6 are identical for clinic since trimmed < 6 min
                g, c, w, v = safe_extract(c_trimmed, clinic_xyz)
                append_result(clinic_configs['full'], g, c, w, v)
                append_result(clinic_configs['first6'], g, c, w, v)
                append_result(clinic_configs['last6'], g, c, w, v)

                # Per-minute: trim 10s from edge of first and last minute only
                n_cmins_raw = len(clinic_xyz) // c_one_min
                clinic_per_min = []
                for mi in range(min(n_cmins_raw, 6)):
                    raw_min = clinic_xyz[mi*c_one_min:(mi+1)*c_one_min]
                    if mi == 0:
                        trimmed_min = raw_min[c_trim_samples:]
                    elif mi == min(n_cmins_raw, 6) - 1:
                        trimmed_min = raw_min[:len(raw_min)-c_trim_samples]
                    else:
                        trimmed_min = raw_min
                    if len(trimmed_min) < 10 * fs_clinic:
                        clinic_per_min.append((None, None, None, False))
                        continue
                    rs_min = resample_uniform(trimmed_min, src_fs=fs_clinic, dst_fs=CFG.target_fs)
                    clinic_per_min.append(safe_extract(rs_min, raw_min))

                # minbymin median
                valid_cmins = [(g, c, w) for g, c, w, v in clinic_per_min if v]
                if valid_cmins:
                    gdf = pd.DataFrame([x[0] for x in valid_cmins])
                    cdf = pd.DataFrame([x[1] for x in valid_cmins])
                    wdf = pd.DataFrame([x[2] for x in valid_cmins])
                    append_result(clinic_configs['minbymin'],
                                  gdf.median().to_dict(), cdf.median().to_dict(),
                                  wdf.median().to_dict(), True)
                else:
                    append_none(clinic_configs['minbymin'])

                for mi in range(6):
                    if mi < len(clinic_per_min):
                        g, c, w, v = clinic_per_min[mi]
                        append_result(clinic_configs[f'min{mi+1}'], g, c, w, v)
                    else:
                        append_none(clinic_configs[f'min{mi+1}'])

            if (i + 1) % 20 == 0:
                bd = len(bout_xyz)/FS if 'bout_xyz' in dir() and bouts else 0
                print(f"  [{i+1}/{n}] {fn}: home longest={bd:.0f}s", flush=True)

        np.savez(cache, home_configs=home_configs, clinic_configs=clinic_configs, allow_pickle=True)
        print("  Cached.", flush=True)

    # ── Run all 10 tables ──
    table_labels = [
        ('full', 'Exp 1: Full Longest Bout'),
        ('first6', 'Exp 2: First 6 min'),
        ('last6', 'Exp 3: Last 6 min'),
        ('minbymin', 'Exp 4: Minute-by-Minute Median'),
        ('min1', 'Exp 5: Minute 1 Only'),
        ('min2', 'Exp 6: Minute 2 Only'),
        ('min3', 'Exp 7: Minute 3 Only'),
        ('min4', 'Exp 8: Minute 4 Only'),
        ('min5', 'Exp 9: Minute 5 Only'),
        ('min6', 'Exp 10: Minute 6 Only'),
    ]

    for config_name, config_label in table_labels:
        h_cfg = home_configs[config_name]
        c_cfg = clinic_configs[config_name]
        both_valid = np.array(h_cfg['valid'], dtype=bool) & np.array(c_cfg['valid'], dtype=bool)
        nv = both_valid.sum()
        if nv < 30:
            print(f"\n  {config_label}: only {nv} valid — skipping")
            continue

        X_gait_h, X_cwt_h, X_ws_h, _ = build_matrices(h_cfg['gait'], h_cfg['cwt'], h_cfg['ws'], both_valid)
        X_gait_c, X_cwt_c, X_ws_c, _ = build_matrices(c_cfg['gait'], c_cfg['cwt'], c_cfg['ws'], both_valid)

        tbl = run_table(f"{config_label} (n={nv})",
                        X_gait_h, X_cwt_h, X_ws_h, X_demo[both_valid],
                        X_gait_c, X_cwt_c, X_ws_c,
                        Ec_pca50[both_valid], Eh_pca50[both_valid],
                        Ec_limu[both_valid], Eh_limu[both_valid],
                        y[both_valid])
        tbl.to_csv(OUT / f'table_{config_name}.csv', index=False)

    print(f"\n{'='*100}")
    print(f"Done in {time.time()-t0:.0f}s")
