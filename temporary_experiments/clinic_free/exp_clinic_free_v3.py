#!/usr/bin/env python3
"""
Clinic-free experiments v3: Self-referencing bout selection + best-daily-segment.
Builds on v2 results (best clinic-free R²=0.441 with Top-20+Demo5).

Exp A: Self-referencing bout selection (consensus signature)
Exp B: Population-mean bout selection
Exp C: Best-daily-segment features (peak 6-min window per day)
Exp D: Hybrid of best approaches
Exp E: Clinic-free longer windows with ENMO ranking
"""
import sys, warnings, time, math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

FS = 30
HOME_DIR = BASE / 'csv_home_daytime'

# Import from v2
from exp_clinic_free_v2 import (detect_walking_bouts_clinicfree, extract_segment_features,
                                 preprocess_segment, eval_loo, best_alpha, report)

# Import walking signature computation
from home.home_hybrid_models_v2 import compute_walking_signature


def load_base_data():
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid = []
    for _, r in ids101.iterrows():
        fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
        clinic_valid.append((PREPROC2 / fn).exists())
    clinic_valid = np.array(clinic_valid)
    ids101 = ids101[clinic_valid].reset_index(drop=True)
    y = ids101['sixmwd'].values.astype(float)

    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids101.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo_5 = p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float)
    for j in range(X_demo_5.shape[1]):
        m = np.isnan(X_demo_5[:, j])
        if m.any(): X_demo_5[m, j] = np.nanmedian(X_demo_5[:, j])

    return ids101, y, X_demo_5


def extract_gait_from_segment(xyz, fs=30.0):
    """Extract gait features from a preprocessed walking segment. Returns dict or None."""
    return extract_segment_features(xyz, fs)


# ══════════════════════════════════════════════════════════════════
# EXP A: Self-referencing bout selection
# ══════════════════════════════════════════════════════════════════
def exp_a_self_referencing(ids101, y, X_demo_5):
    print(f"\n{'─'*90}")
    print("EXP A: Self-referencing bout selection (consensus signature)")
    print(f"{'─'*90}")

    cache = OUT / 'selfref_features_cache.npz'
    if cache.exists():
        print("  Loading from cache...", flush=True)
        d = np.load(cache, allow_pickle=True)
        X_selfref = d['X_selfref']
        X_popref = d['X_popref']
        feat_names = list(d['feat_names'])
        valid = d['valid']
    else:
        print("  Computing self-referencing features...", flush=True)
        all_selfref, all_popref, valid = [], [], []
        feat_names = None

        # First pass: collect all walking signatures to compute population mean
        all_sigs = []
        all_bouts_per_subj = []
        for i, (_, r) in enumerate(ids101.iterrows()):
            fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
            fp = HOME_DIR / fn
            if not fp.exists():
                all_bouts_per_subj.append([]); continue
            xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
            bouts = detect_walking_bouts_clinicfree(xyz, FS, min_bout_sec=10, merge_gap_sec=5)

            subj_sigs = []
            subj_bouts = []
            for s, e in bouts:
                dur = (e - s) / FS
                if dur < 15: continue  # need at least 15s for reliable signature
                sig = compute_walking_signature(xyz[s:e], FS)
                if np.all(np.isfinite(sig)):
                    subj_sigs.append(sig)
                    subj_bouts.append((s, e, sig))
            all_sigs.extend(subj_sigs)
            all_bouts_per_subj.append(subj_bouts)
            if (i+1) % 20 == 0:
                print(f"    Pass 1 [{i+1}/{len(ids101)}] {len(subj_bouts)} bouts", flush=True)

        # Population mean signature
        pop_sig = np.median(np.array(all_sigs), axis=0) if all_sigs else np.zeros(7)
        print(f"  Population signature: {pop_sig}", flush=True)

        # Second pass: rank bouts by self-consensus and population similarity
        for i, (_, r) in enumerate(ids101.iterrows()):
            fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
            fp = HOME_DIR / fn
            subj_bouts = all_bouts_per_subj[i]

            if not subj_bouts or not fp.exists():
                all_selfref.append(None); all_popref.append(None); valid.append(False)
                continue

            xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)

            # Self-consensus: median of all bout signatures
            sigs = np.array([b[2] for b in subj_bouts])
            consensus = np.median(sigs, axis=0)

            # Score bouts by similarity to consensus
            def cosine_sim(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

            scored_self = [(s, e, cosine_sim(sig, consensus)) for s, e, sig in subj_bouts]
            scored_self.sort(key=lambda x: x[2], reverse=True)

            scored_pop = [(s, e, cosine_sim(sig, pop_sig)) for s, e, sig in subj_bouts]
            scored_pop.sort(key=lambda x: x[2], reverse=True)

            # Extract per-bout features from top-20 bouts, aggregate with median
            def aggregate_top_bouts(scored, xyz, n_top=20):
                bout_feats = []
                for s, e, sim in scored[:n_top]:
                    feats = extract_segment_features(xyz[s:e], FS)
                    if feats is not None:
                        bout_feats.append(feats)
                if len(bout_feats) < 2:
                    return None
                keys = sorted(bout_feats[0].keys())
                arr = np.array([[bf.get(k, np.nan) for k in keys] for bf in bout_feats])
                agg = {}
                for j, k in enumerate(keys):
                    col = arr[:, j]
                    v = col[np.isfinite(col)]
                    agg[f'{k}_med'] = np.median(v) if len(v) > 0 else np.nan
                return agg, keys

            result_self = aggregate_top_bouts(scored_self, xyz)
            result_pop = aggregate_top_bouts(scored_pop, xyz)

            if result_self is None:
                all_selfref.append(None); all_popref.append(None); valid.append(False)
            else:
                all_selfref.append(result_self[0])
                all_popref.append(result_pop[0] if result_pop else result_self[0])
                if feat_names is None:
                    feat_names = sorted(result_self[0].keys())
                valid.append(True)

            if (i+1) % 20 == 0:
                print(f"    Pass 2 [{i+1}/{len(ids101)}]", flush=True)

        valid = np.array(valid)

        # Build matrices
        X_selfref = np.full((len(ids101), len(feat_names)), np.nan)
        X_popref = np.full((len(ids101), len(feat_names)), np.nan)
        for i in range(len(ids101)):
            if valid[i]:
                for j, fn in enumerate(feat_names):
                    X_selfref[i, j] = all_selfref[i].get(fn, np.nan)
                    X_popref[i, j] = all_popref[i].get(fn, np.nan)

        np.savez(cache, X_selfref=X_selfref, X_popref=X_popref,
                 feat_names=feat_names, valid=valid)
        print(f"  Cached. Valid: {valid.sum()}/{len(ids101)}", flush=True)

    # Impute and evaluate
    vmask = valid.astype(bool)
    nv = vmask.sum()
    print(f"  Valid subjects: {nv}")

    for label, X_feat in [('Self-consensus', X_selfref), ('Population-mean', X_popref)]:
        Xv = X_feat[vmask].copy()
        yv = y[vmask]
        Dv = X_demo_5[vmask]
        for j in range(Xv.shape[1]):
            m = np.isnan(Xv[:, j])
            if m.any(): Xv[m, j] = np.nanmedian(Xv[:, j])

        # All features
        X = np.column_stack([Xv, Dv])
        r2, mae, rv, rho, a = best_alpha(X, yv)
        report(f"{label} all({Xv.shape[1]})+Demo(5) [n={nv}]", Xv.shape[1]+5, r2, mae, rv, rho, a)

        # Correlation-selected features
        corrs = []
        for j in range(Xv.shape[1]):
            rr, pp = spearmanr(Xv[:, j], yv)
            if np.isfinite(rr): corrs.append((j, abs(rr), rr))
        corrs.sort(key=lambda x: x[1], reverse=True)
        for K in [5, 8, 10, 15]:
            top_idx = [c[0] for c in corrs[:K]]
            X = np.column_stack([Xv[:, top_idx], Dv])
            r2, mae, rv, rho, a = best_alpha(X, yv)
            report(f"{label} top-{K}+Demo(5) [n={nv}]", K+5, r2, mae, rv, rho, a)


# ══════════════════════════════════════════════════════════════════
# EXP C: Best daily segment
# ══════════════════════════════════════════════════════════════════
def exp_c_best_daily_segment(ids101, y, X_demo_5):
    print(f"\n{'─'*90}")
    print("EXP C: Best-daily-segment features (peak 6-min window per day)")
    print(f"{'─'*90}")

    cache = OUT / 'daily_segment_cache.npz'
    if cache.exists():
        print("  Loading from cache...", flush=True)
        d = np.load(cache, allow_pickle=True)
        X_daily = d['X_daily']
        daily_feat_names = list(d['feat_names'])
        daily_valid = d['valid']
    else:
        print("  Finding best daily segments...", flush=True)
        all_daily_feats = []
        daily_valid = []
        daily_feat_names = None

        for i, (_, r) in enumerate(ids101.iterrows()):
            fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
            fp = HOME_DIR / fn
            if not fp.exists():
                all_daily_feats.append(None); daily_valid.append(False); continue

            xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
            n_samples = len(xyz)

            # Split into days (~15h daytime = 15*3600*30 = 1,620,000 samples)
            day_len = 15 * 3600 * FS
            n_days = n_samples // day_len
            if n_days < 1:
                # Treat whole recording as one day
                n_days = 1
                day_len = n_samples

            # Per day: find best 6-min window by ENMO
            seg_len = 6 * 60 * FS  # 360s = 10800 samples
            day_feats = []

            for d_idx in range(n_days):
                start = d_idx * day_len
                end = min(start + day_len, n_samples)
                day_xyz = xyz[start:end]
                if len(day_xyz) < seg_len:
                    continue

                # Compute ENMO per 6-min segment
                vm = np.sqrt(day_xyz[:, 0]**2 + day_xyz[:, 1]**2 + day_xyz[:, 2]**2)
                enmo = np.maximum(vm - 1.0, 0.0)
                n_segs = (len(enmo) - seg_len) // (60 * FS) + 1  # slide by 1 min
                if n_segs < 1: continue

                # Find peak 6-min window
                best_enmo, best_start = -1, 0
                step = 60 * FS  # 1-min step
                for s in range(0, len(enmo) - seg_len + 1, step):
                    seg_enmo = np.mean(enmo[s:s+seg_len])
                    if seg_enmo > best_enmo:
                        best_enmo = seg_enmo
                        best_start = s

                # Extract gait features from best window
                best_xyz = day_xyz[best_start:best_start+seg_len]
                feats = extract_segment_features(best_xyz, FS)
                if feats is not None:
                    day_feats.append(feats)

            # Skip first and last day if ≥ 3 days
            if len(day_feats) >= 3:
                day_feats = day_feats[1:-1]

            if not day_feats:
                all_daily_feats.append(None); daily_valid.append(False); continue

            # Average across days
            if daily_feat_names is None:
                daily_feat_names = sorted(day_feats[0].keys())
            arr = np.array([[df.get(k, np.nan) for k in daily_feat_names] for df in day_feats])
            avg = {}
            for j, k in enumerate(daily_feat_names):
                col = arr[:, j]
                v = col[np.isfinite(col)]
                avg[k] = np.mean(v) if len(v) > 0 else np.nan
            all_daily_feats.append(avg)
            daily_valid.append(True)

            if (i+1) % 20 == 0:
                print(f"    [{i+1}/{len(ids101)}] {len(day_feats)} days with valid features", flush=True)

        daily_valid = np.array(daily_valid)

        # Build matrix
        X_daily = np.full((len(ids101), len(daily_feat_names)), np.nan)
        for i in range(len(ids101)):
            if daily_valid[i] and all_daily_feats[i]:
                for j, k in enumerate(daily_feat_names):
                    X_daily[i, j] = all_daily_feats[i].get(k, np.nan)

        np.savez(cache, X_daily=X_daily, feat_names=daily_feat_names, valid=daily_valid)
        print(f"  Cached. Valid: {daily_valid.sum()}/{len(ids101)}", flush=True)

    vmask = daily_valid.astype(bool)
    nv = vmask.sum()
    print(f"  Valid subjects: {nv}")

    Xv = X_daily[vmask].copy()
    yv = y[vmask]
    Dv = X_demo_5[vmask]
    for j in range(Xv.shape[1]):
        m = np.isnan(Xv[:, j])
        if m.any(): Xv[m, j] = np.nanmedian(Xv[:, j])

    # All features
    X = np.column_stack([Xv, Dv])
    r2, mae, rv, rho, a = best_alpha(X, yv)
    report(f"Daily-best all({Xv.shape[1]})+Demo(5) [n={nv}]", Xv.shape[1]+5, r2, mae, rv, rho, a)

    # Correlation selected
    corrs = []
    for j in range(Xv.shape[1]):
        rr, pp = spearmanr(Xv[:, j], yv)
        if np.isfinite(rr): corrs.append((j, abs(rr), rr, daily_feat_names[j]))
    corrs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 daily-segment features by |ρ|:")
    for j, absrho, rr, name in corrs[:10]:
        print(f"    {name:40s}  ρ={rr:+.3f}")

    for K in [5, 8, 10]:
        top_idx = [c[0] for c in corrs[:K]]
        X = np.column_stack([Xv[:, top_idx], Dv])
        r2, mae, rv, rho, a = best_alpha(X, yv)
        report(f"Daily-best top-{K}+Demo(5) [n={nv}]", K+5, r2, mae, rv, rho, a)


# ══════════════════════════════════════════════════════════════════
# EXP D: Hybrid combinations
# ══════════════════════════════════════════════════════════════════
def exp_d_hybrid(ids101, y, X_demo_5):
    print(f"\n{'─'*90}")
    print("EXP D: Hybrid — combine v2 Top-20 with self-ref and daily-segment")
    print(f"{'─'*90}")

    # Load v2 features
    v2_cache = OUT / 'feature_cache.npz'
    if not v2_cache.exists():
        print("  v2 cache not found — skipping"); return
    v2 = np.load(v2_cache, allow_pickle=True)

    # Load v2 correlation-selected features
    corr_df = pd.read_csv(OUT / 'feature_correlations.csv')
    v2_d = np.load(v2_cache, allow_pickle=True)

    # Rebuild v2 X_all (need the step0d function results)
    # For simplicity, load self-ref and daily caches and combine with v2 top features
    selfref_cache = OUT / 'selfref_features_cache.npz'
    daily_cache = OUT / 'daily_segment_cache.npz'

    if not selfref_cache.exists() or not daily_cache.exists():
        print("  Caches not ready — skipping"); return

    sr = np.load(selfref_cache, allow_pickle=True)
    dc = np.load(daily_cache, allow_pickle=True)

    X_sr = sr['X_selfref']
    sr_valid = sr['valid'].astype(bool)
    sr_names = list(sr['feat_names'])

    X_daily = dc['X_daily']
    daily_valid = dc['valid'].astype(bool)
    daily_names = list(dc['feat_names'])

    # Find subjects valid in both
    both_valid = sr_valid & daily_valid
    nv = both_valid.sum()
    print(f"  Subjects valid in both: {nv}")

    Xsr = X_sr[both_valid]
    Xd = X_daily[both_valid]
    yv = y[both_valid]
    Dv = X_demo_5[both_valid]

    for X in [Xsr, Xd]:
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any(): X[m, j] = np.nanmedian(X[:, j])

    # Self-ref top-K + daily top-K + Demo
    sr_corrs = []
    for j in range(Xsr.shape[1]):
        rr, pp = spearmanr(Xsr[:, j], yv)
        if np.isfinite(rr): sr_corrs.append((j, abs(rr)))
    sr_corrs.sort(key=lambda x: x[1], reverse=True)

    d_corrs = []
    for j in range(Xd.shape[1]):
        rr, pp = spearmanr(Xd[:, j], yv)
        if np.isfinite(rr): d_corrs.append((j, abs(rr)))
    d_corrs.sort(key=lambda x: x[1], reverse=True)

    for K_sr, K_d in [(5, 5), (8, 5), (5, 8), (8, 8), (10, 5)]:
        sr_idx = [c[0] for c in sr_corrs[:K_sr]]
        d_idx = [c[0] for c in d_corrs[:K_d]]
        X = np.column_stack([Xsr[:, sr_idx], Xd[:, d_idx], Dv])
        nf = K_sr + K_d + 5
        r2, mae, rv, rho, a = best_alpha(X, yv)
        report(f"SelfRef-{K_sr}+Daily-{K_d}+Demo(5) [n={nv}]", nf, r2, mae, rv, rho, a)

    # Self-ref only top features
    for K in [5, 8, 10, 12, 15]:
        idx = [c[0] for c in sr_corrs[:K]]
        X = np.column_stack([Xsr[:, idx], Dv])
        r2, mae, rv, rho, a = best_alpha(X, yv)
        report(f"SelfRef-top{K}+Demo(5) [n={nv}]", K+5, r2, mae, rv, rho, a)


# ══════════════════════════════════════════════════════════════════
# EXP E: Clinic-free longer windows
# ══════════════════════════════════════════════════════════════════
def exp_e_longer_windows(ids101, y, X_demo_5):
    print(f"\n{'─'*90}")
    print("EXP E: Clinic-free longer windows (ENMO-ranked bouts, no concatenation)")
    print(f"{'─'*90}")

    cache = OUT / 'longer_clinicfree_cache.npz'
    if cache.exists():
        print("  Loading from cache...", flush=True)
        d = np.load(cache, allow_pickle=True)
        results = d['results'].item()
        feat_names = list(d['feat_names'])
    else:
        print("  Extracting features with different target durations...", flush=True)
        results = {}
        feat_names = None

        for target_sec in [360, 600, 900]:
            all_feats = []
            for i, (_, r) in enumerate(ids101.iterrows()):
                fn = f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"
                fp = HOME_DIR / fn
                if not fp.exists():
                    all_feats.append(None); continue

                xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float64)
                bouts = detect_walking_bouts_clinicfree(xyz, FS, min_bout_sec=10, merge_gap_sec=5)

                # Rank by ENMO intensity
                scored = []
                for s, e in bouts:
                    vm = np.sqrt(xyz[s:e, 0]**2 + xyz[s:e, 1]**2 + xyz[s:e, 2]**2)
                    enmo_mean = float(np.mean(np.maximum(vm - 1.0, 0.0)))
                    scored.append((s, e, enmo_mean))
                scored.sort(key=lambda x: x[2], reverse=True)

                # Collect top bouts up to target_sec (no concatenation of distant bouts)
                # Extract features per bout, aggregate
                bout_feats = []
                total_sec = 0
                for s, e, _ in scored:
                    dur = (e - s) / FS
                    feats = extract_segment_features(xyz[s:e], FS)
                    if feats is not None:
                        bout_feats.append(feats)
                        total_sec += dur
                        if total_sec >= target_sec:
                            break

                if len(bout_feats) < 2:
                    all_feats.append(None); continue

                if feat_names is None:
                    feat_names = sorted(bout_feats[0].keys())

                arr = np.array([[bf.get(k, np.nan) for k in feat_names] for bf in bout_feats])
                agg = {}
                for j, k in enumerate(feat_names):
                    col = arr[:, j]; v = col[np.isfinite(col)]
                    agg[k] = np.median(v) if len(v) > 0 else np.nan
                all_feats.append(agg)

            results[target_sec] = all_feats
            print(f"    {target_sec}s: {sum(1 for f in all_feats if f is not None)} valid", flush=True)

        np.savez(cache, results=results, feat_names=feat_names)
        print("  Cached.", flush=True)

    # Evaluate
    for target_sec in [360, 600, 900]:
        feats_list = results[target_sec]
        valid = np.array([f is not None for f in feats_list])
        nv = valid.sum()
        if nv < 80: print(f"  {target_sec}s: only {nv} valid — skipping"); continue

        X = np.full((len(ids101), len(feat_names)), np.nan)
        for i in range(len(ids101)):
            if feats_list[i]:
                for j, k in enumerate(feat_names):
                    X[i, j] = feats_list[i].get(k, np.nan)

        Xv = X[valid]; yv = y[valid]; Dv = X_demo_5[valid]
        for j in range(Xv.shape[1]):
            m = np.isnan(Xv[:, j])
            if m.any(): Xv[m, j] = np.nanmedian(Xv[:, j])

        # Correlation select
        corrs = []
        for j in range(Xv.shape[1]):
            rr, pp = spearmanr(Xv[:, j], yv)
            if np.isfinite(rr): corrs.append((j, abs(rr)))
        corrs.sort(key=lambda x: x[1], reverse=True)

        for K in [8, 10, 15]:
            top_idx = [c[0] for c in corrs[:K]]
            Xk = np.column_stack([Xv[:, top_idx], Dv])
            r2, mae, rv, rho, a = best_alpha(Xk, yv)
            report(f"ENMO-{target_sec}s top-{K}+Demo(5) [n={nv}]", K+5, r2, mae, rv, rho, a)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    ids101, y, X_demo_5 = load_base_data()
    n = len(y)
    print(f"n={n} subjects")
    print(f"{'='*90}")
    print(f"BASELINES:")
    print(f"  Clinic-informed Gait(8)+Demo(5), α=20:  R²=0.507  MAE=174ft")
    print(f"  v2 clinic-free Top-20+Demo(5):          R²=0.441  MAE=191ft")
    print(f"  Previous clinic-free best (ENMO):       R²=0.381  MAE=199ft")

    exp_a_self_referencing(ids101, y, X_demo_5)
    exp_c_best_daily_segment(ids101, y, X_demo_5)
    exp_d_hybrid(ids101, y, X_demo_5)
    exp_e_longer_windows(ids101, y, X_demo_5)

    # Update experiment log
    with open(OUT / 'experiment_log.txt', 'a') as f:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"V3 EXPERIMENTS — {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"See stdout for full results\n")

    print(f"\n{'='*90}")
    print(f"Done in {time.time()-t0:.0f}s")
