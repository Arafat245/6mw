#!/usr/bin/env python3
"""
Continuous (free-living / home) analog of the clinic WalkSway definitions workbook.

Recomputes the four significant mediolateral-sway features on the HOME free-living
recordings (n=101: 38 POMS, 63 Healthy; M22/M44 excluded), axis-aligned exactly
like the clinic pipeline (step2 preprocess_segment -> true AP/ML/VT), aggregated
over each subject's Top-10 longest clean walking bouts >=60 s (per-bout value, then
median across bouts -- the established home WalkSway convention).

Features (identical definitions to clinic):
  ml_over_enmo       = RMS(ML) / ENMO
  ml_over_vt         = RMS(ML) / RMS(VT)
  ml_energy_frac     = var(ML) / (var(AP)+var(ML)+var(VT))            [new, this work]
  ml_spec_horiz_frac = P_ML / (P_ML+P_AP), band 0.3-10 Hz (Welch PSD) [new, this work]

Stats: two-sided Mann-Whitney U (POMS vs Healthy), Benjamini-Hochberg FDR across the
4 features, Cohen's d, and 95% bootstrap CI of the median (B=2000, seed=42).

Outputs:
  feats/home_walksway_4features.csv                      (per-subject feature values)
  sway/table/home_walksway_significant_features.csv      (group-difference stats)
  sway/table/walksway_significant_feature_definitions_home.xlsx  (3-tab workbook)
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
BASE = Path('/mnt/sdb/arafat/6mw'); sys.path.insert(0, str(BASE))
from home.step2_extract_features import preprocess_segment

FS = 30
NPZ_DIR = BASE / 'home_full_recording_npz'
import pickle
PKL = pickle.load(open(BASE / 'feats/home_walking_bouts.pkl', 'rb'))
BOUTS = PKL['bouts']
KEYS = sorted(BOUTS.keys())            # 101 subjects, M22/M44 already excluded
N_TOP, MIN_SEC = 10, 60

def bandpow(x, lo, hi):
    x = x - np.mean(x)
    f, P = welch(x, fs=FS, nperseg=min(1024, len(x)))
    b = (f >= lo) & (f <= hi)
    return float(np.sum(P[b]))

def feats_for_bout(xyz):
    """4 axis-aligned sway features for one bout, mirroring the clinic definitions."""
    apmlvt, _bp, enmo, _vm = preprocess_segment(xyz, FS)   # gravity-removed, yaw-aligned
    ap, ml, vt = apmlvt[:, 0], apmlvt[:, 1], apmlvt[:, 2]
    ml_rms = float(np.sqrt(np.mean(ml**2))); vt_rms = float(np.sqrt(np.mean(vt**2)))
    enmo_mean = float(np.mean(enmo))
    vAP, vML, vVT = np.var(ap), np.var(ml), np.var(vt)
    mlP, apP = bandpow(ml, 0.3, 10), bandpow(ap, 0.3, 10)
    return {
        'ml_over_enmo':       ml_rms / enmo_mean if enmo_mean > 1e-9 else np.nan,
        'ml_over_vt':         ml_rms / vt_rms if vt_rms > 1e-9 else np.nan,
        'ml_energy_frac':     vML / (vAP + vML + vVT + 1e-12),
        'ml_spec_horiz_frac': mlP / (mlP + apP + 1e-12),
    }

# ── per-subject extraction (Top-10 longest bouts >=60s, median over bouts) ──
FEATCOLS = ['ml_over_enmo', 'ml_over_vt', 'ml_energy_frac', 'ml_spec_horiz_frac']
rows = []
for ki, key in enumerate(KEYS):
    z = np.load(NPZ_DIR / f'{key}.npz'); xyz_all = z['xyz']
    cand = sorted(BOUTS[key], key=lambda se: se[1] - se[0], reverse=True)
    cand = [se for se in cand if (se[1] - se[0]) / FS >= MIN_SEC][:N_TOP]
    per = []
    for s, e in cand:
        seg = xyz_all[s:e].astype(np.float64)
        if len(seg) < int(10 * FS): continue
        try: per.append(feats_for_bout(seg))
        except Exception: continue
    agg = {'key': key, 'cohort': 'POMS' if key[0] == 'M' else 'Healthy', 'n_bouts': len(per)}
    for c in FEATCOLS:
        vals = np.array([p[c] for p in per], float); vals = vals[np.isfinite(vals)]
        agg[c] = float(np.median(vals)) if len(vals) else np.nan
    rows.append(agg)
    if (ki + 1) % 20 == 0: print(f"  [{ki+1}/{len(KEYS)}] {key}", flush=True)

D = pd.DataFrame(rows)
D.to_csv(BASE / 'feats/home_walksway_4features.csv', index=False)
nP = int((D.cohort == 'POMS').sum()); nH = int((D.cohort == 'Healthy').sum())
print(f"\nHome n={len(D)}  POMS={nP}  Healthy={nH}  (median bouts/subj={int(D.n_bouts.median())})")

# ── group-difference stats ──
rng = np.random.default_rng(42)
def med_ci(a, B=2000):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    bs = np.median(a[rng.integers(0, len(a), (B, len(a)))], axis=1)
    return np.median(a), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def cohen_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else np.nan

SRC = {'ml_over_enmo': 'orig', 'ml_over_vt': 'orig', 'ml_energy_frac': 'new', 'ml_spec_horiz_frac': 'new'}
praw = []
for c in FEATCOLS:
    p = D[D.cohort == 'POMS'][c].dropna(); h = D[D.cohort == 'Healthy'][c].dropna()
    praw.append(mannwhitneyu(p, h, alternative='two-sided')[1])
pbh = multipletests(praw, method='fdr_bh')[1]

stat_rows = []
for i, c in enumerate(FEATCOLS):
    p = D[D.cohort == 'POMS'][c].dropna().values; h = D[D.cohort == 'Healthy'][c].dropna().values
    pm, plo, phi = med_ci(p); hm, hlo, hhi = med_ci(h)
    sig = '**' if pbh[i] < 0.01 else '*' if pbh[i] < 0.05 else 'ns'
    stat_rows.append({'Feature': c, 'Source': SRC[c],
                      'POMS_med': pm, 'POMS_lo': plo, 'POMS_hi': phi,
                      'Healthy_med': hm, 'Healthy_lo': hlo, 'Healthy_hi': hhi,
                      'Cohen_d': cohen_d(p, h), 'p_raw': praw[i], 'p_BH': pbh[i], 'sig': sig})
S = pd.DataFrame(stat_rows)
S.to_csv(BASE / 'sway/table/home_walksway_significant_features.csv', index=False)
print("\n=== Home WalkSway group differences (POMS vs Healthy, n=101) ===")
for _, r in S.iterrows():
    print(f"  {r.Feature:<20} POMS {r.POMS_med:.3g}[{r.POMS_lo:.3g},{r.POMS_hi:.3g}]  "
          f"HC {r.Healthy_med:.3g}[{r.Healthy_lo:.3g},{r.Healthy_hi:.3g}]  "
          f"d={r.Cohen_d:+.2f}  p={r.p_raw:.4f}  p_BH={r.p_BH:.4f} {r.sig}")
