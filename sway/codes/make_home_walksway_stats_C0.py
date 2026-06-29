#!/usr/bin/env python3
"""
Recompute home ML-sway group stats with the CORRECTED aggregation (C0):
per-subject = median over ALL sustained (>=60s) walking bouts (n=101: 38 POMS,
63 Control; no wear filter -> full sample). Replaces the earlier top-10-longest
aggregation under which nothing was significant. Uses the cached per-bout table.

Writes:
  feats/home_walksway_4features.csv                  (per-subject values)
  sway/table/home_walksway_significant_features.csv  (group-difference stats)
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
BASE = Path('/mnt/sdb/arafat/6mw')
FC = ['ml_over_enmo', 'ml_over_vt', 'ml_energy_frac', 'ml_spec_horiz_frac']
PB = pd.read_parquet(BASE / 'feats/home_sway_perbout_ge60.parquet')

# C0: per-subject median over all >=60s bouts
D = PB.groupby('key')[FC].median().reset_index()
D['cohort'] = np.where(D.key.str[0] == 'M', 'POMS', 'Control')
D = D[['key', 'cohort'] + FC]
D.to_csv(BASE / 'feats/home_walksway_4features.csv', index=False)
nP = int((D.cohort == 'POMS').sum()); nH = int((D.cohort == 'Control').sum())
print(f"n={len(D)}  POMS={nP}  Control={nH}  (median over all >=60s bouts/subject)")
print(f"bouts/subject: median {int(PB.groupby('key').size().median())}, "
      f"min {int(PB.groupby('key').size().min())}")

rng = np.random.default_rng(42)
def med_ci(a, B=2000):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    bs = np.median(a[rng.integers(0, len(a), (B, len(a)))], axis=1)
    return np.median(a), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def cohen_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sp = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1)) / (len(x)+len(y)-2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else np.nan

SRC = {'ml_over_enmo': 'orig', 'ml_over_vt': 'orig', 'ml_energy_frac': 'new', 'ml_spec_horiz_frac': 'new'}
praw = []
for c in FC:
    p = D[D.cohort == 'POMS'][c].dropna(); h = D[D.cohort == 'Control'][c].dropna()
    praw.append(mannwhitneyu(p, h, alternative='two-sided')[1])
pbh = multipletests(praw, method='fdr_bh')[1]

rows = []
for i, c in enumerate(FC):
    p = D[D.cohort == 'POMS'][c].dropna().values; h = D[D.cohort == 'Control'][c].dropna().values
    pm, plo, phi = med_ci(p); hm, hlo, hhi = med_ci(h)
    sig = '**' if pbh[i] < 0.01 else '*' if pbh[i] < 0.05 else 'ns'
    rows.append({'Feature': c, 'Source': SRC[c],
                 'POMS_med': pm, 'POMS_lo': plo, 'POMS_hi': phi,
                 'Healthy_med': hm, 'Healthy_lo': hlo, 'Healthy_hi': hhi,
                 'Cohen_d': cohen_d(p, h), 'p_raw': praw[i], 'p_BH': pbh[i], 'sig': sig})
S = pd.DataFrame(rows)
S.to_csv(BASE / 'sway/table/home_walksway_significant_features.csv', index=False)
print("\n=== Home ML-sway group differences (C0: all >=60s bouts, n=101) ===")
for _, r in S.iterrows():
    print(f"  {r.Feature:<20} POMS {r.POMS_med:.3g}[{r.POMS_lo:.3g},{r.POMS_hi:.3g}]  "
          f"Ctrl {r.Healthy_med:.3g}[{r.Healthy_lo:.3g},{r.Healthy_hi:.3g}]  "
          f"d={r.Cohen_d:+.2f}  p={r.p_raw:.4f}  p_BH={r.p_BH:.4f} {r.sig}")
