#!/usr/bin/env python3
"""
Faithful Brenton-2022 CONTINUOUS-ACCELEROMETRY procedure applied to the 4 ML-sway
features (free-living home, n=101). Mirrors Brenton's actigraphy analysis:
  - require longest valid consecutive wear >=3 days,
  - analyse per-DAY measures (valid day = >=3 sustained walking bouts as proxy),
  - day-level values are REPEATED MEASURES -> LME with day nested in participant,
    adjusting for age, sex, BMI (age & BMI mean-centered).
MS fixed-effect p = POMS-vs-Control group difference. BH-FDR across the 4 features.

Per day: median sway over that day's up-to-8 longest bouts >=60s.
"""
import sys, re, warnings, pickle
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
BASE = Path('/mnt/sdb/arafat/6mw'); sys.path.insert(0, str(BASE))
from home.step2_extract_features import preprocess_segment
FS = 30; NPZ = BASE / 'home_full_recording_npz'
BOUTS = pickle.load(open(BASE / 'feats/home_walking_bouts.pkl', 'rb'))['bouts']
KEYS = sorted(BOUTS.keys())
FEATCOLS = ['ml_over_enmo', 'ml_over_vt', 'ml_energy_frac', 'ml_spec_horiz_frac']
MIN_BOUTS_DAY, MAX_BOUTS_DAY, MIN_DAYS = 3, 8, 3

# demographics
demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
def key_id(s):
    m = re.match(r'\s*([CM])-?(\d+)', str(s), re.I)
    return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None
demo['key'] = demo['ID'].apply(key_id); DEM = demo.set_index('key')

def bandpow(x, lo, hi):
    x = x - np.mean(x); f, P = welch(x, fs=FS, nperseg=min(1024, len(x)))
    return float(np.sum(P[(f >= lo) & (f <= hi)]))
def feats(xyz):
    if len(xyz) < int(10*FS): return None
    try: a, _b, enmo, _v = preprocess_segment(xyz, FS)
    except Exception: return None
    ap, ml, vt = a[:,0], a[:,1], a[:,2]
    mlr = np.sqrt(np.mean(ml**2)); vtr = np.sqrt(np.mean(vt**2)); em = np.mean(enmo)
    vAP, vML, vVT = np.var(ap), np.var(ml), np.var(vt)
    mlP, apP = bandpow(ml,0.3,10), bandpow(ap,0.3,10)
    return [mlr/em if em>1e-9 else np.nan, mlr/vtr if vtr>1e-9 else np.nan,
            vML/(vAP+vML+vVT+1e-12), mlP/(mlP+apP+1e-12)]

rows = []
for ki, key in enumerate(KEYS):
    if key not in DEM.index: continue
    z = np.load(NPZ / f'{key}.npz'); xyz, ts = z['xyz'], z['timestamps']
    byday = {}
    for s, e in BOUTS[key]:
        if (e-s)/FS < 60: continue
        byday.setdefault(int(ts[s]//86400), []).append((e-s, s, e))
    valid_days = []
    for day, bl in byday.items():
        if len(bl) < MIN_BOUTS_DAY: continue
        bl.sort(reverse=True)
        fs_ = [feats(xyz[s:e].astype(np.float64)) for _, s, e in bl[:MAX_BOUTS_DAY]]
        fs_ = [f for f in fs_ if f]
        if not fs_: continue
        valid_days.append(np.nanmedian(np.array(fs_), axis=0))
    if len(valid_days) < MIN_DAYS: continue
    for di, dv in enumerate(valid_days):
        rows.append({'key': key, 'day': di, 'MS': 1 if key[0]=='M' else 0,
                     'Female': 1 if DEM.loc[key,'Sex']==2 else 0,
                     'Age': DEM.loc[key,'Age'], 'BMI': DEM.loc[key,'BMI'],
                     **{c: dv[j] for j, c in enumerate(FEATCOLS)}})
    if (ki+1) % 25 == 0: print(f"  [{ki+1}/101]", flush=True)

L = pd.DataFrame(rows)
L['Age_c'] = L['Age']-L['Age'].mean(); L['BMI_c'] = L['BMI']-L['BMI'].mean()
nsub = L.key.nunique(); nP = L[L.MS==1].key.nunique(); nH = L[L.MS==0].key.nunique()
print(f"\nDay-level rows={len(L)}  subjects={nsub} (POMS={nP}, Control={nH})  "
      f"median valid days/subj={int(L.groupby('key').size().median())}")

# ---- Brenton LME: feature ~ MS + Female + Age_c + BMI_c, (1|subject), days repeated ----
praw_ms = []; res = []
for c in FEATCOLS:
    m = smf.mixedlm(f"{c} ~ MS + Female + Age_c + BMI_c", L, groups=L['key'])
    f = None
    for meth in ['lbfgs','powell','cg','nm','bfgs']:
        try:
            ff = m.fit(method=meth)
            if np.isfinite(ff.pvalues.get('MS', np.nan)): f = ff; break
            f = f or ff
        except Exception: continue
    est, p = f.params['MS'], f.pvalues['MS']; praw_ms.append(p)
    res.append((c, est, p))
pbh = multipletests(praw_ms, method='fdr_bh')[1]
print("\n=== Brenton continuous-accelerometry LME (MS effect, day-level repeats) ===")
print(f"{'Feature':<20}{'MS est':>10}{'p_raw':>10}{'p_BH':>10}")
for (c, est, p), pb in zip(res, pbh):
    flag = '  <== BH-SIG' if pb < 0.05 else ('  (raw<0.05)' if p < 0.05 else '')
    print(f"  {c:<18}{est:>10.4f}{p:>10.4f}{pb:>10.4f}{flag}")

# ---- companion: per-subject mean of day-medians, Mann-Whitney (unadjusted) ----
PS = L.groupby(['key','MS'])[FEATCOLS].mean().reset_index()
print("\n--- per-subject (mean of valid-day medians), unadjusted Mann-Whitney ---")
pr = []
for c in FEATCOLS:
    pr.append(mannwhitneyu(PS[PS.MS==1][c], PS[PS.MS==0][c], alternative='two-sided')[1])
pbh2 = multipletests(pr, method='fdr_bh')[1]
for c, p, pb in zip(FEATCOLS, pr, pbh2):
    print(f"  {c:<20} p_raw={p:.4f}  p_BH={pb:.4f}{'  <== BH-SIG' if pb<0.05 else ''}")
