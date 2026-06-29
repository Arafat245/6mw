#!/usr/bin/env python3
"""
Does a Brenton-style single-sustained-walk selection recover the clinic ML-sway
group difference in FREE-LIVING data? Compare bout-selection variants for the 4
ML-sway features (axis-aligned, n=101: 38 POMS, 63 Healthy).

Variants:
  top10med   : median over Top-10 longest bouts >=60s   (current home baseline)
  top3med    : median over Top-3 longest bouts
  longest    : single longest sustained bout (full)               <- most clinic-like
  long_cap6  : single longest bout, first 6 min (Brenton 6MWT length, capped)
  long_mid6  : central 6 min of the single longest bout
Plus a Brenton epoch analysis: split each subject's longest >=6min walk into 6
~1-min epochs and run per-epoch Mann-Whitney (POMS vs Healthy) + BH.

Stat per variant: two-sided Mann-Whitney U + Benjamini-Hochberg across the 4 feats.
"""
import sys, warnings, pickle
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
BASE = Path('/mnt/sdb/arafat/6mw'); sys.path.insert(0, str(BASE))
from home.step2_extract_features import preprocess_segment
FS = 30
NPZ = BASE / 'home_full_recording_npz'
BOUTS = pickle.load(open(BASE / 'feats/home_walking_bouts.pkl', 'rb'))['bouts']
KEYS = sorted(BOUTS.keys())
FEATCOLS = ['ml_over_enmo', 'ml_over_vt', 'ml_energy_frac', 'ml_spec_horiz_frac']

def bandpow(x, lo, hi):
    x = x - np.mean(x); f, P = welch(x, fs=FS, nperseg=min(1024, len(x)))
    return float(np.sum(P[(f >= lo) & (f <= hi)]))

def feats(xyz):
    if len(xyz) < int(10 * FS): return None
    try: apmlvt, _bp, enmo, _vm = preprocess_segment(xyz, FS)
    except Exception: return None
    ap, ml, vt = apmlvt[:, 0], apmlvt[:, 1], apmlvt[:, 2]
    ml_rms = np.sqrt(np.mean(ml**2)); vt_rms = np.sqrt(np.mean(vt**2)); em = np.mean(enmo)
    vAP, vML, vVT = np.var(ap), np.var(ml), np.var(vt)
    mlP, apP = bandpow(ml, 0.3, 10), bandpow(ap, 0.3, 10)
    return {'ml_over_enmo': ml_rms/em if em > 1e-9 else np.nan,
            'ml_over_vt': ml_rms/vt_rms if vt_rms > 1e-9 else np.nan,
            'ml_energy_frac': vML/(vAP+vML+vVT+1e-12),
            'ml_spec_horiz_frac': mlP/(mlP+apP+1e-12)}

def med(dicts):
    out = {}
    for c in FEATCOLS:
        v = np.array([d[c] for d in dicts if d], float); v = v[np.isfinite(v)]
        out[c] = np.median(v) if len(v) else np.nan
    return out

CAP6 = 360 * FS
variants = {k: [] for k in ['top10med', 'top3med', 'longest', 'long_cap6', 'long_mid6']}
epoch_rows = []   # for Brenton epoch analysis (longest >=6min walk, 6 epochs)
for ki, key in enumerate(KEYS):
    xyz = np.load(NPZ / f'{key}.npz')['xyz']
    coh = 'POMS' if key[0] == 'M' else 'Healthy'
    cand = sorted([se for se in BOUTS[key] if (se[1]-se[0])/FS >= 60],
                  key=lambda se: se[1]-se[0], reverse=True)
    seg = lambda se: xyz[se[0]:se[1]].astype(np.float64)
    top10 = [feats(seg(se)) for se in cand[:10]]
    top3 = [feats(seg(se)) for se in cand[:3]]
    s, e = cand[0]; L = seg((s, e))
    f_long = feats(L)
    f_cap6 = feats(L[:CAP6])
    mid = max(0, (len(L) - CAP6) // 2); f_mid6 = feats(L[mid:mid + CAP6])
    for name, val in [('top10med', med(top10)), ('top3med', med(top3)),
                      ('longest', f_long), ('long_cap6', f_cap6), ('long_mid6', f_mid6)]:
        row = {'key': key, 'cohort': coh}; row.update(val if val else {c: np.nan for c in FEATCOLS})
        variants[name].append(row)
    # Brenton epochs on longest walk if >=6min
    if (e - s) / FS >= 360:
        bnd = np.linspace(0, len(L), 7).astype(int)
        for ep in range(6):
            fe = feats(L[bnd[ep]:bnd[ep+1]])
            if fe: epoch_rows.append({'key': key, 'cohort': coh, 'epoch': ep+1, **fe})
    if (ki+1) % 25 == 0: print(f"  [{ki+1}/101]", flush=True)

def report(name, rows):
    D = pd.DataFrame(rows)
    praw, ds = [], []
    for c in FEATCOLS:
        p = D[D.cohort=='POMS'][c].dropna(); h = D[D.cohort=='Healthy'][c].dropna()
        praw.append(mannwhitneyu(p, h, alternative='two-sided')[1])
        sp = np.sqrt(((len(p)-1)*p.var()+(len(h)-1)*h.var())/(len(p)+len(h)-2))
        ds.append((p.mean()-h.mean())/sp if sp>0 else np.nan)
    pbh = multipletests(praw, method='fdr_bh')[1]
    print(f"\n--- {name}  (n={len(D)}) ---")
    for c, d, pr, pb in zip(FEATCOLS, ds, praw, pbh):
        flag = '  <== BH-SIG' if pb < 0.05 else ('  (raw<0.05)' if pr < 0.05 else '')
        print(f"  {c:<20} d={d:+.2f}  p_raw={pr:.4f}  p_BH={pb:.4f}{flag}")
    return pbh

print("="*60)
for name in ['top10med', 'top3med', 'longest', 'long_cap6', 'long_mid6']:
    report(name, variants[name])

# Brenton per-epoch analysis on the single longest >=6min walk
print("\n" + "="*60)
E = pd.DataFrame(epoch_rows)
nP = E[E.cohort=='POMS'].key.nunique(); nH = E[E.cohort=='Healthy'].key.nunique()
print(f"Brenton-style per-epoch (longest >=6min walk; POMS={nP}, Healthy={nH} subjects)")
for c in FEATCOLS:
    praw = []
    for ep in range(1, 7):
        sub = E[E.epoch==ep]
        praw.append(mannwhitneyu(sub[sub.cohort=='POMS'][c], sub[sub.cohort=='Healthy'][c],
                                 alternative='two-sided')[1])
    pbh = multipletests(praw, method='fdr_bh')[1]
    sig = ''.join('*' if p<0.05 else '.' for p in pbh)
    best = min(pbh)
    print(f"  {c:<20} epoch p_BH min={best:.4f}  [{sig}]")
