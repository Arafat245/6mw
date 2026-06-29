#!/usr/bin/env python3
"""
Do Brenton's actigraphy COMPLIANCE steps help the free-living ML-sway group test?
Ablate, one screening rule at a time, building on the day-structured analysis.

Caches a per-bout table (all bouts >=60s: key, day, hour, dur, 4 feats) and a
per-day wear table (worn minutes via 1-min non-wear detection). Then evaluates:

  C0  pooled            : median over ALL >=60s bouts, no day structure   (original null)
  C1  day-mean          : per-day median -> mean across ALL days (>=1 bout)
  C2  + >=3 bouts/day   : valid day = >=3 sustained bouts
  C3  + >=10h wear/day  : valid day = >=600 worn min   (Brenton valid day)
  C4  + active-only     : drop bouts in each subject's least-active 8h (sleep proxy)
  C5  + >=3 valid days  : exclude subjects with <3 valid days (Brenton wear screen)

Per condition: per-subject value, two-sided Mann-Whitney (POMS vs Control) + BH.
"""
import sys, re, warnings, pickle
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
BASE = Path('/mnt/sdb/arafat/6mw'); sys.path.insert(0, str(BASE))
from home.step2_extract_features import preprocess_segment
FS = 30; NPZ = BASE / 'home_full_recording_npz'
BOUTS = pickle.load(open(BASE / 'feats/home_walking_bouts.pkl', 'rb'))['bouts']
KEYS = sorted(BOUTS.keys())
FC = ['ml_over_enmo', 'ml_over_vt', 'ml_energy_frac', 'ml_spec_horiz_frac']
BOUT_CACHE = BASE / 'feats/home_sway_perbout_ge60.parquet'
WEAR_CACHE = BASE / 'feats/home_perday_wear.parquet'

def bandpow(x, lo, hi):
    x = x - np.mean(x); f, P = welch(x, fs=FS, nperseg=min(1024, len(x)))
    return float(np.sum(P[(f >= lo) & (f <= hi)]))
def feats(xyz):
    if len(xyz) < int(10*FS): return None
    try: a, _b, enmo, _v = preprocess_segment(xyz, FS)
    except Exception: return None
    ap, ml, vt = a[:,0], a[:,1], a[:,2]
    mlr = np.sqrt(np.mean(ml**2)); vtr = np.sqrt(np.mean(vt**2)); em = np.mean(enmo)
    vAP, vML, vVT = np.var(ap), np.var(ml), np.var(vt); mlP, apP = bandpow(ml,.3,10), bandpow(ap,.3,10)
    return [mlr/em if em>1e-9 else np.nan, mlr/vtr if vtr>1e-9 else np.nan,
            vML/(vAP+vML+vVT+1e-12), mlP/(mlP+apP+1e-12)]

# ── build / load caches ──
if BOUT_CACHE.exists() and WEAR_CACHE.exists():
    PB = pd.read_parquet(BOUT_CACHE); WD = pd.read_parquet(WEAR_CACHE)
    print(f"Loaded cache: {len(PB)} bouts, {len(WD)} subject-days")
else:
    brows, wrows = [], []
    for ki, key in enumerate(KEYS):
        z = np.load(NPZ / f'{key}.npz'); xyz, ts = z['xyz'], z['timestamps']
        # per-minute non-wear detection -> worn minutes per day
        n = len(xyz); spm = 60*FS; vm = np.sqrt((xyz**2).sum(1))
        for mi in range(0, n - spm, spm):
            seg = vm[mi:mi+spm]; worn = np.std(seg) >= 0.013
            wrows.append({'key': key, 'day': int(ts[mi]//86400),
                          'hour': int((ts[mi] % 86400)//3600), 'worn': int(worn)})
        for s, e in BOUTS[key]:
            if (e-s)/FS < 60: continue
            f = feats(xyz[s:e].astype(np.float64))
            if not f: continue
            brows.append({'key': key, 'day': int(ts[s]//86400),
                          'hour': int((ts[s] % 86400)//3600), 'dur': (e-s)/FS,
                          **{c: f[j] for j, c in enumerate(FC)}})
        if (ki+1) % 25 == 0: print(f"  cache [{ki+1}/101]", flush=True)
    PB = pd.DataFrame(brows); PB.to_parquet(BOUT_CACHE)
    Wm = pd.DataFrame(wrows)
    WD = Wm.groupby(['key','day'])['worn'].sum().reset_index().rename(columns={'worn':'wear_min'})
    WD.to_parquet(WEAR_CACHE)
    # hourly activity profile for sleep proxy
    print(f"Cached {len(PB)} bouts, {len(WD)} subject-days")

# hourly worn-fraction -> least-active 8h window per subject (sleep proxy)
Wm = pd.read_parquet(WEAR_CACHE)  # placeholder; recompute hourly from bout hours instead
# sleep proxy: per subject, the 8 consecutive hours with fewest walking bouts
sleep_hours = {}
for key, g in PB.groupby('key'):
    cnt = np.zeros(24)
    for h in g['hour']: cnt[h] += 1
    best_h, best_v = 0, 1e9
    for start in range(24):
        idx = [(start+i) % 24 for i in range(8)]; v = cnt[idx].sum()
        if v < best_v: best_v, best_h = v, start
    sleep_hours[key] = set((best_h+i) % 24 for i in range(8))

demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
def key_id(s):
    m = re.match(r'\s*([CM])-?(\d+)', str(s), re.I)
    return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None
demo['key'] = demo['ID'].apply(key_id); DEM = demo.set_index('key')
WEARMAP = {(r.key, r.day): r.wear_min for r in WD.itertuples()}

def per_subject(mode):
    """Return DataFrame [key, MS, 4 feats] per condition."""
    out = []
    for key, g in PB.groupby('key'):
        g = g.copy()
        if mode in ('C4',): g = g[[h not in sleep_hours[key] for h in g['hour']]]
        if mode == 'C0':
            vals = g[FC].median().values
            out.append((key, vals)); continue
        # day-level medians
        dayvals = {}
        for day, dg in g.groupby('day'):
            if mode in ('C2','C4','C5') and len(dg) < 3: continue
            if mode in ('C3','C4','C5') and WEARMAP.get((key,int(day)),0) < 600: continue
            dayvals[day] = dg[FC].median().values
        if mode == 'C5' and len(dayvals) < 3: continue
        if not dayvals: continue
        out.append((key, np.nanmean(np.array(list(dayvals.values())), axis=0)))
    df = pd.DataFrame({'key':[k for k,_ in out]})
    for j,c in enumerate(FC): df[c] = [v[j] for _,v in out]
    df['MS'] = (df.key.str[0]=='M').astype(int)
    return df

def evaluate(df):
    pr=[]
    for c in FC:
        pr.append(mannwhitneyu(df[df.MS==1][c].dropna(), df[df.MS==0][c].dropna(),
                               alternative='two-sided')[1])
    pbh = multipletests(pr, method='fdr_bh')[1]
    return pr, pbh

CONDS = [('C0','pooled (no day)'),('C1','day-mean, all days'),('C2','+>=3 bouts/day'),
         ('C3','+>=10h wear/day'),('C4','+active-only (8h sleep removed)'),
         ('C5','+>=3 valid days')]
print(f"\n{'cond':<5}{'n':>4}{'nP':>4}{'nH':>4}   " + "  ".join(f"{c.split('_',1)[1][:9]:>9}" for c in FC) + "   (p_BH; *<.05 **<.01)")
for code, desc in CONDS:
    df = per_subject(code); pr, pbh = evaluate(df)
    nP=int((df.MS==1).sum()); nH=int((df.MS==0).sum())
    cells=[]
    for p in pbh:
        s='**' if p<0.01 else '*' if p<0.05 else ''
        cells.append(f"{p:>7.3f}{s:<2}")
    print(f"{code:<5}{len(df):>4}{nP:>4}{nH:>4}   " + "  ".join(cells) + f"   {desc}")
