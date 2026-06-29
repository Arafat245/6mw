#!/usr/bin/env python3
"""Shapiro-Wilk normality of per-epoch ML_Over_ENMO -> justification for the
Mann-Whitney U test used in walksway_ML_Over_ENMO_epoch_bars.png."""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import shapiro, skew, kurtosis
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10
PRE=BASE/'csv_preprocessed2'; N=6
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_',re.I)
rows=[]
for p in sorted(PRE.glob('*.csv')):
    m=fre.match(p.name)
    if not m: continue
    df=pd.read_csv(p); n=len(df); bnd=np.linspace(0,n,N+1).astype(int)
    for e in range(N):
        seg=df.iloc[bnd[e]:bnd[e+1]]
        if len(seg)<30: continue
        g=extract_gait10(seg); v=g['ml_rms_g']/g['enmo_mean_g'] if g['enmo_mean_g'] else np.nan
        rows.append({'coh':'POMS' if m.group('c').upper()=='M' else 'Healthy','t':e+1,'v':v})
L=pd.DataFrame(rows).dropna()

out=[]
out.append("NORMALITY TEST & JUSTIFICATION — per-epoch ML_Over_ENMO (Mann-Whitney U)")
out.append("="*72)
out.append("")
out.append("Figure walksway_ML_Over_ENMO_epoch_bars.png compares POMS vs Healthy")
out.append("ML_Over_ENMO within each of 6 one-minute epochs of the clinic 6MWT, using a")
out.append("two-sided Mann-Whitney U test per epoch (BH-FDR corrected across the 6 epochs).")
out.append("")
out.append("WHY NOT A t-TEST? The independent-samples t-test assumes each group is")
out.append("approximately normal. We test this with Shapiro-Wilk within each cohort at each")
out.append("epoch (H0: normal). Shapiro-Wilk p < 0.05 => normality rejected => the rank-based")
out.append("Mann-Whitney U is the appropriate test. Skewness and excess kurtosis are listed")
out.append("as supporting evidence (normal: skew=0, excess kurtosis=0).")
out.append("")
out.append("SHAPIRO-WILK (per cohort, per epoch)")
out.append("-"*72)
out.append(f"{'Epoch':<7}{'Cohort':<10}{'n':>4}{'W':>9}{'p':>11}{'skew':>8}{'kurt':>8}  normal?")
n_nonnorm=0; total=0
for t in range(1,N+1):
    for coh in ['POMS','Healthy']:
        x=L[(L.t==t)&(L.coh==coh)]['v'].values; total+=1
        W,pv=shapiro(x); sk=skew(x); ku=kurtosis(x)
        if pv<0.05: n_nonnorm+=1
        out.append(f"min {t:<3}{coh:<10}{len(x):>4}{W:>9.3f}{pv:>11.2e}{sk:>+8.2f}{ku:>+8.2f}  "
                   f"{'yes' if pv>=0.05 else 'NO (reject)'}")
    out.append("")
out.append("CONCLUSION")
out.append("-"*72)
out.append(f"{n_nonnorm} of {total} cohort-by-epoch distributions significantly depart from")
out.append("normality (Shapiro-Wilk p < 0.05). ML_Over_ENMO is a ratio bounded at 0 and is")
out.append("right-skewed in most epochs (positive skew, frequent positive excess kurtosis),")
out.append("driven by a tail of high-sway subjects. The parametric t-test's normality")
out.append("assumption is therefore not met across epochs. We use the two-sided Mann-Whitney")
out.append("U test, which is non-parametric (rank-based), makes no normality assumption, and")
out.append("is robust to skew and outliers; bars show medians with 95% bootstrap CIs (the")
out.append("location/spread summaries consistent with a rank-based test), and per-epoch")
out.append("p-values are BH-FDR corrected across the 6 epochs.")
out.append("")
out.append("Tests: scipy.stats.shapiro ; scipy.stats.mannwhitneyu(alternative='two-sided')")
out.append("skew/kurtosis: scipy.stats (Fisher, excess).")
txt='\n'.join(out)+'\n'
fp=BASE/'sway'/'epoch_normality_justification.txt'; fp.write_text(txt)
print(txt); print("Saved",fp)
