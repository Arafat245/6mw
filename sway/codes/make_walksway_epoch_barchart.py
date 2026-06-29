#!/usr/bin/env python3
"""Brenton Figure 1 analog: grouped bar chart of mean ML_Over_ENMO per 1-minute
6MWT epoch, POMS vs Healthy (n=120). Colors match walksway_significant_violins.png."""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
L=pd.DataFrame(rows).dropna()
rng=np.random.default_rng(42)
def med_ci(a,Bn=2000):
    a=np.asarray(a,float); b=np.median(a[rng.integers(0,len(a),(Bn,len(a)))],axis=1)
    return np.median(a),np.percentile(b,2.5),np.percentile(b,97.5)

med={'Healthy':[],'POMS':[]}; lo_e={'Healthy':[],'POMS':[]}; hi_e={'Healthy':[],'POMS':[]}
hi_abs={'Healthy':[],'POMS':[]}; praw=[]
for t in range(1,N+1):
    sub=L[L.t==t]; hp=sub[sub.coh=='Healthy']['v'].values; pp=sub[sub.coh=='POMS']['v'].values
    for coh,vals in [('Healthy',hp),('POMS',pp)]:
        m,clo,chi=med_ci(vals); med[coh].append(m)
        lo_e[coh].append(m-clo); hi_e[coh].append(chi-m); hi_abs[coh].append(chi)
    praw.append(mannwhitneyu(pp,hp,alternative='two-sided')[1])
pbh=multipletests(praw,method='fdr_bh')[1]                      # BH across the 6 epochs
def star(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

CLR_H='#6BAED6'; CLR_P='#FC8D62'
LBL,TIT,SUP,TICK,LEG=13,15,15,12,12
fig,ax=plt.subplots(figsize=(11,6.4))
x=np.arange(1,N+1); w=0.4
ax.bar(x-w/2,med['Healthy'],w,yerr=[lo_e['Healthy'],hi_e['Healthy']],capsize=3,color=CLR_H,
       edgecolor='white',linewidth=0.8,label='Control (all)',error_kw=dict(lw=1,ecolor='#555'))
ax.bar(x+w/2,med['POMS'],w,yerr=[lo_e['POMS'],hi_e['POMS']],capsize=3,color=CLR_P,
       edgecolor='white',linewidth=0.8,label='POMS (all)',error_kw=dict(lw=1,ecolor='#555'))
ax.set_xlabel('Time (minutes)',fontsize=LBL,fontweight='bold')
ax.set_ylabel('Median ML_Over_ENMO',fontsize=LBL,fontweight='bold')
ax.set_title('Median Sway Per 1-Minute Epoch of the 6-Minute Walk in POMS vs Controls',
             fontsize=TIT,fontweight='bold',pad=10)
ax.set_xticks(x); ax.set_xticklabels(x,fontsize=TICK)
ax.tick_params(axis='y',labelsize=TICK)
topbars=max(max(hi_abs['Healthy']),max(hi_abs['POMS']))
loy=min(min(med['Healthy']),min(med['POMS']))-0.12
ax.set_ylim(min(loy,0.85),topbars+0.18)
rng_y=topbars-loy
ax.grid(True,axis='y',alpha=0.25,linestyle='--',linewidth=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=LEG,framealpha=0.95,loc='upper right')
plt.tight_layout()
OUT=BASE/'sway'/'figures'; fp=OUT/'walksway_ML_Over_ENMO_epoch_bars.png'
plt.savefig(fp,dpi=200,bbox_inches='tight'); print("Saved",fp)
print("\nMedian ML_Over_ENMO per epoch (Healthy / POMS) + p_BH:")
for i,t in enumerate(range(1,N+1)):
    print(f"  min {t}: H={med['Healthy'][i]:.3f}  P={med['POMS'][i]:.3f}  p_raw={praw[i]:.4f}  p_BH={pbh[i]:.4f} {star(pbh[i])}")
