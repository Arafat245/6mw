#!/usr/bin/env python3
"""Scatter of the 3 significant WalkSway features vs 6MWD (meters), per-feature
subplots, colored by cohort, with linear trend + Spearman rho/p (n=120)."""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios
from clinic.extract_walking_sway import extract_walking_sway
PRE=BASE/'csv_preprocessed2'; FS=30.0; FT2M=0.3048
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_(?P<yr>\d+)_(?P<d>\d+)\.csv$',re.I)
def bandpow(x,lo,hi):
    f,P=welch(x-np.mean(x),fs=FS,nperseg=min(1024,len(x))); b=(f>=lo)&(f<=hi); return np.sum(P[b])

ids,gait_rows,ws_rows,spec=[],[],[],[]
for p in sorted(PRE.glob('*.csv')):
    m=fre.match(p.name)
    if not m: continue
    df=pd.read_csv(p); ap,ml,vt=df['AP'].values,df['ML'].values,df['VT'].values
    ids.append({'cohort':m.group('c').upper(),'subj_id':int(m.group('id')),'sixmwd':int(m.group('d'))})
    gait_rows.append(extract_gait10(df)); ws_rows.append(extract_walking_sway(ap,ml,vt))
    mlP=bandpow(ml,0.3,10); apP=bandpow(ap,0.3,10); spec.append(mlP/(mlP+apP+1e-12))
ids=pd.DataFrame(ids)
gm=add_sway_ratios(pd.concat([ids.reset_index(drop=True),pd.DataFrame(gait_rows)],axis=1)
                   .merge(compute_vt_rms(PRE),on=['cohort','subj_id','sixmwd'],how='left'))
D=pd.DataFrame({'ML_Over_ENMO':gm['ml_over_enmo'].values,'ML_Over_VT':gm['ml_over_vt'].values,
                'ML_Spec_Frac':spec,'SixMWD_m':ids['sixmwd'].values*FT2M,
                'cohort':np.where(ids['cohort'].values=='M','POMS','Healthy')})

CLR_H='#6BAED6'; CLR_P='#FC8D62'
LBL,TIT,SUP,TICK=13,14,15,12
feats=[('ML_Over_ENMO','ML RMS / ENMO'),('ML_Over_VT','ML RMS / VT RMS'),
       ('ML_Spec_Frac',r'P$_{ML}$ / (P$_{ML}$+P$_{AP}$)')]
fig,axes=plt.subplots(1,3,figsize=(18,6),sharey=True)
fig.suptitle('Mediolateral-Sway Features vs 6-Minute Walk Distance  (n=120: 50 POMS, 70 Healthy)',
             fontsize=SUP,fontweight='bold',y=0.99)
for ax,(col,xlab) in zip(axes,feats):
    for coh,c in [('Healthy',CLR_H),('POMS',CLR_P)]:
        s=D[D.cohort==coh]; ax.scatter(s[col],s['SixMWD_m'],c=c,s=42,alpha=0.85,
                                       edgecolor='white',linewidth=0.6,label=coh)
    # overall linear trend
    x=D[col].values; y=D['SixMWD_m'].values; b,a=np.polyfit(x,y,1)
    xs=np.linspace(x.min(),x.max(),100); ax.plot(xs,a+b*xs,'k--',linewidth=1.6,alpha=0.8)
    rho,pv=spearmanr(x,y)
    ax.set_title(col,fontsize=TIT,fontweight='bold',family='monospace',pad=8)
    ax.set_xlabel(xlab,fontsize=LBL,fontweight='bold')
    ax.tick_params(labelsize=TICK)
    ax.grid(True,alpha=0.25,linestyle='--',linewidth=0.5); ax.set_axisbelow(True)
    ptxt='p < 0.001' if pv<0.001 else f'p = {pv:.3f}'
    ax.text(0.04,0.05,f'Spearman $\\rho$ = {rho:+.3f}\n{ptxt}',transform=ax.transAxes,
            ha='left',va='bottom',fontsize=LBL,fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35',fc='white',ec='gray',alpha=0.85))
axes[0].set_ylabel('6MWD (m)',fontsize=LBL,fontweight='bold')
axes[2].legend(fontsize=LBL,loc='upper right',framealpha=0.9)
plt.tight_layout(rect=[0,0,1,0.96])
OUT=BASE/'sway'/'figures'; fp=OUT/'walksway_6mwd_scatter.png'
plt.savefig(fp,dpi=200,bbox_inches='tight'); print("Saved",fp)
for col,_ in feats:
    rho,pv=spearmanr(D[col],D['SixMWD_m']); print(f"{col:<16} rho={rho:+.3f}  p={pv:.4g}")
