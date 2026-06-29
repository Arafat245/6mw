#!/usr/bin/env python3
"""
Brenton-2022 Table 5 analog (genuine LME) for the 4 significant WalkSway features.

Brenton Table 5: each actigraphy outcome modeled by an LME with the day-to-day
effect as a random effect nested within participant, fixed effects MS + Female +
BMI + Age (BMI/Age mean-centered). Here the repeated measure is the 6 within-walk
epochs of the clinic 6MWT, so:

    feature ~ MS + Female + BMI_c + Age_c ,  random intercept (1 | subject)

over 720 epoch-level rows (120 subjects x 6 epochs). MS = POMS-vs-control,
Female = female-vs-male (Sex==2). n=120 (50 POMS, 70 Healthy).
"""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10
PRE=BASE/'csv_preprocessed2'; FS=30.0; N=6
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_',re.I)
def bandpow(x,lo,hi):
    f,Pp=welch(x-np.mean(x),fs=FS,nperseg=min(512,max(8,len(x)))); b=(f>=lo)&(f<=hi); return np.sum(Pp[b])

demo=pd.read_excel(BASE/'SwayDemographics.xlsx')
def key_id(s):
    m=re.match(r'\s*([CM])-?(\d+)',str(s),re.I); return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None
demo['key']=demo['ID'].apply(key_id); dem=demo.set_index('key')

rows=[]
for p in sorted(PRE.glob('*.csv')):
    m=fre.match(p.name)
    if not m: continue
    key=f"{m.group('c').upper()}{int(m.group('id')):02d}"
    if key not in dem.index: continue
    df=pd.read_csv(p); n=len(df); bnd=np.linspace(0,n,N+1).astype(int)
    for e in range(N):
        seg=df.iloc[bnd[e]:bnd[e+1]]
        if len(seg)<30: continue
        ap,ml,vt=seg['AP'].values,seg['ML'].values,seg['VT'].values
        g=extract_gait10(seg)
        ml_rms=g['ml_rms_g']; enmo=g['enmo_mean_g']; vt_rms=float(np.sqrt(np.mean(vt**2)))
        vAP,vML,vVT=np.var(ap),np.var(ml),np.var(vt); mlP=bandpow(ml,0.3,10); apP=bandpow(ap,0.3,10)
        rows.append({'key':key,'cohort':m.group('c').upper(),
            'ML_Over_ENMO':ml_rms/enmo if enmo else np.nan,
            'ML_Over_VT':ml_rms/vt_rms if vt_rms else np.nan,
            'ML_Energy_Frac':vML/(vAP+vML+vVT+1e-12),
            'ML_Spec_Frac':mlP/(mlP+apP+1e-12),
            'Age':dem.loc[key,'Age'],'Sex':dem.loc[key,'Sex'],'BMI':dem.loc[key,'BMI']})
L=pd.DataFrame(rows)
L['MS']=(L['cohort']=='M').astype(int)
L['Female']=(L['Sex']==2).astype(int)
L['BMI_c']=L['BMI']-L['BMI'].mean(); L['Age_c']=L['Age']-L['Age'].mean()
print(f"rows={len(L)}  subjects={L.key.nunique()}  POMS={L[L.MS==1].key.nunique()}  HC={L[L.MS==0].key.nunique()}\n")

outcomes=['ML_Over_ENMO','ML_Over_VT','ML_Energy_Frac','ML_Spec_Frac']
terms=[('Intercept','(Intercept)'),('MS','MS'),('Female','Female'),('BMI_c','BMI'),('Age_c','Age')]

def fit_lme(o):
    """Random-intercept LME; rescale tiny outcomes to unit variance for conditioning."""
    sub=L.dropna(subset=[o]).copy(); s=1.0/sub[o].std(); sub['_y']=sub[o]*s
    last=None
    for meth in ['lbfgs','powell','cg','nm']:
        try:
            f=smf.mixedlm("_y ~ MS + Female + BMI_c + Age_c",sub,groups=sub['key']).fit(method=meth)
            last=(f,s)
            if np.all(np.isfinite(f.bse[:5])): return f,s
        except Exception: continue
    return last
fits={o:fit_lme(o) for o in outcomes}

table_rows=[]
for kk,lab in terms:
    row={'Fixed effects':lab}
    for o in outcomes:
        f,s=fits[o]; est=f.params[kk]/s; lo,hi=f.conf_int().loc[kk]/s; pv=f.pvalues[kk]
        row[f'{o}|Effect size']=round(est,4); row[f'{o}|95% CI']=f'{lo:.4f} to {hi:.4f}'; row[f'{o}|p Value']=round(pv,4)
    table_rows.append(row)
T5=pd.DataFrame(table_rows)
OUT=BASE/'sway'/'table'; T5.to_csv(OUT/'walksway_table5_lme_group_differences.csv',index=False)
print("Saved",OUT/'walksway_table5_lme_group_differences.csv')
def star(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
for o in outcomes:
    f,s=fits[o]; print(f"\n=== {o} ===")
    print(f"{'effect':<12}{'estimate':>10}{'95% CI':>24}{'p':>10}")
    for kk,lab in terms:
        est=f.params[kk]/s; lo,hi=f.conf_int().loc[kk]/s; pv=f.pvalues[kk]
        print(f"{lab:<12}{est:>10.4f}{('  ['+f'{lo:.4f}, {hi:.4f}'+']'):>24}{pv:>9.4f}{star(pv)}")
