#!/usr/bin/env python3
"""
Brenton-2022 Table 3 analog: LME of the within-walk ML_Over_ENMO trajectory.
Outcome = ML_Over_ENMO (vs Brenton's minute-by-minute 6MW gait speed); BMI is
continuous & mean-centered (vs Brenton's Overweight/Obese dummies).

Model:  ML_Over_ENMO ~ MS + t + t2 + BMI_c + Age_c + Female + MS:t + MS:t2 , (1|subject)
6 within-walk epochs/subject; t=0..5 (intercept = first epoch). n=120.
"""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10
PRE=BASE/'csv_preprocessed2'; N=6
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_',re.I)
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
        g=extract_gait10(seg); v=g['ml_rms_g']/g['enmo_mean_g'] if g['enmo_mean_g'] else np.nan
        rows.append({'key':key,'ML_Over_ENMO':v,'t':e,
                     'Age':dem.loc[key,'Age'],'Sex':dem.loc[key,'Sex'],'BMI':dem.loc[key,'BMI'],
                     'MS':1 if m.group('c').upper()=='M' else 0})
L=pd.DataFrame(rows).dropna(subset=['ML_Over_ENMO'])
L['Female']=(L['Sex']==2).astype(int); L['t2']=L['t']**2
L['BMI_c']=L['BMI']-L['BMI'].mean(); L['Age_c']=L['Age']-L['Age'].mean()
print(f"rows={len(L)}  subjects={L.key.nunique()}  POMS={L[L.MS==1].key.nunique()}  HC={L[L.MS==0].key.nunique()}\n")

mdl=smf.mixedlm("ML_Over_ENMO ~ MS + t + t2 + BMI_c + Age_c + Female + MS:t + MS:t2",L,groups=L['key'])
fit=None
for meth in ['powell','lbfgs','cg','nm','bfgs']:
    try:
        f=mdl.fit(method=meth)
        if np.all(np.isfinite(f.bse[:9])): fit=f; break
        fit=fit or f
    except Exception: continue
order=[('Intercept','(Intercept)'),('MS','MS'),('t','Time'),('t2','Time$^2$'),
       ('BMI_c','BMI'),('Age_c','Age'),('Female','Female'),
       ('MS:t','MS $\\times$ Time'),('MS:t2','MS $\\times$ Time$^2$')]
plain={'Intercept':'(Intercept)','MS':'MS','t':'Time','t2':'Time^2','BMI_c':'BMI','Age_c':'Age',
       'Female':'Female','MS:t':'MS x Time','MS:t2':'MS x Time^2'}
ci=fit.conf_int()
def star(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

rows_out=[]
for kk,_ in order:
    rows_out.append({'Fixed effects':plain[kk],'Estimate':round(fit.params[kk],4),
                     '95% CI':f'{ci.loc[kk,0]:.4f} to {ci.loc[kk,1]:.4f}','p Value':round(fit.pvalues[kk],4)})
tab=pd.DataFrame(rows_out)
OUT=BASE/'sway'/'table'; tab.to_csv(OUT/'ml_over_enmo_table3_trajectory.csv',index=False)

print(f"=== Table 3 analog: LME of ML_Over_ENMO trajectory (BMI continuous) ===")
print(f"{'Fixed effects':<16}{'Estimate':>10}{'95% CI':>26}{'p':>10}")
for kk,_ in order:
    est=fit.params[kk]; lo,hi=ci.loc[kk]; pv=fit.pvalues[kk]
    print(f"{plain[kk]:<16}{est:>10.4f}{('  ['+f'{lo:.4f}, {hi:.4f}'+']'):>26}{pv:>9.4f}{star(pv)}")

# ---- Brenton Table 3 style LaTeX ----
def fp(p):
    s='\\textless0.001' if p<0.001 else f'{p:.3f}'
    return f'\\textbf{{{s}}}' if p<0.05 else s
Lx=['% Requires \\usepackage{booktabs}','\\begin{table}[t]','\\centering',
    '\\caption{\\textbf{Linear Mixed-Effects Modeling of the ML\\_Over\\_ENMO Trajectory '
    'Across the 6-Minute Walk}}','\\label{tab:ml_over_enmo_traj}',
    '\\begin{tabular}{l ccc}','\\toprule',
    '\\multicolumn{4}{l}{Outcome: minute-by-minute ML\\_Over\\_ENMO during 6MWT} \\\\','\\midrule',
    'Fixed effects & Estimate & 95\\% CI & $p$ Value \\\\','\\midrule']
for kk,lab in order:
    label=lab if kk=='Intercept' else f'\\textbf{{{lab}}}'
    est=fit.params[kk]; lo,hi=ci.loc[kk]; pv=fit.pvalues[kk]
    Lx.append(f'{label} & {est:.3f} & {lo:.3f} to {hi:.3f} & {fp(pv)} \\\\')
Lx+=['\\bottomrule','\\end{tabular}','\\\\[2pt]',
     '{\\footnotesize Abbreviations: BMI = body mass index; MS = multiple sclerosis; '
     'POMS = pediatric-onset MS. Outcome is ML\\_Over\\_ENMO (mediolateral RMS / ENMO; dimensionless) '
     'computed in 6 within-walk epochs ($\\sim$1 min each) of the clinic 6-minute walk; $n=120$ '
     '(50 POMS, 70 healthy controls; 720 epoch-level observations). Estimates are from a linear '
     'mixed-effects model with a random intercept per participant. ``Time'' (epoch index $0$--$5$) '
     'captures the linear within-walk change and ``Time$^2$'' the quadratic change for controls; '
     '``MS'' captures the POMS-vs-control difference at the first epoch, and ``MS $\\times$ Time'' / '
     '``MS $\\times$ Time$^2$'' the difference in trajectory shape for POMS. BMI and age are '
     'mean-centered; ``Female'' is the female-vs-male difference. CIs are 95\\% CIs. Significant '
     'values ($p<0.05$) are shown in bold.}','\\end{table}']
(OUT/'ml_over_enmo_table3_trajectory.tex').write_text('\n'.join(Lx)+'\n')
print("\nSaved",OUT/'ml_over_enmo_table3_trajectory.csv')
print("Saved",OUT/'ml_over_enmo_table3_trajectory.tex')
