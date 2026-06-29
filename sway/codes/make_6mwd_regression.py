#!/usr/bin/env python3
"""
OLS regression of 6MWD on ML_Over_ENMO + demographics (Brenton-style table).

Model:  6MWD_m ~ ML_Over_ENMO + MS + Female + BMI_c + Age_c
One whole-walk value per subject; n=120 (50 POMS, 70 Healthy). 6MWD converted
feet -> meters (x0.3048). BMI and Age mean-centered. MS = POMS(1)/control(0),
Female = Sex==2. 6MWD taken from the clinic-6MWT filename (verified identical to
feats/target_6mwd.csv on all 101 labelled subjects).
"""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios
from clinic.extract_walking_sway import extract_walking_sway
PRE=BASE/'csv_preprocessed2'; FT2M=0.3048
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_(?P<yr>\d+)_(?P<d>\d+)\.csv$',re.I)

ids,gait_rows,ws_rows=[],[],[]
for p in sorted(PRE.glob('*.csv')):
    m=fre.match(p.name)
    if not m: continue
    df=pd.read_csv(p)
    ids.append({'cohort':m.group('c').upper(),'subj_id':int(m.group('id')),'sixmwd':int(m.group('d'))})
    gait_rows.append(extract_gait10(df)); ws_rows.append(extract_walking_sway(df['AP'].values,df['ML'].values,df['VT'].values))
ids=pd.DataFrame(ids)
gm=add_sway_ratios(pd.concat([ids.reset_index(drop=True),pd.DataFrame(gait_rows)],axis=1)
                   .merge(compute_vt_rms(PRE),on=['cohort','subj_id','sixmwd'],how='left'))
D=pd.DataFrame({'ML_Over_ENMO':gm['ml_over_enmo'].values,
                'SixMWD_m':ids['sixmwd'].values*FT2M})
D['key']=ids['cohort']+ids['subj_id'].map(lambda x:f"{x:02d}")
demo=pd.read_excel(BASE/'SwayDemographics.xlsx')
def key_id(s):
    m=re.match(r'\s*([CM])-?(\d+)',str(s),re.I); return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None
demo['key']=demo['ID'].apply(key_id)
D=D.merge(demo[['key','Sex','Age','BMI']],on='key',how='left')
D['MS']=(D['key'].str[0]=='M').astype(int)
D['Female']=(D['Sex']==2).astype(int)
D['BMI_c']=D['BMI']-D['BMI'].mean(); D['Age_c']=D['Age']-D['Age'].mean()
print(f"n={len(D)}  POMS={int(D.MS.sum())}  Healthy={int((D.MS==0).sum())}  outcome=6MWD (m)\n")

fit=smf.ols('SixMWD_m ~ ML_Over_ENMO + MS + Female + BMI_c + Age_c',data=D).fit()
terms=[('Intercept','(Intercept)'),('ML_Over_ENMO','ML\\_Over\\_ENMO'),
       ('MS','MS'),('Female','Female'),('BMI_c','BMI'),('Age_c','Age')]
plain={'Intercept':'(Intercept)','ML_Over_ENMO':'ML_Over_ENMO','MS':'MS','Female':'Female','BMI_c':'BMI','Age_c':'Age'}
def star(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

rows=[]
for kk,_ in terms:
    est=fit.params[kk]; lo,hi=fit.conf_int().loc[kk]; pv=fit.pvalues[kk]
    rows.append({'Fixed effects':plain[kk],'Effect size':round(est,3),
                 '95% CI':f'{lo:.3f} to {hi:.3f}','p Value':round(pv,4)})
tab=pd.DataFrame(rows)
OUT=BASE/'sway'/'table'; tab.to_csv(OUT/'sixmwd_on_ml_over_enmo.csv',index=False)

print(f"=== OLS: 6MWD (m) ~ ML_Over_ENMO + MS + Female + BMI + Age   (R2={fit.rsquared:.3f}, n={int(fit.nobs)}) ===")
print(f"{'Fixed effects':<14}{'Effect size':>12}{'95% CI':>24}{'p':>10}")
for kk,_ in terms:
    est=fit.params[kk]; lo,hi=fit.conf_int().loc[kk]; pv=fit.pvalues[kk]
    print(f"{plain[kk]:<14}{est:>12.3f}{('  ['+f'{lo:.2f}, {hi:.2f}'+']'):>24}{pv:>9.4f}{star(pv)}")

# ---- Brenton Table 5 / Table 3 style LaTeX (single outcome) ----
def fp(p):
    s='\\textless0.001' if p<0.001 else f'{p:.3f}'
    return f'\\textbf{{{s}}}' if p<0.05 else s
L=['% Requires \\usepackage{booktabs}','\\begin{table}[t]','\\centering',
   '\\caption{\\textbf{Association of Mediolateral Sway (ML\\_Over\\_ENMO) With 6-Minute Walk '
   'Distance Using Linear Regression}}','\\label{tab:sixmwd_ml_over_enmo}',
   '\\begin{tabular}{l ccc}','\\toprule',
   'Fixed effects & Effect size & 95\\% CI & $p$ Value \\\\','\\midrule']
for kk,lab in terms:
    label=lab if kk=='Intercept' else f'\\textbf{{{lab}}}'
    est=fit.params[kk]; lo,hi=fit.conf_int().loc[kk]; pv=fit.pvalues[kk]
    L.append(f'{label} & {est:.3f} & {lo:.3f} to {hi:.3f} & {fp(pv)} \\\\')
L+=['\\bottomrule','\\end{tabular}','\\\\[2pt]',
    '{\\footnotesize Abbreviations: BMI = body mass index; MS = multiple sclerosis; '
    'POMS = pediatric-onset MS. Outcome is 6-minute walk distance (6MWD) in meters; $n=120$ '
    '(50 POMS, 70 healthy controls). Effect size represents the estimate from a linear regression '
    f'(ordinary least squares; $R^2={fit.rsquared:.3f}$) of 6MWD on ML\\_Over\\_ENMO, MS status, sex, '
    'BMI, and age, using one whole-walk value per participant from the clinic 6-minute walk. CIs '
    'represent the 95\\% CIs. BMI and age were mean-centered. ML\\_Over\\_ENMO (mediolateral RMS '
    'acceleration divided by ENMO) is dimensionless, so its effect size is meters of 6MWD per unit. '
    '``MS\'\' represents the group difference between the POMS and control group; ``Female\'\' '
    'represents the group difference between all females and males. Significant values ($p<0.05$) '
    'are shown in bold.}','\\end{table}']
(OUT/'sixmwd_on_ml_over_enmo.tex').write_text('\n'.join(L)+'\n')
print("\nSaved",OUT/'sixmwd_on_ml_over_enmo.csv')
print("Saved",OUT/'sixmwd_on_ml_over_enmo.tex')
