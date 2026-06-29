#!/usr/bin/env python3
"""
Brenton Table-5-style OLS with ML_Over_ENMO as the outcome.

    ML_Over_ENMO ~ MS + Female + BMI_c + Age_c   (OLS, n=120)

One whole-walk value per subject. BMI and Age mean-centered. MS = POMS(1)/
control(0); Female = Sex==2. Effect size = OLS estimate; CI = 95% CI.
"""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios
from clinic.extract_walking_sway import extract_walking_sway
PRE=BASE/'csv_preprocessed2'
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_(?P<yr>\d+)_(?P<d>\d+)\.csv$',re.I)

ids,gait_rows=[],[]
for p in sorted(PRE.glob('*.csv')):
    m=fre.match(p.name)
    if not m: continue
    df=pd.read_csv(p); extract_walking_sway(df['AP'].values,df['ML'].values,df['VT'].values)
    ids.append({'cohort':m.group('c').upper(),'subj_id':int(m.group('id')),'sixmwd':int(m.group('d'))})
    gait_rows.append(extract_gait10(df))
ids=pd.DataFrame(ids)
gm=add_sway_ratios(pd.concat([ids.reset_index(drop=True),pd.DataFrame(gait_rows)],axis=1)
                   .merge(compute_vt_rms(PRE),on=['cohort','subj_id','sixmwd'],how='left'))
D=pd.DataFrame({'ML_Over_ENMO':gm['ml_over_enmo'].values})
D['key']=ids['cohort']+ids['subj_id'].map(lambda x:f"{x:02d}")
demo=pd.read_excel(BASE/'SwayDemographics.xlsx')
def key_id(s):
    m=re.match(r'\s*([CM])-?(\d+)',str(s),re.I); return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None
demo['key']=demo['ID'].apply(key_id)
D=D.merge(demo[['key','Sex','Age','BMI']],on='key',how='left')
D['MS']=(D['key'].str[0]=='M').astype(int); D['Female']=(D['Sex']==2).astype(int)
D['BMI_c']=D['BMI']-D['BMI'].mean(); D['Age_c']=D['Age']-D['Age'].mean()
print(f"n={len(D)}  POMS={int(D.MS.sum())}  Healthy={int((D.MS==0).sum())}  outcome=ML_Over_ENMO\n")

fit=smf.ols('ML_Over_ENMO ~ MS + Female + BMI_c + Age_c',data=D).fit()
terms=[('Intercept','(Intercept)'),('MS','MS'),('Female','Female'),('BMI_c','BMI'),('Age_c','Age')]
def star(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

rows=[]
for kk,lab in terms:
    est=fit.params[kk]; lo,hi=fit.conf_int().loc[kk]; pv=fit.pvalues[kk]
    rows.append({'Fixed effects':lab,'Effect size':round(est,4),
                 '95% CI':f'{lo:.4f} to {hi:.4f}','p Value':round(pv,4)})
tab=pd.DataFrame(rows)
OUT=BASE/'sway'/'table'; tab.to_csv(OUT/'ml_over_enmo_ols.csv',index=False)

print(f"=== OLS: ML_Over_ENMO ~ MS + Female + BMI + Age   (R2={fit.rsquared:.3f}, n={int(fit.nobs)}) ===")
print(f"{'Fixed effects':<14}{'Effect size':>12}{'95% CI':>26}{'p':>10}")
for kk,lab in terms:
    est=fit.params[kk]; lo,hi=fit.conf_int().loc[kk]; pv=fit.pvalues[kk]
    print(f"{lab:<14}{est:>12.4f}{('  ['+f'{lo:.4f}, {hi:.4f}'+']'):>26}{pv:>9.4f}{star(pv)}")

# ---- Brenton Table 5 style LaTeX ----
def fp(p):
    s='\\textless0.001' if p<0.001 else f'{p:.3f}'
    return f'\\textbf{{{s}}}' if p<0.05 else s
L=['% Requires \\usepackage{booktabs}','\\begin{table}[t]','\\centering',
   '\\caption{\\textbf{Demographic and Disease Predictors of Mediolateral Sway '
   '(ML\\_Over\\_ENMO) Using Linear Regression}}','\\label{tab:ml_over_enmo_ols}',
   '\\begin{tabular}{l ccc}','\\toprule',
   'Fixed effects & Effect size & 95\\% CI & $p$ Value \\\\','\\midrule']
for kk,lab in terms:
    label=lab if kk=='Intercept' else f'\\textbf{{{lab}}}'
    est=fit.params[kk]; lo,hi=fit.conf_int().loc[kk]; pv=fit.pvalues[kk]
    L.append(f'{label} & {est:.3f} & {lo:.3f} to {hi:.3f} & {fp(pv)} \\\\')
L+=['\\bottomrule','\\end{tabular}','\\\\[2pt]',
    '{\\footnotesize Abbreviations: BMI = body mass index; MS = multiple sclerosis; '
    'POMS = pediatric-onset MS. Outcome is ML\\_Over\\_ENMO (mediolateral RMS acceleration '
    'divided by ENMO; dimensionless), one whole-walk value per participant from the clinic '
    f'6-minute walk; $n=120$ (50 POMS, 70 healthy controls). Effect size is the estimate from a '
    f'linear regression (ordinary least squares; $R^2={fit.rsquared:.3f}$) of ML\\_Over\\_ENMO on MS '
    'status, sex, BMI, and age. CIs represent the 95\\% CIs. BMI and age were mean-centered. '
    '``MS\'\' represents the group difference between the POMS and control group; ``Female\'\' '
    'represents the group difference between all females and males. Significant values ($p<0.05$) '
    'are shown in bold.}','\\end{table}']
(OUT/'ml_over_enmo_ols.tex').write_text('\n'.join(L)+'\n')
print("\nSaved",OUT/'ml_over_enmo_ols.csv')
print("Saved",OUT/'ml_over_enmo_ols.tex')
