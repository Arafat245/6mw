#!/usr/bin/env python3
"""
Brenton-2022-style Table 5 for the significant clinic WalkSway features.

Brenton et al. (Neurology 2022) Table 5 models each actigraphy outcome with a
linear mixed-effects model (day-to-day random effect nested within participant,
adjusting for sex, age, BMI; BMI and age mean-centered). Our WalkSway features
are SINGLE per-subject values from one clinic 6MWT -> no repeated measures, so
the LME reduces exactly to OLS with the same fixed-effect structure:

    feature ~ MS + Female + BMI_centered + Age_centered

Fixed effects reported (like Brenton): (Intercept), MS, Female, BMI, Age,
each with Effect size, 95% CI, p Value.  MS = POMS-vs-control difference,
Female = female-vs-male difference (Sex==2 is female per the project convention).
Cohort = full clinic cohort n=120 (50 POMS, 70 Healthy).
"""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios
from clinic.extract_walking_sway import extract_walking_sway
PRE=BASE/'csv_preprocessed2'; FS=30.0
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_(?P<yr>\d+)_(?P<d>\d+)\.csv$',re.I)
def bandpow(x,lo,hi):
    f,Pp=welch(x-np.mean(x),fs=FS,nperseg=min(1024,len(x))); b=(f>=lo)&(f<=hi); return np.sum(Pp[b])

# --- extract the 4 significant sway features (120 subjects) ---
ids,gait_rows,ws_rows,new_rows=[],[],[],[]
for p in sorted(PRE.glob('*.csv')):
    m=fre.match(p.name)
    if not m: continue
    df=pd.read_csv(p); ap,ml,vt=df['AP'].values,df['ML'].values,df['VT'].values
    ids.append({'cohort':m.group('c').upper(),'subj_id':int(m.group('id')),'sixmwd':int(m.group('d'))})
    gait_rows.append(extract_gait10(df)); ws_rows.append(extract_walking_sway(ap,ml,vt))
    vAP,vML,vVT=np.var(ap),np.var(ml),np.var(vt); mlP=bandpow(ml,0.3,10); apP=bandpow(ap,0.3,10)
    new_rows.append({'ML_Energy_Frac':vML/(vAP+vML+vVT+1e-12),'ML_Spec_Frac':mlP/(mlP+apP+1e-12)})
ids=pd.DataFrame(ids)
gm=add_sway_ratios(pd.concat([ids.reset_index(drop=True),pd.DataFrame(gait_rows)],axis=1)
                   .merge(compute_vt_rms(PRE),on=['cohort','subj_id','sixmwd'],how='left'))
D=pd.DataFrame({'ML_Over_ENMO':gm['ml_over_enmo'].values,'ML_Over_VT':gm['ml_over_vt'].values,
               'ML_Energy_Frac':pd.DataFrame(new_rows)['ML_Energy_Frac'].values,
               'ML_Spec_Frac':pd.DataFrame(new_rows)['ML_Spec_Frac'].values})
D['key']=ids['cohort']+ids['subj_id'].map(lambda x:f"{x:02d}")

# --- merge demographics (Sex/Age/BMI) ---
demo=pd.read_excel(BASE/'SwayDemographics.xlsx')
def key_id(s):
    m=re.match(r'\s*([CM])-?(\d+)',str(s),re.I); return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None
demo['key']=demo['ID'].apply(key_id)
D=D.merge(demo[['key','Sex','Age','BMI']],on='key',how='left')
D['MS']=(D['key'].str[0]=='M').astype(int)         # 1 = POMS
D['Female']=(D['Sex']==2).astype(int)              # 2 = female (project convention)
D['BMI_c']=D['BMI']-D['BMI'].mean()                # mean-centered (Brenton)
D['Age_c']=D['Age']-D['Age'].mean()
n_poms=int(D['MS'].sum()); n_hc=int((D['MS']==0).sum())
print(f"n={len(D)}  POMS={n_poms}  Healthy={n_hc}  (LME -> OLS, single 6MWT per subject)\n")

outcomes=['ML_Over_ENMO','ML_Over_VT','ML_Energy_Frac','ML_Spec_Frac']
terms=[('Intercept','Intercept'),('MS','MS'),('Female','Female'),('BMI_c','BMI'),('Age_c','Age')]
rows=[]
fits={}
for out in outcomes:
    fit=smf.ols(f'{out} ~ MS + Female + BMI_c + Age_c',data=D).fit()
    fits[out]=fit
for label_key,label in terms:
    row={'Fixed effects':label}
    for out in outcomes:
        f=fits[out]; est=f.params[label_key]; lo,hi=f.conf_int().loc[label_key]; pv=f.pvalues[label_key]
        row[f'{out}|Effect size']=round(est,4)
        row[f'{out}|95% CI']=f'{lo:.4f} to {hi:.4f}'
        row[f'{out}|p Value']=round(pv,4)
    rows.append(row)
tab=pd.DataFrame(rows)
OUT=BASE/'sway'/'table'; OUT.mkdir(parents=True,exist_ok=True)
tab.to_csv(OUT/'walksway_lme_group_differences.csv',index=False)
print("Saved",OUT/'walksway_lme_group_differences.csv')

# pretty print
def star(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
for out in outcomes:
    f=fits[out]; print(f"\n=== {out}  (R²={f.rsquared:.3f}) ===")
    print(f"{'effect':<10}{'estimate':>11}{'95% CI':>26}{'p':>10}")
    for lk,lb in terms:
        est=f.params[lk]; lo,hi=f.conf_int().loc[lk]; pv=f.pvalues[lk]
        print(f"{lb:<10}{est:>11.4f}{f'  [{lo:.4f}, {hi:.4f}]':>26}{pv:>9.4f}{star(pv)}")

# ---- LaTeX export, styled like Brenton (2022) Table 5 ----
# CI as "x to y"; significant p (<0.05) bold; fixed-effect labels bold; abbreviation footnote.
def fp(p):
    s = '\\textless0.001' if p<0.001 else f'{p:.3f}'
    return f'\\textbf{{{s}}}' if p<0.05 else s
hdr={'ML_Over_ENMO':'ML\\_Over\\_ENMO','ML_Over_VT':'ML\\_Over\\_VT',
     'ML_Energy_Frac':'ML\\_Energy\\_Frac','ML_Spec_Frac':'ML\\_Spec\\_Frac'}
L=[]
L.append('% Requires \\usepackage{booktabs,graphicx}')
L.append('\\begin{table}[t]')
L.append('\\centering')
L.append('\\caption{\\textbf{Group Differences in Mediolateral WalkSway Features Using Linear '
         'Regression}}')
L.append('\\label{tab:walksway_group_diff}')
L.append('\\resizebox{\\textwidth}{!}{%')
L.append('\\begin{tabular}{l ccc ccc ccc ccc}')
L.append('\\toprule')
L.append(' & \\multicolumn{3}{c}{'+hdr['ML_Over_ENMO']+'} & \\multicolumn{3}{c}{'+hdr['ML_Over_VT']
         +'} & \\multicolumn{3}{c}{'+hdr['ML_Energy_Frac']+'} & \\multicolumn{3}{c}{'
         +hdr['ML_Spec_Frac']+'} \\\\')
L.append('\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}')
L.append('Fixed effects & Effect size & CI & $p$ Value & Effect size & CI & $p$ Value '
         '& Effect size & CI & $p$ Value & Effect size & CI & $p$ Value \\\\')
L.append('\\midrule')
for lk,lb in terms:
    label = lb if lb=='Intercept' else f'\\textbf{{{lb}}}'
    label = '(Intercept)' if lb=='Intercept' else label
    cells=[label]
    for out in outcomes:
        f=fits[out]; est=f.params[lk]; lo,hi=f.conf_int().loc[lk]; pv=f.pvalues[lk]
        cells += [f'{est:.3f}', f'{lo:.3f} to {hi:.3f}', fp(pv)]
    L.append(' & '.join(cells)+' \\\\')
L.append('\\bottomrule')
L.append('\\end{tabular}}')
L.append('\\\\[2pt]')
L.append('{\\footnotesize Abbreviations: BMI = body mass index; MS = multiple sclerosis; '
         'POMS = pediatric-onset MS. $n=120$ (50 POMS, 70 healthy controls). Effect size represents '
         'the estimate from linear regression (ordinary least squares) of each WalkSway feature on '
         'MS status, sex, BMI, and age; one whole-walk value per participant from the clinic 6-minute '
         'walk. CIs represent the 95\\% CIs. BMI and age were mean-centered. ``MS\'\' represents the '
         'group difference between the POMS and control group; ``Female\'\' represents the group '
         'difference between all females and males. All WalkSway features are dimensionless. '
         'Significant values ($p<0.05$) are shown in bold.}')
L.append('\\end{table}')
tex='\n'.join(L)+'\n'
(OUT/'walksway_group_differences.tex').write_text(tex)
print("\nSaved",OUT/'walksway_group_differences.tex')
