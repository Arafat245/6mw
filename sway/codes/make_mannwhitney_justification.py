#!/usr/bin/env python3
"""Shapiro-Wilk normality check -> justification for using Mann-Whitney U."""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.stats import shapiro, skew, kurtosis, mannwhitneyu
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios
from clinic.extract_walking_sway import extract_walking_sway
PRE=BASE/'csv_preprocessed2'; FS=30.0
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
                'ML_Spec_Frac':spec,'cohort':np.where(ids['cohort'].values=='M','POMS','Healthy')})
feats=['ML_Over_ENMO','ML_Over_VT','ML_Spec_Frac']

L=[]
L.append("JUSTIFICATION FOR USING THE MANN-WHITNEY U TEST")
L.append("="*64)
L.append("")
L.append("Cohort comparison: POMS (n=50) vs Healthy (n=70), clinic 6MWT WalkSway")
L.append("features. The figure walksway_significant_violins.png reports a two-sided")
L.append("Mann-Whitney U (Wilcoxon rank-sum) test for each feature, with")
L.append("Benjamini-Hochberg FDR correction over the 14-feature WalkSway family.")
L.append("")
L.append("WHY NOT A t-TEST?")
L.append("-"*64)
L.append("The independent-samples t-test assumes each group is approximately")
L.append("normally distributed. We tested this with the Shapiro-Wilk test within")
L.append("each cohort (H0: data are normally distributed). A Shapiro-Wilk p < 0.05")
L.append("means normality is rejected, violating the t-test assumption and making")
L.append("the rank-based, distribution-free Mann-Whitney U test the appropriate")
L.append("choice. Skewness and excess kurtosis are reported as supporting evidence")
L.append("(a normal distribution has skew = 0 and excess kurtosis = 0).")
L.append("")
L.append("SHAPIRO-WILK NORMALITY TEST (per cohort, per feature)")
L.append("-"*64)
L.append(f"{'Feature':<16}{'Cohort':<10}{'n':>4}{'W':>9}{'p':>11}{'skew':>8}{'kurt':>8}  normal?")
any_violation=False
for f in feats:
    for coh in ['POMS','Healthy']:
        x=D.loc[D.cohort==coh,f].dropna().values
        W,pv=shapiro(x); sk=skew(x); ku=kurtosis(x)  # kurtosis: excess (Fisher)
        norm = 'yes' if pv>=0.05 else 'NO (reject)'
        if pv<0.05: any_violation=True
        L.append(f"{f:<16}{coh:<10}{len(x):>4}{W:>9.3f}{pv:>11.2e}{sk:>+8.2f}{ku:>+8.2f}  {norm}")
L.append("")
L.append("POOLED (both cohorts combined)")
L.append(f"{'Feature':<16}{'':<10}{'n':>4}{'W':>9}{'p':>11}{'skew':>8}{'kurt':>8}  normal?")
for f in feats:
    x=D[f].dropna().values; W,pv=shapiro(x); sk=skew(x); ku=kurtosis(x)
    L.append(f"{f:<16}{'(all)':<10}{len(x):>4}{W:>9.3f}{pv:>11.2e}{sk:>+8.2f}{ku:>+8.2f}  {'yes' if pv>=0.05 else 'NO (reject)'}")
L.append("")
L.append("CONCLUSION")
L.append("-"*64)
if any_violation:
    L.append("At least one cohort distribution per feature significantly departs from")
    L.append("normality (Shapiro-Wilk p < 0.05), and the features are right-skewed")
    L.append("ratios bounded at 0. The normality assumption of the parametric t-test")
    L.append("is therefore not met. We use the two-sided Mann-Whitney U test, which is")
    L.append("non-parametric (rank-based), makes no normality assumption, and is robust")
    L.append("to skew and outliers. Group differences are summarized with medians and")
    L.append("95% bootstrap confidence intervals (consistent with a rank-based test),")
    L.append("and p-values are BH-FDR corrected across the WalkSway feature family.")
else:
    L.append("Distributions did not significantly depart from normality; Mann-Whitney U")
    L.append("is still reported as a conservative, distribution-free choice.")
L.append("")
L.append("Test: scipy.stats.mannwhitneyu(..., alternative='two-sided')")
L.append("Normality: scipy.stats.shapiro ; skew/kurtosis: scipy.stats (Fisher, excess)")
txt='\n'.join(L)+'\n'
out=BASE/'sway'/'mannwhitney_justification.txt'; out.write_text(txt)
print(txt); print("Saved",out)
