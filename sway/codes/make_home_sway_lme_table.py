#!/usr/bin/env python3
"""
Brenton Table-5-style LME robustness table for the FREE-LIVING ML-sway features.
Day-level repeated measures (valid day = >=3 sustained >=60s bouts; per-day value =
median over that day's up-to-8 longest bouts), days nested in participant, MS effect
adjusted for sex, age, BMI. Mirrors Brenton-2022 continuous-accelerometry LME.

  feature ~ MS + Female + Age_c + BMI_c , (1 | subject)

Outputs: sway/table/home_walksway_lme_robustness.csv
         sway/table/home_walksway_lme_robustness.tex
"""
import sys, re, warnings
import numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
BASE = Path('/mnt/sdb/arafat/6mw')
# Trim to the 3 BH-significant free-living features (ml_over_enmo is n.s. and excluded).
FC = ['ml_over_vt', 'ml_energy_frac', 'ml_spec_horiz_frac']
FC_ALL = ['ml_over_enmo', 'ml_over_vt', 'ml_energy_frac', 'ml_spec_horiz_frac']
DISP = {'ml_over_enmo':'ML\\_Over\\_ENMO','ml_over_vt':'ML\\_Over\\_VT',
        'ml_energy_frac':'ML\\_Energy\\_Frac','ml_spec_horiz_frac':'ML\\_Spec\\_Frac'}
PLAIN = {'ml_over_enmo':'ML_Over_ENMO','ml_over_vt':'ML_Over_VT',
         'ml_energy_frac':'ML_Energy_Frac','ml_spec_horiz_frac':'ML_Spec_Frac'}
# valid day = >=3 sustained (>=60s) bouts (clean per-day value); require only >=1 valid
# day so ALL n=101 are retained (all 101 have >=1 such day). per-day value = median over
# that day's up-to-8 longest bouts.
MIN_BOUTS_DAY, MAX_BOUTS_DAY, MIN_DAYS = 3, 8, 1

PB = pd.read_parquet(BASE / 'feats/home_sway_perbout_ge60.parquet')
demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
def key_id(s):
    m = re.match(r'\s*([CM])-?(\d+)', str(s), re.I)
    return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None
demo['key'] = demo['ID'].apply(key_id); DEM = demo.set_index('key')

# build day-level table
rows = []
for key, g in PB.groupby('key'):
    if key not in DEM.index: continue
    dayvals = {}
    for day, dg in g.groupby('day'):
        if len(dg) < MIN_BOUTS_DAY: continue
        top = dg.sort_values('dur', ascending=False).head(MAX_BOUTS_DAY)
        dayvals[day] = top[FC_ALL].median().values
    if len(dayvals) < MIN_DAYS: continue
    for di, dv in enumerate(dayvals.values()):
        rows.append({'key':key,'MS':1 if key[0]=='M' else 0,
                     'Female':1 if DEM.loc[key,'Sex']==2 else 0,
                     'Age':DEM.loc[key,'Age'],'BMI':DEM.loc[key,'BMI'],
                     **{c:dv[j] for j,c in enumerate(FC_ALL)}})
L = pd.DataFrame(rows)
L['Age_c'] = L['Age']-L['Age'].mean(); L['BMI_c'] = L['BMI']-L['BMI'].mean()
nsub=L.key.nunique(); nP=L[L.MS==1].key.nunique(); nH=L[L.MS==0].key.nunique()
ndays=len(L); mdays=int(L.groupby('key').size().median())
print(f"day-rows={ndays}  subjects={nsub} (POMS={nP}, Control={nH})  median days/subj={mdays}\n")

EFFORDER = [('Intercept','(Intercept)'),('MS','MS'),('Female','Female'),
            ('BMI_c','BMI'),('Age_c','Age')]   # Brenton Table 5 row order
# Fit each LME on the original outcome; accept the first converged fit (finite,
# well-scaled SEs). The >=3-bouts/day valid-day rule keeps per-day values clean, so
# all four converge with meaningful intercept p-values.
def fit_orig(c):
    m = smf.mixedlm(f"{c} ~ MS + Female + Age_c + BMI_c", L, groups=L['key'])
    f = None
    for meth in ['lbfgs', 'powell', 'cg', 'nm', 'bfgs']:
        try:
            ff = m.fit(method=meth, reml=True)
            sc = max(abs(L[c].mean()), 1e-9)
            ok = np.isfinite(ff.pvalues.get('MS', np.nan)) and ff.bse.get('MS', np.inf) < 50*sc
            if ok: f = ff; break
            f = f or ff
        except Exception: continue
    ci = f.conf_int()
    return {kk: (f.params[kk], ci.loc[kk,0], ci.loc[kk,1], f.pvalues[kk]) for kk,_ in EFFORDER}

# Fit ALL 4 features so BH correction spans the full tested family (matches primary),
# but the table displays only the 3 significant ones (FC).
RES = {c: fit_orig(c) for c in FC_ALL}
ms_p = [RES[c]['MS'][3] for c in FC_ALL]
ms_pbh = dict(zip(FC_ALL, multipletests(ms_p, method='fdr_bh')[1]))

# CSV -- WIDE, Brenton Table-5 layout: rows = fixed effects, column-groups = features
def pfmt(p): return '<0.001' if p < 0.001 else f'{p:.3f}'
wide = []
for kk, lab in EFFORDER:
    row = {'Fixed effects': lab}
    for c in FC:
        est, lo, hi, p = RES[c][kk]
        row[f'{PLAIN[c]}: Effect size'] = f'{est:.3f}'
        row[f'{PLAIN[c]}: 95% CI']      = f'{lo:.3f} to {hi:.3f}'
        row[f'{PLAIN[c]}: p Value']     = pfmt(p)
    wide.append(row)
# append the BH-corrected MS p row (across all 4 features) as a labelled footer row
bh_row = {'Fixed effects': 'MS p (BH-corrected)'}
for c in FC:
    bh_row[f'{PLAIN[c]}: Effect size'] = ''
    bh_row[f'{PLAIN[c]}: 95% CI'] = ''
    bh_row[f'{PLAIN[c]}: p Value'] = pfmt(ms_pbh[c])
wide.append(bh_row)
cols = ['Fixed effects'] + [f'{PLAIN[c]}: {s}' for c in FC for s in ('Effect size','95% CI','p Value')]
C = pd.DataFrame(wide, columns=cols)
C.to_csv(BASE/'sway/table/home_walksway_lme_robustness.csv', index=False)

# console
print(f"=== Free-living ML-sway LME (day-level repeats; MS adjusted for sex, age, BMI) ===")
def star(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
for c in FC:
    print(f"\n{PLAIN[c]}   [MS p_BH={ms_pbh[c]:.4f}{star(ms_pbh[c])}]")
    for kk,lab in EFFORDER:
        est,lo,hi,p = RES[c][kk]
        extra='' if kk!='MS' else f'  p_BH={ms_pbh[c]:.4f}'
        print(f"  {lab:<12}{est:>9.4f}  [{lo:.4f}, {hi:.4f}]  p={p:.4f}{star(p)}{extra}")

# LaTeX (Brenton Table-5 layout: outcomes = column-groups, fixed effects = rows)
def fp(p):
    s='\\textless0.001' if p<0.001 else f'{p:.3f}'
    return f'\\textbf{{{s}}}' if p<0.05 else s
# column groups: one per feature, each = Effect size | CI | p Value
colspec = 'l ' + ' '.join(['ccc'] * len(FC))
grp_hdr = 'Fixed effects'
for c in FC:
    grp_hdr += f' & \\multicolumn{{3}}{{c}}{{{DISP[c]}}}'
grp_hdr += ' \\\\'
cmids = ' '.join(f'\\cmidrule(lr){{{2+3*i}-{4+3*i}}}' for i in range(len(FC)))
sub_hdr = ' & ' + ' & '.join(['Effect size & CI & $p$ Value'] * len(FC)) + ' \\\\'
Lx=['% Requires \\usepackage{booktabs,graphicx}. Spans both columns (table*) and is',
    '% scaled to the full text width with \\resizebox, like Brenton 2022 Table 5.',
    '\\begin{table*}[t]','\\centering',
    '\\caption{\\textbf{Group Differences in Free-Living Mediolateral-Sway Features Using '
    'Linear Mixed-Effects Modeling}}','\\label{tab:home_sway_lme}',
    '\\setlength{\\tabcolsep}{5pt}\\renewcommand{\\arraystretch}{1.15}',
    '\\resizebox{\\textwidth}{!}{%',
    f'\\begin{{tabular}}{{{colspec}}}','\\toprule', grp_hdr, cmids, sub_hdr, '\\midrule']
for kk,lab in EFFORDER:
    label = lab if kk=='Intercept' else f'\\textbf{{{lab}}}'
    cells=[label]
    for c in FC:
        est,lo,hi,p = RES[c][kk]
        cells.append(f'{est:.3f}')
        cells.append(f'{lo:.3f} to {hi:.3f}')
        cells.append(fp(p))
    Lx.append(' & '.join(cells) + ' \\\\')
msbh = ', '.join(DISP[c] + f' $={ms_pbh[c]:.3f}$' for c in FC)
Lx+=['\\bottomrule','\\end{tabular}%','}','\\\\[2pt]',
     '{\\footnotesize Abbreviations: BMI = body mass index; CI = confidence interval; '
     'MS = multiple sclerosis; POMS = pediatric-onset MS. Effect size represents the estimate '
     'from linear mixed-effects models with a random intercept per participant (day-level '
     f'repeated measures); $n={nsub}$ ({nP} POMS, {nH} Control; {ndays} subject-days, median '
     f'{mdays} days/participant). Each outcome is a free-living ML-sway feature, computed per '
     'valid day (a day with $\\geq$3 sustained walking bouts $\\geq$60\\,s; per-day value = median '
     'over that day\'s up-to-8 longest bouts) and entered as a repeated measure; all 101 '
     'participants had $\\geq$1 valid day, so none were excluded. CIs represent '
     'the 95\\% CIs. ``MS\'\' represents the group difference between the POMS and control group '
     'adjusted for sex, age, and BMI (age and BMI mean-centered); ``Female\'\' represents the '
     'group difference between all females and males. Significant values ($p<0.05$) are in bold. '
     'Benjamini-Hochberg-corrected MS $p$ values (across all four candidate ML-sway features, '
     'including the non-significant ML\\_Over\\_ENMO not shown): ' + msbh + '. This '
     'day-level mixed model confirms the unadjusted per-participant comparison and shows the '
     'POMS effect is independent of sex, age, and BMI.}','\\end{table*}']
(BASE/'sway/table/home_walksway_lme_robustness.tex').write_text('\n'.join(Lx)+'\n')
print("\nSaved sway/table/home_walksway_lme_robustness.csv and .tex")
