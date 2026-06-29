#!/usr/bin/env python3
"""
Brenton-2022 Table 3 / Figure 2 analog, using ML_Over_ENMO instead of 6MW gait speed.

Brenton modeled minute-by-minute 6MW gait speed with an LME (Time, Time^2, MS,
BMI category, Age, Female, MS x Time interactions) and plotted the modeled
trajectories by BMI category for Control vs MS (Figure 2).

Here the within-walk "trajectory" is ML_Over_ENMO computed in 6 equal epochs of
the clinic 6MWT (epoch ~57 s ~= 1 minute). This gives repeated measures per
subject, so a genuine LME (random intercept per subject) applies.

Model:  ML_Over_ENMO ~ MS + t + t2 + Overweight + Obese + Age_c + Female
                       + MS:t + MS:t2,   (1 | subject)
t = epoch index 0..5 (so intercept = first epoch); Age mean-centered;
reference BMI = healthy weight; Female = Sex==2.  n=120 (50 POMS, 70 Healthy).
"""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
BASE=Path('/mnt/sdb/arafat/6mw'); import sys; sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10
PRE=BASE/'csv_preprocessed2'
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_',re.I)
N_EPOCH=6

# --- demographics ---
demo=pd.read_excel(BASE/'SwayDemographics.xlsx')
def key_id(s):
    m=re.match(r'\s*([CM])-?(\d+)',str(s),re.I); return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None
demo['key']=demo['ID'].apply(key_id)
dem=demo.set_index('key')

# --- per-epoch ML_Over_ENMO trajectory ---
rows=[]
for p in sorted(PRE.glob('*.csv')):
    m=fre.match(p.name)
    if not m: continue
    key=f"{m.group('c').upper()}{int(m.group('id')):02d}"
    if key not in dem.index: continue
    df=pd.read_csv(p); n=len(df); bnd=np.linspace(0,n,N_EPOCH+1).astype(int)
    for e in range(N_EPOCH):
        seg=df.iloc[bnd[e]:bnd[e+1]]
        if len(seg)<30: continue
        g=extract_gait10(seg); val=g['ml_rms_g']/g['enmo_mean_g'] if g['enmo_mean_g'] else np.nan
        rows.append({'key':key,'cohort':m.group('c').upper(),'t':e,'ML_Over_ENMO':val,
                     'Age':dem.loc[key,'Age'],'Sex':dem.loc[key,'Sex'],'BMICat':dem.loc[key,'BMI Cat']})
L=pd.DataFrame(rows).dropna(subset=['ML_Over_ENMO'])
L['MS']=(L['cohort']=='M').astype(int)
L['Female']=(L['Sex']==2).astype(int)
L['Overweight']=(L['BMICat']==3).astype(int)
L['Obese']=(L['BMICat']==4).astype(int)
L['Age_c']=L['Age']-L['Age'].mean()
L['t2']=L['t']**2
print(f"rows={len(L)}  subjects={L.key.nunique()}  POMS={L[L.MS==1].key.nunique()}  HC={L[L.MS==0].key.nunique()}\n")

# --- LME (Table 3 analog) ---
fit=smf.mixedlm("ML_Over_ENMO ~ MS + t + t2 + Overweight + Obese + Age_c + Female + MS:t + MS:t2",
                L, groups=L['key']).fit(method='lbfgs')
order=[('Intercept','(Intercept)'),('MS','MS'),('t','Time'),('t2','Time^2'),
       ('Overweight','Overweight'),('Obese','Obese'),('Age_c','Age'),('Female','Female'),
       ('MS:t','MS x Time'),('MS:t2','MS x Time^2')]
ci=fit.conf_int()
out=[]
for kk,lab in order:
    out.append({'Fixed effects':lab,'Estimate':round(fit.params[kk],4),
                '95% CI':f'{ci.loc[kk,0]:.4f} to {ci.loc[kk,1]:.4f}','p Value':round(fit.pvalues[kk],4)})
T3=pd.DataFrame(out)
OUT_T=BASE/'sway'/'table'; T3.to_csv(OUT_T/'walksway_ML_Over_ENMO_trajectory_LME.csv',index=False)
def star(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
print("=== Table 3 analog: LME of ML_Over_ENMO trajectory ===")
print(f"{'Fixed effects':<14}{'Estimate':>10}{'95% CI':>26}{'p':>10}")
for r in out:
    print(f"{r['Fixed effects']:<14}{r['Estimate']:>10.4f}{('  ['+r['95% CI'].replace(' to ',', ')+']'):>26}{r['p Value']:>9.4f}{star(r['p Value'])}")
print("Saved",OUT_T/'walksway_ML_Over_ENMO_trajectory_LME.csv')

# --- Figure 2 analog: modeled trajectories by BMI cat, Control vs MS ---
CLR={'Healthy':'#1f77b4','Overweight':'#e377c2','Obese':'#2ca02c'}
grid=[]
for ms in [0,1]:
    for cat,(ov,ob) in {'Healthy':(0,0),'Overweight':(1,0),'Obese':(0,1)}.items():
        for t in range(N_EPOCH):
            grid.append({'MS':ms,'t':t,'t2':t**2,'Overweight':ov,'Obese':ob,
                         'Age_c':0.0,'Female':0,'cat':cat})
G=pd.DataFrame(grid); G['pred']=fit.predict(G)
fig,axes=plt.subplots(1,2,figsize=(12,5.2),sharey=True)
for ax,(ms,name) in zip(axes,[(0,'Control'),(1,'MS')]):
    for cat in ['Healthy','Overweight','Obese']:
        g=G[(G.MS==ms)&(G.cat==cat)]
        ax.plot(g['t']+1,g['pred'],marker='o',color=CLR[cat],linewidth=2.4,markersize=6,label=cat)
    ax.set_title(name,fontsize=14,fontweight='bold')
    ax.set_xlabel('6MWT epoch (~1 min)',fontsize=12,fontweight='bold')
    ax.grid(True,alpha=0.25,linestyle='--',linewidth=0.5); ax.set_axisbelow(True)
    ax.set_xticks(range(1,N_EPOCH+1))
axes[0].set_ylabel('ML_Over_ENMO  (modeled)',fontsize=12,fontweight='bold')
axes[1].legend(title='BMI',fontsize=11,title_fontsize=11)
fig.suptitle('Modeled ML_Over_ENMO Trajectory over the 6MWT by BMI Category — Control vs POMS\n'
             '(LME, male reference, age at mean; n=120)',fontsize=13,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
OUT_F=BASE/'sway'/'figures'; fp=OUT_F/'walksway_ML_Over_ENMO_trajectory.png'
plt.savefig(fp,dpi=200,bbox_inches='tight'); print("Saved",fp)
