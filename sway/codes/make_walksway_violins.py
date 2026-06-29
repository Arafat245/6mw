import sys, re, warnings, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from scipy.signal import welch
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore'); BASE=Path('/mnt/sdb/arafat/6mw'); sys.path.insert(0,str(BASE))
from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios
from clinic.extract_walking_sway import extract_walking_sway
PRE=BASE/'csv_preprocessed2'; FS=30.0
fre=re.compile(r'^(?P<c>[CM])(?P<id>\d+)_(?P<yr>\d+)_(?P<d>\d+)\.csv$',re.I)
def bandpow(x,lo,hi):
    f,Pp=welch(x-np.mean(x),fs=FS,nperseg=min(1024,len(x))); b=(f>=lo)&(f<=hi); return np.sum(Pp[b])

ids,gait_rows,ws_rows,new_rows=[],[],[],[]
for p in sorted(PRE.glob('*.csv')):
    m=fre.match(p.name)
    if not m: continue
    df=pd.read_csv(p); ap,ml,vt=df['AP'].values,df['ML'].values,df['VT'].values
    ids.append({'cohort':m.group('c').upper(),'subj_id':int(m.group('id')),'sixmwd':int(m.group('d'))})
    gait_rows.append(extract_gait10(df)); ws_rows.append(extract_walking_sway(ap,ml,vt))
    vAP,vML,vVT=np.var(ap),np.var(ml),np.var(vt); mlP=bandpow(ml,0.3,10); apP=bandpow(ap,0.3,10)
    new_rows.append({'ml_energy_frac':vML/(vAP+vML+vVT+1e-12),'ml_spec_horiz_frac':mlP/(mlP+apP+1e-12)})
ids=pd.DataFrame(ids)
gm=add_sway_ratios(pd.concat([ids.reset_index(drop=True),pd.DataFrame(gait_rows)],axis=1)
                   .merge(compute_vt_rms(PRE),on=['cohort','subj_id','sixmwd'],how='left'))
W=pd.DataFrame(ws_rows); W['ml_over_enmo']=gm['ml_over_enmo'].values; W['ml_over_vt']=gm['ml_over_vt'].values
D=pd.concat([W,pd.DataFrame(new_rows)],axis=1)
D['cohort']=np.where(ids['cohort'].values=='M','POMS','Healthy')

CLR_HEALTHY='#6BAED6'; CLR_POMS='#FC8D62'; palette={'Healthy':CLR_HEALTHY,'POMS':CLR_POMS}
LBL,TIT,SUP,TICK=13,14,15,12
# (col, ylabel_formula, bh_key, panel_letter, construct_title, display_name, unit)
# All four are ratios of like-dimensioned quantities -> dimensionless (no physical unit).
feats=[('ml_over_enmo',     'ML RMS / ENMO',                 'ml_over_enmo',     'a','ML Sway per Effort','ML_Over_ENMO','dimensionless'),
       ('ml_over_vt',       'ML RMS / VT RMS',               'ml_over_vt',       'b','ML vs Vertical Sway','ML_Over_VT','dimensionless'),
       ('ml_spec_horiz_frac',r'P$_{ML}$ / (P$_{ML}$+P$_{AP}$)','ml_spec_horiz_frac','d','ML Spectral Fraction','ML_Spec_Frac','dimensionless')]

fig,axes=plt.subplots(1,len(feats),figsize=(5*len(feats),6.2))
fig.suptitle('Clinic 6MWT Mediolateral-Sway Features by Cohort  (n=120: 50 POMS, 70 Control)',
             fontsize=SUP,fontweight='bold',y=0.99)
# Raw (uncorrected) two-sided Mann-Whitney p-values, following Brenton et al.
# (2022) who defined significance at p<0.05 with no multiple-comparison correction.
allcols=list(W.columns)+['ml_energy_frac','ml_spec_horiz_frac']
praw=dict(zip(allcols,[mannwhitneyu(D.loc[D.cohort=='POMS',c],D.loc[D.cohort=='Healthy',c],alternative='two-sided')[1] for c in allcols]))

for ax,(col,ylab,key,letter,construct,disp,unit) in zip(axes,feats):
    sub=D[['cohort',col]].copy()
    sns.violinplot(data=sub,x='cohort',y=col,order=['Healthy','POMS'],palette=palette,
                   inner=None,cut=0,linewidth=1.2,saturation=0.9,width=0.85,ax=ax)
    for coll in ax.collections: coll.set_alpha(0.55); coll.set_edgecolor('none')
    sns.stripplot(data=sub,x='cohort',y=col,order=['Healthy','POMS'],palette=palette,
                  size=4.2,jitter=0.18,alpha=0.9,edgecolor='white',linewidth=0.6,ax=ax)
    for i,coh in enumerate(['Healthy','POMS']):
        med=float(sub.loc[sub.cohort==coh,col].median())
        ax.plot([i-0.22,i+0.22],[med,med],color='black',linewidth=2.0,zorder=5)
    ax.set_xticklabels(['Control','POMS'],fontsize=TICK,fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel(ylab,fontsize=LBL,fontweight='bold')
    ax.set_title(disp,fontsize=TIT,fontweight='bold',pad=8,family='monospace')
    ax.tick_params(axis='y',labelsize=TICK)
    ax.grid(True,axis='y',which='both',alpha=0.25,linestyle='--',linewidth=0.5); ax.set_axisbelow(True)
    q=praw[key]; stars='***' if q<0.001 else '**' if q<0.01 else '*' if q<0.05 else 'ns'
    ymin,ymax=sub[col].min(),sub[col].max(); rng=ymax-ymin
    ybar=ymax+rng*0.06; ytxt=ymax+rng*0.10
    ax.plot([0,0,1,1],[ybar-rng*0.015,ybar,ybar,ybar-rng*0.015],color='black',linewidth=1.1)
    ax.text(0.5,ytxt,f'{stars}  (p = {q:.1e})',ha='center',va='bottom',
            fontsize=LBL-1,fontweight='bold')
    ax.set_ylim(top=ymax+rng*0.22)

plt.tight_layout(rect=[0,0,1,0.97])
OUT=BASE/'sway'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
fp=OUT/'walksway_significant_violins.png'
plt.savefig(fp,dpi=200,bbox_inches='tight'); print('Saved',fp)
