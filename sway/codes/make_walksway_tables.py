import sys, re, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
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
    gait_rows.append(extract_gait10(df))
    ws_rows.append(extract_walking_sway(ap,ml,vt))
    vAP,vML,vVT=np.var(ap),np.var(ml),np.var(vt); mlP=bandpow(ml,0.3,10); apP=bandpow(ap,0.3,10)
    new_rows.append({'ml_energy_frac':vML/(vAP+vML+vVT+1e-12),
                     'ml_spec_horiz_frac':mlP/(mlP+apP+1e-12)})
ids=pd.DataFrame(ids); poms=(ids['cohort']=='M').values; heal=~poms
gm=pd.concat([ids.reset_index(drop=True),pd.DataFrame(gait_rows)],axis=1)
gm=add_sway_ratios(gm.merge(compute_vt_rms(PRE),on=['cohort','subj_id','sixmwd'],how='left'))

W=pd.DataFrame(ws_rows); W['ml_over_enmo']=gm['ml_over_enmo'].values; W['ml_over_vt']=gm['ml_over_vt'].values
WN=pd.concat([W,pd.DataFrame(new_rows)],axis=1)        # 12 walksway + 2 new = 14
ws_names=list(WN.columns); X=WN.values.astype(float)
X=np.where(np.isnan(X),np.nanmedian(X,0),X)
print(f"WalkSway family = {len(ws_names)} features (12 original + 2 new) | POMS={poms.sum()} Healthy={heal.sum()}\n")

pr=[mannwhitneyu(X[poms,j],X[heal,j],alternative='two-sided')[1] for j in range(X.shape[1])]
q=multipletests(pr,method='fdr_bh')[1]
rng=np.random.default_rng(42)
def med_ci(x,B=2000):
    x=x[~np.isnan(x)]; b=np.median(x[rng.integers(0,len(x),(B,len(x)))],axis=1)
    return np.median(x),np.percentile(b,2.5),np.percentile(b,97.5)
def cohens_d(a,b):
    a,b=a[~np.isnan(a)],b[~np.isnan(b)]; n1,n2=len(a),len(b)
    s=np.sqrt(((n1-1)*np.var(a,ddof=1)+(n2-1)*np.var(b,ddof=1))/(n1+n2-2)); return (a.mean()-b.mean())/s if s>0 else 0
def stars(v): return '***' if v<0.001 else '**' if v<0.01 else '*' if v<0.05 else 'ns'

rows=[]
for j,f in enumerate(ws_names):
    pm,pl,pu=med_ci(X[poms,j]); hm,hl,hu=med_ci(X[heal,j])
    rows.append([f,'new' if f in('ml_energy_frac','ml_spec_horiz_frac') else 'orig',
                 pm,pl,pu,hm,hl,hu,round(cohens_d(X[poms,j],X[heal,j]),3),pr[j],q[j],stars(q[j])])
out=pd.DataFrame(rows,columns=['Feature','Source','POMS_med','POMS_lo','POMS_hi',
    'Healthy_med','Healthy_lo','Healthy_hi','Cohen_d','p_raw','p_BH','sig'])
# final deliverable: significant (BH<0.05) WalkSway features, EXCLUDING ap_range_norm
final=out[(out.p_BH<0.05)&(out.Feature!='ap_range_norm')].sort_values('p_BH')
out.to_csv('/mnt/sdb/arafat/6mw/sway/table/walksway_all14_ms_vs_healthy.csv',index=False)
final.to_csv('/mnt/sdb/arafat/6mw/sway/table/walksway_significant_features.csv',index=False)

print("=== ALL 14 (raw p, BH p over 14-feature family) ===")
for _,r in out.sort_values('p_raw').iterrows():
    print(f"{r.Feature:<20}{r.Source:<5} d={r.Cohen_d:+.2f}  raw={r.p_raw:.4f}  BH={r.p_BH:.4f}  {r.sig}")
print("\n=== FINAL: significant WalkSway features (BH<0.05), excluding ap_range_norm ===")
for _,r in final.iterrows():
    print(f"{r.Feature:<20}({r.Source}) d={r.Cohen_d:+.2f} BH={r.p_BH:.4f} {r.sig}")
