#!/usr/bin/env python3
"""
Experiment 1: Walking Gait Features + ML Baselines
===================================================
Features: 35 gait features from walking_segments/ (or csv_processed_home fallback)
Models: Ridge, ElasticNet, RF, XGBoost, SVR
Configs: A1, A2, C1, C2
CV: LOO
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch, find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
WALK_DIR = OUT / "walking_segments"
OUT.mkdir(exist_ok=True)
FS = 30.0
BASIC_DEMO = ["cohort_M", "Age", "Sex", "Height", "Weight", "BMI"]
CLINIC_COLS = ["cadence_hz_c","step_time_cv_pct_c","acf_step_regularity_c","hr_ap_c","hr_vt_c",
               "ml_rms_g_c","ml_spectral_entropy_c","jerk_mean_abs_gps_c","enmo_mean_g_c","cadence_slope_per_min_c"]

def load_table():
    home = pd.read_csv(BASE/"sway_features_home.csv").rename(columns={"year_x":"year"})
    clinic = pd.read_csv(BASE/"features_top10.csv")
    cc = {c: c+"_c" for c in clinic.columns if c not in ("cohort","subj_id","year","sixmwd","fs")}
    clinic = clinic.rename(columns=cc).drop(columns=["sixmwd","fs"], errors="ignore")
    p = home[["cohort","subj_id","year","sixmwd"]].merge(clinic, on=["cohort","subj_id","year"], how="inner")
    demo = pd.read_excel(BASE/"SwayDemographics.xlsx")
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    p = p.merge(demo, on=["cohort","subj_id"], how="left")
    p["cohort_M"] = (p["cohort"]=="M").astype(int)
    for c in ["Sex","Age","Height","Weight","BMI"]: p[c] = pd.to_numeric(p[c], errors="coerce")
    return p

def fname(r): return f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"

def extract_gait(sig, fs=FS):
    ap,ml,vt = sig[:,0],sig[:,1],sig[:,2]
    vm = np.sqrt(ap**2+ml**2+vt**2)
    f = {}
    for nm,ax in [("ap",ap),("ml",ml),("vt",vt),("vm",vm)]:
        f[f"{nm}_rms"]=np.sqrt(np.mean(ax**2)); f[f"{nm}_std"]=np.std(ax)
        f[f"{nm}_iqr"]=np.percentile(ax,75)-np.percentile(ax,25)
    f["sma"]=np.mean(np.abs(ap)+np.abs(ml)+np.abs(vt))
    for nm,ax in [("ap",ap),("ml",ml),("vt",vt)]:
        f[f"{nm}_jerk_rms"]=np.sqrt(np.mean((np.diff(ax)*fs)**2))
    for nm,ax in [("vt",vt),("ap",ap)]:
        ac=ax-ax.mean(); acf=np.correlate(ac,ac,"full")[len(ac)-1:]; acf/=(acf[0]+1e-12)
        ml2=min(int(1.5*fs),len(acf)-1); s=acf[int(0.3*fs):ml2]
        pk,pr=find_peaks(s,height=0.0)
        f[f"{nm}_step_reg"]=pr["peak_heights"][0] if len(pk)>=1 else 0
        f[f"{nm}_step_time"]=(pk[0]+int(0.3*fs))/fs if len(pk)>=1 else 0
        f[f"{nm}_stride_reg"]=pr["peak_heights"][1] if len(pk)>=2 else 0
        f[f"{nm}_step_sym"]=f[f"{nm}_step_reg"]/(f[f"{nm}_stride_reg"]+1e-8) if len(pk)>=2 else 1
    for nm,ax in [("vt",vt),("ap",ap),("ml",ml)]:
        if len(ax)>64:
            fr,ps=welch(ax,fs=fs,nperseg=min(256,len(ax))); gb=(fr>=0.5)&(fr<=3.5)
            f[f"{nm}_dom_freq"]=fr[gb][np.argmax(ps[gb])] if gb.any() else 0
            gp=np.trapz(ps[gb],fr[gb]) if gb.any() else 0; tp=np.trapz(ps,fr)+1e-12
            f[f"{nm}_gait_pwr"]=gp/tp
            pn=ps/(ps.sum()+1e-12); pn=pn[pn>0]; f[f"{nm}_spec_ent"]=-np.sum(pn*np.log2(pn+1e-12))
        else: f[f"{nm}_dom_freq"]=0; f[f"{nm}_gait_pwr"]=0; f[f"{nm}_spec_ent"]=0
    fund=f.get("vt_dom_freq",0)
    if fund>0:
        fv=np.abs(rfft(vt)); ff=rfftfreq(len(vt),1/fs); ep,op=0,0
        for h in range(1,11):
            idx=np.argmin(np.abs(ff-h*fund))
            if h%2==0: ep+=fv[idx]**2
            else: op+=fv[idx]**2
        f["vt_hr"]=ep/(op+1e-12)
    else: f["vt_hr"]=0
    w2=int(2*fs); rw=np.array([np.sqrt(np.mean(vm[i:i+w2]**2)) for i in range(0,max(1,len(vm)-w2+1),w2)])
    f["act_rms_cv"]=rw.std()/(rw.mean()+1e-12) if len(rw)>1 else 0
    return f

def loo(X,y,mfn):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(X[tr]),y[tr]); p[te]=m.predict(sc.transform(X[te]))
    return p

def loo_fs(X,y,mfn,k=15):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        cors=np.array([abs(spearmanr(X[tr,j],y[tr])[0]) for j in range(X.shape[1])])
        idx=np.argsort(cors)[-k:]
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(X[tr][:,idx]),y[tr]); p[te]=m.predict(sc.transform(X[te][:,idx]))
    return p

def met(y,yh):
    return {"R2":round(r2_score(y,yh),4),"MAE":round(mean_absolute_error(y,yh),1),
            "r":round(pearsonr(y,yh)[0],4),"rho":round(spearmanr(y,yh)[0],4)}

def main():
    print("="*60)
    print("Exp 1: Walking Gait Features + ML")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)
    print(f"  {n} subjects")

    X_clinic = paired[CLINIC_COLS].values.astype(float)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])

    # Load walking signals and extract features
    print("  Extracting gait features...")
    rows = []
    for _,r in paired.iterrows():
        fn = fname(r)
        wp = WALK_DIR/fn
        if wp.exists():
            sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
        else:
            sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
        rows.append(extract_gait(sig))
    X = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    for c in X.columns:
        if X[c].isna().any(): X[c]=X[c].fillna(X[c].median())
    X = X.values.astype(float)
    print(f"  {X.shape[1]} features")

    results = []
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
    models = {"Ridge":lambda:Ridge(alpha=10),"ElasticNet":lambda:ElasticNet(alpha=1,l1_ratio=0.5,max_iter=10000),
              "RF":lambda:RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=-1),
              "XGBoost":xgb,"SVR":lambda:SVR(kernel="rbf",C=100,epsilon=50)}

    print("\n  Running LOO CV...")
    for mname,mfn in models.items():
        for cfg,Xin,desc in [
            ("A1", None, "Home gait (feat sel)"),
            ("A2", np.column_stack([X,X_demo]), "Home gait + demo"),
            ("C1", X_clinic, "Clinic accel"),
            ("C2", np.column_stack([X_clinic,X_demo]), "Clinic accel + demo"),
        ]:
            if cfg == "A1":
                p = loo_fs(X, y, mfn, k=15)
            else:
                p = loo(Xin, y, mfn)
            m = met(y, p)
            results.append({"model":mname,"config":cfg,"desc":desc,**m})
            print(f"    {mname:15s} {cfg:4s} R²={m['R2']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp1_gait_ml.csv", index=False)
    print("\n" + "="*60)
    pivot = df.pivot_table(index="model",columns="config",values="R2",aggfunc="first")
    print(pivot[["A1","A2","C1","C2"]].to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/exp1_gait_ml.csv")

if __name__ == "__main__":
    main()
