#!/usr/bin/env python3
"""
Experiment 2: Activity Profile Features + ML
=============================================
Features: ~15 activity-level features from daytime_segments/
Models: Ridge, ElasticNet, RF, XGBoost, SVR
Configs: A1, A2 (home only — activity profiles don't apply to clinic 6MW test)
CV: LOO
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
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
DAY_DIR = OUT / "daytime_segments"
BASIC_DEMO = ["cohort_M","Age","Sex","Height","Weight","BMI"]
FS = 30.0

def load_table():
    home = pd.read_csv(BASE/"sway_features_home.csv").rename(columns={"year_x":"year"})
    demo = pd.read_excel(BASE/"SwayDemographics.xlsx")
    demo["cohort"]=demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"]=demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    p = home[["cohort","subj_id","year","sixmwd"]].merge(demo,on=["cohort","subj_id"],how="left")
    p["cohort_M"]=(p["cohort"]=="M").astype(int)
    for c in ["Sex","Age","Height","Weight","BMI"]: p[c]=pd.to_numeric(p[c],errors="coerce")
    return p

def fname(r): return f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"

def extract_activity(xyz, fs=FS):
    vm = np.sqrt(xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2)
    bs = int(fs); n_bins = len(vm)//bs
    f = {}
    if n_bins < 10:
        return {k:0 for k in ["ai_mean","ai_std","ai_iqr","ai_entropy","ai_pct_low","ai_pct_mod",
                               "ai_pct_high","ai_cv","bout_count","bout_mean_dur","bout_total_dur",
                               "bout_mean_accel","bout_dur_cv","total_hours","walk_pct"]}
    ai = np.array([np.sqrt(np.mean(vm[i*bs:(i+1)*bs]**2)) for i in range(n_bins)])
    f["ai_mean"]=np.mean(ai); f["ai_std"]=np.std(ai)
    f["ai_iqr"]=np.percentile(ai,75)-np.percentile(ai,25)
    hist,_=np.histogram(ai,bins=20,density=True); hist=hist[hist>0]; hist=hist/hist.sum()
    f["ai_entropy"]=-np.sum(hist*np.log2(hist+1e-12))
    lo,hi=np.percentile(ai,30),np.percentile(ai,85)
    f["ai_pct_low"]=np.mean(ai<lo); f["ai_pct_mod"]=np.mean((ai>=lo)&(ai<hi)); f["ai_pct_high"]=np.mean(ai>=hi)
    f["ai_cv"]=np.std(ai)/(np.mean(ai)+1e-12)
    active=(ai>lo)&(ai<hi)
    bouts=[]; in_b,bstart=False,0
    for i in range(len(active)):
        if active[i] and not in_b: bstart=i; in_b=True
        elif not active[i] and in_b:
            if (i-bstart)>=5: bouts.append((bstart,i))
            in_b=False
    if in_b and (len(active)-bstart)>=5: bouts.append((bstart,len(active)))
    f["bout_count"]=len(bouts)
    if bouts:
        durs=[(e-s) for s,e in bouts]; accels=[np.mean(ai[s:e]) for s,e in bouts]
        f["bout_mean_dur"]=np.mean(durs); f["bout_total_dur"]=np.sum(durs)
        f["bout_mean_accel"]=np.mean(accels); f["bout_dur_cv"]=np.std(durs)/(np.mean(durs)+1e-12)
    else:
        f["bout_mean_dur"]=0; f["bout_total_dur"]=0; f["bout_mean_accel"]=0; f["bout_dur_cv"]=0
    f["total_hours"]=len(ai)/3600; f["walk_pct"]=f["bout_total_dur"]/(len(ai)+1e-12)
    return f

def loo(X,y,mfn):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(X[tr]),y[tr]); p[te]=m.predict(sc.transform(X[te]))
    return p

def met(y,yh):
    return {"R2":round(r2_score(y,yh),4),"MAE":round(mean_absolute_error(y,yh),1),
            "r":round(pearsonr(y,yh)[0],4),"rho":round(spearmanr(y,yh)[0],4)}

def main():
    print("="*60)
    print("Exp 2: Activity Profile Features + ML")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])

    print("  Extracting activity features...")
    rows = []
    for _,r in paired.iterrows():
        fn = fname(r)
        dp = DAY_DIR/fn
        if dp.exists():
            sig = pd.read_csv(dp)[["X","Y","Z"]].values.astype(np.float32)
        else:
            # Fallback: use walking segment as proxy
            wp = OUT/"walking_segments"/fn
            if wp.exists():
                sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
            else:
                sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
        rows.append(extract_activity(sig))
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
        p = loo(X, y, mfn)
        m = met(y,p); results.append({"model":mname,"config":"A1",**m})
        print(f"    {mname:15s} A1   R²={m['R2']:.4f}")

        p = loo(np.column_stack([X,X_demo]), y, mfn)
        m = met(y,p); results.append({"model":mname,"config":"A2",**m})
        print(f"    {mname:15s} A2   R²={m['R2']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp2_activity_ml.csv", index=False)
    print("\n" + "="*60)
    pivot = df.pivot_table(index="model",columns="config",values="R2",aggfunc="first")
    print(pivot.to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/exp2_activity_ml.csv")

if __name__ == "__main__":
    main()
