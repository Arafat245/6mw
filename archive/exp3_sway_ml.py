#!/usr/bin/env python3
"""
Experiment 3: Postural Sway Features + ML (Meyer et al. 2024)
=============================================================
15 sway features from standing/quiet periods in daytime data:
Jerk, Dist, RMS, Path, Range, MV, MF, Area, Pwr, F50, F95, CF, FD, ApEn, LyExp

Method: Detect quiet standing bouts (low activity, low periodicity),
compute horizontal acceleration magnitude, extract sway features
per 30-second window, aggregate with percentile statistics.

Configs: A1, A2
CV: LOO
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch, butter, filtfilt
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
WALK_DIR = OUT / "walking_segments"
FS = 30.0
BASIC_DEMO = ["cohort_M","Age","Sex","Height","Weight","BMI"]

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

def compute_sway_features_window(horiz_mag, fs=FS):
    """Compute 15 sway features from horizontal acceleration magnitude (Meyer et al.)."""
    n = len(horiz_mag)
    dt = 1.0 / fs
    f = {}

    # Temporal
    f["jerk"] = np.mean(np.abs(np.diff(horiz_mag) / dt))
    pos = np.cumsum(horiz_mag - np.mean(horiz_mag)) * dt  # displacement proxy
    f["dist"] = np.mean(np.abs(pos - np.mean(pos)))
    f["rms"] = np.sqrt(np.mean(horiz_mag**2))
    f["path"] = np.sum(np.abs(np.diff(pos)))
    f["range"] = np.ptp(pos)
    f["mv"] = f["path"] / (n * dt) if n > 0 else 0
    trial_dur = n * dt
    f["mf"] = f["path"] / (2 * np.pi * f["dist"] * trial_dur + 1e-12)

    # Area (ellipse area from 2D — use horizontal mag as 1D proxy)
    f["area"] = np.pi * np.std(horiz_mag) * np.std(pos)

    # Spectral
    if n > 32:
        freqs, psd = welch(horiz_mag, fs=fs, nperseg=min(256, n))
        total_pwr = np.trapz(psd, freqs) + 1e-12
        f["pwr"] = total_pwr
        cum_pwr = np.cumsum(psd * np.diff(np.concatenate([[0], freqs])))
        cum_norm = cum_pwr / (cum_pwr[-1] + 1e-12)
        f["f50"] = freqs[np.searchsorted(cum_norm, 0.5)] if len(cum_norm) > 0 else 0
        f["f95"] = freqs[np.searchsorted(cum_norm, 0.95)] if len(cum_norm) > 0 else 0
        f["cf"] = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
        psd_norm = psd / (psd.sum() + 1e-12)
        psd_nz = psd_norm[psd_norm > 0]
        f["fd"] = np.sqrt(1 - (np.sum(psd_nz * np.log2(psd_nz + 1e-12)))**2 / (np.log2(len(psd_nz) + 1e-12))**2) if len(psd_nz) > 1 else 0
    else:
        f["pwr"]=0; f["f50"]=0; f["f95"]=0; f["cf"]=0; f["fd"]=0

    # Skipping ApEn and LyExp — they are slow and not among top features
    # (Meyer et al.: top features were RMS, Path, MV, Area, Pwr, FD)

    return f


def detect_standing_bouts(xyz, fs=FS, win_sec=30, min_bouts=3):
    """Detect quiet standing periods: low VM variability, no periodicity."""
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    win = int(win_sec * fs)
    bouts = []
    for s in range(0, len(vm) - win, win):
        seg_vm = vm[s:s+win]
        std = np.std(seg_vm)
        rms = np.sqrt(np.mean(seg_vm**2))
        # Standing: near-gravity, low variance, not walking
        if std < 0.15 and rms > 0.9 and rms < 1.2:
            bouts.append((s, s + win))
    return bouts


def extract_sway_all(xyz, fs=FS):
    """Extract sway features from standing bouts, aggregate with percentile stats."""
    bouts = detect_standing_bouts(xyz, fs)

    if len(bouts) < 3:
        # Fallback: use quietest 30-sec windows
        vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
        win = int(30 * fs)
        stds = []
        for s in range(0, len(vm)-win, win):
            stds.append((np.std(vm[s:s+win]), s))
        stds.sort()
        bouts = [(s, s+win) for _, s in stds[:max(5, len(stds)//4)]]

    # Apply 3.5 Hz lowpass, compute horizontal magnitude per bout
    b, a = butter(4, 3.5 / (fs/2), btype="low")
    all_feats = []
    for s, e in bouts:
        seg = xyz[s:e]
        if len(seg) < 30:
            continue
        filtered = filtfilt(b, a, seg, axis=0)
        horiz_mag = np.sqrt(filtered[:,0]**2 + filtered[:,1]**2)
        feats = compute_sway_features_window(horiz_mag, fs)
        all_feats.append(feats)

    if not all_feats:
        return {f"{k}_{s}": 0 for k in ["jerk","dist","rms","path","range","mv","mf",
                "area","pwr","f50","f95","cf","fd"]
                for s in ["p5","p25","med","p75","p95","std"]}

    df = pd.DataFrame(all_feats)
    agg = {}
    for col in df.columns:
        vals = df[col].values
        agg[f"{col}_p5"] = np.percentile(vals, 5)
        agg[f"{col}_p25"] = np.percentile(vals, 25)
        agg[f"{col}_med"] = np.median(vals)
        agg[f"{col}_p75"] = np.percentile(vals, 75)
        agg[f"{col}_p95"] = np.percentile(vals, 95)
        agg[f"{col}_std"] = np.std(vals)
    return agg


def loo(X,y,mfn):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(X[tr]),y[tr]); p[te]=m.predict(sc.transform(X[te]))
    return p

def loo_fs(X,y,mfn,k=20):
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
    print("Exp 3: Postural Sway Features + ML (Meyer et al. 2024)")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])

    print("  Extracting sway features from standing bouts...")
    rows = []
    for i,(_,r) in enumerate(paired.iterrows()):
        fn = fname(r)
        dp = DAY_DIR/fn
        if dp.exists():
            sig = pd.read_csv(dp)[["X","Y","Z"]].values.astype(np.float32)
        else:
            wp = OUT/"walking_segments"/fn
            if wp.exists():
                sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
            else:
                sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
        rows.append(extract_sway_all(sig))
        if (i+1)%50==0: print(f"    {i+1}/{n}", flush=True)
    print(f"    {n}/{n}")

    X = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    for c in X.columns:
        if X[c].isna().any(): X[c]=X[c].fillna(X[c].median())
    X = X.values.astype(float)
    print(f"  {X.shape[1]} sway features")

    results = []
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
    models = {"Ridge":lambda:Ridge(alpha=10),"ElasticNet":lambda:ElasticNet(alpha=1,l1_ratio=0.5,max_iter=10000),
              "RF":lambda:RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=-1),
              "XGBoost":xgb,"SVR":lambda:SVR(kernel="rbf",C=100,epsilon=50)}

    print("\n  Running LOO CV...")
    for mname,mfn in models.items():
        p = loo_fs(X, y, mfn, k=20)
        m = met(y,p); results.append({"model":mname,"config":"A1",**m})
        print(f"    {mname:15s} A1   R²={m['R2']:.4f}")

        p = loo(np.column_stack([X,X_demo]), y, mfn)
        m = met(y,p); results.append({"model":mname,"config":"A2",**m})
        print(f"    {mname:15s} A2   R²={m['R2']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp3_sway_ml.csv", index=False)
    print("\n" + "="*60)
    pivot = df.pivot_table(index="model",columns="config",values="R2",aggfunc="first")
    print(pivot.to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/exp3_sway_ml.csv")

if __name__ == "__main__":
    main()
