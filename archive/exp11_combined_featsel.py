#!/usr/bin/env python3
"""
Experiment 11: Combined Best Features + Feature Selection
=========================================================
Combine useful features from all previous experiments:
  - Activity Profile (15) — best A1/A2
  - Gait (35) — best C1
  - CWT temporal dynamics (28) — best C2 boost

Then apply per-fold feature selection methods to reduce redundancy:
  1. Spearman correlation with 6MWD (relevancy)
  2. Correlation filter (remove inter-feature redundancy > 0.9)
  3. Combined: filter redundancy first, then select top-k by relevancy

Configs: A1, A2, C1, C2
CV: LOO
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import pywt
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
WALK_DIR = OUT / "walking_segments"
DAY_DIR = OUT / "daytime_segments"
FS = 30.0
BASIC_DEMO = ["cohort_M","Age","Sex","Height","Weight","BMI"]
CLINIC_COLS = ["cadence_hz_c","step_time_cv_pct_c","acf_step_regularity_c","hr_ap_c","hr_vt_c",
               "ml_rms_g_c","ml_spectral_entropy_c","jerk_mean_abs_gps_c","enmo_mean_g_c","cadence_slope_per_min_c"]

def load_table():
    home = pd.read_csv(BASE/"sway_features_home.csv").rename(columns={"year_x":"year"})
    clinic = pd.read_csv(BASE/"features_top10.csv")
    cc = {c:c+"_c" for c in clinic.columns if c not in ("cohort","subj_id","year","sixmwd","fs")}
    clinic = clinic.rename(columns=cc).drop(columns=["sixmwd","fs"],errors="ignore")
    p = home[["cohort","subj_id","year","sixmwd"]].merge(clinic,on=["cohort","subj_id","year"],how="inner")
    demo = pd.read_excel(BASE/"SwayDemographics.xlsx")
    demo["cohort"]=demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"]=demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    p = p.merge(demo,on=["cohort","subj_id"],how="left")
    p["cohort_M"]=(p["cohort"]=="M").astype(int)
    for c in ["Sex","Age","Height","Weight","BMI"]: p[c]=pd.to_numeric(p[c],errors="coerce")
    return p

def fname(r): return f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"

# Reuse feature extractors
from exp1_gait_ml import extract_gait
from exp2_activity_ml import extract_activity

def compute_cwt_power(sig_1d, fs, freqs):
    scales = fs / (freqs + 1e-12)
    coeffs, _ = pywt.cwt(sig_1d, scales, 'morl', sampling_period=1.0/fs)
    return np.abs(coeffs)**2

def extract_cwt_segment(sig, fs=FS):
    vm = np.sqrt(sig[:,0]**2+sig[:,1]**2+sig[:,2]**2)
    vm_n = vm - vm.mean(); vm_n = vm_n/(np.max(np.abs(vm_n))+1e-12)
    freqs = np.linspace(0.5,12,50)
    power = compute_cwt_power(vm_n, fs, freqs)
    mp = power.mean(axis=1)
    f = {}
    f["cwt_mean_energy"]=np.mean(power)
    hm=freqs>=3.5; f["cwt_high_freq_energy"]=np.mean(power[hm]) if hm.any() else 0
    f["cwt_dominant_freq"]=freqs[np.argmax(mp)]
    gm=(freqs>=0.5)&(freqs<=3.5); gp=mp.copy(); gp[~gm]=0
    f["cwt_cadence"]=freqs[np.argmax(gp)]*60
    nw=max(1,power.shape[1]//int(fs))
    df=[freqs[np.argmax(power[:,w*int(fs):min((w+1)*int(fs),power.shape[1])].mean(1))] for w in range(nw)]
    f["cwt_freq_var"]=np.std(df); f["cwt_freq_cv"]=np.std(df)/(np.mean(df)+1e-12)
    pn=mp/(mp.sum()+1e-12); pnz=pn[pn>0]
    f["cwt_entropy"]=-np.sum(pnz*np.log2(pnz+1e-12))
    fv=np.abs(rfft(vm_n)); ff=rfftfreq(len(vm_n),1/fs)
    gb=(ff>=0.5)&(ff<=3.5)
    if gb.any():
        f0=ff[gb][np.argmax(fv[gb])]; f["cwt_f0"]=f0
        if f0>0:
            ep,op=0,0
            for h in range(1,11):
                idx=np.argmin(np.abs(ff-h*f0))
                if h%2==0: ep+=fv[idx]**2
                else: op+=fv[idx]**2
            f["cwt_hr"]=ep/(op+1e-12)
        else: f["cwt_hr"]=0
    else: f["cwt_f0"]=0; f["cwt_hr"]=0
    return f

def extract_cwt_temporal(sig, fs=FS, n_seg=6):
    T=len(sig); sl=T//n_seg
    sfs=[]
    for i in range(n_seg):
        s,e=i*sl,min((i+1)*sl,T)
        if e-s<int(2*fs): continue
        sfs.append(extract_cwt_segment(sig[s:e],fs))
    if not sfs: sfs=[extract_cwt_segment(sig,fs)]
    df=pd.DataFrame(sfs)
    f={f"{k}_mean":df[k].mean() for k in df.columns}
    f.update({f"{k}_std":df[k].std() for k in df.columns})
    for key in ["cwt_mean_energy","cwt_high_freq_energy","cwt_freq_var","cwt_entropy"]:
        if key in df.columns and len(df)>=3:
            sl2,_,rv,_,_=linregress(np.arange(len(df)),df[key].values)
            f[f"{key}_slope"]=sl2; f[f"{key}_slope_r"]=rv
        else:
            f[f"{key}_slope"]=0; f[f"{key}_slope_r"]=0
    return f


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION METHODS
# ══════════════════════════════════════════════════════════════════════════════

def remove_redundant(X_tr, X_te, threshold=0.9):
    """Remove features with inter-correlation > threshold (on training set)."""
    corr = np.abs(np.corrcoef(X_tr.T))
    np.fill_diagonal(corr, 0)
    keep = list(range(X_tr.shape[1]))
    for i in range(len(keep)):
        if i >= len(keep): break
        to_drop = []
        for j in range(i+1, len(keep)):
            if j >= len(keep): break
            if corr[keep[i], keep[j]] > threshold:
                to_drop.append(j)
        for j in sorted(to_drop, reverse=True):
            if j < len(keep):
                keep.pop(j)
    return X_tr[:, keep], X_te[:, keep], keep

def select_top_k(X_tr, X_te, y_tr, k):
    """Select top-k features by |Spearman correlation| with target."""
    cors = np.array([abs(spearmanr(X_tr[:,j], y_tr)[0]) for j in range(X_tr.shape[1])])
    idx = np.argsort(cors)[-k:]
    return X_tr[:, idx], X_te[:, idx], idx

def combined_selection(X_tr, X_te, y_tr, redundancy_thresh=0.9, top_k=20):
    """Remove redundancy first, then select top-k by relevancy."""
    X_tr_r, X_te_r, keep1 = remove_redundant(X_tr, X_te, redundancy_thresh)
    if X_tr_r.shape[1] <= top_k:
        return X_tr_r, X_te_r
    X_tr_s, X_te_s, _ = select_top_k(X_tr_r, X_te_r, y_tr, top_k)
    return X_tr_s, X_te_s


# ══════════════════════════════════════════════════════════════════════════════
# CV
# ══════════════════════════════════════════════════════════════════════════════

def loo_no_sel(X, y, mfn):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(X[tr]),y[tr]); p[te]=m.predict(sc.transform(X[te]))
    return p

def loo_topk(X, y, mfn, k):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        Xtr_s, Xte_s, _ = select_top_k(X[tr], X[te], y[tr], k)
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(Xtr_s),y[tr]); p[te]=m.predict(sc.transform(Xte_s))
    return p

def loo_combined(X, y, mfn, thresh=0.9, k=20):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        Xtr_s, Xte_s = combined_selection(X[tr], X[te], y[tr], thresh, k)
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(Xtr_s),y[tr]); p[te]=m.predict(sc.transform(Xte_s))
    return p

def met(y,yh):
    return {"R2":round(r2_score(y,yh),4),"MAE":round(mean_absolute_error(y,yh),1),
            "r":round(pearsonr(y,yh)[0],4)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("Exp 11: Combined Features + Feature Selection")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])
    X_clinic = paired[CLINIC_COLS].values.astype(float)

    # Extract all features
    print("\n  Extracting features...")
    gait_r, act_r, cwt_r = [], [], []
    for i,(_,r) in enumerate(paired.iterrows()):
        fn = fname(r)
        wp = WALK_DIR/fn
        if wp.exists():
            sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
        else:
            sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
        gait_r.append(extract_gait(sig))
        cwt_r.append(extract_cwt_temporal(sig))
        dp = DAY_DIR/fn
        if dp.exists():
            day = pd.read_csv(dp)[["X","Y","Z"]].values.astype(np.float32)
        else:
            day = sig
        act_r.append(extract_activity(day))
        if (i+1)%50==0: print(f"    {i+1}/{n}", flush=True)
    print(f"    {n}/{n}")

    def to_mat(rows):
        df=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
        for c in df.columns:
            if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
        return df.values.astype(float), list(df.columns)

    X_gait, gn = to_mat(gait_r)
    X_act, an = to_mat(act_r)
    X_cwt, cn = to_mat(cwt_r)

    # Combined feature sets for HOME
    X_all_home = np.column_stack([X_act, X_gait, X_cwt])
    X_act_gait = np.column_stack([X_act, X_gait])
    X_act_cwt = np.column_stack([X_act, X_cwt])

    # Combined for CLINIC
    X_all_clinic = np.column_stack([X_clinic, X_cwt])

    print(f"  Activity: {X_act.shape[1]}, Gait: {X_gait.shape[1]}, CWT: {X_cwt.shape[1]}")
    print(f"  Combined home: {X_all_home.shape[1]}, Combined clinic: {X_all_clinic.shape[1]}")

    results = []
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,
                                subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
    models = {"Ridge":lambda:Ridge(alpha=10),
              "ElasticNet":lambda:ElasticNet(alpha=1,l1_ratio=0.5,max_iter=10000),
              "RF":lambda:RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,
                                                 random_state=42,n_jobs=-1),
              "XGBoost":xgb}

    def rec(model, config, feat_name, sel_type, preds, nf):
        m = met(y, preds)
        results.append({"model":model,"config":config,"features":feat_name,
                        "selection":sel_type,"n_features":nf,**m})
        print(f"    {model:12s} {config:4s} {feat_name:20s} {sel_type:15s} ({nf:3d}f) R²={m['R2']:.4f}")

    print("\n  Running LOO CV...")

    # A series: test combined home features with different selection methods
    home_combos = [
        ("Act+Gait+CWT", X_all_home),
        ("Act+Gait", X_act_gait),
        ("Act+CWT", X_act_cwt),
        ("Activity", X_act),
    ]

    for combo_name, X_h in home_combos:
        nf = X_h.shape[1]
        for mname, mfn in models.items():
            # No selection
            rec(mname, "A1", combo_name, "none", loo_no_sel(X_h, y, mfn), nf)
            # Top-k
            k = min(15, nf)
            if nf > k:
                rec(mname, "A1", combo_name, f"top{k}", loo_topk(X_h, y, mfn, k), k)
            # Combined (redundancy + relevancy)
            if nf > 20:
                rec(mname, "A1", combo_name, "combined20", loo_combined(X_h, y, mfn, 0.9, 20), 20)

            # A2
            X_a2 = np.column_stack([X_h, X_demo])
            nf2 = X_a2.shape[1]
            rec(mname, "A2", combo_name, "none", loo_no_sel(X_a2, y, mfn), nf2)
            if nf2 > 20:
                rec(mname, "A2", combo_name, "combined20", loo_combined(X_a2, y, mfn, 0.9, 20), 20)

    # C series: clinic + CWT
    clinic_combos = [
        ("Clinic+CWT", X_all_clinic),
        ("Clinic", X_clinic),
    ]
    for combo_name, X_c in clinic_combos:
        nf = X_c.shape[1]
        for mname, mfn in models.items():
            rec(mname, "C1", combo_name, "none", loo_no_sel(X_c, y, mfn), nf)
            X_c2 = np.column_stack([X_c, X_demo])
            nf2 = X_c2.shape[1]
            rec(mname, "C2", combo_name, "none", loo_no_sel(X_c2, y, mfn), nf2)
            if nf2 > 15:
                rec(mname, "C2", combo_name, "combined15", loo_combined(X_c2, y, mfn, 0.9, 15), 15)

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp11_combined.csv", index=False)

    # Summary
    print("\n"+"="*60)
    print("BEST PER CONFIG")
    print("="*60)
    for cfg in ["A1","A2","C1","C2"]:
        sub = df[df["config"]==cfg]
        if sub.empty: continue
        best = sub.loc[sub["R2"].idxmax()]
        print(f"  {cfg}: R²={best['R2']:.4f} ({best['model']}, {best['features']}, {best['selection']}, {int(best['n_features'])}f)")

    print(f"\nSaved to {OUT}/exp11_combined.csv")


if __name__ == "__main__":
    main()
