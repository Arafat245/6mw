#!/usr/bin/env python3
"""
Experiment 8: Feature Selection Ablation
========================================
Run ALL feature sets with consistent per-fold feature selection (top-k by Spearman)
vs no feature selection. Standardized across all configs.

Feature sets: Gait (35), Activity (15), Sway (78), Clustered (18)
Configs: A1, A2, C1, C2
Models: RF, XGBoost, ElasticNet (best performers from Exp 1-4)
CV: LOO
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
BASIC_DEMO = ["cohort_M","Age","Sex","Height","Weight","BMI"]

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

CLINIC_COLS = ["cadence_hz_c","step_time_cv_pct_c","acf_step_regularity_c","hr_ap_c","hr_vt_c",
               "ml_rms_g_c","ml_spectral_entropy_c","jerk_mean_abs_gps_c","enmo_mean_g_c","cadence_slope_per_min_c"]

def loo_no_sel(X, y, mfn):
    p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = mfn()
        m.fit(sc.fit_transform(X[tr]), y[tr]); p[te] = m.predict(sc.transform(X[te]))
    return p

def loo_with_sel(X, y, mfn, top_k):
    p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        cors = np.array([abs(spearmanr(X[tr,j], y[tr])[0]) for j in range(X.shape[1])])
        idx = np.argsort(cors)[-top_k:]
        sc = StandardScaler(); m = mfn()
        m.fit(sc.fit_transform(X[tr][:,idx]), y[tr]); p[te] = m.predict(sc.transform(X[te][:,idx]))
    return p

def met(y, yh):
    return {"R2": round(r2_score(y,yh), 4), "MAE": round(mean_absolute_error(y,yh), 1),
            "r": round(pearsonr(y,yh)[0], 4)}

def main():
    print("="*60)
    print("Exp 8: Feature Selection Ablation")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])
    X_clinic = paired[CLINIC_COLS].values.astype(float)

    # Load all feature sets
    print("  Loading feature sets...")
    feat_sets = {}

    # Gait (from Exp 1)
    gait_csv = OUT/"exp1_gait_ml.csv"  # need raw features, not results
    # Re-extract or load from saved
    home_gait_csv = OUT/"home_gait_features.csv"
    if home_gait_csv.exists():
        feat_sets["gait"] = pd.read_csv(home_gait_csv).values.astype(float)
    else:
        # Re-extract from walking segments
        from exp1_gait_ml import extract_gait, fname
        WALK_DIR = OUT/"walking_segments"
        rows = []
        for _,r in paired.iterrows():
            fn = fname(r)
            wp = WALK_DIR/fn
            if wp.exists():
                sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
            else:
                sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
            rows.append(extract_gait(sig))
        df = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
        for c in df.columns:
            if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
        feat_sets["gait"] = df.values.astype(float)
        df.to_csv(home_gait_csv, index=False)

    # Activity (from Exp 2)
    from exp2_activity_ml import extract_activity, fname as fname2
    DAY_DIR = OUT/"daytime_segments"
    rows = []
    for _,r in paired.iterrows():
        fn = fname2(r)
        dp = DAY_DIR/fn
        if dp.exists():
            sig = pd.read_csv(dp)[["X","Y","Z"]].values.astype(np.float32)
        else:
            wp = OUT/"walking_segments"/fn
            if wp.exists():
                sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
            else:
                sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
        rows.append(extract_activity(sig))
    df = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    for c in df.columns:
        if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
    feat_sets["activity"] = df.values.astype(float)

    # Sway (from Exp 3)
    from exp3_sway_ml import extract_sway_all, fname as fname3
    rows = []
    for i,(_,r) in enumerate(paired.iterrows()):
        fn = fname3(r)
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
        if (i+1)%50==0: print(f"    sway {i+1}/{len(paired)}", flush=True)
    print(f"    sway {len(paired)}/{len(paired)}")
    df = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    for c in df.columns:
        if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
    feat_sets["sway"] = df.values.astype(float)

    # Clustered (from Exp 4)
    from exp4_clustering import compute_windowed_activity_features, cluster_and_select, aggregate_features, fname as fname4
    rows = []
    for _,r in paired.iterrows():
        fn = fname4(r)
        dp = DAY_DIR/fn
        if dp.exists():
            sig = pd.read_csv(dp)[["X","Y","Z"]].values.astype(np.float32)
        else:
            wp = OUT/"walking_segments"/fn
            if wp.exists():
                sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
            else:
                sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
        wf = compute_windowed_activity_features(sig)
        if len(wf) >= 5:
            selected,_,_ = cluster_and_select(wf)
            rows.append(aggregate_features(selected))
        else:
            rows.append(aggregate_features(wf))
    df = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    for c in df.columns:
        if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
    feat_sets["clustered"] = df.values.astype(float)

    for k,v in feat_sets.items():
        print(f"  {k}: {v.shape[1]} features")

    # Models
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,
                                colsample_bytree=0.8,random_state=42,verbosity=0)
    models = {"ElasticNet":lambda:ElasticNet(alpha=1,l1_ratio=0.5,max_iter=10000),
              "RF":lambda:RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=-1),
              "XGBoost":xgb}

    results = []

    def rec(model, config, feat_type, sel_type, preds):
        m = met(y, preds)
        results.append({"model":model,"config":config,"features":feat_type,"selection":sel_type,**m})
        print(f"    {model:12s} {config:4s} {feat_type:10s} {sel_type:8s} R²={m['R2']:.4f}")

    print("\n  Running ablation...")
    top_k_map = {"gait": 15, "activity": 10, "sway": 20, "clustered": 12}

    for feat_name, X_feat in feat_sets.items():
        n_feat = X_feat.shape[1]
        top_k = top_k_map.get(feat_name, min(15, n_feat))

        for mname, mfn in models.items():
            # A1: no selection
            rec(mname, "A1", feat_name, "none", loo_no_sel(X_feat, y, mfn))
            # A1: with selection
            if n_feat > top_k:
                rec(mname, "A1", feat_name, f"top{top_k}", loo_with_sel(X_feat, y, mfn, top_k))

            # A2: no selection
            X_a2 = np.column_stack([X_feat, X_demo])
            rec(mname, "A2", feat_name, "none", loo_no_sel(X_a2, y, mfn))
            # A2: with selection
            top_k2 = min(top_k + 6, X_a2.shape[1])  # allow all demo features + top_k
            rec(mname, "A2", feat_name, f"top{top_k2}", loo_with_sel(X_a2, y, mfn, top_k2))

    # C configs
    for mname, mfn in models.items():
        rec(mname, "C1", "clinic", "none", loo_no_sel(X_clinic, y, mfn))
        X_c2 = np.column_stack([X_clinic, X_demo])
        rec(mname, "C2", "clinic", "none", loo_no_sel(X_c2, y, mfn))
        rec(mname, "C2", "clinic", "top12", loo_with_sel(X_c2, y, mfn, 12))

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp8_featsel.csv", index=False)

    print("\n"+"="*60)
    print("FEATURE SELECTION ABLATION (best per feat_type + config)")
    print("="*60)

    for feat_name in list(feat_sets.keys()) + ["clinic"]:
        sub = df[df["features"]==feat_name]
        if sub.empty: continue
        print(f"\n  --- {feat_name} ---")
        pivot = sub.pivot_table(index=["model","selection"],columns="config",values="R2",aggfunc="first")
        co = [c for c in ["A1","A2","C1","C2"] if c in pivot.columns]
        if co:
            print(pivot[co].to_string(float_format="{:.4f}".format))

    # Best overall per config
    print("\n  --- BEST OVERALL ---")
    for cfg in ["A1","A2","C1","C2"]:
        sub = df[df["config"]==cfg]
        if sub.empty: continue
        best = sub.loc[sub["R2"].idxmax()]
        print(f"    {cfg}: R²={best['R2']:.4f} ({best['model']}, {best['features']}, {best['selection']})")

    print(f"\nSaved to {OUT}/exp8_featsel.csv")

if __name__ == "__main__":
    main()
