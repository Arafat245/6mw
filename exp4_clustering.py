#!/usr/bin/env python3
"""
Experiment 4: K-means Clustering on Best Feature Set (Meyer et al. method)
==========================================================================
Apply subject-specific k-means clustering to identify the most informative
subset of data. Train models per-cluster and compare to unclustered.

Uses the best performing feature set from Exp 1-3:
  - Activity profile features (best A1)
  - Walking gait features (best C1)

Method (adapted from Meyer et al. 2024):
  1. Compute features per time window (not aggregated)
  2. K-means cluster per subject (k chosen by Davies-Bouldin)
  3. Sort clusters by Frequency Dispersion (or dominant feature)
  4. Train models on each cluster's features separately
  5. Compare per-cluster vs all-data performance

CV: LOO
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import welch, find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
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


def compute_windowed_activity_features(xyz, fs=FS, win_sec=60):
    """Compute activity features per window (for clustering)."""
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    win = int(win_sec * fs)
    windows = []
    for s in range(0, len(vm) - win, win):
        seg = vm[s:s+win]
        ai = np.sqrt(np.mean(seg**2))
        std = np.std(seg)
        iqr = np.percentile(seg, 75) - np.percentile(seg, 25)
        # Spectral
        if len(seg) > 32:
            fr, ps = welch(seg, fs=fs, nperseg=min(128, len(seg)))
            total = np.trapz(ps, fr) + 1e-12
            gait_band = (fr >= 0.5) & (fr <= 3.5)
            gait_pwr = np.trapz(ps[gait_band], fr[gait_band]) / total if gait_band.any() else 0
            cf = np.sum(fr * ps) / (np.sum(ps) + 1e-12)
            pn = ps / (ps.sum() + 1e-12); pn = pn[pn > 0]
            fd = 1 - (np.sum(pn * np.cos(2*np.pi*fr[:len(pn)]/fs)))**2 if len(pn) > 1 else 0
        else:
            gait_pwr, cf, fd = 0, 0, 0
        windows.append({"ai": ai, "std": std, "iqr": iqr,
                        "gait_pwr": gait_pwr, "cf": cf, "fd": abs(fd)})
    return pd.DataFrame(windows) if windows else pd.DataFrame()


def cluster_and_select(window_feats, n_clusters_range=(2, 5)):
    """K-means clustering on windowed features. Select best cluster.
    Best cluster = highest mean gait_pwr (most walking-like).
    """
    if len(window_feats) < 5:
        return window_feats, 0, 1

    X = window_feats.values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # Find optimal k by Davies-Bouldin
    best_k, best_db = 2, np.inf
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        if k >= len(Xs):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        db = davies_bouldin_score(Xs, labels)
        if db < best_db:
            best_db = db
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)

    # Select cluster with highest gait_pwr (most walking-like)
    cluster_gait = []
    for c in range(best_k):
        mask = labels == c
        if mask.sum() > 0:
            cluster_gait.append((c, window_feats.loc[mask, "gait_pwr"].mean()))
        else:
            cluster_gait.append((c, 0))
    best_cluster = max(cluster_gait, key=lambda x: x[1])[0]

    return window_feats[labels == best_cluster], best_cluster, best_k


def aggregate_features(window_df):
    """Aggregate windowed features into subject-level summary."""
    if len(window_df) == 0:
        return {f"{c}_{s}": 0 for c in ["ai","std","iqr","gait_pwr","cf","fd"]
                for s in ["mean","med","std"]}
    f = {}
    for c in window_df.columns:
        v = window_df[c].values
        f[f"{c}_mean"] = np.mean(v)
        f[f"{c}_med"] = np.median(v)
        f[f"{c}_std"] = np.std(v)
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
    print("Exp 4: K-means Clustering (Meyer et al. method)")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])

    # Extract windowed features, cluster, select best cluster, aggregate
    print("  Computing windowed features + clustering...")
    all_data_rows = []  # no clustering
    clustered_rows = []  # with clustering
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

        wf = compute_windowed_activity_features(sig)

        # All data (no clustering)
        all_data_rows.append(aggregate_features(wf))

        # Clustered (select best cluster)
        if len(wf) >= 5:
            selected, best_c, n_k = cluster_and_select(wf)
            clustered_rows.append(aggregate_features(selected))
        else:
            clustered_rows.append(aggregate_features(wf))

        if (i+1)%50==0: print(f"    {i+1}/{n}", flush=True)
    print(f"    {n}/{n}")

    X_all = pd.DataFrame(all_data_rows).replace([np.inf,-np.inf],np.nan)
    X_clust = pd.DataFrame(clustered_rows).replace([np.inf,-np.inf],np.nan)
    for c in X_all.columns:
        if X_all[c].isna().any(): X_all[c]=X_all[c].fillna(X_all[c].median())
    for c in X_clust.columns:
        if X_clust[c].isna().any(): X_clust[c]=X_clust[c].fillna(X_clust[c].median())
    X_all = X_all.values.astype(float)
    X_clust = X_clust.values.astype(float)
    print(f"  All data: {X_all.shape[1]} features, Clustered: {X_clust.shape[1]} features")

    results = []
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
    models = {"Ridge":lambda:Ridge(alpha=10),"ElasticNet":lambda:ElasticNet(alpha=1,l1_ratio=0.5,max_iter=10000),
              "RF":lambda:RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=-1),
              "XGBoost":xgb}

    print("\n  Running LOO CV...")
    for mname,mfn in models.items():
        # All data A1
        p = loo(X_all, y, mfn)
        m = met(y,p); results.append({"model":mname,"method":"all_data","config":"A1",**m})
        print(f"    {mname:15s} all_data  A1 R²={m['R2']:.4f}")

        # Clustered A1
        p = loo(X_clust, y, mfn)
        m = met(y,p); results.append({"model":mname,"method":"clustered","config":"A1",**m})
        print(f"    {mname:15s} clustered A1 R²={m['R2']:.4f}")

        # All data A2
        p = loo(np.column_stack([X_all, X_demo]), y, mfn)
        m = met(y,p); results.append({"model":mname,"method":"all_data","config":"A2",**m})
        print(f"    {mname:15s} all_data  A2 R²={m['R2']:.4f}")

        # Clustered A2
        p = loo(np.column_stack([X_clust, X_demo]), y, mfn)
        m = met(y,p); results.append({"model":mname,"method":"clustered","config":"A2",**m})
        print(f"    {mname:15s} clustered A2 R²={m['R2']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp4_clustering.csv", index=False)
    print("\n" + "="*60)
    print("All data vs Clustered:")
    for method in ["all_data", "clustered"]:
        sub = df[df["method"]==method]
        pivot = sub.pivot_table(index="model",columns="config",values="R2",aggfunc="first")
        print(f"\n  {method}:")
        print(pivot.to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/exp4_clustering.csv")

if __name__ == "__main__":
    main()
