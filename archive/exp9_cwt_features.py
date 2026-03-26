#!/usr/bin/env python3
"""
Experiment 9: CWT-based Time-Frequency Features
================================================
Features from POMS_Methods_Results:
  - Estimated cadence, dominant frequency, high-freq energy
  - Wavelet entropy, max power frequency, frequency variability/CV
  - Harmonic ratio, fundamental frequency, mean energy

Plus temporal dynamics features:
  - Slope of mean_energy over walking bout segments (fatigue indicator)
  - Slope of high_freq_energy over segments

Tested alone and combined with existing best features (Activity Profile, Gait).

Configs: A1, A2, C1, C2
CV: LOO
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import pywt
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
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


# ══════════════════════════════════════════════════════════════════════════════
# CWT FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_cwt_power(sig_1d, fs, freqs):
    """Compute CWT power spectrum using PyWavelets Morlet wavelet."""
    scales = fs / (freqs + 1e-12)
    coeffs, _ = pywt.cwt(sig_1d, scales, 'morl', sampling_period=1.0/fs)
    power = np.abs(coeffs)**2
    return power  # (n_freqs, n_time)


def extract_cwt_features_segment(sig, fs=FS):
    """Extract CWT time-frequency features from a signal segment.
    sig: (T, 3) array [AP, ML, VT] or [X, Y, Z]
    """
    # Compute resultant magnitude
    vm = np.sqrt(sig[:,0]**2 + sig[:,1]**2 + sig[:,2]**2)

    # Normalize to [-1, 1]
    vm_norm = vm - vm.mean()
    vmax = np.max(np.abs(vm_norm)) + 1e-12
    vm_norm = vm_norm / vmax

    f = {}

    # CWT analysis
    freqs = np.linspace(0.5, 12, 50)  # 0.5-12 Hz range
    power = compute_cwt_power(vm_norm, fs, freqs)
    mean_power = power.mean(axis=1)  # average over time

    # Mean energy (total CWT power)
    f["mean_energy"] = np.mean(power)

    # High-frequency energy (3.5-12 Hz)
    high_mask = freqs >= 3.5
    f["high_freq_energy"] = np.mean(power[high_mask]) if high_mask.any() else 0

    # Dominant frequency (freq with highest mean power)
    f["dominant_freq"] = freqs[np.argmax(mean_power)]

    # Estimated cadence (from dominant freq in gait range 0.5-3.5 Hz)
    gait_mask = (freqs >= 0.5) & (freqs <= 3.5)
    gait_power = mean_power.copy()
    gait_power[~gait_mask] = 0
    f["estimated_cadence"] = freqs[np.argmax(gait_power)] * 60  # steps/min

    # Max power frequency (global max in time-freq matrix)
    max_idx = np.unravel_index(np.argmax(power), power.shape)
    f["max_power_freq"] = freqs[max_idx[0]]

    # Frequency variability: dominant freq per time window
    n_windows = max(1, power.shape[1] // int(fs))
    dom_freqs = []
    for w in range(n_windows):
        s = w * int(fs)
        e = min(s + int(fs), power.shape[1])
        window_power = power[:, s:e].mean(axis=1)
        dom_freqs.append(freqs[np.argmax(window_power)])
    dom_freqs = np.array(dom_freqs)
    f["freq_variability"] = np.std(dom_freqs)
    f["freq_cv"] = np.std(dom_freqs) / (np.mean(dom_freqs) + 1e-12)

    # Wavelet entropy
    p_norm = mean_power / (mean_power.sum() + 1e-12)
    p_nz = p_norm[p_norm > 0]
    f["wavelet_entropy"] = -np.sum(p_nz * np.log2(p_nz + 1e-12))

    # Harmonic ratio (from FFT)
    fft_vals = np.abs(rfft(vm_norm))
    fft_freqs = rfftfreq(len(vm_norm), 1/fs)
    gait_band = (fft_freqs >= 0.5) & (fft_freqs <= 3.5)
    if gait_band.any():
        f0 = fft_freqs[gait_band][np.argmax(fft_vals[gait_band])]
        f["fundamental_freq"] = f0
        if f0 > 0:
            even_p, odd_p = 0, 0
            for h in range(1, 11):
                idx = np.argmin(np.abs(fft_freqs - h * f0))
                if h % 2 == 0: even_p += fft_vals[idx]**2
                else: odd_p += fft_vals[idx]**2
            f["harmonic_ratio"] = even_p / (odd_p + 1e-12)
        else:
            f["harmonic_ratio"] = 0
    else:
        f["fundamental_freq"] = 0
        f["harmonic_ratio"] = 0

    return f


def extract_cwt_with_temporal(sig, fs=FS, n_segments=6):
    """Extract CWT features + temporal dynamics (slopes over segments).
    Splits signal into n_segments equal parts and computes features per segment.
    Then computes slope of key features over segments (fatigue dynamics).
    """
    T = len(sig)
    seg_len = T // n_segments

    # Per-segment features
    seg_feats = []
    for i in range(n_segments):
        s = i * seg_len
        e = min(s + seg_len, T)
        if e - s < int(2 * fs):
            continue
        seg_feats.append(extract_cwt_features_segment(sig[s:e], fs))

    if not seg_feats:
        seg_feats = [extract_cwt_features_segment(sig, fs)]

    df = pd.DataFrame(seg_feats)

    # Aggregate: mean of each feature across segments
    f = {f"cwt_{k}_mean": df[k].mean() for k in df.columns}
    f.update({f"cwt_{k}_std": df[k].std() for k in df.columns})

    # Temporal slopes (fatigue dynamics)
    for key in ["mean_energy", "high_freq_energy", "freq_variability", "wavelet_entropy"]:
        if key in df.columns and len(df) >= 3:
            x = np.arange(len(df))
            slope, _, r_value, _, _ = linregress(x, df[key].values)
            f[f"cwt_{key}_slope"] = slope
            f[f"cwt_{key}_slope_r"] = r_value
        else:
            f[f"cwt_{key}_slope"] = 0
            f[f"cwt_{key}_slope_r"] = 0

    return f


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING FEATURE EXTRACTION (reuse from exp1 and exp2)
# ══════════════════════════════════════════════════════════════════════════════

def extract_gait(sig, fs=FS):
    """Same 35 gait features as exp1."""
    from exp1_gait_ml import extract_gait as _eg
    return _eg(sig, fs)

def extract_activity(xyz, fs=FS):
    """Same 15 activity features as exp2."""
    from exp2_activity_ml import extract_activity as _ea
    return _ea(xyz, fs)


# ══════════════════════════════════════════════════════════════════════════════
# CV
# ══════════════════════════════════════════════════════════════════════════════

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
            "r":round(pearsonr(y,yh)[0],4)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("Exp 9: CWT Time-Frequency Features + Temporal Dynamics")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])
    X_clinic = paired[CLINIC_COLS].values.astype(float)

    # Extract CWT features from home walking segments
    print("\n  Extracting CWT features (home walking)...")
    cwt_home_rows = []
    gait_rows = []
    activity_rows = []
    for i,(_,r) in enumerate(paired.iterrows()):
        fn = fname(r)
        # Home walking
        wp = WALK_DIR/fn
        if wp.exists():
            sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
        else:
            sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
        cwt_home_rows.append(extract_cwt_with_temporal(sig))
        gait_rows.append(extract_gait(sig))

        # Activity profile from daytime
        dp = DAY_DIR/fn
        if dp.exists():
            day_sig = pd.read_csv(dp)[["X","Y","Z"]].values.astype(np.float32)
        else:
            day_sig = sig
        activity_rows.append(extract_activity(day_sig))

        if (i+1)%20==0: print(f"    {i+1}/{n}", flush=True)
    print(f"    {n}/{n}")

    # CWT from clinic signals
    print("  Extracting CWT features (clinic)...")
    cwt_clinic_rows = []
    for _,r in paired.iterrows():
        fn = fname(r)
        sig = pd.read_csv(BASE/"csv_raw2"/fn)[["X","Y","Z"]].values.astype(np.float32)
        cwt_clinic_rows.append(extract_cwt_with_temporal(sig, n_segments=6))
    print(f"    {n}/{n}")

    # Build feature matrices
    def to_mat(rows):
        df = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
        for c in df.columns:
            if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
        return df.values.astype(float), list(df.columns)

    X_cwt_h, cwt_names = to_mat(cwt_home_rows)
    X_cwt_c, _ = to_mat(cwt_clinic_rows)
    X_gait, _ = to_mat(gait_rows)
    X_act, _ = to_mat(activity_rows)

    print(f"  CWT home: {X_cwt_h.shape[1]} features")
    print(f"  CWT clinic: {X_cwt_c.shape[1]} features")
    print(f"  Gait: {X_gait.shape[1]}, Activity: {X_act.shape[1]}")

    # Feature combinations
    combos = {
        "CWT only":               {"home": X_cwt_h, "clinic": X_cwt_c},
        "Gait only":              {"home": X_gait, "clinic": X_clinic},
        "Activity only":          {"home": X_act, "clinic": None},
        "Gait + CWT":             {"home": np.column_stack([X_gait, X_cwt_h]),
                                   "clinic": np.column_stack([X_clinic, X_cwt_c])},
        "Activity + CWT":         {"home": np.column_stack([X_act, X_cwt_h]),
                                   "clinic": None},
        "Activity + Gait":        {"home": np.column_stack([X_act, X_gait]),
                                   "clinic": None},
        "Activity + Gait + CWT":  {"home": np.column_stack([X_act, X_gait, X_cwt_h]),
                                   "clinic": None},
    }

    results = []
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,
                                colsample_bytree=0.8,random_state=42,verbosity=0)
    models = {"Ridge":lambda:Ridge(alpha=10),
              "ElasticNet":lambda:ElasticNet(alpha=1,l1_ratio=0.5,max_iter=10000),
              "RF":lambda:RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,
                                                 random_state=42,n_jobs=-1),
              "XGBoost":xgb}

    def rec(model, config, feat_name, preds, n_feat):
        m = met(y, preds)
        results.append({"model":model,"config":config,"features":feat_name,"n_features":n_feat,**m})
        print(f"    {model:12s} {config:4s} {feat_name:25s} ({n_feat:3d}f) R²={m['R2']:.4f}")

    print("\n  Running LOO CV...")
    for combo_name, data in combos.items():
        X_h = data["home"]
        X_c = data.get("clinic")
        nf_h = X_h.shape[1]

        for mname, mfn in models.items():
            # A1
            if nf_h > 20:
                rec(mname, "A1", combo_name, loo_fs(X_h, y, mfn, k=min(20, nf_h)), nf_h)
            else:
                rec(mname, "A1", combo_name, loo(X_h, y, mfn), nf_h)

            # A2
            X_a2 = np.column_stack([X_h, X_demo])
            rec(mname, "A2", combo_name, loo(X_a2, y, mfn), X_a2.shape[1])

            # C1, C2 (only if clinic features available)
            if X_c is not None:
                nf_c = X_c.shape[1]
                rec(mname, "C1", combo_name, loo(X_c, y, mfn), nf_c)
                X_c2 = np.column_stack([X_c, X_demo])
                rec(mname, "C2", combo_name, loo(X_c2, y, mfn), X_c2.shape[1])

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp9_cwt.csv", index=False)

    # Summary
    print("\n"+"="*60)
    print("BEST PER CONFIG AND FEATURE SET")
    print("="*60)
    for cfg in ["A1","A2","C1","C2"]:
        sub = df[df["config"]==cfg]
        if sub.empty: continue
        print(f"\n  --- {cfg} ---")
        for feat in sub["features"].unique():
            fsub = sub[sub["features"]==feat]
            best = fsub.loc[fsub["R2"].idxmax()]
            print(f"    {feat:25s} {best['model']:12s} R²={best['R2']:.4f}")

    # Overall best
    print("\n  --- OVERALL BEST ---")
    for cfg in ["A1","A2","C1","C2"]:
        sub = df[df["config"]==cfg]
        if sub.empty: continue
        best = sub.loc[sub["R2"].idxmax()]
        print(f"    {cfg}: R²={best['R2']:.4f} ({best['model']}, {best['features']}, {int(best['n_features'])}f)")

    print(f"\nSaved to {OUT}/exp9_cwt.csv")


if __name__ == "__main__":
    main()
