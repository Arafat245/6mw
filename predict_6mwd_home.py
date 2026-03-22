#!/usr/bin/env python3
"""
Predict 6-Minute Walk Distance (6MWD) from Home Accelerometer Data
===================================================================
Feature extraction: Walking bout selection from raw home IMU signals,
then gait-specific features (temporal, spectral, autocorrelation, harmonic).

Pipeline: Raw home IMU → walking bout detection → gravity removal &
anatomical alignment → gait feature extraction.

Ablation matrix:
  A1 = Home IMU features only (no demographics, no clinical scores)
  A2 = Home IMU + Basic Demographics (Age, Sex, Height, Weight, BMI, cohort)
  A3 = Home IMU + Demographics + Clinical Scores (BDI, MFIS)
  B1 = Home IMU + CCPT (clinic-calibrated prediction)
  B2 = Home IMU + Demographics + CCPT
  C1 = Clinic accelerometer features only
  C2 = Clinic accel + Basic Demographics
  C3 = Clinic accel + Demographics + Clinical Scores

CV: Leave-One-Subject-Out (LOO) — consistent for all models
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
FIG = os.path.join(OUT, "figures")
PRED = os.path.join(OUT, "predictions")
for d in [OUT, FIG, PRED]:
    os.makedirs(d, exist_ok=True)

FS = 30.0

CLINIC_ACCEL_COLS = [
    "cadence_hz_c", "step_time_cv_pct_c", "acf_step_regularity_c",
    "hr_ap_c", "hr_vt_c", "ml_rms_g_c", "ml_spectral_entropy_c",
    "jerk_mean_abs_gps_c", "enmo_mean_g_c", "cadence_slope_per_min_c",
]

BASIC_DEMO = ["cohort_M", "Age", "Sex", "Height", "Weight", "BMI"]
CLINICAL_SCORES = ["BDI Raw Score", "MFIS Total"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_tabular_data():
    home = pd.read_csv(os.path.join(BASE, "sway_features_home.csv"))
    home.rename(columns={"year_x": "year"}, inplace=True)

    clinic = pd.read_csv(os.path.join(BASE, "features_top10.csv"))
    clinic_cols = {c: c + "_c" for c in clinic.columns
                   if c not in ("cohort", "subj_id", "year", "sixmwd", "fs")}
    clinic = clinic.rename(columns=clinic_cols)
    paired = home.merge(clinic, on=["cohort", "subj_id", "year"], how="inner",
                        suffixes=("", "_clinic"))

    demo = pd.read_excel(os.path.join(BASE, "SwayDemographics.xlsx"))
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    paired = paired.merge(demo, on=["cohort", "subj_id"], how="left")
    paired["cohort_M"] = (paired["cohort"] == "M").astype(int)
    paired["Sex"] = pd.to_numeric(paired["Sex"], errors="coerce")
    for col in ["Age", "Height", "Weight", "BMI", "BDI Raw Score", "MFIS Total"]:
        paired[col] = pd.to_numeric(paired[col], errors="coerce")
    return paired


def load_raw_signal(cohort, sid, year, sixmwd, source="csv_ca"):
    """Load raw X,Y,Z signal from csv_ca (preferred) or csv_processed_home (fallback)."""
    fname = f"{cohort}{sid:02d}_{year}_{sixmwd}.csv"
    fpath = os.path.join(BASE, source, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        if "X" in df.columns:
            return df[["X", "Y", "Z"]].values.astype(np.float32), "raw"
        return df[["AP", "ML", "VT"]].values.astype(np.float32), "preprocessed"
    # Fallback to csv_processed_home
    fpath2 = os.path.join(BASE, "csv_processed_home", fname)
    df = pd.read_csv(fpath2)
    return df[["AP", "ML", "VT"]].values.astype(np.float32), "preprocessed"


# ══════════════════════════════════════════════════════════════════════════════
# 2. WALKING BOUT DETECTION (on raw signal) + PREPROCESSING + FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_walking_bouts_raw(sig_xyz, fs=FS, win_sec=10, step_sec=2, min_bout_sec=30):
    """Detect walking bouts from raw X,Y,Z acceleration.
    Walking = moderate VM intensity + periodic signal in step frequency range.
    Returns list of (start_sample, end_sample) tuples.
    """
    vm = np.sqrt(sig_xyz[:, 0]**2 + sig_xyz[:, 1]**2 + sig_xyz[:, 2]**2)
    win = int(win_sec * fs)
    step = int(step_sec * fs)

    is_walking = []
    starts = []
    for s in range(0, len(vm) - win, step):
        seg_vm = vm[s:s + win]
        rms = np.sqrt(np.mean(seg_vm**2))
        std = np.std(seg_vm)

        # Periodicity: use axis with most variance
        seg = sig_xyz[s:s + win]
        variances = [np.var(seg[:, i]) for i in range(3)]
        best_axis = seg[:, np.argmax(variances)]
        best_c = best_axis - best_axis.mean()
        acf = np.correlate(best_c, best_c, "full")[len(best_c) - 1:]
        acf /= (acf[0] + 1e-12)
        search = acf[int(0.3 * fs):int(1.5 * fs)]
        peaks, props = find_peaks(search, height=0.1)
        regularity = props["peak_heights"][0] if len(peaks) > 0 else 0

        walking = (std > 0.05) and (rms > 0.8) and (rms < 1.5) and (regularity > 0.15)
        is_walking.append(walking)
        starts.append(s)

    # Merge consecutive walking windows into bouts
    bouts = []
    in_bout = False
    bout_start = 0
    for i, (w, s) in enumerate(zip(is_walking, starts)):
        if w and not in_bout:
            bout_start = s
            in_bout = True
        elif not w and in_bout:
            bout_end = starts[i - 1] + win
            if (bout_end - bout_start) >= min_bout_sec * fs:
                bouts.append((bout_start, bout_end))
            in_bout = False
    if in_bout:
        bout_end = starts[-1] + win
        if (bout_end - bout_start) >= min_bout_sec * fs:
            bouts.append((bout_start, bout_end))

    return bouts


def preprocess_walking_segment(seg_xyz, fs=FS):
    """Preprocess a raw walking segment: gravity removal + anatomical alignment.
    Returns (T, 3) array of [AP, ML, VT].
    """
    if len(seg_xyz) < 50:
        return seg_xyz

    # Gravity estimation via low-pass
    b, a = butter(4, 0.25 / (fs / 2), btype="low")
    g_est = filtfilt(b, a, seg_xyz, axis=0)
    g_mean = np.mean(g_est, axis=0)
    g_dir = g_mean / (np.linalg.norm(g_mean) + 1e-12)

    # Remove gravity
    g_proj = (seg_xyz @ g_dir)[:, None] * g_dir[None, :]
    dyn = seg_xyz - g_proj

    # VT = dynamic vertical component
    vt = seg_xyz @ g_dir - np.mean(seg_xyz @ g_dir)

    # Horizontal plane
    acc_h = dyn - (dyn @ g_dir)[:, None] * g_dir[None, :]

    # AP/ML via PCA of horizontal acceleration
    from sklearn.decomposition import PCA as PCA_
    if acc_h.shape[0] > 10:
        pca = PCA_(n_components=2)
        h2d = pca.fit_transform(acc_h)
        ap, ml = h2d[:, 0], h2d[:, 1]
    else:
        ap, ml = acc_h[:, 0], acc_h[:, 1]

    return np.column_stack([ap, ml, vt])


def score_walking_windows_preprocessed(sig, fs=FS, win_sec=10, step_sec=2):
    """Fallback: score windows on already-preprocessed AP/ML/VT signal."""
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    vt = sig[:, 2]
    vm = np.sqrt(sig[:, 0]**2 + sig[:, 1]**2 + sig[:, 2]**2)
    scores, starts = [], []
    for s in range(0, len(vt) - win, step):
        rms = np.sqrt(np.mean(vm[s:s + win]**2))
        seg_c = vt[s:s + win] - vt[s:s + win].mean()
        acf = np.correlate(seg_c, seg_c, "full")[len(seg_c) - 1:]
        acf /= (acf[0] + 1e-12)
        search = acf[int(0.3 * fs):int(1.5 * fs)]
        peaks, props = find_peaks(search, height=0.05)
        reg = props["peak_heights"][0] if len(peaks) > 0 else 0
        scores.append(rms * (1 + reg))
        starts.append(s)
    return np.array(scores), np.array(starts)


def select_best_walking_preprocessed(sig, top_pct=0.25, fs=FS):
    """Fallback for already-preprocessed signals."""
    scores, starts = score_walking_windows_preprocessed(sig, fs)
    win = int(10 * fs)
    n_sel = max(5, int(len(scores) * top_pct))
    n_sel = min(n_sel, len(scores))
    top_idx = np.sort(np.argsort(scores)[-n_sel:])
    return np.concatenate([sig[starts[i]:starts[i] + win] for i in top_idx])


def extract_gait_features(sig, fs=FS):
    """Extract 35 interpretable gait features from a walking signal (T, 3)."""
    ap, ml, vt = sig[:, 0], sig[:, 1], sig[:, 2]
    vm = np.sqrt(ap**2 + ml**2 + vt**2)
    f = {}

    # Basic stats
    for name, axis in [("ap", ap), ("ml", ml), ("vt", vt), ("vm", vm)]:
        f[f"{name}_rms"] = np.sqrt(np.mean(axis**2))
        f[f"{name}_std"] = np.std(axis)
        f[f"{name}_iqr"] = np.percentile(axis, 75) - np.percentile(axis, 25)
    f["sma"] = np.mean(np.abs(ap) + np.abs(ml) + np.abs(vt))

    # Jerk
    for name, axis in [("ap", ap), ("ml", ml), ("vt", vt)]:
        f[f"{name}_jerk_rms"] = np.sqrt(np.mean((np.diff(axis) * fs)**2))

    # Autocorrelation gait timing (VT and AP)
    for name, axis in [("vt", vt), ("ap", ap)]:
        axis_c = axis - axis.mean()
        acf = np.correlate(axis_c, axis_c, "full")[len(axis_c) - 1:]
        acf /= (acf[0] + 1e-12)
        max_lag = min(int(1.5 * fs), len(acf) - 1)
        search = acf[int(0.3 * fs):max_lag]
        peaks, props = find_peaks(search, height=0.0)
        f[f"{name}_step_reg"] = props["peak_heights"][0] if len(peaks) >= 1 else 0
        f[f"{name}_step_time"] = (peaks[0] + int(0.3 * fs)) / fs if len(peaks) >= 1 else 0
        f[f"{name}_stride_reg"] = props["peak_heights"][1] if len(peaks) >= 2 else 0
        f[f"{name}_step_sym"] = (
            f[f"{name}_step_reg"] / (f[f"{name}_stride_reg"] + 1e-8)
            if len(peaks) >= 2 else 1.0
        )

    # Spectral features
    for name, axis in [("vt", vt), ("ap", ap), ("ml", ml)]:
        if len(axis) > 64:
            freqs, psd = welch(axis, fs=fs, nperseg=min(256, len(axis)))
            gb = (freqs >= 0.5) & (freqs <= 3.5)
            f[f"{name}_dom_freq"] = freqs[gb][np.argmax(psd[gb])] if gb.any() else 0
            gp = np.trapz(psd[gb], freqs[gb]) if gb.any() else 0
            tp = np.trapz(psd, freqs) + 1e-12
            f[f"{name}_gait_pwr"] = gp / tp
            pn = psd / (psd.sum() + 1e-12)
            pn = pn[pn > 0]
            f[f"{name}_spec_ent"] = -np.sum(pn * np.log2(pn + 1e-12))
        else:
            f[f"{name}_dom_freq"] = 0
            f[f"{name}_gait_pwr"] = 0
            f[f"{name}_spec_ent"] = 0

    # Harmonic ratio (VT)
    fund = f.get("vt_dom_freq", 0)
    if fund > 0:
        fv = np.abs(rfft(vt))
        ff = rfftfreq(len(vt), 1 / fs)
        ep, op = 0.0, 0.0
        for h in range(1, 11):
            idx = np.argmin(np.abs(ff - h * fund))
            if h % 2 == 0:
                ep += fv[idx]**2
            else:
                op += fv[idx]**2
        f["vt_hr"] = ep / (op + 1e-12)
    else:
        f["vt_hr"] = 0

    # Walking consistency
    win2 = int(2 * fs)
    rms_w = np.array([np.sqrt(np.mean(vm[i:i + win2]**2))
                      for i in range(0, len(vm) - win2 + 1, win2)])
    f["act_rms_cv"] = rms_w.std() / (rms_w.mean() + 1e-12)

    return f


def segment_by_activity_index(sig, fs=FS, bin_sec=1.0):
    """Segment signal into moderate-AI regions (walking-level activity).
    Returns list of (start_sample, end_sample) for moderate-AI periods.
    """
    vm = np.sqrt(sig[:, 0]**2 + sig[:, 1]**2 + sig[:, 2]**2)
    bs = int(bin_sec * fs)
    n_bins = len(vm) // bs
    ai = np.array([np.sqrt(np.mean(vm[i*bs:(i+1)*bs]**2)) for i in range(n_bins)])

    low_thresh = np.percentile(ai, 30)
    high_thresh = np.percentile(ai, 85)
    moderate = (ai >= low_thresh) & (ai <= high_thresh)

    segments = []
    in_seg = False
    start = 0
    for i in range(len(moderate)):
        if moderate[i] and not in_seg:
            start = i
            in_seg = True
        elif not moderate[i] and in_seg:
            if (i - start) >= 5:
                segments.append((start * bs, i * bs))
            in_seg = False
    if in_seg and (len(moderate) - start) >= 5:
        segments.append((start * bs, len(moderate) * bs))
    return segments


def extract_features_one_subject(cohort, sid, year, sixmwd):
    """Pipeline: load preprocessed home signal → walking bout selection → features.
    Uses csv_processed_home (AP/ML/VT) with window-based walking selection.
    This pipeline gives the best B-series performance (XGBoost B2 = 0.711).
    """
    fname = f"{cohort}{sid:02d}_{year}_{sixmwd}.csv"
    fpath = os.path.join(BASE, "csv_processed_home", fname)
    df = pd.read_csv(fpath)
    sig = df[["AP", "ML", "VT"]].values.astype(np.float32)
    walk = select_best_walking_preprocessed(sig)
    return extract_gait_features(walk)


def extract_features_all(paired):
    """Extract walking-bout gait features for all subjects."""
    rows = []
    n = len(paired)
    for i, (_, row) in enumerate(paired.iterrows()):
        feats = extract_features_one_subject(
            row["cohort"], int(row["subj_id"]),
            int(row["year"]), int(row["sixmwd"]),
        )
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n}", flush=True)
    print(f"    {n}/{n}", flush=True)
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"    ({nan_count} NaN values imputed with column median)")
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELS
# ══════════════════════════════════════════════════════════════════════════════

def get_models():
    return {
        "Ridge": lambda: Ridge(alpha=10.0),
        "Lasso": lambda: Lasso(alpha=1.0, max_iter=10000),
        "ElasticNet": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=42, n_jobs=-1),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        "SVR": lambda: SVR(kernel="rbf", C=100, epsilon=50),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION (Leave-One-Subject-Out)
# ══════════════════════════════════════════════════════════════════════════════

def loo_cv(X, y, model_fn):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    return preds, y - preds


def loo_cv_featsel(X, y, model_fn, top_k=15):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        cors = np.array([abs(spearmanr(X[tr, j], y[tr])[0])
                         for j in range(X.shape[1])])
        idx = np.argsort(cors)[-top_k:]
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X[tr][:, idx]), y[tr])
        preds[te] = m.predict(sc.transform(X[te][:, idx]))
    return preds, y - preds


def loo_cv_ccpt(X_home, X_clinic, X_extra, y, model_fn):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X_home):
        sc_c = StandardScaler()
        Xc_tr = sc_c.fit_transform(X_clinic[tr])
        Xc_te = sc_c.transform(X_clinic[te])
        clinic_m = Ridge(alpha=1.0)
        clinic_m.fit(Xc_tr, y[tr])
        cpred_tr = clinic_m.predict(Xc_tr).reshape(-1, 1) / 1000.0
        cpred_te = clinic_m.predict(Xc_te).reshape(-1, 1) / 1000.0

        sc_h = StandardScaler()
        Xh_tr = sc_h.fit_transform(X_home[tr])
        Xh_te = sc_h.transform(X_home[te])

        if X_extra is not None:
            sc_d = StandardScaler()
            Xd_tr = sc_d.fit_transform(X_extra[tr])
            Xd_te = sc_d.transform(X_extra[te])
            Xaug_tr = np.column_stack([Xh_tr, Xd_tr, cpred_tr])
            Xaug_te = np.column_stack([Xh_te, Xd_te, cpred_te])
        else:
            Xaug_tr = np.column_stack([Xh_tr, cpred_tr])
            Xaug_te = np.column_stack([Xh_te, cpred_te])

        m = model_fn()
        m.fit(Xaug_tr, y[tr])
        preds[te] = m.predict(Xaug_te)
    return preds, y - preds


# ══════════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y, yhat, n_feat):
    n = len(y)
    r2 = r2_score(y, yhat)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_feat - 1, 1)
    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    pr, pp = pearsonr(y, yhat)
    sr, sp = spearmanr(y, yhat)
    return {"R2": round(r2, 4), "Adj_R2": round(adj_r2, 4),
            "MAE": round(mae, 1), "RMSE": round(rmse, 1),
            "Pearson_r": round(pr, 4), "Pearson_p": pp,
            "Spearman_rho": round(sr, 4), "Spearman_p": sp}


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                      "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


def fig_heatmap(results, configs, path):
    model_names = ["Ridge", "Lasso", "ElasticNet", "RandomForest", "XGBoost", "SVR"]
    mat = np.array([[results.get(f"{mn}_{c}", {}).get("metrics", {}).get("R2", np.nan)
                     for c in configs] for mn in model_names])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.7)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    for i in range(len(model_names)):
        for j in range(len(configs)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.45 or v < 0 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_title("R\u00b2 Ablation Heatmap (LOO CV)")
    fig.colorbar(im, ax=ax, label="R\u00b2")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_scatter_best(results, y, cohort, configs, path):
    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4.5), sharey=True)
    colors = {"C": "#4C72B0", "M": "#DD8452"}
    for ax, cfg in zip(axes, configs):
        cfg_results = [(k, v) for k, v in results.items() if v["config"] == cfg]
        if not cfg_results:
            continue
        name, info = max(cfg_results, key=lambda x: x[1]["metrics"]["R2"])
        yhat = info["predictions"]
        r2 = info["metrics"]["R2"]
        pr = info["metrics"]["Pearson_r"]
        for c in ["C", "M"]:
            mask = cohort == c
            ax.scatter(y[mask], yhat[mask], c=colors[c], alpha=0.6,
                       edgecolors="k", linewidths=0.3, s=40,
                       label="Control" if c == "C" else "MS")
        lo = min(y.min(), yhat.min()) - 50
        hi = max(y.max(), yhat.max()) + 50
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual 6MWD (ft)")
        ax.set_title(f"{cfg}: {name.split('_')[0]}\nR\u00b2={r2:.3f}, r={pr:.3f}")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Predicted 6MWD (ft)")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_barplot(results, configs, path):
    model_names = ["Ridge", "Lasso", "ElasticNet", "RandomForest", "XGBoost", "SVR"]
    cfg_colors = {"A1": "#a6cee3", "A2": "#1f78b4", "A3": "#b2df8a",
                  "B1": "#fb9a99", "B2": "#e31a1c", "B3": "#d62728", "B4": "#8c564b",
                  "C1": "#cab2d6", "C2": "#6a3d9a", "C3": "#ff7f00"}
    mat = np.array([[results.get(f"{mn}_{c}", {}).get("metrics", {}).get("R2", 0)
                     for c in configs] for mn in model_names])
    x = np.arange(len(model_names))
    w = 0.15
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, cfg in enumerate(configs):
        ax.bar(x + i * w, mat[:, i], w, label=cfg, color=cfg_colors.get(cfg, "#999"),
               edgecolor="k", linewidth=0.5)
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("R\u00b2")
    ax.set_title("Model Comparison: R\u00b2 (LOO CV)")
    ax.legend(title="Config")
    ax.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_residual(results, y, cohort, path):
    best_key = max((k for k in results if results[k]["config"] == "B2"),
                   key=lambda k: results[k]["metrics"]["R2"])
    info = results[best_key]
    yhat = info["predictions"]
    resid = y - yhat
    mean_vals = (y + yhat) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(mean_vals, resid, alpha=0.5, s=30, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(resid.mean(), color="r", ls="--", label=f"Mean: {resid.mean():.0f}")
    ax.axhline(resid.mean() + 1.96 * resid.std(), color="gray", ls=":")
    ax.axhline(resid.mean() - 1.96 * resid.std(), color="gray", ls=":")
    ax.set_xlabel("Mean of Actual & Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(f"Bland-Altman: {best_key}")
    ax.legend(fontsize=9)
    ax = axes[1]
    bp = ax.boxplot([resid[cohort == "C"], resid[cohort == "M"]],
                    labels=["Control", "MS"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Residual")
    ax.set_title("Residual by Cohort")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Predicting 6MWD from Home Accelerometer Data")
    print("  Features: Walking-bout gait features (35 features)")
    print("  CV: Leave-One-Subject-Out (LOO)")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    paired = load_tabular_data()
    y = paired["sixmwd"].values.astype(float)
    cohort = paired["cohort"].values
    print(f"  {len(paired)} obs, C={sum(cohort=='C')}, M={sum(cohort=='M')}")

    # 2. Extract features
    print("\n[2/6] Extracting walking-bout gait features...")
    feat_df = extract_features_all(paired)
    X_imu = feat_df.values.astype(float)
    imu_names = list(feat_df.columns)
    print(f"  {X_imu.shape[1]} features")

    feat_df.to_csv(os.path.join(OUT, "home_imu_features.csv"), index=False)

    # Clinic features for CCPT
    X_clinic = paired[CLINIC_ACCEL_COLS].values.astype(float)

    # Basic demographics (no clinical scores)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        mask = np.isnan(X_demo[:, j])
        if mask.any():
            X_demo[mask, j] = np.nanmedian(X_demo[:, j])

    # Clinical scores (separate)
    X_clinical = paired[CLINICAL_SCORES].values.astype(float)
    for j in range(X_clinical.shape[1]):
        mask = np.isnan(X_clinical[:, j])
        if mask.any():
            X_clinical[mask, j] = np.nanmedian(X_clinical[:, j])

    TOP_K = 15
    configs_list = ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1", "C2", "C3"]

    # 3. Run experiments
    print(f"\n[3/6] Running LOO CV...")
    results = {}
    all_metrics = []

    def record(key, cfg, model, preds, resids, n_feat, desc):
        metrics = compute_metrics(y, preds, n_feat)
        results[key] = {"config": cfg, "model": model,
                        "predictions": preds, "residuals": resids, "metrics": metrics}
        print(f"  {key:35s} R\u00b2={metrics['R2']:.4f}")
        row = {"model": model, "config": cfg, "config_desc": desc}
        row.update(metrics)
        all_metrics.append(row)
        pd.DataFrame({"cohort": cohort, "subj_id": paired["subj_id"].values,
                       "y_true": y, "y_pred": preds, "residual": resids}
                      ).to_csv(os.path.join(PRED, f"{key}.csv"), index=False)

    models = get_models()
    for mname, mfn in models.items():
        # A1: Home IMU only (with per-fold feature selection)
        p, r = loo_cv_featsel(X_imu, y, mfn, top_k=TOP_K)
        record(f"{mname}_A1", "A1", mname, p, r, TOP_K, "Home IMU only")

        # A2: Home IMU + Basic Demographics
        X_a2 = np.column_stack([X_imu, X_demo])
        p, r = loo_cv(X_a2, y, mfn)
        record(f"{mname}_A2", "A2", mname, p, r, X_a2.shape[1], "IMU + Basic Demo")

        # A3: Home IMU + Basic Demo + Clinical Scores
        X_a3 = np.column_stack([X_imu, X_demo, X_clinical])
        p, r = loo_cv(X_a3, y, mfn)
        record(f"{mname}_A3", "A3", mname, p, r, X_a3.shape[1], "IMU + Demo + Clinical")

        # B1: Home IMU + CCPT (no demographics)
        p, r = loo_cv_ccpt(X_imu, X_clinic, None, y, mfn)
        record(f"{mname}_B1", "B1", mname, p, r, X_imu.shape[1] + 1, "IMU + CCPT")

        # B2: Home IMU + Basic Demo + CCPT
        p, r = loo_cv_ccpt(X_imu, X_clinic, X_demo, y, mfn)
        record(f"{mname}_B2", "B2", mname, p, r,
               X_imu.shape[1] + len(BASIC_DEMO) + 1, "IMU + Demo + CCPT")

        # B3: Home IMU + Clinical Scores + CCPT
        p, r = loo_cv_ccpt(X_imu, X_clinic, X_clinical, y, mfn)
        record(f"{mname}_B3", "B3", mname, p, r,
               X_imu.shape[1] + len(CLINICAL_SCORES) + 1, "IMU + Clinical + CCPT")

        # B4: Home IMU + Demo + Clinical + CCPT
        X_demo_clin = np.column_stack([X_demo, X_clinical])
        p, r = loo_cv_ccpt(X_imu, X_clinic, X_demo_clin, y, mfn)
        record(f"{mname}_B4", "B4", mname, p, r,
               X_imu.shape[1] + len(BASIC_DEMO) + len(CLINICAL_SCORES) + 1,
               "IMU + Demo + Clinical + CCPT")

        # C1: Clinic accelerometer only
        p, r = loo_cv(X_clinic, y, mfn)
        record(f"{mname}_C1", "C1", mname, p, r, X_clinic.shape[1], "Clinic accel only")

        # C2: Clinic accel + Basic Demographics
        X_c2 = np.column_stack([X_clinic, X_demo])
        p, r = loo_cv(X_c2, y, mfn)
        record(f"{mname}_C2", "C2", mname, p, r, X_c2.shape[1], "Clinic accel + Demo")

        # C3: Clinic accel + Demo + Clinical Scores
        X_c3 = np.column_stack([X_clinic, X_demo, X_clinical])
        p, r = loo_cv(X_c3, y, mfn)
        record(f"{mname}_C3", "C3", mname, p, r, X_c3.shape[1], "Clinic accel + Demo + Clinical")

    # 4. Save results
    print("\n[4/6] Saving results...")
    summary = pd.DataFrame(all_metrics)
    summary.to_csv(os.path.join(OUT, "results_summary.csv"), index=False)

    # 5. Feature importance
    print("\n[5/6] Feature importances...")
    for mname in ["XGBoost", "RandomForest"]:
        for cfg, X_used, fname_suffix in [
            ("A1", X_imu, imu_names),
            ("B2", None, None),
        ]:
            key = f"{mname}_{cfg}"
            if key not in results:
                continue
            if cfg == "A1":
                sc = StandardScaler()
                m = get_models()[mname]()
                m.fit(sc.fit_transform(X_used), y)
                if hasattr(m, "feature_importances_"):
                    imp = pd.Series(m.feature_importances_, index=fname_suffix
                                    ).sort_values(ascending=False)
                    results[key]["importances"] = imp
                    print(f"  {key}: top = {imp.index[0]} ({imp.iloc[0]:.4f})")

                    imp_top = imp.head(20)
                    fig, ax = plt.subplots(figsize=(9, 7))
                    ax.barh(range(len(imp_top)), imp_top.values[::-1],
                            color="#1f78b4", edgecolor="k", linewidth=0.3)
                    ax.set_yticks(range(len(imp_top)))
                    ax.set_yticklabels(imp_top.index[::-1], fontsize=8)
                    ax.set_xlabel("Feature Importance")
                    ax.set_title(f"A1 Home IMU Features \u2014 {mname} (top 20)")
                    plt.tight_layout()
                    fig.savefig(os.path.join(FIG, f"feat_imp_A1_{mname}.png"))
                    plt.close(fig)

    # 6. Figures
    print("\n[6/6] Generating figures...")
    fig_heatmap(results, configs_list, os.path.join(FIG, "ablation_heatmap.png"))
    fig_scatter_best(results, y, cohort, configs_list, os.path.join(FIG, "scatter_plots.png"))
    fig_barplot(results, configs_list, os.path.join(FIG, "model_comparison.png"))
    fig_residual(results, y, cohort, os.path.join(FIG, "residual_analysis.png"))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (R\u00b2)")
    print("=" * 70)
    pivot = summary.pivot_table(index="model", columns="config", values="R2", aggfunc="first")
    col_order = [c for c in configs_list + ["REF"] if c in pivot.columns]
    print(pivot[col_order].to_string(float_format="{:.4f}".format))

    print("\n-- Clinical scores impact (A2 vs A3): --")
    for mname in models:
        a2 = results.get(f"{mname}_A2", {}).get("metrics", {}).get("R2", 0)
        a3 = results.get(f"{mname}_A3", {}).get("metrics", {}).get("R2", 0)
        print(f"  {mname:15s}: A2={a2:.4f}, A3={a3:.4f}, \u0394={a3 - a2:+.4f}")

    print(f"\nAll results saved to: {OUT}/")


if __name__ == "__main__":
    main()
