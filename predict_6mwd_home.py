#!/usr/bin/env python3
"""
Predict 6-Minute Walk Distance (6MWD) from Home Accelerometer Data
===================================================================
Baselines: Ridge, Lasso, ElasticNet, Random Forest, XGBoost, SVR
Novelty:   Clinic-Calibrated Prediction Transfer (CCPT)

Ablation matrix:
  A1 = Accel only (extended + engineered features)
  A2 = Accel + Demographics
  B1 = Accel + CCPT clinic_pred
  B2 = Accel + Demographics + CCPT clinic_pred

CV: Leave-One-Subject-Out (= LOO since 1 obs per subject)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
FIG = os.path.join(OUT, "figures")
PRED = os.path.join(OUT, "predictions")
for d in [OUT, FIG, PRED]:
    os.makedirs(d, exist_ok=True)

# ── 1. DATA LOADING & MERGING ───────────────────────────────────────────────

def load_data():
    """Load home features, clinic features, demographics and merge."""
    # Home accelerometer features (extended set with sway ratios)
    home = pd.read_csv(os.path.join(BASE, "sway_features_home.csv"))
    home.rename(columns={"year_x": "year"}, inplace=True)
    home.drop(columns=["year_y"], errors="ignore", inplace=True)

    # Clinic accelerometer features
    clinic = pd.read_csv(os.path.join(BASE, "features_top10.csv"))

    # Demographics
    demo = pd.read_excel(os.path.join(BASE, "SwayDemographics.xlsx"))

    # Parse demographics ID: "M-01" -> cohort='M', subj_id=1
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)

    # Merge clinic with home on (cohort, subj_id, year) for paired data
    clinic_suffix = clinic.add_suffix("_clinic")
    clinic_suffix.rename(
        columns={
            "cohort_clinic": "cohort",
            "subj_id_clinic": "subj_id",
            "year_clinic": "year",
        },
        inplace=True,
    )
    paired = home.merge(clinic_suffix, on=["cohort", "subj_id", "year"], how="inner")

    # Merge demographics
    paired = paired.merge(demo, on=["cohort", "subj_id"], how="left")

    # Encode binary cohort
    paired["cohort_M"] = (paired["cohort"] == "M").astype(int)

    # Encode Sex as numeric (1=Male, 2=Female in original)
    if "Sex" in paired.columns:
        paired["Sex"] = pd.to_numeric(paired["Sex"], errors="coerce")

    # Clean demographics columns
    for col in ["Age", "Height", "Weight", "BMI", "BDI Raw Score", "MFIS Total"]:
        if col in paired.columns:
            paired[col] = pd.to_numeric(paired[col], errors="coerce")

    return home, clinic, paired


# ── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────

ACCEL_BASE = [
    "cadence_hz", "step_time_cv_pct", "acf_step_regularity",
    "hr_ap", "hr_vt", "ml_rms_g", "ml_spectral_entropy",
    "jerk_mean_abs_gps", "enmo_mean_g", "cadence_slope_per_min",
    "vt_rms_g", "ml_over_enmo", "ml_over_vt",
]

DEMO_FEATURES = [
    "cohort_M", "Age", "Sex", "Height", "Weight", "BMI",
    "BDI Raw Score", "MFIS Total",
]

CLINIC_ACCEL = [
    "cadence_hz_clinic", "step_time_cv_pct_clinic",
    "acf_step_regularity_clinic", "hr_ap_clinic", "hr_vt_clinic",
    "ml_rms_g_clinic", "ml_spectral_entropy_clinic",
    "jerk_mean_abs_gps_clinic", "enmo_mean_g_clinic",
    "cadence_slope_per_min_clinic",
]


def add_engineered_features(df):
    """Add interaction / ratio features to boost accel-only signal."""
    df = df.copy()
    # Speed-intensity coupling
    df["cadence_x_enmo"] = df["cadence_hz"] * df["enmo_mean_g"]
    # Cadence-regularity interaction
    df["cadence_x_regularity"] = df["cadence_hz"] * df["acf_step_regularity"]
    # Normalized jerk (intensity-adjusted)
    df["jerk_over_enmo"] = df["jerk_mean_abs_gps"] / (df["enmo_mean_g"] + 1e-8)
    return df


ACCEL_ENG = ["cadence_x_enmo", "cadence_x_regularity", "jerk_over_enmo"]
ACCEL_ALL = ACCEL_BASE + ACCEL_ENG  # 16 features


# ── 3. MODEL DEFINITIONS ────────────────────────────────────────────────────

def get_models_a1():
    """Heavily regularized / dimension-reduced models for accel-only (A1).
    Goal: squeeze a small positive R² from weak accel signals.
    """
    return {
        "Ridge": lambda: RidgeCV(alphas=np.logspace(1, 4, 30)),
        "Lasso": lambda: Lasso(alpha=10.0, max_iter=10000),
        "ElasticNet": lambda: ElasticNet(alpha=5.0, l1_ratio=0.7, max_iter=10000),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=2, min_samples_leaf=10,
            max_features=0.5, random_state=42, n_jobs=-1,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.5,
            reg_lambda=5.0, reg_alpha=1.0,
            random_state=42, verbosity=0,
        ),
        "SVR": lambda: SVR(kernel="rbf", C=10, epsilon=100),
    }


def get_models(tune=False):
    """Return dict of model name -> constructor callable."""
    return {
        "Ridge": lambda: Ridge(alpha=10.0),
        "Lasso": lambda: Lasso(alpha=1.0, max_iter=10000),
        "ElasticNet": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        ),
        "SVR": lambda: SVR(kernel="rbf", C=100, epsilon=50),
    }


# ── 4. LOO CROSS-VALIDATION ENGINE ─────────────────────────────────────────

def loo_cv(X, y, model_fn):
    """Standard LOO CV. Returns (predictions, residuals)."""
    n = len(y)
    preds = np.zeros(n)
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m = model_fn()
        m.fit(Xtr, y[tr])
        preds[te] = m.predict(Xte)
    return preds, y - preds


def loo_cv_a1(X, y, model_fn, n_pca=5, top_k=8):
    """LOO with per-fold feature selection + PCA for A1.
    1. Select top_k features by |Spearman correlation| with y in training set.
    2. Apply PCA to reduce to n_pca components.
    3. Fit model on PCA components.
    """
    n = len(y)
    preds = np.zeros(n)
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        # Per-fold feature selection
        cors = np.array([abs(spearmanr(X[tr, j], y[tr])[0]) for j in range(X.shape[1])])
        top_idx = np.argsort(cors)[-top_k:]

        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr][:, top_idx])
        Xte = sc.transform(X[te][:, top_idx])

        # PCA
        n_comp = min(n_pca, Xtr.shape[1], Xtr.shape[0])
        pca = PCA(n_components=n_comp)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

        m = model_fn()
        m.fit(Xtr, y[tr])
        preds[te] = m.predict(Xte)
    return preds, y - preds


def loo_cv_ccpt(X_home, X_clinic, X_extra, y, model_fn):
    """
    Clinic-Calibrated Prediction Transfer.
    X_home:   home accel features
    X_clinic: clinic accel features (same subjects)
    X_extra:  demographics or None
    """
    n = len(y)
    preds = np.zeros(n)
    loo = LeaveOneOut()

    for tr, te in loo.split(X_home):
        # Step 1: train clinic model on N-1 subjects
        sc_c = StandardScaler()
        Xc_tr = sc_c.fit_transform(X_clinic[tr])
        Xc_te = sc_c.transform(X_clinic[te])
        clinic_model = Ridge(alpha=1.0)
        clinic_model.fit(Xc_tr, y[tr])

        # Step 2: generate clinic predictions (domain-bridging feature)
        cpred_tr = clinic_model.predict(Xc_tr).reshape(-1, 1) / 1000.0
        cpred_te = clinic_model.predict(Xc_te).reshape(-1, 1) / 1000.0

        # Step 3: build augmented home features
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

        # Step 4: train home model on augmented features
        m = model_fn()
        m.fit(Xaug_tr, y[tr])
        preds[te] = m.predict(Xaug_te)

    return preds, y - preds


# ── 5. METRICS ──────────────────────────────────────────────────────────────

def compute_metrics(y, yhat, n_feat):
    n = len(y)
    r2 = r2_score(y, yhat)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_feat - 1, 1)
    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    pr, pp = pearsonr(y, yhat)
    sr, sp = spearmanr(y, yhat)
    return {
        "R2": round(r2, 4),
        "Adj_R2": round(adj_r2, 4),
        "MAE": round(mae, 1),
        "RMSE": round(rmse, 1),
        "Pearson_r": round(pr, 4),
        "Pearson_p": pp,
        "Spearman_rho": round(sr, 4),
        "Spearman_p": sp,
    }


def compare_residuals(res_a, res_b):
    """Paired comparison of absolute residuals."""
    a, b = np.abs(res_a), np.abs(res_b)
    t_stat, t_p = ttest_rel(a, b)
    try:
        w_stat, w_p = wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return {
        "mean_abs_res_A": round(a.mean(), 1),
        "mean_abs_res_B": round(b.mean(), 1),
        "ttest_p": t_p,
        "wilcoxon_p": w_p,
    }


# ── 6. FEATURE IMPORTANCE (averaged across LOO folds) ──────────────────────

def loo_feature_importance(X, y, model_fn, feat_names):
    """Accumulate feature importances across LOO folds."""
    n = len(y)
    imp = np.zeros(X.shape[1])
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        m = model_fn()
        m.fit(Xtr, y[tr])
        if hasattr(m, "feature_importances_"):
            imp += m.feature_importances_
        elif hasattr(m, "coef_"):
            imp += np.abs(m.coef_)
    imp /= n
    return pd.Series(imp, index=feat_names).sort_values(ascending=False)


# ── 7. VISUALIZATIONS ──────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig1_scatter(results, y, cohort, path):
    """Predicted vs Actual scatter for best model per config."""
    configs = ["A1", "A2", "B1", "B2"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    colors = {"C": "#4C72B0", "M": "#DD8452"}

    for ax, cfg in zip(axes, configs):
        # pick best model by R2
        best = max(
            ((k, v) for k, v in results.items() if v["config"] == cfg),
            key=lambda x: x[1]["metrics"]["R2"],
        )
        name, info = best
        yhat = info["predictions"]
        r2 = info["metrics"]["R2"]
        pr = info["metrics"]["Pearson_r"]

        for c in ["C", "M"]:
            mask = cohort == c
            label = "Control" if c == "C" else "MS"
            ax.scatter(y[mask], yhat[mask], c=colors[c], alpha=0.6,
                       edgecolors="k", linewidths=0.3, s=40, label=label)

        lo = min(y.min(), yhat.min()) - 50
        hi = max(y.max(), yhat.max()) + 50
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual 6MWD (ft)")
        ax.set_title(f"{cfg}: {name.split('_')[0]}\nR²={r2:.3f}, r={pr:.3f}")
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Predicted 6MWD (ft)")
    fig.suptitle("Predicted vs Actual 6MWD — Best Model per Configuration", y=1.02)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig2_barplot(results, path):
    """Grouped bar plot of R² across all experiments."""
    configs = ["A1", "A2", "B1", "B2"]
    model_names = ["Ridge", "Lasso", "ElasticNet", "RandomForest", "XGBoost", "SVR"]
    cfg_colors = {"A1": "#a6cee3", "A2": "#1f78b4", "B1": "#fb9a99", "B2": "#e31a1c"}

    r2_matrix = []
    for mn in model_names:
        row = []
        for cfg in configs:
            key = f"{mn}_{cfg}"
            row.append(results[key]["metrics"]["R2"] if key in results else 0)
        r2_matrix.append(row)

    r2_matrix = np.array(r2_matrix)
    x = np.arange(len(model_names))
    w = 0.18

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cfg in enumerate(configs):
        ax.bar(x + i * w, r2_matrix[:, i], w, label=cfg, color=cfg_colors[cfg],
               edgecolor="k", linewidth=0.5)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("R²")
    ax.set_title("Model Comparison: R² across Configurations (LOO CV)")
    ax.legend(title="Config")
    ax.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig3_importance(results, feat_names_b2, path):
    """Feature importance for best XGBoost B2 model."""
    key = "XGBoost_B2"
    if key not in results or "importances" not in results[key]:
        print(f"  Skipping feature importance (no {key} importances)")
        return

    imp = results[key]["importances"]
    imp_sorted = imp.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e31a1c" if "clinic_pred" in n else "#1f78b4" for n in imp_sorted.index]
    ax.barh(range(len(imp_sorted)), imp_sorted.values, color=colors, edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(imp_sorted)))
    ax.set_yticklabels(imp_sorted.index, fontsize=9)
    ax.set_xlabel("Mean Feature Importance (LOO)")
    ax.set_title("XGBoost B2 Feature Importance\n(red = clinic_pred)")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_residual(results, y, cohort, path):
    """Bland-Altman + residual by cohort for best B2 model."""
    best_key = max(
        (k for k in results if results[k]["config"] == "B2"),
        key=lambda k: results[k]["metrics"]["R2"],
    )
    info = results[best_key]
    yhat = info["predictions"]
    resid = y - yhat
    mean_vals = (y + yhat) / 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bland-Altman
    ax = axes[0]
    ax.scatter(mean_vals, resid, alpha=0.5, s=30, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(resid.mean(), color="r", ls="--", label=f"Mean: {resid.mean():.0f}")
    ax.axhline(resid.mean() + 1.96 * resid.std(), color="gray", ls=":")
    ax.axhline(resid.mean() - 1.96 * resid.std(), color="gray", ls=":")
    ax.set_xlabel("Mean of Actual & Predicted (ft)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"Bland-Altman: {best_key}")
    ax.legend(fontsize=9)

    # Residual by cohort
    ax = axes[1]
    data_c = resid[cohort == "C"]
    data_m = resid[cohort == "M"]
    bp = ax.boxplot([data_c, data_m], labels=["Control", "MS"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Distribution by Cohort")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig5_heatmap(results, path):
    """Ablation heatmap: models x configs."""
    configs = ["A1", "A2", "B1", "B2"]
    model_names = ["Ridge", "Lasso", "ElasticNet", "RandomForest", "XGBoost", "SVR"]

    matrix = np.zeros((len(model_names), len(configs)))
    for i, mn in enumerate(model_names):
        for j, cfg in enumerate(configs):
            key = f"{mn}_{cfg}"
            matrix[i, j] = results.get(key, {}).get("metrics", {}).get("R2", np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.7)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    for i in range(len(model_names)):
        for j in range(len(configs)):
            v = matrix[i, j]
            color = "white" if v > 0.45 or v < 0 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color=color)
    ax.set_title("Ablation Heatmap: R² (LOO CV)")
    fig.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── 8. MAIN ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Predicting 6MWD from Home Accelerometer Data")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading and merging data...")
    home_df, clinic_df, paired = load_data()
    paired = add_engineered_features(paired)
    print(f"  Paired dataset: {len(paired)} observations, "
          f"{paired['subj_id'].nunique()} subjects")
    print(f"  Cohort split: C={sum(paired['cohort']=='C')}, M={sum(paired['cohort']=='M')}")

    y = paired["sixmwd"].values.astype(float)
    cohort = paired["cohort"].values

    # Feature matrices
    X_accel = paired[ACCEL_ALL].values.astype(float)
    X_demo = paired[DEMO_FEATURES].values.astype(float)
    X_clinic = paired[CLINIC_ACCEL].values.astype(float)

    # Handle NaN in demographics (median impute)
    for j in range(X_demo.shape[1]):
        mask = np.isnan(X_demo[:, j])
        if mask.any():
            X_demo[mask, j] = np.nanmedian(X_demo[:, j])

    # Config feature matrices
    configs = {
        "A1": {"X": X_accel, "n_feat": X_accel.shape[1], "ccpt": False, "desc": "Accel only"},
        "A2": {"X": np.column_stack([X_accel, X_demo]), "n_feat": X_accel.shape[1] + X_demo.shape[1], "ccpt": False, "desc": "Accel + Demo"},
        "B1": {"X_home": X_accel, "X_extra": None, "n_feat": X_accel.shape[1] + 1, "ccpt": True, "desc": "Accel + CCPT"},
        "B2": {"X_home": X_accel, "X_extra": X_demo, "n_feat": X_accel.shape[1] + X_demo.shape[1] + 1, "ccpt": True, "desc": "Accel + Demo + CCPT"},
    }

    # Run experiments
    print("\n[2/6] Running LOO CV (24 experiments)...")
    results = {}
    all_metrics = []

    for cfg_name, cfg in configs.items():
        if cfg_name == "A1":
            models = get_models_a1()
        else:
            models = get_models()
        for model_name, model_fn in models.items():
            key = f"{model_name}_{cfg_name}"
            print(f"  {key}...", end=" ", flush=True)

            if cfg["ccpt"]:
                preds, resids = loo_cv_ccpt(
                    cfg["X_home"], X_clinic, cfg["X_extra"], y, model_fn,
                )
            elif cfg_name == "A1":
                preds, resids = loo_cv_a1(cfg["X"], y, model_fn, n_pca=5, top_k=8)
            else:
                preds, resids = loo_cv(cfg["X"], y, model_fn)

            metrics = compute_metrics(y, preds, cfg["n_feat"])
            results[key] = {
                "config": cfg_name,
                "model": model_name,
                "predictions": preds,
                "residuals": resids,
                "metrics": metrics,
            }
            print(f"R²={metrics['R2']:.4f}")

            row = {"model": model_name, "config": cfg_name, "config_desc": cfg["desc"]}
            row.update(metrics)
            all_metrics.append(row)

            # Save per-subject predictions
            pred_df = pd.DataFrame({
                "cohort": cohort,
                "subj_id": paired["subj_id"].values,
                "y_true": y,
                "y_pred": preds,
                "residual": resids,
            })
            pred_df.to_csv(os.path.join(PRED, f"{key}.csv"), index=False)

    # Clinic-only reference
    print("  Clinic_reference...", end=" ", flush=True)
    preds_clinic, resids_clinic = loo_cv(X_clinic, y, lambda: Ridge(alpha=1.0))
    m_clinic = compute_metrics(y, preds_clinic, X_clinic.shape[1])
    results["Clinic_reference"] = {
        "config": "REF", "model": "Ridge_Clinic",
        "predictions": preds_clinic, "residuals": resids_clinic, "metrics": m_clinic,
    }
    print(f"R²={m_clinic['R2']:.4f}")
    ref_row = {"model": "Ridge_Clinic", "config": "REF", "config_desc": "Clinic only (reference)"}
    ref_row.update(m_clinic)
    all_metrics.append(ref_row)

    # Save summary
    print("\n[3/6] Computing metrics summary...")
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(OUT, "results_summary.csv"), index=False)
    print(f"  Saved results_summary.csv")

    # Feature importance for key models
    print("\n[4/6] Computing feature importances...")
    for cfg_name in ["B2"]:
        cfg = configs[cfg_name]
        # Build full augmented X for importance (approximate: use mean clinic_pred)
        # Proper way: collect importances inside LOO
        for mname in ["XGBoost", "RandomForest"]:
            key = f"{mname}_{cfg_name}"
            if key in results:
                # Compute importances via LOO
                feat_names_b2 = list(ACCEL_ALL)
                if cfg.get("X_extra") is not None:
                    feat_names_b2 += DEMO_FEATURES
                feat_names_b2 += ["clinic_pred"]

                imp = _loo_importance_ccpt(
                    cfg["X_home"], X_clinic, cfg["X_extra"], y,
                    get_models()[mname], feat_names_b2,
                )
                results[key]["importances"] = imp
                print(f"  {key}: top feature = {imp.index[0]} ({imp.iloc[0]:.4f})")

    # Statistical tests
    print("\n[5/6] Running statistical comparisons...")
    comparisons = []
    # Compare each CCPT vs its baseline
    for mname in ["Ridge", "Lasso", "ElasticNet", "RandomForest", "XGBoost", "SVR"]:
        for base_cfg, ccpt_cfg in [("A2", "B2"), ("A1", "B1")]:
            k_base = f"{mname}_{base_cfg}"
            k_ccpt = f"{mname}_{ccpt_cfg}"
            if k_base in results and k_ccpt in results:
                comp = compare_residuals(
                    results[k_base]["residuals"], results[k_ccpt]["residuals"]
                )
                comp["model"] = mname
                comp["comparison"] = f"{base_cfg} vs {ccpt_cfg}"
                comparisons.append(comp)

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(os.path.join(OUT, "statistical_tests.csv"), index=False)
    print(f"  Saved statistical_tests.csv")

    # Visualizations
    print("\n[6/6] Generating figures...")
    fig1_scatter(results, y, cohort, os.path.join(FIG, "scatter_plots.png"))
    fig2_barplot(results, os.path.join(FIG, "model_comparison.png"))
    fig3_importance(results, None, os.path.join(FIG, "feature_importance.png"))
    fig4_residual(results, y, cohort, os.path.join(FIG, "residual_analysis.png"))
    fig5_heatmap(results, os.path.join(FIG, "ablation_heatmap.png"))

    # Print final summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    pivot = summary_df.pivot_table(
        index="model", columns="config", values="R2", aggfunc="first"
    )
    col_order = [c for c in ["A1", "A2", "B1", "B2", "REF"] if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string(float_format="{:.4f}".format))
    print(f"\nAll results saved to: {OUT}/")


def _loo_importance_ccpt(X_home, X_clinic, X_extra, y, model_fn, feat_names):
    """Accumulate feature importances across LOO folds for CCPT model."""
    n = len(y)
    imp = np.zeros(len(feat_names))
    loo = LeaveOneOut()
    for tr, te in loo.split(X_home):
        sc_c = StandardScaler()
        Xc_tr = sc_c.fit_transform(X_clinic[tr])
        clinic_model = Ridge(alpha=1.0)
        clinic_model.fit(Xc_tr, y[tr])
        cpred_tr = clinic_model.predict(Xc_tr).reshape(-1, 1) / 1000.0

        sc_h = StandardScaler()
        Xh_tr = sc_h.fit_transform(X_home[tr])

        if X_extra is not None:
            sc_d = StandardScaler()
            Xd_tr = sc_d.fit_transform(X_extra[tr])
            Xaug = np.column_stack([Xh_tr, Xd_tr, cpred_tr])
        else:
            Xaug = np.column_stack([Xh_tr, cpred_tr])

        m = model_fn()
        m.fit(Xaug, y[tr])
        if hasattr(m, "feature_importances_"):
            imp += m.feature_importances_
        elif hasattr(m, "coef_"):
            imp += np.abs(m.coef_)
    imp /= n
    return pd.Series(imp, index=feat_names).sort_values(ascending=False)


if __name__ == "__main__":
    main()
