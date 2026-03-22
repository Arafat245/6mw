#!/usr/bin/env python3
"""
Predict 6MWD from Clinic Accelerometer Data (C series)
======================================================
Uses pre-extracted clinic accelerometer features from structured 6MW test.

Ablation:
  C1 = Clinic accel features only
  C2 = Clinic accel + Basic Demographics
  C3 = Clinic accel + Demographics + Clinical Scores

CV: Leave-One-Subject-Out (LOO)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
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

CLINIC_ACCEL_COLS = [
    "cadence_hz_c", "step_time_cv_pct_c", "acf_step_regularity_c",
    "hr_ap_c", "hr_vt_c", "ml_rms_g_c", "ml_spectral_entropy_c",
    "jerk_mean_abs_gps_c", "enmo_mean_g_c", "cadence_slope_per_min_c",
]
BASIC_DEMO = ["cohort_M", "Age", "Sex", "Height", "Weight", "BMI"]
CLINICAL_SCORES = ["BDI Raw Score", "MFIS Total"]


def load_data():
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


def loo_cv(X, y, model_fn):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    return preds, y - preds


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


def main():
    print("=" * 70)
    print("Predicting 6MWD from Clinic Accelerometer Data (C series)")
    print("  CV: Leave-One-Subject-Out (LOO)")
    print("=" * 70)

    paired = load_data()
    y = paired["sixmwd"].values.astype(float)
    cohort = paired["cohort"].values
    print(f"  {len(paired)} obs, C={sum(cohort=='C')}, M={sum(cohort=='M')}")

    X_clinic = paired[CLINIC_ACCEL_COLS].values.astype(float)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        mask = np.isnan(X_demo[:, j])
        if mask.any():
            X_demo[mask, j] = np.nanmedian(X_demo[:, j])
    X_clinical = paired[CLINICAL_SCORES].values.astype(float)
    for j in range(X_clinical.shape[1]):
        mask = np.isnan(X_clinical[:, j])
        if mask.any():
            X_clinical[mask, j] = np.nanmedian(X_clinical[:, j])

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

    print("\nRunning LOO CV...")
    models = get_models()
    for mname, mfn in models.items():
        # C1: Clinic accel only
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

    # Save
    summary = pd.DataFrame(all_metrics)
    summary.to_csv(os.path.join(OUT, "results_clinic.csv"), index=False)

    # Feature importance
    for mname in ["XGBoost", "RandomForest"]:
        sc = StandardScaler()
        m = get_models()[mname]()
        m.fit(sc.fit_transform(X_clinic), y)
        if hasattr(m, "feature_importances_"):
            imp = pd.Series(m.feature_importances_,
                            index=[c.replace("_c", "") for c in CLINIC_ACCEL_COLS]
                            ).sort_values(ascending=False)
            print(f"\n  {mname} C1 top features: {imp.head(5).to_dict()}")

            imp_top = imp.head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(range(len(imp_top)), imp_top.values[::-1],
                    color="#6a3d9a", edgecolor="k", linewidth=0.3)
            ax.set_yticks(range(len(imp_top)))
            ax.set_yticklabels(imp_top.index[::-1], fontsize=9)
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"C1 Clinic Features \u2014 {mname}")
            plt.tight_layout()
            fig.savefig(os.path.join(FIG, f"feat_imp_C1_{mname}.png"))
            plt.close(fig)

    # Summary
    print("\n" + "=" * 70)
    print("CLINIC RESULTS (R\u00b2)")
    print("=" * 70)
    pivot = summary.pivot_table(index="model", columns="config", values="R2", aggfunc="first")
    print(pivot[["C1", "C2", "C3"]].to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/results_clinic.csv")


if __name__ == "__main__":
    main()
