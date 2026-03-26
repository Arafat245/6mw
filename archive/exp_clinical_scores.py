#!/usr/bin/env python3
"""
Predict Clinical Scores (MFIS Total, EDSS) from Accelerometer Features
=======================================================================
Targets: MFIS Total (n=103), EDSS (n=40 MS only)

ML Baselines: Ridge, ElasticNet, RF, XGBoost
Novelty 1: Attention-weighted multi-source feature fusion
Novelty 2: MOMENT embeddings + hand-crafted feature fusion

Configs: A1, A2, C1, C2
CV: LOO
"""
import os, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
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
FEATS = BASE / "feats"
OUT = BASE / "results_raw_pipeline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    ids = pd.read_csv(FEATS / "target_6mwd.csv")
    demo_raw = pd.read_excel(BASE / "SwayDemographics.xlsx")
    demo_raw["cohort"] = demo_raw["ID"].str.extract(r"^([A-Z])")[0]
    demo_raw["subj_id"] = demo_raw["ID"].str.extract(r"(\d+)")[0].astype(int)
    p = ids.merge(demo_raw, on=["cohort", "subj_id"], how="left")
    for c in ["MFIS Total", "EDSS Total", "Sex", "Age", "Height", "Weight", "BMI"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["cohort_M"] = (p["cohort"] == "M").astype(int)

    def load_feat(name):
        df = pd.read_csv(FEATS / name).drop(columns=["cohort","subj_id","year","sixmwd"], errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan)
        for c in df.columns:
            if df[c].isna().any(): df[c] = df[c].fillna(df[c].median())
        return df.values.astype(float), list(df.columns)

    X_act, act_n = load_feat("home_activity_profile.csv")
    X_gait, gait_n = load_feat("home_gait.csv")
    X_cwt, cwt_n = load_feat("home_cwt.csv")
    X_sway, sway_n = load_feat("home_sway.csv")
    X_clinic, clinic_n = load_feat("clinic_accel_features.csv")
    X_ccwt, ccwt_n = load_feat("clinic_cwt.csv")
    X_csway, csway_n = load_feat("clinic_sway.csv")

    demo_cols = ["cohort_M", "Age", "Sex", "Height", "Weight", "BMI"]
    X_demo = p[demo_cols].values.astype(float)
    for j in range(X_demo.shape[1]):
        m = np.isnan(X_demo[:, j])
        if m.any(): X_demo[m, j] = np.nanmedian(X_demo[:, j])

    # MOMENT embeddings (cached)
    E_mom_h = np.load(OUT / "emb_moment_home.npy") if (OUT / "emb_moment_home.npy").exists() else None
    E_mom_c = np.load(OUT / "emb_moment_clinic.npy") if (OUT / "emb_moment_clinic.npy").exists() else None

    return p, {
        "activity": X_act, "gait": X_gait, "cwt_h": X_cwt, "sway_h": X_sway,
        "clinic": X_clinic, "cwt_c": X_ccwt, "sway_c": X_csway,
        "demo": X_demo, "moment_h": E_mom_h, "moment_c": E_mom_c,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CV AND METRICS
# ══════════════════════════════════════════════════════════════════════════════

def loo(X, y, mfn):
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = mfn()
        m.fit(sc.fit_transform(X[tr]), y[tr]); p[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, p), pearsonr(y, p)[0], len(y)

def met_str(r2, r, n):
    return f"R²={r2:.4f} r={r:.4f} n={n}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. NOVELTY 1: ATTENTION-WEIGHTED FEATURE FUSION
# ══════════════════════════════════════════════════════════════════════════════

class AttentionFusion(nn.Module):
    def __init__(self, source_dims, hidden=16):
        super().__init__()
        self.n_sources = len(source_dims)
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU())
            for d in source_dims
        ])
        self.attn_w = nn.Linear(hidden * self.n_sources, self.n_sources)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(hidden, 1),
        )

    def forward(self, sources):
        projected = [proj(src) for proj, src in zip(self.projectors, sources)]
        stacked = torch.stack(projected, dim=1)  # (batch, n_sources, hidden)
        concat = torch.cat(projected, dim=1)  # (batch, n_sources*hidden)
        alpha = F.softmax(self.attn_w(concat), dim=1)  # (batch, n_sources)
        fused = (stacked * alpha.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return self.head(fused).squeeze(-1), alpha


def loo_attention(source_arrays, y, epochs=80, lr=5e-4):
    """LOO with attention fusion model."""
    mask = ~np.isnan(y)
    sources = [s[mask] for s in source_arrays]
    y_clean = y[mask]
    n = len(y_clean)
    dims = [s.shape[1] for s in sources]
    preds = np.zeros(n)
    all_alphas = np.zeros((n, len(sources)))

    for i in range(n):
        tr = [j for j in range(n) if j != i]
        scalers = [StandardScaler() for _ in sources]
        src_tr = [torch.from_numpy(sc.fit_transform(s[tr]).astype(np.float32)).to(DEVICE)
                  for sc, s in zip(scalers, sources)]
        src_te = [torch.from_numpy(sc.transform(s[i:i+1]).astype(np.float32)).to(DEVICE)
                  for sc, s in zip(scalers, sources)]
        yt = torch.from_numpy(y_clean[tr].astype(np.float32)).to(DEVICE)

        model = AttentionFusion(dims, hidden=16).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred, _ = model(src_tr)
            F.mse_loss(pred, yt).backward()
            opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            pred, alpha = model(src_te)
            preds[i] = pred.cpu().item()
            all_alphas[i] = alpha.cpu().numpy()

    r2 = r2_score(y_clean, preds)
    r = pearsonr(y_clean, preds)[0]
    return r2, r, n, all_alphas.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. NOVELTY 2: MOMENT + HAND-CRAFTED FUSION
# ══════════════════════════════════════════════════════════════════════════════

def loo_moment_fusion(moment_emb, handcrafted, demo, y, epochs=80, lr=5e-4):
    """LOO with MOMENT embeddings + hand-crafted features via attention."""
    from sklearn.decomposition import PCA
    mask = ~np.isnan(y)
    mom = moment_emb[mask]
    hc = handcrafted[mask]
    dm = demo[mask]
    y_clean = y[mask]

    # PCA reduce MOMENT
    mom_pca = PCA(n_components=min(30, mom.shape[1], len(y_clean)-2)).fit_transform(mom)

    sources = [mom_pca, hc, dm]
    dims = [s.shape[1] for s in sources]
    n = len(y_clean)
    preds = np.zeros(n)
    all_alphas = np.zeros((n, 3))

    for i in range(n):
        tr = [j for j in range(n) if j != i]
        scalers = [StandardScaler() for _ in sources]
        src_tr = [torch.from_numpy(sc.fit_transform(s[tr]).astype(np.float32)).to(DEVICE)
                  for sc, s in zip(scalers, sources)]
        src_te = [torch.from_numpy(sc.transform(s[i:i+1]).astype(np.float32)).to(DEVICE)
                  for sc, s in zip(scalers, sources)]
        yt = torch.from_numpy(y_clean[tr].astype(np.float32)).to(DEVICE)

        model = AttentionFusion(dims, hidden=16).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred, _ = model(src_tr)
            F.mse_loss(pred, yt).backward()
            opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            pred, alpha = model(src_te)
            preds[i] = pred.cpu().item()
            all_alphas[i] = alpha.cpu().numpy()

    r2 = r2_score(y_clean, preds)
    r = pearsonr(y_clean, preds)[0]
    return r2, r, n, all_alphas.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Predicting Clinical Scores: MFIS Total & EDSS")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    p, data = load_all()

    xgb = lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    models = {"Ridge": lambda: Ridge(alpha=10),
              "ElasticNet": lambda: ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000),
              "RF": lambda: RandomForestRegressor(n_estimators=200, max_depth=5,
                                                   min_samples_leaf=5, random_state=42, n_jobs=-1),
              "XGBoost": xgb}

    targets = {
        "MFIS_Total": p["MFIS Total"].values.astype(float),
        "EDSS": p["EDSS Total"].values.astype(float),
    }

    # Feature configs
    feat_configs = {
        # A series (home)
        "Activity|A1": data["activity"],
        "Gait|A1": data["gait"],
        "CWT_H|A1": data["cwt_h"],
        "Sway_H|A1": data["sway_h"],
        "Activity|A2": np.column_stack([data["activity"], data["demo"]]),
        "Gait|A2": np.column_stack([data["gait"], data["demo"]]),
        "CWT_H|A2": np.column_stack([data["cwt_h"], data["demo"]]),
        # C series (clinic)
        "Clinic|C1": data["clinic"],
        "CWT_C|C1": data["cwt_c"],
        "Sway_C|C1": data["sway_c"],
        "Clinic+CWT_C|C1": np.column_stack([data["clinic"], data["cwt_c"]]),
        "Clinic|C2": np.column_stack([data["clinic"], data["demo"]]),
        "CWT_C|C2": np.column_stack([data["cwt_c"], data["demo"]]),
        "Clinic+CWT_C|C2": np.column_stack([data["clinic"], data["cwt_c"], data["demo"]]),
    }

    for target_name, y in targets.items():
        n_valid = (~np.isnan(y)).sum()
        print(f"\n{'='*70}")
        print(f"TARGET: {target_name} (n={n_valid})")
        print(f"{'='*70}")

        results = []

        # ── ML Baselines ──
        print(f"\n  --- ML Baselines ---")
        for fc_name, X in feat_configs.items():
            feat_set, config = fc_name.split("|")
            for mname, mfn in models.items():
                r2, r, n = loo(X, y, mfn)
                results.append({"model": mname, "config": config, "features": feat_set,
                                "R2": round(r2, 4), "r": round(r, 4), "n": n})
                print(f"    {mname:12s} {config:4s} {feat_set:15s} R²={r2:.4f} r={r:.4f}")

        # ── Novelty 1: Attention Fusion (Home A2) ──
        print(f"\n  --- Novelty 1: Attention Fusion (Home) ---")
        src_names_h = ["Gait", "CWT", "Activity", "Demo"]
        r2, r, n, alphas = loo_attention(
            [data["gait"], data["cwt_h"], data["activity"], data["demo"]], y)
        results.append({"model": "AttnFusion", "config": "A2", "features": "All_Home",
                        "R2": round(r2, 4), "r": round(r, 4), "n": n})
        print(f"    AttnFusion   A2   All_Home        R²={r2:.4f} r={r:.4f}")
        print(f"    Attention weights: {dict(zip(src_names_h, [f'{a:.3f}' for a in alphas]))}")

        # Attention Fusion (Clinic C2)
        print(f"\n  --- Novelty 1: Attention Fusion (Clinic) ---")
        src_names_c = ["Clinic", "CWT_C", "Demo"]
        r2, r, n, alphas = loo_attention(
            [data["clinic"], data["cwt_c"], data["demo"]], y)
        results.append({"model": "AttnFusion", "config": "C2", "features": "All_Clinic",
                        "R2": round(r2, 4), "r": round(r, 4), "n": n})
        print(f"    AttnFusion   C2   All_Clinic      R²={r2:.4f} r={r:.4f}")
        print(f"    Attention weights: {dict(zip(src_names_c, [f'{a:.3f}' for a in alphas]))}")

        # ── Novelty 2: MOMENT + Hand-crafted Fusion ──
        if data["moment_h"] is not None:
            print(f"\n  --- Novelty 2: MOMENT + Feature Fusion (Home) ---")
            # Best home hand-crafted = activity (for MFIS) or gait (for EDSS)
            best_hc = data["activity"] if "MFIS" in target_name else data["gait"]
            r2, r, n, alphas = loo_moment_fusion(
                data["moment_h"], best_hc, data["demo"], y)
            results.append({"model": "MOMENT+Fusion", "config": "A2", "features": "MOMENT+HC",
                            "R2": round(r2, 4), "r": round(r, 4), "n": n})
            fm_names = ["MOMENT", "HandCrafted", "Demo"]
            print(f"    MOMENT+Fus   A2   MOMENT+HC       R²={r2:.4f} r={r:.4f}")
            print(f"    Attention weights: {dict(zip(fm_names, [f'{a:.3f}' for a in alphas]))}")

        if data["moment_c"] is not None:
            print(f"\n  --- Novelty 2: MOMENT + Feature Fusion (Clinic) ---")
            best_hc_c = np.column_stack([data["clinic"], data["cwt_c"]])
            r2, r, n, alphas = loo_moment_fusion(
                data["moment_c"], best_hc_c, data["demo"], y)
            results.append({"model": "MOMENT+Fusion", "config": "C2", "features": "MOMENT+HC",
                            "R2": round(r2, 4), "r": round(r, 4), "n": n})
            print(f"    MOMENT+Fus   C2   MOMENT+HC       R²={r2:.4f} r={r:.4f}")
            print(f"    Attention weights: {dict(zip(fm_names, [f'{a:.3f}' for a in alphas]))}")

        # Save
        df = pd.DataFrame(results)
        suffix = "mfis" if "MFIS" in target_name else "edss"
        df.to_csv(OUT / f"exp_clinical_{suffix}.csv", index=False)

        # Summary
        print(f"\n  --- BEST PER CONFIG ({target_name}) ---")
        for cfg in ["A1", "A2", "C1", "C2"]:
            sub = df[df["config"] == cfg]
            if sub.empty: continue
            best = sub.loc[sub["R2"].idxmax()]
            print(f"    {cfg}: R²={best['R2']:.4f} r={best['r']:.4f} ({best['model']}, {best['features']})")

    print(f"\nAll results saved to {OUT}/")


if __name__ == "__main__":
    main()
