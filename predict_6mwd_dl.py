#!/usr/bin/env python3
"""
Foundation Model & DL Experiments for 6MWD Prediction
=====================================================
Foundation Models (pre-trained, frozen embeddings):
  FM1: MOMENT (Goswami et al., ICML 2024) — time series FM, 512-step input
  FM2: Chronos (Ansari et al., 2024) — probabilistic time series FM

Downstream regressors on FM embeddings:
  Ridge, XGBoost, MLP, TCN, LSTM, Transformer

Configs: A1, A2, B1, B2, C1, C2

CV: Leave-One-Subject-Out (LOO) — consistent with ML baselines
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
PRED = os.path.join(OUT, "predictions")
for d in [OUT, PRED]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 30.0

BASIC_DEMO = ["cohort_M", "Age", "Sex", "Height", "Weight", "BMI"]
CLINIC_ACCEL_COLS = [
    "cadence_hz_c", "step_time_cv_pct_c", "acf_step_regularity_c",
    "hr_ap_c", "hr_vt_c", "ml_rms_g_c", "ml_spectral_entropy_c",
    "jerk_mean_abs_gps_c", "enmo_mean_g_c", "cadence_slope_per_min_c",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    home = pd.read_csv(os.path.join(BASE, "sway_features_home.csv"))
    home.rename(columns={"year_x": "year"}, inplace=True)
    clinic = pd.read_csv(os.path.join(BASE, "features_top10.csv"))
    cc = {c: c + "_c" for c in clinic.columns if c not in ("cohort","subj_id","year","sixmwd","fs")}
    clinic = clinic.rename(columns=cc)
    paired = home.merge(clinic, on=["cohort","subj_id","year"], how="inner", suffixes=("","_clinic"))
    demo = pd.read_excel(os.path.join(BASE, "SwayDemographics.xlsx"))
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    paired = paired.merge(demo, on=["cohort","subj_id"], how="left")
    paired["cohort_M"] = (paired["cohort"]=="M").astype(int)
    paired["Sex"] = pd.to_numeric(paired["Sex"], errors="coerce")
    for col in ["Age","Height","Weight","BMI"]:
        paired[col] = pd.to_numeric(paired[col], errors="coerce")
    return paired


def load_signal(cohort, sid, year, sixmwd, source_dir):
    fname = f"{cohort}{sid:02d}_{year}_{sixmwd}.csv"
    df = pd.read_csv(os.path.join(BASE, source_dir, fname))
    if "AP" in df.columns:
        return df[["AP","ML","VT"]].values.astype(np.float32)
    return df[["X","Y","Z"]].values.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FOUNDATION MODEL EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_moment_embeddings(signals, batch_size=8):
    """MOMENT FM: extract embeddings (frozen). Input: 3-channel, 512 steps."""
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(DEVICE)
    model.eval()

    embeddings = []
    for sig in signals:
        # Segment into 512-step windows
        segs = []
        for s in range(0, len(sig) - 512 + 1, 256):
            segs.append(sig[s:s+512].T)  # (3, 512)
        if not segs:
            pad = np.zeros((3, 512), dtype=np.float32)
            pad[:, :len(sig)] = sig[:512].T
            segs.append(pad)
        segs = np.array(segs, dtype=np.float32)

        seg_embeds = []
        for j in range(0, len(segs), batch_size):
            batch = torch.from_numpy(segs[j:j+batch_size]).to(DEVICE)
            with torch.no_grad():
                out = model(x_enc=batch)
                seg_embeds.append(out.embeddings.cpu().numpy())
        embeddings.append(np.concatenate(seg_embeds).mean(axis=0))

    del model; torch.cuda.empty_cache()
    return np.array(embeddings)


def extract_chronos_embeddings(signals, batch_size=8):
    """Chronos FM: extract embeddings (frozen). Processes each axis separately."""
    from chronos import ChronosPipeline
    model = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=DEVICE,
        torch_dtype=torch.float32,
    )

    embeddings = []
    for sig in signals:
        # Chronos expects 1D time series — process each axis, concatenate
        axis_embeds = []
        for ax in range(sig.shape[1]):
            ts = torch.from_numpy(sig[:, ax]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                # Use encoder embeddings
                tok = model.tokenizer(ts)
                enc_out = model.model.encoder(
                    input_ids=tok.to(DEVICE)
                ).last_hidden_state  # (1, seq_len, hidden)
                axis_embed = enc_out.mean(dim=1).cpu().numpy()  # (1, hidden)
                axis_embeds.append(axis_embed.squeeze())
        embeddings.append(np.concatenate(axis_embeds))

    del model; torch.cuda.empty_cache()
    return np.array(embeddings)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DOWNSTREAM MODELS
# ══════════════════════════════════════════════════════════════════════════════

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class TCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden=32, n_layers=3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            inc = in_dim if i == 0 else hidden
            d = 2**i
            pad = 2 * d
            layers.extend([
                nn.Conv1d(inc, hidden, 3, dilation=d, padding=pad),
                nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            ])
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, x):
        # x: (batch, in_dim, seq_len)
        for layer in self.tcn:
            x = layer(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden*2, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1)).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, in_dim, d_model=32, nhead=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                           dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_model, 1))
    def forward(self, x):
        x = self.proj(x)
        x = self.enc(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CV ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def loo_sklearn(X, y, model_fn):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    return preds


def loo_torch(X, y, model_cls, model_kwargs, epochs=60, lr=5e-4, reshape=None):
    """LOO with PyTorch model. reshape: None, 'seq', or 'conv1d'."""
    n, d = X.shape
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]).astype(np.float32)
        Xte = sc.transform(X[i:i+1]).astype(np.float32)

        if reshape in ("seq", "conv1d"):
            sl = 32
            fd = d // sl
            Xtr = Xtr[:, :sl*fd].reshape(-1, sl, fd)
            Xte = Xte[:, :sl*fd].reshape(-1, sl, fd)
            if reshape == "conv1d":
                Xtr = np.transpose(Xtr, (0, 2, 1))
                Xte = np.transpose(Xte, (0, 2, 1))

        model = model_cls(**model_kwargs).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        Xt = torch.from_numpy(Xtr).to(DEVICE)
        yt = torch.from_numpy(y[tr].astype(np.float32)).to(DEVICE)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            F.mse_loss(model(Xt), yt).backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            preds[i] = model(torch.from_numpy(Xte).to(DEVICE)).cpu().item()
    return preds


def loo_ccpt(X_home, X_clinic_feats, X_extra, y, model_fn):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X_home):
        sc_c = StandardScaler()
        cm = Ridge(alpha=1.0)
        cm.fit(sc_c.fit_transform(X_clinic_feats[tr]), y[tr])
        cp_tr = cm.predict(sc_c.fit_transform(X_clinic_feats[tr])).reshape(-1,1)/1000
        cp_te = cm.predict(sc_c.transform(X_clinic_feats[te])).reshape(-1,1)/1000
        sc_h = StandardScaler()
        Xh_tr = sc_h.fit_transform(X_home[tr])
        Xh_te = sc_h.transform(X_home[te])
        if X_extra is not None:
            sc_d = StandardScaler()
            Xd_tr = sc_d.fit_transform(X_extra[tr])
            Xd_te = sc_d.transform(X_extra[te])
            Xa_tr = np.column_stack([Xh_tr, Xd_tr, cp_tr])
            Xa_te = np.column_stack([Xh_te, Xd_te, cp_te])
        else:
            Xa_tr = np.column_stack([Xh_tr, cp_tr])
            Xa_te = np.column_stack([Xh_te, cp_te])
        m = model_fn()
        m.fit(Xa_tr, y[tr])
        preds[te] = m.predict(Xa_te)
    return preds


def metrics(y, yhat):
    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    pr, _ = pearsonr(y, yhat)
    sr, _ = spearmanr(y, yhat)
    return {"R2": round(r2, 4), "MAE": round(mae, 1),
            "Pearson_r": round(pr, 4), "Spearman_rho": round(sr, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Foundation Model & DL Experiments for 6MWD Prediction")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    paired = load_data()
    y = paired["sixmwd"].values.astype(float)
    cohort = paired["cohort"].values
    n = len(y)
    print(f"  {n} subjects")

    X_cf = paired[CLINIC_ACCEL_COLS].values.astype(float)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m = np.isnan(X_demo[:, j])
        if m.any(): X_demo[m, j] = np.nanmedian(X_demo[:, j])

    # Load signals
    print("\n[1/4] Loading signals...")
    home_sigs, clinic_sigs = [], []
    for _, row in paired.iterrows():
        c, s, yr, d = row["cohort"], int(row["subj_id"]), int(row["year"]), int(row["sixmwd"])
        home_sigs.append(load_signal(c, s, yr, d, "csv_processed_home"))
        clinic_sigs.append(load_signal(c, s, yr, d, "csv_raw2"))

    # Extract FM embeddings
    print("\n[2/4] Extracting MOMENT embeddings...")
    mom_home = extract_moment_embeddings(home_sigs)
    mom_clinic = extract_moment_embeddings(clinic_sigs)
    np.save(os.path.join(OUT, "moment_home.npy"), mom_home)
    np.save(os.path.join(OUT, "moment_clinic.npy"), mom_clinic)
    print(f"  MOMENT: home {mom_home.shape}, clinic {mom_clinic.shape}")

    print("\n[3/4] Extracting Chronos embeddings...")
    try:
        chr_home = extract_chronos_embeddings(home_sigs)
        chr_clinic = extract_chronos_embeddings(clinic_sigs)
        np.save(os.path.join(OUT, "chronos_home.npy"), chr_home)
        np.save(os.path.join(OUT, "chronos_clinic.npy"), chr_clinic)
        print(f"  Chronos: home {chr_home.shape}, clinic {chr_clinic.shape}")
        has_chronos = True
    except Exception as e:
        print(f"  Chronos failed: {e}")
        has_chronos = False

    # Run experiments
    print(f"\n[4/4] Running LOO CV experiments...")
    results = []

    def rec(model, config, preds, desc):
        m = metrics(y, preds)
        results.append({"model": model, "config": config, "desc": desc, **m})
        print(f"  {model:30s} {config:4s} R\u00b2={m['R2']:.4f}")

    xgb = lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbosity=0)

    for fm_name, Eh, Ec in [("MOMENT", mom_home, mom_clinic)] + \
                             ([("Chronos", chr_home, chr_clinic)] if has_chronos else []):
        print(f"\n  {'═'*5} {fm_name} (embed_dim={Eh.shape[1]}) {'═'*5}")
        d = Eh.shape[1]
        sl = 32
        fd = d // sl

        # A1
        print(f"\n  --- A1: {fm_name} home only ---")
        rec(f"{fm_name}+Ridge", "A1", loo_sklearn(Eh, y, lambda: Ridge(alpha=10)), "")
        rec(f"{fm_name}+XGBoost", "A1", loo_sklearn(Eh, y, xgb), "")
        rec(f"{fm_name}+MLP", "A1",
            loo_torch(Eh, y, MLPRegressor, {"in_dim": d, "hidden": 32}, epochs=60), "")
        if fd > 0:
            rec(f"{fm_name}+TCN", "A1",
                loo_torch(Eh, y, TCNRegressor, {"in_dim": fd, "hidden": 32},
                          epochs=50, reshape="conv1d"), "")
            rec(f"{fm_name}+LSTM", "A1",
                loo_torch(Eh, y, LSTMRegressor, {"in_dim": fd, "hidden": 32},
                          epochs=50, reshape="seq"), "")
            rec(f"{fm_name}+Transformer", "A1",
                loo_torch(Eh, y, TransformerRegressor, {"in_dim": fd, "d_model": 32, "nhead": 4},
                          epochs=50, reshape="seq"), "")

        # A2
        print(f"\n  --- A2: {fm_name} + Demo ---")
        Xa2 = np.column_stack([Eh, X_demo])
        rec(f"{fm_name}+Ridge", "A2", loo_sklearn(Xa2, y, lambda: Ridge(alpha=10)), "")
        rec(f"{fm_name}+XGBoost", "A2", loo_sklearn(Xa2, y, xgb), "")

        # B1
        print(f"\n  --- B1: {fm_name} + CCPT ---")
        rec(f"{fm_name}+Ridge", "B1", loo_ccpt(Eh, X_cf, None, y, lambda: Ridge(alpha=10)), "")
        rec(f"{fm_name}+XGBoost", "B1", loo_ccpt(Eh, X_cf, None, y, xgb), "")

        # B2
        print(f"\n  --- B2: {fm_name} + Demo + CCPT ---")
        rec(f"{fm_name}+Ridge", "B2", loo_ccpt(Eh, X_cf, X_demo, y, lambda: Ridge(alpha=10)), "")
        rec(f"{fm_name}+XGBoost", "B2", loo_ccpt(Eh, X_cf, X_demo, y, xgb), "")

        # C1
        print(f"\n  --- C1: {fm_name} clinic only ---")
        rec(f"{fm_name}+Ridge", "C1", loo_sklearn(Ec, y, lambda: Ridge(alpha=10)), "")
        rec(f"{fm_name}+XGBoost", "C1", loo_sklearn(Ec, y, xgb), "")
        rec(f"{fm_name}+MLP", "C1",
            loo_torch(Ec, y, MLPRegressor, {"in_dim": Ec.shape[1], "hidden": 32}, epochs=60), "")
        if fd > 0:
            rec(f"{fm_name}+TCN", "C1",
                loo_torch(Ec, y, TCNRegressor, {"in_dim": fd, "hidden": 32},
                          epochs=50, reshape="conv1d"), "")
            rec(f"{fm_name}+LSTM", "C1",
                loo_torch(Ec, y, LSTMRegressor, {"in_dim": fd, "hidden": 32},
                          epochs=50, reshape="seq"), "")
            rec(f"{fm_name}+Transformer", "C1",
                loo_torch(Ec, y, TransformerRegressor, {"in_dim": fd, "d_model": 32, "nhead": 4},
                          epochs=50, reshape="seq"), "")

        # C2
        print(f"\n  --- C2: {fm_name} + Demo ---")
        Xc2 = np.column_stack([Ec, X_demo])
        rec(f"{fm_name}+Ridge", "C2", loo_sklearn(Xc2, y, lambda: Ridge(alpha=10)), "")
        rec(f"{fm_name}+XGBoost", "C2", loo_sklearn(Xc2, y, xgb), "")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "results_dl.csv"), index=False)
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    pivot = df.pivot_table(index="model", columns="config", values="R2", aggfunc="first")
    co = [c for c in ["A1","A2","B1","B2","C1","C2"] if c in pivot.columns]
    print(pivot[co].to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/results_dl.csv")


if __name__ == "__main__":
    main()
