#!/usr/bin/env python3
"""
Run All Models for 6MWD Prediction (A and C configs)
====================================================
Requires: preprocess_raw.py to have been run first.

Feature types (reported separately):
  - Walking gait features (35) from walking_segments/
  - Activity profile features (~15) from daytime_segments/

Models:
  ML: Ridge, ElasticNet, RandomForest, XGBoost, SVR
  FM: MOMENT (1024-d), Chronos (512-d per axis), LimuBERT (72-d)
      + PCA reduction variants to avoid overfitting
  DL: TCN, LSTM, Transformer (on raw walking segments, separate from FM)

Walking detection comparison:
  - Heuristic (current) vs LimuBERT-based activity recognition

Configs: A1 (accel only), A2 (+ demo), C1 (clinic only), C2 (+ demo)
CV: Leave-One-Subject-Out (LOO)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.signal import welch, find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
WALK_DIR = OUT / "walking_segments"
DAY_DIR = OUT / "daytime_segments"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 30.0
BASIC_DEMO = ["cohort_M", "Age", "Sex", "Height", "Weight", "BMI"]
CLINIC_COLS = [
    "cadence_hz_c", "step_time_cv_pct_c", "acf_step_regularity_c",
    "hr_ap_c", "hr_vt_c", "ml_rms_g_c", "ml_spectral_entropy_c",
    "jerk_mean_abs_gps_c", "enmo_mean_g_c", "cadence_slope_per_min_c",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_subject_table():
    home = pd.read_csv(BASE / "sway_features_home.csv")
    home.rename(columns={"year_x": "year"}, inplace=True)
    clinic = pd.read_csv(BASE / "features_top10.csv")
    cc = {c: c + "_c" for c in clinic.columns if c not in ("cohort","subj_id","year","sixmwd","fs")}
    clinic = clinic.rename(columns=cc).drop(columns=["sixmwd","fs"], errors="ignore")
    paired = home[["cohort","subj_id","year","sixmwd"]].merge(
        clinic, on=["cohort","subj_id","year"], how="inner")
    demo = pd.read_excel(BASE / "SwayDemographics.xlsx")
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    paired = paired.merge(demo, on=["cohort","subj_id"], how="left")
    paired["cohort_M"] = (paired["cohort"] == "M").astype(int)
    for col in ["Sex","Age","Height","Weight","BMI"]:
        paired[col] = pd.to_numeric(paired[col], errors="coerce")
    return paired


def fname(row):
    return f"{row['cohort']}{int(row['subj_id']):02d}_{int(row['year'])}_{int(row['sixmwd'])}.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 2. WALKING GAIT FEATURES (35 features from walking_segments/)
# ══════════════════════════════════════════════════════════════════════════════

def extract_gait_features(sig, fs=FS):
    ap, ml, vt = sig[:,0], sig[:,1], sig[:,2]
    vm = np.sqrt(ap**2 + ml**2 + vt**2)
    f = {}
    for nm, ax in [("ap",ap),("ml",ml),("vt",vt),("vm",vm)]:
        f[f"{nm}_rms"] = np.sqrt(np.mean(ax**2))
        f[f"{nm}_std"] = np.std(ax)
        f[f"{nm}_iqr"] = np.percentile(ax,75)-np.percentile(ax,25)
    f["sma"] = np.mean(np.abs(ap)+np.abs(ml)+np.abs(vt))
    for nm, ax in [("ap",ap),("ml",ml),("vt",vt)]:
        f[f"{nm}_jerk_rms"] = np.sqrt(np.mean((np.diff(ax)*fs)**2))
    for nm, ax in [("vt",vt),("ap",ap)]:
        ac = ax - ax.mean()
        acf = np.correlate(ac, ac, "full")[len(ac)-1:]
        acf /= (acf[0]+1e-12)
        ml2 = min(int(1.5*fs), len(acf)-1)
        s = acf[int(0.3*fs):ml2]
        pk, pr = find_peaks(s, height=0.0)
        f[f"{nm}_step_reg"] = pr["peak_heights"][0] if len(pk)>=1 else 0
        f[f"{nm}_step_time"] = (pk[0]+int(0.3*fs))/fs if len(pk)>=1 else 0
        f[f"{nm}_stride_reg"] = pr["peak_heights"][1] if len(pk)>=2 else 0
        f[f"{nm}_step_sym"] = f[f"{nm}_step_reg"]/(f[f"{nm}_stride_reg"]+1e-8) if len(pk)>=2 else 1
    for nm, ax in [("vt",vt),("ap",ap),("ml",ml)]:
        if len(ax) > 64:
            fr, ps = welch(ax, fs=fs, nperseg=min(256,len(ax)))
            gb = (fr>=0.5)&(fr<=3.5)
            f[f"{nm}_dom_freq"] = fr[gb][np.argmax(ps[gb])] if gb.any() else 0
            gp = np.trapz(ps[gb], fr[gb]) if gb.any() else 0
            tp = np.trapz(ps, fr)+1e-12
            f[f"{nm}_gait_pwr"] = gp/tp
            pn = ps/(ps.sum()+1e-12); pn = pn[pn>0]
            f[f"{nm}_spec_ent"] = -np.sum(pn*np.log2(pn+1e-12))
        else:
            f[f"{nm}_dom_freq"]=0; f[f"{nm}_gait_pwr"]=0; f[f"{nm}_spec_ent"]=0
    fund = f.get("vt_dom_freq", 0)
    if fund > 0:
        fv = np.abs(rfft(vt)); ff = rfftfreq(len(vt), 1/fs)
        ep, op = 0, 0
        for h in range(1,11):
            idx = np.argmin(np.abs(ff-h*fund))
            if h%2==0: ep += fv[idx]**2
            else: op += fv[idx]**2
        f["vt_hr"] = ep/(op+1e-12)
    else:
        f["vt_hr"] = 0
    w2 = int(2*fs)
    rw = np.array([np.sqrt(np.mean(vm[i:i+w2]**2)) for i in range(0,max(1,len(vm)-w2+1),w2)])
    f["act_rms_cv"] = rw.std()/(rw.mean()+1e-12) if len(rw)>1 else 0
    return f


# ══════════════════════════════════════════════════════════════════════════════
# 3. ACTIVITY PROFILE FEATURES (~15 from daytime_segments/)
# ══════════════════════════════════════════════════════════════════════════════

def extract_activity_features(xyz, fs=FS):
    """Activity-level features from full daytime recording."""
    vm = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    f = {}

    # Activity Index per second
    bs = int(fs)
    n_bins = len(vm) // bs
    if n_bins < 10:
        return {k: 0 for k in ["ai_mean","ai_std","ai_iqr","ai_entropy","ai_pct_low",
                                "ai_pct_mod","ai_pct_high","ai_cv","bout_count",
                                "bout_mean_dur","bout_total_dur","bout_mean_accel",
                                "bout_dur_cv","total_hours","walk_pct"]}
    ai = np.array([np.sqrt(np.mean(vm[i*bs:(i+1)*bs]**2)) for i in range(n_bins)])

    f["ai_mean"] = np.mean(ai)
    f["ai_std"] = np.std(ai)
    f["ai_iqr"] = np.percentile(ai,75)-np.percentile(ai,25)
    hist, _ = np.histogram(ai, bins=20, density=True)
    hist = hist[hist>0]; hist = hist/hist.sum()
    f["ai_entropy"] = -np.sum(hist*np.log2(hist+1e-12))
    lo, hi = np.percentile(ai,30), np.percentile(ai,85)
    f["ai_pct_low"] = np.mean(ai < lo)
    f["ai_pct_mod"] = np.mean((ai >= lo) & (ai < hi))
    f["ai_pct_high"] = np.mean(ai >= hi)
    f["ai_cv"] = np.std(ai)/(np.mean(ai)+1e-12)

    # Walking bout statistics (from heuristic detection on raw)
    active = (ai > lo) & (ai < hi)  # moderate = likely walking
    bouts = []
    in_b, bstart = False, 0
    for i in range(len(active)):
        if active[i] and not in_b: bstart=i; in_b=True
        elif not active[i] and in_b:
            if (i-bstart) >= 5: bouts.append((bstart, i))
            in_b = False
    if in_b and (len(active)-bstart)>=5: bouts.append((bstart, len(active)))

    f["bout_count"] = len(bouts)
    if bouts:
        durs = [(e-s) for s,e in bouts]
        accels = [np.mean(ai[s:e]) for s,e in bouts]
        f["bout_mean_dur"] = np.mean(durs)
        f["bout_total_dur"] = np.sum(durs)
        f["bout_mean_accel"] = np.mean(accels)
        f["bout_dur_cv"] = np.std(durs)/(np.mean(durs)+1e-12)
    else:
        f["bout_mean_dur"]=0; f["bout_total_dur"]=0; f["bout_mean_accel"]=0; f["bout_dur_cv"]=0

    f["total_hours"] = len(ai) / 3600
    f["walk_pct"] = f["bout_total_dur"] / (len(ai)+1e-12)
    return f


# ══════════════════════════════════════════════════════════════════════════════
# 4. FM EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def segment_512(sig):
    segs = []
    for s in range(0, len(sig)-512+1, 256):
        segs.append(sig[s:s+512].T)
    if not segs:
        p = np.zeros((3,512), dtype=np.float32)
        L = min(len(sig),512); p[:,:L] = sig[:L].T
        segs.append(p)
    return np.array(segs, dtype=np.float32)


def extract_moment_embeddings(signals, batch_size=8):
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",
                                            model_kwargs={"task_name": "embedding"})
    model.init(); model = model.to(DEVICE); model.eval()
    embs = []
    for sig in signals:
        segs = segment_512(sig)
        se = []
        for j in range(0, len(segs), batch_size):
            b = torch.from_numpy(segs[j:j+batch_size]).to(DEVICE)
            with torch.no_grad():
                se.append(model(x_enc=b).embeddings.cpu().numpy())
        embs.append(np.concatenate(se).mean(axis=0))
    del model; torch.cuda.empty_cache()
    return np.array(embs)


def extract_chronos_embeddings(signals, batch_size=1):
    from chronos import ChronosPipeline
    model = ChronosPipeline.from_pretrained("amazon/chronos-t5-small",
                                             device_map=DEVICE, dtype=torch.float32)
    embs = []
    for sig in signals:
        axis_embs = []
        for ax in range(min(sig.shape[1], 3)):
            ts = torch.from_numpy(sig[:, ax]).unsqueeze(0)
            with torch.no_grad():
                emb, _ = model.embed(ts)
                axis_embs.append(emb.mean(dim=1).cpu().numpy().squeeze())
        embs.append(np.concatenate(axis_embs))
    del model; torch.cuda.empty_cache()
    return np.array(embs)


def extract_limubert_embeddings(signals, batch_size=32):
    sys.path.insert(0, str(BASE / "limubert_repo"))
    from config import PretrainModelConfig
    from models import LIMUBertModel4Pretrain

    with open(BASE / "limubert_repo" / "config" / "limu_bert.json") as f:
        cfg_dict = json.load(f)["base_v4"]
    cfg = PretrainModelConfig.from_json(cfg_dict)
    model = LIMUBertModel4Pretrain(cfg, output_embed=True)
    state = torch.load(BASE / "limubert_repo" / "weights" / "limu_pretrain.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE); model.eval()

    embs = []
    for sig in signals:
        # Pad 3ch→6ch (zero gyro), segment into 20-step windows
        if sig.shape[1] == 3:
            sig6 = np.column_stack([sig, np.zeros_like(sig)])
        else:
            sig6 = sig
        segs = []
        for s in range(0, len(sig6)-20+1, 10):
            segs.append(sig6[s:s+20])
        if not segs:
            p = np.zeros((20, 6), dtype=np.float32)
            L = min(len(sig6), 20)
            p[:L] = sig6[:L]
            segs.append(p)
        segs = np.array(segs, dtype=np.float32)

        se = []
        for j in range(0, len(segs), batch_size):
            b = torch.from_numpy(segs[j:j+batch_size]).to(DEVICE)
            with torch.no_grad():
                out = model(b)  # (batch, 20, 72)
                se.append(out.mean(dim=1).cpu().numpy())  # (batch, 72)
        embs.append(np.concatenate(se).mean(axis=0))  # (72,)

    del model; torch.cuda.empty_cache()
    return np.array(embs)


# ══════════════════════════════════════════════════════════════════════════════
# 5. DL MODELS (on raw segments, SEPARATE from FM)
# ══════════════════════════════════════════════════════════════════════════════

class TCN1D(nn.Module):
    def __init__(self, c=3, h=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c,h,15,padding=7),nn.BatchNorm1d(h),nn.ReLU(),nn.MaxPool1d(4),nn.Dropout(0.3),
            nn.Conv1d(h,h,11,padding=5),nn.BatchNorm1d(h),nn.ReLU(),nn.MaxPool1d(4),nn.Dropout(0.3),
            nn.Conv1d(h,h,7,padding=3),nn.BatchNorm1d(h),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(h,1))
    def forward(self,x): return self.head(self.net(x).squeeze(-1)).squeeze(-1)

class LSTM1D(nn.Module):
    def __init__(self, c=3, h=16):
        super().__init__()
        self.lstm = nn.LSTM(c,h,1,batch_first=True,bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(h*2,1))
    def forward(self,x):
        o,_ = self.lstm(x.transpose(1,2)); return self.head(o.mean(1)).squeeze(-1)

class Transformer1D(nn.Module):
    def __init__(self, c=3, d=16, nh=4):
        super().__init__()
        self.proj = nn.Linear(c,d)
        l = nn.TransformerEncoderLayer(d,nh,dim_feedforward=32,dropout=0.3,batch_first=True)
        self.enc = nn.TransformerEncoder(l,2)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(d,1))
    def forward(self,x):
        x = self.proj(x.transpose(1,2)); x = self.enc(x)
        return self.head(x.mean(1)).squeeze(-1)


def seg300(sig):
    segs = []
    for s in range(0, len(sig)-300+1, 150):
        segs.append(sig[s:s+300].T)
    if not segs:
        p = np.zeros((3,300),dtype=np.float32)
        L=min(len(sig),300); p[:,:L]=sig[:L].T; segs.append(p)
    return np.array(segs, dtype=np.float32)


def loo_dl(signals, y, model_cls, kw, epochs=30, lr=1e-3):
    n = len(y); preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j!=i]
        all_s, all_l = [], []
        for j in tr:
            for s in seg300(signals[j]):
                all_s.append(s); all_l.append(y[j])
        all_s = np.array(all_s); all_l = np.array(all_l, dtype=np.float32)
        m = model_cls(**kw).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
        m.train()
        bs = 64
        for _ in range(epochs):
            idx = np.random.permutation(len(all_s))
            for b in range(0, len(idx), bs):
                bi = idx[b:b+bs]
                xb = torch.from_numpy(all_s[bi]).to(DEVICE) + torch.randn(len(bi),3,300,device=DEVICE)*0.005
                yb = torch.from_numpy(all_l[bi]).to(DEVICE)
                opt.zero_grad(); F.mse_loss(m(xb),yb).backward(); opt.step()
        m.eval()
        ts = torch.from_numpy(seg300(signals[i])).to(DEVICE)
        with torch.no_grad(): preds[i] = m(ts).cpu().numpy().mean()
        if (i+1)%20==0: print(f"    fold {i+1}/{n}", flush=True)
    print(f"    {n}/{n}")
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# 6. CV ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def loo(X, y, mfn):
    p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = mfn()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        p[te] = m.predict(sc.transform(X[te]))
    return p

def loo_fs(X, y, mfn, k=15):
    p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        cors = np.array([abs(spearmanr(X[tr,j],y[tr])[0]) for j in range(X.shape[1])])
        idx = np.argsort(cors)[-k:]
        sc = StandardScaler(); m = mfn()
        m.fit(sc.fit_transform(X[tr][:,idx]), y[tr])
        p[te] = m.predict(sc.transform(X[te][:,idx]))
    return p

def metrics(y, yh):
    return {"R2": round(r2_score(y,yh),4), "MAE": round(mean_absolute_error(y,yh),1),
            "r": round(pearsonr(y,yh)[0],4), "rho": round(spearmanr(y,yh)[0],4)}


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("All Models: Walking Features + Activity Profile + FM + DL")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    paired = load_subject_table()
    y = paired["sixmwd"].values.astype(float)
    cohort = paired["cohort"].values
    n = len(y)
    print(f"  {n} subjects (C={sum(cohort=='C')}, M={sum(cohort=='M')})")

    X_clinic = paired[CLINIC_COLS].values.astype(float)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m = np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j] = np.nanmedian(X_demo[:,j])

    # ── Load processed signals ──
    print("\n[1/8] Loading processed signals...")
    home_walk, home_day, clinic_raw = [], [], []
    missing = 0
    for _, row in paired.iterrows():
        fn = fname(row)
        wp = WALK_DIR / fn
        dp = DAY_DIR / fn
        cp = BASE / "csv_raw2" / fn

        if wp.exists():
            home_walk.append(pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32))
        else:
            # Fallback: use csv_processed_home
            fb = BASE / "csv_processed_home" / fn
            home_walk.append(pd.read_csv(fb)[["AP","ML","VT"]].values.astype(np.float32))
            missing += 1

        if dp.exists():
            home_day.append(pd.read_csv(dp)[["X","Y","Z"]].values.astype(np.float32))
        else:
            home_day.append(home_walk[-1])  # fallback

        clinic_raw.append(pd.read_csv(cp)[["X","Y","Z"]].values.astype(np.float32))

    print(f"  Loaded: {n} home walking, {n} home daytime, {n} clinic")
    if missing: print(f"  ({missing} used fallback from csv_processed_home)")

    # ── Extract features ──
    print("\n[2/8] Extracting walking gait features...")
    gait_rows = [extract_gait_features(s) for s in home_walk]
    X_gait = pd.DataFrame(gait_rows).replace([np.inf,-np.inf], np.nan)
    for c in X_gait.columns:
        if X_gait[c].isna().any(): X_gait[c] = X_gait[c].fillna(X_gait[c].median())
    X_gait = X_gait.values.astype(float)
    print(f"  {X_gait.shape[1]} walking gait features")

    print("\n[3/8] Extracting activity profile features...")
    act_rows = [extract_activity_features(s) for s in home_day]
    X_act = pd.DataFrame(act_rows).replace([np.inf,-np.inf], np.nan)
    for c in X_act.columns:
        if X_act[c].isna().any(): X_act[c] = X_act[c].fillna(X_act[c].median())
    X_act = X_act.values.astype(float)
    print(f"  {X_act.shape[1]} activity profile features")

    # ── FM embeddings ──
    print("\n[4/8] Extracting FM embeddings...")

    print("  MOMENT (home)...", flush=True)
    E_mom_h = extract_moment_embeddings(home_walk)
    print(f"    {E_mom_h.shape}")
    print("  MOMENT (clinic)...", flush=True)
    E_mom_c = extract_moment_embeddings(clinic_raw)
    print(f"    {E_mom_c.shape}")

    print("  Chronos (home)...", flush=True)
    E_chr_h = extract_chronos_embeddings(home_walk)
    print(f"    {E_chr_h.shape}")
    print("  Chronos (clinic)...", flush=True)
    E_chr_c = extract_chronos_embeddings(clinic_raw)
    print(f"    {E_chr_c.shape}")

    print("  LimuBERT (home)...", flush=True)
    E_lim_h = extract_limubert_embeddings(home_walk)
    print(f"    {E_lim_h.shape}")
    print("  LimuBERT (clinic)...", flush=True)
    E_lim_c = extract_limubert_embeddings(clinic_raw)
    print(f"    {E_lim_c.shape}")

    # PCA-reduced FM embeddings (to avoid overfitting)
    def pca_reduce(X, n_comp=50):
        pca = PCA(n_components=min(n_comp, X.shape[1], X.shape[0]-1))
        return pca.fit_transform(X)

    E_mom_h_pca = pca_reduce(E_mom_h, 50)
    E_mom_c_pca = pca_reduce(E_mom_c, 50)
    E_chr_h_pca = pca_reduce(E_chr_h, 50)
    E_chr_c_pca = pca_reduce(E_chr_c, 50)
    # LimuBERT is only 72-d, no PCA needed

    # Save embeddings
    for name, arr in [("mom_h", E_mom_h), ("mom_c", E_mom_c),
                      ("chr_h", E_chr_h), ("chr_c", E_chr_c),
                      ("lim_h", E_lim_h), ("lim_c", E_lim_c)]:
        np.save(OUT / f"emb_{name}.npy", arr)

    # ── Run all experiments ──
    print("\n[5/8] Running ML baselines (LOO)...")
    results = []

    def rec(model, config, feat_type, preds):
        m = metrics(y, preds)
        results.append({"model": model, "config": config, "feat_type": feat_type, **m})
        print(f"  {model:25s} {config:4s} {feat_type:12s} R²={m['R2']:.4f}")

    xgb = lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    ml = {"Ridge": lambda: Ridge(alpha=10), "ElasticNet": lambda: ElasticNet(alpha=1,l1_ratio=0.5,max_iter=10000),
          "RF": lambda: RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=-1),
          "XGBoost": xgb, "SVR": lambda: SVR(kernel="rbf",C=100,epsilon=50)}

    for mname, mfn in ml.items():
        # Walking gait features
        rec(mname, "A1", "gait", loo_fs(X_gait, y, mfn, k=15))
        rec(mname, "A2", "gait", loo(np.column_stack([X_gait, X_demo]), y, mfn))
        rec(mname, "C1", "gait_clinic", loo(X_clinic, y, mfn))
        rec(mname, "C2", "gait_clinic", loo(np.column_stack([X_clinic, X_demo]), y, mfn))

        # Activity profile features
        rec(mname, "A1", "activity", loo(X_act, y, mfn))
        rec(mname, "A2", "activity", loo(np.column_stack([X_act, X_demo]), y, mfn))

    # ── FM experiments ──
    print("\n[6/8] FM + downstream regressors...")
    fm_downstream = {"Ridge": lambda: Ridge(alpha=10), "XGBoost": xgb}

    for fm_name, Eh, Ec, Eh_pca, Ec_pca in [
        ("MOMENT", E_mom_h, E_mom_c, E_mom_h_pca, E_mom_c_pca),
        ("Chronos", E_chr_h, E_chr_c, E_chr_h_pca, E_chr_c_pca),
        ("LimuBERT", E_lim_h, E_lim_c, E_lim_h, E_lim_c),  # no PCA for 72-d
    ]:
        for dname, dfn in fm_downstream.items():
            label = f"{fm_name}+{dname}"
            # Full embeddings
            rec(label, "A1", "fm_full", loo(Eh, y, dfn))
            rec(label, "A2", "fm_full", loo(np.column_stack([Eh, X_demo]), y, dfn))
            rec(label, "C1", "fm_full", loo(Ec, y, dfn))
            rec(label, "C2", "fm_full", loo(np.column_stack([Ec, X_demo]), y, dfn))

            # PCA-reduced (skip for LimuBERT)
            if fm_name != "LimuBERT":
                label_p = f"{fm_name}_PCA+{dname}"
                rec(label_p, "A1", "fm_pca", loo(Eh_pca, y, dfn))
                rec(label_p, "A2", "fm_pca", loo(np.column_stack([Eh_pca, X_demo]), y, dfn))
                rec(label_p, "C1", "fm_pca", loo(Ec_pca, y, dfn))
                rec(label_p, "C2", "fm_pca", loo(np.column_stack([Ec_pca, X_demo]), y, dfn))

    # ── DL on raw signals (separate from FM) ──
    print("\n[7/8] TCN/LSTM/Transformer on raw walking signals...")
    for dl_name, dl_cls, dl_kw in [
        ("TCN", TCN1D, {"c": 3, "h": 16}),
        ("LSTM", LSTM1D, {"c": 3, "h": 16}),
        ("Transformer", Transformer1D, {"c": 3, "d": 16, "nh": 4}),
    ]:
        print(f"  {dl_name} A1...", flush=True)
        rec(dl_name, "A1", "dl_raw", loo_dl(home_walk, y, dl_cls, dl_kw, epochs=30))
        print(f"  {dl_name} C1...", flush=True)
        rec(dl_name, "C1", "dl_raw", loo_dl(clinic_raw, y, dl_cls, dl_kw, epochs=30))

    # ── Save and report ──
    print("\n[8/8] Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(OUT / "results_all.csv", index=False)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for ft in df["feat_type"].unique():
        sub = df[df["feat_type"] == ft]
        print(f"\n--- Feature type: {ft} ---")
        pivot = sub.pivot_table(index="model", columns="config", values="R2", aggfunc="first")
        co = [c for c in ["A1","A2","C1","C2"] if c in pivot.columns]
        if co:
            print(pivot[co].to_string(float_format="{:.4f}".format))

    print(f"\nSaved to {OUT}/results_all.csv")


if __name__ == "__main__":
    main()
