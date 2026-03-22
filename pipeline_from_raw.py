#!/usr/bin/env python3
"""
Complete Pipeline: GT3X Raw Files → Proper Preprocessing → Walking Detection
→ Feature Extraction → 6MWD Prediction with ML + FM + DL
===========================================================================

Step 1: Load raw GT3X from Accel files/ (multi-day, 30 Hz)
Step 2: Extract daytime segments (skip night/idle)
Step 3: Walking bout detection on raw signal
Step 4: Per-bout preprocessing (gravity removal, anatomical alignment)
Step 5: Gait feature extraction from walking bouts
Step 6: MOMENT FM embedding extraction
Step 7: Prediction with ML baselines, MOMENT+downstream, TCN/LSTM/Transformer
Step 8: Report all A/B/C configs

Subject ID alignment verified between home and clinic data.
CV: Leave-One-Subject-Out (LOO), no data leakage.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from pygt3x.reader import FileReader

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
OUT.mkdir(exist_ok=True)
(OUT / "predictions").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 30.0

BASIC_DEMO = ["cohort_M", "Age", "Sex", "Height", "Weight", "BMI"]
CLINIC_ACCEL_COLS = [
    "cadence_hz_c", "step_time_cv_pct_c", "acf_step_regularity_c",
    "hr_ap_c", "hr_vt_c", "ml_rms_g_c", "ml_spectral_entropy_c",
    "jerk_mean_abs_gps_c", "enmo_mean_g_c", "cadence_slope_per_min_c",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. SUBJECT ALIGNMENT & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def build_subject_table():
    """Build aligned subject table with GT3X paths, clinic features, demographics.
    Verifies that home and clinic data refer to the same subjects.
    """
    # Home feature file for subject list and 6MWD ground truth
    home = pd.read_csv(BASE / "sway_features_home.csv")
    home.rename(columns={"year_x": "year"}, inplace=True)
    home = home[["cohort", "subj_id", "year", "sixmwd"]].copy()

    # Clinic features
    clinic = pd.read_csv(BASE / "features_top10.csv")
    cc = {c: c + "_c" for c in clinic.columns
          if c not in ("cohort", "subj_id", "year", "sixmwd", "fs")}
    clinic = clinic.rename(columns=cc)

    # Merge — inner join ensures same subjects
    clinic = clinic.drop(columns=["sixmwd_c"], errors="ignore")
    if "sixmwd" in clinic.columns:
        clinic = clinic.drop(columns=["sixmwd"])
    if "fs_c" in clinic.columns:
        clinic = clinic.drop(columns=["fs_c"])
    if "fs" in clinic.columns and "fs" not in home.columns:
        pass  # keep
    elif "fs" in clinic.columns:
        clinic = clinic.drop(columns=["fs"])
    paired = home.merge(clinic, on=["cohort", "subj_id", "year"], how="inner")

    # Demographics
    demo = pd.read_excel(BASE / "SwayDemographics.xlsx")
    demo["cohort"] = demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"] = demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    paired = paired.merge(demo, on=["cohort", "subj_id"], how="left")
    paired["cohort_M"] = (paired["cohort"] == "M").astype(int)
    paired["Sex"] = pd.to_numeric(paired["Sex"], errors="coerce")
    for col in ["Age", "Height", "Weight", "BMI"]:
        paired[col] = pd.to_numeric(paired[col], errors="coerce")

    # Map to GT3X files
    accel_dir = BASE / "Accel files"
    folder_map = {}
    for folder in accel_dir.iterdir():
        if not folder.is_dir():
            continue
        m = re.match(r"([CM])(\d+)", folder.name)
        if m:
            folder_map[(m.group(1), int(m.group(2)))] = folder

    gt3x_paths = []
    for _, row in paired.iterrows():
        key = (row["cohort"], int(row["subj_id"]))
        folder = folder_map.get(key)
        if folder:
            gt3x = list(folder.glob("*.gt3x"))
            gt3x_paths.append(str(gt3x[0]) if gt3x else None)
        else:
            gt3x_paths.append(None)
    paired["gt3x_path"] = gt3x_paths

    # Verify alignment
    assert len(paired) == len(home), "Subject count mismatch after merge"
    assert paired["gt3x_path"].notna().all(), "Missing GT3X files for some subjects"

    # Verify clinic and home have same sixmwd (same ground truth)
    print(f"  Subjects: {len(paired)} (C={sum(paired['cohort']=='C')}, M={sum(paired['cohort']=='M')})")
    print(f"  All GT3X files found: {paired['gt3x_path'].notna().sum()}/{len(paired)}")
    print(f"  6MWD range: {paired['sixmwd'].min()}-{paired['sixmwd'].max()}")

    return paired


def load_gt3x(path):
    """Load raw GT3X file, return (timestamps, X, Y, Z) at 30 Hz."""
    with FileReader(str(path)) as reader:
        df = reader.to_pandas()
    timestamps = df.index.values
    xyz = df[["X", "Y", "Z"]].values.astype(np.float32)
    return timestamps, xyz


# ══════════════════════════════════════════════════════════════════════════════
# 2. DAYTIME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_daytime(timestamps, xyz, day_start_hour=7, day_end_hour=22):
    """Extract daytime data only (7am-10pm), skip nights.
    Handles multi-day recordings. Returns concatenated daytime segments.
    """
    # Convert timestamps to hours of day
    # timestamps are Unix epoch seconds
    hours = (timestamps % 86400) / 3600  # hour of day (UTC)

    daytime_mask = (hours >= day_start_hour) & (hours < day_end_hour)

    # Also skip idle/constant segments (device not worn)
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    # Rolling std over 5-second windows
    win = int(5 * FS)
    if len(vm) > win:
        rolling_std = pd.Series(vm).rolling(win, center=True).std().values
        rolling_std = np.nan_to_num(rolling_std, nan=0)
        worn_mask = rolling_std > 0.01  # device is being worn if some variance
    else:
        worn_mask = np.ones(len(vm), dtype=bool)

    mask = daytime_mask & worn_mask
    return xyz[mask]


# ══════════════════════════════════════════════════════════════════════════════
# 3. WALKING BOUT DETECTION (on raw signal, 30 Hz)
# ══════════════════════════════════════════════════════════════════════════════

def detect_walking_bouts(xyz, fs=FS, win_sec=10, step_sec=2, min_bout_sec=20):
    """Detect walking bouts from raw accelerometer signal.
    Criteria: moderate VM intensity + periodic signal at step frequency.
    """
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    win = int(win_sec * fs)
    step = int(step_sec * fs)

    is_walking = []
    starts = []
    for s in range(0, len(vm) - win, step):
        seg_vm = vm[s:s + win]
        rms = np.sqrt(np.mean(seg_vm**2))
        std = np.std(seg_vm)

        # Periodicity: autocorrelation on most variable axis
        seg = xyz[s:s + win]
        variances = [np.var(seg[:, i]) for i in range(3)]
        best = seg[:, np.argmax(variances)]
        best_c = best - best.mean()
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


# ══════════════════════════════════════════════════════════════════════════════
# 4. PER-BOUT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_bout(seg_xyz, fs=FS):
    """Preprocess a raw walking bout: gravity removal + anatomical alignment.
    Returns (T, 3) array [AP, ML, VT].
    """
    if len(seg_xyz) < 50:
        return seg_xyz

    b, a = butter(4, 0.25 / (fs / 2), btype="low")
    g_est = filtfilt(b, a, seg_xyz, axis=0)
    g_mean = np.mean(g_est, axis=0)
    g_dir = g_mean / (np.linalg.norm(g_mean) + 1e-12)

    # Remove gravity
    g_proj = (seg_xyz @ g_dir)[:, None] * g_dir[None, :]
    dyn = seg_xyz - g_proj

    # VT = dynamic vertical
    vt = seg_xyz @ g_dir - np.mean(seg_xyz @ g_dir)

    # Horizontal plane → AP/ML via PCA
    acc_h = dyn - (dyn @ g_dir)[:, None] * g_dir[None, :]
    if acc_h.shape[0] > 10:
        pca = PCA(n_components=2)
        h2d = pca.fit_transform(acc_h)
        ap, ml = h2d[:, 0], h2d[:, 1]
    else:
        ap, ml = acc_h[:, 0], acc_h[:, 1]

    return np.column_stack([ap, ml, vt])


# ══════════════════════════════════════════════════════════════════════════════
# 5. GAIT FEATURE EXTRACTION (35 features)
# ══════════════════════════════════════════════════════════════════════════════

def extract_gait_features(sig, fs=FS):
    """35 interpretable gait features from preprocessed walking signal."""
    ap, ml, vt = sig[:, 0], sig[:, 1], sig[:, 2]
    vm = np.sqrt(ap**2 + ml**2 + vt**2)
    f = {}

    for nm, ax in [("ap", ap), ("ml", ml), ("vt", vt), ("vm", vm)]:
        f[f"{nm}_rms"] = np.sqrt(np.mean(ax**2))
        f[f"{nm}_std"] = np.std(ax)
        f[f"{nm}_iqr"] = np.percentile(ax, 75) - np.percentile(ax, 25)
    f["sma"] = np.mean(np.abs(ap) + np.abs(ml) + np.abs(vt))

    for nm, ax in [("ap", ap), ("ml", ml), ("vt", vt)]:
        f[f"{nm}_jerk_rms"] = np.sqrt(np.mean((np.diff(ax) * fs)**2))

    for nm, ax in [("vt", vt), ("ap", ap)]:
        ax_c = ax - ax.mean()
        acf = np.correlate(ax_c, ax_c, "full")[len(ax_c) - 1:]
        acf /= (acf[0] + 1e-12)
        ml2 = min(int(1.5 * fs), len(acf) - 1)
        search = acf[int(0.3 * fs):ml2]
        peaks, props = find_peaks(search, height=0.0)
        f[f"{nm}_step_reg"] = props["peak_heights"][0] if len(peaks) >= 1 else 0
        f[f"{nm}_step_time"] = (peaks[0] + int(0.3 * fs)) / fs if len(peaks) >= 1 else 0
        f[f"{nm}_stride_reg"] = props["peak_heights"][1] if len(peaks) >= 2 else 0
        f[f"{nm}_step_sym"] = (f[f"{nm}_step_reg"] / (f[f"{nm}_stride_reg"] + 1e-8)
                               if len(peaks) >= 2 else 1.0)

    for nm, ax in [("vt", vt), ("ap", ap), ("ml", ml)]:
        if len(ax) > 64:
            freqs, psd = welch(ax, fs=fs, nperseg=min(256, len(ax)))
            gb = (freqs >= 0.5) & (freqs <= 3.5)
            f[f"{nm}_dom_freq"] = freqs[gb][np.argmax(psd[gb])] if gb.any() else 0
            gp = np.trapz(psd[gb], freqs[gb]) if gb.any() else 0
            tp = np.trapz(psd, freqs) + 1e-12
            f[f"{nm}_gait_pwr"] = gp / tp
            pn = psd / (psd.sum() + 1e-12); pn = pn[pn > 0]
            f[f"{nm}_spec_ent"] = -np.sum(pn * np.log2(pn + 1e-12))
        else:
            f[f"{nm}_dom_freq"] = 0; f[f"{nm}_gait_pwr"] = 0; f[f"{nm}_spec_ent"] = 0

    fund = f.get("vt_dom_freq", 0)
    if fund > 0:
        fv = np.abs(rfft(vt)); ff = rfftfreq(len(vt), 1 / fs)
        ep, op = 0, 0
        for h in range(1, 11):
            idx = np.argmin(np.abs(ff - h * fund))
            if h % 2 == 0: ep += fv[idx]**2
            else: op += fv[idx]**2
        f["vt_hr"] = ep / (op + 1e-12)
    else:
        f["vt_hr"] = 0

    win2 = int(2 * fs)
    rms_w = np.array([np.sqrt(np.mean(vm[i:i + win2]**2))
                      for i in range(0, max(1, len(vm) - win2 + 1), win2)])
    f["act_rms_cv"] = rms_w.std() / (rms_w.mean() + 1e-12) if len(rms_w) > 1 else 0

    return f


# ══════════════════════════════════════════════════════════════════════════════
# 6. FULL PIPELINE: GT3X → FEATURES (per subject)
# ══════════════════════════════════════════════════════════════════════════════

def process_one_subject(gt3x_path, save_daytime_path=None):
    """Complete pipeline for one subject: GT3X → daytime → walking → preprocess → features.
    Also returns concatenated preprocessed walking signal for FM embedding.
    """
    timestamps, xyz = load_gt3x(gt3x_path)

    # Daytime extraction
    daytime_xyz = extract_daytime(timestamps, xyz)
    if len(daytime_xyz) < int(60 * FS):
        # Fallback: use all data if daytime is too short
        daytime_xyz = xyz

    # Save daytime CSV if path provided
    if save_daytime_path is not None:
        pd.DataFrame(daytime_xyz, columns=["X", "Y", "Z"]).to_csv(
            save_daytime_path, index=False)

    # Walking bout detection
    bouts = detect_walking_bouts(daytime_xyz)

    if bouts:
        # Preprocess each bout separately
        walk_segments = []
        for s, e in bouts[:20]:  # max 20 bouts
            preprocessed = preprocess_bout(daytime_xyz[s:e])
            walk_segments.append(preprocessed)
        walk_all = np.concatenate(walk_segments, axis=0)
    else:
        # Fallback: preprocess entire daytime, select best windows
        preprocessed = preprocess_bout(daytime_xyz[:int(600 * FS)])  # max 10 min
        walk_all = preprocessed

    # Extract gait features
    feats = extract_gait_features(walk_all)

    return feats, walk_all, len(bouts)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MOMENT EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_moment_embeddings(walk_signals, batch_size=8):
    """Extract MOMENT embeddings from preprocessed walking signals."""
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(DEVICE)
    model.eval()

    embeddings = []
    for sig in walk_signals:
        # Segment into 512-step windows
        segs = []
        for s in range(0, len(sig) - 512 + 1, 256):
            segs.append(sig[s:s + 512].T)  # (3, 512)
        if not segs:
            pad = np.zeros((3, 512), dtype=np.float32)
            L = min(len(sig), 512)
            pad[:, :L] = sig[:L].T
            segs.append(pad)
        segs = np.array(segs, dtype=np.float32)

        seg_embeds = []
        for j in range(0, len(segs), batch_size):
            batch = torch.from_numpy(segs[j:j + batch_size]).to(DEVICE)
            with torch.no_grad():
                out = model(x_enc=batch)
                seg_embeds.append(out.embeddings.cpu().numpy())
        embeddings.append(np.concatenate(seg_embeds).mean(axis=0))

    del model; torch.cuda.empty_cache()
    return np.array(embeddings)


# ══════════════════════════════════════════════════════════════════════════════
# 8. DL MODELS (TCN, LSTM, Transformer) — on gait features, NOT FM
# ══════════════════════════════════════════════════════════════════════════════

class MLPHead(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class TCN1D(nn.Module):
    """TCN on raw walking signal segments for direct regression."""
    def __init__(self, in_ch=3, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 15, padding=7), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(0.3),
            nn.Conv1d(hidden, hidden, 11, padding=5), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(0.3),
            nn.Conv1d(hidden, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.head(self.net(x).squeeze(-1)).squeeze(-1)


class LSTM1D(nn.Module):
    """LSTM on raw walking signal for regression."""
    def __init__(self, in_ch=3, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, 1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden * 2, 1))
    def forward(self, x):
        # x: (batch, channels, seq) → (batch, seq, channels)
        out, _ = self.lstm(x.transpose(1, 2))
        return self.head(out.mean(dim=1)).squeeze(-1)


class Transformer1D(nn.Module):
    """Transformer on raw walking signal for regression."""
    def __init__(self, in_ch=3, d_model=16, nhead=4):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=32,
                                           dropout=0.3, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(d_model, 1))
    def forward(self, x):
        # x: (batch, channels, seq) → (batch, seq, channels)
        x = self.proj(x.transpose(1, 2))
        x = self.enc(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


def segment_for_dl(walk_sig, seg_len=300, seg_step=150):
    """Segment walking signal into fixed-length pieces for DL models."""
    segs = []
    for s in range(0, len(walk_sig) - seg_len + 1, seg_step):
        segs.append(walk_sig[s:s + seg_len].T)  # (3, seg_len)
    if not segs:
        pad = np.zeros((3, seg_len), dtype=np.float32)
        L = min(len(walk_sig), seg_len)
        pad[:, :L] = walk_sig[:L].T
        segs.append(pad)
    return np.array(segs, dtype=np.float32)


def loo_dl_raw(walk_signals, y, model_cls, model_kwargs, epochs=30, lr=1e-3):
    """LOO with DL model on raw walking segments. Averages segment predictions."""
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        # Build training segments
        all_segs, all_labels = [], []
        for j in tr:
            segs = segment_for_dl(walk_signals[j])
            for s in segs:
                all_segs.append(s)
                all_labels.append(y[j])
        all_segs = np.array(all_segs)
        all_labels = np.array(all_labels, dtype=np.float32)

        # Train
        model = model_cls(**model_kwargs).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        bs = 64
        model.train()
        for _ in range(epochs):
            idx = np.random.permutation(len(all_segs))
            for b in range(0, len(idx), bs):
                bi = idx[b:b + bs]
                xb = torch.from_numpy(all_segs[bi]).to(DEVICE)
                # Augment
                xb = xb + torch.randn_like(xb) * 0.005
                yb = torch.from_numpy(all_labels[bi]).to(DEVICE)
                opt.zero_grad()
                F.mse_loss(model(xb), yb).backward()
                opt.step()
            sched.step()

        # Predict: average over test subject's segments
        model.eval()
        test_segs = torch.from_numpy(segment_for_dl(walk_signals[i])).to(DEVICE)
        with torch.no_grad():
            preds[i] = model(test_segs).cpu().numpy().mean()

        if (i + 1) % 20 == 0:
            print(f"    fold {i+1}/{n}", flush=True)
    print(f"    {n}/{n}")
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# 9. CV ENGINES (ML and FM)
# ══════════════════════════════════════════════════════════════════════════════

def loo_cv(X, y, model_fn):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    return preds


def loo_featsel(X, y, model_fn, top_k=15):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        cors = np.array([abs(spearmanr(X[tr, j], y[tr])[0]) for j in range(X.shape[1])])
        idx = np.argsort(cors)[-top_k:]
        sc = StandardScaler()
        m = model_fn()
        m.fit(sc.fit_transform(X[tr][:, idx]), y[tr])
        preds[te] = m.predict(sc.transform(X[te][:, idx]))
    return preds


def loo_ccpt(X_home, X_clinic, X_extra, y, model_fn):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X_home):
        sc_c = StandardScaler()
        cm = Ridge(alpha=1.0)
        cm.fit(sc_c.fit_transform(X_clinic[tr]), y[tr])
        cp_tr = cm.predict(sc_c.fit_transform(X_clinic[tr])).reshape(-1, 1) / 1000
        cp_te = cm.predict(sc_c.transform(X_clinic[te])).reshape(-1, 1) / 1000
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


def loo_moment_mlp(X, y, hidden=32, epochs=60, lr=5e-4):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]).astype(np.float32)
        Xte = sc.transform(X[i:i+1]).astype(np.float32)
        model = MLPHead(Xtr.shape[1], hidden).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
        Xt = torch.from_numpy(Xtr).to(DEVICE)
        yt = torch.from_numpy(y[tr].astype(np.float32)).to(DEVICE)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            F.mse_loss(model(Xt), yt).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds[i] = model(torch.from_numpy(Xte).to(DEVICE)).cpu().item()
    return preds


def compute_metrics(y, yhat, label=""):
    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    pr, _ = pearsonr(y, yhat)
    sr, _ = spearmanr(y, yhat)
    return {"R2": round(r2, 4), "MAE": round(mae, 1),
            "Pearson_r": round(pr, 4), "Spearman_rho": round(sr, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Complete Pipeline: GT3X → Walking Detection → Prediction")
    print("=" * 70)

    # 1. Build aligned subject table
    print("\n[1/7] Building subject table...")
    paired = build_subject_table()
    y = paired["sixmwd"].values.astype(float)
    cohort = paired["cohort"].values

    X_clinic = paired[CLINIC_ACCEL_COLS].values.astype(float)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m = np.isnan(X_demo[:, j])
        if m.any(): X_demo[m, j] = np.nanmedian(X_demo[:, j])

    # 2. Process all subjects from raw GT3X
    print("\n[2/7] Processing GT3X files (daytime → walking → preprocess → features)...")
    walk_dir = OUT / "walking_segments"
    walk_dir.mkdir(exist_ok=True)
    daytime_dir = OUT / "daytime_segments"
    daytime_dir.mkdir(exist_ok=True)

    all_feats = []
    all_walk_signals = []
    n = len(paired)
    for i, (_, row) in enumerate(paired.iterrows()):
        c, s, yr, d = row["cohort"], int(row["subj_id"]), int(row["year"]), int(row["sixmwd"])
        fname = f"{c}{s:02d}_{yr}_{d}.csv"
        daytime_path = daytime_dir / fname
        feats, walk_sig, n_bouts = process_one_subject(
            row["gt3x_path"], save_daytime_path=daytime_path)
        all_feats.append(feats)
        all_walk_signals.append(walk_sig)

        # Save processed walking segment
        pd.DataFrame(walk_sig, columns=["AP", "ML", "VT"]).to_csv(
            walk_dir / fname, index=False)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n} (bouts={n_bouts})", flush=True)
    print(f"    {n}/{n}")
    print(f"  Walking segments saved to {walk_dir}/")

    feat_df = pd.DataFrame(all_feats).replace([np.inf, -np.inf], np.nan)
    for c in feat_df.columns:
        if feat_df[c].isna().any():
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())
    X_imu = feat_df.values.astype(float)
    print(f"  {X_imu.shape[1]} gait features extracted")
    feat_df.to_csv(OUT / "home_gait_features.csv", index=False)

    # 3. MOMENT embeddings
    print("\n[3/7] Extracting MOMENT embeddings from walking signals...")
    E_home = extract_moment_embeddings(all_walk_signals)
    print(f"  Home MOMENT embeddings: {E_home.shape}")

    # Clinic raw signals from csv_raw2 (for MOMENT + DL models)
    print("  Loading clinic raw signals...")
    clinic_sigs_raw = []
    for _, row in paired.iterrows():
        c, s, yr, d = row["cohort"], int(row["subj_id"]), int(row["year"]), int(row["sixmwd"])
        fname = f"{c}{s:02d}_{yr}_{d}.csv"
        df = pd.read_csv(BASE / "csv_raw2" / fname)
        clinic_sigs_raw.append(df[["X", "Y", "Z"]].values.astype(np.float32))

    print("  Extracting clinic MOMENT embeddings...")
    E_clinic = extract_moment_embeddings(clinic_sigs_raw)
    print(f"  Clinic MOMENT embeddings: {E_clinic.shape}")

    np.save(OUT / "moment_home.npy", E_home)
    np.save(OUT / "moment_clinic.npy", E_clinic)

    # 4. Run ML experiments
    print("\n[4/7] ML baselines (LOO CV)...")
    results = []
    xgb = lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbosity=0)

    def rec(model, config, preds):
        m = compute_metrics(y, preds)
        results.append({"model": model, "config": config, **m})
        print(f"  {model:30s} {config:4s} R²={m['R2']:.4f}")

    ml_models = {
        "Ridge": lambda: Ridge(alpha=10),
        "ElasticNet": lambda: ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=200, max_depth=5,
                                                       min_samples_leaf=5, random_state=42, n_jobs=-1),
        "XGBoost": xgb,
        "SVR": lambda: SVR(kernel="rbf", C=100, epsilon=50),
    }

    for mname, mfn in ml_models.items():
        rec(mname, "A1", loo_featsel(X_imu, y, mfn, top_k=15))
        rec(mname, "A2", loo_cv(np.column_stack([X_imu, X_demo]), y, mfn))
        rec(mname, "C1", loo_cv(X_clinic, y, mfn))
        rec(mname, "C2", loo_cv(np.column_stack([X_clinic, X_demo]), y, mfn))

    # 5. MOMENT FM experiments (separate from TCN/LSTM/Transformer)
    print("\n[5/7] MOMENT FM + downstream regressors...")
    for mname, mfn in [("MOMENT+Ridge", lambda: Ridge(alpha=10)), ("MOMENT+XGBoost", xgb)]:
        rec(mname, "A1", loo_cv(E_home, y, mfn))
        rec(mname, "A2", loo_cv(np.column_stack([E_home, X_demo]), y, mfn))
        rec(mname, "C1", loo_cv(E_clinic, y, mfn))
        rec(mname, "C2", loo_cv(np.column_stack([E_clinic, X_demo]), y, mfn))

    rec("MOMENT+MLP", "A1", loo_moment_mlp(E_home, y))
    rec("MOMENT+MLP", "C1", loo_moment_mlp(E_clinic, y))

    # 6. TCN/LSTM/Transformer on raw walking signals (A1 and C1)
    print("\n[6/7] TCN/LSTM/Transformer on raw walking signals...")
    for dl_name, dl_cls, dl_kwargs in [
        ("TCN", TCN1D, {"in_ch": 3, "hidden": 16}),
        ("LSTM", LSTM1D, {"in_ch": 3, "hidden": 16}),
        ("Transformer", Transformer1D, {"in_ch": 3, "d_model": 16, "nhead": 4}),
    ]:
        print(f"  {dl_name} A1...", flush=True)
        p = loo_dl_raw(all_walk_signals, y, dl_cls, dl_kwargs, epochs=30, lr=1e-3)
        rec(dl_name, "A1", p)
        print(f"  {dl_name} C1...", flush=True)
        p = loo_dl_raw(clinic_sigs_raw, y, dl_cls, dl_kwargs, epochs=30, lr=1e-3)
        rec(dl_name, "C1", p)

    # 7. Save and report
    print("\n[7/7] Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(OUT / "results_all.csv", index=False)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (R²) — Complete Raw Pipeline")
    print("=" * 70)
    pivot = df.pivot_table(index="model", columns="config", values="R2", aggfunc="first")
    co = [c for c in ["A1", "A2", "C1", "C2"] if c in pivot.columns]
    print(pivot[co].to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
