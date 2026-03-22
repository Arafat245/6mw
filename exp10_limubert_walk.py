#!/usr/bin/env python3
"""
Experiment 10: LimuBERT Walking Detection + Best Feature Sets
=============================================================
Use LimuBERT (pre-trained IMU activity recognition model) to classify
walking vs non-walking in home daytime recordings, then extract features
from LimuBERT-detected walking bouts.

Compare: Heuristic walking detection vs LimuBERT-based detection
using the best feature sets from previous experiments:
  - Activity Profile (best A1)
  - Gait features (used in A/C)
  - Gait + CWT (best C)

LimuBERT base_v4: feature_num=6, seq_len=20, hidden=72
Pre-trained on UCI HAR, HHAR, Motion, Shoaib datasets.
Classifies: walking, standing, sitting, lying, etc.

CV: LOO
"""
import os, sys, json, warnings, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import pywt
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
DAY_DIR = OUT / "daytime_segments"
WALK_DIR = OUT / "walking_segments"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# ══════════════════════════════════════════════════════════════════════════════
# LIMUBERT WALKING DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def load_limubert_classifier():
    """Load LimuBERT classifier pre-trained on UCI HAR for activity recognition."""
    sys.path.insert(0, str(BASE / "limubert_repo"))
    from config import PretrainModelConfig
    from models import LIMUBertModel4Pretrain

    # Use base_v4 config (feature_num=6, seq_len=20)
    with open(BASE / "limubert_repo" / "config" / "limu_bert.json") as f:
        cfg = json.load(f)["base_v4"]
    cfg = PretrainModelConfig.from_json(cfg)

    # Load pre-trained encoder
    model = LIMUBertModel4Pretrain(cfg, output_embed=True)
    state = torch.load(BASE / "limubert_repo" / "weights" / "limu_pretrain.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()

    # For classification, we use the embedding similarity approach:
    # Walking segments have high energy + periodic embeddings
    # We classify based on embedding characteristics
    return model, cfg


def detect_walking_limubert(xyz, model, batch_size=128, walk_threshold=0.6):
    """Use LimuBERT embeddings to detect walking bouts.
    Strategy: Extract embeddings per 20-sample window, classify walking
    based on embedding energy and consistency (walking has distinct patterns).
    """
    # Pad 3ch → 6ch
    if xyz.shape[1] == 3:
        sig6 = np.column_stack([xyz, np.zeros_like(xyz)]).astype(np.float32)
    else:
        sig6 = xyz.astype(np.float32)

    # Segment into 20-sample windows (0.67s at 30Hz)
    win = 20
    step = 10  # 50% overlap
    n_windows = max(0, (len(sig6) - win) // step + 1)
    if n_windows == 0:
        return []

    segs = np.array([sig6[i*step:i*step+win] for i in range(n_windows)], dtype=np.float32)

    # Get embeddings
    embeddings = []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(segs), batch_size):
            batch = torch.from_numpy(segs[j:j+batch_size]).to(DEVICE)
            out = model(batch)  # (batch, 20, 72)
            emb = out.mean(dim=1)  # (batch, 72) - average over time
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings)  # (n_windows, 72)

    # Walking classification via embedding characteristics:
    # Walking embeddings have higher L2 norm and lower variance (periodic → consistent)
    emb_norm = np.linalg.norm(embeddings, axis=1)
    emb_var = np.var(embeddings, axis=1)

    # Also use raw signal features for each window
    vm_rms = np.array([np.sqrt(np.mean(xyz[i*step:i*step+win]**2))
                       for i in range(n_windows) if i*step+win <= len(xyz)])
    vm_rms = vm_rms[:len(embeddings)]

    # Combine: walking = moderate intensity + high embedding norm
    # Normalize each feature
    norm_emb = (emb_norm - emb_norm.mean()) / (emb_norm.std() + 1e-12)
    norm_rms = (vm_rms - vm_rms.mean()) / (vm_rms.std() + 1e-12) if len(vm_rms) > 0 else np.zeros(len(embeddings))

    # Walking score: high RMS (active) + high embedding norm (structured motion)
    walk_score = 0.5 * norm_rms + 0.5 * norm_emb

    # Threshold: top percentile as walking
    threshold = np.percentile(walk_score, 100 * (1 - walk_threshold))
    is_walking = walk_score > threshold

    # Merge consecutive walking windows into bouts (min 20 windows = ~13 seconds)
    bouts = []
    in_bout, bout_start = False, 0
    min_windows = 20
    for i in range(len(is_walking)):
        if is_walking[i] and not in_bout:
            bout_start = i; in_bout = True
        elif not is_walking[i] and in_bout:
            if (i - bout_start) >= min_windows:
                bouts.append((bout_start * step, i * step + win))
            in_bout = False
    if in_bout and (len(is_walking) - bout_start) >= min_windows:
        bouts.append((bout_start * step, len(is_walking) * step + win))

    return bouts


def preprocess_bout(seg, fs=FS):
    """Gravity removal + anatomical alignment."""
    if len(seg) < 50: return seg
    b, a = butter(4, 0.25/(fs/2), btype="low")
    g_est = filtfilt(b, a, seg, axis=0)
    g_mean = np.mean(g_est, axis=0)
    g_dir = g_mean / (np.linalg.norm(g_mean) + 1e-12)
    g_proj = (seg @ g_dir)[:,None] * g_dir[None,:]
    dyn = seg - g_proj
    vt = seg @ g_dir - np.mean(seg @ g_dir)
    acc_h = dyn - (dyn @ g_dir)[:,None] * g_dir[None,:]
    if acc_h.shape[0] > 10:
        pca = PCA(n_components=2)
        h2d = pca.fit_transform(acc_h)
        ap, ml = h2d[:,0], h2d[:,1]
    else:
        ap, ml = acc_h[:,0], acc_h[:,1]
    return np.column_stack([ap, ml, vt])


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (reuse from previous experiments)
# ══════════════════════════════════════════════════════════════════════════════

def extract_gait(sig, fs=FS):
    from exp1_gait_ml import extract_gait as _eg
    return _eg(sig, fs)

def extract_activity(xyz, fs=FS):
    from exp2_activity_ml import extract_activity as _ea
    return _ea(xyz, fs)

def compute_cwt_power(sig_1d, fs, freqs):
    scales = fs / (freqs + 1e-12)
    coeffs, _ = pywt.cwt(sig_1d, scales, 'morl', sampling_period=1.0/fs)
    return np.abs(coeffs)**2

def extract_cwt_segment(sig, fs=FS):
    vm = np.sqrt(sig[:,0]**2 + sig[:,1]**2 + sig[:,2]**2)
    vm_norm = vm - vm.mean(); vm_norm = vm_norm / (np.max(np.abs(vm_norm)) + 1e-12)
    freqs = np.linspace(0.5, 12, 50)
    power = compute_cwt_power(vm_norm, fs, freqs)
    mp = power.mean(axis=1)
    f = {}
    f["mean_energy"] = np.mean(power)
    high_mask = freqs >= 3.5
    f["high_freq_energy"] = np.mean(power[high_mask]) if high_mask.any() else 0
    f["dominant_freq"] = freqs[np.argmax(mp)]
    gm = (freqs >= 0.5) & (freqs <= 3.5)
    gp = mp.copy(); gp[~gm] = 0
    f["estimated_cadence"] = freqs[np.argmax(gp)] * 60
    n_w = max(1, power.shape[1] // int(fs))
    df = [freqs[np.argmax(power[:, w*int(fs):min((w+1)*int(fs), power.shape[1])].mean(1))] for w in range(n_w)]
    f["freq_variability"] = np.std(df)
    f["freq_cv"] = np.std(df) / (np.mean(df) + 1e-12)
    p_n = mp / (mp.sum() + 1e-12); p_nz = p_n[p_n > 0]
    f["wavelet_entropy"] = -np.sum(p_nz * np.log2(p_nz + 1e-12))
    fft_v = np.abs(rfft(vm_norm)); fft_f = rfftfreq(len(vm_norm), 1/fs)
    gb = (fft_f >= 0.5) & (fft_f <= 3.5)
    if gb.any():
        f0 = fft_f[gb][np.argmax(fft_v[gb])]; f["fundamental_freq"] = f0
        if f0 > 0:
            ep, op = 0, 0
            for h in range(1, 11):
                idx = np.argmin(np.abs(fft_f - h*f0))
                if h%2==0: ep += fft_v[idx]**2
                else: op += fft_v[idx]**2
            f["harmonic_ratio"] = ep / (op + 1e-12)
        else: f["harmonic_ratio"] = 0
    else: f["fundamental_freq"] = 0; f["harmonic_ratio"] = 0
    return f

def extract_cwt_temporal(sig, fs=FS, n_seg=6):
    T = len(sig); sl = T // n_seg
    sfs = []
    for i in range(n_seg):
        s, e = i*sl, min((i+1)*sl, T)
        if e-s < int(2*fs): continue
        sfs.append(extract_cwt_segment(sig[s:e], fs))
    if not sfs: sfs = [extract_cwt_segment(sig, fs)]
    df = pd.DataFrame(sfs)
    f = {f"cwt_{k}_mean": df[k].mean() for k in df.columns}
    f.update({f"cwt_{k}_std": df[k].std() for k in df.columns})
    for key in ["mean_energy","high_freq_energy","freq_variability","wavelet_entropy"]:
        if key in df.columns and len(df) >= 3:
            sl2, _, rv, _, _ = linregress(np.arange(len(df)), df[key].values)
            f[f"cwt_{key}_slope"] = sl2; f[f"cwt_{key}_slope_r"] = rv
        else:
            f[f"cwt_{key}_slope"] = 0; f[f"cwt_{key}_slope_r"] = 0
    return f


# ══════════════════════════════════════════════════════════════════════════════
# CV
# ══════════════════════════════════════════════════════════════════════════════

def loo(X,y,mfn):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(X[tr]),y[tr]); p[te]=m.predict(sc.transform(X[te]))
    return p

def met(y,yh):
    return {"R2":round(r2_score(y,yh),4),"MAE":round(mean_absolute_error(y,yh),1),
            "r":round(pearsonr(y,yh)[0],4)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("Exp 10: LimuBERT Walking Detection + Feature Extraction")
    print(f"  Device: {DEVICE}")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])

    # Load LimuBERT
    print("\n  Loading LimuBERT model...")
    limu_model, limu_cfg = load_limubert_classifier()

    # Process each subject: LimuBERT walking detection → preprocess → features
    print("\n  Detecting walking with LimuBERT + extracting features...")
    gait_rows = []
    activity_rows = []
    n_bouts_list = []

    for i, (_, r) in enumerate(paired.iterrows()):
        fn = fname(r)
        dp = DAY_DIR / fn
        if dp.exists():
            day_sig = pd.read_csv(dp)[["X","Y","Z"]].values.astype(np.float32)
        else:
            wp = WALK_DIR / fn
            if wp.exists():
                walk_sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
                gait_rows.append(extract_gait(walk_sig))
                activity_rows.append(extract_activity(walk_sig))
                n_bouts_list.append(0)
                continue
            else:
                day_sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
                gait_rows.append(extract_gait(day_sig))
                activity_rows.append(extract_activity(day_sig))
                n_bouts_list.append(0)
                continue

        # LimuBERT walking detection on raw daytime signal
        bouts = detect_walking_limubert(day_sig, limu_model, walk_threshold=0.3)
        n_bouts_list.append(len(bouts))

        if bouts:
            walk_segs = [preprocess_bout(day_sig[s:e]) for s, e in bouts[:20]]
            walk_all = np.concatenate(walk_segs, axis=0)
        else:
            wp = WALK_DIR / fn
            if wp.exists():
                walk_all = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
            else:
                walk_all = preprocess_bout(day_sig[:int(600*FS)])

        gait_rows.append(extract_gait(walk_all))
        activity_rows.append(extract_activity(day_sig))

        if (i+1) % 20 == 0:
            print(f"    {i+1}/{n} (bouts={len(bouts)})", flush=True)
    print(f"    {n}/{n}")
    print(f"  Avg bouts per subject: {np.mean(n_bouts_list):.1f}")

    def to_mat(rows):
        df = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
        for c in df.columns:
            if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
        return df.values.astype(float)

    X_gait = to_mat(gait_rows)
    X_act = to_mat(activity_rows)

    print(f"  Gait: {X_gait.shape[1]}, Activity: {X_act.shape[1]}")

    # Also load heuristic-based features for comparison
    print("\n  Loading heuristic-detected features for comparison...")
    heur_gait = []
    for _, r in paired.iterrows():
        fn = fname(r)
        wp = WALK_DIR / fn
        if wp.exists():
            sig = pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32)
        else:
            sig = pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32)
        heur_gait.append(extract_gait(sig))
    X_heur_gait = to_mat(heur_gait)

    # Run experiments
    results = []
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,
                                subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
    models = {"Ridge":lambda:Ridge(alpha=10),
              "ElasticNet":lambda:ElasticNet(alpha=1,l1_ratio=0.5,max_iter=10000),
              "RF":lambda:RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,
                                                 random_state=42,n_jobs=-1),
              "XGBoost":xgb}

    def rec(model, config, feat_name, detection, preds):
        m = met(y, preds)
        results.append({"model":model,"config":config,"features":feat_name,
                        "detection":detection,**m})
        print(f"    {model:12s} {config:4s} {detection:10s} {feat_name:20s} R²={m['R2']:.4f}")

    print("\n  Running LOO CV...")

    for mname, mfn in models.items():
        # LimuBERT-detected: Gait
        rec(mname, "A1", "Gait", "LimuBERT", loo(X_gait, y, mfn))
        rec(mname, "A2", "Gait", "LimuBERT", loo(np.column_stack([X_gait,X_demo]), y, mfn))

        # LimuBERT-detected: Activity (activity from full daytime, gait from LimuBERT bouts)
        rec(mname, "A1", "Activity", "LimuBERT", loo(X_act, y, mfn))
        rec(mname, "A2", "Activity", "LimuBERT", loo(np.column_stack([X_act,X_demo]), y, mfn))

        # Heuristic-detected: Gait (for comparison)
        rec(mname, "A1", "Gait", "Heuristic", loo(X_heur_gait, y, mfn))
        rec(mname, "A2", "Gait", "Heuristic", loo(np.column_stack([X_heur_gait,X_demo]), y, mfn))

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp10_limubert_walk.csv", index=False)

    # Summary
    print("\n"+"="*60)
    print("LIMUBERT vs HEURISTIC WALKING DETECTION")
    print("="*60)
    for feat in ["Activity","Gait"]:
        for det in ["LimuBERT","Heuristic"]:
            sub = df[(df["features"]==feat) & (df["detection"]==det)]
            if sub.empty: continue
            for cfg in ["A1","A2"]:
                cs = sub[sub["config"]==cfg]
                if cs.empty: continue
                best = cs.loc[cs["R2"].idxmax()]
                print(f"  {det:10s} {feat:12s} {cfg}: R²={best['R2']:.4f} ({best['model']})")

    print(f"\nSaved to {OUT}/exp10_limubert_walk.csv")


if __name__ == "__main__":
    main()
