#!/usr/bin/env python3
"""
Experiment 5: Foundation Models (MOMENT, Chronos, LimuBERT)
===========================================================
Extract frozen embeddings, use downstream Ridge/XGBoost.
PCA reduction for large embeddings to avoid overfitting.
Configs: A1, A2, C1, C2
CV: LOO
"""
import os, sys, json, warnings, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
WALK_DIR = OUT / "walking_segments"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 30.0
BASIC_DEMO = ["cohort_M","Age","Sex","Height","Weight","BMI"]
CLINIC_COLS = ["cadence_hz_c","step_time_cv_pct_c","acf_step_regularity_c","hr_ap_c","hr_vt_c",
               "ml_rms_g_c","ml_spectral_entropy_c","jerk_mean_abs_gps_c","enmo_mean_g_c","cadence_slope_per_min_c"]

def load_table():
    home = pd.read_csv(BASE/"sway_features_home.csv").rename(columns={"year_x":"year"})
    clinic = pd.read_csv(BASE/"features_top10.csv")
    cc = {c:c+"_c" for c in clinic.columns if c not in ("cohort","subj_id","year","sixmwd","fs")}
    clinic = clinic.rename(columns=cc).drop(columns=["sixmwd","fs"], errors="ignore")
    p = home[["cohort","subj_id","year","sixmwd"]].merge(clinic,on=["cohort","subj_id","year"],how="inner")
    demo = pd.read_excel(BASE/"SwayDemographics.xlsx")
    demo["cohort"]=demo["ID"].str.extract(r"^([A-Z])")[0]
    demo["subj_id"]=demo["ID"].str.extract(r"(\d+)")[0].astype(int)
    p = p.merge(demo,on=["cohort","subj_id"],how="left")
    p["cohort_M"]=(p["cohort"]=="M").astype(int)
    for c in ["Sex","Age","Height","Weight","BMI"]: p[c]=pd.to_numeric(p[c],errors="coerce")
    return p

def fname(r): return f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"

# ── MOMENT ──
def get_moment_emb(signals, bs=8):
    from momentfm import MOMENTPipeline
    m = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",model_kwargs={"task_name":"embedding"})
    m.init(); m=m.to(DEVICE); m.eval()
    embs = []
    for sig in signals:
        segs=[]
        for s in range(0,len(sig)-512+1,256): segs.append(sig[s:s+512].T)
        if not segs:
            p=np.zeros((3,512),dtype=np.float32); L=min(len(sig),512); p[:,:L]=sig[:L].T; segs.append(p)
        segs=np.array(segs,dtype=np.float32); se=[]
        for j in range(0,len(segs),bs):
            b=torch.from_numpy(segs[j:j+bs]).to(DEVICE)
            with torch.no_grad(): se.append(m(x_enc=b).embeddings.cpu().numpy())
        embs.append(np.concatenate(se).mean(0))
    del m; torch.cuda.empty_cache()
    return np.array(embs)

# ── Chronos ──
def get_chronos_emb(signals):
    from chronos import ChronosPipeline
    m = ChronosPipeline.from_pretrained("amazon/chronos-t5-small",device_map=DEVICE,dtype=torch.float32)
    embs = []
    for sig in signals:
        ae=[]
        for ax in range(min(sig.shape[1],3)):
            ts=torch.from_numpy(sig[:,ax]).unsqueeze(0)
            with torch.no_grad():
                e,_=m.embed(ts); ae.append(e.mean(1).cpu().numpy().squeeze())
        embs.append(np.concatenate(ae))
    del m; torch.cuda.empty_cache()
    return np.array(embs)

# ── LimuBERT ──
def get_limubert_emb(signals, bs=64):
    sys.path.insert(0, str(BASE/"limubert_repo"))
    from config import PretrainModelConfig
    from models import LIMUBertModel4Pretrain
    with open(BASE/"limubert_repo"/"config"/"limu_bert.json") as f:
        cfg = PretrainModelConfig.from_json(json.load(f)["base_v4"])
    m = LIMUBertModel4Pretrain(cfg, output_embed=True)
    st = torch.load(BASE/"limubert_repo"/"weights"/"limu_pretrain.pt",map_location="cpu",weights_only=False)
    m.load_state_dict(st, strict=False); m=m.to(DEVICE); m.eval()
    embs = []
    for sig in signals:
        if sig.shape[1]==3: sig6=np.column_stack([sig,np.zeros_like(sig)])
        else: sig6=sig
        segs=[]
        for s in range(0,len(sig6)-20+1,10): segs.append(sig6[s:s+20])
        if not segs:
            p=np.zeros((20,6),dtype=np.float32); L=min(len(sig6),20); p[:L]=sig6[:L]; segs.append(p)
        segs=np.array(segs,dtype=np.float32); se=[]
        for j in range(0,len(segs),bs):
            b=torch.from_numpy(segs[j:j+bs]).to(DEVICE)
            with torch.no_grad(): se.append(m(b).mean(1).cpu().numpy())
        embs.append(np.concatenate(se).mean(0))
    del m; torch.cuda.empty_cache()
    return np.array(embs)

def loo(X,y,mfn):
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        sc=StandardScaler(); m=mfn(); m.fit(sc.fit_transform(X[tr]),y[tr]); p[te]=m.predict(sc.transform(X[te]))
    return p

def met(y,yh):
    return {"R2":round(r2_score(y,yh),4),"MAE":round(mean_absolute_error(y,yh),1),
            "r":round(pearsonr(y,yh)[0],4)}

def main():
    print("="*60)
    print("Exp 5: Foundation Models (MOMENT, Chronos, LimuBERT)")
    print(f"  Device: {DEVICE}")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])

    # Load signals
    print("\n  Loading signals...")
    home_sigs, clinic_sigs = [], []
    for _,r in paired.iterrows():
        fn = fname(r)
        wp = WALK_DIR/fn
        if wp.exists():
            home_sigs.append(pd.read_csv(wp)[["AP","ML","VT"]].values.astype(np.float32))
        else:
            home_sigs.append(pd.read_csv(BASE/"csv_processed_home"/fn)[["AP","ML","VT"]].values.astype(np.float32))
        clinic_sigs.append(pd.read_csv(BASE/"csv_raw2"/fn)[["X","Y","Z"]].values.astype(np.float32))

    results = []
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)

    def rec(model,config,preds):
        m=met(y,preds); results.append({"model":model,"config":config,**m})
        print(f"    {model:30s} {config:4s} R²={m['R2']:.4f}")

    for fm_name, extract_fn in [("MOMENT", get_moment_emb), ("Chronos", get_chronos_emb), ("LimuBERT", get_limubert_emb)]:
        print(f"\n  --- {fm_name} ---")
        print(f"  Extracting home embeddings...", flush=True)
        Eh = extract_fn(home_sigs)
        print(f"    Home: {Eh.shape}")
        print(f"  Extracting clinic embeddings...", flush=True)
        Ec = extract_fn(clinic_sigs)
        print(f"    Clinic: {Ec.shape}")

        # PCA reduction if dim > 100
        if Eh.shape[1] > 100:
            pca_h = PCA(n_components=50).fit(Eh)
            Eh_pca = pca_h.transform(Eh)
            pca_c = PCA(n_components=50).fit(Ec)
            Ec_pca = pca_c.transform(Ec)
            variants = [(f"{fm_name}", Eh, Ec), (f"{fm_name}_PCA50", Eh_pca, Ec_pca)]
        else:
            variants = [(f"{fm_name}", Eh, Ec)]

        for label, Eh_v, Ec_v in variants:
            for dname, dfn in [("Ridge", lambda: Ridge(alpha=10)), ("XGBoost", xgb)]:
                tag = f"{label}+{dname}"
                rec(tag, "A1", loo(Eh_v, y, dfn))
                rec(tag, "A2", loo(np.column_stack([Eh_v, X_demo]), y, dfn))
                rec(tag, "C1", loo(Ec_v, y, dfn))
                rec(tag, "C2", loo(np.column_stack([Ec_v, X_demo]), y, dfn))

        # Save embeddings
        np.save(OUT/f"emb_{fm_name.lower()}_home.npy", Eh)
        np.save(OUT/f"emb_{fm_name.lower()}_clinic.npy", Ec)

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp5_fm.csv", index=False)
    print("\n"+"="*60)
    pivot = df.pivot_table(index="model",columns="config",values="R2",aggfunc="first")
    co = [c for c in ["A1","A2","C1","C2"] if c in pivot.columns]
    print(pivot[co].to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/exp5_fm.csv")

if __name__ == "__main__":
    main()
