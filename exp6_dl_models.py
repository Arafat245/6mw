#!/usr/bin/env python3
"""
Experiment 6: TCN, LSTM, Transformer on Raw Walking Signals
============================================================
DL models trained directly on raw accelerometer segments.
Separate from foundation models — these are trained from scratch.
Configs: A1, C1
CV: LOO
"""
import os, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
WALK_DIR = OUT / "walking_segments"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_table():
    home = pd.read_csv(BASE/"sway_features_home.csv").rename(columns={"year_x":"year"})
    return home[["cohort","subj_id","year","sixmwd"]]

def fname(r): return f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv"

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
        x = self.proj(x.transpose(1,2)); x = self.enc(x); return self.head(x.mean(1)).squeeze(-1)

def seg300(sig):
    segs=[]
    for s in range(0,len(sig)-300+1,150): segs.append(sig[s:s+300].T)
    if not segs:
        p=np.zeros((3,300),dtype=np.float32); L=min(len(sig),300); p[:,:L]=sig[:L].T; segs.append(p)
    return np.array(segs,dtype=np.float32)

def loo_dl(signals, y, model_cls, kw, epochs=30, lr=1e-3):
    n=len(y); preds=np.zeros(n)
    for i in range(n):
        tr=[j for j in range(n) if j!=i]
        all_s,all_l=[],[]
        for j in tr:
            for s in seg300(signals[j]): all_s.append(s); all_l.append(y[j])
        all_s=np.array(all_s); all_l=np.array(all_l,dtype=np.float32)
        m=model_cls(**kw).to(DEVICE)
        opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-2); bs=64
        m.train()
        for _ in range(epochs):
            idx=np.random.permutation(len(all_s))
            for b in range(0,len(idx),bs):
                bi=idx[b:b+bs]
                xb=torch.from_numpy(all_s[bi]).to(DEVICE)+torch.randn(len(bi),3,300,device=DEVICE)*0.005
                yb=torch.from_numpy(all_l[bi]).to(DEVICE)
                opt.zero_grad(); F.mse_loss(m(xb),yb).backward(); opt.step()
        m.eval()
        ts=torch.from_numpy(seg300(signals[i])).to(DEVICE)
        with torch.no_grad(): preds[i]=m(ts).cpu().numpy().mean()
        if (i+1)%20==0: print(f"      fold {i+1}/{n}",flush=True)
    print(f"      {n}/{n}")
    return preds

def met(y,yh):
    return {"R2":round(r2_score(y,yh),4),"MAE":round(mean_absolute_error(y,yh),1),
            "r":round(pearsonr(y,yh)[0],4)}

def main():
    print("="*60)
    print("Exp 6: TCN/LSTM/Transformer on Raw Signals")
    print(f"  Device: {DEVICE}")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    n = len(y)

    print("  Loading signals...")
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
    dl_models = [
        ("TCN", TCN1D, {"c":3,"h":16}),
        ("LSTM", LSTM1D, {"c":3,"h":16}),
        ("Transformer", Transformer1D, {"c":3,"d":16,"nh":4}),
    ]

    for dl_name, dl_cls, dl_kw in dl_models:
        print(f"\n  {dl_name} A1...", flush=True)
        p = loo_dl(home_sigs, y, dl_cls, dl_kw, epochs=30)
        m = met(y,p); results.append({"model":dl_name,"config":"A1",**m})
        print(f"    {dl_name:15s} A1 R²={m['R2']:.4f}")

        print(f"  {dl_name} C1...", flush=True)
        p = loo_dl(clinic_sigs, y, dl_cls, dl_kw, epochs=30)
        m = met(y,p); results.append({"model":dl_name,"config":"C1",**m})
        print(f"    {dl_name:15s} C1 R²={m['R2']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp6_dl.csv", index=False)
    print("\n"+"="*60)
    pivot = df.pivot_table(index="model",columns="config",values="R2",aggfunc="first")
    print(pivot.to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/exp6_dl.csv")

if __name__ == "__main__":
    main()
