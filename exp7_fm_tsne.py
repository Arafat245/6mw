#!/usr/bin/env python3
"""
Experiment 7: FM Embeddings with t-SNE and without reduction
============================================================
Compare: Full embeddings vs PCA vs t-SNE for FM downstream regression.
Uses cached embeddings from Exp 5.
Configs: A1, A2, C1, C2
CV: LOO
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
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
    print("Exp 7: FM Embeddings — Full vs PCA vs t-SNE")
    print("="*60)
    paired = load_table()
    y = paired["sixmwd"].values.astype(float)
    X_demo = paired[BASIC_DEMO].values.astype(float)
    for j in range(X_demo.shape[1]):
        m=np.isnan(X_demo[:,j])
        if m.any(): X_demo[m,j]=np.nanmedian(X_demo[:,j])

    # Load cached embeddings
    fm_data = {}
    for fm in ["moment", "chronos", "limubert"]:
        hp = OUT/f"emb_{fm}_home.npy"
        cp = OUT/f"emb_{fm}_clinic.npy"
        if hp.exists() and cp.exists():
            fm_data[fm] = {"home": np.load(hp), "clinic": np.load(cp)}
            print(f"  {fm}: home {fm_data[fm]['home'].shape}, clinic {fm_data[fm]['clinic'].shape}")

    results = []
    xgb = lambda: XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,
                                colsample_bytree=0.8,random_state=42,verbosity=0)

    def rec(model,config,preds):
        m=met(y,preds); results.append({"model":model,"config":config,**m})
        print(f"    {model:35s} {config:4s} R²={m['R2']:.4f}")

    for fm_name, data in fm_data.items():
        Eh, Ec = data["home"], data["clinic"]
        dim = Eh.shape[1]
        print(f"\n  --- {fm_name.upper()} (dim={dim}) ---")

        # Reduction methods
        reductions = [("full", Eh, Ec)]

        if dim > 50:
            # PCA
            Eh_pca = PCA(n_components=50).fit_transform(Eh)
            Ec_pca = PCA(n_components=50).fit_transform(Ec)
            reductions.append(("PCA50", Eh_pca, Ec_pca))

            # t-SNE (reduce to 10 components via PCA first, then t-SNE to 3)
            # t-SNE needs perplexity < n_samples
            for n_comp in [3, 10]:
                Eh_tsne = TSNE(n_components=min(n_comp, 3), perplexity=min(30, len(Eh)-1),
                               random_state=42, init="pca").fit_transform(Eh)
                Ec_tsne = TSNE(n_components=min(n_comp, 3), perplexity=min(30, len(Ec)-1),
                               random_state=42, init="pca").fit_transform(Ec)
                reductions.append((f"tSNE{min(n_comp,3)}", Eh_tsne, Ec_tsne))
        else:
            # Small dim (LimuBERT=72) — just t-SNE
            for n_comp in [3]:
                Eh_tsne = TSNE(n_components=n_comp, perplexity=min(30, len(Eh)-1),
                               random_state=42, init="pca").fit_transform(Eh)
                Ec_tsne = TSNE(n_components=n_comp, perplexity=min(30, len(Ec)-1),
                               random_state=42, init="pca").fit_transform(Ec)
                reductions.append((f"tSNE{n_comp}", Eh_tsne, Ec_tsne))

        for red_name, Eh_r, Ec_r in reductions:
            for dname, dfn in [("Ridge", lambda: Ridge(alpha=10)), ("XGBoost", xgb)]:
                tag = f"{fm_name}_{red_name}+{dname}"
                rec(tag, "A1", loo(Eh_r, y, dfn))
                rec(tag, "A2", loo(np.column_stack([Eh_r, X_demo]), y, dfn))
                rec(tag, "C1", loo(Ec_r, y, dfn))
                rec(tag, "C2", loo(np.column_stack([Ec_r, X_demo]), y, dfn))

    df = pd.DataFrame(results)
    df.to_csv(OUT/"exp7_fm_tsne.csv", index=False)
    print("\n"+"="*60)
    pivot = df.pivot_table(index="model",columns="config",values="R2",aggfunc="first")
    co = [c for c in ["A1","A2","C1","C2"] if c in pivot.columns]
    print(pivot[co].to_string(float_format="{:.4f}".format))
    print(f"\nSaved to {OUT}/exp7_fm_tsne.csv")

if __name__ == "__main__":
    main()
