#!/usr/bin/env python3
"""
Home-to-Clinic Domain Adaptation for 6MWD Prediction
=====================================================
Learn a mapping from home gait features → clinic gait feature space
using paired data (same subjects, both recordings).

Methods:
  1. Linear Mapping (Ridge Regression)
  2. Nonlinear Mapping (MLP Adapter)
  3. CCA (Canonical Correlation Analysis)

No data leakage: mapping learned on N-1 subjects per LOO fold.
The mapping is UNSUPERVISED (no 6MWD labels used for alignment).

CV: LOO
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

FEATS = Path('feats')
OUT = Path('results_raw_pipeline')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ids = pd.read_csv(FEATS / 'target_6mwd.csv')
y = ids['sixmwd'].values.astype(float)
cohort = ids['cohort'].values

def load(name):
    df = pd.read_csv(FEATS / name).drop(columns=['cohort','subj_id','year','sixmwd'], errors='ignore')
    X = df.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        if X[c].isna().any(): X[c] = X[c].fillna(X[c].median())
    return X.values.astype(float)

X_home = load('home_gait10.csv')      # 10 features
X_clinic = load('clinic_gait10.csv')  # 10 features (same columns)
X_act = load('home_activity_profile_v2.csv')  # 16 features

demo = pd.read_csv(FEATS / 'demographics.csv').drop(columns=['cohort','subj_id','year','sixmwd'])
for c in demo.columns: demo[c] = pd.to_numeric(demo[c], errors='coerce')
for c in demo.columns:
    if demo[c].isna().any(): demo[c] = demo[c].fillna(demo[c].median())
X_demo4 = demo[['cohort_M','Age','Sex','BMI']].values.astype(float)


# ══════════════════════════════════════════════════════════════════
# METHOD 1: Linear Mapping (Ridge)
# ══════════════════════════════════════════════════════════════════

def loo_linear_adapt(X_home, X_clinic, X_extra, y, model_fn, alpha=1.0):
    """LOO with linear domain adaptation: home → clinic mapping."""
    n = len(y)
    preds = np.zeros(n)
    mapping_errors = []

    for tr, te in LeaveOneOut().split(X_home):
        # Step 1: Learn mapping on training subjects (UNSUPERVISED — no y used)
        sc_h = StandardScaler()
        sc_c = StandardScaler()
        Xh_tr = sc_h.fit_transform(X_home[tr])
        Xc_tr = sc_c.fit_transform(X_clinic[tr])

        # Ridge regression: home → clinic (per-feature)
        mapper = Ridge(alpha=alpha)
        mapper.fit(Xh_tr, Xc_tr)

        # Step 2: Map test subject's home features
        Xh_te = sc_h.transform(X_home[te])
        Xh_te_mapped = mapper.predict(Xh_te)

        # Map training home features too (for predictor training)
        Xh_tr_mapped = mapper.predict(Xh_tr)

        # Mapping quality
        mapping_errors.append(np.mean((Xh_tr_mapped - Xc_tr)**2))

        # Step 3: Add extra features if any
        if X_extra is not None:
            sc_e = StandardScaler()
            Xe_tr = sc_e.fit_transform(X_extra[tr])
            Xe_te = sc_e.transform(X_extra[te])
            X_tr_final = np.column_stack([Xh_tr_mapped, Xe_tr])
            X_te_final = np.column_stack([Xh_te_mapped, Xe_te])
        else:
            X_tr_final = Xh_tr_mapped
            X_te_final = Xh_te_mapped

        # Step 4: Train predictor on mapped features → 6MWD
        m = model_fn()
        m.fit(X_tr_final, y[tr])
        preds[te] = m.predict(X_te_final)

    return r2_score(y, preds), np.mean(mapping_errors), preds


# ══════════════════════════════════════════════════════════════════
# METHOD 2: Nonlinear Mapping (MLP Adapter)
# ══════════════════════════════════════════════════════════════════

class MLPAdapter(nn.Module):
    def __init__(self, in_dim=10, hidden=16, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def loo_mlp_adapt(X_home, X_clinic, X_extra, y, model_fn, epochs=100, lr=1e-3):
    """LOO with MLP domain adaptation."""
    n = len(y)
    preds = np.zeros(n)

    for tr, te in LeaveOneOut().split(X_home):
        sc_h = StandardScaler()
        sc_c = StandardScaler()
        Xh_tr = sc_h.fit_transform(X_home[tr]).astype(np.float32)
        Xc_tr = sc_c.fit_transform(X_clinic[tr]).astype(np.float32)
        Xh_te = sc_h.transform(X_home[te]).astype(np.float32)

        # Train MLP adapter: home → clinic (UNSUPERVISED)
        adapter = MLPAdapter(X_home.shape[1], 16, X_clinic.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=1e-3)

        Xh_t = torch.from_numpy(Xh_tr).to(DEVICE)
        Xc_t = torch.from_numpy(Xc_tr).to(DEVICE)

        adapter.train()
        for _ in range(epochs):
            opt.zero_grad()
            mapped = adapter(Xh_t)
            loss = F.mse_loss(mapped, Xc_t)
            loss.backward()
            opt.step()

        # Map
        adapter.eval()
        with torch.no_grad():
            Xh_tr_mapped = adapter(Xh_t).cpu().numpy()
            Xh_te_mapped = adapter(torch.from_numpy(Xh_te).to(DEVICE)).cpu().numpy()

        # Add extra features
        if X_extra is not None:
            sc_e = StandardScaler()
            Xe_tr = sc_e.fit_transform(X_extra[tr])
            Xe_te = sc_e.transform(X_extra[te])
            X_tr_final = np.column_stack([Xh_tr_mapped, Xe_tr])
            X_te_final = np.column_stack([Xh_te_mapped, Xe_te])
        else:
            X_tr_final = Xh_tr_mapped
            X_te_final = Xh_te_mapped

        m = model_fn()
        m.fit(X_tr_final, y[tr])
        preds[te] = m.predict(X_te_final)

    return r2_score(y, preds), preds


# ══════════════════════════════════════════════════════════════════
# METHOD 3: CCA
# ══════════════════════════════════════════════════════════════════

def loo_cca_adapt(X_home, X_clinic, X_extra, y, model_fn, n_components=5):
    """LOO with CCA domain adaptation."""
    n = len(y)
    preds = np.zeros(n)

    for tr, te in LeaveOneOut().split(X_home):
        sc_h = StandardScaler()
        sc_c = StandardScaler()
        Xh_tr = sc_h.fit_transform(X_home[tr])
        Xc_tr = sc_c.fit_transform(X_clinic[tr])
        Xh_te = sc_h.transform(X_home[te])

        # CCA: find shared space (UNSUPERVISED)
        cca = CCA(n_components=n_components)
        cca.fit(Xh_tr, Xc_tr)
        Xh_tr_cca, _ = cca.transform(Xh_tr, Xc_tr)
        Xh_te_cca = cca.transform(Xh_te)
        # CCA transform returns tuple for single input
        if isinstance(Xh_te_cca, tuple):
            Xh_te_cca = Xh_te_cca[0]

        if X_extra is not None:
            sc_e = StandardScaler()
            Xe_tr = sc_e.fit_transform(X_extra[tr])
            Xe_te = sc_e.transform(X_extra[te])
            X_tr_final = np.column_stack([Xh_tr_cca, Xe_tr])
            X_te_final = np.column_stack([Xh_te_cca, Xe_te])
        else:
            X_tr_final = Xh_tr_cca
            X_te_final = Xh_te_cca

        m = model_fn()
        m.fit(X_tr_final, y[tr])
        preds[te] = m.predict(X_te_final)

    return r2_score(y, preds), preds


# ══════════════════════════════════════════════════════════════════
# RAW BASELINE (no adaptation)
# ══════════════════════════════════════════════════════════════════

def loo_raw(X, y, model_fn):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = model_fn()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, preds), preds


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Home-to-Clinic Domain Adaptation for 6MWD Prediction")
    print("=" * 70)

    xgb = lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    rf = lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                        random_state=42, n_jobs=-1)
    models = {'Ridge': lambda: Ridge(alpha=10),
              'ElasticNet': lambda: ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000),
              'RF': rf, 'XGBoost': xgb}

    results = []

    def rec(method, config, model, r2, preds=None):
        results.append({'method': method, 'config': config, 'model': model, 'R2': round(r2, 4)})
        print(f"  {method:25s} {config:4s} {model:12s} R²={r2:.4f}")
        if preds is not None:
            # Save predictions
            ms = cohort == 'M'
            ct = cohort == 'C'
            r2_ms = r2_score(y[ms], preds[ms]) if ms.sum() > 2 else np.nan
            r2_ct = r2_score(y[ct], preds[ct]) if ct.sum() > 2 else np.nan

    # ── BASELINES ──
    print("\n--- Baselines (no adaptation) ---")
    for mname, mfn in models.items():
        r2, p = loo_raw(X_home, y, mfn)
        rec('Raw home gait', 'A1', mname, r2)

    for mname, mfn in models.items():
        r2, p = loo_raw(np.column_stack([X_home, X_demo4]), y, mfn)
        rec('Raw home gait+demo', 'A2', mname, r2)

    for mname, mfn in models.items():
        r2, p = loo_raw(X_act, y, mfn)
        rec('Activity Profile', 'A1', mname, r2)

    for mname, mfn in models.items():
        r2, p = loo_raw(np.column_stack([X_act, X_demo4]), y, mfn)
        rec('Activity+demo', 'A2', mname, r2)

    for mname, mfn in models.items():
        r2, p = loo_raw(X_clinic, y, mfn)
        rec('Clinic gait (upper bound)', 'C1', mname, r2)

    # ── METHOD 1: Linear Mapping ──
    print("\n--- Method 1: Linear Mapping (Ridge) ---")
    for alpha in [0.1, 1.0, 10.0]:
        for mname, mfn in models.items():
            r2, err, p = loo_linear_adapt(X_home, X_clinic, None, y, mfn, alpha=alpha)
            rec(f'Linear(α={alpha})', 'A1', mname, r2)

    # Best alpha with demo
    for mname, mfn in models.items():
        r2, err, p = loo_linear_adapt(X_home, X_clinic, X_demo4, y, mfn, alpha=1.0)
        rec('Linear(α=1)+demo', 'A2', mname, r2)

    # Linear + Activity
    for mname, mfn in models.items():
        r2, err, p = loo_linear_adapt(X_home, X_clinic, X_act, y, mfn, alpha=1.0)
        rec('Linear+Activity', 'A1', mname, r2)

    for mname, mfn in models.items():
        r2, err, p = loo_linear_adapt(X_home, X_clinic, np.column_stack([X_act, X_demo4]), y, mfn, alpha=1.0)
        rec('Linear+Activity+demo', 'A2', mname, r2)

    # ── METHOD 2: MLP Adapter ──
    print("\n--- Method 2: MLP Adapter ---")
    for mname, mfn in models.items():
        r2, p = loo_mlp_adapt(X_home, X_clinic, None, y, mfn)
        rec('MLP Adapter', 'A1', mname, r2)

    for mname, mfn in models.items():
        r2, p = loo_mlp_adapt(X_home, X_clinic, X_demo4, y, mfn)
        rec('MLP Adapter+demo', 'A2', mname, r2)

    for mname, mfn in models.items():
        r2, p = loo_mlp_adapt(X_home, X_clinic, np.column_stack([X_act, X_demo4]), y, mfn)
        rec('MLP+Activity+demo', 'A2', mname, r2)

    # ── METHOD 3: CCA ──
    print("\n--- Method 3: CCA ---")
    for n_comp in [3, 5, 7]:
        for mname, mfn in models.items():
            r2, p = loo_cca_adapt(X_home, X_clinic, None, y, mfn, n_components=n_comp)
            rec(f'CCA(k={n_comp})', 'A1', mname, r2)

    for mname, mfn in models.items():
        r2, p = loo_cca_adapt(X_home, X_clinic, X_demo4, y, mfn, n_components=5)
        rec('CCA(k=5)+demo', 'A2', mname, r2)

    for mname, mfn in models.items():
        r2, p = loo_cca_adapt(X_home, X_clinic, np.column_stack([X_act, X_demo4]), y, mfn, n_components=5)
        rec('CCA+Activity+demo', 'A2', mname, r2)

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUT / 'exp_domain_adaptation.csv', index=False)

    # Summary
    print("\n" + "=" * 70)
    print("BEST PER METHOD AND CONFIG")
    print("=" * 70)
    for cfg in ['A1', 'A2', 'C1']:
        sub = df[df['config'] == cfg]
        if sub.empty: continue
        best = sub.loc[sub['R2'].idxmax()]
        print(f"  {cfg}: R²={best['R2']:.4f} ({best['model']}, {best['method']})")

    # Save best predictions
    print("\n--- Saving best predictions ---")
    for cfg, method_fn, X_extra_best in [
        ('A1', lambda mfn: loo_linear_adapt(X_home, X_clinic, None, y, mfn, alpha=1.0),
         None),
        ('A2', lambda mfn: loo_linear_adapt(X_home, X_clinic, np.column_stack([X_act, X_demo4]), y, mfn, alpha=1.0),
         None),
    ]:
        # Use best model from results
        sub = df[df['config'] == cfg]
        best = sub.loc[sub['R2'].idxmax()]
        best_model = models[best['model']]
        r2, _, preds = method_fn(best_model) if 'Linear' in best['method'] else (0, 0, np.zeros(len(y)))
        if r2 > 0:
            ms = cohort == 'M'
            ct = cohort == 'C'
            pred_df = pd.DataFrame({
                'cohort': cohort, 'subj_id': ids['subj_id'].values,
                'y_true': y, 'y_pred': preds, 'residual': y - preds,
            })
            pred_df.to_csv(OUT / f'predictions_v2/DA_{cfg}_{best["model"]}.csv', index=False)
            print(f"  {cfg}: R²={r2:.4f}, MS={r2_score(y[ms], preds[ms]):.4f}, Ctrl={r2_score(y[ct], preds[ct]):.4f}")

    print(f"\nSaved to {OUT}/exp_domain_adaptation.csv")


if __name__ == '__main__':
    main()
