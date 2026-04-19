#!/usr/bin/env python3
"""
Cadence-only parsimonious baselines for clinic 6MWD (LOO CV, n=101, m).

Reference / sanity-check script — NOT part of the paper results table.
Headline clinic model is Ridge over Gait+CWT+WS+Demo (55f, R²=0.806).

Configs:
  A. cadence alone (1f)                     → R²=0.471, MAE=57.1m
  B. cadence + Height (2f)                  → R²=0.680, MAE=45.3m
  C. cadence + Demo(4) (5f)                 → R²=0.712, MAE=43.3m
  D. mechanistic d = K*cadence*Height*360s  → R²=0.656, MAE=46.8m
  E. mechanistic + Demo residual Ridge      → R²=0.711, MAE=43.6m

Run: python clinic/predict_cadence_only.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent.parent
FT2M = 0.3048


def loo_ridge(X, y, alpha=5):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    return preds


def report(name, y_ft, pred_ft):
    y_m = y_ft * FT2M
    p_m = pred_ft * FT2M
    r2 = r2_score(y_ft, pred_ft)
    mae = mean_absolute_error(y_m, p_m)
    rho = spearmanr(y_ft, pred_ft)[0]
    r_v = pearsonr(y_ft, pred_ft)[0]
    print(f"  {name:42s}  R²={r2:+.3f}  MAE={mae:5.1f} m  r={r_v:+.3f}  ρ={rho:+.3f}")


def main():
    subj_df = pd.read_csv(BASE / 'home_full_recording_npz' / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)

    g = pd.read_csv(BASE / 'feats' / 'clinic_gait_features.csv')
    assert list(subj_df['key']) == list(g['key'])
    cadence = g['cadence_hz'].values.astype(float)  # Hz

    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    height_cm = p['Height'].fillna(p['Height'].median()).values
    height_m = height_cm / 100.0
    demo4 = p[['cohort_POMS', 'Age', 'Sex', 'Height']].copy()
    for c in demo4.columns:
        demo4[c] = pd.to_numeric(demo4[c], errors='coerce').fillna(demo4[c].median())
    demo4 = demo4.values

    print(f"\nCadence-only clinic baselines (LOO CV, n={len(y)})")
    print(f"  cadence_hz: mean={cadence.mean():.2f} Hz ({cadence.mean()*60:.1f} steps/min)")
    print(f"  6MWD true:  mean={y.mean()*FT2M:.1f} m\n")

    # A. cadence alone
    Xa = cadence.reshape(-1, 1)
    report("A. cadence alone", y, loo_ridge(Xa, y, alpha=1))

    # B. cadence + Height
    Xb = np.column_stack([cadence, height_cm])
    report("B. cadence + Height", y, loo_ridge(Xb, y, alpha=1))

    # C. cadence + Demo(4)
    Xc = np.column_stack([cadence, demo4])
    report("C. cadence + Demo(4)", y, loo_ridge(Xc, y, alpha=5))

    # D. mechanistic distance: d = K * cadence(Hz) * Height(m) * 360s
    #    LOO-calibrate scalar K from training fold
    raw = cadence * height_m * 360.0  # m, before K
    raw_ft = raw / FT2M
    pred_d = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(raw_ft):
        K = np.mean(y[tr]) / np.mean(raw_ft[tr])
        pred_d[te] = K * raw_ft[te]
    report("D. K * cadence * Height * 360s (LOO K)", y, pred_d)

    # E. mechanistic + Ridge residual on Demo(4)  [linear correction]
    pred_e = pred_d.copy()
    resid = y - pred_d
    pred_e_corr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(demo4):
        sc = StandardScaler()
        m = Ridge(alpha=5)
        m.fit(sc.fit_transform(demo4[tr]), resid[tr])
        pred_e_corr[te] = pred_e[te] + m.predict(sc.transform(demo4[te]))
    report("E. mechanistic + Demo residual Ridge", y, pred_e_corr)

    print(f"\n  Reference: Ridge full (Gait+CWT+WS+Demo, 55f) → R²=0.806, MAE=31.2 m")
    print(f"             Demo only (4f)                       → R²=0.362, MAE=60.8 m")


if __name__ == '__main__':
    main()
