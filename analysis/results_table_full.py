#!/usr/bin/env python3
"""
Full results tables: all feature set combinations with and without feature selection.

Table 1: No feature selection on Gait/CWT/WS (appended as-is to fixed blocks)
Table 2: Spearman Top-20 on Gait/CWT/WS (selection inside LOO)

Fixed blocks (no selection applied):
  - PerBout-Top20: Spearman Top-20 inside LOO from 153/124 accel features
  - Demo: 4 features (cohort_POMS, Age, Sex, Height for clinic / BMI for home)
  - Demo-only row: uses BMI for both clinic and home

Input:  feats/*.csv + SwayDemographics.xlsx
Output: results/results_no_selection.csv
        results/results_spearman_top20.csv

Run:  python analysis/results_table_full.py
"""
import warnings, time, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
NPZ_DIR = BASE / 'home_full_recording_npz'
FT2M = 0.3048
C_ALPHA = 5
H_ALPHA = 20


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_combined(X_parts_fixed, X_parts_select, y, K_select=20, alpha=20):
    """
    LOO CV with fixed and selectable feature blocks.
    Fixed blocks: appended as-is (PerBout uses its own Spearman Top-20 per fold).
    Selectable blocks: concatenated, then Spearman Top-K applied inside LOO.
    """
    n = len(y)
    pr = np.zeros(n)

    # Separate PerBout from other fixed parts
    pb_info = None
    fixed_no_pb = []
    for X_part, label in X_parts_fixed:
        if label == 'PerBout':
            pb_info = X_part
        else:
            fixed_no_pb.append(X_part)

    # Concatenate selectable features
    if X_parts_select:
        X_selectable = np.column_stack([x for x, _ in X_parts_select])
        n_selectable = X_selectable.shape[1]
    else:
        X_selectable = None
        n_selectable = 0

    for i in range(n):
        tr = np.ones(n, dtype=bool); tr[i] = False
        parts_tr = []; parts_te = []

        # Fixed parts (Demo, etc.)
        for X_part in fixed_no_pb:
            parts_tr.append(X_part[tr])
            parts_te.append(X_part[i:i + 1])

        # PerBout-Top20 (own selection per fold)
        if pb_info is not None:
            n_pb = pb_info.shape[1]
            corrs = [abs(spearmanr(pb_info[tr, j], y[tr])[0])
                     if np.std(pb_info[tr, j]) > 0 else 0 for j in range(n_pb)]
            top_k_pb = sorted(range(n_pb), key=lambda j: corrs[j], reverse=True)[:20]
            parts_tr.append(pb_info[tr][:, top_k_pb])
            parts_te.append(pb_info[i:i + 1][:, top_k_pb])

        # Selectable parts (Spearman Top-K)
        if X_selectable is not None and n_selectable > 0:
            K_use = min(K_select, n_selectable)
            corrs = [abs(spearmanr(X_selectable[tr, j], y[tr])[0])
                     if np.std(X_selectable[tr, j]) > 0 else 0 for j in range(n_selectable)]
            top_k_sel = sorted(range(n_selectable), key=lambda j: corrs[j], reverse=True)[:K_use]
            parts_tr.append(X_selectable[tr][:, top_k_sel])
            parts_te.append(X_selectable[i:i + 1][:, top_k_sel])

        X_tr = np.column_stack(parts_tr) if parts_tr else np.zeros((tr.sum(), 0))
        X_te = np.column_stack(parts_te) if parts_te else np.zeros((1, 0))

        if X_tr.shape[1] == 0:
            pr[i] = y[tr].mean()
            continue

        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_tr), y[tr])
        pr[i] = m.predict(sc.transform(X_te))[0]

    r2 = r2_score(y, pr)
    mae = mean_absolute_error(y * FT2M, pr * FT2M)
    rho = spearmanr(y, pr)[0]
    return r2, mae, rho


def run_table(table_name, use_selection, sel_sets_c, sel_sets_h, sel_names,
              c_pb, h_pb, X_demo_clinic, X_demo_home, X_demo_shared, y):
    """Run all combinations and return results."""
    print(f'\n{"=" * 120}')
    print(f'{table_name}')
    print(f'{"=" * 120}')
    print(f'{"Row":<55s} {"C R²":>6s} {"C MAE":>6s} {"C ρ":>6s}  {"H R²":>6s} {"H MAE":>6s} {"H ρ":>6s}')
    print('-' * 120)

    rows = []

    # All combos of selectable sets (including empty)
    sel_combos = []
    for r in range(0, len(sel_names) + 1):
        for combo in itertools.combinations(sel_names, r):
            sel_combos.append(combo)

    for sel_combo in sel_combos:
        for use_pb in [False, True]:
            for use_demo in [False, True]:
                if not sel_combo and not use_pb and not use_demo:
                    continue

                label_parts = list(sel_combo)
                if use_pb:
                    label_parts.append('PerBout-Top20')
                if use_demo:
                    label_parts.append('Demo')
                label = '+'.join(label_parts)

                # --- CLINIC ---
                c_fixed = []; c_select = []
                if use_demo:
                    if not sel_combo and not use_pb:
                        c_fixed.append((X_demo_shared, 'Demo'))
                    else:
                        c_fixed.append((X_demo_clinic, 'Demo'))
                if use_pb:
                    c_fixed.append((c_pb, 'PerBout'))
                for sn in sel_combo:
                    if use_selection:
                        c_select.append((sel_sets_c[sn], sn))
                    else:
                        c_fixed.append((sel_sets_c[sn], sn))
                K_c = min(20, sum(x.shape[1] for x, _ in c_select)) if c_select else 0
                cr2, cmae, crho = loo_combined(c_fixed, c_select, y, K_select=K_c, alpha=C_ALPHA)

                # --- HOME ---
                h_fixed = []; h_select = []
                if use_demo:
                    if not sel_combo and not use_pb:
                        h_fixed.append((X_demo_shared, 'Demo'))
                    else:
                        h_fixed.append((X_demo_home, 'Demo'))
                if use_pb:
                    h_fixed.append((h_pb, 'PerBout'))
                for sn in sel_combo:
                    if use_selection:
                        h_select.append((sel_sets_h[sn], sn))
                    else:
                        h_fixed.append((sel_sets_h[sn], sn))
                K_h = min(20, sum(x.shape[1] for x, _ in h_select)) if h_select else 0
                hr2, hmae, hrho = loo_combined(h_fixed, h_select, y, K_select=K_h, alpha=H_ALPHA)

                print(f'{label:<55s} {cr2:>6.3f} {cmae:>6.1f} {crho:>6.3f}  '
                      f'{hr2:>6.3f} {hmae:>6.1f} {hrho:>6.3f}')
                rows.append({
                    'Feature Set': label,
                    'Clinic R²': round(cr2, 3), 'Clinic MAE (m)': round(cmae, 1),
                    'Clinic ρ': round(crho, 3),
                    'Home R²': round(hr2, 3), 'Home MAE (m)': round(hmae, 1),
                    'Home ρ': round(hrho, 3),
                })

    return rows


if __name__ == '__main__':
    t0 = time.time()

    # Load subjects + targets
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(y)

    # Demographics
    demo_xl = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo_xl['cohort'] = demo_xl['ID'].str.extract(r'^([A-Z])')[0]
    demo_xl['subj_id'] = demo_xl['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo_xl, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')

    # Demo: clinic=Height, home=BMI. Demo-only row uses BMI (shared).
    X_demo_clinic = impute(p[['cohort_POMS', 'Age', 'Sex', 'Height']].values.astype(float))
    X_demo_home = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))
    X_demo_shared = X_demo_home  # BMI for demo-only row

    # Load cached features
    FEATS = BASE / 'feats'
    c_gait = impute(pd.read_csv(FEATS / 'clinic_gait_features.csv').drop(columns='key').values.astype(float))
    c_cwt = impute(pd.read_csv(FEATS / 'clinic_cwt_features.csv').drop(columns='key').values.astype(float))
    c_ws = impute(pd.read_csv(FEATS / 'clinic_walksway_features.csv').drop(columns='key').values.astype(float))
    c_pb = impute(pd.read_csv(FEATS / 'clinic_perbout_features.csv').drop(columns='key').values.astype(float))
    h_gait = impute(pd.read_csv(FEATS / 'home_gait_features.csv').drop(columns='key').values.astype(float))
    h_cwt = impute(pd.read_csv(FEATS / 'home_cwt_features.csv').drop(columns='key').values.astype(float))
    h_ws = impute(pd.read_csv(FEATS / 'home_walksway_features.csv').drop(columns='key').values.astype(float))
    h_pb = impute(pd.read_csv(FEATS / 'home_perbout_features.csv').drop(columns='key').values.astype(float))

    # Selectable feature sets (Gait, CWT, WalkSway)
    sel_sets_c = {'Gait': c_gait, 'CWT': c_cwt, 'WalkSway': c_ws}
    sel_sets_h = {'Gait': h_gait, 'CWT': h_cwt, 'WalkSway': h_ws}
    sel_names = ['Gait', 'CWT', 'WalkSway']

    print(f'n={n}, LOO CV, Clinic α={C_ALPHA}, Home α={H_ALPHA}')
    print(f'PerBout-Top20: Spearman inside LOO (fixed block)')
    print(f'Demo-only row: cohort_POMS, Age, Sex, BMI (same for both)')
    print(f'Demo in combos: Clinic=Height, Home=BMI')

    # Table 1: No selection on Gait/CWT/WS
    t1 = run_table('TABLE 1: No Feature Selection on Gait/CWT/WS',
                    False, sel_sets_c, sel_sets_h, sel_names,
                    c_pb, h_pb, X_demo_clinic, X_demo_home, X_demo_shared, y)

    # Table 2: Spearman Top-20 on Gait/CWT/WS
    t2 = run_table('TABLE 2: Spearman Top-20 on Gait/CWT/WS',
                    True, sel_sets_c, sel_sets_h, sel_names,
                    c_pb, h_pb, X_demo_clinic, X_demo_home, X_demo_shared, y)

    # Save
    RESULTS = BASE / 'results'
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(t1).to_csv(RESULTS / 'results_no_selection.csv', index=False)
    pd.DataFrame(t2).to_csv(RESULTS / 'results_spearman_top20.csv', index=False)
    print(f'\nSaved results/results_no_selection.csv and results/results_spearman_top20.csv')
    print(f'Done in {time.time() - t0:.0f}s')
