#!/usr/bin/env python3
"""
Comprehensive results table: all feature sets and combinations.
Home: First 6 min of longest bout (clinic-free) + PerBout-Top20
Clinic: Full 6MWT
Same preprocessing, same feature counts.
"""
import sys, warnings, time, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent
sys.path.insert(0, str(BASE))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


def eval_loo(X, y, alpha=10):
    n, p = X.shape
    pr = np.zeros(n)
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    r2 = r2_score(y, pr)
    mae = mean_absolute_error(y, pr)
    rho = spearmanr(y, pr)[0]
    # Adjusted R²
    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan
    return r2, mae, rho, adj_r2


def best_alpha(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 0, 10)
    for a in alphas:
        r2, mae, rho, adj_r2 = eval_loo(X, y, a)
        if r2 > best[0]: best = (r2, mae, rho, adj_r2, a)
    return best


if __name__ == '__main__':
    t0 = time.time()

    # ── Load cached features from previous run ──
    cache = OUT / 'longest_bout_proper_cache.npz'
    if not cache.exists():
        print("ERROR: Run exp_longest_bout_segments.py first to generate cache.")
        sys.exit(1)

    d = np.load(cache, allow_pickle=True)
    home_configs = d['home_configs'].item()
    clinic_configs = d['clinic_configs'].item()

    # Use first6 for home, full for clinic
    h_cfg = home_configs['first6']
    c_cfg = clinic_configs['full']  # full = first6 = last6 for clinic

    h_valid = np.array(h_cfg['valid'], dtype=bool)
    c_valid = np.array(c_cfg['valid'], dtype=bool)
    both_valid = h_valid & c_valid
    nv = both_valid.sum()

    # Build Gait/CWT/WalkSway matrices
    def build_matrix(feat_list, valid_mask):
        df = pd.DataFrame([f for f, v in zip(feat_list, valid_mask) if v])
        X = df.replace([np.inf, -np.inf], np.nan).values.astype(float)
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.all(): X[:, j] = 0
            elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
        return X

    X_gait_h = build_matrix(h_cfg['gait'], both_valid)
    X_cwt_h = build_matrix(h_cfg['cwt'], both_valid)
    X_ws_h = build_matrix(h_cfg['ws'], both_valid)

    X_gait_c = build_matrix(c_cfg['gait'], both_valid)
    X_cwt_c = build_matrix(c_cfg['cwt'], both_valid)
    X_ws_c = build_matrix(c_cfg['ws'], both_valid)

    # Load subjects
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid_mask = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                                  for _, r in ids101.iterrows()])
    ids_v = ids101[clinic_valid_mask].reset_index(drop=True)
    y = ids_v['sixmwd'].values.astype(float)[both_valid[clinic_valid_mask]]

    # Wait — need to align both_valid to the right indexing
    # both_valid has length = len(ids_v) = n subjects with clinic files
    y = ids_v['sixmwd'].values.astype(float)
    y = y[both_valid]

    # Demo(4)
    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    # Demo(4) for clinic, Demo(5) for home
    X_demo4_all = p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float)
    X_demo5_all = p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float)
    for X in [X_demo4_all, X_demo5_all]:
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any(): X[m, j] = np.nanmedian(X[:, j])
    X_demo4 = X_demo4_all[both_valid]
    X_demo5 = X_demo5_all[both_valid]

    # PerBout-Top20 (clinic-free, from NPZ)
    cf_npz = np.load(BASE / 'feats' / 'home_clinicfree_top20.npz')
    X_perbout_all = cf_npz['X']  # (101, 20) aligned to ids101
    # Map to ids_v[both_valid]
    _ids101 = ids101
    cidx = np.where(clinic_valid_mask)[0]
    pb_idx = []
    for idx in np.where(both_valid)[0]:
        r = ids_v.iloc[idx]
        match = _ids101[(_ids101['cohort'] == r['cohort']) & (_ids101['subj_id'] == r['subj_id']) & (_ids101['sixmwd'] == r['sixmwd'])]
        if len(match) > 0: pb_idx.append(match.index[0])
    X_perbout = X_perbout_all[pb_idx]
    for j in range(X_perbout.shape[1]):
        m = np.isnan(X_perbout[:, j])
        if m.any(): X_perbout[m, j] = np.nanmedian(X_perbout[:, j])

    # Foundation models
    valid_102 = ~((ids['cohort'] == 'M') & (ids['subj_id'] == 22))
    all_cidx = np.where(clinic_valid_mask)[0]
    Ec_pca50 = PCA(n_components=50).fit_transform(
        np.load(str(BASE / 'feats/moment_clinic_raw.npy'))[valid_102.values][all_cidx])
    Eh_pca50 = PCA(n_components=50).fit_transform(
        np.load(str(BASE / 'feats/moment_home_raw.npy'))[valid_102.values][all_cidx])
    Ec_limu = np.load(str(BASE / 'results_raw_pipeline/emb_limubert_clinic.npy'))[valid_102.values][all_cidx]
    Eh_limu = np.load(str(BASE / 'results_raw_pipeline/emb_limubert_home.npy'))[valid_102.values][all_cidx]

    Ec_pca50 = Ec_pca50[both_valid]; Eh_pca50 = Eh_pca50[both_valid]
    Ec_limu = Ec_limu[both_valid]; Eh_limu = Eh_limu[both_valid]

    n = len(y)
    print(f"n={n} subjects\n")

    # ── Define feature blocks ──
    # Home uses Demo(5) with BMI, Clinic uses Demo(4) without BMI
    blocks_home = {
        'Gait': X_gait_h,
        'CWT': X_cwt_h,
        'WalkSway': X_ws_h,
        'Demo': X_demo5,
        'PerBout-Top20': X_perbout,
        'MOMENT': Eh_pca50,
        'LimuBERT': Eh_limu,
    }
    blocks_clinic = {
        'Gait': X_gait_c,
        'CWT': X_cwt_c,
        'WalkSway': X_ws_c,
        'Demo': X_demo4,
        'PerBout-Top20': X_perbout,  # home-only, no clinic equivalent
        'MOMENT': Ec_pca50,
        'LimuBERT': Ec_limu,
    }

    block_names = ['Gait', 'CWT', 'WalkSway', 'Demo', 'PerBout-Top20', 'MOMENT', 'LimuBERT']

    # ── Generate all combinations ──
    results = []

    # Singles
    combos = [(name,) for name in block_names]

    # All pairs
    for combo in itertools.combinations(block_names, 2):
        combos.append(combo)

    # All triples
    for combo in itertools.combinations(block_names, 3):
        combos.append(combo)

    # Key 4+ combos (not all — too many)
    key_combos = [
        ('Gait', 'CWT', 'WalkSway', 'Demo'),
        ('Gait', 'Demo', 'PerBout-Top20'),
        ('Gait', 'CWT', 'Demo', 'PerBout-Top20'),
        ('Gait', 'CWT', 'WalkSway', 'Demo', 'PerBout-Top20'),
        ('Gait', 'Demo', 'MOMENT'),
        ('Gait', 'Demo', 'LimuBERT'),
        ('Gait', 'CWT', 'WalkSway', 'Demo', 'MOMENT'),
        ('Gait', 'CWT', 'WalkSway', 'Demo', 'LimuBERT'),
        ('Gait', 'CWT', 'WalkSway', 'Demo', 'PerBout-Top20', 'MOMENT'),
        ('Gait', 'CWT', 'WalkSway', 'Demo', 'PerBout-Top20', 'LimuBERT'),
        ('Gait', 'CWT', 'WalkSway', 'Demo', 'PerBout-Top20', 'MOMENT', 'LimuBERT'),
    ]
    for combo in key_combos:
        if combo not in combos:
            combos.append(combo)

    print(f"{'Feature Set':<55s} {'#f':>3s}  {'H MAE':>6s} {'H ρ':>6s} {'H R²':>6s} {'H adjR²':>7s}  {'C MAE':>6s} {'C ρ':>6s} {'C R²':>6s} {'C adjR²':>7s}")
    print("-" * 115)

    for combo in combos:
        label = '+'.join(combo)

        # Home
        X_h_parts = [blocks_home[name] for name in combo]
        X_h = np.column_stack(X_h_parts)
        nf = X_h.shape[1]

        # Clinic — PerBout-Top20 is home-only, use '—' for clinic
        has_perbout = 'PerBout-Top20' in combo
        clinic_combo = [name for name in combo if name != 'PerBout-Top20']

        h_r2, h_mae, h_rho, h_adj = best_alpha(X_h, y)[:4]

        if clinic_combo:
            X_c_parts = [blocks_clinic[name] for name in clinic_combo]
            X_c = np.column_stack(X_c_parts)
            c_r2, c_mae, c_rho, c_adj = best_alpha(X_c, y)[:4]
            c_nf = X_c.shape[1]
        else:
            c_r2, c_mae, c_rho, c_adj, c_nf = np.nan, np.nan, np.nan, np.nan, 0

        results.append({
            'Feature Set': label, '#f': nf,
            'Home MAE (ft)': round(h_mae, 1), 'Home Spearman': round(h_rho, 3),
            'Home R²': round(h_r2, 4), 'Home adj R²': round(h_adj, 4),
            'Clinic MAE (ft)': round(c_mae, 1) if np.isfinite(c_mae) else '—',
            'Clinic Spearman': round(c_rho, 3) if np.isfinite(c_rho) else '—',
            'Clinic R²': round(c_r2, 4) if np.isfinite(c_r2) else '—',
            'Clinic adj R²': round(c_adj, 4) if np.isfinite(c_adj) else '—',
        })

        c_r2_str = f"{c_r2:.4f}" if np.isfinite(c_r2) else "     —"
        c_mae_str = f"{c_mae:.0f}" if np.isfinite(c_mae) else "    —"
        c_rho_str = f"{c_rho:.3f}" if np.isfinite(c_rho) else "    —"
        c_adj_str = f"{c_adj:.4f}" if np.isfinite(c_adj) else "     —"

        print(f"  {label:<53s} {nf:>3d}  {h_mae:>6.0f} {h_rho:>6.3f} {h_r2:>6.4f} {h_adj:>7.4f}  {c_mae_str:>6s} {c_rho_str:>6s} {c_r2_str:>6s} {c_adj_str:>7s}")

    # Save
    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT / 'full_results_table.csv', index=False)
    print(f"\nSaved full_results_table.csv ({len(results)} rows)")
    print(f"Done in {time.time()-t0:.0f}s")
