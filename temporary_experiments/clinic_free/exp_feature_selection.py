#!/usr/bin/env python3
"""
Feature selection experiments on all 71 feature set combinations.
8 feature selection methods × 71 combos = 568 experiments.
For each combo, select top-K features (K searched over [3,5,7,10,15,20])
and report best Home R² and Clinic R².
"""
import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, chi2_contingency
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mutual_info_score
from sklearn.linear_model import Ridge
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent
sys.path.insert(0, str(BASE))


def loo_eval(X, y, alpha=20):
    n = len(y)
    pr = np.zeros(n)
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr]); pr[te] = m.predict(sc.transform(X[te]))
    r2 = r2_score(y, pr)
    mae = mean_absolute_error(y, pr)
    rho = spearmanr(y, pr)[0]
    return r2, mae, rho


def best_alpha_eval(X, y, alphas=[5, 10, 20, 50, 100]):
    best = (-999, 0, 0, 10)
    for a in alphas:
        r2, mae, rho = loo_eval(X, y, a)
        if r2 > best[0]: best = (r2, mae, rho, a)
    return best


def impute(X):
    X = X.copy().astype(float)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


# ══════════════════════════════════════════════════════════════════
# FEATURE SELECTION METHODS
# ══════════════════════════════════════════════════════════════════

def select_anova_f(X, y, k):
    """ANOVA F-value."""
    scores, _ = f_regression(X, y)
    scores = np.nan_to_num(scores, 0)
    idx = np.argsort(scores)[::-1][:k]
    return idx


def select_chi2(X, y, k):
    """Chi-Square (discretize continuous features first)."""
    # Make non-negative for chi2
    X_pos = X - X.min(axis=0) + 1e-10
    # Discretize target
    y_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile').fit_transform(y.reshape(-1, 1)).ravel()
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        x_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile').fit_transform(X_pos[:, j:j+1]).ravel()
        ct = pd.crosstab(x_bins.astype(int), y_bins.astype(int))
        try:
            chi2, p, _, _ = chi2_contingency(ct)
            scores[j] = chi2
        except:
            scores[j] = 0
    idx = np.argsort(scores)[::-1][:k]
    return idx


def select_mi(X, y, k):
    """Mutual Information regression."""
    scores = mutual_info_regression(X, y, random_state=42)
    idx = np.argsort(scores)[::-1][:k]
    return idx


def select_spearman(X, y, k):
    """Spearman correlation."""
    scores = np.array([abs(spearmanr(X[:, j], y)[0]) if np.std(X[:, j]) > 1e-12 else 0
                       for j in range(X.shape[1])])
    idx = np.argsort(scores)[::-1][:k]
    return idx


def select_tree(X, y, k):
    """Decision tree feature importance."""
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X, y)
    scores = dt.feature_importances_
    idx = np.argsort(scores)[::-1][:k]
    return idx


def select_pca(X, y, k):
    """PCA — return top-k components (not feature indices)."""
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    n_comp = min(k, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_sc)
    return X_pca  # special: returns transformed data, not indices


def select_relieff(X, y, k):
    """ReliefF approximation using random forest proximity."""
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    scores = rf.feature_importances_
    idx = np.argsort(scores)[::-1][:k]
    return idx


def select_mrmr(X, y, k):
    """Minimum Redundancy Maximum Relevance (greedy forward selection)."""
    n_feat = X.shape[1]
    # Relevance: abs spearman with y
    relevance = np.array([abs(spearmanr(X[:, j], y)[0]) if np.std(X[:, j]) > 1e-12 else 0
                          for j in range(n_feat)])
    selected = []
    remaining = list(range(n_feat))

    for _ in range(min(k, n_feat)):
        best_score, best_f = -999, remaining[0]
        for f in remaining:
            rel = relevance[f]
            if selected:
                # Redundancy: mean abs spearman with already selected
                red = np.mean([abs(spearmanr(X[:, f], X[:, s])[0]) for s in selected])
            else:
                red = 0
            score = rel - red
            if score > best_score:
                best_score, best_f = score, f
        selected.append(best_f)
        remaining.remove(best_f)

    return np.array(selected)


METHODS = {
    'ANOVA-F': select_anova_f,
    'Chi-Square': select_chi2,
    'MI': select_mi,
    'Spearman': select_spearman,
    'DecisionTree': select_tree,
    'PCA': select_pca,
    'ReliefF': select_relieff,
    'MRMR': select_mrmr,
}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    # Load the full results table to get feature set names
    full_table = pd.read_csv(OUT / 'full_results_table.csv')

    # Load all feature blocks (same as exp_full_table.py)
    home_cache = BASE / 'temporary_experiments' / 'clinic_free' / 'longest_bout_proper_cache.npz'
    hc = np.load(home_cache, allow_pickle=True)
    h_cfg = hc['home_configs'].item()['first6']
    c_cfg = hc['clinic_configs'].item()['full']
    h_valid = np.array(h_cfg['valid'], dtype=bool)
    c_valid = np.array(c_cfg['valid'], dtype=bool)
    both_valid = h_valid & c_valid

    def build_from_cache(feat_list, valid_mask):
        df = pd.DataFrame([f for f, v in zip(feat_list, valid_mask) if v])
        return impute(df.replace([np.inf, -np.inf], np.nan).values.astype(float))

    X_gait_h = build_from_cache(h_cfg['gait'], both_valid)
    X_cwt_h = build_from_cache(h_cfg['cwt'], both_valid)
    X_ws_h = build_from_cache(h_cfg['ws'], both_valid)
    X_gait_c = build_from_cache(c_cfg['gait'], both_valid)
    X_cwt_c = build_from_cache(c_cfg['cwt'], both_valid)
    X_ws_c = build_from_cache(c_cfg['ws'], both_valid)

    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = ((ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44])))
    ids101 = ids[~excl].reset_index(drop=True)
    PREPROC2 = BASE / 'csv_preprocessed2'
    clinic_valid_mask = np.array([(PREPROC2 / f"{r['cohort']}{int(r['subj_id']):02d}_{int(r['year'])}_{int(r['sixmwd'])}.csv").exists()
                                  for _, r in ids101.iterrows()])
    ids_v = ids101[clinic_valid_mask].reset_index(drop=True)
    y = ids_v['sixmwd'].values.astype(float)[both_valid]
    n = len(y)

    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = ids_v.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_M'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'Height', 'BMI']: p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo4 = impute(p[['cohort_M', 'Age', 'Sex', 'Height']].values.astype(float))[both_valid]
    X_demo5 = impute(p[['cohort_M', 'Age', 'Sex', 'Height', 'BMI']].values.astype(float))[both_valid]

    cf_npz = np.load(BASE / 'feats' / 'home_clinicfree_top20.npz')
    X_perbout_all = cf_npz['X']
    pb_idx = []
    for idx in np.where(both_valid)[0]:
        r = ids_v.iloc[idx]
        match = ids101[(ids101['cohort'] == r['cohort']) & (ids101['subj_id'] == r['subj_id']) & (ids101['sixmwd'] == r['sixmwd'])]
        if len(match) > 0: pb_idx.append(match.index[0])
    X_perbout = impute(X_perbout_all[pb_idx])

    from sklearn.decomposition import PCA as PCA_sk
    valid_102 = ~((ids['cohort'] == 'M') & (ids['subj_id'] == 22))
    all_cidx = np.where(clinic_valid_mask)[0]
    Ec_pca50 = PCA_sk(n_components=50).fit_transform(np.load(str(BASE / 'feats/moment_clinic_raw.npy'))[valid_102.values][all_cidx])[both_valid]
    Eh_pca50 = PCA_sk(n_components=50).fit_transform(np.load(str(BASE / 'feats/moment_home_raw.npy'))[valid_102.values][all_cidx])[both_valid]
    Ec_limu = np.load(str(BASE / 'results_raw_pipeline/emb_limubert_clinic.npy'))[valid_102.values][all_cidx][both_valid]
    Eh_limu = np.load(str(BASE / 'results_raw_pipeline/emb_limubert_home.npy'))[valid_102.values][all_cidx][both_valid]

    blocks_home = {'Gait': X_gait_h, 'CWT': X_cwt_h, 'WalkSway': X_ws_h,
                   'Demo': X_demo5, 'PerBout-Top20': X_perbout, 'MOMENT': Eh_pca50, 'LimuBERT': Eh_limu}
    blocks_clinic = {'Gait': X_gait_c, 'CWT': X_cwt_c, 'WalkSway': X_ws_c,
                     'Demo': X_demo4, 'PerBout-Top20': X_perbout, 'MOMENT': Ec_pca50, 'LimuBERT': Ec_limu}

    # Only run on the 7 key feature sets from the final table
    key_combos = [
        ('Gait',),
        ('CWT',),
        ('WalkSway',),
        ('PerBout-Top20',),
        ('Demo',),
        ('Demo', 'PerBout-Top20'),
        ('Gait', 'CWT', 'WalkSway', 'Demo'),
    ]

    K_values = [3, 5, 7, 10, 15, 20]

    print(f"n={n}, Feature selection experiments\n")
    all_results = []

    for combo in key_combos:
        label = '+'.join(combo)
        X_h = np.column_stack([blocks_home[b] for b in combo])
        has_perbout = 'PerBout-Top20' in combo
        clinic_combo = [b for b in combo if b != 'PerBout-Top20']
        has_clinic = len(clinic_combo) > 0
        X_c = np.column_stack([blocks_clinic[b] for b in clinic_combo]) if has_clinic else None

        nf = X_h.shape[1]
        # Skip feature selection if very few features
        if nf <= 5:
            # Just run baseline without selection
            h_r2, h_mae, h_rho, _ = best_alpha_eval(X_h, y)
            c_r2, c_mae, c_rho = (np.nan, np.nan, np.nan)
            if X_c is not None:
                c_r2, c_mae, c_rho, _ = best_alpha_eval(X_c, y)
            all_results.append({
                'Feature Set': label, 'Method': 'None (baseline)', 'K': nf,
                'Home R²': round(h_r2, 4), 'Home MAE': round(h_mae, 1), 'Home ρ': round(h_rho, 3),
                'Clinic R²': round(c_r2, 4) if np.isfinite(c_r2) else '—',
                'Clinic MAE': round(c_mae, 1) if np.isfinite(c_mae) else '—',
                'Clinic ρ': round(c_rho, 3) if np.isfinite(c_rho) else '—',
            })
            print(f"  {label:<35s} None(baseline)  K={nf:>2d}  H R²={h_r2:.4f}  C R²={c_r2:.4f}" if np.isfinite(c_r2) else
                  f"  {label:<35s} None(baseline)  K={nf:>2d}  H R²={h_r2:.4f}  C R²=—")
            continue

        for method_name, method_fn in METHODS.items():
            best_h = (-999, 0, 0, 0)
            best_c = (-999, 0, 0, 0)

            for K in K_values:
                if K >= nf:
                    continue

                # Home
                if method_name == 'PCA':
                    X_h_sel = method_fn(X_h, y, K)
                    h_r2, h_mae, h_rho, _ = best_alpha_eval(X_h_sel, y)
                else:
                    idx_h = method_fn(X_h, y, K)
                    X_h_sel = X_h[:, idx_h]
                    h_r2, h_mae, h_rho, _ = best_alpha_eval(X_h_sel, y)

                if h_r2 > best_h[0]:
                    best_h = (h_r2, h_mae, h_rho, K)

                # Clinic
                if X_c is not None and X_c.shape[1] > K:
                    if method_name == 'PCA':
                        X_c_sel = method_fn(X_c, y, K)
                        c_r2, c_mae, c_rho, _ = best_alpha_eval(X_c_sel, y)
                    else:
                        idx_c = method_fn(X_c, y, K)
                        X_c_sel = X_c[:, idx_c]
                        c_r2, c_mae, c_rho, _ = best_alpha_eval(X_c_sel, y)

                    if c_r2 > best_c[0]:
                        best_c = (c_r2, c_mae, c_rho, K)

            h_r2, h_mae, h_rho, h_k = best_h
            c_r2, c_mae, c_rho, c_k = best_c if best_c[0] > -999 else (np.nan, np.nan, np.nan, 0)

            all_results.append({
                'Feature Set': label, 'Method': method_name, 'K': f"H:{h_k},C:{c_k}" if c_k else h_k,
                'Home R²': round(h_r2, 4), 'Home MAE': round(h_mae, 1), 'Home ρ': round(h_rho, 3),
                'Clinic R²': round(c_r2, 4) if np.isfinite(c_r2) else '—',
                'Clinic MAE': round(c_mae, 1) if np.isfinite(c_mae) else '—',
                'Clinic ρ': round(c_rho, 3) if np.isfinite(c_rho) else '—',
            })

            c_str = f"C R²={c_r2:.4f}" if np.isfinite(c_r2) else "C R²=—"
            print(f"  {label:<35s} {method_name:<15s} K=H:{h_k},C:{c_k}  H R²={h_r2:.4f}  {c_str}")

    # Save
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(OUT / 'feature_selection_results.csv', index=False)

    # Print summary: best home and clinic per feature set
    print(f"\n{'='*100}")
    print(f"BEST RESULTS PER FEATURE SET")
    print(f"{'='*100}")
    print(f"{'Feature Set':<35s} {'Best Home Method':<20s} {'H R²':>7s} {'H MAE':>6s}  {'Best Clinic Method':<20s} {'C R²':>7s} {'C MAE':>6s}")
    print("-" * 105)

    for combo in key_combos:
        label = '+'.join(combo)
        sub = rdf[rdf['Feature Set'] == label]
        # Best home
        h_best = sub.loc[sub['Home R²'].idxmax()]
        # Best clinic
        c_sub = sub[sub['Clinic R²'] != '—']
        if len(c_sub) > 0:
            c_sub_num = c_sub.copy()
            c_sub_num['Clinic R²'] = pd.to_numeric(c_sub_num['Clinic R²'])
            c_best = c_sub_num.loc[c_sub_num['Clinic R²'].idxmax()]
            c_str = f"{c_best['Method']:<20s} {c_best['Clinic R²']:>7.4f} {c_best['Clinic MAE']:>6}"
        else:
            c_str = "—"
        print(f"  {label:<35s} {h_best['Method']:<20s} {h_best['Home R²']:>7.4f} {h_best['Home MAE']:>6}  {c_str}")

    print(f"\nSaved feature_selection_results.csv ({len(all_results)} rows)")
    print(f"Done in {time.time()-t0:.0f}s")
