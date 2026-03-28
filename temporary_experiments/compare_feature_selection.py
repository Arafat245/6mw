#!/usr/bin/env python3
"""Compare 8 feature selection methods inside LOO (no leakage)."""
import numpy as np, pandas as pd, warnings, time
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, chi2_contingency
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
NPZ_DIR = BASE / 'csv_home_daytime_npz'
subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
y = subj_df['sixmwd'].values.astype(float)
n = len(y)

orig_df = pd.read_csv(BASE / 'feats' / 'home_clinicfree_features.csv')
orig_cols = [c for c in orig_df.columns if c != 'key']
demo_data = pd.read_excel(BASE / 'Accel files' / 'PedMSWalkStudy_Demographic.xlsx')
demo_data['cohort'] = demo_data['ID'].str.extract(r'^([A-Z])')[0]
demo_data['subj_id'] = demo_data['ID'].str.extract(r'(\d+)')[0].astype(int)
p = subj_df.merge(demo_data, on=['cohort', 'subj_id'], how='left')
p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
for c in ['Age', 'Sex', 'Height', 'BMI']:
    p[c] = pd.to_numeric(p[c], errors='coerce')
demo_cols = ['cohort_POMS', 'Age', 'Sex', 'Height', 'BMI']

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X

X_orig = impute(orig_df[orig_cols].values.astype(float))
X_demo = impute(p[demo_cols].values.astype(float))
X_all = np.column_stack([X_orig, X_demo])
all_names = orig_cols + demo_cols
n_accel = len(orig_cols)
demo_idx = list(range(n_accel, n_accel + len(demo_cols)))

print(f"X={X_all.shape}, n={n}, accel={n_accel}, demo={len(demo_cols)}")


def select_spearman(X_tr, y_tr, K):
    scores = [abs(spearmanr(X_tr[:, j], y_tr)[0]) if np.std(X_tr[:, j]) > 0 else 0 for j in range(n_accel)]
    return sorted(range(n_accel), key=lambda j: scores[j], reverse=True)[:K]


def select_pearson(X_tr, y_tr, K):
    scores = [abs(pearsonr(X_tr[:, j], y_tr)[0]) if np.std(X_tr[:, j]) > 0 else 0 for j in range(n_accel)]
    return sorted(range(n_accel), key=lambda j: scores[j], reverse=True)[:K]


def select_anova_f(X_tr, y_tr, K):
    F, _ = f_regression(X_tr[:, :n_accel], y_tr)
    F = np.nan_to_num(F, nan=0)
    return sorted(range(n_accel), key=lambda j: F[j], reverse=True)[:K]


def select_mutual_info(X_tr, y_tr, K):
    mi = mutual_info_regression(X_tr[:, :n_accel], y_tr, random_state=42)
    return sorted(range(n_accel), key=lambda j: mi[j], reverse=True)[:K]


def select_chi2(X_tr, y_tr, K):
    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    y_binned = kbd.fit_transform(y_tr.reshape(-1, 1)).ravel()
    scores = []
    for j in range(n_accel):
        col = X_tr[:, j]
        if np.std(col) == 0:
            scores.append(0)
            continue
        try:
            col_binned = kbd.fit_transform(col.reshape(-1, 1)).ravel()
            ct = pd.crosstab(col_binned.astype(int), y_binned.astype(int))
            chi2, pval, _, _ = chi2_contingency(ct)
            scores.append(chi2)
        except:
            scores.append(0)
    return sorted(range(n_accel), key=lambda j: scores[j], reverse=True)[:K]


def select_decision_tree(X_tr, y_tr, K):
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_tr[:, :n_accel], y_tr)
    imp = dt.feature_importances_
    return sorted(range(n_accel), key=lambda j: imp[j], reverse=True)[:K]


def select_relieff(X_tr, y_tr, K):
    n_tr = len(y_tr)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_tr[:, :n_accel])
    weights = np.zeros(n_accel)
    n_iter = min(n_tr, 50)
    rng = np.random.RandomState(42)
    for _ in range(n_iter):
        i = rng.randint(n_tr)
        dists = np.sum((X_sc - X_sc[i])**2, axis=1)
        dists[i] = np.inf
        y_diff = np.abs(y_tr - y_tr[i])
        median_diff = np.median(y_diff[y_diff > 0])
        hits = np.where(y_diff <= median_diff)[0]
        misses = np.where(y_diff > median_diff)[0]
        if len(hits) == 0 or len(misses) == 0:
            continue
        nearest_hit = hits[np.argmin(dists[hits])]
        nearest_miss = misses[np.argmin(dists[misses])]
        weights += (X_sc[i] - X_sc[nearest_miss])**2 - (X_sc[i] - X_sc[nearest_hit])**2
    return sorted(range(n_accel), key=lambda j: weights[j], reverse=True)[:K]


def select_mrmr(X_tr, y_tr, K):
    mi_y = mutual_info_regression(X_tr[:, :n_accel], y_tr, random_state=42)
    selected = []
    remaining = list(range(n_accel))
    for _ in range(K):
        best_score, best_idx = -np.inf, -1
        for j in remaining:
            relevance = mi_y[j]
            if len(selected) > 0:
                redundancy = np.mean([
                    mutual_info_regression(X_tr[:, j].reshape(-1, 1), X_tr[:, s], random_state=42)[0]
                    for s in selected
                ])
            else:
                redundancy = 0
            score = relevance - redundancy
            if score > best_score:
                best_score = score
                best_idx = j
        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)
    return selected


K = 20
alpha = 20
print(f"\nK={K} accel features + 5 demo, Ridge alpha={alpha}, LOO CV (inside LOO, no leakage)")
print(f"{'Method':<16s}  {'R2':>8s}  {'MAE':>6s}  {'rho':>7s}  {'time':>5s}")
print("-" * 52)

methods = {
    'Spearman': select_spearman,
    'Pearson': select_pearson,
    'ANOVA-F': select_anova_f,
    'Mutual Info': select_mutual_info,
    'Chi-Square': select_chi2,
    'Decision Tree': select_decision_tree,
    'ReliefF': select_relieff,
    'mRMR': select_mrmr,
}

for name, select_fn in methods.items():
    t0 = time.time()
    preds = np.zeros(n)
    for tr, te in LeaveOneOut().split(X_all):
        selected = select_fn(X_all[tr], y[tr], K) + demo_idx
        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_all[tr][:, selected]), y[tr])
        preds[te] = m.predict(sc.transform(X_all[te][:, selected]))
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rho = spearmanr(y, preds)[0]
    elapsed = time.time() - t0
    print(f"{name:<16s}  R2={r2:.4f}  {mae:.0f}  {rho:.3f}  {elapsed:.0f}s", flush=True)

# PCA separately (no feature selection, dimensionality reduction)
print()
for K_pca in [5, 10, 15, 20, 30]:
    t0 = time.time()
    preds = np.zeros(n)
    for tr, te in LeaveOneOut().split(X_all):
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_all[tr])
        X_te_sc = sc.transform(X_all[te])
        pca = PCA(n_components=K_pca)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_te_pca = pca.transform(X_te_sc)
        m = Ridge(alpha=alpha)
        m.fit(X_tr_pca, y[tr])
        preds[te] = m.predict(X_te_pca)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rho = spearmanr(y, preds)[0]
    elapsed = time.time() - t0
    print(f"PCA (K={K_pca:2d})       R2={r2:.4f}  {mae:.0f}  {rho:.3f}  {elapsed:.0f}s", flush=True)
