#!/usr/bin/env python3
"""
Generate all paper figures from cached features and data.

Output: results/paper_figures/*.png + *.svg  (7 figures)
Run:    python analysis/generate_paper_figures.py  (~1 min)
"""
import warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
FEATS = BASE / 'feats'
OUT = BASE / 'results' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)
POMS_FIGURES = BASE / 'POMS' / 'figures'  # paper-side copies (LaTeX-adjacent)
POMS_FIGURES.mkdir(parents=True, exist_ok=True)
FT2M = 0.3048


def save_paper_figure(fig, filename, dpi=200):
    """Write to both results/paper_figures/ and POMS/figures/ so the paper-side stays in sync."""
    fig.savefig(OUT / filename, dpi=dpi, bbox_inches='tight')
    fig.savefig(POMS_FIGURES / filename, dpi=dpi, bbox_inches='tight')

CLR_HEALTHY = '#6BAED6'
CLR_POMS = '#FC8D62'


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def loo_ridge_preds(X, y, alpha):
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pr[te] = m.predict(sc.transform(X[te]))
    return pr


def loo_spearman_preds(X_accel, X_demo, y, K=20, alpha=20):
    n_accel = X_accel.shape[1]
    pr = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), dtype=bool); tr[i] = False
        corrs = [abs(spearmanr(X_accel[tr, j], y[tr])[0])
                 if np.std(X_accel[tr, j]) > 0 else 0 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:min(K, n_accel)]
        if X_demo is not None:
            X_tr = np.column_stack([X_accel[tr][:, top_k], X_demo[tr]])
            X_te = np.column_stack([X_accel[i:i+1][:, top_k], X_demo[i:i+1]])
        else:
            X_tr = X_accel[tr][:, top_k]
            X_te = X_accel[i:i+1][:, top_k]
        sc = StandardScaler(); m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_tr), y[tr])
        pr[i] = m.predict(sc.transform(X_te))[0]
    return pr


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
t0 = time.time()
print("Loading data...")

subj_df = pd.read_csv(BASE / 'home_full_recording_npz' / '_subjects.csv')
y_ft = subj_df['sixmwd'].values.astype(float)
y_m = y_ft * FT2M
n = len(subj_df)
cohorts = subj_df['cohort'].values
is_poms = cohorts == 'M'
is_healthy = cohorts == 'C'

demo_xl = pd.read_excel(BASE / 'SwayDemographics.xlsx')
demo_xl['cohort'] = demo_xl['ID'].str.extract(r'^([A-Z])')[0]
demo_xl['subj_id'] = demo_xl['ID'].str.extract(r'(\d+)')[0].astype(int)
p = subj_df.merge(demo_xl, on=['cohort', 'subj_id'], how='left')
for c in ['Age', 'Sex', 'Height', 'Weight', 'BMI', 'EDSS Total', 'MS Dur',
          'MFIS Total', 'MFIS Phys', 'MFIS Cog', 'MFIS Psych', 'BDI Raw Score']:
    p[c] = pd.to_numeric(p[c], errors='coerce')
p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)

X_demo_bmi = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))
X_demo_height = impute(p[['cohort_POMS', 'Age', 'Sex', 'Height']].values.astype(float))

c_gait_df = pd.read_csv(FEATS / 'clinic_gait_features.csv')
c_cwt_df = pd.read_csv(FEATS / 'clinic_cwt_features.csv')
c_ws_df = pd.read_csv(FEATS / 'clinic_walksway_features.csv')
c_gait = impute(c_gait_df.drop(columns='key').values.astype(float))
c_cwt = impute(c_cwt_df.drop(columns='key').values.astype(float))
c_ws = impute(c_ws_df.drop(columns='key').values.astype(float))
c_gait_cols = list(c_gait_df.columns[1:])
c_cwt_cols = list(c_cwt_df.columns[1:])
c_ws_cols = list(c_ws_df.columns[1:])

h_pb_df = pd.read_csv(FEATS / 'home_perbout_features.csv')
h_pb_cols = list(h_pb_df.columns[1:])
h_pb = impute(h_pb_df.drop(columns='key').values.astype(float))

h_gait_df = pd.read_csv(FEATS / 'home_gait_features.csv')
h_cwt_df = pd.read_csv(FEATS / 'home_cwt_features.csv')
h_ws_df = pd.read_csv(FEATS / 'home_walksway_features.csv')
h_gait_med_cols = [c for c in h_gait_df.columns if c.endswith('_med')]
h_cwt_med_cols = [c for c in h_cwt_df.columns if c.endswith('_med')]
h_ws_med_cols = [c for c in h_ws_df.columns if c.endswith('_med')]
h_gait_med = impute(h_gait_df[h_gait_med_cols].values.astype(float))
h_cwt_med = impute(h_cwt_df[h_cwt_med_cols].values.astype(float))
h_ws_med = impute(h_ws_df[h_ws_med_cols].values.astype(float))


# ══════════════════════════════════════════════════════════════════
# COMPUTE LOO PREDICTIONS
# ══════════════════════════════════════════════════════════════════
print("Computing LOO predictions...")

X_clinic_all = np.column_stack([c_gait, c_cwt, c_ws, X_demo_height])
pred_clinic_ft = loo_ridge_preds(X_clinic_all, y_ft, alpha=5)
pred_home_ft = loo_spearman_preds(h_pb, X_demo_bmi, y_ft, K=20, alpha=20)

pred_clinic_m = pred_clinic_ft * FT2M
pred_home_m = pred_home_ft * FT2M

c_r2 = r2_score(y_m, pred_clinic_m)
c_mae = mean_absolute_error(y_m, pred_clinic_m)
c_rho = spearmanr(y_m, pred_clinic_m)[0]
h_r2 = r2_score(y_m, pred_home_m)
h_mae = mean_absolute_error(y_m, pred_home_m)
h_rho = spearmanr(y_m, pred_home_m)[0]
print(f"  Clinic R2={c_r2:.3f} MAE={c_mae:.1f}m  Home R2={h_r2:.3f} MAE={h_mae:.1f}m")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5: Predicted vs Actual
# ══════════════════════════════════════════════════════════════════
print("\nFig 5: fig_predicted_vs_actual.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, pred_m, title, r2, mae, rho in [
    (ax1, pred_home_m, 'Home: Bout+Act-Top20+Demo(4)', h_r2, h_mae, h_rho),
    (ax2, pred_clinic_m, 'Clinic: Gait+CWT+WalkSway+Demo', c_r2, c_mae, c_rho),
]:
    # Per-cohort R2
    r2_h = r2_score(y_m[is_healthy], pred_m[is_healthy])
    r2_p = r2_score(y_m[is_poms], pred_m[is_poms])

    ax.scatter(y_m[is_healthy], pred_m[is_healthy], c=CLR_HEALTHY, s=25, alpha=0.7,
               label=f'Healthy (n={is_healthy.sum()}, R\u00b2={r2_h:.3f})', zorder=3)
    ax.scatter(y_m[is_poms], pred_m[is_poms], c=CLR_POMS, s=25, alpha=0.7,
               label=f'POMS (n={is_poms.sum()}, R\u00b2={r2_p:.3f})', zorder=3)

    # y=x line
    lo = min(y_m.min(), pred_m.min()) * 0.95
    hi = max(y_m.max(), pred_m.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1, alpha=0.6)

    # OLS fit
    slope, intercept = np.polyfit(y_m, pred_m, 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, slope * xs + intercept, '-', color='black', lw=1.2)

    ax.set_xlabel('Actual 6MWD (m)')
    ax.set_ylabel('Predicted 6MWD (m)')
    ax.set_title(f'{title}\nR\u00b2={r2:.3f}, MAE={mae:.1f} m, \u03c1={rho:.3f}', fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

plt.tight_layout()
save_paper_figure(fig, 'fig_predicted_vs_actual.png')
plt.close()
print(f"  Saved fig_predicted_vs_actual.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6: Bland-Altman
# ══════════════════════════════════════════════════════════════════
print("\nFig 6: fig_bland_altman.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, pred_m, title in [
    (ax1, pred_home_m, 'Home: Bout+Act-Top20+Demo(4)'),
    (ax2, pred_clinic_m, 'Clinic: Gait+CWT+WalkSway+Demo'),
]:
    mean_vals = (y_m + pred_m) / 2
    diff_vals = y_m - pred_m
    bias = diff_vals.mean()
    sd = diff_vals.std()

    ax.scatter(mean_vals[is_healthy], diff_vals[is_healthy], c=CLR_HEALTHY,
               s=25, alpha=0.7, label='Healthy', zorder=3)
    ax.scatter(mean_vals[is_poms], diff_vals[is_poms], c=CLR_POMS,
               s=25, alpha=0.7, label='POMS', zorder=3)

    lo, hi = mean_vals.min() * 0.95, mean_vals.max() * 1.05
    ax.axhline(bias, color='black', lw=1.2, label=f'Bias: {bias:+.1f} m')
    ax.axhline(bias + 1.96 * sd, color='gray', ls='--', lw=1,
               label=f'+1.96 SD: {bias + 1.96 * sd:+.1f} m')
    ax.axhline(bias - 1.96 * sd, color='gray', ls='--', lw=1,
               label=f'\u22121.96 SD: {bias - 1.96 * sd:+.1f} m')
    ax.axhline(0, color='lightgray', lw=0.5)

    ax.set_xlabel('Mean of Actual & Predicted (m)')
    ax.set_ylabel('Actual \u2212 Predicted (m)')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
save_paper_figure(fig, 'fig_bland_altman.png')
plt.close()
print(f"  Saved fig_bland_altman.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 7: SHAP Feature Importance (linear model)
# ══════════════════════════════════════════════════════════════════
print("\nFig 7: fig_shap_importance.png")


def linear_shap(X, y, alpha, feature_names, max_show=10):
    """Compute SHAP values for Ridge (exact for linear models)."""
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    m = Ridge(alpha=alpha).fit(X_s, y)
    # SHAP for linear: coef * (x_scaled)
    shap_vals = X_s * m.coef_[np.newaxis, :]
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(mean_abs)[-max_show:][::-1]
    return shap_vals, X, feature_names, top_idx


def plot_shap_beeswarm(ax, shap_vals, feat_vals, feat_names, top_idx, title):
    """SHAP beeswarm for selected features."""
    n_show = len(top_idx)
    for pos, idx in enumerate(reversed(top_idx)):
        sv = shap_vals[:, idx]
        fv = feat_vals[:, idx]
        fmin, fmax = np.nanmin(fv), np.nanmax(fv)
        if fmax > fmin:
            fn = (fv - fmin) / (fmax - fmin)
        else:
            fn = np.full_like(fv, 0.5)
        np.random.seed(idx)
        jitter = np.random.normal(0, 0.12, len(sv))
        ax.scatter(sv, np.full_like(sv, pos) + jitter, c=fn,
                   cmap='coolwarm', s=6, alpha=0.6, edgecolors='none',
                   vmin=0, vmax=1)

    ax.set_yticks(range(n_show))
    ax.set_yticklabels([feat_names[i].replace('_', ' ') for i in reversed(top_idx)],
                        fontsize=8)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) Clinic: Gait+CWT+WalkSway (51 features, no demo for interpretability)
X_clinic_gcw = np.column_stack([c_gait, c_cwt, c_ws])
clinic_gcw_names = c_gait_cols + c_cwt_cols + c_ws_cols
shap_c, feat_c, names_c, top_c = linear_shap(X_clinic_gcw, y_ft, alpha=5,
                                               feature_names=clinic_gcw_names)
plot_shap_beeswarm(ax1, shap_c, feat_c, names_c, top_c,
                    '(a) Clinic: Gait+CWT+WalkSway')

# (b) Home: PerBout Top-20 (use all-data Spearman to pick representative features)
corrs = [abs(spearmanr(h_pb[:, j], y_ft)[0]) if np.std(h_pb[:, j]) > 0 else 0
         for j in range(h_pb.shape[1])]
top20_idx = sorted(range(len(corrs)), key=lambda j: corrs[j], reverse=True)[:20]
X_home_top20 = h_pb[:, top20_idx]
home_top20_names = [h_pb_cols[j] for j in top20_idx]
shap_h, feat_h, names_h, top_h = linear_shap(X_home_top20, y_ft, alpha=20,
                                               feature_names=home_top20_names)
plot_shap_beeswarm(ax2, shap_h, feat_h, names_h, top_h,
                    '(b) Home: PerBout Top-20')

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.8, aspect=30, pad=0.02)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Low', '', 'High'])
cbar.set_label('Feature value', fontsize=9)

plt.tight_layout()
save_paper_figure(fig, 'fig_shap_importance.png')
plt.close()
print(f"  Saved fig_shap_importance.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Feature-6MWD Correlation Heatmap
# ══════════════════════════════════════════════════════════════════
print("\nFig 2: heatmap_feature_6mwd_corr.png")

# Build clinic and home feature arrays with matching names
all_clinic_X = np.column_stack([c_gait, c_cwt, c_ws])
all_clinic_names = c_gait_cols + c_cwt_cols + c_ws_cols
all_clinic_cats = ['Gait'] * len(c_gait_cols) + ['CWT'] * len(c_cwt_cols) + ['WalkSway'] * len(c_ws_cols)

all_home_X = np.column_stack([h_gait_med, h_cwt_med, h_ws_med])
h_gait_base = [c.rsplit('_med', 1)[0] for c in h_gait_med_cols]
h_cwt_base = [c.rsplit('_med', 1)[0] for c in h_cwt_med_cols]
h_ws_base = [c.rsplit('_med', 1)[0] for c in h_ws_med_cols]
all_home_names = h_gait_base + h_cwt_base + h_ws_base

common = sorted(set(all_clinic_names) & set(all_home_names))

columns = ['Home\n(All)', 'Clinic\n(All)', 'Home\n(POMS)', 'Home\n(Healthy)',
           'Clinic\n(POMS)', 'Clinic\n(Healthy)']

heat_data = []
heat_labels = []
heat_annot = []

age = p['Age'].values.astype(float)

for feat in common + ['_age_']:
    if feat == '_age_':
        cat = 'Demo'
        display = 'Demo: Age'
    else:
        c_idx = all_clinic_names.index(feat)
        cat = all_clinic_cats[c_idx]
        display = f"{cat}: {feat.replace('_', ' ')}"

    row_rho = []
    row_annot = []
    for mask_label, mask in [('all', np.ones(n, bool)), ('all', np.ones(n, bool)),
                              ('poms', is_poms), ('healthy', is_healthy),
                              ('poms', is_poms), ('healthy', is_healthy)]:
        if feat == '_age_':
            vals = age[mask]
        else:
            is_home = columns[len(row_rho)].startswith('Home')
            if is_home:
                h_idx = all_home_names.index(feat)
                vals = all_home_X[mask, h_idx]
            else:
                vals = all_clinic_X[mask, c_idx]

        rho, pval = spearmanr(vals, y_ft[mask])
        row_rho.append(rho)
        row_annot.append(f"{rho:+.2f}{sig_stars(pval)}")

    if max(abs(r) for r in row_rho) > 0.3:
        heat_data.append(row_rho)
        heat_labels.append(display)
        heat_annot.append(row_annot)

# Sort by max |rho|
order = sorted(range(len(heat_data)), key=lambda i: max(abs(r) for r in heat_data[i]), reverse=True)
heat_data = [heat_data[i] for i in order]
heat_labels = [heat_labels[i] for i in order]
heat_annot = [heat_annot[i] for i in order]

data_arr = np.array(heat_data)
n_feat = len(heat_labels)

fig, ax = plt.subplots(figsize=(8, max(4, n_feat * 0.4 + 1)))
im = ax.imshow(data_arr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(6))
ax.set_xticklabels(columns, fontsize=9)
ax.set_yticks(range(n_feat))
ax.set_yticklabels(heat_labels, fontsize=8)

for i in range(n_feat):
    for j in range(6):
        ax.text(j, i, heat_annot[i][j], ha='center', va='center', fontsize=7,
                color='white' if abs(data_arr[i, j]) > 0.6 else 'black')

ax.set_title(f'Feature Correlations with 6MWD (|\u03c1| > 0.3, n={n}, '
             f'POMS={is_poms.sum()}, Healthy={is_healthy.sum()})', fontsize=10)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Spearman \u03c1', fontsize=9)

plt.tight_layout()
save_paper_figure(fig, 'heatmap_feature_6mwd_corr.png')
plt.close()
print(f"  Saved heatmap_feature_6mwd_corr.png ({n_feat} features)")


# ══════════════════════════════════════════════════════════════════
# FIGURES 3 & 4: Clinical Score Correlation Heatmaps
# ══════════════════════════════════════════════════════════════════
print("\nFig 3: heatmap_clinical_corr_clinic.png")
print("Fig 4: heatmap_clinical_corr_home.png")

clinical_scores = {
    'EDSS': p.loc[is_poms, 'EDSS Total'].values.astype(float),
    'MFIS Total': p.loc[is_poms, 'MFIS Total'].values.astype(float),
    'MFIS Phys': p.loc[is_poms, 'MFIS Phys'].values.astype(float),
    'MFIS Cog': p.loc[is_poms, 'MFIS Cog'].values.astype(float),
    'MFIS Psych': p.loc[is_poms, 'MFIS Psych'].values.astype(float),
    'BDI': p.loc[is_poms, 'BDI Raw Score'].values.astype(float),
    'POMS Dur': p.loc[is_poms, 'MS Dur'].values.astype(float),
}


def clinical_heatmap(X, feat_names, feat_cats, score_names, title, fname, mask):
    """Generate clinical correlation heatmap."""
    X_ms = X[mask]
    rows_data, rows_annot, rows_labels = [], [], []

    for j, (feat, cat) in enumerate(zip(feat_names, feat_cats)):
        rhos, annots = [], []
        has_sig = False
        for sn in score_names:
            scores = clinical_scores[sn]
            valid = ~(np.isnan(X_ms[:, j]) | np.isnan(scores))
            if valid.sum() < 5:
                rhos.append(0); annots.append('---')
                continue
            rho, pval = spearmanr(X_ms[valid, j], scores[valid])
            rhos.append(rho)
            annots.append(f"{rho:+.2f}{sig_stars(pval)}")
            if pval < 0.05:
                has_sig = True
        if has_sig:
            rows_data.append(rhos)
            rows_annot.append(annots)
            rows_labels.append(f"{cat}: {feat.replace('_', ' ')}")

    if not rows_data:
        print(f"  No significant features for {fname}")
        return

    # Sort by min p-value (most significant first)
    data_arr = np.array(rows_data)
    n_feat = len(rows_labels)
    n_scores = len(score_names)

    fig, ax = plt.subplots(figsize=(max(6, n_scores * 1.2), max(3, n_feat * 0.4 + 1)))
    vmax = max(0.3, min(0.8, np.max(np.abs(data_arr)) + 0.1))
    im = ax.imshow(data_arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(n_scores))
    ax.set_xticklabels(score_names, fontsize=9)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(rows_labels, fontsize=8)

    for i in range(n_feat):
        for j in range(n_scores):
            ax.text(j, i, rows_annot[i][j], ha='center', va='center', fontsize=7,
                    color='white' if abs(data_arr[i, j]) > vmax * 0.75 else 'black')

    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman \u03c1', fontsize=9)

    plt.tight_layout()
    save_paper_figure(fig, fname)
    plt.close()
    print(f"  Saved {fname} ({n_feat} features)")


# Clinic features
clinic_all_X = np.column_stack([c_gait, c_cwt, c_ws])
clinic_all_names = c_gait_cols + c_cwt_cols + c_ws_cols
clinic_all_cats = ['Gait'] * len(c_gait_cols) + ['CWT'] * len(c_cwt_cols) + ['WalkSway'] * len(c_ws_cols)

clinical_heatmap(clinic_all_X, clinic_all_names, clinic_all_cats,
                 ['EDSS', 'MFIS Total', 'MFIS Phys', 'MFIS Cog', 'MFIS Psych', 'BDI', 'POMS Dur'],
                 f'Clinic Wearable Feature - Clinical Score Correlations (POMS only, n={is_poms.sum()})',
                 'heatmap_clinical_corr_clinic.png', is_poms)

# Home features: Bout + Act from the headline home_perbout_features.csv pool
# (matches the 6MWD heatmap and ms_vs_healthy table conventions)
h_bout_cols = [c for c in h_pb_cols if c.startswith('g_')]
h_act_cols  = [c for c in h_pb_cols if c.startswith('act_')]
h_bout_idx  = [h_pb_cols.index(c) for c in h_bout_cols]
h_act_idx   = [h_pb_cols.index(c) for c in h_act_cols]
home_all_X    = np.column_stack([h_pb[:, h_bout_idx], h_pb[:, h_act_idx]])
home_all_names = h_bout_cols + h_act_cols
home_all_cats  = ['Bout'] * len(h_bout_cols) + ['Act'] * len(h_act_cols)

clinical_heatmap(home_all_X, home_all_names, home_all_cats,
                 ['EDSS', 'MFIS Total', 'MFIS Phys', 'MFIS Cog', 'MFIS Psych', 'BDI'],
                 f'Home Wearable Feature - Clinical Score Correlations (POMS only, n={is_poms.sum()})',
                 'heatmap_clinical_corr_home.png', is_poms)


print(f"\nAll 6 figures saved to {OUT}/")
print(f"Done in {time.time() - t0:.0f}s")
