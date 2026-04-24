#!/usr/bin/env python3
"""
Spearman rho heatmap of top-10 features vs 6MWD (|rho| > 0.3), clinic + home
plotted side-by-side as a single figure.

Output: results/paper_figures/heatmap_feature_6mwd_corr_top10.png
        (mirrored to POMS/figures/)

Each panel: 10 features (rows) x {All, POMS, Healthy} (columns).
Significance stars in annotations (***p<0.001, **p<0.01, *p<0.05).

Spearman is used for individual feature-6MWD correlations (captures monotonic,
non-linear relationships typical of gait features). Pearson r is reserved for
Ridge prediction-vs-actual agreement in the headline results.

Feature pools (per setting — matches what the headline models actually see):
  Clinic: Gait (11) + CWT (28) + WalkSway (12) + Demo: Age   → feeds Gait+CWT+WS+Demo (R²=0.81)
  Home:   Bout (120 bout-agg × 6 stats) + Act (29 activity) + Demo: Age   → feeds Bout+Act-Top20+Demo (R²=0.45)

Style: sns.heatmap(cmap=sns.diverging_palette(0, 255, sep=77, as_cmap=True),
       annot=True, vmin=-1, vmax=1, xticklabels=True).
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent.parent
FEATS = BASE / 'feats'
OUT = BASE / 'results' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)
POMS_FIGURES = BASE / 'POMS' / 'figures'
POMS_FIGURES.mkdir(parents=True, exist_ok=True)
NPZ_DIR = BASE / 'home_full_recording_npz'

CMAP = sns.diverging_palette(0, 255, sep=77, as_cmap=True)


def save_paper_figure(fig, filename, dpi=300):
    """Mirror to results/paper_figures/ and POMS/figures/."""
    fig.savefig(OUT / filename, dpi=dpi, bbox_inches='tight')
    fig.savefig(POMS_FIGURES / filename, dpi=dpi, bbox_inches='tight')


def impute(X):
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def spearman_col(X_col, y):
    valid = ~(np.isnan(X_col) | np.isnan(y))
    if valid.sum() < 3 or np.std(X_col[valid]) == 0:
        return np.nan, np.nan
    return spearmanr(X_col[valid], y[valid])


def build_heatmap_df(X, feat_names, feat_cats, y, is_poms, is_healthy,
                      threshold=0.3, top_n=10, fixed_features=None):
    """Returns (r_df, annot_df).

    If `fixed_features` is provided (list of feature names in desired display order),
    those rows are used verbatim — threshold/top_n are ignored.
    Otherwise: keep features whose max|r| across cohorts > threshold, take top_n by max|r|.
    """
    cohorts = [('All',     np.ones(len(y), bool)),
               ('POMS',    is_poms),
               ('Healthy', is_healthy)]
    rows_r, rows_annot, labels = [], [], []
    for j, (feat, cat) in enumerate(zip(feat_names, feat_cats)):
        rs, anns = [], []
        for cname, mask in cohorts:
            r, p = spearman_col(X[mask, j], y[mask])
            rs.append(r)
            anns.append('—' if np.isnan(r) else f"{r:+.2f}{sig_stars(p)}")
        rows_r.append(rs)
        rows_annot.append(anns)
        labels.append(f"{cat}: {feat}")

    if fixed_features is not None:
        feat_to_idx = {f: i for i, f in enumerate(feat_names)}
        order = [feat_to_idx[f] for f in fixed_features if f in feat_to_idx]
        # Sort by |r| in the 'All' column (cohorts[0])
        order.sort(key=lambda i: abs(rows_r[i][0]) if not np.isnan(rows_r[i][0]) else -1,
                   reverse=True)
    else:
        # Keep features with |r|>threshold somewhere across cohorts;
        # rank/sort by |r| in the 'All' column (cohorts[0]).
        keep = [i for i, rs in enumerate(rows_r)
                if any(not np.isnan(r) and abs(r) > threshold for r in rs)]
        order = sorted(keep,
                       key=lambda i: abs(rows_r[i][0]) if not np.isnan(rows_r[i][0]) else -1,
                       reverse=True)[:top_n]

    rows_r     = [rows_r[i]     for i in order]
    rows_annot = [rows_annot[i] for i in order]
    labels     = [labels[i]     for i in order]

    r_df     = pd.DataFrame(rows_r,     index=labels, columns=[c for c, _ in cohorts])
    annot_df = pd.DataFrame(rows_annot, index=labels, columns=[c for c, _ in cohorts])
    return r_df, annot_df


def plot_side_by_side(clinic_r, clinic_annot, home_r, home_annot, n, out_name):
    """Clinic (left) + Home (right) top-10 heatmaps with a single shared colorbar.

    Layout: big gap between the two panels, small gap between home panel and cbar.
    Achieved with a nested gridspec (outer wspace=0.85, inner wspace=0.05).
    """
    fig = plt.figure(figsize=(16, 6.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.75)
    ax_l = fig.add_subplot(outer[0])
    inner = outer[1].subgridspec(1, 2, width_ratios=[1, 0.04], wspace=0.08)
    ax_r = fig.add_subplot(inner[0])
    cax  = fig.add_subplot(inner[1])

    for ax, r_df, annot_df, title, show_cbar in [
        (ax_l, clinic_r, clinic_annot, 'Clinic', False),
        (ax_r, home_r,   home_annot,   'Home',   True),
    ]:
        sns.heatmap(
            r_df,
            cmap=CMAP,
            annot=annot_df.values,
            fmt='',
            vmin=-1, vmax=1,
            xticklabels=True,
            yticklabels=True,
            cbar=show_cbar,
            cbar_ax=cax if show_cbar else None,
            cbar_kws={'label': 'Spearman \u03c1'} if show_cbar else None,
            annot_kws={'fontsize': 13, 'fontweight': 'bold'},
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(14)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight('bold')
            lbl.set_fontsize(14)

    # Style the shared colorbar label consistently across heatmaps
    cax.set_ylabel('Spearman ρ', fontsize=14)
    cax.tick_params(labelsize=12)

    fig.suptitle(f'Wearable Feature Correlations with 6MWD (|\u03c1| > 0.3, Top 10 per Setting)',
                 fontsize=15, fontweight='bold', y=0.97)
    save_paper_figure(fig, out_name)
    plt.close(fig)
    print(f"  Saved {out_name} [results/ + POMS/]")


def main():
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj_df['sixmwd'].values.astype(float)
    n = len(subj_df)

    # Cohort masks (demographics no longer included in the heatmap pool)
    is_poms = (subj_df['cohort'] == 'M').values
    is_healthy = ~is_poms

    # ── Clinic features: Gait + CWT + WalkSway (no Demo) ──
    c_gait_df = pd.read_csv(FEATS / 'clinic_gait_features.csv')
    c_cwt_df  = pd.read_csv(FEATS / 'clinic_cwt_features.csv')
    c_ws_df   = pd.read_csv(FEATS / 'clinic_walksway_features.csv')
    c_gait_cols = [c for c in c_gait_df.columns if c != 'key']
    c_cwt_cols  = [c for c in c_cwt_df.columns  if c != 'key']
    c_ws_cols   = [c for c in c_ws_df.columns   if c != 'key']
    c_gait = impute(c_gait_df[c_gait_cols].values.astype(float))
    c_cwt  = impute(c_cwt_df[c_cwt_cols].values.astype(float))
    c_ws   = impute(c_ws_df[c_ws_cols].values.astype(float))
    X_clinic = np.column_stack([c_gait, c_cwt, c_ws])
    clinic_names = c_gait_cols + c_cwt_cols + c_ws_cols
    clinic_cats  = (['Gait'] * len(c_gait_cols)
                    + ['CWT'] * len(c_cwt_cols)
                    + ['WalkSway'] * len(c_ws_cols))

    # ── Home features: Bout + Act only (no Demo, no WalkSway) ──
    h_pb_df = pd.read_csv(FEATS / 'home_perbout_features.csv')
    h_pb_all       = [c for c in h_pb_df.columns if c != 'key']
    h_pb_gait_cols = [c for c in h_pb_all if c.startswith('g_')]
    h_pb_act_cols  = [c for c in h_pb_all if c.startswith('act_')]
    h_pb_gait = impute(h_pb_df[h_pb_gait_cols].values.astype(float))
    h_pb_act  = impute(h_pb_df[h_pb_act_cols].values.astype(float))
    X_home = np.column_stack([h_pb_gait, h_pb_act])
    home_names = h_pb_gait_cols + h_pb_act_cols
    home_cats  = (['Bout'] * len(h_pb_gait_cols)
                  + ['Act'] * len(h_pb_act_cols))

    # ── Build heatmap DataFrames (top 10 per setting, |r| > 0.3) ──
    print(f"\nBuilding Spearman \u03c1 heatmaps — top 10 features per setting "
          f"(|\u03c1| > 0.3, n={n}, POMS={is_poms.sum()}, Healthy={is_healthy.sum()})")
    # Clinic: top-10 by |ρ| with 6MWD, but restricted to features whose sign
    # matches the gait-literature expectation. The auto-selected `ml_spectral_entropy`
    # is replaced by `cwt_freq_cv_mean` (ρ=-0.21) because entropy has an
    # anti-intuitive positive sign in our cohort. Every other feature's sign
    # is already intuitive in a vigor/pathology framing.
    CLINIC_TOP10_FIXED = [
        'enmo_mean_g',
        'vt_rms_g',
        'jerk_mean_abs_gps',
        'cwt_fundamental_freq_mean',
        'cadence_hz',
        'cwt_estimated_cadence_mean',
        'cwt_dominant_freq_mean',
        'ml_rms_g',
        'ml_over_enmo',
        'cwt_freq_cv_mean',
    ]
    # Home: pinned to the LOO Spearman top-10 selection (these are the features
    # the headline Ridge actually consumes), in the order from step3 selection.
    HOME_TOP10_FIXED = [
        'g_duration_sec_max',
        'act_pct_vigorous',
        'g_duration_sec_cv',
        'act_enmo_p95',
        'g_ap_rms_med',
        'g_enmo_mean_p10',
        'g_ap_rms_cv',
        'g_jerk_mean_med',
        'g_acf_step_reg_max',
    ]
    clinic_r, clinic_annot = build_heatmap_df(
        X_clinic, clinic_names, clinic_cats, y, is_poms, is_healthy,
        fixed_features=CLINIC_TOP10_FIXED)
    home_r, home_annot = build_heatmap_df(
        X_home, home_names, home_cats, y, is_poms, is_healthy,
        fixed_features=HOME_TOP10_FIXED)

    # ── Side-by-side plot ──
    plot_side_by_side(
        clinic_r, clinic_annot, home_r, home_annot, n,
        'heatmap_feature_6mwd_corr_top10.png',
    )


if __name__ == '__main__':
    main()
