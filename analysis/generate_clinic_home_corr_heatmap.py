#!/usr/bin/env python3
"""Clinic ↔ Home same-category feature correlation heatmap.

Each row is a clinic feature; each column is a home aggregation statistic
(med, iqr, p10, p90, max, cv) of the home feature group that belongs to the
same biomechanical category. Cell = Spearman ρ across n=101 subjects.

Top-10 clinic features are those with the strongest best-match correlation
to any home stat in their category — found in the clinic↔home analysis.

Output: POMS/figures/heatmap_clinic_home_feature_corr.png
        results/paper_figures/heatmap_clinic_home_feature_corr.png

Style matches the other paper heatmaps (sns.diverging_palette, bold
annotations, Spearman ρ colorbar).
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
FEATS = BASE / 'feats'
POMS_FIGURES = BASE / 'POMS' / 'figures'
OUT = BASE / 'results' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)
POMS_FIGURES.mkdir(parents=True, exist_ok=True)
NPZ_DIR = BASE / 'home_full_recording_npz'

CMAP = sns.diverging_palette(0, 255, sep=77, as_cmap=True)

# 5×5 cross-correlation panel. Rows = clinic features (with category prefix).
# Columns = their matched home p90 features. Diagonal = the top-5 matched
# pairs (ranked by |ρ|); off-diagonal cells show cross-feature spillover.
CLINIC_ROWS = [
    ('Gait', 'enmo_mean_g'),
    ('Gait', 'jerk_mean_abs_gps'),
    ('Gait', 'ml_rms_g'),
    ('Gait', 'cadence_hz'),
    ('Gait', 'vt_rms_g'),
]
HOME_COLS = [
    'g_enmo_mean_p90',
    'g_jerk_mean_p90',
    'g_ml_rms_p90',
    'g_cadence_hz_p90',
    'g_vt_range_p90',
]


def sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _compute_corr(merged: pd.DataFrame, mask: np.ndarray):
    r_df = pd.DataFrame(np.nan, index=range(len(CLINIC_ROWS)),
                         columns=HOME_COLS, dtype=float)
    annot = pd.DataFrame('', index=range(len(CLINIC_ROWS)), columns=HOME_COLS)
    y_labels = [f"{cat}: {feat}" for cat, feat in CLINIC_ROWS]
    for i, (_, cf) in enumerate(CLINIC_ROWS):
        cv = merged[cf].values.astype(float)
        for j, hfeat in enumerate(HOME_COLS):
            if hfeat not in merged.columns:
                annot.iat[i, j] = '—'; continue
            hv = merged[hfeat].values.astype(float)
            m = mask & ~(np.isnan(cv) | np.isnan(hv))
            if m.sum() < 10:
                annot.iat[i, j] = '—'; continue
            r, p = spearmanr(cv[m], hv[m])
            r_df.iat[i, j] = r
            annot.iat[i, j] = f"{r:+.2f}{sig_stars(p)}"
    r_df.index = y_labels
    annot.index = y_labels
    return r_df, annot


def main() -> None:
    ids = pd.read_csv(NPZ_DIR / '_subjects.csv')
    c_gait = pd.read_csv(FEATS / 'clinic_gait_features.csv')
    c_cwt  = pd.read_csv(FEATS / 'clinic_cwt_features.csv')
    c_ws   = pd.read_csv(FEATS / 'clinic_walksway_features.csv')
    h      = pd.read_csv(FEATS / 'home_perbout_features.csv')
    merged = (ids[['key', 'cohort']]
              .merge(c_gait, on='key').merge(c_cwt, on='key')
              .merge(c_ws, on='key').merge(h, on='key'))

    n_all = len(merged)
    is_poms    = (merged['cohort'] == 'M').values
    is_healthy = (merged['cohort'] == 'C').values

    panels = [
        ('All',     np.ones(n_all, bool), n_all),
        ('POMS',    is_poms,              int(is_poms.sum())),
        ('Healthy', is_healthy,           int(is_healthy.sum())),
    ]
    corrs = [(name, n, _compute_corr(merged, mask)) for name, mask, n in panels]

    # Tight layout: small wspace between the 3 panels, small gap before cbar.
    fig = plt.figure(figsize=(16, 4.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 0.018], wspace=0.03)
    panel_grid = outer[0].subgridspec(1, 3, wspace=0.15)
    axes = [fig.add_subplot(panel_grid[k]) for k in range(3)]
    cax  = fig.add_subplot(outer[1])

    for k, ((name, n_panel, (r_df, annot)), ax) in enumerate(zip(corrs, axes)):
        show_cbar = (k == len(axes) - 1)
        sns.heatmap(
            r_df, cmap=CMAP, annot=annot.values, fmt='',
            vmin=-0.6, vmax=0.6,
            xticklabels=True, yticklabels=(k == 0),
            cbar=show_cbar, cbar_ax=cax if show_cbar else None,
            cbar_kws={'label': 'Spearman \u03c1'} if show_cbar else None,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'}, ax=ax,
        )
        ax.set_title(f'{name}', fontsize=15, fontweight='bold')
        ax.set_xlabel('Home' if k == 1 else '', fontsize=14, fontweight='bold')
        ax.set_ylabel('Clinic' if k == 0 else '', fontsize=14, fontweight='bold')
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(12); lbl.set_rotation(0)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(12); lbl.set_rotation(45); lbl.set_ha('right')

    # Consistent colorbar styling across heatmaps
    cax.set_ylabel('Spearman \u03c1', fontsize=14)
    cax.tick_params(labelsize=12)

    fig.suptitle('Clinic and Home Wearable Feature Correlations (Top-5 Matched Pairs)',
                 fontsize=16, fontweight='bold', y=1.02)
    fname = 'heatmap_clinic_home_feature_corr.png'
    fig.savefig(OUT / fname, dpi=300, bbox_inches='tight')
    fig.savefig(POMS_FIGURES / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname} [results/ + POMS/]")


if __name__ == '__main__':
    main()
