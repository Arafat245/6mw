#!/usr/bin/env python3
"""Combined supplementary figure: (a) device wear-time violin and (b)
per-subject median bout duration violin by cohort.

Styling is intentionally aligned with ``bout_distribution_overview.png`` so
the supplementary figure matches the main-text figures in palette, fonts,
and violin/scatter conventions.

Output:
  POMS/figures/fig_supp_wear_median_bout.png
  results/paper_figures/fig_supp_wear_median_bout.png
"""
from __future__ import annotations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
FEATS = BASE / 'feats'
POMS_FIGURES = BASE / 'POMS' / 'figures'
OUT = BASE / 'results' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)
POMS_FIGURES.mkdir(parents=True, exist_ok=True)
NPZ_DIR = BASE / 'home_full_recording_npz'

# Same palette + font sizes as bout_distribution_overview.png.
CLR_HEALTHY = '#6BAED6'
CLR_POMS    = '#FC8D62'
LBL, TIT, SUP, LEG, TICK = 12, 13, 14, 11, 11


def _sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def main() -> None:
    subj = pd.read_csv(NPZ_DIR / '_subjects.csv')
    wear = pd.read_csv(FEATS / 'home_wear_time.csv')
    pb   = pd.read_csv(FEATS / 'home_perbout_features.csv')

    d = (subj[['key', 'cohort', 'sixmwd']]
         .merge(wear[['key', 'wear_hours']], on='key')
         .merge(pb[['key', 'g_duration_sec_med']], on='key'))
    d['cohort_label'] = np.where(d['cohort'] == 'M', 'POMS', 'Healthy')

    fig = plt.figure(figsize=(10, 5.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.32)
    ax_v = fig.add_subplot(outer[0])
    ax_d = fig.add_subplot(outer[1])

    # ─────────────────────────── Panel (a) ───────────────────────────
    palette = {'Healthy': CLR_HEALTHY, 'POMS': CLR_POMS}
    order = ['Healthy', 'POMS']
    sns.violinplot(
        data=d, x='cohort_label', y='wear_hours',
        order=order, palette=palette,
        inner=None, cut=0, linewidth=1.2,
        saturation=0.9, width=0.85, ax=ax_v,
    )
    for coll in ax_v.collections:
        coll.set_alpha(0.55)
        coll.set_edgecolor('none')
    sns.stripplot(
        data=d, x='cohort_label', y='wear_hours',
        order=order, palette=palette,
        size=4.2, jitter=0.18, alpha=0.9,
        edgecolor='white', linewidth=0.6, ax=ax_v,
    )
    for i, cohort in enumerate(order):
        med = float(d.loc[d['cohort_label'] == cohort, 'wear_hours'].median())
        ax_v.plot([i - 0.22, i + 0.22], [med, med],
                   color='black', linewidth=2.0, zorder=5)

    hc = d.loc[d['cohort_label'] == 'Healthy', 'wear_hours']
    ms = d.loc[d['cohort_label'] == 'POMS',    'wear_hours']
    _, pval = mannwhitneyu(hc, ms, alternative='two-sided')
    stars = _sig_stars(pval)
    if stars:
        y_top = float(d['wear_hours'].max())
        y_bar = y_top + (y_top * 0.05)
        ax_v.plot([0, 0, 1, 1],
                   [y_bar - (y_top * 0.01), y_bar, y_bar, y_bar - (y_top * 0.01)],
                   color='black', linewidth=1.1)
        ax_v.text(0.5, y_bar + (y_top * 0.02),
                  f'{stars}  (p = {pval:.2g})',
                  ha='center', va='bottom', fontsize=LBL, fontweight='bold')

    ax_v.set_xticklabels(['Healthy', 'POMS'],
                          fontsize=TICK, fontweight='bold')
    ax_v.set_xlabel('')
    ax_v.set_ylabel('Wear time (hours)', fontsize=LBL, fontweight='bold')
    ax_v.set_title('Device Wear Time',
                    fontsize=TIT, fontweight='bold', pad=8)
    ax_v.tick_params(axis='y', labelsize=TICK)
    ax_v.grid(True, axis='y', which='both', alpha=0.25,
               linestyle='--', linewidth=0.5)
    ax_v.set_axisbelow(True)

    # ─────────────────────────── Panel (b) ───────────────────────────
    # Per-subject median bout duration — one value per subject, so each
    # subject contributes a single point to the cohort violin.
    sns.violinplot(
        data=d, x='cohort_label', y='g_duration_sec_med',
        order=order, palette=palette,
        inner=None, cut=0, linewidth=1.2,
        saturation=0.9, width=0.85, ax=ax_d,
    )
    for coll in ax_d.collections:
        coll.set_alpha(0.55)
        coll.set_edgecolor('none')
    sns.stripplot(
        data=d, x='cohort_label', y='g_duration_sec_med',
        order=order, palette=palette,
        size=4.2, jitter=0.18, alpha=0.9,
        edgecolor='white', linewidth=0.6, ax=ax_d,
    )
    for i, cohort in enumerate(order):
        med = float(d.loc[d['cohort_label'] == cohort, 'g_duration_sec_med'].median())
        ax_d.plot([i - 0.22, i + 0.22], [med, med],
                  color='black', linewidth=2.0, zorder=5)

    hc_d = d.loc[d['cohort_label'] == 'Healthy', 'g_duration_sec_med']
    ms_d = d.loc[d['cohort_label'] == 'POMS',    'g_duration_sec_med']

    ax_d.set_xticklabels(['Healthy', 'POMS'],
                         fontsize=TICK, fontweight='bold')
    ax_d.set_xlabel('')
    ax_d.set_ylabel('Median bout duration (s)', fontsize=LBL, fontweight='bold')
    ax_d.set_title('Per-Subject Median Bout Duration',
                   fontsize=TIT, fontweight='bold', pad=8)
    ax_d.tick_params(axis='y', labelsize=TICK)
    ax_d.grid(True, axis='y', which='both', alpha=0.25,
              linestyle='--', linewidth=0.5)
    ax_d.set_axisbelow(True)

    fig.suptitle('Home Monitoring Summary',
                  fontsize=SUP, fontweight='bold', y=1.02)
    out_name = 'fig_supp_wear_median_bout.png'
    fig.savefig(POMS_FIGURES / out_name, dpi=300, bbox_inches='tight')
    fig.savefig(OUT / out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_name} [POMS/ + results/]')


if __name__ == '__main__':
    main()
