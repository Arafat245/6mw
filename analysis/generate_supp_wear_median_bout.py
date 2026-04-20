#!/usr/bin/env python3
"""Combined supplementary figure: (a) device wear-time violin and (b) scatter
of number of walking bouts vs median bout duration per subject.

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
from matplotlib.lines import Line2D
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

# 6MWD size scaling for panel (b) — same parameters and legend strategy as
# bout_distribution_overview.png.
SZ_MIN, SZ_MAX = 25, 260


def _sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _size_by_6mwd(vals, lo, hi):
    v = np.asarray(vals, float)
    return SZ_MIN + (v - lo) / max(hi - lo, 1e-9) * (SZ_MAX - SZ_MIN)


def main() -> None:
    subj = pd.read_csv(NPZ_DIR / '_subjects.csv')
    wear = pd.read_csv(FEATS / 'home_wear_time.csv')
    pb   = pd.read_csv(FEATS / 'home_perbout_features.csv')

    d = (subj[['key', 'cohort', 'sixmwd']]
         .merge(wear[['key', 'wear_hours']], on='key')
         .merge(pb[['key', 'g_duration_sec_med']], on='key'))
    bouts_dir = BASE / 'walking_bouts'
    d['n_bouts'] = [len(list((bouts_dir / k).glob('*.csv')))
                    if (bouts_dir / k).exists() else np.nan
                    for k in d['key']]
    d['sixmwd_m'] = d['sixmwd'] * 0.3048
    d['cohort_label'] = np.where(d['cohort'] == 'M', 'POMS', 'Healthy')

    fig = plt.figure(figsize=(14, 5.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.55, 1.0], wspace=0.22)
    ax_v = fig.add_subplot(outer[0])
    ax_s = fig.add_subplot(outer[1])

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

    ax_v.set_xticklabels([f'Healthy\n(n={len(hc)})', f'POMS\n(n={len(ms)})'],
                          fontsize=TICK, fontweight='bold')
    ax_v.set_xlabel('')
    ax_v.set_ylabel('Wear time (hours)', fontsize=LBL, fontweight='bold')
    ax_v.set_title('Device Wear Time Over 7-Day Monitoring',
                    fontsize=TIT, fontweight='bold', pad=8)
    ax_v.tick_params(axis='y', labelsize=TICK)
    ax_v.grid(True, axis='y', which='both', alpha=0.25,
               linestyle='--', linewidth=0.5)
    ax_v.set_axisbelow(True)

    # ─────────────────────────── Panel (b) ───────────────────────────
    smwd_lo = float(d['sixmwd_m'].min()); smwd_hi = float(d['sixmwd_m'].max())
    _lo  = int(np.floor(smwd_lo / 50) * 50)
    _hi  = int(np.ceil(smwd_hi / 50) * 50)
    _mid = int(((_lo + _hi) / 2) / 50) * 50
    SZ_LEG_VALS = [_lo, _mid, _hi]
    # In-panel scatter: sizes vary by 6MWD (same as main figure).
    for cohort, color in [('Healthy', CLR_HEALTHY), ('POMS', CLR_POMS)]:
        g = d[d['cohort_label'] == cohort].dropna(subset=['n_bouts', 'g_duration_sec_med'])
        ax_s.scatter(
            g['n_bouts'], g['g_duration_sec_med'],
            s=_size_by_6mwd(g['sixmwd_m'].values, smwd_lo, smwd_hi),
            c=color, alpha=0.7, edgecolor='white', linewidth=0.7,
            zorder=3,
        )
    ax_s.set_xscale('log'); ax_s.set_yscale('log')
    ax_s.set_xlabel('Number of walking bouts per subject',
                     fontsize=LBL, fontweight='bold')
    ax_s.set_ylabel('Median bout duration (s)',
                     fontsize=LBL, fontweight='bold')
    ax_s.set_title('Per-Subject Bout Count vs Median Bout Duration',
                    fontsize=TIT, fontweight='bold', pad=8)
    ax_s.tick_params(axis='both', labelsize=TICK)
    ax_s.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.5)
    # Cohort legend uses fixed-size proxy dots so the Healthy/POMS markers are
    # identical in size across figures (the in-panel dots still vary by 6MWD).
    COHORT_LEG_S = 90
    plt.sca(ax_s)
    cohort_handles = [
        plt.scatter([], [], s=COHORT_LEG_S, c=CLR_HEALTHY,
                    edgecolor='white', linewidth=0.7, alpha=0.85),
        plt.scatter([], [], s=COHORT_LEG_S, c=CLR_POMS,
                    edgecolor='white', linewidth=0.7, alpha=0.85),
    ]
    n_hc = int((d['cohort_label'] == 'Healthy').sum())
    n_ms = int((d['cohort_label'] == 'POMS').sum())
    leg_cohort = ax_s.legend(
        cohort_handles, [f'Healthy (n={n_hc})', f'POMS (n={n_ms})'],
        loc='upper right', frameon=True, fontsize=LEG)
    ax_s.add_artist(leg_cohort)
    # 6MWD size legend — identical place, format, and proxy style as
    # bout_distribution_overview.png (plt.scatter proxies, loc='lower right',
    # title='6MWD', fontsize LEG-1).
    plt.sca(ax_s)
    size_handles = [plt.scatter([], [],
                                 s=float(_size_by_6mwd(np.array([v]), smwd_lo, smwd_hi)[0]),
                                 c='gray', alpha=0.6, edgecolor='white')
                    for v in SZ_LEG_VALS]
    ax_s.legend(size_handles, [f'{v} m' for v in SZ_LEG_VALS],
                 loc='lower right', title='6MWD', frameon=True,
                 fontsize=LEG - 1, title_fontsize=LEG - 1)

    fig.suptitle('Home Monitoring Summary',
                  fontsize=SUP, fontweight='bold', y=1.02)
    out_name = 'fig_supp_wear_median_bout.png'
    fig.savefig(POMS_FIGURES / out_name, dpi=300, bbox_inches='tight')
    fig.savefig(OUT / out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_name} [POMS/ + results/]')


if __name__ == '__main__':
    main()
