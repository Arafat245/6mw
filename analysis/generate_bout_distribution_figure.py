#!/usr/bin/env python3
"""
Three-panel bout-distribution figure (mirrored to POMS/figures/):

  bout_distribution_overview.png
    LEFT   panel: violin of longest sustained bout per cohort (duration).
    MIDDLE panel: violin of per-subject 90th-pct bout ENMO (intensity).
    RIGHT  panel: pooled bout-duration survival per cohort (1-ECDF) with bootstrap CI.
"""
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent.parent
FEATS = BASE / 'feats'
NPZ_DIR = BASE / 'home_full_recording_npz'
OUT = BASE / 'results' / 'paper_figures'; OUT.mkdir(parents=True, exist_ok=True)
POMS_FIGURES = BASE / 'POMS' / 'figures'; POMS_FIGURES.mkdir(parents=True, exist_ok=True)
FS = 30
CLR_HEALTHY = '#6BAED6'
CLR_POMS = '#FC8D62'
POINT_SIZE = 90

# Match heatmap_feature_6mwd_corr_top10.png font sizes for paper consistency:
#   suptitle 14, subplot title 13, tick/annot 11, axis label 12, legend 11
LBL = 13   # axis label font (+1)
TIT = 14   # subplot title font (+1)
SUP = 15   # suptitle font (+1)
LEG = 12   # legend font (+1)
TICK = 12  # tick font (+1)


def save_paper_figure(fig, filename, dpi=300):
    fig.savefig(OUT / filename, dpi=dpi, bbox_inches='tight')
    fig.savefig(POMS_FIGURES / filename, dpi=dpi, bbox_inches='tight')


def main():
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    with open(FEATS / 'home_walking_bouts.pkl', 'rb') as f:
        bouts_per_subj = pickle.load(f)['bouts']

    # Per-subject max/median bout duration — pulled from the FEATURE CSV so
    # the plots are consistent with g_duration_sec_max / _med used in the
    # heatmap, ms_vs_healthy table, and LOO Ridge features. (The raw pickle
    # inflates max duration for non-30Hz subjects because sample indices are
    # divided by a hardcoded FS=30; the feature CSV uses per-subject FS and
    # also drops bouts where gait features could not be computed.)
    feat_df = pd.read_csv(FEATS / 'home_perbout_features.csv')
    max_by_key = feat_df.set_index('key')['g_duration_sec_max'].to_dict()
    enmo_p90_by_key = feat_df.set_index('key')['g_enmo_mean_p90'].to_dict()

    rows = []
    for _, r in subj_df.iterrows():
        key = r['key']
        if key not in bouts_per_subj or key not in max_by_key:
            continue
        n_bouts = len(bouts_per_subj[key])
        rows.append({
            'key': key,
            'cohort': 'POMS' if r['cohort'] == 'M' else 'Healthy',
            'n_bouts': n_bouts,
            'max_dur':    float(max_by_key[key]),
            'enmo_p90':   float(enmo_p90_by_key.get(key, np.nan)),
        })
    df = pd.DataFrame(rows).dropna(subset=['max_dur'])

    n_p = (df.cohort == 'POMS').sum()
    n_h = (df.cohort == 'Healthy').sum()
    print(f"Subjects with bouts: {len(df)} (POMS={n_p}, Healthy={n_h})")

    pooled = {'Healthy': [], 'POMS': []}
    for _, r in subj_df.iterrows():
        if r['key'] not in bouts_per_subj:
            continue
        durs = [(end - start) / FS for start, end in bouts_per_subj[r['key']]]
        pooled['POMS' if r['cohort'] == 'M' else 'Healthy'].extend(durs)

    # ─────────────────────────────────────────────────────────────────
    # Combined figure: outer 1×3 (duration | intensity | survival)
    # ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 7))
    outer = fig.add_gridspec(1, 3, width_ratios=[0.45, 0.45, 1.0], wspace=0.28)

    # LEFT: violin + jitter of longest sustained bout per cohort
    ax_bx = fig.add_subplot(outer[0])
    poms_max    = df[df.cohort == 'POMS']['max_dur'].values
    healthy_max = df[df.cohort == 'Healthy']['max_dur'].values
    _, pval = mannwhitneyu(poms_max, healthy_max, alternative='two-sided')
    stars = ('***' if pval < 0.001
             else '**' if pval < 0.01
             else '*' if pval < 0.05
             else 'ns')

    # Linear scale in minutes
    df_bx = df[['cohort', 'max_dur']].copy()
    df_bx['max_dur_min'] = df_bx['max_dur'] / 60.0

    palette = {'Healthy': CLR_HEALTHY, 'POMS': CLR_POMS}
    sns.violinplot(
        data=df_bx, x='cohort', y='max_dur_min',
        order=['Healthy', 'POMS'], palette=palette,
        inner=None, cut=0, linewidth=1.2,
        saturation=0.9, width=0.85, ax=ax_bx,
    )
    for coll in ax_bx.collections:
        coll.set_alpha(0.55)
        coll.set_edgecolor('none')

    sns.stripplot(
        data=df_bx, x='cohort', y='max_dur_min',
        order=['Healthy', 'POMS'], palette=palette,
        size=4.2, jitter=0.18, alpha=0.9,
        edgecolor='white', linewidth=0.6, ax=ax_bx,
    )

    # Median ticks drawn in black on top of each violin
    for i, cohort in enumerate(['Healthy', 'POMS']):
        med = float(df_bx.loc[df_bx.cohort == cohort, 'max_dur_min'].median())
        ax_bx.plot([i - 0.22, i + 0.22], [med, med],
                   color='black', linewidth=2.0, zorder=5)

    ax_bx.set_xticklabels(['Healthy', 'POMS'],
                           fontsize=TICK, fontweight='bold')
    ax_bx.set_xlabel('')
    ax_bx.set_ylabel('Longest sustained bout (min)',
                      fontsize=LBL, fontweight='bold')
    ax_bx.set_title('Longest Bout Distribution',
                     fontsize=TIT, fontweight='bold', pad=8)
    ax_bx.tick_params(axis='y', labelsize=TICK)
    ax_bx.grid(True, axis='y', which='both', alpha=0.25,
               linestyle='--', linewidth=0.5)
    ax_bx.set_axisbelow(True)

    # Significance bracket + stars (linear space)
    y_top = float(df_bx['max_dur_min'].max())
    y_bar = y_top * 1.05
    y_txt = y_top * 1.09
    ax_bx.plot([0, 0, 1, 1],
                [y_bar - y_top * 0.012, y_bar, y_bar, y_bar - y_top * 0.012],
                color='black', linewidth=1.1)
    ax_bx.text(0.5, y_txt, f'{stars}  (p = {pval:.1e})',
                ha='center', va='bottom',
                fontsize=LBL, fontweight='bold')
    ax_bx.set_ylim(top=y_top * 1.18)

    # MIDDLE: violin of per-subject 90th-percentile bout intensity (ENMO, g)
    ax_in = fig.add_subplot(outer[1])
    df_in = df[['cohort', 'enmo_p90']].dropna().copy()
    poms_int    = df_in.loc[df_in.cohort == 'POMS',    'enmo_p90'].values
    healthy_int = df_in.loc[df_in.cohort == 'Healthy', 'enmo_p90'].values
    _, pval_in = mannwhitneyu(poms_int, healthy_int, alternative='two-sided')
    stars_in = ('***' if pval_in < 0.001
                else '**' if pval_in < 0.01
                else '*' if pval_in < 0.05
                else 'ns')

    sns.violinplot(
        data=df_in, x='cohort', y='enmo_p90',
        order=['Healthy', 'POMS'], palette=palette,
        inner=None, cut=0, linewidth=1.2,
        saturation=0.9, width=0.85, ax=ax_in,
    )
    for coll in ax_in.collections:
        coll.set_alpha(0.55)
        coll.set_edgecolor('none')

    sns.stripplot(
        data=df_in, x='cohort', y='enmo_p90',
        order=['Healthy', 'POMS'], palette=palette,
        size=4.2, jitter=0.18, alpha=0.9,
        edgecolor='white', linewidth=0.6, ax=ax_in,
    )

    for i, cohort in enumerate(['Healthy', 'POMS']):
        med = float(df_in.loc[df_in.cohort == cohort, 'enmo_p90'].median())
        ax_in.plot([i - 0.22, i + 0.22], [med, med],
                   color='black', linewidth=2.0, zorder=5)

    ax_in.set_xticklabels(['Healthy', 'POMS'],
                           fontsize=TICK, fontweight='bold')
    ax_in.set_xlabel('')
    ax_in.set_ylabel('90th percentile bout intensity (g)',
                      fontsize=LBL, fontweight='bold')
    ax_in.set_title('Bout Intensity Distribution',
                     fontsize=TIT, fontweight='bold', pad=8)
    ax_in.tick_params(axis='y', labelsize=TICK)
    ax_in.grid(True, axis='y', which='both', alpha=0.25,
               linestyle='--', linewidth=0.5)
    ax_in.set_axisbelow(True)

    y_top_in = float(df_in['enmo_p90'].max())
    y_bar_in = y_top_in * 1.04
    y_txt_in = y_top_in * 1.08
    ax_in.plot([0, 0, 1, 1],
                [y_bar_in - y_top_in * 0.01, y_bar_in, y_bar_in,
                 y_bar_in - y_top_in * 0.01],
                color='black', linewidth=1.1)
    ax_in.text(0.5, y_txt_in, f'{stars_in}  (p = {pval_in:.1e})',
                ha='center', va='bottom',
                fontsize=LBL, fontweight='bold')
    ax_in.set_ylim(top=y_top_in * 1.18)

    # RIGHT: survival curves (x-axis in minutes)
    ax_sv = fig.add_subplot(outer[2])
    for cohort, color in [('Healthy', CLR_HEALTHY), ('POMS', CLR_POMS)]:
        durs = np.array(sorted(pooled[cohort]))  # seconds
        n_b = len(durs)
        surv = 1.0 - np.arange(n_b) / n_b
        ax_sv.plot(durs / 60.0, surv, color=color, linewidth=2.4,
                   label=f'{cohort}  ({n_b:,} pooled bouts)')
        rng = np.random.default_rng(42)
        subj_keys = [k for k in subj_df['key']
                     if k in bouts_per_subj
                     and (subj_df.loc[subj_df.key == k, 'cohort'].iloc[0]
                          == ('M' if cohort == 'POMS' else 'C'))]
        boot_grids = []
        grid = np.logspace(np.log10(10), np.log10(durs.max()), 80)  # seconds
        for _ in range(300):
            sample_keys = rng.choice(subj_keys, size=len(subj_keys), replace=True)
            boot_durs = []
            for k in sample_keys:
                boot_durs.extend([(e - s) / FS for s, e in bouts_per_subj[k]])
            if not boot_durs:
                continue
            bd = np.array(sorted(boot_durs))
            boot_surv = 1.0 - np.searchsorted(bd, grid, side='left') / len(bd)
            boot_grids.append(boot_surv)
        boot_arr = np.array(boot_grids)
        lo, hi = np.percentile(boot_arr, [2.5, 97.5], axis=0)
        ax_sv.fill_between(grid / 60.0, lo, hi, color=color, alpha=0.18)

    ax_sv.set_xscale('log'); ax_sv.set_yscale('log')
    x_ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax_sv.set_xticks(x_ticks)
    ax_sv.set_xticklabels([('%g' % t) for t in x_ticks])
    ax_sv.set_xticks([], minor=True)
    ax_sv.set_xlabel('Bout duration t (min)', fontsize=LBL, fontweight='bold')
    ax_sv.set_ylabel('Fraction of bouts with duration \u2265 t',
                     fontsize=LBL, fontweight='bold')
    ax_sv.set_title('Pooled Bout-Duration Survival (95% Bootstrap CI)',
                    fontsize=TIT, fontweight='bold', pad=8)
    ax_sv.tick_params(axis='both', labelsize=TICK)
    ax_sv.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.5)
    ax_sv.legend(loc='upper right', frameon=True, fontsize=LEG)
    for t_thresh in (30, 60, 120, 300):  # seconds
        for cohort in ('Healthy', 'POMS'):
            durs = np.array(pooled[cohort])
            frac = float((durs >= t_thresh).mean())
            color = CLR_HEALTHY if cohort == 'Healthy' else CLR_POMS
            ax_sv.scatter([t_thresh / 60.0], [frac], s=24, c=color,
                          edgecolor='black', linewidth=0.5, zorder=4)

    fig.suptitle('Home Walking-Bout Characteristics by Cohort',
                 fontsize=SUP, fontweight='bold', y=1.0)
    save_paper_figure(fig, 'bout_distribution_overview.png')
    plt.close(fig)
    print("  Saved bout_distribution_overview.png [results/ + POMS/]")

    print("\n  Fraction of bouts \u2265 threshold:")
    print(f"  {'Threshold':>12s}  {'Healthy':>10s}  {'POMS':>10s}")
    for t_thresh in (30, 60, 120, 300, 600):
        h = (np.array(pooled['Healthy']) >= t_thresh).mean()
        p = (np.array(pooled['POMS'])   >= t_thresh).mean()
        print(f"  {t_thresh/60.0:>8.2f} min  {h:>10.2%}  {p:>10.2%}")

if __name__ == '__main__':
    main()
