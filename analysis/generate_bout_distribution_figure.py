#!/usr/bin/env python3
"""
Side-by-side bout-distribution figure (mirrored to POMS/figures/):

  bout_distribution_overview.png
    LEFT  panel: joint scatter (n_bouts vs longest sustained bout) with KDE marginals.
    RIGHT panel: pooled bout-duration survival per cohort (1-ECDF) with bootstrap CI.
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
LBL = 12   # axis label font
TIT = 13   # subplot title font
SUP = 14   # suptitle font
LEG = 11   # legend font
TICK = 11  # tick font


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
    med_by_key = feat_df.set_index('key')['g_duration_sec_med'].to_dict()

    rows = []
    durs_by_subj = {}
    for _, r in subj_df.iterrows():
        key = r['key']
        if key not in bouts_per_subj or key not in max_by_key:
            continue
        durations = np.array([(end - start) / FS for start, end in bouts_per_subj[key]])
        durs_by_subj[key] = durations
        rows.append({
            'key': key,
            'cohort': 'POMS' if r['cohort'] == 'M' else 'Healthy',
            'n_bouts': len(durations),
            'max_dur':    float(max_by_key[key]),
            'median_dur': float(med_by_key[key]),
        })
    df = pd.DataFrame(rows).dropna(subset=['max_dur'])
    # Attach 6MWD (m) and map to point sizes — used by both joint scatters
    sixmwd_m = (subj_df.set_index('key')['sixmwd'].astype(float) * 0.3048).to_dict()
    df['sixmwd_m'] = df['key'].map(sixmwd_m)
    SZ_MIN, SZ_MAX = 25, 260
    smwd_lo, smwd_hi = df['sixmwd_m'].min(), df['sixmwd_m'].max()

    def _size_by_6mwd(vals):
        v = np.asarray(vals, float)
        return SZ_MIN + (v - smwd_lo) / max(smwd_hi - smwd_lo, 1e-9) * (SZ_MAX - SZ_MIN)

    # Three tick values for size legend (rounded to nearest 50)
    _lo = int(np.floor(smwd_lo / 50) * 50)
    _hi = int(np.ceil(smwd_hi / 50) * 50)
    _mid = int((_lo + _hi) / 2 / 50) * 50
    SZ_LEG_VALS = [_lo, _mid, _hi]

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
    # Combined figure: outer 1×3 (boxplot | joint+marginals | survival)
    # ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 7))
    outer = fig.add_gridspec(1, 3, width_ratios=[0.45, 1.05, 1], wspace=0.18)

    # LEFT: violin + jitter of longest sustained bout per cohort
    ax_bx = fig.add_subplot(outer[0])
    poms_max    = df[df.cohort == 'POMS']['max_dur'].values
    healthy_max = df[df.cohort == 'Healthy']['max_dur'].values
    _, pval = mannwhitneyu(poms_max, healthy_max, alternative='two-sided')
    stars = ('***' if pval < 0.001
             else '**' if pval < 0.01
             else '*' if pval < 0.05
             else 'ns')

    # Work in log10 space so the violin kernel fits a long-tailed distribution
    df_bx = df[['cohort', 'max_dur']].copy()
    df_bx['log_max'] = np.log10(df_bx['max_dur'])

    palette = {'Healthy': CLR_HEALTHY, 'POMS': CLR_POMS}
    sns.violinplot(
        data=df_bx, x='cohort', y='log_max',
        order=['Healthy', 'POMS'], palette=palette,
        inner=None, cut=0, linewidth=1.2,
        saturation=0.9, width=0.85, ax=ax_bx,
    )
    for coll in ax_bx.collections:
        coll.set_alpha(0.55)
        coll.set_edgecolor('none')

    sns.stripplot(
        data=df_bx, x='cohort', y='log_max',
        order=['Healthy', 'POMS'], palette=palette,
        size=4.2, jitter=0.18, alpha=0.9,
        edgecolor='white', linewidth=0.6, ax=ax_bx,
    )

    # Median ticks drawn in black on top of each violin
    for i, cohort in enumerate(['Healthy', 'POMS']):
        med = np.median(np.log10(df_bx.loc[df_bx.cohort == cohort, 'max_dur']))
        ax_bx.plot([i - 0.22, i + 0.22], [med, med],
                   color='black', linewidth=2.0, zorder=5)

    # Pretty log-scale y-ticks (we plot log10 but label as seconds)
    log_lo = np.floor(df_bx['log_max'].min())
    log_hi = np.ceil(df_bx['log_max'].max())
    major_ticks = np.arange(log_lo, log_hi + 1)
    ax_bx.set_yticks(major_ticks)
    ax_bx.set_yticklabels([f'$10^{{{int(t)}}}$' for t in major_ticks])
    minor_ticks = np.concatenate([np.log10(np.arange(2, 10)) + t
                                   for t in major_ticks[:-1]])
    ax_bx.set_yticks(minor_ticks, minor=True)

    ax_bx.set_xticklabels([f'Healthy\n(n={len(healthy_max)})',
                            f'POMS\n(n={len(poms_max)})'],
                           fontsize=TICK, fontweight='bold')
    ax_bx.set_xlabel('')
    ax_bx.set_ylabel('Longest sustained bout (s)',
                      fontsize=LBL, fontweight='bold')
    ax_bx.set_title('Cohort Comparison',
                     fontsize=TIT, fontweight='bold', pad=8)
    ax_bx.tick_params(axis='y', labelsize=TICK)
    ax_bx.grid(True, axis='y', which='both', alpha=0.25,
               linestyle='--', linewidth=0.5)
    ax_bx.set_axisbelow(True)

    # Significance bracket + stars (in log10 space)
    y_top = df_bx['log_max'].max()
    y_bar = y_top + 0.22
    y_txt = y_bar + 0.10
    ax_bx.plot([0, 0, 1, 1],
                [y_bar - 0.05, y_bar, y_bar, y_bar - 0.05],
                color='black', linewidth=1.1)
    ax_bx.text(0.5, y_txt, f'{stars}  (p = {pval:.1e})',
                ha='center', va='bottom',
                fontsize=LBL, fontweight='bold')
    ax_bx.set_ylim(top=y_top + 0.6)

    # MIDDLE: joint scatter with marginal KDEs (5×5 subgridspec, ratio 4:1)
    left = outer[1].subgridspec(5, 5, wspace=0.06, hspace=0.06)
    ax_jn = fig.add_subplot(left[1:, :-1])           # joint scatter
    ax_mx = fig.add_subplot(left[0, :-1], sharex=ax_jn)   # top marginal
    ax_my = fig.add_subplot(left[1:, -1], sharey=ax_jn)   # right marginal

    for cohort, color in [('Healthy', CLR_HEALTHY), ('POMS', CLR_POMS)]:
        m = (df.cohort == cohort).values
        ax_jn.scatter(df.loc[m, 'n_bouts'], df.loc[m, 'max_dur'],
                      s=_size_by_6mwd(df.loc[m, 'sixmwd_m'].values),
                      c=color, alpha=0.7,
                      edgecolor='white', linewidth=0.7,
                      zorder=3)
    ax_jn.set_xscale('log'); ax_jn.set_yscale('log')
    ax_jn.set_xlabel('Number of walking bouts per subject', fontsize=LBL, fontweight='bold')
    ax_jn.set_ylabel('Longest sustained bout (s)', fontsize=LBL, fontweight='bold')
    ax_jn.tick_params(axis='both', labelsize=TICK)
    ax_jn.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.5)
    # Fixed-size cohort legend proxies — consistent across paper figures.
    COHORT_LEG_S = 90
    plt.sca(ax_jn)
    cohort_handles = [
        plt.scatter([], [], s=COHORT_LEG_S, c=CLR_HEALTHY,
                    edgecolor='white', linewidth=0.7, alpha=0.85),
        plt.scatter([], [], s=COHORT_LEG_S, c=CLR_POMS,
                    edgecolor='white', linewidth=0.7, alpha=0.85),
    ]
    n_hc = int((df.cohort == 'Healthy').sum())
    n_ms = int((df.cohort == 'POMS').sum())
    leg_cohort = ax_jn.legend(
        cohort_handles, [f'Healthy (n={n_hc})', f'POMS (n={n_ms})'],
        loc='upper left', frameon=True, fontsize=LEG)
    ax_jn.add_artist(leg_cohort)
    size_handles = [plt.scatter([], [], s=_size_by_6mwd([v])[0],
                                 c='gray', alpha=0.6, edgecolor='white')
                    for v in SZ_LEG_VALS]
    ax_jn.legend(size_handles, [f'{v} m' for v in SZ_LEG_VALS],
                  loc='lower right', title='6MWD',
                  frameon=True, fontsize=LEG - 1, title_fontsize=LEG - 1)

    sns.kdeplot(data=df, x='n_bouts', hue='cohort',
                palette={'Healthy': CLR_HEALTHY, 'POMS': CLR_POMS},
                hue_order=['Healthy', 'POMS'], fill=True, alpha=0.4,
                linewidth=1.5, bw_adjust=0.8, log_scale=True,
                ax=ax_mx, legend=False)
    sns.kdeplot(data=df, y='max_dur', hue='cohort',
                palette={'Healthy': CLR_HEALTHY, 'POMS': CLR_POMS},
                hue_order=['Healthy', 'POMS'], fill=True, alpha=0.4,
                linewidth=1.5, bw_adjust=0.8, log_scale=True,
                ax=ax_my, legend=False)
    for ax in (ax_mx, ax_my):
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ('top', 'right', 'left', 'bottom'):
            ax.spines[s].set_visible(False)

    ax_mx.set_title('Per-Subject Walking-Bout Distribution',
                    fontsize=TIT, fontweight='bold', pad=8)

    # RIGHT: survival curves
    ax_sv = fig.add_subplot(outer[2])
    for cohort, color in [('Healthy', CLR_HEALTHY), ('POMS', CLR_POMS)]:
        durs = np.array(sorted(pooled[cohort]))
        n_b = len(durs)
        surv = 1.0 - np.arange(n_b) / n_b
        ax_sv.plot(durs, surv, color=color, linewidth=2.4,
                   label=f'{cohort}  ({n_b:,} pooled bouts)')
        rng = np.random.default_rng(42)
        subj_keys = [k for k in subj_df['key']
                     if k in bouts_per_subj
                     and (subj_df.loc[subj_df.key == k, 'cohort'].iloc[0]
                          == ('M' if cohort == 'POMS' else 'C'))]
        boot_grids = []
        grid = np.logspace(np.log10(10), np.log10(durs.max()), 80)
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
        ax_sv.fill_between(grid, lo, hi, color=color, alpha=0.18)

    ax_sv.set_xscale('log'); ax_sv.set_yscale('log')
    ax_sv.set_xlabel('Bout duration t (s)', fontsize=LBL, fontweight='bold')
    ax_sv.set_ylabel('Fraction of bouts with duration \u2265 t',
                     fontsize=LBL, fontweight='bold')
    ax_sv.set_title('Pooled Bout-Duration Survival (95% Bootstrap CI)',
                    fontsize=TIT, fontweight='bold', pad=8)
    ax_sv.tick_params(axis='both', labelsize=TICK)
    ax_sv.grid(True, which='both', alpha=0.25, linestyle='--', linewidth=0.5)
    ax_sv.legend(loc='lower left', frameon=True, fontsize=LEG)
    for t_thresh in (30, 60, 120, 300):
        for cohort in ('Healthy', 'POMS'):
            durs = np.array(pooled[cohort])
            frac = float((durs >= t_thresh).mean())
            color = CLR_HEALTHY if cohort == 'Healthy' else CLR_POMS
            ax_sv.scatter([t_thresh], [frac], s=24, c=color,
                          edgecolor='black', linewidth=0.5, zorder=4)

    fig.suptitle('Home Walking-Bout Distribution and Pooled Survival',
                 fontsize=SUP, fontweight='bold', y=1.02)
    save_paper_figure(fig, 'bout_distribution_overview.png')
    plt.close(fig)
    print("  Saved bout_distribution_overview.png [results/ + POMS/]")

    print("\n  Fraction of bouts \u2265 threshold:")
    print(f"  {'Threshold':>10s}  {'Healthy':>10s}  {'POMS':>10s}")
    for t_thresh in (30, 60, 120, 300, 600):
        h = (np.array(pooled['Healthy']) >= t_thresh).mean()
        p = (np.array(pooled['POMS'])   >= t_thresh).mean()
        print(f"  {t_thresh:>8d} s  {h:>10.2%}  {p:>10.2%}")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE: per-subject bout-duration boxplots (POMS top, Healthy bottom)
    # Sorted by median bout duration.
    # ─────────────────────────────────────────────────────────────────
    poms_keys    = df[df.cohort == 'POMS'].sort_values('median_dur')['key'].tolist()
    healthy_keys = df[df.cohort == 'Healthy'].sort_values('median_dur')['key'].tolist()

    def _boxes(ax, keys, color, title):
        data = [durs_by_subj[k] for k in keys]
        ax.boxplot(data, positions=np.arange(len(keys)),
                   widths=0.7, patch_artist=True,
                   whis=(5, 95),
                   showfliers=True,
                   flierprops=dict(marker='o', markersize=1.8,
                                   markerfacecolor=color,
                                   markeredgecolor='none', alpha=0.35),
                   medianprops=dict(color='black', linewidth=1.0),
                   whiskerprops=dict(color=color, linewidth=0.9),
                   capprops=dict(color=color, linewidth=0.9),
                   boxprops=dict(facecolor=color, alpha=0.75,
                                 edgecolor=color, linewidth=0.7))
        ax.set_yscale('log')
        ax.set_xlim(-0.8, len(keys) - 0.2)
        ax.set_xticks([])
        ax.set_ylabel('Bout duration (s)', fontsize=LBL, fontweight='bold')
        ax.set_title(title, fontsize=TIT, fontweight='bold', pad=6)
        ax.tick_params(axis='y', labelsize=TICK)
        ax.grid(True, axis='y', which='both', alpha=0.25,
                linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    fig_bp, (ax_p, ax_h) = plt.subplots(
        2, 1, figsize=(13, 6.5),
        gridspec_kw={'height_ratios': [len(poms_keys), len(healthy_keys)],
                     'hspace': 0.35},
        sharey=True,
    )
    _boxes(ax_p, poms_keys, CLR_POMS,
           f'POMS (n={len(poms_keys)}) \u2014 subjects sorted by median bout duration')
    _boxes(ax_h, healthy_keys, CLR_HEALTHY,
           f'Healthy (n={len(healthy_keys)}) \u2014 subjects sorted by median bout duration')
    ax_h.set_xlabel('Subjects (sorted by median bout duration, low \u2192 high)',
                     fontsize=LBL, fontweight='bold')
    fig_bp.suptitle('Per-Subject Bout-Duration Distribution (Home)',
                     fontsize=SUP, fontweight='bold', y=1.00)
    save_paper_figure(fig_bp, 'bout_duration_boxplots_per_subject.png')
    plt.close(fig_bp)
    print("  Saved bout_duration_boxplots_per_subject.png [results/ + POMS/]")


if __name__ == '__main__':
    main()
