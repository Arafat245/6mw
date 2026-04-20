#!/usr/bin/env python3
"""Supplementary figure: 6MWD prediction MAE vs walking-bout minimum-duration
threshold. Re-aggregates cached per-bout features from the home pipeline at
increasing minimum-duration floors, keeps the full 29 whole-recording activity
features (they do not depend on bout detection), then runs the headline home
pipeline (Spearman Top-20 of Bout+Act + Demo(4), Ridge alpha=20, LOO CV).

Thresholds: 10, 30, 60, 120, 240 s (the 10 s floor is baked into the cached
pkl and step2 feature extraction; higher thresholds are obtained by post-
filtering the cached per-bout features).

Output:
  POMS/figures/fig_supp_bout_threshold_sensitivity.png
  results/paper_figures/fig_supp_bout_threshold_sensitivity.png
"""
from __future__ import annotations
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
FEATS = BASE / 'feats'
NPZ_DIR = BASE / 'home_full_recording_npz'
OUT = BASE / 'results' / 'paper_figures'; OUT.mkdir(parents=True, exist_ok=True)
POMS_FIGURES = BASE / 'POMS' / 'figures'; POMS_FIGURES.mkdir(parents=True, exist_ok=True)

THRESHOLDS_SEC = [10, 30, 60, 120, 240]
N_BOOT = 2000
RNG = np.random.default_rng(42)
FT2M = 0.3048

CLR_HEALTHY = '#6BAED6'
CLR_POMS    = '#FC8D62'
STAT_NAMES  = ('med', 'iqr', 'p10', 'p90', 'max', 'cv')


def impute(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.all(): X[:, j] = 0
        elif m.any(): X[m, j] = np.nanmedian(X[~m, j])
    return X


def aggregate_bout_feats(bout_feats: list[dict]) -> dict:
    """Rebuild the g_*_<stat> columns from the per-bout feature dicts —
    matches the aggregation used in home/step2_extract_features.py."""
    if not bout_feats:
        return {}
    names = sorted({k for bf in bout_feats for k in bf})
    arr = np.array([[bf.get(k, np.nan) for k in names] for bf in bout_feats])
    row: dict = {}
    for j, name in enumerate(names):
        col = arr[:, j]; valid = col[np.isfinite(col)]
        if len(valid) < 2:
            for st in STAT_NAMES:
                row[f'g_{name}_{st}'] = np.nan
            continue
        row[f'g_{name}_med'] = float(np.median(valid))
        row[f'g_{name}_iqr'] = float(np.percentile(valid, 75) - np.percentile(valid, 25))
        row[f'g_{name}_p10'] = float(np.percentile(valid, 10))
        row[f'g_{name}_p90'] = float(np.percentile(valid, 90))
        row[f'g_{name}_max'] = float(np.max(valid))
        row[f'g_{name}_cv']  = float(np.std(valid) / (np.mean(valid) + 1e-12))
    durs = [bf.get('duration_sec', 0) for bf in bout_feats]
    row['g_total_walk_sec'] = float(sum(durs))
    row['g_mean_bout_dur']  = float(np.mean(durs)) if durs else 0.0
    # g_bout_dur_cv removed — duplicate of g_duration_sec_cv
    return row


def loo_bout_act_top20_demo(X_accel, X_demo, y, K=20, alpha=20):
    """Home headline pipeline: Spearman Top-K inside LOO + Demo, Ridge."""
    n = len(y); n_accel = X_accel.shape[1]
    preds = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, dtype=bool); tr[i] = False
        corrs = [abs(spearmanr(X_accel[tr, j], y[tr])[0])
                 if np.std(X_accel[tr, j]) > 0 else 0
                 for j in range(n_accel)]
        top_k = sorted(range(n_accel), key=lambda j: corrs[j], reverse=True)[:K]
        X_tr = np.column_stack([X_accel[tr][:, top_k], X_demo[tr]])
        X_te = np.column_stack([X_accel[i:i+1][:, top_k], X_demo[i:i+1]])
        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X_tr), y[tr])
        preds[i] = m.predict(sc.transform(X_te))[0]
    return preds


def metrics_with_ci(y_ft, pr_ft, n_boot=N_BOOT, rng=RNG):
    y_m = y_ft * FT2M; pr_m = pr_ft * FT2M
    r2 = r2_score(y_ft, pr_ft)
    mae = mean_absolute_error(y_m, pr_m)
    r_v = pearsonr(y_ft, pr_ft)[0]
    n = len(y_ft)
    mae_b = np.empty(n_boot); r2_b = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mae_b[b] = mean_absolute_error(y_ft[idx] * FT2M, pr_ft[idx] * FT2M)
        try: r2_b[b] = r2_score(y_ft[idx], pr_ft[idx])
        except Exception: r2_b[b] = np.nan
    pct = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))
    return r2, pct(r2_b), mae, pct(mae_b), r_v


def main() -> None:
    subj = pd.read_csv(NPZ_DIR / '_subjects.csv')
    y = subj['sixmwd'].values.astype(float)

    with open(FEATS / 'home_walking_bouts.pkl', 'rb') as f:
        bout_feats_by_key = pickle.load(f)['bout_feats']

    pb = pd.read_csv(FEATS / 'home_perbout_features.csv')
    act_cols = [c for c in pb.columns if c.startswith('act_')]
    assert list(subj['key']) == list(pb['key'])
    X_act = impute(pb[act_cols].values.astype(float))

    demo = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo['cohort'] = demo['ID'].str.extract(r'^([A-Z])')[0]
    demo['subj_id'] = demo['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj.merge(demo, on=['cohort', 'subj_id'], how='left')
    p['cohort_POMS'] = (p['cohort'] == 'M').astype(int)
    for c in ['Age', 'Sex', 'BMI']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    X_demo = impute(p[['cohort_POMS', 'Age', 'Sex', 'BMI']].values.astype(float))

    print(f"Thresholds: {THRESHOLDS_SEC} s  |  N_BOOT={N_BOOT}")
    rows = []
    for t_sec in THRESHOLDS_SEC:
        per_subj_rows = []
        kept_counts = []
        for k in subj['key']:
            subj_bouts = [b for b in bout_feats_by_key.get(k, [])
                          if b.get('duration_sec', 0) >= t_sec]
            kept_counts.append(len(subj_bouts))
            per_subj_rows.append(aggregate_bout_feats(subj_bouts))
        agg = pd.DataFrame(per_subj_rows)

        # Union of gait columns across subjects, fill missing with NaN
        gait_cols = sorted([c for c in agg.columns if c.startswith('g_') and
                             not c.endswith(('_walk_sec', '_bout_dur', '_dur_cv'))])
        meta_cols = ['g_total_walk_sec', 'g_mean_bout_dur']
        for c in gait_cols + meta_cols:
            if c not in agg.columns:
                agg[c] = np.nan
        X_gait = impute(agg[gait_cols].values.astype(float))
        X_meta = impute(agg[meta_cols].values.astype(float))
        X_accel = np.column_stack([X_gait, X_meta, X_act])

        preds = loo_bout_act_top20_demo(X_accel, X_demo, y, K=20, alpha=20)
        r2, r2_ci, mae, mae_ci, r_v = metrics_with_ci(y, preds)

        n_zero = sum(1 for k in kept_counts if k == 0)
        rows.append({
            't_sec':   t_sec,
            'R2':      r2, 'R2_lo': r2_ci[0], 'R2_hi': r2_ci[1],
            'MAE':     mae, 'MAE_lo': mae_ci[0], 'MAE_hi': mae_ci[1],
            'r':       r_v,
            'mean_n_bouts_kept': float(np.mean(kept_counts)),
            'subj_with_zero_bouts': int(n_zero),
        })
        print(f"  t>={t_sec:>3d}s  R²={r2:+.3f} [{r2_ci[0]:+.2f},{r2_ci[1]:+.2f}]  "
              f"MAE={mae:5.1f} [{mae_ci[0]:4.1f},{mae_ci[1]:4.1f}] m  "
              f"mean_bouts={np.mean(kept_counts):6.0f}  zero={n_zero}/101")

    res = pd.DataFrame(rows)
    res.to_csv(BASE / 'results' / 'bout_threshold_sensitivity.csv', index=False)

    # ─────────────── Plot MAE vs threshold ───────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = res['t_sec'].values
    ax.errorbar(
        x, res['MAE'].values,
        yerr=[res['MAE'].values - res['MAE_lo'].values,
              res['MAE_hi'].values - res['MAE'].values],
        fmt='o-', color=CLR_POMS, markersize=9,
        linewidth=2.2, capsize=5, capthick=1.4,
        elinewidth=1.4, markerfacecolor=CLR_POMS,
        markeredgecolor='white', zorder=3,
    )
    for xi, row in zip(x, rows):
        ax.annotate(f"{row['MAE']:.1f}", xy=(xi, row['MAE']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(x); ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel('Minimum bout duration threshold (s, log scale)',
                   fontsize=12, fontweight='bold')
    ax.set_ylabel('6MWD Prediction MAE (m)',
                   fontsize=12, fontweight='bold')
    ax.set_title('Home 6MWD Prediction Sensitivity to Bout-Duration Threshold',
                  fontsize=13, fontweight='bold', pad=8)
    ymin = min(r['MAE_lo'] for r in rows) - 3
    ymax = max(r['MAE_hi'] for r in rows) + 8
    ax.set_ylim(ymin, ymax)
    # Note on the 10-s floor
    ax.text(0.02, 0.97,
            'Each point = LOO 6MWD prediction with\n'
            'Bout+Act-Top20+Demo (Ridge $\\alpha=20$).\n'
            'Bar = 95% percentile bootstrap CI (B=2000).',
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='0.7', alpha=0.9))

    plt.tight_layout()
    out_name = 'fig_supp_bout_threshold_sensitivity.png'
    fig.savefig(OUT / out_name, dpi=300, bbox_inches='tight')
    fig.savefig(POMS_FIGURES / out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_name} [POMS/ + results/]')


if __name__ == '__main__':
    main()
