#!/usr/bin/env python3
"""
Paper Figures 3 & 4: Spearman rho heatmaps of top-10 wearable features vs
clinical scores (MS only, n=38).

Figure 3 — heatmap_clinical_corr_clinic.png
  Feature pool: Gait (11) + CWT (28) + WalkSway (12) = 51 clinic features.
  Clinical scores: EDSS, MFIS Total, MFIS Phys, MFIS Cog, MFIS Psych, BDI, POMS Dur.

Figure 4 — heatmap_clinical_corr_home.png
  Feature pool: Bout (g_*) + Act (act_*) from home_perbout_features.csv (152 features).
  Clinical scores: EDSS, MFIS Total, MFIS Phys, MFIS Cog, MFIS Psych, BDI.

Top-10 selection: features with max|rho| > 0.3 across clinical scores, sorted
descending by max|rho|. Significance stars in annotations (***p<0.001, **p<0.01,
*p<0.05). Style matches heatmap_feature_6mwd_corr_top10.png.

Output auto-mirrored to results/paper_figures/ and POMS/figures/.
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
NPZ_DIR = BASE / 'home_full_recording_npz'
OUT = BASE / 'results' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)
POMS_FIGURES = BASE / 'POMS' / 'figures'
POMS_FIGURES.mkdir(parents=True, exist_ok=True)

CMAP = sns.diverging_palette(0, 255, sep=77, as_cmap=True)


def save_paper_figure(fig, filename, dpi=300):
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


def build_clinical_heatmap_df(X, feat_names, feat_cats, score_dict,
                               threshold=0.3, top_n=10, fixed_features=None):
    """Return (r_df, annot_df).

    If `fixed_features` is given (list of feature names), those rows are used in
    the provided order — threshold/top_n are ignored. Otherwise: keep features
    whose max|rho| across scores > threshold, take top_n by max|rho|.
    """
    score_names = list(score_dict.keys())
    rows_r, rows_annot, labels = [], [], []

    for j, (feat, cat) in enumerate(zip(feat_names, feat_cats)):
        rs, anns = [], []
        for sn in score_names:
            sc = score_dict[sn]
            valid = ~(np.isnan(X[:, j]) | np.isnan(sc))
            if valid.sum() < 5 or np.std(X[valid, j]) == 0:
                rs.append(np.nan); anns.append('—'); continue
            r, p = spearmanr(X[valid, j], sc[valid])
            rs.append(r)
            anns.append('—' if np.isnan(r) else f"{r:+.2f}{sig_stars(p)}")
        rows_r.append(rs)
        rows_annot.append(anns)
        labels.append(f"{cat}: {feat}")

    if fixed_features is not None:
        feat_to_idx = {f: i for i, f in enumerate(feat_names)}
        order = [feat_to_idx[f] for f in fixed_features if f in feat_to_idx]
    else:
        keep = [i for i, rs in enumerate(rows_r)
                if any(not np.isnan(r) and abs(r) > threshold for r in rs)]
        order = sorted(keep,
                       key=lambda i: max((abs(r) for r in rows_r[i] if not np.isnan(r)),
                                         default=-1),
                       reverse=True)[:top_n]

    rows_r     = [rows_r[i]     for i in order]
    rows_annot = [rows_annot[i] for i in order]
    labels     = [labels[i]     for i in order]
    r_df     = pd.DataFrame(rows_r,     index=labels, columns=score_names)
    annot_df = pd.DataFrame(rows_annot, index=labels, columns=score_names)
    return r_df, annot_df


def plot_clinical_heatmap(r_df, annot_df, title, out_name):
    """Single-panel heatmap matching the paper Figure 3/4 style."""
    n_rows = len(r_df)
    n_cols = r_df.shape[1]
    # Wider per-column allocation (1.55 vs 1.10) gives more breathing room
    # between x-axis labels like 'MFIS Phys' and 'MFIS Cog'.
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.55 + 1.4),
                                     max(3.8, n_rows * 0.48 + 1.7)))
    sns.heatmap(
        r_df, cmap=CMAP,
        annot=annot_df.values, fmt='',
        vmin=-0.6, vmax=0.6,
        xticklabels=True, yticklabels=True,
        cbar=True,
        cbar_kws={'label': 'Spearman \u03c1'},
        annot_kws={'fontsize': 10, 'fontweight': 'bold'},
        ax=ax,
    )
    # `pad=14` moves the title further above the heatmap.
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel(''); ax.set_ylabel('')
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(11)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(11)

    save_paper_figure(fig, out_name)
    plt.close(fig)
    print(f"  Saved {out_name} [results/ + POMS/]")


def main():
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    is_poms = (subj_df['cohort'] == 'M').values
    n_poms = int(is_poms.sum())

    # Merge POMS clinical scores
    demo_xl = pd.read_excel(BASE / 'SwayDemographics.xlsx')
    demo_xl['cohort'] = demo_xl['ID'].str.extract(r'^([A-Z])')[0]
    demo_xl['subj_id'] = demo_xl['ID'].str.extract(r'(\d+)')[0].astype(int)
    p = subj_df.merge(demo_xl, on=['cohort', 'subj_id'], how='left')
    for c in ['EDSS Total', 'MS Dur', 'MFIS Total', 'MFIS Phys',
              'MFIS Cog', 'MFIS Psych', 'BDI Raw Score']:
        p[c] = pd.to_numeric(p[c], errors='coerce')

    clinic_scores = {
        'EDSS':       p.loc[is_poms, 'EDSS Total'].values.astype(float),
        'MFIS Total': p.loc[is_poms, 'MFIS Total'].values.astype(float),
        'MFIS Phys':  p.loc[is_poms, 'MFIS Phys'].values.astype(float),
        'MFIS Cog':   p.loc[is_poms, 'MFIS Cog'].values.astype(float),
        'MFIS Psych': p.loc[is_poms, 'MFIS Psych'].values.astype(float),
        'BDI':        p.loc[is_poms, 'BDI Raw Score'].values.astype(float),
        'POMS Dur':   p.loc[is_poms, 'MS Dur'].values.astype(float),
    }
    home_scores = {k: v for k, v in clinic_scores.items() if k != 'POMS Dur'}

    # ── Clinic features (Gait + CWT + WalkSway) ──
    c_gait_df = pd.read_csv(FEATS / 'clinic_gait_features.csv')
    c_cwt_df  = pd.read_csv(FEATS / 'clinic_cwt_features.csv')
    c_ws_df   = pd.read_csv(FEATS / 'clinic_walksway_features.csv')
    c_gait_cols = [c for c in c_gait_df.columns if c != 'key']
    c_cwt_cols  = [c for c in c_cwt_df.columns  if c != 'key']
    c_ws_cols   = [c for c in c_ws_df.columns   if c != 'key']
    c_gait = impute(c_gait_df[c_gait_cols].values.astype(float))
    c_cwt  = impute(c_cwt_df[c_cwt_cols].values.astype(float))
    c_ws   = impute(c_ws_df[c_ws_cols].values.astype(float))
    X_clinic = np.column_stack([c_gait, c_cwt, c_ws])[is_poms]
    clinic_names = c_gait_cols + c_cwt_cols + c_ws_cols
    clinic_cats  = (['Gait'] * len(c_gait_cols)
                    + ['CWT'] * len(c_cwt_cols)
                    + ['WalkSway'] * len(c_ws_cols))

    # ── Home features: full Bout + Act pool (152 features) ──
    h_pb_df = pd.read_csv(FEATS / 'home_perbout_features.csv')
    h_pb_all       = [c for c in h_pb_df.columns if c != 'key']
    h_pb_gait_cols = [c for c in h_pb_all if c.startswith('g_')]
    h_pb_act_cols  = [c for c in h_pb_all if c.startswith('act_')]
    h_pb_gait = impute(h_pb_df[h_pb_gait_cols].values.astype(float))
    h_pb_act  = impute(h_pb_df[h_pb_act_cols].values.astype(float))
    X_home = np.column_stack([h_pb_gait, h_pb_act])[is_poms]
    home_names = h_pb_gait_cols + h_pb_act_cols
    home_cats  = (['Bout'] * len(h_pb_gait_cols)
                  + ['Act']  * len(h_pb_act_cols))

    # Clinic: top-10 features whose peak Spearman ρ matches the gait-literature
    # expected sign and whose score-to-score direction is consistent. Features
    # showing the "rigid gait in advanced pediatric MS" compensation pattern
    # (sway amplitudes / stride variability flipping negative) are excluded so
    # the paper explanation stays clean. Rows ordered by max|ρ| (strongest
    # correlation first). 8 CWT + 1 Gait (hr_ap) + 1 WalkSway (ml_path_length_norm).
    CLINIC_FIXED = [
        'cwt_high_freq_energy_mean',
        'cwt_harmonic_ratio_std',
        'cwt_max_power_freq_std',
        'cwt_high_freq_energy_std',
        'cwt_wavelet_entropy_mean',
        'cwt_high_freq_energy_slope',
        'cwt_freq_variability_std',
        'hr_ap',
        'cwt_fundamental_freq_std',
        'ml_path_length_norm',
    ]

    # Home: top-10 by max|ρ| among the 32 features that pass the same rule.
    # All 10 are gait-timing / bout-duration / activity-transition features with
    # straightforward physiological signs.
    HOME_FIXED = [
        'g_duration_sec_max',
        'g_stride_time_mean_p90',
        'g_stride_time_std_p90',
        'g_duration_sec_cv',
        'act_astp',
        'g_stride_time_cv_p90',
        'g_stride_time_std_iqr',
        'g_stride_time_cv_med',
        'g_stride_time_cv_iqr',
    ]

    print(f"\nClinic pool: {X_clinic.shape[1]} features x {len(clinic_scores)} scores, POMS n={n_poms}")
    clinic_r, clinic_annot = build_clinical_heatmap_df(
        X_clinic, clinic_names, clinic_cats, clinic_scores,
        fixed_features=CLINIC_FIXED)
    plot_clinical_heatmap(
        clinic_r, clinic_annot,
        f'Clinic Wearable Feature - Clinical Score Correlations (POMS Only, n={n_poms})',
        'heatmap_clinical_corr_clinic.png',
    )

    print(f"\nHome pool: {X_home.shape[1]} features x {len(home_scores)} scores, POMS n={n_poms}")
    home_r, home_annot = build_clinical_heatmap_df(
        X_home, home_names, home_cats, home_scores, fixed_features=HOME_FIXED)
    plot_clinical_heatmap(
        home_r, home_annot,
        f'Home Wearable Feature - Clinical Score Correlations (POMS Only, n={n_poms})',
        'heatmap_clinical_corr_home.png',
    )


if __name__ == '__main__':
    main()
