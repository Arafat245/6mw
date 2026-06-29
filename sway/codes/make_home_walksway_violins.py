#!/usr/bin/env python3
"""Violin plot of the home (free-living) ML-sway features by cohort, mirroring the
clinic walksway_significant_violins.png. Per-subject values = median over ALL >=60s
sustained walking bouts (C0 aggregation, n=101: 38 POMS, 63 Control). Shows the
3 features significant at raw p<0.05; p_raw from sway/table/home_walksway_significant_features.csv
(reported uncorrected, following Brenton et al. 2022)."""
import warnings, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')
BASE = Path('/mnt/sdb/arafat/6mw')
D = pd.read_csv(BASE / 'feats/home_walksway_4features.csv')
S = pd.read_csv(BASE / 'sway/table/home_walksway_significant_features.csv').set_index('Feature')

CLR_CTRL = '#6BAED6'; CLR_POMS = '#FC8D62'; palette = {'Control': CLR_CTRL, 'POMS': CLR_POMS}
LBL, TIT, SUP, TICK = 13, 14, 15, 12
# (col, ylabel, display_name) -- the 3 BH-significant home features
feats = [('ml_over_vt',         'ML RMS / VT RMS',                          'ML_Over_VT'),
         ('ml_energy_frac',     r'var(ML) / $\Sigma$var(AP,ML,VT)',         'ML_Energy_Frac'),
         ('ml_spec_horiz_frac', r'P$_{ML}$ / (P$_{ML}$+P$_{AP}$)',          'ML_Spec_Frac')]

fig, axes = plt.subplots(1, len(feats), figsize=(5*len(feats), 6.2))
fig.suptitle('Free-Living Mediolateral-Sway Features by Cohort  (n=101: 38 POMS, 63 Control)',
             fontsize=SUP, fontweight='bold', y=0.99)

for ax, (col, ylab, disp) in zip(axes, feats):
    sub = D[['cohort', col]].copy()
    sns.violinplot(data=sub, x='cohort', y=col, order=['Control', 'POMS'], palette=palette,
                   inner=None, cut=0, linewidth=1.2, saturation=0.9, width=0.85, ax=ax)
    for coll in ax.collections: coll.set_alpha(0.55); coll.set_edgecolor('none')
    sns.stripplot(data=sub, x='cohort', y=col, order=['Control', 'POMS'], palette=palette,
                  size=4.2, jitter=0.18, alpha=0.9, edgecolor='white', linewidth=0.6, ax=ax)
    for i, coh in enumerate(['Control', 'POMS']):
        med = float(sub.loc[sub.cohort == coh, col].median())
        ax.plot([i-0.22, i+0.22], [med, med], color='black', linewidth=2.0, zorder=5)
    ax.set_xticklabels(['Control', 'POMS'], fontsize=TICK, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel(ylab, fontsize=LBL, fontweight='bold')
    ax.set_title(disp, fontsize=TIT, fontweight='bold', pad=8, family='monospace')
    ax.tick_params(axis='y', labelsize=TICK)
    ax.grid(True, axis='y', which='both', alpha=0.25, linestyle='--', linewidth=0.5); ax.set_axisbelow(True)
    q = float(S.loc[col, 'p_raw'])
    stars = '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else 'ns'
    ymin, ymax = sub[col].min(), sub[col].max(); rng = ymax - ymin
    ybar = ymax + rng*0.06; ytxt = ymax + rng*0.10
    ax.plot([0, 0, 1, 1], [ybar-rng*0.015, ybar, ybar, ybar-rng*0.015], color='black', linewidth=1.1)
    ax.text(0.5, ytxt, f'{stars}  (p = {q:.1e})', ha='center', va='bottom',
            fontsize=LBL-1, fontweight='bold')
    ax.set_ylim(top=ymax + rng*0.22)

plt.tight_layout(rect=[0, 0, 1, 0.97])
OUT = BASE / 'sway' / 'figures'; OUT.mkdir(parents=True, exist_ok=True)
fp = OUT / 'home_walksway_significant_violins.png'
plt.savefig(fp, dpi=200, bbox_inches='tight'); print('Saved', fp)
