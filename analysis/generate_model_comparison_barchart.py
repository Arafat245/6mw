#!/usr/bin/env python3
"""Bar chart of MAE (m) per model, Clinic and Home side-by-side per model.

Input : POMS/tables/model_comparison.csv (produced by model_comparison_table.py)
Output: POMS/figures/fig_model_comparison_mae.png
        results/paper_figures/fig_model_comparison_mae.png

Error-bar convention: 95% bootstrap CI half-width / 1.96 (i.e. σ-equivalent
under a normal approximation). Asymmetric about the point estimate is
preserved by averaging the two half-widths.
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
TABLES = BASE / 'POMS' / 'tables'
OUT_POMS = BASE / 'POMS' / 'figures'
OUT_RESULTS = BASE / 'results' / 'paper_figures'
OUT_POMS.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.mkdir(parents=True, exist_ok=True)

Z95 = 1.959964  # two-sided 95% z-score


def parse_mae_ci(cell: str) -> tuple[float, float]:
    """Parse strings like '31.2 [25.7, 37.7]' → (mean, sd_equivalent).

    sd_equivalent = ((upper - mean) + (mean - lower)) / 2 / Z95
                  = (upper - lower) / (2 * Z95)
    """
    m = re.match(r'\s*([-\d.]+)\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', cell)
    mean = float(m.group(1)); lo = float(m.group(2)); hi = float(m.group(3))
    sd = (hi - lo) / (2 * Z95)
    return mean, sd


def main() -> None:
    df = pd.read_csv(TABLES / 'model_comparison.csv')
    # Rename "Ridge (best)" → "Ridge" in the source table labels for display.
    df['Model'] = df['Model'].str.replace(r'\s*\(best\)\s*', '', regex=True)
    models = df['Model'].tolist()
    clinic = [parse_mae_ci(v) for v in df['Clinic MAE (m) [95% CI]']]
    home   = [parse_mae_ci(v) for v in df['Home MAE (m) [95% CI]']]
    c_m = np.array([c[0] for c in clinic]); c_sd = np.array([c[1] for c in clinic])
    h_m = np.array([c[0] for c in home]);   h_sd = np.array([c[1] for c in home])

    # Put Ridge first (it's the reported final model in both settings), then
    # order the remaining models by ascending clinic MAE.
    ridge_idx = next(i for i, m in enumerate(models) if m == 'Ridge')
    other_idx = [i for i in np.argsort(c_m) if i != ridge_idx]
    order = [ridge_idx, *other_idx]
    models_o = [models[i] for i in order]
    c_m_o = c_m[order]; c_sd_o = c_sd[order]
    h_m_o = h_m[order]; h_sd_o = h_sd[order]

    x = np.arange(len(models_o))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars_c = ax.bar(x - width/2, c_m_o, width, yerr=c_sd_o,
                    color='#1f77b4', edgecolor='black', linewidth=0.6,
                    capsize=4, label='Clinic', error_kw={'elinewidth': 1.2})
    bars_h = ax.bar(x + width/2, h_m_o, width, yerr=h_sd_o,
                    color='#d62728', edgecolor='black', linewidth=0.6,
                    capsize=4, label='Home', error_kw={'elinewidth': 1.2})

    # Value labels above the error bars
    for bars, vals, sds in [(bars_c, c_m_o, c_sd_o), (bars_h, h_m_o, h_sd_o)]:
        for b, v, s in zip(bars, vals, sds):
            ax.text(b.get_x() + b.get_width()/2, v + s + 1.2, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models_o, fontsize=11)
    ax.set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
    ax.set_title('6MWD Prediction Performance Comparison of Different Models',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, frameon=True)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(h_m_o.max() + h_sd_o.max(), c_m_o.max() + c_sd_o.max()) * 1.15)

    fig.tight_layout()
    for path in (OUT_POMS / 'fig_model_comparison_mae.png',
                 OUT_RESULTS / 'fig_model_comparison_mae.png'):
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_model_comparison_mae.png [POMS/ + results/]')


if __name__ == '__main__':
    main()
