"""
16_make_fig7_comparison.py — Figure 7: LSTM vs tuned Ridge comparison.

Three-panel figure across all 20 retained POD modes:
  (a) RMSE per mode for LSTM and tuned Ridge.
  (b) Relative RMSE difference (%) between the two models, with the
      mean difference marked by a dashed horizontal line.
  (c) RMSE versus mode variance, with mode labels M1-M6 highlighting
      the six leading modes.

Input : results/TLv2_rmse_lstm.npy, rmse_ridge_tuned.npy
        data/pod_variance_ratio.npy
Output: figures/fig7_comparison.png, fig7_comparison.pdf

Paper : Section 4.3; Figure 7
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config

mpl.rcParams.update({
    'font.family'    : 'DejaVu Serif',
    'font.size'      : 7,
    'axes.linewidth' : 0.5,
    'axes.labelsize' : 8,
    'axes.titlesize' : 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
})


def main():
    print("Figure 7: Comparison (paper-style)")
    rmse_lstm = np.load(config.TLV2_RMSE_LSTM)
    rmse_ridge = np.load(config.RIDGE_RMSE)
    var = np.load(config.POD_VAR)[:config.N_MODES]

    n = config.N_MODES
    modes = np.arange(1, n + 1)
    improve = (rmse_ridge - rmse_lstm) / rmse_ridge * 100
    mean_imp = improve.mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---------- Panel (a): RMSE per mode ----------
    ax1 = axes[0]
    x = np.arange(n); w = 0.35
    ax1.bar(x - w/2, rmse_lstm, w, label='LSTM (ours)',
            color='#C0392B', alpha=0.85, edgecolor='white', linewidth=0.4)
    ax1.bar(x + w/2, rmse_ridge, w, label='Ridge (tuned)',
            color='#2980B9', alpha=0.85, edgecolor='white', linewidth=0.4)
    ax1.set_xlabel('POD Mode Number', fontsize=10)
    ax1.set_ylabel('RMSE', fontsize=10)
    ax1.set_title('(a) RMSE per Mode', fontsize=11, pad=6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, fontsize=8)
    ax1.legend(fontsize=8, framealpha=0.85, loc='upper left')
    ax1.grid(axis='y', alpha=0.25, linewidth=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=7, length=2.5, width=0.4)

    # ---------- Panel (b): LSTM vs Ridge % ----------
    ax2 = axes[1]
    colors_imp = ['#27AE60' if v > 0 else '#E74C3C' for v in improve]
    ax2.bar(modes, improve, color=colors_imp, alpha=0.9,
            edgecolor='white', linewidth=0.4)
    ax2.axhline(mean_imp, color='black', linewidth=1.2, linestyle='--',
                label=f'Mean: {mean_imp:.1f}%')
    ax2.set_xlabel('POD Mode Number', fontsize=10)
    ax2.set_ylabel('RMSE Difference (%)', fontsize=10)
    ax2.set_title('(b) LSTM vs Tuned Ridge', fontsize=11, pad=6)
    ax2.legend(fontsize=8, framealpha=0.85, loc='lower right')
    ax2.grid(axis='y', alpha=0.25, linewidth=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=7, length=2.5, width=0.4)

    y_rng = max(improve) - min(improve)
    off = 0.04 * y_rng if y_rng > 0 else 0.1
    for m, v in zip(modes, improve):
        va = 'bottom' if v >= 0 else 'top'
        yt = v + off if v >= 0 else v - off
        ax2.text(m, yt, f'{v:.1f}%', ha='center', va=va,
                 fontsize=7, color='#1A5276')

    # ---------- Panel (c): Scatter RMSE vs Variance ----------
    ax3 = axes[2]
    var_pct = var * 100
    ax3.scatter(var_pct, rmse_lstm, color='#C0392B', s=60,
                zorder=3, label='LSTM', alpha=0.85)
    ax3.scatter(var_pct, rmse_ridge, color='#2980B9', s=60,
                zorder=3, label='Ridge (tuned)', alpha=0.85, marker='s')
    for i in range(min(6, n)):
        ax3.annotate(f'M{i+1}', (var_pct[i], rmse_lstm[i]),
                     fontsize=7, color='#C0392B',
                     xytext=(3, 3), textcoords='offset points')
    ax3.set_xlabel('Mode Variance (%)', fontsize=10)
    ax3.set_ylabel('RMSE', fontsize=10)
    ax3.set_title('(c) RMSE vs Mode Variance', fontsize=11, pad=6)
    ax3.legend(fontsize=8, framealpha=0.85)
    ax3.grid(alpha=0.25, linewidth=0.4)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(labelsize=7, length=2.5, width=0.4)

    fig.suptitle('Model Comparison: LSTM POD-ROM vs Validation-Tuned Ridge\n'
                 'North Sea and Baltic Sea \u2014 Test Period 2018-2024',
                 fontsize=11, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig7_comparison.pdf'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig7_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/fig7_comparison.png")


if __name__ == "__main__":
    main()
