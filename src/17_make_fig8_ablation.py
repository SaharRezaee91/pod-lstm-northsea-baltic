"""17_make_fig8_ablation.py - Figure 8 (paper-style, forced)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import config


def main():
    print("Figure 8: Ablation study")

    # Force rcParams AFTER config import
    mpl.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 8,
        'axes.linewidth': 0.5,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6.5,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    lstm_modes = np.load(config.ABLATION_LSTM_MODES)
    gru_modes = np.load(config.ABLATION_GRU_MODES)
    uv_modes = np.load(config.ABLATION_UV_MODES)
    lstm_seeds = np.load(config.ABLATION_LSTM_SEEDS)
    gru_seeds = np.load(config.ABLATION_GRU_SEEDS)
    uv_seeds = np.load(config.ABLATION_UV_SEEDS)

    t_stat, p_val = stats.ttest_rel(lstm_seeds, gru_seeds)

    modes = np.arange(1, config.N_MODES + 1)
    colors = {'LSTM': '#C0392B', 'GRU': '#2980B9', 'UV': '#27AE60'}

    # Wider aspect but same font size
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    fig.suptitle('Ablation Study: Multivariate vs. Univariate Embedding '
                 'and Recurrent Architecture', fontsize=10, y=1.02)

    # ---------- Panel (a) ----------
    ax1 = axes[0]
    ax1.plot(modes, lstm_modes, 'o-', color=colors['LSTM'],
             label='LSTM Multivariate', markersize=2.5,
             markerfacecolor='white', markeredgewidth=1.0,
             linewidth=1.0)
    ax1.plot(modes, gru_modes, 's--', color=colors['GRU'],
             label='GRU Multivariate', markersize=2.5,
             markerfacecolor='white', markeredgewidth=1.0,
             linewidth=1.0)
    ax1.plot(modes, uv_modes, '^:', color=colors['UV'],
             label='LSTM Univariate', markersize=2.5,
             markerfacecolor='white', markeredgewidth=1.0,
             linewidth=1.0)
    ax1.set_xlabel('POD Mode Number', fontsize=8)
    ax1.set_ylabel('RMSE', fontsize=8)
    ax1.set_title('(a) Mode-by-mode Comparison', fontsize=9, pad=5)
    ax1.legend(fontsize=6, framealpha=0.9, edgecolor='gray',
               loc='upper left', handlelength=2.0, borderpad=0.3,
               labelspacing=0.3)
    ax1.grid(alpha=0.2, linewidth=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(length=2.5, width=0.4, labelsize=7)

    # ---------- Panel (b) ----------
    ax2 = axes[1]
    means = [lstm_modes.mean(), gru_modes.mean(), uv_modes.mean()]
    x_pos = np.arange(3)
    bars = ax2.bar(x_pos, means,
                   color=[colors['LSTM'], colors['GRU'], colors['UV']],
                   alpha=0.85, width=0.55)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.008, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['LSTM\nMultiv.', 'GRU\nMultiv.', 'LSTM\nUniv.'])
    ax2.set_ylabel('Mean Test RMSE', fontsize=8)
    ax2.set_title('(b) 10-Seed Mean Performance', fontsize=9, pad=5)
    ax2.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(length=2.5, width=0.4, labelsize=7)
    ax2.set_ylim(0, 1.0)

    # ---------- Panel (c) ----------
    ax3 = axes[2]
    means_r = [lstm_seeds.mean(), gru_seeds.mean(), uv_seeds.mean()]
    stds_r  = [lstm_seeds.std(),  gru_seeds.std(),  uv_seeds.std()]
    x_pos = np.arange(3)
    ax3.bar(x_pos, means_r, yerr=stds_r,
            color=[colors['LSTM'], colors['GRU'], colors['UV']],
            alpha=0.85, width=0.55, capsize=2.5,
            error_kw={'linewidth': 0.7})
    p_str = f"{p_val:.3f}" if p_val < 0.05 else f"{p_val:.2f}"
    for i, (m, s) in enumerate(zip(means_r, stds_r)):
        ax3.text(i, m + s + 0.010, f'{m:.3f}\u00b1{s:.3f}',
                 ha='center', va='bottom', fontsize=6)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['LSTM\nMultiv.', 'GRU\nMultiv.', 'LSTM\nUniv.'])
    ax3.set_ylabel('Mean Test RMSE', fontsize=8)
    ax3.set_title(f'(c) 10-Seed Robustness (p={p_str})', fontsize=9, pad=5)
    ax3.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(length=2.5, width=0.4, labelsize=7)
    ax3.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig8_ablation.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig8_ablation.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Paired t-test LSTM vs GRU: p={p_val:.4f}")
    print("Saved: figures/fig8_ablation.png")


if __name__ == "__main__":
    main()
