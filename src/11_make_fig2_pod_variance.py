"""
11_make_fig2_pod_variance.py — Figure 2: POD variance spectrum.

Two-panel figure:
  (a) Variance per mode, first 60 modes; six leading modes highlighted.
  (b) Cumulative variance with reference lines at 50%, 75%, 90%, and an
      annotation at the 20-mode cutoff (79.6% of total variance).

Input : data/pod_variance_ratio.npy
Output: figures/fig2_pod_variance.png, fig2_pod_variance.pdf

Paper : Section 4.1; Figure 2
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config


def main():
    print("Figure 2: POD variance")
    var_ratio = np.load(config.POD_VAR)
    cum_var = np.cumsum(var_ratio) * 100
    var_pct = var_ratio * 100

    DISPLAY_MODES = 60
    var_plot = var_pct[:DISPLAY_MODES]
    cum_plot = cum_var[:DISPLAY_MODES]
    modes = np.arange(1, DISPLAY_MODES + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel a
    ax1 = axes[0]
    bars = ax1.bar(modes, var_plot, color='steelblue',
                   edgecolor='white', linewidth=0.4, alpha=0.85)
    for i in range(min(6, DISPLAY_MODES)):
        bars[i].set_color('#C0392B')
        bars[i].set_alpha(0.9)
    ax1.set_xlabel('POD Mode Number', fontsize=11)
    ax1.set_ylabel('Explained Variance (%)', fontsize=11)
    ax1.set_title('(a) Variance per Mode', fontsize=12, pad=8)
    ax1.set_xlim(0.5, DISPLAY_MODES + 0.5)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for i in range(min(6, DISPLAY_MODES)):
        ax1.text(i + 1, var_plot[i] * 0.5, f'{var_plot[i]:.1f}%',
                 ha='center', va='center', fontsize=7.5,
                 color='black', fontweight='bold', rotation=90)

    # Panel b
    ax2 = axes[1]
    ax2.plot(modes, cum_plot, color='steelblue', linewidth=2,
             marker='o', markersize=3, markerfacecolor='white',
             markeredgewidth=1)
    thresholds = [(50, '#E74C3C', '--', '50% variance'),
                  (75, '#E67E22', '--', '75% variance'),
                  (90, '#27AE60', '-.', '90% variance')]
    for th, col, ls, lbl in thresholds:
        idx = int(np.argmax(cum_var >= th))
        ax2.axhline(th, color=col, linewidth=1, linestyle=ls,
                    alpha=0.7, label=lbl)
        if idx + 1 <= DISPLAY_MODES:
            ax2.axvline(idx + 1, color=col, linewidth=0.8,
                        linestyle=ls, alpha=0.5)
            ax2.annotate(f'{idx+1} modes', xy=(idx+1, th),
                         xytext=(idx+3, th-6), fontsize=8, color=col,
                         arrowprops=dict(arrowstyle='->', color=col, lw=0.8))
    ax2.set_xlabel('POD Mode Number', fontsize=11)
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=11)
    ax2.set_title('(b) Cumulative Variance', fontsize=12, pad=8)
    ax2.set_xlim(0.5, DISPLAY_MODES + 0.5)
    ax2.set_ylim(0, 102)
    ax2.grid(alpha=0.3, linewidth=0.5)
    ax2.legend(fontsize=9, loc='lower right', framealpha=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    if DISPLAY_MODES >= 20:
        ax2.axvline(20, color='gray', linewidth=1.2, linestyle=':')
        ax2.annotate(f'20 modes\n({cum_var[19]:.1f}%)',
                     xy=(20, cum_var[19]),
                     xytext=(25, cum_var[19]-10), fontsize=8, color='gray',
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    fig.suptitle('POD Variance Distribution of the Training Period (1979-2010)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig2_pod_variance.pdf'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig2_pod_variance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/fig2_pod_variance.png")


if __name__ == "__main__":
    main()
