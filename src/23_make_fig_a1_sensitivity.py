"""
23_make_fig_a1_sensitivity.py — Figure A1: hyperparameter sensitivity.

Two-panel figure documenting the two key design choices:
  (a) Best validation loss vs. input sequence length (12, 24, 36 months).
      The 24-month window gives the lowest validation loss.
  (b) Best validation loss vs. number of retained POD modes
      (5, 10, 15, 20, 25, 30). Beyond 20 modes, the higher-dimensional
      output mapping becomes harder to learn from the 384-month record,
      so 20 modes is chosen as the balance point.

Values are taken from the corresponding validation runs of the main
pipeline (see Appendix A of the paper).

Input : none (data hardcoded from Appendix A)
Output: figures/figA1_sensitivity.png, figA1_sensitivity.pdf

Paper : Appendix A; Figure A1
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config

mpl.rcParams.update({
    'font.family'    : 'DejaVu Serif',
    'font.size'      : 10,
    'axes.linewidth' : 0.8,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
})


def main():
    print("Figure A1: Sensitivity analysis (paper-exact)")

    # ============================================================
    # Data from paper (Section 3.4 / Appendix A)
    # ============================================================
    seq_lengths = [12, 24, 36]
    seq_losses  = [0.7498, 0.7490, 0.7498]

    n_modes_list = [5, 10, 15, 20, 25, 30]
    mode_losses  = [0.635, 0.700, 0.750, 0.750, 0.775, 0.805]

    # ============================================================
    # Plot — taller figure (was 3.5, now 4.2)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---------- Panel (a): Sequence Length ----------
    ax1 = axes[0]
    ax1.plot(seq_lengths, seq_losses, 'o-',
             color='#1A5276', linewidth=1.4,
             markersize=7, markerfacecolor='white',
             markeredgewidth=1.5, zorder=3)

    ax1.plot(24, 0.7490, 's', color='#C0392B',
             markersize=12, zorder=4,
             label='Chosen configuration (24 months)')

    ax1.set_xlabel('Input Sequence Length (months)', fontsize=10)
    ax1.set_ylabel('Best Validation Loss (MSE)', fontsize=10)
    ax1.set_title('(a) Sensitivity to Sequence Length',
                  fontsize=11, pad=6)
    ax1.grid(alpha=0.3, linewidth=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=9, length=3, width=0.5)
    ax1.set_xticks(seq_lengths)
    ax1.legend(fontsize=8, loc='upper center', framealpha=0.9,
               edgecolor='#CCCCCC')

    # ---------- Panel (b): Number of Modes ----------
    ax2 = axes[1]
    ax2.plot(n_modes_list, mode_losses, 'o-',
             color='#1A5276', linewidth=1.4,
             markersize=7, markerfacecolor='white',
             markeredgewidth=1.5, zorder=3)

    ax2.plot(20, 0.750, 's', color='#C0392B',
             markersize=12, zorder=4,
             label='Chosen configuration (20 modes)')

    ax2.set_xlabel('Number of Retained POD Modes', fontsize=10)
    ax2.set_ylabel('Best Validation Loss (MSE)', fontsize=10)
    ax2.set_title('(b) Sensitivity to Number of Modes',
                  fontsize=11, pad=6)
    ax2.grid(alpha=0.3, linewidth=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=9, length=3, width=0.5)
    ax2.set_xticks(n_modes_list)
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.9,
               edgecolor='#CCCCCC')

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'figA1_sensitivity.pdf'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'figA1_sensitivity.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/figA1_sensitivity.png")


if __name__ == "__main__":
    main()
