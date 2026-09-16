"""15_make_fig6_predictions.py - Figure 6 (matching paper)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config

mpl.rcParams.update({
    'font.family'    : 'DejaVu Serif',
    'font.size'      : 8,
    'axes.linewidth' : 0.6,
    'axes.labelsize' : 8,
    'axes.titlesize' : 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
})


def main():
    print("Figure 6: Predictions (paper-style)")
    true_vals = np.load(config.TRUE_TEST)
    pred_lstm = np.load(config.TLV2_PRED_LSTM)
    pred_ridge = np.load(config.RIDGE_PRED)
    rmse_lstm = np.load(config.TLV2_RMSE_LSTM)
    rmse_ridge = np.load(config.RIDGE_RMSE)

    colors = {
        'true':  '#2C3E50',   # dark blue-gray
        'lstm':  '#C0392B',   # red
        'ridge': '#2980B9',   # blue
    }
    t = np.arange(len(true_vals))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for idx, mode in enumerate(range(6)):
        ax = axes[idx // 2, idx % 2]

        ax.plot(t, true_vals[:, mode], color=colors['true'],
                linewidth=1.6, label='Observed', zorder=3)
        ax.plot(t, pred_lstm[:, mode], color=colors['lstm'],
                linewidth=1.4, linestyle='--', label='LSTM', zorder=2)
        ax.plot(t, pred_ridge[:, mode], color=colors['ridge'],
                linewidth=1.2, linestyle=':', label='Ridge (tuned)',
                alpha=0.85, zorder=1)

        ax.set_title(
            f'({chr(97+idx)}) Mode {mode+1} | RMSE: '
            f'LSTM={rmse_lstm[mode]:.3f}  Ridge={rmse_ridge[mode]:.3f}',
            fontsize=10, pad=5
        )
        ax.set_xlabel('Test Time Step (months)', fontsize=9)
        ax.set_ylabel(f'$a^{{({mode+1})}}(t)$', fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8, length=3, width=0.5)

        # Legend only on panel (a)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9,
                      edgecolor='#CCCCCC')

    fig.suptitle('POD Coefficient Prediction: LSTM vs Tuned Ridge\n'
                 'Test Period (2018\u20132024)', fontsize=12, y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig6_predictions.pdf'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig6_predictions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/fig6_predictions.png")


if __name__ == "__main__":
    main()
