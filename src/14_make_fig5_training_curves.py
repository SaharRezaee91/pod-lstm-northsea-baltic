"""
14_make_fig5_training_curves.py — Figure 5: LSTM learning curves.

Two-panel figure of training and validation loss over 300 epochs:
  (a) Training MSE, decreasing monotonically from ~1.0 to ~0.88.
  (b) Validation MSE, evaluated every 10 epochs. The dashed vertical
      line marks the best checkpoint (lowest validation loss), which
      is the state retained for all downstream evaluations.

Input : results/TLv2_train_losses.npy, TLv2_val_losses.npy
Output: figures/fig5_training_curves.png, fig5_training_curves.pdf

Paper : Section 4.2; Figure 5
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config

# === OVERRIDE global rcParams IMMEDIATELY (after config import) ===
mpl.rcParams.update({
    'font.family'    : 'DejaVu Serif',
    'font.size'      : 8,
    'axes.linewidth' : 0.6,
    'axes.labelsize' : 8,
    'axes.titlesize' : 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
})


def main():
    print("Figure 5: Training curves")

    train_loss = np.load(config.TLV2_TRAIN_LOSS)
    val_loss = np.load(config.TLV2_VAL_LOSS)

    epochs = np.arange(1, len(train_loss) + 1)
    val_epochs = np.arange(10, len(train_loss) + 1, 10)

    best_idx = int(np.argmin(val_loss))
    best_epoch = val_epochs[best_idx]
    best_val = val_loss[best_idx]

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 2.8))

    # ---- Panel (a) ----
    ax[0].plot(epochs, train_loss, color='lightsteelblue',
               linewidth=0.7, label='Train Loss')
    ax[0].set_title('(a) Training Loss', fontsize=9, pad=5)
    ax[0].set_xlabel('Epoch', fontsize=8)
    ax[0].set_ylabel('MSE Loss', fontsize=8)
    ax[0].grid(alpha=0.25, linewidth=0.4)
    ax[0].legend(fontsize=6, loc='upper right', framealpha=0.85,
                 borderpad=0.3, handlelength=1.8)
    ax[0].tick_params(length=3, width=0.5)

    # ---- Panel (b) ----
    ax[1].plot(val_epochs, val_loss, color='firebrick',
               marker='o', markersize=2, linewidth=0.7,
               label='Validation Loss')
    ax[1].axvline(best_epoch, color='seagreen',
                  linestyle='--', linewidth=0.8,
                  label=f'Best validation checkpoint (epoch {best_epoch})')
    ax[1].set_title('(b) Validation Loss', fontsize=9, pad=5)
    ax[1].set_xlabel('Epoch', fontsize=8)
    ax[1].set_ylabel('MSE Loss', fontsize=8)
    ax[1].grid(alpha=0.25, linewidth=0.4)
    ax[1].legend(fontsize=6, loc='upper right', framealpha=0.85,
                 borderpad=0.3, handlelength=1.8)
    ax[1].tick_params(length=3, width=0.5)

    # suptitle: two lines, close together
    fig.suptitle('LSTM Training \u2014 Learning Curves\n'
                 f'N_modes={config.N_MODES} | SEQ_LEN={config.SEQ_LEN}',
                 fontsize=9, y=0.99)

    plt.subplots_adjust(top=0.78, bottom=0.15, left=0.09, right=0.98,
                        wspace=0.30)
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig5_training_curves.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig5_training_curves.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"Best epoch: {best_epoch}, Best val loss: {best_val:.4f}")
    print("Saved: figures/fig5_training_curves.png")


if __name__ == "__main__":
    main()
