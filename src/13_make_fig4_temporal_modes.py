"""13_make_fig4_temporal_modes.py - Figure 4: Temporal coefficients."""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import config


def main():
    print("Figure 4: Temporal modes")
    ds = xr.open_dataset(config.ERA5_FULL)
    U = np.load(config.POD_U)
    S = np.load(config.POD_S)
    var = np.load(config.POD_VAR)
    coeffs = U * S[np.newaxis, :]
    time = ds.valid_time.values

    colors = ['#C0392B', '#2980B9', '#27AE60',
              '#8E44AD', '#E67E22', '#16A085']
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)

    warm_periods = [
        ('1982-07', '1983-06'), ('1991-07', '1992-06'),
        ('1997-07', '1998-06'), ('2009-07', '2010-06'),
        ('2015-07', '2016-06'), ('2023-07', '2024-06'),
    ]

    for i, ax in enumerate(axes):
        a = coeffs[:, i]
        ax.plot(time, a, color=colors[i], linewidth=1.2, zorder=3)
        ax.axhline(0, color='gray', linewidth=0.5,
                   linestyle='--', alpha=0.6)
        for j, (start, end) in enumerate(warm_periods):
            ax.axvspan(np.datetime64(start), np.datetime64(end),
                       alpha=0.10, color='#E74C3C',
                       label='Warm event' if (i == 0 and j == 0) else '')
        ax.set_ylabel(f'$a^{{({i+1})}}(t)$', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.01, 0.85, f'Mode {i+1} — {var[i]*100:.1f}% variance',
                transform=ax.transAxes, fontsize=10,
                color=colors[i], fontweight='bold')
        if i == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.7)

    axes[-1].set_xlabel('Time', fontsize=11)
    fig.suptitle('Temporal POD Coefficients (6 Leading Modes)\n'
                 'North Sea and Baltic Sea (1979-2024)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig4_temporal_modes.pdf'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig4_temporal_modes.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/fig4_temporal_modes.png")


if __name__ == "__main__":
    main()
