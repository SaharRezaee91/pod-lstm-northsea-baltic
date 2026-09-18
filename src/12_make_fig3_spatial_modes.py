"""
12_make_fig3_spatial_modes.py — Figure 3: spatial POD modes (SST block).

Six-panel map showing the SST component of the six leading POD modes.
The POD basis itself is built from the joint five-variable state vector;
only the SST block of Vt is shown here because it admits the most direct
physical interpretation.

Input : data/pod_Vt_sst.npy, data/pod_variance_ratio.npy
        data/era5_processed_1979_2024.nc  (for the geographic grid)
Output: figures/fig3_spatial_modes.png, fig3_spatial_modes.pdf

Paper : Section 4.1; Figure 3
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import config


def main():
    print("Figure 3: Spatial modes")

    mpl.rcParams.update({
        'font.family'    : 'DejaVu Serif',
        'font.size'      : 10,
        'axes.linewidth' : 0.8,
        'savefig.dpi'    : 300,
        'savefig.bbox'   : 'tight',
    })

    ds = xr.open_dataset(config.ERA5_FULL)
    Vt_sst = np.load(config.POD_SST_VT)
    var = np.load(config.POD_VAR)

    lats = ds.latitude.values
    lons = ds.longitude.values
    n_lat = len(lats)
    n_lon = len(lons)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.0),
                             subplot_kw={'projection': proj})
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        ax.set_extent([-5, 30, 50, 65], crs=proj)

        ax.add_feature(cfeature.LAND, facecolor='#DDDDDD', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3,
                       linestyle='--', alpha=0.6, zorder=3)

        gl = ax.gridlines(draw_labels=False,
                          linewidth=0.3, color='gray',
                          alpha=0.5, linestyle='--')
        gl.xlocator = mpl.ticker.FixedLocator([0, 5, 10, 15, 20, 25])
        gl.ylocator = mpl.ticker.FixedLocator([52, 54, 56, 58, 60, 62, 64])

        for lon_tick in [0, 5, 10, 15, 20, 25]:
            ax.text(lon_tick, 49.4, f'{lon_tick}\u00b0E',
                    transform=proj, fontsize=7.5,
                    ha='center', va='top', color='#333333',
                    clip_on=False, zorder=10)

        for lat_tick in [52, 54, 56, 58, 60, 62, 64]:
            ax.text(-5.7, lat_tick, f'{lat_tick}\u00b0N',
                    transform=proj, fontsize=7.5,
                    ha='right', va='center', color='#333333',
                    clip_on=False, zorder=10)

        mode_sst = Vt_sst[i].reshape(n_lat, n_lon)
        vmax = np.nanpercentile(np.abs(mode_sst), 98)
        im = ax.pcolormesh(lons, lats, mode_sst,
                           cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax,
                           transform=proj, zorder=1)

        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          pad=0.08, shrink=0.82, aspect=28, fraction=0.04)
        cb.ax.tick_params(labelsize=7, length=2, width=0.5)
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor('black')

        ax.set_title(f'({chr(97+i)}) Mode {i+1}  '
                     f'[{var[i]*100:.1f}% variance]', fontsize=10, pad=4)

    fig.suptitle('Spatial POD Modes (SST Component)\n'
                 'North Sea and Baltic Sea (1979\u20132024)',
                 fontsize=11.5, y=1.00)

    # top REDUCED to bring row 1 down closer to suptitle
    plt.subplots_adjust(top=0.94, bottom=0.10, left=0.05, right=0.99,
                        wspace=0.10, hspace=0.35)

    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig3_spatial_modes.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig3_spatial_modes.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print("Saved: figures/fig3_spatial_modes.png")


if __name__ == "__main__":
    main()
