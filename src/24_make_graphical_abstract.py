"""
24_make_graphical_abstract.py — Graphical abstract for Elsevier.

Layout:
  Top    — large map of the study region showing July 2020 SST anomaly
  Bottom — 4-stage pipeline (ERA5 → POD → LSTM → Evaluation)
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import config

mpl.rcParams.update({
    'font.family'    : 'DejaVu Serif',
    'font.size'      : 11,
    'axes.linewidth' : 0.8,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
})


def main():
    print("Building Graphical Abstract (final v2)")

    fig = plt.figure(figsize=(12, 7.5), facecolor='white')

    # ==================================================================
    # TOP: large map of the study region (Jul 2020 heatwave)
    # ==================================================================
    ax_map = fig.add_axes([0.03, 0.32, 0.94, 0.60],
                          projection=ccrs.PlateCarree())
    ax_map.set_extent([-5, 30, 50, 65], crs=ccrs.PlateCarree())

    # Load SST anomaly (Jul 2020)
    ds = xr.open_dataset(config.ERA5_SST)
    sst = ds['sst'].values - 273.15
    train_end = config.TRAIN_END
    train_3d = sst[:train_end].reshape(train_end // 12, 12, 61, 141)
    clim = train_3d.mean(axis=0)

    # July 2020 = index 498
    idx = 498
    m = idx % 12
    anom = sst[idx] - clim[m]

    lats = ds.latitude.values
    lons = ds.longitude.values

    # Coastline / features
    ax_map.add_feature(cfeature.LAND, facecolor='#E8E8E8',
                       zorder=2, edgecolor='#888888', linewidth=0.4)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7,
                       edgecolor='#2C3E50', zorder=3)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3,
                       linestyle=':', alpha=0.5, zorder=3)

    # SST anomaly
    im = ax_map.pcolormesh(
        lons, lats, anom, cmap='RdBu_r',
        vmin=-3, vmax=3,
        transform=ccrs.PlateCarree(),
        shading='auto', zorder=1,
    )

    # Coastline over the data
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6,
                       edgecolor='#2C3E50', zorder=4)

    # Colorbar inside the map
    cbar = plt.colorbar(
        im, ax=ax_map, orientation='horizontal',
        pad=0.02, shrink=0.6, aspect=30, fraction=0.04,
    )
    cbar.ax.tick_params(labelsize=9, pad=1, length=3, width=0.5)
    cbar.set_label('SST anomaly (°C)', fontsize=10, labelpad=2)
    cbar.outline.set_linewidth(0.4)

    # Title inside map
    ax_map.set_title(
        'July 2020 European marine heatwave (ERA5 observed anomaly)',
        fontsize=12, pad=6, fontweight='bold', color='#2C3E50',
    )

    # Axis labels (subtle)
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.25,
                          color='gray', alpha=0.35, linestyle='--')
    gl.xlocator = mpl.ticker.FixedLocator([0, 10, 20])
    gl.ylocator = mpl.ticker.FixedLocator([52, 56, 60, 64])
    for lon in [0, 10, 20]:
        ax_map.text(lon, 49.5, f'{lon}°E', fontsize=8,
                    ha='center', va='top', color='#333333',
                    transform=ccrs.PlateCarree())
    for lat in [52, 56, 60, 64]:
        ax_map.text(-6.5, lat, f'{lat}°N', fontsize=8,
                    ha='right', va='center', color='#333333',
                    transform=ccrs.PlateCarree())

    # ==================================================================
    # BOTTOM: 4-stage pipeline
    # ==================================================================
    ax_pipe = fig.add_axes([0.03, 0.02, 0.94, 0.24])
    ax_pipe.set_xlim(0, 10)
    ax_pipe.set_ylim(0, 2.0)
    ax_pipe.axis('off')

    stages = [
        ('ERA5 Reanalysis',
         '5 coupled surface\nfields (1979-2024)',
         '#34495E'),
        ('POD',
         '20 modes\n79.6% variance',
         '#2980B9'),
        ('POD-LSTM',
         '24-month input\n→ 1-month forecast',
         '#C0392B'),
        ('Evaluation',
         'RMSE 0.878\nvs Ridge / ARIMA',
         '#8E44AD'),
    ]

    n = len(stages)
    box_w = 2.0
    box_h = 1.55
    gap = (10 - n * box_w) / (n + 1)

    for i, (title, sub, color) in enumerate(stages):
        x = gap + i * (box_w + gap)
        y = 0.15

        # Outer box
        rect = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle='round,pad=0.04,rounding_size=0.15',
            edgecolor=color, facecolor='white',
            linewidth=2.2, zorder=2,
        )
        ax_pipe.add_patch(rect)

        # Colored strip
        strip = FancyBboxPatch(
            (x, y + box_h - 0.45), box_w, 0.45,
            boxstyle='round,pad=0.02,rounding_size=0.10',
            edgecolor=color, facecolor=color,
            linewidth=0, zorder=3,
        )
        ax_pipe.add_patch(strip)

        # Title inside strip
        ax_pipe.text(x + box_w/2, y + box_h - 0.22, title,
                     ha='center', va='center', fontsize=12,
                     fontweight='bold', color='white', zorder=4)

        # Subtitle
        ax_pipe.text(x + box_w/2, y + box_h - 0.75, sub,
                     ha='center', va='top', fontsize=10.5,
                     color='#2C3E50', zorder=4)

        # Arrow to next
        if i < n - 1:
            ax_pipe.add_patch(FancyArrowPatch(
                (x + box_w + 0.04, y + box_h/2),
                (x + box_w + gap - 0.04, y + box_h/2),
                arrowstyle='-|>', mutation_scale=22,
                linewidth=2.5, color='#7F8C8D', zorder=5))

    # Main title above pipeline
    fig.text(0.5, 0.285,
             'Multivariate POD-LSTM for ocean surface prediction',
             ha='center', va='bottom', fontsize=15,
             fontweight='bold', color='#2C3E50')

    # Citation at bottom
    fig.text(0.5, 0.005,
             'Rezaee, Ardestani, Pourroostaei Ardakani (2026)  ·  Computers & Geosciences',
             ha='center', va='bottom', fontsize=10,
             color='#7F8C8D', style='italic')

    plt.savefig(os.path.join(config.FIGURES_DIR, 'graphical_abstract.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.10,
                facecolor='white')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'graphical_abstract.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.10,
                facecolor='white')
    plt.close()
    print("Saved: figures/graphical_abstract.png")


if __name__ == "__main__":
    main()