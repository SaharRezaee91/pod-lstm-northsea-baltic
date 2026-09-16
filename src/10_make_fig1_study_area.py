"""10_make_fig1_study_area.py - Figure 1 (exact copy of your code)."""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import config

os.makedirs(config.FIGURES_DIR, exist_ok=True)

mpl.rcParams.update({
    'font.family'    : 'DejaVu Serif',
    'font.size'      : 10,
    'axes.linewidth' : 0.8,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
})

# load SST
ds  = xr.open_dataset(config.ERA5_SST)
sst = ds['sst'] - 273.15

lats = sst.latitude.values
lons = sst.longitude.values

sst_mean = sst.mean(dim='valid_time')
sst_std  = sst.std(dim='valid_time')
sst_anom = sst.isel(valid_time=-1) - sst_mean

proj = ccrs.PlateCarree()

fig, axes = plt.subplots(
    1, 3, figsize=(15, 5),
    subplot_kw={'projection': proj}
)

datasets = [sst_mean, sst_std, sst_anom]
titles   = [
    '(a) Long-term Mean SST (\u00b0C)',
    '(b) Standard Deviation (\u00b0C)',
    '(c) Anomaly \u2014 Dec 2024 (\u00b0C)',
]
cmaps = ['RdYlBu_r', 'YlOrRd', 'RdBu_r']

for ax, data, title, cmap in zip(axes, datasets, titles, cmaps):
    ax.set_extent([-5, 30, 50, 65], crs=proj)

    ax.add_feature(cfeature.LAND, facecolor='#DDDDDD', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4,
                   linestyle='--', alpha=0.6, zorder=3)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.4, zorder=3)

    # Gridlines without labels (cartopy bug workaround)
    gl = ax.gridlines(draw_labels=False,
                      linewidth=0.4, color='gray',
                      alpha=0.5, linestyle='--')
    gl.xlocator = mpl.ticker.FixedLocator([0, 5, 10, 15, 20, 25])
    gl.ylocator = mpl.ticker.FixedLocator([50, 52, 54, 56, 58, 60, 62, 64])

    # Manual labels (same style as gridliner would produce)
    for lon_tick in [0, 5, 10, 15, 20, 25]:
        ax.text(lon_tick, 49.4, f'{lon_tick}\u00b0E',
                transform=proj, fontsize=8,
                ha='center', va='top', color='#333333', clip_on=False)
    for lat_tick in [50, 52, 54, 56, 58, 60, 62, 64]:
        ax.text(-5.5, lat_tick, f'{lat_tick}\u00b0N',
                transform=proj, fontsize=8,
                ha='right', va='center', color='#333333', clip_on=False)

    vals = data.values
    vmax = np.nanpercentile(np.abs(vals), 98)
    if cmap == 'RdBu_r':
        vmin, vmax = -vmax, vmax
    else:
        vmin = np.nanpercentile(vals, 2)

    im = ax.pcolormesh(
        lons, lats, vals,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        transform=proj, zorder=1
    )

    cb = plt.colorbar(im, ax=ax,
                      orientation='horizontal',
                      pad=0.06, shrink=0.88, aspect=25)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, pad=6)

fig.suptitle(
    'Study Area: North Sea and Baltic Sea\n'
    'ERA5 Reanalysis Sea Surface Temperature (1979-2024)',
    fontsize=12, y=0.75
)

plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'fig1_study_area.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(config.FIGURES_DIR, 'fig1_study_area.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved successfully")
print("Path: figures/fig1_study_area.png")
