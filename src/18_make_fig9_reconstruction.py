"""
18_make_fig9_reconstruction.py — Figure 9: spatial reconstruction of SST.

Three-row, three-column figure over three representative test months
(Jan 2018, Jul 2020, Dec 2024):

  Row 1 (Observed anomaly):
      Reconstruction from the observed POD coefficients (true_test_v2).
  Row 2 (LSTM POD-ROM Reconstruction):
      Reconstruction from the LSTM-predicted coefficients.
  Row 3 (Error = LSTM - Observed):
      Residual field, displayed on the +/- 0.3 normalized-unit scale
      used in the paper.

All three rows are in the same normalized coefficient space, so the
error is directly comparable.

Input : data/pod_Vt.npy (SST block = first 8601 columns)
        results/true_test_v2.npy, TLv2_pred_lstm.npy
        data/pod_coeffs_mean.npy, pod_coeffs_std.npy
Output: figures/fig9_reconstruction.png, fig9_reconstruction.pdf

Paper : Section 4.4; Figure 9
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import config


TIME_STEPS = [0, 30, 83]
LABELS = ['Jan 2018', 'Jul 2020', 'Dec 2024']
SST_LIMIT = 4.0
ERROR_LIMIT = 0.3


def reconstruct(coeff_norm, Vt_sst, coeffs_mean, coeffs_std):
    coeff = coeff_norm * coeffs_std[:config.N_MODES] + coeffs_mean[:config.N_MODES]
    field = coeff @ Vt_sst
    return field.reshape(61, 141)


def main():
    print("Figure 9: SST reconstruction (FINAL)")

    mpl.rcParams.update({
        'font.family'    : 'DejaVu Serif',
        'font.size'      : 9,
        'axes.linewidth' : 0.6,
        'savefig.dpi'    : 300,
        'savefig.bbox'   : 'tight',
    })

    ds = xr.open_dataset(config.ERA5_FULL)
    lats = ds.latitude.values
    lons = ds.longitude.values

    Vt = np.load(config.POD_VT)
    Vt_sst = Vt[:config.N_MODES, :8601]

    true_vals = np.load(config.TRUE_TEST)
    pred_lstm = np.load(config.TLV2_PRED_LSTM)
    coeffs_mean = np.load(config.POD_MEAN)
    coeffs_std = np.load(config.POD_STD)

    obs  = [reconstruct(true_vals[t], Vt_sst, coeffs_mean, coeffs_std) for t in TIME_STEPS]
    lstm = [reconstruct(pred_lstm[t], Vt_sst, coeffs_mean, coeffs_std) for t in TIME_STEPS]
    err  = [lstm[i] - obs[i] for i in range(3)]

    # ============================================================
    # Layout: 3 map rows + 3 THIN colorbar rows
    # ============================================================
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(13.5, 9.5))

    # 6 rows: [maps, cbar, maps, cbar, maps, cbar]
    # Colorbar rows now MUCH thinner (0.05 instead of 0.12)
    gs = fig.add_gridspec(
        6, 3,
        height_ratios=[1.0, 0.05, 1.0, 0.05, 1.0, 0.05],
        left=0.09, right=0.99,
        top=0.92, bottom=0.05,
        wspace=0.03, hspace=0.10,
    )

    map_axes = [[fig.add_subplot(gs[r, c], projection=proj)
                 for c in range(3)]
                for r in (0, 2, 4)]

    cbar_axes = [fig.add_subplot(gs[r, :]) for r in (1, 3, 5)]

    row_titles = ['Observed anomaly',
                  'LSTM POD-ROM Reconstruction',
                  'Error (LSTM \u2212 Observed)']

    # ---------- Draw maps ----------
    ims = [[None]*3 for _ in range(3)]

    for row in range(3):
        for col in range(3):
            ax = map_axes[row][col]
            ax.set_extent([config.LON_MIN, config.LON_MAX,
                           config.LAT_MIN, config.LAT_MAX], crs=proj)
            ax.add_feature(cfeature.LAND, facecolor='#EEEEEE', zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)

            gl = ax.gridlines(draw_labels=False,
                              linewidth=0.25, color='gray',
                              alpha=0.35, linestyle='--')
            gl.xlocator = mpl.ticker.FixedLocator([0, 5, 10, 15, 20, 25])
            gl.ylocator = mpl.ticker.FixedLocator([52, 54, 56, 58, 60, 62, 64])

            # ---------- Latitude labels: on ALL panels in col 0 ----------
            if col == 0:
                for lat_tick in [52, 54, 56, 58, 60, 62, 64]:
                    ax.text(-6.0, lat_tick, f'{lat_tick}\u00b0N',
                            transform=proj, fontsize=7,
                            ha='right', va='center', color='#333333',
                            clip_on=False, zorder=10)
            # ---------- Longitude labels: on ALL panels in row 2 ----------
            if row == 2:
                for lon_tick in [0, 5, 10, 15, 20, 25]:
                    ax.text(lon_tick, 49.0, f'{lon_tick}\u00b0E',
                            transform=proj, fontsize=7,
                            ha='center', va='top', color='#333333',
                            clip_on=False, zorder=10)

            if row == 0:
                field, vmin, vmax = obs[col], -SST_LIMIT, SST_LIMIT
            elif row == 1:
                field, vmin, vmax = lstm[col], -SST_LIMIT, SST_LIMIT
            else:
                field, vmin, vmax = err[col], -ERROR_LIMIT, ERROR_LIMIT

            im = ax.pcolormesh(lons, lats, field, cmap='RdBu_r',
                               vmin=vmin, vmax=vmax,
                               transform=proj, zorder=1)
            ims[row][col] = im

            if col == 0:
                ax.text(-0.16, 0.5, row_titles[row],
                        transform=ax.transAxes, rotation=90,
                        va='center', ha='right',
                        fontsize=9.5, fontweight='bold')

            if row == 0:
                ax.set_title(LABELS[col], fontsize=10, pad=3,
                             fontweight='bold')

    # ---------- Colorbars in dedicated rows (thin) ----------
    cb1 = fig.colorbar(ims[0][1], cax=cbar_axes[0], orientation='horizontal')
    cb1.ax.tick_params(labelsize=7, pad=1, length=2, width=0.4)
    cb1.set_label('SST Anomaly (\u00b0C)', fontsize=8, labelpad=1)
    cb1.outline.set_linewidth(0.4)

    cb2 = fig.colorbar(ims[1][1], cax=cbar_axes[1], orientation='horizontal')
    cb2.ax.tick_params(labelsize=7, pad=1, length=2, width=0.4)
    cb2.set_label('SST Anomaly (\u00b0C)', fontsize=8, labelpad=1)
    cb2.outline.set_linewidth(0.4)

    cb3 = fig.colorbar(ims[2][1], cax=cbar_axes[2], orientation='horizontal')
    cb3.ax.tick_params(labelsize=7, pad=1, length=2, width=0.4)
    cb3.set_label('SST Anomaly Error (\u00b0C)', fontsize=8, labelpad=1)
    cb3.outline.set_linewidth(0.4)

    # Shrink colorbars horizontally so they align with middle panel
    for cb in (cb1, cb2, cb3):
        pos = cb.ax.get_position()
        new_width = pos.width * 0.55
        new_x = pos.x0 + (pos.width - new_width) / 2
        cb.ax.set_position([new_x, pos.y0, new_width, pos.height])

    # ---------- suptitle ----------
    fig.suptitle('Spatial Reconstruction of SST Anomalies using LSTM POD-ROM\n'
                 'Observed vs LSTM Reconstruction '
                 '(North Sea and Baltic Sea, Test Period 2018-2024)',
                 fontsize=10.5, y=0.97)

    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig9_reconstruction.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig9_reconstruction.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print("Saved: figures/fig9_reconstruction.png")


if __name__ == "__main__":
    main()
