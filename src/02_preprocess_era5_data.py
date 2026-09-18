"""
02_preprocess_era5_data.py — Preprocess ERA5 into a standardized state matrix.

Loads the five coupled surface fields, builds the ocean mask from the
training-period SST field, applies a train-only z-score per grid point
(no information from validation or test leaks into normalization), and
concatenates the fields into a single state matrix X.

Input : data/era5_sst_1979_2024.nc
        data/era5_slp_1979_2024.nc
        data/era5_heat_flux_1979_2024.nc
        data/era5_wind_1979_2024.nc
        data/era5_precip_evap_1979_2024.nc
Output: data/ocean_mask.npy
        data/X_train.npy  (384, 17525)
        data/X_val.npy    ( 84, 17525)
        data/X_test.npy   ( 84, 17525)

Paper : Section 3.2 (Data preprocessing)
"""
import os
import numpy as np
import xarray as xr
import config

# Paths to raw ERA5 files (from original project)
RAW_DIR = '/Users/sahar/ocean_pod_project/data'


def load_raw_variables():
    """Load 5 raw variables from separate files."""
    print("Loading raw ERA5 variables...")

    sst_ds = xr.open_dataset(os.path.join(RAW_DIR, 'era5_sst_1979_2024.nc'))
    slp_ds = xr.open_dataset(os.path.join(RAW_DIR, 'era5_slp_1979_2024.nc'))
    heat_ds = xr.open_dataset(os.path.join(RAW_DIR, 'era5_heat_flux_1979_2024.nc'))
    wind_ds = xr.open_dataset(os.path.join(RAW_DIR, 'era5_wind_1979_2024.nc'))
    pe_ds = xr.open_dataset(os.path.join(RAW_DIR, 'era5_precip_evap_1979_2024.nc'))

    n_time = sst_ds.sizes['valid_time']
    n_lat = sst_ds.sizes['latitude']
    n_lon = sst_ds.sizes['longitude']
    n_space = n_lat * n_lon

    print(f"  Grid: {n_lat}x{n_lon}={n_space}")
    print(f"  Time: {n_time} months")

    # ---- sst: K -> C ----
    sst = (sst_ds['sst'].values - 273.15).reshape(n_time, -1)

    # ---- msl: already in Pa ----
    msl = slp_ds['msl'].values.reshape(n_time, -1)

    # ---- heat flux: sshf + slhf ----
    heat = (heat_ds['sshf'].values + heat_ds['slhf'].values).reshape(n_time, -1)

    # ---- wind speed: sqrt(u10^2 + v10^2) ----
    u10 = wind_ds['u10'].values.reshape(n_time, -1)
    v10 = wind_ds['v10'].values.reshape(n_time, -1)
    ws = np.sqrt(u10**2 + v10**2)

    # ---- net precip: tp - e ----
    tp = pe_ds['tp'].values.reshape(n_time, -1)
    e = pe_ds['e'].values.reshape(n_time, -1)
    net_precip = tp - e

    fields = {
        'sst': sst,
        'msl': msl,
        'heat': heat,
        'wind': ws,
        'pe': net_precip,
    }

    return fields, n_lat, n_lon, n_space, n_time


def build_ocean_mask(sst, n_space):
    """Build mask from training SST (finite + ocean-like values)."""
    sst_tr = sst[:config.TRAIN_END]
    valid = np.mean(
        np.isfinite(sst_tr) & (sst_tr > -5) & (sst_tr < 40), axis=0
    )
    mask = valid > 0.95
    print(f"  Ocean mask: {mask.sum()}/{n_space} points")
    return mask


def standardize_train_locked(data, train_end, eps=1e-10):
    """Z-score using TRAIN-only stats."""
    tr = data[:train_end]
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd < eps] = 1.0
    return (data - mu) / sd, mu, sd


def main():
    print("=" * 60)
    print("Preprocessing ERA5 (from raw files)")
    print("=" * 60)

    fields, n_lat, n_lon, n_space, n_time = load_raw_variables()

    # Ocean mask from training SST
    mask = build_ocean_mask(fields['sst'], n_space)
    np.save(os.path.join(config.DATA_DIR, 'ocean_mask.npy'), mask)

    # Save variable names
    order = ['sst', 'msl', 'heat', 'wind', 'pe']
    np.save(os.path.join(config.DATA_DIR, 'var_names.npy'), np.array(order))

    # Standardize each field using train stats
    X_all = []
    for name in order:
        d = fields[name][:, mask]
        d_std, mu, sd = standardize_train_locked(d, config.TRAIN_END)
        X_all.append(d_std)
        np.save(os.path.join(config.DATA_DIR, f'{name}_mean.npy'), mu)
        np.save(os.path.join(config.DATA_DIR, f'{name}_std.npy'), sd)
        print(f"  {name}: {d_std.shape}, "
              f"std_before={d.std():.4f}, std_after={d_std[:config.TRAIN_END].std():.4f}")

    # Concatenate
    X = np.concatenate(X_all, axis=1)
    print(f"\nFull X: {X.shape}")

    X_train = X[:config.TRAIN_END]
    X_val = X[config.TRAIN_END:config.VAL_END]
    X_test = X[config.VAL_END:]

    print(f"  Train: {X_train.shape}")
    print(f"  Val  : {X_val.shape}")
    print(f"  Test : {X_test.shape}")

    np.save(os.path.join(config.DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(config.DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.DATA_DIR, 'X_full.npy'), X)

    print("\nDone.")


if __name__ == "__main__":
    main()
