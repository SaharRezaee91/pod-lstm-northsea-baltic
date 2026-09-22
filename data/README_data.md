# Data directory

This directory contains the preprocessed data required to reproduce the results in the paper without re-downloading the full ERA5 record (~10 GB).

## State matrices (standardized)

- `X_train.npy` — training period (Jan 1979 – Dec 2010), shape (384, 17525). Each row is one month; columns are the concatenated fields of the five variables (5 × 3505 active ocean points).
- `X_val.npy` — validation period (2011-2017), shape (84, 17525).
- `X_test.npy` — test period (2018-2024), shape (84, 17525).
- `X_full.npy` — full record (1979-2024), shape (552, 17525).

## Per-variable statistics (computed from the training period only)

- `sst_mean.npy`, `sst_std.npy`
- `slp_mean.npy`, `slp_std.npy`
- `heat_mean.npy`, `heat_std.npy`
- `wind_mean.npy`, `wind_std.npy`
- `pe_mean.npy`, `pe_std.npy`

## POD outputs

- `pod_U.npy` (552, 384) — temporal singular vectors
- `pod_S.npy` (384,) — singular values
- `pod_Vt.npy` (384, 43005) — spatial modes on the full 8601-point grid. Columns corresponding to non-ocean grid points are zero (25324 zero columns out of 43005). Non-zero columns: 17681.
- `pod_variance_ratio.npy` (384,) — fraction of variance per mode. The first 20 modes cumulatively explain 79.6% of the variance, matching the value reported in the paper.
- `pod_A_train.npy` (384, 384) — standardized POD coefficients for the training period.
- `pod_A_val.npy` (84, 384) — standardized POD coefficients for validation.
- `pod_A_test.npy` (84, 384) — standardized POD coefficients for test.
- `pod_coeffs_mean.npy` (20,) — mean of the first 20 coefficients (from training).
- `pod_coeffs_std.npy` (20,) — std of the first 20 coefficients (from training).
- `pod_Vt_sst.npy` (20, 8601) — SST block of the first 20 POD modes.

## Derived convenience files

- `V20.npy` (20, 43005) — first 20 rows of `pod_Vt.npy`.
- `scaler_stats.npz` — combined per-variable mean/std arrays, for convenience.
- `lat.npy`, `lon.npy` — latitude and longitude grids (61 × 141).
- `ocean_mask.npy` (8601,) — boolean mask of ocean grid points.

## Important note on dimensionality

The multivariate state matrix `X_train` has 17525 columns (5 × 3505 active ocean points). The full spatial basis `pod_Vt.npy` has 43005 columns (5 × 8601 grid points), where non-ocean points carry zero loadings. Both representations are consistent: the reduced-order model operates on the first 20 POD coefficients, and the variance explained by these modes matches the values reported in the paper (51.3% at 3 modes, 75.2% at 14 modes, 79.6% at 20 modes).

## Regenerating the data from scratch

    python src/01_download_era5_data.py
    python src/02_preprocess_era5_data.py
    python src/03_pod_decomposition.py

Requires CDS API credentials (`~/.cdsapirc`) and ~10 GB of free disk space.
