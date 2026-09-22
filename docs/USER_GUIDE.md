# User Guide

This guide describes the inputs, outputs, and options of each script in the `src/` directory.

## Pipeline order

Scripts are numbered in execution order. Run them from inside `src/`:

    01_download_era5_data.py
    02_preprocess_era5_data.py
    03_pod_decomposition.py
    04_train_pod_lstm.py
    05_train_baselines.py
    06_ablation_multiseed.py
    10_make_fig1_study_area.py ... 18_make_fig9_reconstruction.py
    20_make_table3.py ... 23_make_fig_a1_sensitivity.py

## 01_download_era5_data.py

Downloads ERA5 monthly reanalysis fields from the Copernicus Climate Data Store (CDS).

- Inputs: CDS API credentials in `~/.cdsapirc`
- Outputs: raw NetCDF files in `data/`
- Runtime: 1-2 hours, ~10 GB

## 02_preprocess_era5_data.py

Preprocesses raw ERA5 fields and saves standardized state matrices.

- Inputs: raw NetCDF files in `data/era5_processed_1979_2024.nc`
- Outputs: `X_train.npy` (384, 17525), `X_val.npy` (84, 17525), `X_test.npy` (84, 17525), `X_full.npy` (552, 17525), `ocean_mask.npy` (8601,)
- Runtime: < 1 minute
- Key detail: statistics and ocean mask are computed from the training period only (1979-2010), following Section 3.2 of the paper.

## 03_pod_decomposition.py

Performs economy SVD on the training-period state matrix.

- Inputs: `X_train.npy` (384, 17525)
- Outputs:
  - `pod_U.npy` (552, 384) — temporal singular vectors
  - `pod_S.npy` (384,) — singular values
  - `pod_Vt.npy` (384, 43005) — spatial modes on the full 8601-point grid
  - `pod_variance_ratio.npy` (384,) — variance fraction per mode
  - `pod_A_train.npy` (384, 384) — standardized coefficients, training
  - `pod_A_val.npy` (84, 384) — standardized coefficients, validation
  - `pod_A_test.npy` (84, 384) — standardized coefficients, test
  - `pod_coeffs_mean.npy` (20,) — coefficient means (from training)
  - `pod_coeffs_std.npy` (20,) — coefficient stds (from training)
  - `pod_Vt_sst.npy` (20, 8601) — SST block of first 20 modes
- Runtime: < 10 seconds

## 04_train_pod_lstm.py

Trains the multivariate POD-LSTM.

- Inputs: `pod_A_train.npy`, `pod_A_val.npy`
- Outputs: `results/TLv2_pred_lstm.npy`, `results/TLv2_rmse_lstm.npy`, `results/TLv2_mae_lstm.npy`, `results/TLv2_train_losses.npy`, `results/TLv2_val_losses.npy`, `results/true_test_v2.npy`
- Runtime: ~5 seconds on CPU (300 epochs)

## 05_train_baselines.py

Trains Ridge and ARIMA baselines.

- Inputs: `pod_A_train.npy`, `pod_A_val.npy`, `pod_A_test.npy`
- Outputs: `results/pred_ridge_tuned.npy`, `results/TLv2_pred_arima.npy`, and the corresponding RMSE/MAE/R2 arrays
- Runtime: 2-5 minutes

## 06_ablation_multiseed.py

Runs the ablation study (GRU, univariate LSTM, Transformer) across 10 random seeds.

- Inputs: `pod_A_train.npy`, `pod_A_val.npy`, `pod_A_test.npy`
- Outputs: `results/fig8_*_FINAL.npy`
- Runtime: 3-5 minutes

## Figure scripts (10-18)

Each script reproduces one figure from the paper. All read from `data/` and `results/` and write to `figures/`. Runtime < 1 minute each.

## Table scripts (20-23)

Each script reproduces one table or significance test from the paper. Runtime < 1 minute each.

## Environment

See `requirements.txt` or `environment.yml` at the repository root.
