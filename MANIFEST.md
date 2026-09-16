# MANIFEST — File → Paper element mapping

This document maps every script and output file to the corresponding
element in the paper.

---

## Paper: A Multivariate POD-LSTM Reduced-Order Model for Ocean Surface
## Dynamics in the North Sea and Baltic Sea
### Rezaee, S., Ardestani, M., Pourroostaei Ardakani, S.
### Computers & Geosciences (submitted)

---

## 1. Source Code (`src/`)

### 1.1 Data preparation

| Script | Purpose | Input | Output |
|---|---|---|---|
| `01_download_era5_data.py` | Download ERA5 monthly means from CDS | CDS API | `data/era5_*.nc` |
| `02_preprocess_era5_data.py` | Standardize + ocean mask | `data/era5_*.nc` | `data/X_train.npy`, `X_val.npy`, `X_test.npy` |
| `03_pod_decomposition.py` | SVD on training period only | `data/X_train.npy` | `data/pod_*.npy` |

### 1.2 Model training

| Script | Purpose | Output |
|---|---|---|
| `04_train_pod_lstm.py` | Main POD-LSTM (Table 2) | `results/TLv2_pred_lstm.npy` |
| `05_train_baselines.py` | Ridge (alpha=3000) + ARIMA(2,0,1) | `results/pred_ridge_tuned.npy`, `TLv2_pred_arima.npy` |
| `06_ablation_multiseed.py` | GRU / univariate / 10 seeds | `results/fig8_*_FINAL.npy` |

### 1.3 Figures

| Script | Paper element | Output |
|---|---|---|
| `10_make_fig1_study_area.py` | Figure 1 | `figures/fig1_study_area.png` |
| `11_make_fig2_pod_variance.py` | Figure 2 | `figures/fig2_pod_variance.png` |
| `12_make_fig3_spatial_modes.py` | Figure 3 | `figures/fig3_spatial_modes.png` |
| `13_make_fig4_temporal_modes.py` | Figure 4 | `figures/fig4_temporal_modes.png` |
| `14_make_fig5_training_curves.py` | Figure 5 | `figures/fig5_training_curves.png` |
| `15_make_fig6_predictions.py` | Figure 6 | `figures/fig6_predictions.png` |
| `16_make_fig7_comparison.py` | Figure 7 | `figures/fig7_comparison.png` |
| `17_make_fig8_ablation.py` | Figure 8 | `figures/fig8_ablation.png` |
| `18_make_fig9_reconstruction.py` | Figure 9 | `figures/fig9_reconstruction.png` |
| `23_make_fig_a1_sensitivity.py` | Figure A1 | `figures/figA1_sensitivity.png` |

### 1.4 Tables and statistics

| Script | Paper element | Output |
|---|---|---|
| `20_make_table3.py` | Table 3 | `results/Table3.xlsx` |
| `21_make_table_a2.py` | Tables A1, A2 | `results/Tables_A1_A2.xlsx` |
| `22_significance_tests.py` | Wilcoxon + Diebold-Mariano | `results/significance_tests.txt` |

---

## 2. Key Result Files (`results/`)

| File | Shape | Description |
|---|---|---|
| `TLv2_pred_lstm.npy` | (84, 20) | LSTM test predictions |
| `TLv2_rmse_lstm.npy` | (20,) | LSTM per-mode RMSE |
| `TLv2_mae_lstm.npy` | (20,) | LSTM per-mode MAE |
| `TLv2_train_losses.npy` | (300,) | Training loss curve |
| `TLv2_val_losses.npy` | (30,) | Validation loss curve |
| `pred_ridge_tuned.npy` | (84, 20) | Ridge test predictions |
| `rmse_ridge_tuned.npy` | (20,) | Ridge per-mode RMSE |
| `TLv2_pred_arima.npy` | (84, 20) | ARIMA test predictions |
| `TLv2_rmse_arima.npy` | (20,) | ARIMA per-mode RMSE |
| `true_test_v2.npy` | (84, 20) | Ground truth (observed coefficients) |
| `fig8_lstm_modes_FINAL.npy` | (20,) | LSTM per-mode mean RMSE (10 seeds) |
| `fig8_gru_modes_FINAL.npy` | (20,) | GRU per-mode mean RMSE |
| `fig8_uv_modes_FINAL.npy` | (20,) | Univariate per-mode mean RMSE |
| `fig8_lstm_seeds_FINAL.npy` | (10,) | LSTM per-seed mean RMSE |
| `fig8_gru_seeds_FINAL.npy` | (10,) | GRU per-seed mean RMSE |
| `fig8_uv_seeds_FINAL.npy` | (10,) | Univariate per-seed mean RMSE |
| `r2_lstm_FINAL.npy` | (20,) | R² per mode (LSTM) |
| `pearson_lstm_FINAL.npy` | (20,) | Pearson r per mode (LSTM) |

---

## 3. POD Files (`data/`)

| File | Shape | Description |
|---|---|---|
| `pod_A_train.npy` | (384, 384) | Training-period POD coefficients (raw) |
| `pod_A_val.npy` | (84, 384) | Validation-period coefficients |
| `pod_A_test.npy` | (84, 384) | Test-period coefficients |
| `pod_Vt.npy` | (384, 43005) | Spatial basis (all 5 variables) |
| `pod_U.npy` | (552, 384) | Temporal basis (all 552 months) |
| `pod_S.npy` | (384,) | Singular values |
| `pod_variance_ratio.npy` | (384,) | Variance spectrum |
| `pod_coeffs_mean.npy` | (20,) | Mean for z-scoring (first 20 modes) |
| `pod_coeffs_std.npy` | (20,) | Std for z-scoring (first 20 modes) |

**Note:** `pod_Vt.npy` has 43,005 columns = 5 variables × 8,601 grid points.
The SST block is the first 8,601 columns.

---

## 4. Repository structure summary

pod-lstm-northsea-baltic/
├── README.md
├── LICENSE
├── MANIFEST.md # This file
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/ # ERA5 + POD (not in git)
├── src/ # 21 Python scripts
├── results/ # Numeric outputs
└── figures/ # PNG + PDF figures

---

## 5. Runtime summary

| Step | Time |
|---|---|
| ERA5 download | 1-2 hours |
| Preprocessing | < 1 minute |
| POD (SVD) | < 10 seconds |
| LSTM training (CPU) | ~5 seconds |
| Ridge + ARIMA | 2-5 minutes |
| Ablation (10 seeds) | 3-5 minutes |
| All figures | < 1 minute |

**Total (after download): approximately 15 minutes on a standard laptop CPU.**
