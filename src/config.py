"""
config.py — Shared configuration for all pipeline scripts.

Centralizes paths, hyperparameters, and plotting style. Every script in
src/ imports from this module, so values such as N_MODES and TRAIN_END
have a single source of truth.

Input : none
Output: none (only defines constants and file paths)
Paper : Sections 3.2-3.5 (data splits, hyperparameters, baselines)
"""
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Study domain
LAT_MIN, LAT_MAX = 50.0, 65.0
LON_MIN, LON_MAX = -5.0, 30.0
GRID_RES = 0.25

# Time
YEAR_START, YEAR_END = 1979, 2024
N_MONTHS = 552
TRAIN_END = 384
VAL_END   = 468

# POD
N_MODES = 20
SEQ_LEN = 24

# LSTM
HIDDEN       = 32
N_EPOCHS     = 300
LR           = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT      = 0.3
EARLY_STOP   = 20
GRAD_CLIP    = 1.0

# Ridge
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]

# Ablation
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

# Raw ERA5
ERA5_FULL = os.path.join(DATA_DIR, 'era5_processed_1979_2024.nc')
ERA5_SST  = os.path.join(DATA_DIR, 'era5_sst_1979_2024.nc')

# ============================================================
# POD - using ORIGINAL paper files (pod_*.npy, not pod_TL_*)
# ============================================================
POD_A_TRAIN = os.path.join(DATA_DIR, 'pod_A_train.npy')  # (384, 384) raw
POD_A_VAL   = os.path.join(DATA_DIR, 'pod_A_val.npy')    # (84, 384) raw
POD_A_TEST  = os.path.join(DATA_DIR, 'pod_A_test.npy')   # (84, 384) raw
POD_VT      = os.path.join(DATA_DIR, 'pod_Vt.npy')       # (384, 43005)
POD_U       = os.path.join(DATA_DIR, 'pod_U.npy')        # (552, 384)
POD_S       = os.path.join(DATA_DIR, 'pod_S.npy')        # (384,)
POD_VAR     = os.path.join(DATA_DIR, 'pod_variance_ratio.npy')  # (384,)

# Coefficients mean/std (computed from A_train)
POD_MEAN = os.path.join(DATA_DIR, 'pod_coeffs_mean.npy')
POD_STD  = os.path.join(DATA_DIR, 'pod_coeffs_std.npy')

# SST block of Vt (for spatial reconstruction)
POD_SST_VT = os.path.join(DATA_DIR, 'pod_Vt_sst.npy')

# LSTM
TLV2_PRED_LSTM  = os.path.join(RESULTS_DIR, 'TLv2_pred_lstm.npy')
TLV2_RMSE_LSTM  = os.path.join(RESULTS_DIR, 'TLv2_rmse_lstm.npy')
TLV2_MAE_LSTM   = os.path.join(RESULTS_DIR, 'TLv2_mae_lstm.npy')
TLV2_TRAIN_LOSS = os.path.join(RESULTS_DIR, 'TLv2_train_losses.npy')
TLV2_VAL_LOSS   = os.path.join(RESULTS_DIR, 'TLv2_val_losses.npy')

# Ridge
RIDGE_PRED      = os.path.join(RESULTS_DIR, 'pred_ridge_tuned.npy')
RIDGE_RMSE      = os.path.join(RESULTS_DIR, 'rmse_ridge_tuned.npy')
RIDGE_MAE       = os.path.join(RESULTS_DIR, 'mae_ridge_tuned.npy')
RIDGE_R2        = os.path.join(RESULTS_DIR, 'r2_ridge_tuned.npy')
RIDGE_R         = os.path.join(RESULTS_DIR, 'r_ridge_tuned.npy')
RIDGE_SELECTION = os.path.join(RESULTS_DIR, 'ridge_alpha_selection.txt')

# ARIMA
ARIMA_PRED = os.path.join(RESULTS_DIR, 'TLv2_pred_arima.npy')
ARIMA_RMSE = os.path.join(RESULTS_DIR, 'TLv2_rmse_arima.npy')
ARIMA_MAE  = os.path.join(RESULTS_DIR, 'TLv2_mae_arima.npy')

# Ground truth
TRUE_TEST = os.path.join(RESULTS_DIR, 'true_test_v2.npy')

# Ablation
ABLATION_LSTM_MODES = os.path.join(RESULTS_DIR, 'fig8_lstm_modes_FINAL.npy')
ABLATION_GRU_MODES  = os.path.join(RESULTS_DIR, 'fig8_gru_modes_FINAL.npy')
ABLATION_UV_MODES   = os.path.join(RESULTS_DIR, 'fig8_uv_modes_FINAL.npy')
ABLATION_LSTM_SEEDS = os.path.join(RESULTS_DIR, 'fig8_lstm_seeds_FINAL.npy')
ABLATION_GRU_SEEDS  = os.path.join(RESULTS_DIR, 'fig8_gru_seeds_FINAL.npy')
ABLATION_UV_SEEDS   = os.path.join(RESULTS_DIR, 'fig8_uv_seeds_FINAL.npy')

# Metrics
R2_LSTM  = os.path.join(RESULTS_DIR, 'r2_lstm_FINAL.npy')
R2_RIDGE = os.path.join(RESULTS_DIR, 'r2_ridge_FINAL.npy')
R2_ARIMA = os.path.join(RESULTS_DIR, 'r2_arima_FINAL.npy')
R_LSTM   = os.path.join(RESULTS_DIR, 'pearson_lstm_FINAL.npy')
R_RIDGE  = os.path.join(RESULTS_DIR, 'pearson_ridge_FINAL.npy')
R_ARIMA  = os.path.join(RESULTS_DIR, 'pearson_arima_FINAL.npy')

# Plotting
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'true': '#2C3E50',
    'lstm': '#C0392B',
    'ridge': '#2980B9',
    'arima': '#27AE60',
    'gru': '#2980B9',
    'univariate': '#27AE60',
    'transformer': '#8E44AD',
}
