"""
21_make_table_a2.py — Tables A1 and A2 (Appendix A).

Table A1: continuation of Table 3, reporting RMSE and MAE for modes
          11-20 for LSTM, tuned Ridge, and ARIMA.

Table A2: coefficient of determination (R²) and Pearson correlation
          coefficient (r) for all 20 retained modes, computed against
          the observed test-period coefficients. Mean across modes is
          reported in the last row.

Input : results/true_test_v2.npy
        results/TLv2_pred_lstm.npy, pred_ridge_tuned.npy, TLv2_pred_arima.npy
        results/TLv2_rmse_lstm.npy, rmse_ridge_tuned.npy, TLv2_rmse_arima.npy
        results/TLv2_mae_lstm.npy,  mae_ridge_tuned.npy,  TLv2_mae_arima.npy
        data/pod_variance_ratio.npy
Output: results/Tables_A1_A2.xlsx
        results/r2_*_FINAL.npy, pearson_*_FINAL.npy
        results/r2_ridge_tuned.npy, r_ridge_tuned.npy

Paper : Appendix A; Tables A1, A2
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import config


def compute_r2_pearson(y_true, y_pred):
    n = y_true.shape[1]
    r2 = np.zeros(n)
    r = np.zeros(n)
    for m in range(n):
        yt = y_true[:, m]
        yp = y_pred[:, m]
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        ss_res = np.sum((yt - yp) ** 2)
        r2[m] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
            r[m], _ = stats.pearsonr(yt, yp)
        else:
            r[m] = np.nan
    return r2, r


def main():
    print("Tables A1 and A2")

    true_vals = np.load(config.TRUE_TEST)
    pred_lstm = np.load(config.TLV2_PRED_LSTM)
    pred_ridge = np.load(config.RIDGE_PRED)
    pred_arima = np.load(config.ARIMA_PRED)

    shapes = {'lstm': pred_lstm.shape, 'ridge': pred_ridge.shape,
              'arima': pred_arima.shape, 'true': true_vals.shape}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Shape mismatch: {shapes}")

    r2_lstm, r_lstm = compute_r2_pearson(true_vals, pred_lstm)
    r2_ridge, r_ridge = compute_r2_pearson(true_vals, pred_ridge)
    r2_arima, r_arima = compute_r2_pearson(true_vals, pred_arima)

    np.save(config.R2_LSTM, r2_lstm)
    np.save(config.R2_RIDGE, r2_ridge)
    np.save(config.R2_ARIMA, r2_arima)
    np.save(config.R_LSTM, r_lstm)
    np.save(config.R_RIDGE, r_ridge)
    np.save(config.R_ARIMA, r_arima)

    rmse_lstm = np.load(config.TLV2_RMSE_LSTM)
    rmse_ridge = np.load(config.RIDGE_RMSE)
    rmse_arima = np.load(config.ARIMA_RMSE)
    mae_lstm = np.load(config.TLV2_MAE_LSTM)
    mae_ridge = np.load(config.RIDGE_MAE)
    mae_arima = np.load(config.ARIMA_MAE)
    var = np.load(config.POD_VAR)[:config.N_MODES]

    # Table A2
    rows_a2 = []
    for i in range(config.N_MODES):
        rows_a2.append({
            'Mode': i + 1,
            'R² LSTM': round(r2_lstm[i], 3),
            'R² Ridge': round(r2_ridge[i], 3),
            'R² ARIMA': round(r2_arima[i], 3),
            'r LSTM': round(r_lstm[i], 3),
            'r Ridge': round(r_ridge[i], 3),
            'r ARIMA': round(r_arima[i], 3),
        })
    rows_a2.append({
        'Mode': 'Mean',
        'R² LSTM': round(np.nanmean(r2_lstm), 3),
        'R² Ridge': round(np.nanmean(r2_ridge), 3),
        'R² ARIMA': round(np.nanmean(r2_arima), 3),
        'r LSTM': round(np.nanmean(r_lstm), 3),
        'r Ridge': round(np.nanmean(r_ridge), 3),
        'r ARIMA': round(np.nanmean(r_arima), 3),
    })
    df_a2 = pd.DataFrame(rows_a2)

    # Table A1
    rows_a1 = []
    for i in range(10, config.N_MODES):
        imp = (rmse_ridge[i] - rmse_lstm[i]) / rmse_ridge[i] * 100
        rows_a1.append({
            'Mode': i + 1,
            'Variance (%)': round(var[i] * 100, 1),
            'LSTM RMSE': round(rmse_lstm[i], 4),
            'Ridge RMSE': round(rmse_ridge[i], 4),
            'ARIMA RMSE': round(rmse_arima[i], 4),
            'Δ vs Ridge (%)': round(imp, 1),
            'LSTM MAE': round(mae_lstm[i], 4),
            'Ridge MAE': round(mae_ridge[i], 4),
            'ARIMA MAE': round(mae_arima[i], 4),
        })
    df_a1 = pd.DataFrame(rows_a1)

    out = os.path.join(config.RESULTS_DIR, 'Tables_A1_A2.xlsx')
    with pd.ExcelWriter(out) as w:
        df_a1.to_excel(w, sheet_name='TableA1', index=False)
        df_a2.to_excel(w, sheet_name='TableA2', index=False)

    print("\nTable A1 (Modes 11-20):")
    print(df_a1.to_string(index=False))
    print("\nTable A2 (R² and r):")
    print(df_a2.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
