"""
20_make_table3.py — Table 3: RMSE and MAE for the first 10 POD modes.

Builds Table 3 of the paper: per-mode RMSE and MAE for LSTM, tuned
Ridge, and ARIMA baselines, plus the relative RMSE change (LSTM vs
Ridge). The Mean row reports averages across all 20 retained modes.
Results are exported to Excel.

Input : results/TLv2_rmse_lstm.npy, rmse_ridge_tuned.npy, TLv2_rmse_arima.npy
        results/TLv2_mae_lstm.npy,  mae_ridge_tuned.npy,  TLv2_mae_arima.npy
        data/pod_variance_ratio.npy
Output: results/Table3.xlsx

Paper : Section 4.3; Table 3
"""
import os
import numpy as np
import pandas as pd
import config


def main():
    print("Table 3: RMSE/MAE")
    rmse_lstm = np.load(config.TLV2_RMSE_LSTM)
    rmse_ridge = np.load(config.RIDGE_RMSE)
    rmse_arima = np.load(config.ARIMA_RMSE)
    mae_lstm = np.load(config.TLV2_MAE_LSTM)
    mae_ridge = np.load(config.RIDGE_MAE)
    mae_arima = np.load(config.ARIMA_MAE)
    var_ratio = np.load(config.POD_VAR)[:config.N_MODES]

    n = config.N_MODES
    rows = []
    for i in range(10):
        imp = (rmse_ridge[i] - rmse_lstm[i]) / rmse_ridge[i] * 100
        rows.append({
            'Mode': i + 1,
            'Variance (%)': round(var_ratio[i] * 100, 1),
            'LSTM RMSE': round(rmse_lstm[i], 3),
            'Ridge RMSE': round(rmse_ridge[i], 3),
            'ARIMA RMSE': round(rmse_arima[i], 3),
            'Δ vs Ridge (%)': f'{imp:+.1f}',
            'LSTM MAE': round(mae_lstm[i], 3),
            'Ridge MAE': round(mae_ridge[i], 3),
            'ARIMA MAE': round(mae_arima[i], 3),
        })

    mean_imp = (rmse_ridge.mean() - rmse_lstm.mean()) / rmse_ridge.mean() * 100
    rows.append({
        'Mode': 'Mean',
        'Variance (%)': '—',
        'LSTM RMSE': round(rmse_lstm.mean(), 3),
        'Ridge RMSE': round(rmse_ridge.mean(), 3),
        'ARIMA RMSE': round(rmse_arima.mean(), 3),
        'Δ vs Ridge (%)': f'{mean_imp:+.1f}',
        'LSTM MAE': round(mae_lstm.mean(), 3),
        'Ridge MAE': round(mae_ridge.mean(), 3),
        'ARIMA MAE': round(mae_arima.mean(), 3),
    })

    df = pd.DataFrame(rows)
    out_xlsx = os.path.join(config.RESULTS_DIR, 'Table3.xlsx')
    df.to_excel(out_xlsx, index=False)

    print("\n" + "=" * 100)
    print("TABLE 3 — RMSE and MAE of POD coefficient predictions")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print(f"\nSaved: {out_xlsx}")


if __name__ == "__main__":
    main()
