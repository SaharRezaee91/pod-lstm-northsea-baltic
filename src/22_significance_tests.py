"""22_significance_tests.py - Wilcoxon + Diebold-Mariano tests."""
import os
import numpy as np
from scipy import stats
import config


def dm_test(y_true, pred1, pred2, h=1):
    """Diebold-Mariano test on squared errors."""
    e1 = (y_true - pred1) ** 2
    e2 = (y_true - pred2) ** 2
    d = e1 - e2
    n = len(d)
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_val


def main():
    print("=" * 60)
    print("Significance tests")
    print("=" * 60)

    true_vals = np.load(config.TRUE_TEST)
    pred_lstm = np.load(config.TLV2_PRED_LSTM)
    pred_ridge = np.load(config.RIDGE_PRED)
    pred_arima = np.load(config.ARIMA_PRED)
    rmse_lstm = np.load(config.TLV2_RMSE_LSTM)
    rmse_ridge = np.load(config.RIDGE_RMSE)
    rmse_arima = np.load(config.ARIMA_RMSE)

    lines = []
    lines.append("Significance tests (LSTM vs Ridge, LSTM vs ARIMA)")
    lines.append("=" * 60)

    # ---- Wilcoxon mode-level ----
    w1, p1 = stats.wilcoxon(rmse_lstm, rmse_ridge)
    lines.append(f"\nMode-level Wilcoxon (LSTM vs Ridge): W={w1:.1f}, p={p1:.4f}")
    print(f"Wilcoxon LSTM vs Ridge: W={w1:.1f}, p={p1:.4f}")

    w2, p2 = stats.wilcoxon(rmse_lstm, rmse_arima)
    lines.append(f"Mode-level Wilcoxon (LSTM vs ARIMA): W={w2:.1f}, p={p2:.4f}")
    print(f"Wilcoxon LSTM vs ARIMA: W={w2:.1f}, p={p2:.4f}")

    # ---- Diebold-Mariano aggregate (across modes, per timestep) ----
    err_lstm = (true_vals - pred_lstm) ** 2
    err_ridge = (true_vals - pred_ridge) ** 2
    err_arima = (true_vals - pred_arima) ** 2

    agg_lstm = err_lstm.mean(axis=1)
    agg_ridge = err_ridge.mean(axis=1)
    agg_arima = err_arima.mean(axis=1)

    dm1, dp1 = dm_test(true_vals.mean(axis=1), pred_lstm.mean(axis=1),
                       pred_ridge.mean(axis=1))
    lines.append(f"\nDM test (aggregate LSTM vs Ridge): DM={dm1:.3f}, p={dp1:.4f}")
    print(f"DM LSTM vs Ridge: DM={dm1:.3f}, p={dp1:.4f}")

    dm2, dp2 = dm_test(true_vals.mean(axis=1), pred_lstm.mean(axis=1),
                       pred_arima.mean(axis=1))
    lines.append(f"DM test (aggregate LSTM vs ARIMA): DM={dm2:.3f}, p={dp2:.4f}")
    print(f"DM LSTM vs ARIMA: DM={dm2:.3f}, p={dp2:.4f}")

    # ---- Mode-specific DM ----
    lines.append("\nMode-specific DM (LSTM vs ARIMA):")
    for m in range(config.N_MODES):
        dm, dp = dm_test(true_vals[:, m], pred_lstm[:, m], pred_arima[:, m])
        lines.append(f"  Mode {m+1:>2}: DM={dm:>7.3f}, p={dp:.5f}")

    out = os.path.join(config.RESULTS_DIR, 'significance_tests.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
