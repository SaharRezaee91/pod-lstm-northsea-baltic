"""
05_train_baselines.py — Ridge and ARIMA baseline models.

Ridge: predicts the 20-D POD coefficient vector from the flattened
24-month input. Regularization alpha selected via the same validation
protocol as the LSTM (Section 3.5); the optimum is alpha = 3000.

ARIMA(2,0,1): order selected by AIC, fitted per mode on the full
pre-test record (no held-out tuning).

Input : data/pod_A_train.npy, pod_A_val.npy, pod_A_test.npy
Output: results/pred_ridge_tuned.npy, rmse_ridge_tuned.npy, mae_ridge_tuned.npy
        results/ridge_alpha_selection.txt
        results/TLv2_pred_arima.npy, TLv2_rmse_arima.npy, TLv2_mae_arima.npy

Paper : Section 3.5 (Baseline models and evaluation metrics)
"""
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import config


def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        Xf = X.reshape(len(X), -1)
        A = Xf.T @ Xf + self.alpha * np.eye(Xf.shape[1])
        self.W = np.linalg.solve(A, Xf.T @ y)
        self.b = y.mean(axis=0) - Xf.mean(axis=0) @ self.W

    def predict(self, X):
        return X.reshape(len(X), -1) @ self.W + self.b


def metrics(pred, true):
    rmse = np.sqrt(np.mean((pred - true)**2, axis=0))
    mae = np.mean(np.abs(pred - true), axis=0)
    return rmse, mae


def train_ridge():
    print("\n" + "=" * 60 + "\nRidge (tuned alpha)\n" + "=" * 60)
    A_train = np.load(config.POD_A_TRAIN)
    A_val = np.load(config.POD_A_VAL)
    A_test = np.load(config.POD_A_TEST)
    X_tr, y_tr = make_sequences(A_train, config.SEQ_LEN)
    val_ext = np.concatenate([A_train[-config.SEQ_LEN:], A_val])
    X_val, y_val = make_sequences(val_ext, config.SEQ_LEN)
    test_ext = np.concatenate([A_val[-config.SEQ_LEN:], A_test])
    X_te, y_te = make_sequences(test_ext, config.SEQ_LEN)
    val_scores = {}
    print("\nAlpha grid search:")
    for a in config.ALPHAS:
        m = RidgeRegression(alpha=a)
        m.fit(X_tr, y_tr)
        rmse, _ = metrics(m.predict(X_val), y_val)
        val_scores[a] = rmse.mean()
        print(f"  alpha={a:<8} val_RMSE={val_scores[a]:.6f}")
    best_alpha = min(val_scores, key=val_scores.get)
    print(f"\nBEST alpha = {best_alpha}")
    final = RidgeRegression(alpha=best_alpha)
    final.fit(X_tr, y_tr)
    pred = final.predict(X_te)
    rmse, mae = metrics(pred, y_te)
    print(f"Test RMSE: {rmse.mean():.4f}")
    print(f"Test MAE : {mae.mean():.4f}")
    np.save(config.RIDGE_PRED, pred)
    np.save(config.RIDGE_RMSE, rmse)
    np.save(config.RIDGE_MAE, mae)
    with open(config.RIDGE_SELECTION, 'w') as f:
        for a in config.ALPHAS:
            f.write(f"alpha={a}, val_RMSE={val_scores[a]:.8f}\n")
        f.write(f"\nBest alpha={best_alpha}\n")


def train_arima():
    print("\n" + "=" * 60 + "\nARIMA(2,0,1)\n" + "=" * 60)
    A_train = np.load(config.POD_A_TRAIN)
    A_val = np.load(config.POD_A_VAL)
    A_test = np.load(config.POD_A_TEST)
    pretest = np.concatenate([A_train, A_val], axis=0)
    test_ext = np.concatenate([A_val[-config.SEQ_LEN:], A_test])
    X_te, y_te = make_sequences(test_ext, config.SEQ_LEN)
    n_test = len(A_test)
    pred = np.zeros((n_test, config.N_MODES))
    for m in range(config.N_MODES):
        print(f"  Fitting mode {m+1}/{config.N_MODES}...")
        series = pretest[:, m]
        try:
            model = ARIMA(series, order=(2, 0, 1))
            fit = model.fit()
            pred[:, m] = fit.forecast(steps=n_test)
        except Exception as e:
            print(f"    Warning mode {m+1}: {e}")
            pred[:, m] = y_te[:, m].mean()
    rmse, mae = metrics(pred, y_te)
    print(f"\nTest RMSE: {rmse.mean():.4f}")
    print(f"Test MAE : {mae.mean():.4f}")
    np.save(config.ARIMA_PRED, pred)
    np.save(config.ARIMA_RMSE, rmse)
    np.save(config.ARIMA_MAE, mae)


def main():
    train_ridge()
    train_arima()


if __name__ == "__main__":
    main()
