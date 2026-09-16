"""03_pod_decomposition.py - Leakage-free POD (SVD on train only)."""
import os
import numpy as np
import config


def main():
    print("=" * 60 + "\nPOD decomposition (leakage-free)\n" + "=" * 60)
    X_train = np.load(os.path.join(config.DATA_DIR, 'X_train.npy'))
    X_val = np.load(os.path.join(config.DATA_DIR, 'X_val.npy'))
    X_test = np.load(os.path.join(config.DATA_DIR, 'X_test.npy'))
    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")
    print("\nSVD on TRAIN only...")
    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    print(f"  U : {U.shape}")
    print(f"  S : {S.shape}")
    print(f"  Vt: {Vt.shape}")
    var_ratio = (S**2) / np.sum(S**2)
    cum = np.cumsum(var_ratio) * 100
    for k in [3, 14, 20, 52]:
        print(f"  {k:>3} modes: {cum[k-1]:.2f}%")
    A_train = U * S
    A_val = (X_val @ Vt.T)[:, :config.N_MODES]
    A_test = (X_test @ Vt.T)[:, :config.N_MODES]
    A_train_N = A_train[:, :config.N_MODES]
    coeffs_mean = A_train_N.mean(axis=0)
    coeffs_std = A_train_N.std(axis=0)
    coeffs_std[coeffs_std < 1e-10] = 1.0
    A_train_std = (A_train_N - coeffs_mean) / coeffs_std
    A_val_std = (A_val - coeffs_mean) / coeffs_std
    A_test_std = (A_test - coeffs_mean) / coeffs_std
    n_space_per_var = 61 * 141
    Vt_sst = Vt[:config.N_MODES, :n_space_per_var]
    np.save(config.POD_U, U)
    np.save(config.POD_S, S)
    np.save(config.POD_VT, Vt)
    np.save(config.POD_VAR, var_ratio)
    np.save(config.POD_A_TRAIN, A_train_std)
    np.save(config.POD_A_VAL, A_val_std)
    np.save(config.POD_A_TEST, A_test_std)
    np.save(config.POD_MEAN, coeffs_mean)
    np.save(config.POD_STD, coeffs_std)
    np.save(config.POD_SST_VT, Vt_sst)
    print(f"\nSaved POD outputs -> {config.DATA_DIR}/")


if __name__ == "__main__":
    main()
