"""06_ablation_multiseed.py - 10-seed ablation (LSTM/GRU/univariate)."""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import config


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


class POD_LSTM(nn.Module):
    def __init__(self, n_modes, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_modes, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, n_modes))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class POD_GRU(nn.Module):
    def __init__(self, n_modes, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_modes, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, n_modes))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))


def train_model(model, X_tr, y_tr, X_val, y_val):
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR,
                           weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=20, factor=0.5, min_lr=1e-6)
    best_val, no_imp, best_state = float('inf'), 0, None
    for epoch in range(config.N_EPOCHS):
        model.train()
        loss = nn.MSELoss()(model(Xtr), ytr)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        opt.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = nn.MSELoss()(model(Xv), yv).item()
            sched.step(vl)
            if vl < best_val:
                best_val = vl; no_imp = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 10
            if no_imp >= config.EARLY_STOP:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
    return np.sqrt(np.mean((pred - y_te)**2, axis=0))


def prepare_uv_data():
    print("Building univariate SST-only embedding...")
    X_train_full = np.load(os.path.join(config.DATA_DIR, 'X_train.npy'))
    X_val_full = np.load(os.path.join(config.DATA_DIR, 'X_val.npy'))
    X_test_full = np.load(os.path.join(config.DATA_DIR, 'X_test.npy'))
    n_sst = 61 * 141
    sst_tr = X_train_full[:, :n_sst]
    sst_val = X_val_full[:, :n_sst]
    sst_te = X_test_full[:, :n_sst]
    U, S, Vt = np.linalg.svd(sst_tr, full_matrices=False)
    A_tr = (U * S)[:, :config.N_MODES]
    A_val = (sst_val @ Vt.T)[:, :config.N_MODES]
    A_te = (sst_te @ Vt.T)[:, :config.N_MODES]
    mu = A_tr.mean(0); sd = A_tr.std(0); sd[sd < 1e-10] = 1.0
    return (A_tr - mu) / sd, (A_val - mu) / sd, (A_te - mu) / sd


def main():
    print("=" * 60 + f"\nAblation study: {config.N_SEEDS} seeds\n" + "=" * 60)
    A_train = np.load(config.POD_A_TRAIN)
    A_val = np.load(config.POD_A_VAL)
    A_test = np.load(config.POD_A_TEST)
    X_tr_mv, y_tr_mv = make_sequences(A_train, config.SEQ_LEN)
    val_ext = np.concatenate([A_train[-config.SEQ_LEN:], A_val])
    X_val_mv, y_val_mv = make_sequences(val_ext, config.SEQ_LEN)
    test_ext = np.concatenate([A_val[-config.SEQ_LEN:], A_test])
    X_te_mv, y_te_mv = make_sequences(test_ext, config.SEQ_LEN)
    uv_tr, uv_val, uv_te = prepare_uv_data()
    X_tr_uv, y_tr_uv = make_sequences(uv_tr, config.SEQ_LEN)
    val_ext_uv = np.concatenate([uv_tr[-config.SEQ_LEN:], uv_val])
    X_val_uv, y_val_uv = make_sequences(val_ext_uv, config.SEQ_LEN)
    test_ext_uv = np.concatenate([uv_val[-config.SEQ_LEN:], uv_te])
    X_te_uv, y_te_uv = make_sequences(test_ext_uv, config.SEQ_LEN)
    all_lstm_mv = np.zeros((config.N_SEEDS, config.N_MODES))
    all_gru_mv = np.zeros((config.N_SEEDS, config.N_MODES))
    all_uv = np.zeros((config.N_SEEDS, config.N_MODES))
    for i, seed in enumerate(config.SEEDS):
        print(f"\n--- Seed {seed+1}/{config.N_SEEDS} ---")
        set_seed(seed)
        m1 = train_model(POD_LSTM(config.N_MODES, config.HIDDEN),
                         X_tr_mv, y_tr_mv, X_val_mv, y_val_mv)
        all_lstm_mv[i] = evaluate(m1, X_te_mv, y_te_mv)
        set_seed(seed)
        m2 = train_model(POD_GRU(config.N_MODES, config.HIDDEN),
                         X_tr_mv, y_tr_mv, X_val_mv, y_val_mv)
        all_gru_mv[i] = evaluate(m2, X_te_mv, y_te_mv)
        set_seed(seed)
        m3 = train_model(POD_LSTM(config.N_MODES, config.HIDDEN),
                         X_tr_uv, y_tr_uv, X_val_uv, y_val_uv)
        all_uv[i] = evaluate(m3, X_te_uv, y_te_uv)
    lstm_seeds = all_lstm_mv.mean(axis=1)
    gru_seeds = all_gru_mv.mean(axis=1)
    uv_seeds = all_uv.mean(axis=1)
    t_stat, p_t = stats.ttest_rel(lstm_seeds, gru_seeds)
    print(f"\nPaired t-test LSTM vs GRU: t={t_stat:.3f}, p={p_t:.4f}")
    np.save(config.ABLATION_LSTM_MODES, all_lstm_mv.mean(axis=0))
    np.save(config.ABLATION_GRU_MODES, all_gru_mv.mean(axis=0))
    np.save(config.ABLATION_UV_MODES, all_uv.mean(axis=0))
    np.save(config.ABLATION_LSTM_SEEDS, lstm_seeds)
    np.save(config.ABLATION_GRU_SEEDS, gru_seeds)
    np.save(config.ABLATION_UV_SEEDS, uv_seeds)
    print("Saved ablation arrays.")


if __name__ == "__main__":
    main()
