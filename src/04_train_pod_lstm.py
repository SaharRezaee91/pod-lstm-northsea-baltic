"""04_train_pod_lstm.py - Main POD-LSTM (Table 2 of paper)."""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import config


def set_seed(s=42):
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
    train_losses, val_losses = [], []
    for epoch in range(config.N_EPOCHS):
        model.train()
        loss = nn.MSELoss()(model(Xtr), ytr)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        opt.step()
        train_losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = nn.MSELoss()(model(Xv), yv).item()
            val_losses.append(vl)
            sched.step(vl)
            if vl < best_val:
                best_val = vl
                no_imp = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 10
            if no_imp >= config.EARLY_STOP:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
    rmse = np.sqrt(np.mean((pred - y_te)**2, axis=0))
    mae = np.mean(np.abs(pred - y_te), axis=0)
    return pred, rmse, mae


def main():
    set_seed(42)
    print("=" * 60 + "\nTraining main POD-LSTM\n" + "=" * 60)
    A_train = np.load(config.POD_A_TRAIN)
    A_val = np.load(config.POD_A_VAL)
    A_test = np.load(config.POD_A_TEST)
    X_tr, y_tr = make_sequences(A_train, config.SEQ_LEN)
    val_ext = np.concatenate([A_train[-config.SEQ_LEN:], A_val])
    X_val, y_val = make_sequences(val_ext, config.SEQ_LEN)
    test_ext = np.concatenate([A_val[-config.SEQ_LEN:], A_test])
    X_te, y_te = make_sequences(test_ext, config.SEQ_LEN)
    print(f"X_tr : {X_tr.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_te : {X_te.shape}")
    np.save(config.TRUE_TEST, y_te)
    model = POD_LSTM(config.N_MODES, config.HIDDEN)
    model, train_losses, val_losses = train_model(model, X_tr, y_tr, X_val, y_val)
    pred, rmse, mae = evaluate(model, X_te, y_te)
    print(f"\nTest RMSE: {rmse.mean():.4f}")
    print(f"Test MAE : {mae.mean():.4f}")
    print(f"Final epoch: {len(train_losses)}")
    np.save(config.TLV2_PRED_LSTM, pred)
    np.save(config.TLV2_RMSE_LSTM, rmse)
    np.save(config.TLV2_MAE_LSTM, mae)
    np.save(config.TLV2_TRAIN_LOSS, np.array(train_losses))
    np.save(config.TLV2_VAL_LOSS, np.array(val_losses))


if __name__ == "__main__":
    main()
