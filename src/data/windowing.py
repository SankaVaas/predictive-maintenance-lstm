# src/data/windowing.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def make_windows(df: pd.DataFrame, feature_cols: list,
                 seq_len: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X : (N, seq_len, n_features)
        y : (N,)  — RUL at the last step of each window
    """
    X, y = [], []
    for unit_id, group in df.groupby("unit_id"):
        data   = group[feature_cols].values          # (T, F)
        labels = group["RUL"].values                 # (T,)

        if len(data) < seq_len:
            continue  # skip engines with too few cycles

        for i in range(len(data) - seq_len + 1):
            X.append(data[i : i + seq_len])
            y.append(labels[i + seq_len - 1])       # label = RUL at window end

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class CMAPSSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, seq_len, features)
        self.y = torch.from_numpy(y)   # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# src/data/windowing.py  — replace get_loaders entirely
def get_loaders(train_df, test_df, feature_cols,
                seq_len=30, batch_size=256, val_split=0.1):

    X_train, y_train = make_windows(train_df, feature_cols, seq_len)

    # validation split
    n_val = int(len(X_train) * val_split)
    X_val,   y_val   = X_train[-n_val:],  y_train[-n_val:]
    X_train, y_train = X_train[:-n_val],  y_train[:-n_val]

    # ── test: build one window per engine manually ────────────────────────────
    # Take the last seq_len rows of each engine; label = RUL_true from file
    rul_true = (
        test_df[test_df["RUL"] != -1]          # rows that have a real label
        .groupby("unit_id")["RUL"].first()      # one RUL per engine
    )

    X_test_list, y_test_list = [], []
    for unit_id, group in test_df.groupby("unit_id"):
        data = group[feature_cols].values        # (T, F)
        if unit_id not in rul_true.index:
            continue
        # pad with zeros if engine has fewer than seq_len cycles
        if len(data) < seq_len:
            pad = np.zeros((seq_len - len(data), data.shape[1]), dtype=np.float32)
            data = np.vstack([pad, data])
        window = data[-seq_len:]                 # last seq_len cycles
        X_test_list.append(window)
        y_test_list.append(rul_true[unit_id])

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    train_loader = DataLoader(CMAPSSDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(CMAPSSDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(CMAPSSDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train windows : {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    print(f"Input shape   : {X_train.shape}  →  (batch, {seq_len}, {X_train.shape[2]})")

    return train_loader, val_loader, test_loader, X_train.shape[2]