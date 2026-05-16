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
    
# src/data/windowing.py — add this function, call it inside get_loaders

def augment_short_life(train_df: pd.DataFrame, feature_cols: list,
                        seq_len: int, lifetime_threshold: int = 150,
                        oversample_factor: int = 3) -> tuple:
    """
    Oversample windows from engines with short lifetimes.
    These are underrepresented and cause over-prediction on test engines like #67.
    """
    max_cycles = train_df.groupby("unit_id")["cycle"].max()
    short_units = max_cycles[max_cycles < lifetime_threshold].index.tolist()

    if not short_units:
        return np.array([]), np.array([])

    short_df = train_df[train_df["unit_id"].isin(short_units)]
    X_short, y_short = make_windows(short_df, feature_cols, seq_len)

    # repeat oversample_factor times
    X_aug = np.tile(X_short, (oversample_factor, 1, 1))
    y_aug = np.tile(y_short, oversample_factor)

    print(f"  Short-life engines (<{lifetime_threshold} cycles): {len(short_units)} engines")
    print(f"  Augmented windows added: {len(X_aug):,}  (factor={oversample_factor}x)")
    return X_aug, y_aug


# src/data/windowing.py — add this function, call it inside get_loaders

def augment_short_life(train_df: pd.DataFrame, feature_cols: list,
                        seq_len: int, lifetime_threshold: int = 200,
                        oversample_factor: int = 5) -> tuple:
    """
    Oversample windows from engines with short lifetimes.
    These are underrepresented and cause over-prediction on test engines like #67.
    """
    max_cycles = train_df.groupby("unit_id")["cycle"].max()
    short_units = max_cycles[max_cycles < lifetime_threshold].index.tolist()

    if not short_units:
        return np.array([]), np.array([])

    short_df = train_df[train_df["unit_id"].isin(short_units)]
    X_short, y_short = make_windows(short_df, feature_cols, seq_len)

    # repeat oversample_factor times
    X_aug = np.tile(X_short, (oversample_factor, 1, 1))
    y_aug = np.tile(y_short, oversample_factor)

    print(f"  Short-life engines (<{lifetime_threshold} cycles): {len(short_units)} engines")
    print(f"  Augmented windows added: {len(X_aug):,}  (factor={oversample_factor}x)")
    return X_aug, y_aug


def get_loaders(train_df, test_df, feature_cols,
                seq_len=30, batch_size=256, val_split=0.1,
                stratified=False):

    if stratified:
        all_units   = sorted(train_df["unit_id"].unique())
        n_val_units = max(1, int(len(all_units) * val_split))
        val_units   = all_units[-n_val_units:]
        train_units = all_units[:-n_val_units]
        tr_df  = train_df[train_df["unit_id"].isin(train_units)]
        vl_df  = train_df[train_df["unit_id"].isin(val_units)]
        X_train, y_train = make_windows(tr_df, feature_cols, seq_len)
        X_val,   y_val   = make_windows(vl_df, feature_cols, seq_len)
        print(f"Train engines : {len(train_units)}  |  Val engines: {n_val_units}")
    else:
        X_all, y_all = make_windows(train_df, feature_cols, seq_len)
        n_val        = int(len(X_all) * val_split)
        X_val,   y_val   = X_all[-n_val:],  y_all[-n_val:]
        X_train, y_train = X_all[:-n_val],  y_all[:-n_val]

    # augment short-life engines
    X_aug, y_aug = augment_short_life(train_df, feature_cols, seq_len,
                                       lifetime_threshold=200, oversample_factor=5)
    if len(X_aug):
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)

        # re-shuffle after augmentation
        idx     = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  Train windows after augmentation: {len(X_train):,}")
        
    # test — last window per engine
    rul_true = (
        test_df[test_df["RUL"] != -1]
        .groupby("unit_id")["RUL"].first()
    )
    X_test_list, y_test_list = [], []
    for unit_id, group in test_df.groupby("unit_id"):
        data = group[feature_cols].values
        if unit_id not in rul_true.index:
            continue
        if len(data) < seq_len:
            pad  = np.zeros((seq_len - len(data), data.shape[1]), dtype=np.float32)
            data = np.vstack([pad, data])
        X_test_list.append(data[-seq_len:])
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