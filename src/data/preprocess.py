# src/data/preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

COLS = (
    ["unit_id", "cycle"] +
    [f"op_{i}" for i in range(1, 4)] +
    [f"s{i}" for i in range(1, 22)]
)

# Sensors to drop — near-zero variance on FD001
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Cap RUL at 125 — engine is "healthy" until ~125 cycles to go
RUL_CAP = 125


def load_raw(split: str, dataset: str = "FD001") -> pd.DataFrame:
    path = Path(f"data/raw/{split}_{dataset}.txt")
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add Remaining Useful Life column to training data."""
    max_cycle = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CAP)
    return df.drop(columns=["max_cycle"])


def add_rolling_features(df: pd.DataFrame, sensors: list, window: int = 5) -> pd.DataFrame:
    """Rolling mean and std per engine — captures local degradation trend."""
    for col in sensors:
        df[f"{col}_rmean"] = (
            df.groupby("unit_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{col}_rstd"] = (
            df.groupby("unit_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    return df


def preprocess(dataset: str = "FD001", window: int = 5, save: bool = True):
    # ── Load ──────────────────────────────────────────────────────────────────
    train_df = load_raw("train", dataset)
    test_df  = load_raw("test",  dataset)

    # ── RUL for train ─────────────────────────────────────────────────────────
    train_df = add_rul(train_df)

    # ── RUL for test — assign true RUL only to each engine's last row ─────────
    rul_test = pd.read_csv(
        f"data/raw/RUL_{dataset}.txt", header=None, names=["RUL_true"]
    )
    rul_test["unit_id"] = range(1, len(rul_test) + 1)  # unit IDs are 1-indexed

    test_df["RUL"] = -1  # default: unlabeled rows
    for _, row in rul_test.iterrows():
        uid  = int(row["unit_id"])
        last = test_df[test_df["unit_id"] == uid]["cycle"].idxmax()
        test_df.loc[last, "RUL"] = min(row["RUL_true"], RUL_CAP)

    # ── Drop flat sensors ─────────────────────────────────────────────────────
    sensor_cols = [c for c in COLS[5:] if c not in DROP_SENSORS]

    # ── Rolling features ──────────────────────────────────────────────────────
    train_df = add_rolling_features(train_df, sensor_cols, window)
    test_df  = add_rolling_features(test_df,  sensor_cols, window)

    # ── Feature columns ───────────────────────────────────────────────────────
    feature_cols = (
        sensor_cols +
        [f"{s}_rmean" for s in sensor_cols] +
        [f"{s}_rstd"  for s in sensor_cols] +
        [f"op_{i}" for i in range(1, 4)]
    )

    # ── Normalize — fit on train only, transform both ─────────────────────────
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        out = Path("data/processed")
        out.mkdir(exist_ok=True)
        train_df.to_parquet(out / f"train_{dataset}.parquet", index=False)
        test_df.to_parquet(out  / f"test_{dataset}.parquet",  index=False)
        with open(out / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"Saved to data/processed/  |  features: {len(feature_cols)}")

    return train_df, test_df, feature_cols


if __name__ == "__main__":
    train, test, features = preprocess("FD001")
    print(train[["unit_id", "cycle", "RUL"]].tail())
    print(f"\nFeature count : {len(features)}")
    print(f"Train shape   : {train.shape}")
    print(f"Test shape    : {test.shape}")
    print(f"\nTest RUL sample (labeled rows only):")
    print(test[test["RUL"] != -1][["unit_id", "cycle", "RUL"]].head(10))