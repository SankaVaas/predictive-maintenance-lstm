# run this in a notebook or python shell to verify the download
import pandas as pd

COLS = (
    ["unit_id", "cycle"] +
    [f"op_{i}" for i in range(1, 4)] +
    [f"s{i}" for i in range(1, 22)]
)

df = pd.read_csv("data/raw/train_FD001.txt", sep=r"\s+", header=None, names=COLS)

print(df.shape)          # expect (20631, 26)
print(df["unit_id"].nunique())   # expect 100 engines
print(df.head())