# diagnose_engines.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data.preprocess import preprocess
from src.data.windowing  import get_loaders
from src.models.lstm     import PredictiveMaintenanceLSTM

CKPT   = torch.load("data/processed/best_model.pt", map_location="cpu", weights_only=False)
CONFIG = CKPT["config"]

model = PredictiveMaintenanceLSTM(CKPT["n_features"], CONFIG["hidden_size"],
                                   CONFIG["num_layers"], CONFIG["dropout"])
model.load_state_dict(CKPT["model_state"])
model.eval()

# ── raw data for context ───────────────────────────────────────────────────────
COLS = (["unit_id", "cycle"] + [f"op_{i}" for i in range(1,4)] +
        [f"s{i}" for i in range(1,22)])
test_raw = pd.read_csv("data/raw/test_FD001.txt",
                        sep=r"\s+", header=None, names=COLS)
rul_true = pd.read_csv("data/raw/RUL_FD001.txt",
                        header=None, names=["RUL_true"])
rul_true["unit_id"] = range(1, 101)

# lifetime of each test engine (how many cycles of data we have)
test_lengths = test_raw.groupby("unit_id")["cycle"].max()

print("── Problematic engines ───────────────────────────────────────")
print(f"{'Eng':>4}  {'Test cycles':>11}  {'True RUL':>9}  {'Lifetime':>9}")
print("─" * 45)
for uid in [67, 79, 48, 86, 97, 6, 98]:
    true_rul = rul_true[rul_true["unit_id"] == uid]["RUL_true"].values[0]
    cycles   = test_lengths[uid]
    total    = cycles + true_rul   # estimated total lifetime
    print(f"{uid:>4}  {cycles:>11}  {true_rul:>9}  {total:>9}")

print("\n── All test engine lifetimes (total = test_cycles + true_RUL) ─")
all_totals = []
for uid in range(1, 101):
    true_rul = rul_true[rul_true["unit_id"] == uid]["RUL_true"].values[0]
    cycles   = test_lengths[uid]
    all_totals.append(cycles + true_rul)

all_totals = np.array(all_totals)
print(f"Min total lifetime : {all_totals.min():.0f} cycles  (engine {all_totals.argmin()+1})")
print(f"Max total lifetime : {all_totals.max():.0f} cycles  (engine {all_totals.argmax()+1})")
print(f"Mean total lifetime: {all_totals.mean():.0f} cycles")
print(f"Engines < 150 total cycles: {(all_totals < 150).sum()}")
print(f"Engines < 120 total cycles: {(all_totals < 120).sum()}")

# ── plot total lifetime distribution ──────────────────────────────────────────
plt.figure(figsize=(10, 3))
plt.hist(all_totals, bins=30, color="#185FA5", edgecolor="white")
for uid in [67, 79]:
    plt.axvline(all_totals[uid-1], color="red", linestyle="--",
                label=f"Engine {uid} ({all_totals[uid-1]:.0f} cycles)")
plt.xlabel("Total engine lifetime (cycles)")
plt.ylabel("Count")
plt.title("Test engine lifetime distribution  |  red = problem engines")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/engine_lifetime_dist.png", dpi=150)
plt.show()