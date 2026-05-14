# diagnose.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from src.data.preprocess import preprocess
from src.data.windowing  import get_loaders
from src.models.lstm     import PredictiveMaintenanceLSTM

# ── load ──────────────────────────────────────────────────────────────────────
CKPT      = torch.load("data/processed/best_model.pt", map_location="cpu", weights_only=False)
CONFIG    = CKPT["config"]
N_FEAT    = CKPT["n_features"]

model = PredictiveMaintenanceLSTM(N_FEAT, CONFIG["hidden_size"],
                                   CONFIG["num_layers"], CONFIG["dropout"])
model.load_state_dict(CKPT["model_state"])
model.eval()

train_df, test_df, feature_cols = preprocess("FD001", save=False)
_, _, test_loader, _ = get_loaders(train_df, test_df, feature_cols,
                                    seq_len=CONFIG["seq_len"], batch_size=256)

# ── collect per-engine predictions ────────────────────────────────────────────
with torch.no_grad():
    all_preds, all_targets = [], []
    for X, y in test_loader:
        all_preds.append(model(X).numpy())
        all_targets.append(y.numpy())

preds   = np.concatenate(all_preds)
targets = np.concatenate(all_targets)
errors  = preds - targets   # positive = over-predicted (bad for NASA score)

# ── per-engine breakdown ───────────────────────────────────────────────────────
print(f"\n{'Eng':>4}  {'True RUL':>9}  {'Pred RUL':>9}  {'Error':>8}  {'NASA contrib':>13}")
print("─" * 55)
nasa_contribs = np.where(errors < 0,
                          np.exp(-errors / 13) - 1,
                          np.exp( errors / 10) - 1)
order = np.argsort(nasa_contribs)[::-1]   # worst first
for i in order[:20]:                       # top 20 worst engines
    print(f"{i+1:>4}  {targets[i]:>9.1f}  {preds[i]:>9.1f}  "
          f"{errors[i]:>+8.1f}  {nasa_contribs[i]:>13.2f}")

print(f"\nTotal NASA score : {nasa_contribs.sum():.2f}")
print(f"Top 5 engines contribute : {nasa_contribs[order[:5]].sum():.2f}  "
      f"({100*nasa_contribs[order[:5]].sum()/nasa_contribs.sum():.0f}% of total)")

# ── visualise error distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(targets, preds, alpha=0.6, s=20, c=errors, cmap="RdYlGn_r")
axes[0].plot([0,125],[0,125],"r--")
axes[0].set_xlabel("True RUL"); axes[0].set_ylabel("Pred RUL")
axes[0].set_title("Predicted vs True (colour = error)")

axes[1].bar(range(len(nasa_contribs)), nasa_contribs[order], color="#185FA5")
axes[1].set_xlabel("Engine rank (worst first)")
axes[1].set_ylabel("NASA score contribution")
axes[1].set_title("Per-engine NASA contribution")

axes[2].hist(errors, bins=25, color="#3B6D11", edgecolor="white")
axes[2].axvline(0, color="red", linestyle="--", label="Zero error")
axes[2].axvline(errors.mean(), color="orange", linestyle="--",
                label=f"Mean={errors.mean():.1f}")
axes[2].set_xlabel("Error (pred − true)"); axes[2].legend()
axes[2].set_title("Error distribution")

plt.tight_layout()
plt.savefig("notebooks/diagnosis.png", dpi=150)
plt.show()

# ── key insight ────────────────────────────────────────────────────────────────
over  = (errors > 0).sum()
under = (errors < 0).sum()
print(f"\nOver-predictions  (bad): {over}  engines")
print(f"Under-predictions (ok) : {under}  engines")
print(f"Mean error             : {errors.mean():+.2f}  (negative = conservative = good)")