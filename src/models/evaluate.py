# src/models/evaluate.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_test(model, test_loader, rul_cap=125):
    model.eval().to(DEVICE)
    all_preds, all_targets = [], []

    for X, y in test_loader:
        pred = model(X.to(DEVICE)).cpu()
        all_preds.append(pred)
        all_targets.append(y)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae  = mean_absolute_error(targets, preds)
    score = nasa_scoring(preds, targets)   # industry standard metric

    print(f"\n── Test Results ──────────────────────────────")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  NASA score : {score:.2f}  (lower is better)")

    # Predicted vs actual plot
    plt.figure(figsize=(10, 4))
    plt.scatter(targets, preds, alpha=0.6, s=20, color="#185FA5")
    plt.plot([0, rul_cap], [0, rul_cap], "r--", linewidth=1, label="Perfect prediction")
    plt.xlabel("True RUL"); plt.ylabel("Predicted RUL")
    plt.title(f"Predicted vs True RUL  |  RMSE={rmse:.2f}  MAE={mae:.2f}")
    plt.legend(); plt.tight_layout()
    plt.savefig("notebooks/test_results.png", dpi=150)
    plt.show()

    return preds, targets, rmse, mae


def nasa_scoring(preds, targets):
    """
    NASA's asymmetric scoring function — penalises late predictions
    (predicting failure AFTER it happens) more than early ones.
    """
    d = preds - targets
    score = np.where(d < 0,
                     np.exp(-d / 13) - 1,
                     np.exp( d / 10) - 1)
    return np.sum(score)