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


# src/models/evaluate.py  — add this function, then call it from evaluate_test

@torch.no_grad()
def mc_dropout_predict(model, X: torch.Tensor, n_passes: int = 10) -> torch.Tensor:
    """Monte Carlo dropout — keep dropout ON during inference, average passes."""
    model.train()   # dropout active
    preds = torch.stack([model(X) for _ in range(n_passes)], dim=0)  # (n_passes, batch)
    model.eval()
    return preds.mean(dim=0), preds.std(dim=0)   # mean prediction + uncertainty


# src/models/evaluate.py — update evaluate_test signature
@torch.no_grad()
def evaluate_test(model, test_loader, rul_cap=125, calibrator=None):
    model.eval().to(DEVICE)
    all_preds, all_targets = [], []

    for X, y in test_loader:
        pred = model(X.to(DEVICE)).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # apply calibration if provided
    if calibrator is not None:
        preds_cal = calibrator.predict(preds)
        print("\n── Uncalibrated ──────────────────────────────")
        _print_metrics(preds, targets)
        print("\n── Calibrated ────────────────────────────────")
        _print_metrics(preds_cal, targets)
        preds = preds_cal
    else:
        print("\n── Test Results ──────────────────────────────")
        _print_metrics(preds, targets)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(targets, preds, alpha=0.7, s=25, color="#185FA5")
    axes[0].plot([0, rul_cap], [0, rul_cap], "r--", linewidth=1)
    axes[0].set_xlabel("True RUL"); axes[0].set_ylabel("Predicted RUL")
    axes[0].set_title(f"Predicted vs True RUL")

    errors = preds - targets
    axes[1].hist(errors, bins=25, color="#3B6D11", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].axvline(errors.mean(), color="orange", linestyle="--",
                    label=f"Mean={errors.mean():.1f}")
    axes[1].set_xlabel("Error (pred − true)"); axes[1].legend()
    axes[1].set_title("Error distribution")

    plt.tight_layout()
    plt.savefig("notebooks/test_results.png", dpi=150)
    plt.show()
    return preds, targets


def _print_metrics(preds, targets):
    rmse_val = np.sqrt(np.mean((preds - targets) ** 2))
    mae_val  = np.mean(np.abs(preds - targets))
    score    = nasa_scoring(preds, targets)
    over     = (preds > targets).sum()
    print(f"  RMSE       : {rmse_val:.4f}")
    print(f"  MAE        : {mae_val:.4f}")
    print(f"  NASA score : {score:.2f}")
    print(f"  Over-preds : {over}/100 engines")