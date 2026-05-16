# src/models/evaluate.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nasa_scoring(preds, targets):
    """
    NASA's asymmetric scoring function.
    Penalises late predictions (over-predicting RUL) more than early ones.
    """
    d = preds - targets
    score = np.where(d < 0,
                     np.exp(-d / 13) - 1,
                     np.exp( d / 10) - 1)
    return np.sum(score)


def _print_metrics(preds, targets):
    rmse_val = np.sqrt(np.mean((preds - targets) ** 2))
    mae_val  = np.mean(np.abs(preds - targets))
    score    = nasa_scoring(preds, targets)
    over     = (preds > targets).sum()
    mean_err = np.mean(preds - targets)
    print(f"  RMSE       : {rmse_val:.4f}")
    print(f"  MAE        : {mae_val:.4f}")
    print(f"  NASA score : {score:.2f}  (lower is better)")
    print(f"  Over-preds : {over}/{len(preds)} engines")
    print(f"  Mean error : {mean_err:+.4f}  (negative = conservative = good)")


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

    # ── metrics ───────────────────────────────────────────────────────────────
    if calibrator is not None:
        from src.models.calibrate import calibrate as apply_calibration
        preds_cal = apply_calibration(preds, calibrator)
        print("\n── Uncalibrated ──────────────────────────────")
        _print_metrics(preds, targets)
        print("\n── Calibrated ────────────────────────────────")
        _print_metrics(preds_cal, targets)
        preds = preds_cal
    else:
        print("\n── Test Results ──────────────────────────────")
        _print_metrics(preds, targets)

    # ── per-engine breakdown (top 10 worst) ───────────────────────────────────
    errors        = preds - targets
    nasa_contribs = np.where(errors < 0,
                              np.exp(-errors / 13) - 1,
                              np.exp( errors / 10) - 1)
    order = np.argsort(nasa_contribs)[::-1]

    print(f"\n── Top 10 worst engines ──────────────────────")
    print(f"{'Eng':>4}  {'True RUL':>9}  {'Pred RUL':>9}  {'Error':>8}  {'NASA contrib':>13}")
    print("─" * 55)
    for i in order[:10]:
        print(f"{i+1:>4}  {targets[i]:>9.1f}  {preds[i]:>9.1f}  "
              f"{errors[i]:>+8.1f}  {nasa_contribs[i]:>13.2f}")
    top5_pct = 100 * nasa_contribs[order[:5]].sum() / nasa_contribs.sum()
    print(f"\nTop 5 engines contribute: {nasa_contribs[order[:5]].sum():.2f}  "
          f"({top5_pct:.0f}% of total score)")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. predicted vs true
    sc = axes[0].scatter(targets, preds, alpha=0.7, s=25,
                          c=errors, cmap="RdYlGn_r", vmin=-40, vmax=40)
    axes[0].plot([0, rul_cap], [0, rul_cap], "r--", linewidth=1, label="Perfect")
    plt.colorbar(sc, ax=axes[0], label="Error (pred−true)")
    axes[0].set_xlabel("True RUL")
    axes[0].set_ylabel("Predicted RUL")
    rmse_val = np.sqrt(np.mean((preds - targets) ** 2))
    axes[0].set_title(f"Predicted vs True  |  RMSE={rmse_val:.2f}")
    axes[0].legend()

    # 2. error distribution
    axes[1].hist(errors, bins=25, color="#3B6D11", edgecolor="white")
    axes[1].axvline(0, color="red",    linestyle="--", label="Zero error")
    axes[1].axvline(errors.mean(), color="orange", linestyle="--",
                     label=f"Mean={errors.mean():+.1f}")
    axes[1].set_xlabel("Error (pred − true)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error distribution")
    axes[1].legend()

    # 3. per-engine NASA contribution (sorted)
    axes[2].bar(range(len(nasa_contribs)),
                nasa_contribs[order], color="#185FA5", width=1.0)
    axes[2].set_xlabel("Engine rank (worst → best)")
    axes[2].set_ylabel("NASA score contribution")
    axes[2].set_title("Per-engine NASA contribution")

    plt.tight_layout()
    plt.savefig("notebooks/test_results.png", dpi=150)
    plt.show()

    return preds, targets