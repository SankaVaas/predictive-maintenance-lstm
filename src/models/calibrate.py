# src/models/calibrate.py — replace everything with this simpler version
import numpy as np
import pickle
import torch


def collect_predictions(model, loader, device="cpu"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def fit_calibrator(raw_preds, targets):
    """Simple bias correction: subtract mean over-prediction."""
    bias = np.mean(raw_preds - targets)   # e.g. +3.2 means model over-predicts by 3.2
    print(f"  Val bias (mean error): {bias:+.4f}")
    return {"type": "bias", "bias": bias}


def save_calibrator(calibrator, path="data/processed/calibrator.pkl"):
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"  Calibrator saved → {path}")


def load_calibrator(path="data/processed/calibrator.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def calibrate(raw_preds: np.ndarray, calibrator: dict) -> np.ndarray:
    if calibrator["type"] == "bias":
        corrected = raw_preds - calibrator["bias"]
        return np.clip(corrected, 0, 125)   # never predict negative RUL or > cap
    return raw_preds