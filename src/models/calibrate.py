# src/models/calibrate.py
"""
Learns a simple isotonic regression calibration on val set predictions.
Corrects systematic over-prediction, especially on short-RUL engines.
"""
import numpy as np
import pickle
import torch
from pathlib import Path
from sklearn.isotonic import IsotonicRegression


def collect_val_predictions(model, val_loader, device="cpu"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            preds.append(model(X.to(device)).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def fit_calibrator(raw_preds: np.ndarray, targets: np.ndarray):
    """
    Isotonic regression: fits a monotone mapping raw_pred → calibrated_pred.
    Naturally corrects the over-prediction bias without hard-coding rules.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_preds, targets)
    return calibrator


def save_calibrator(calibrator, path="data/processed/calibrator.pkl"):
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved → {path}")


def load_calibrator(path="data/processed/calibrator.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def calibrate(raw_preds: np.ndarray, calibrator) -> np.ndarray:
    return calibrator.predict(raw_preds)