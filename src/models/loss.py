# src/models/loss.py
import torch
import torch.nn as nn


class AsymmetricMSELoss(nn.Module):
    """
    Penalises over-prediction (pred > target) more than under-prediction.
    Mirrors the NASA scoring asymmetry directly in the loss.

    over_penalty  > 1.0  →  model learns to predict conservatively
    """
    def __init__(self, over_penalty: float = 2.0):
        super().__init__()
        self.over_penalty = over_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff    = pred - target                          # positive = over-predicted
        weights = torch.where(diff > 0,
                              torch.full_like(diff, self.over_penalty),
                              torch.ones_like(diff))
        return (weights * diff ** 2).mean()


class HuberAsymmetricLoss(nn.Module):
    """
    Huber base (robust to outlier engines) + asymmetric weighting.
    Best of both worlds.
    """
    def __init__(self, delta: float = 15.0, over_penalty: float = 1.5):
        super().__init__()
        self.delta       = delta
        self.over_penalty = over_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff    = pred - target
        abs_d   = diff.abs()
        huber   = torch.where(abs_d < self.delta,
                              0.5 * diff ** 2,
                              self.delta * (abs_d - 0.5 * self.delta))
        weights = torch.where(diff > 0,
                              torch.full_like(diff, self.over_penalty),
                              torch.ones_like(diff))
        return (weights * huber).mean()