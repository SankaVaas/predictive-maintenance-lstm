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
    
# src/models/loss.py  — add WeightedHuberAsymmetricLoss at the bottom

class WeightedHuberAsymmetricLoss(nn.Module):
    """
    Huber + asymmetric penalty + extra weight on low-RUL windows.
    Low RUL windows matter most for NASA score — weight them higher.
    """
    def __init__(self, delta=12.0, over_penalty=1.5,
                 low_rul_threshold=50, low_rul_weight=3.0):
        super().__init__()
        self.delta             = delta
        self.over_penalty      = over_penalty
        self.low_rul_threshold = low_rul_threshold
        self.low_rul_weight    = low_rul_weight

    def forward(self, pred, target):
        diff  = pred - target
        abs_d = diff.abs()

        # huber base
        huber = torch.where(abs_d < self.delta,
                            0.5 * diff ** 2,
                            self.delta * (abs_d - 0.5 * self.delta))

        # asymmetric weight
        asym  = torch.where(diff > 0,
                            torch.full_like(diff, self.over_penalty),
                            torch.ones_like(diff))

        # low-RUL weight — penalise errors near failure more
        rul_w = torch.where(target < self.low_rul_threshold,
                            torch.full_like(target, self.low_rul_weight),
                            torch.ones_like(target))

        return (rul_w * asym * huber).mean()


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