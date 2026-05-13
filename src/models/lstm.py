# src/models/lstm.py
import torch
import torch.nn as nn


class PredictiveMaintenanceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)          # single RUL value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        out, _ = self.lstm(x)         # (batch, seq_len, hidden_size)
        last    = out[:, -1, :]       # (batch, hidden_size) — last timestep
        return self.head(last).squeeze(-1)  # (batch,)