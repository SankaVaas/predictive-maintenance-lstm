# src/models/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import mlflow
import mlflow.pytorch
from src.models.lstm import PredictiveMaintenanceLSTM
from src.models.loss import HuberAsymmetricLoss, AsymmetricMSELoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(nn.functional.mse_loss(preds, targets)).item()


def get_criterion(config: dict):
    name = config.get("loss", "mse")
    if name == "huber_asymmetric":
        return HuberAsymmetricLoss(
            delta=config.get("huber_delta", 15.0),
            over_penalty=config.get("over_penalty", 1.5)
        )
    elif name == "asymmetric_mse":
        return AsymmetricMSELoss(over_penalty=config.get("over_penalty", 2.0))
    else:
        return nn.MSELoss()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item() * len(y)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return total_loss / len(loader.dataset), rmse(preds, targets)


def train(train_loader: DataLoader, val_loader: DataLoader,
          n_features: int, config: dict) -> PredictiveMaintenanceLSTM:

    model = PredictiveMaintenanceLSTM(
        input_size  = n_features,
        hidden_size = config["hidden_size"],
        num_layers  = config["num_layers"],
        dropout     = config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["lr"],
                                 weight_decay=config["weight_decay"])

    # cosine annealing — LR decays smoothly to near-zero, no sudden drops
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )

    criterion = get_criterion(config)

    best_val_rmse    = float("inf")
    patience_counter = 0
    best_weights     = None

    print(f"\nTraining on : {DEVICE}")
    print(f"Loss        : {config.get('loss', 'mse')}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    mlflow.set_experiment("predictive_maintenance")
    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(1, config["epochs"] + 1):
            train_loss          = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_rmse  = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            mlflow.log_metrics({"train_loss": train_loss,
                                 "val_loss":   val_loss,
                                 "val_rmse":   val_rmse}, step=epoch)

            print(f"Epoch {epoch:3d}  |  train_loss: {train_loss:.4f}  "
                  f"val_loss: {val_loss:.4f}  val_rmse: {val_rmse:.4f}")

            if val_rmse < best_val_rmse - 0.001:
                best_val_rmse    = val_rmse
                best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"\nEarly stopping at epoch {epoch}  "
                          f"|  best val RMSE: {best_val_rmse:.4f}")
                    break

        model.load_state_dict(best_weights)
        mlflow.pytorch.log_model(model, "model")

        ckpt_path = Path("data/processed/best_model.pt")
        torch.save({"model_state": best_weights,
                    "config":      config,
                    "n_features":  n_features}, ckpt_path)
        print(f"\nCheckpoint saved → {ckpt_path}")

    return model