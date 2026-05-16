# quick_eval.py  — run this instead of full run_training.py
import torch
import numpy as np
from src.data.preprocess  import preprocess
from src.data.windowing    import get_loaders
from src.models.lstm       import PredictiveMaintenanceLSTM
from src.models.evaluate   import evaluate_test
from src.models.calibrate  import load_calibrator

CKPT  = torch.load("data/processed/best_model.pt", map_location="cpu", weights_only=False)
CONFIG = CKPT["config"]

model = PredictiveMaintenanceLSTM(CKPT["n_features"], CONFIG["hidden_size"],
                                   CONFIG["num_layers"], CONFIG["dropout"])
model.load_state_dict(CKPT["model_state"])

train_df, test_df, feature_cols = preprocess("FD001", save=False)
_, _, test_loader, _ = get_loaders(train_df, test_df, feature_cols,
                                    seq_len=CONFIG["seq_len"], batch_size=256)

calibrator = load_calibrator()
evaluate_test(model, test_loader, calibrator=calibrator)