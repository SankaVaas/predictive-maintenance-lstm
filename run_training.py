# run_training.py
from src.data.preprocess    import preprocess
from src.data.windowing      import get_loaders
from src.models.train        import train
from src.models.evaluate     import evaluate_test
from src.models.calibrate    import (collect_val_predictions,
                                     fit_calibrator, save_calibrator,
                                     load_calibrator, calibrate)
import numpy as np
import torch

CONFIG = {
    "hidden_size"  : 128,
    "num_layers"   : 2,
    "dropout"      : 0.3,
    "lr"           : 5e-4,
    "weight_decay" : 1e-5,
    "epochs"       : 150,
    "patience"     : 20,
    "seq_len"      : 50,
    "batch_size"   : 256,
    "loss"         : "huber_asymmetric",
    "over_penalty" : 4.0,
    "huber_delta"  : 10.0,
}

if __name__ == "__main__":
    train_df, test_df, feature_cols = preprocess("FD001")

    train_loader, val_loader, test_loader, n_features = get_loaders(
        train_df, test_df, feature_cols,
        seq_len    = CONFIG["seq_len"],
        batch_size = CONFIG["batch_size"],
        stratified = False,          # random split — best generalisation
    )

    # ── train ─────────────────────────────────────────────────────────────────
    model = train(train_loader, val_loader, n_features, CONFIG)

    # ── fit calibrator on val set ──────────────────────────────────────────────
    print("\nFitting calibrator on val predictions...")
    raw_val_preds, val_targets = collect_val_predictions(model, val_loader)
    calibrator = fit_calibrator(raw_val_preds, val_targets)
    save_calibrator(calibrator)

    # ── evaluate with calibration ──────────────────────────────────────────────
    evaluate_test(model, test_loader, calibrator=calibrator)