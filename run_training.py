# run_training.py
from src.data.preprocess  import preprocess
from src.data.windowing    import get_loaders
from src.models.train      import train
from src.models.evaluate   import evaluate_test
from src.models.calibrate  import (collect_predictions, fit_calibrator,
                                    save_calibrator, calibrate)

CONFIG = {
    "hidden_size"  : 128,
    "num_layers"   : 2,
    "dropout"      : 0.3,
    "lr"           : 5e-4,
    "weight_decay" : 1e-5,
    "epochs"       : 120,
    "patience"     : 15,
    "seq_len"      : 50,
    "batch_size"   : 256,
    "loss"         : "huber_asymmetric",
    "over_penalty" : 1.5,
    "huber_delta"  : 12.0,
}

if __name__ == "__main__":
    train_df, test_df, feature_cols = preprocess("FD001")

    train_loader, val_loader, test_loader, n_features = get_loaders(
        train_df, test_df, feature_cols,
        seq_len    = CONFIG["seq_len"],
        batch_size = CONFIG["batch_size"],
        stratified = False,
    )

    model = train(train_loader, val_loader, n_features, CONFIG)

    # fit bias calibrator on val
    print("\nFitting bias calibrator on val set...")
    raw_val, val_targets = collect_predictions(model, val_loader)
    calibrator = fit_calibrator(raw_val, val_targets)
    save_calibrator(calibrator)

    # evaluate both ways
    evaluate_test(model, test_loader, calibrator=calibrator)