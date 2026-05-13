# verify_preprocessing.py  — run this to confirm everything works
from src.data.preprocess import preprocess
from src.data.windowing import get_loaders

train_df, test_df, feature_cols = preprocess("FD001")
train_loader, val_loader, test_loader, n_features = get_loaders(
    train_df, test_df, feature_cols, seq_len=30, batch_size=256
)

# Grab one batch and inspect
X_batch, y_batch = next(iter(train_loader))
print(f"\nBatch X shape : {X_batch.shape}")   # expect (256, 30, n_features)
print(f"Batch y shape : {y_batch.shape}")     # expect (256,)
print(f"RUL range     : {y_batch.min():.1f} – {y_batch.max():.1f}")
print(f"n_features    : {n_features}")