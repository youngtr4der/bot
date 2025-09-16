import pandas as pd
import numpy as np
import os

from storage import load_data
from validation import get_purged_walk_forward_splits
from model import build_lstm_model

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(X, y, time_steps=10):
    """
    Converts 2D feature/label data into 3D sequences for LSTM models.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def main():
    """
    Main function to run the full model training and validation pipeline.
    """
    print("--- Starting Phase 3: Model Training Pipeline ---")

    # --- 1. Load Data ---
    data_path = os.path.join('data', 'featured_labeled_data.parquet')
    df = load_data(data_path)
    if df.empty:
        print("Loaded data is empty. Stopping.")
        return

    X = df.drop(columns=['label', 'touch_time'])
    y = df['label']
    touch_times = df['touch_time']

    # --- 2. Set up Cross-Validation ---
    n_splits = 4
    embargo_size = 5
    time_steps = 10
    splits = get_purged_walk_forward_splits(X, touch_times, n_splits=n_splits, embargo_size=embargo_size)

    all_scores = []

    # --- 3. Loop Through CV Splits ---
    for fold, (train_indices, val_indices) in enumerate(splits):
        print(f"\n========== FOLD {fold + 1}/{n_splits} ==========")

        X_train, X_val = X.loc[train_indices], X.loc[val_indices]
        y_train, y_val = y.loc[train_indices], y.loc[val_indices]

        # --- 4. Scale Features ---
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)

        # --- 5. One-Hot Encode Labels ---
        y_train_cat = to_categorical(y_train + 1, num_classes=3)
        y_val_cat = to_categorical(y_val + 1, num_classes=3)
        y_train_cat_series = pd.Series(list(y_train_cat), index=y_train.index)
        y_val_cat_series = pd.Series(list(y_val_cat), index=y_val.index)

        # --- 6. Create 3D Sequences ---
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_cat_series, time_steps)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_cat_series, time_steps)

        if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
            print("Not enough data to create sequences in this fold. Skipping.")
            continue

        # --- 7. Build and Train Model ---
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = build_lstm_model(input_shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train_seq, y_train_seq,
            epochs=50, batch_size=8,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[early_stopping], verbose=0 # Set verbose to 0 to reduce log spam
        )

        # --- 8. Evaluate ---
        print("--- Evaluating on Validation Set ---")
        loss, accuracy = model.evaluate(X_val_seq, y_val_seq, verbose=0)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        all_scores.append(accuracy)

    print("\n--- Cross-Validation Complete ---")
    if all_scores:
        print(f"Average CV Accuracy: {np.mean(all_scores):.4f}")

if __name__ == '__main__':
    main()
