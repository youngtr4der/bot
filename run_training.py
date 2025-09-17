import pandas as pd
import numpy as np
import os
import joblib

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

    # --- 2. Set up and Run Cross-Validation ---
    n_splits = 4
    embargo_size = 5
    time_steps = 10
    splits = get_purged_walk_forward_splits(X, touch_times, n_splits=n_splits, embargo_size=embargo_size)

    all_scores = []

    for fold, (train_indices, val_indices) in enumerate(splits):
        print(f"\n========== FOLD {fold + 1}/{n_splits} ==========")
        X_train, X_val = X.loc[train_indices], X.loc[val_indices]
        y_train, y_val = y.loc[train_indices], y.loc[val_indices]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)

        y_train_cat = to_categorical(y_train + 1, num_classes=3)
        y_val_cat = to_categorical(y_val + 1, num_classes=3)
        y_train_cat_series = pd.Series(list(y_train_cat), index=y_train.index)
        y_val_cat_series = pd.Series(list(y_val_cat), index=y_val.index)

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_cat_series, time_steps)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_cat_series, time_steps)

        if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
            print("Not enough data to create sequences in this fold. Skipping.")
            continue

        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = build_lstm_model(input_shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train_seq, y_train_seq, epochs=50, batch_size=8,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[early_stopping], verbose=0
        )

        print("--- Evaluating on Validation Set ---")
        loss, accuracy = model.evaluate(X_val_seq, y_val_seq, verbose=0)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        all_scores.append(accuracy)

    print("\n--- Cross-Validation Complete ---")
    if all_scores:
        print(f"Average CV Accuracy: {np.mean(all_scores):.4f}")

    # --- 9. Train and Save Final Model on All Data ---
    print("\n--- Training Final Model on All Data ---")

    final_scaler = StandardScaler()
    X_scaled = pd.DataFrame(final_scaler.fit_transform(X), index=X.index, columns=X.columns)

    y_shifted = y + 1
    y_cat = to_categorical(y_shifted, num_classes=3)
    y_cat_series = pd.Series(list(y_cat), index=y.index)
    X_seq, y_seq = create_sequences(X_scaled, y_cat_series, time_steps)

    if X_seq.shape[0] > 0:
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        final_model = build_lstm_model(input_shape)

        print("Training final model for 15 epochs...")
        final_model.fit(X_seq, y_seq, epochs=15, batch_size=8, verbose=0)

        print("Saving final model to 'final_model.keras' and scaler to 'final_scaler.joblib'...")
        final_model.save('final_model.keras')
        joblib.dump(final_scaler, 'final_scaler.joblib')
        print("Final model and scaler saved successfully.")
    else:
        print("Not enough data to train a final model.")

if __name__ == '__main__':
    main()
