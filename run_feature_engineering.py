import pandas as pd
import os
from feature_engineering import generate_all_features
from labeling import get_triple_barrier_labels
from storage import save_to_parquet

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a Parquet file, handling potential errors.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'. Please ensure Phase 1 was completed successfully.")
        return pd.DataFrame()

    try:
        print(f"Loading data from '{file_path}'...")
        df = pd.read_parquet(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"An error occurred while loading the Parquet file: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to run the full feature engineering and labeling pipeline.
    """
    print("--- Starting Phase 2: Feature Engineering & Labeling ---")

    # Define paths
    input_file = os.path.join('data', 'xbtusd_1h.parquet')
    output_file = os.path.join('data', 'featured_labeled_data.parquet')

    # Load the raw data from Phase 1
    price_data = load_data(input_file)
    if price_data.empty:
        print("Stopping pipeline due to data loading failure.")
        return

    # --- 1. Generate Features ---
    # This function creates technical indicators, stats, etc.
    featured_data = generate_all_features(price_data)
    if featured_data.empty:
        print("Stopping pipeline after feature generation resulted in an empty DataFrame.")
        return

    # --- 2. Generate Labels ---
    # Labels are generated based on the original price data to avoid lookahead bias from features.
    print("\n--- Starting Label Generation ---")
    labels = get_triple_barrier_labels(
        prices=price_data,
        holding_period=24, # e.g., hold for a maximum of 24 hours
        tp_multiplier=2.0, # Take profit at 2x ATR
        sl_multiplier=1.5  # Stop loss at 1.5x ATR
    )
    labels.name = 'label'

    # --- 3. Align Features and Labels ---
    print("\n--- Aligning Features and Labels ---")
    # An inner join ensures that we only keep timestamps for which we have both valid features and a valid label.
    final_df = featured_data.join(labels, how='inner')

    if final_df.empty:
        print("Final DataFrame is empty after aligning features and labels. Nothing to save.")
        return

    print(f"Final DataFrame created with shape: {final_df.shape}")
    print("Final columns:", final_df.columns.tolist())

    # --- 4. Save the Final Dataset ---
    print(f"\n--- Saving Final Dataset to '{output_file}' ---")
    save_to_parquet(final_df, output_file)

    print("\n--- Phase 2: Feature Engineering and Labeling Pipeline Complete! ---")


if __name__ == "__main__":
    main()
