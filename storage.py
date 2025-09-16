import pandas as pd
import os

def save_to_parquet(df: pd.DataFrame, file_path: str):
    """
    Saves a pandas DataFrame to a Parquet file.
    Creates the directory if it does not exist.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Warning: Input is not a valid, non-empty DataFrame. Nothing to save.")
        return

    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        df.to_parquet(file_path, index=True)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving to Parquet file {file_path}: {e}")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a Parquet file, handling potential errors.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'.")
        return pd.DataFrame()

    try:
        print(f"Loading data from '{file_path}'...")
        df = pd.read_parquet(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"An error occurred while loading the Parquet file: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print("--- Testing Storage Module (Save and Load) ---")

    data = {
        'open': [100, 102], 'high': [103, 104],
        'low': [99, 101], 'close': [102, 103],
        'volume': [1000, 1200]
    }
    index = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00'])
    dummy_df = pd.DataFrame(data, index=index)

    output_path = os.path.join('data', 'storage_test.parquet')

    # Test saving
    save_to_parquet(dummy_df, output_path)

    # Test loading
    loaded_df = load_data(output_path)

    if not loaded_df.empty and dummy_df.equals(loaded_df):
        print("\nVerification successful: Saved and loaded DataFrames are identical.")
    else:
        print("\nVerification failed: DataFrames do not match or loading failed.")

    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Cleaned up test file '{output_path}'.")
