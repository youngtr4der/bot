import pandas as pd
import os

def save_to_parquet(df: pd.DataFrame, file_path: str):
    """
    Saves a pandas DataFrame to a Parquet file.
    Creates the directory if it does not exist.

    :param df: The DataFrame to save.
    :param file_path: The full path (including filename) for the output Parquet file.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Warning: Input is not a valid, non-empty DataFrame. Nothing to save.")
        return

    try:
        # Ensure the target directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Save the DataFrame to a Parquet file
        df.to_parquet(file_path, index=True)
        print(f"Successfully saved data to {file_path}")

    except Exception as e:
        print(f"An error occurred while saving to Parquet file {file_path}: {e}")

if __name__ == '__main__':
    # Example Usage for demonstrating the function
    print("Running storage.py example...")

    # 1. Create a dummy DataFrame
    data = {
        'open': [100, 102, 101, 105],
        'high': [103, 104, 102, 106],
        'low': [99, 101, 100, 104],
        'close': [102, 103, 101, 105],
        'volume': [1000, 1200, 1100, 1500]
    }
    index = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00', '2023-01-01 03:00:00'])
    dummy_df = pd.DataFrame(data, index=index)
    dummy_df.index.name = 'open_time'

    # 2. Define an output path in a 'data' subdirectory
    output_path = os.path.join('data', 'test_data.parquet')

    # 3. Use the function to save the data
    save_to_parquet(dummy_df, output_path)

    # 4. Verify that the file was created
    if os.path.exists(output_path):
        print(f"\nVerification successful: File '{output_path}' was created.")
        # Optional: Clean up the created file and directory
        try:
            os.remove(output_path)
            os.rmdir('data')
            print("Cleaned up test file and directory.")
        except OSError as e:
            print(f"Error during cleanup: {e}")
    else:
        print(f"\nVerification failed: File '{output_path}' was not created.")
