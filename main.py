import pandas as pd
import os
from connectors import KrakenConnector # Using Kraken due to Binance geo-blocking
from storage import save_to_parquet

def transform_data(df: pd.DataFrame, target_interval: str = '1H') -> pd.DataFrame:
    """
    Transforms raw OHLCV data by resampling to a target interval and handling missing values.

    :param df: The input DataFrame with high-frequency data.
    :param target_interval: The target resampling interval (e.g., '1H', '4H').
    :return: The resampled and cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Warning: Input is not a valid, non-empty DataFrame. No transformation will be performed.")
        return pd.DataFrame()

    print(f"Original data points: {len(df)}")
    print(f"Resampling data to '{target_interval}' interval...")

    # Define aggregation rules for resampling
    resample_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Resample the data
    resampled_df = df.resample(target_interval).apply(resample_rules)

    # The user's plan specifies handling missing values. Forward-fill is a reasonable strategy.
    # It propagates the last valid observation forward to the next valid.
    original_len = len(resampled_df)
    resampled_df.dropna(inplace=True) # Drop rows where no data was available to aggregate
    print(f"Data points after resampling and dropping NaN rows: {len(resampled_df)}")

    if resampled_df.empty:
        return resampled_df

    # Let's create a full date range to see if there are gaps to fill
    full_range_df = resampled_df.asfreq(target_interval)
    missing_rows = full_range_df[full_range_df.isnull().any(axis=1)]

    if not missing_rows.empty:
        print(f"Found {len(missing_rows)} missing rows after resampling. Applying forward-fill...")
        resampled_df = resampled_df.asfreq(target_interval, method='ffill')

    print("Transformation complete.")
    return resampled_df

def run_etl(symbol: str, raw_interval: int, target_interval: str, output_dir: str = 'data'):
    """
    Runs the full ETL (Extract, Transform, Load) process using Kraken data.
    """
    print(f"--- Starting ETL for {symbol} ---")

    # 1. EXTRACT
    print(f"Extracting {raw_interval}-minute data from Kraken...")
    connector = KrakenConnector()
    raw_data = connector.fetch_ohlcv(symbol, interval=raw_interval)

    if raw_data.empty:
        print(f"Extraction failed or returned no data for {symbol}. ETL process stopped.")
        return

    print(f"Successfully extracted {len(raw_data)} rows of data.")

    # 2. TRANSFORM
    transformed_data = transform_data(raw_data, target_interval)

    if transformed_data.empty:
        print("Transformation resulted in an empty DataFrame. ETL process stopped.")
        return

    # 3. LOAD
    output_filename = f"{symbol.lower()}_{target_interval.lower()}.parquet"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Loading transformed data to '{output_path}'...")
    save_to_parquet(transformed_data, output_path)

    print(f"--- ETL for {symbol} finished successfully. ---")

if __name__ == '__main__':
    # --- Configuration for Kraken ---
    # Kraken uses 'XBT' for Bitcoin instead of 'BTC'
    SYMBOL_TO_FETCH = 'XBTUSD'
    # Fetch higher frequency data (e.g., 5-minute). Kraken uses integers for minutes.
    RAW_INTERVAL = 5
    # Resample to a lower frequency (e.g., 1-hour)
    TARGET_INTERVAL = '1H'

    run_etl(SYMBOL_TO_FETCH, RAW_INTERVAL, TARGET_INTERVAL)
