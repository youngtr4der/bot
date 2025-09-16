import pandas as pd
import os
import time
from connectors import KrakenConnector # Using Kraken due to Binance geo-blocking
from storage import save_to_parquet

def transform_data(df: pd.DataFrame, target_interval: str = '1H') -> pd.DataFrame:
    """
    Transforms raw OHLCV data by resampling to a target interval and handling missing values.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Warning: Input is not a valid, non-empty DataFrame. No transformation will be performed.")
        return pd.DataFrame()

    print(f"Original data points: {len(df)}")
    print(f"Resampling data to '{target_interval}' interval...")

    resample_rules = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }
    resampled_df = df.resample(target_interval).apply(resample_rules)

    resampled_df.dropna(inplace=True)
    print(f"Data points after resampling and dropping NaN rows: {len(resampled_df)}")

    if resampled_df.empty:
        return resampled_df

    # Forward-fill any gaps
    full_range_df = resampled_df.asfreq(target_interval)
    if full_range_df.isnull().values.any():
        resampled_df = resampled_df.asfreq(target_interval, method='ffill')

    print("Transformation complete.")
    return resampled_df

def run_etl(symbol: str, raw_interval: int, target_interval: str, output_dir: str = 'data', n_pages: int = 8):
    """
    Runs the full ETL process, fetching multiple pages of historical data.
    """
    print(f"--- Starting ETL for {symbol} ---")

    # 1. EXTRACT
    print(f"Extracting {n_pages} pages of {raw_interval}-minute data from Kraken...")
    connector = KrakenConnector()

    all_dfs = []
    last_timestamp = None

    for i in range(n_pages):
        print(f"Fetching page {i+1}/{n_pages}...")
        df, last_timestamp = connector.fetch_ohlcv(symbol, interval=raw_interval, since=last_timestamp)

        if df.empty:
            print("Received empty data, stopping pagination.")
            break

        all_dfs.append(df)

        if last_timestamp is None:
            break # No more data to paginate

        time.sleep(1) # Be respectful to the API

    if not all_dfs:
        print("Extraction failed or returned no data. ETL process stopped.")
        return

    # Combine all pages and remove duplicates
    raw_data = pd.concat(all_dfs)
    raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
    raw_data.sort_index(inplace=True)

    print(f"Successfully extracted a total of {len(raw_data)} rows of data.")

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
    SYMBOL_TO_FETCH = 'XBTUSD'
    # Fetch 1-hour data
    RAW_INTERVAL = 60
    # Resample to 2-hour bars to get a decent number of samples (~360)
    TARGET_INTERVAL = '2H'

    # Run with a single page, as pagination logic is flawed for historical data.
    run_etl(SYMBOL_TO_FETCH, RAW_INTERVAL, TARGET_INTERVAL, n_pages=1)
