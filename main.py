import pandas as pd
import os
import time
from connectors import KrakenConnector
from storage import save_to_parquet

def transform_data(df: pd.DataFrame, target_interval: str = '1H') -> pd.DataFrame:
    """
    Transforms raw OHLCV data by resampling to a target interval and handling missing values.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    print(f"Resampling data to '{target_interval}' interval...")

    resample_rules = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }
    resampled_df = df.resample(target_interval).apply(resample_rules)
    resampled_df.dropna(inplace=True)

    if resampled_df.empty:
        return resampled_df

    # Forward-fill any gaps
    full_range_df = resampled_df.asfreq(target_interval)
    if full_range_df.isnull().values.any():
        resampled_df = resampled_df.asfreq(target_interval, method='ffill')

    return resampled_df

def run_etl(symbol: str, raw_interval: int, target_interval: str, output_dir: str = 'data'):
    """
    Runs the full ETL process using Kraken data.
    """
    print(f"--- Starting ETL for {symbol} ---")

    connector = KrakenConnector()
    # The fetch_ohlcv now returns a tuple (df, last_timestamp), we only need the df here.
    raw_data, _ = connector.fetch_ohlcv(symbol, interval=raw_interval)

    if raw_data.empty:
        print(f"Extraction failed or returned no data for {symbol}. ETL process stopped.")
        return

    print(f"Successfully extracted {len(raw_data)} rows of data.")

    transformed_data = transform_data(raw_data, target_interval)

    if transformed_data.empty:
        print("Transformation resulted in an empty DataFrame. ETL process stopped.")
        return

    output_filename = f"{symbol.lower()}_{target_interval.lower()}.parquet"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Loading transformed data to '{output_path}'...")
    save_to_parquet(transformed_data, output_path)

    print(f"--- ETL for {symbol} finished successfully. ---")

if __name__ == '__main__':
    SYMBOL_TO_FETCH = 'XBTUSD'
    RAW_INTERVAL = 60 # 1 hour
    TARGET_INTERVAL = '2H' # Resample to 2-hour bars

    run_etl(SYMBOL_TO_FETCH, RAW_INTERVAL, TARGET_INTERVAL)
