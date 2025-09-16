import pandas as pd
import numpy as np
import pandas_ta as ta
import pywt

def get_weights_expanding_window(d: float, num_weights: int) -> np.ndarray:
    """
    Generates weights for fractional differentiation using the expanding window method.
    This is based on the recursive formula:
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k
    """
    weights = [1.0]
    for k in range(1, num_weights):
        weight = -weights[-1] * (d - k + 1) / k
        weights.append(weight)
    return np.array(weights)

def fractional_differentiate(series: pd.Series, d: float) -> pd.Series:
    """
    Applies fractional differentiation using a robust, non-FFT, expanding window method.
    """
    n = len(series)
    weights = get_weights_expanding_window(d, n)

    # Reverse weights for convolution-like dot product
    weights_rev = weights[::-1]

    differentiated_values = []
    for i in range(n):
        # The window grows with each step
        window = series.iloc[:i + 1]
        # We need the corresponding slice of reversed weights
        weight_slice = weights_rev[-(i + 1):]
        differentiated_values.append(np.dot(weight_slice, window))

    return pd.Series(
        differentiated_values,
        index=series.index,
        name=f"{series.name}_fracdiff_{d}"
    )

def generate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates technical indicators (RSI, MACD, Bollinger Bands) using pandas-ta.
    """
    print("Generating technical indicators (RSI, MACD, BBands)...")
    df.ta.rsi(close=df['close'], append=True)
    df.ta.macd(close=df['close'], append=True)
    df.ta.bbands(close=df['close'], append=True)
    return df

def generate_rolling_stats(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Generates rolling window statistics (volatility, skewness, kurtosis).
    """
    print(f"Generating rolling statistics with window={window}...")
    returns = df['close'].pct_change()
    df[f'volatility_{window}'] = returns.rolling(window=window).std()
    df[f'skew_{window}'] = df['close'].rolling(window=window).skew()
    df[f'kurtosis_{window}'] = df['close'].rolling(window=window).kurt()
    return df

def generate_wavelet_features(df: pd.DataFrame, wavelet: str = 'db4') -> pd.DataFrame:
    """
    Generates single-level Discrete Wavelet Transform (DWT) coefficients as features.
    """
    print(f"Generating wavelet features with wavelet='{wavelet}'...")
    # Force the creation of a new, writable numpy array to avoid read-only buffer errors.
    ts_values = df['close_fracdiff'].dropna().to_numpy(copy=True)
    if len(ts_values) < 2:
        df['wavelet_cA'] = np.nan
        df['wavelet_cD'] = np.nan
        return df

    cA, cD = pywt.dwt(ts_values, wavelet=wavelet)

    pad_len = len(df) - len(cA)
    df['wavelet_cA'] = np.concatenate(([np.nan] * pad_len, cA))
    df['wavelet_cD'] = np.concatenate(([np.nan] * pad_len, cD))
    return df

def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrator function to generate all features for the model.
    """
    print("\n--- Starting Feature Generation ---")
    df_featured = df.copy()

    df_featured['close_fracdiff'] = fractional_differentiate(df_featured['close'], d=0.5)

    df_featured = generate_technical_indicators(df_featured)
    df_featured = generate_rolling_stats(df_featured)
    df_featured = generate_wavelet_features(df_featured)

    df_featured.dropna(inplace=True)

    print("--- Feature Generation Complete ---")
    return df_featured

if __name__ == '__main__':
    print("\n--- Testing Full Feature Generation ---")
    np.random.seed(42)
    data = {
        'open': np.random.rand(200) * 10 + 100,
        'high': np.random.rand(200) * 10 + 105,
        'low': np.random.rand(200) * 10 + 95,
        'close': np.random.rand(200) * 10 + 100,
        'volume': np.random.rand(200) * 1000,
    }
    index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200, freq='H'))
    sample_df = pd.DataFrame(data, index=index)

    print(f"Original DataFrame shape: {sample_df.shape}")

    try:
        featured_df = generate_all_features(sample_df)

        print(f"DataFrame shape after feature generation: {featured_df.shape}")
        print("\nGenerated Features (last 5 rows):")
        print(featured_df.tail())
        print("\nColumns:")
        print(featured_df.columns.tolist())
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()
