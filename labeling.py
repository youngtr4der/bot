import pandas as pd
import numpy as np
import pandas_ta as ta

def get_triple_barrier_labels(
    prices: pd.DataFrame,
    holding_period: int,
    tp_multiplier: float,
    sl_multiplier: float,
    atr_period: int = 14
) -> pd.Series:
    """
    Implements the Triple Barrier Method for labeling financial time series data.

    This method sets three barriers for each entry point:
    1. Upper Barrier: Take-profit level.
    2. Lower Barrier: Stop-loss level.
    3. Vertical Barrier: Maximum holding period before exiting.

    The label is determined by which barrier is hit first.

    :param prices: DataFrame with 'high', 'low', 'close' columns and a DatetimeIndex.
    :param holding_period: Maximum number of bars to hold a position (vertical barrier).
    :param tp_multiplier: Multiplier for ATR to set the take-profit barrier.
    :param sl_multiplier: Multiplier for ATR to set the stop-loss barrier.
    :param atr_period: The lookback period for calculating ATR.
    :return: A pandas Series with labels (1 for take-profit, -1 for stop-loss, 0 for timeout).
    """
    print("Starting triple barrier labeling...")

    # 1. Calculate daily volatility (ATR) to set dynamic barriers
    print(f"Calculating ATR with a period of {atr_period}...")
    atr = prices.ta.atr(high=prices['high'], low=prices['low'], close=prices['close'], length=atr_period)

    # Drop NaNs from ATR calculation
    atr = atr.dropna()
    prices = prices.loc[atr.index]

    # 2. Define the barriers for each timestamp
    take_profit_levels = prices['close'] + (atr * tp_multiplier)
    stop_loss_levels = prices['close'] - (atr * sl_multiplier)

    labels = pd.Series(np.nan, index=prices.index)

    # 3. Determine the outcome for each timestamp by looking into the future
    print(f"Iterating through {len(prices)} timestamps to find outcomes...")
    for i in range(len(prices)):
        # The vertical barrier is defined by the holding period
        future_path = prices.iloc[i + 1 : i + 1 + holding_period]

        if future_path.empty:
            continue

        # Check for take-profit hit
        tp_hits = future_path[future_path['high'] >= take_profit_levels.iloc[i]]

        # Check for stop-loss hit
        sl_hits = future_path[future_path['low'] <= stop_loss_levels.iloc[i]]

        # Determine which barrier was hit first
        first_tp_time = tp_hits.index[0] if not tp_hits.empty else pd.NaT
        first_sl_time = sl_hits.index[0] if not sl_hits.empty else pd.NaT

        if pd.notna(first_tp_time) and pd.notna(first_sl_time):
            # If both hit, choose the one that happened first
            if first_tp_time < first_sl_time:
                labels.iloc[i] = 1
            else:
                labels.iloc[i] = -1
        elif pd.notna(first_tp_time):
            labels.iloc[i] = 1
        elif pd.notna(first_sl_time):
            labels.iloc[i] = -1
        else:
            # If neither barrier was hit within the holding period, it's a timeout
            labels.iloc[i] = 0

    print("Labeling complete.")
    return labels.dropna().astype(int)

if __name__ == '__main__':
    print("--- Testing Triple Barrier Labeling ---")
    np.random.seed(42)
    # Create a sample price series with some trends
    price_moves = (np.random.randn(200) * 0.5).cumsum()
    base_price = 100
    close_prices = base_price + price_moves

    data = {
        'high': close_prices + np.random.rand(200) * 0.5,
        'low': close_prices - np.random.rand(200) * 0.5,
        'close': close_prices,
    }
    index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200, freq='H'))
    sample_prices = pd.DataFrame(data, index=index)

    print("Sample price data created.")

    labels = get_triple_barrier_labels(
        sample_prices,
        holding_period=20,
        tp_multiplier=1.5,
        sl_multiplier=1.5,
        atr_period=14
    )

    print("\nGenerated Labels (first 20):")
    print(labels.head(20))
    print("\nLabel Counts:")
    print(labels.value_counts())
    # A healthy distribution is expected, not just timeouts.
    assert 0 in labels.value_counts()
    assert 1 in labels.value_counts()
    assert -1 in labels.value_counts()
    print("\nTest passed: All label types were generated.")
