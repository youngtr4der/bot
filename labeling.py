import pandas as pd
import numpy as np
import pandas_ta as ta

def get_triple_barrier_labels(
    prices: pd.DataFrame,
    holding_period: int,
    tp_multiplier: float,
    sl_multiplier: float,
    atr_period: int = 14
) -> pd.DataFrame:
    """
    Implements the Triple Barrier Method and returns labels and event timestamps.

    This method sets three barriers for each entry point and determines which is hit first.
    The timestamp of the event is crucial for purging in cross-validation.

    :param prices: DataFrame with 'high', 'low', 'close' columns and a DatetimeIndex.
    :param holding_period: Maximum number of bars to hold a position (vertical barrier).
    :param tp_multiplier: Multiplier for ATR to set the take-profit barrier.
    :param sl_multiplier: Multiplier for ATR to set the stop-loss barrier.
    :param atr_period: The lookback period for calculating ATR.
    :return: A pandas DataFrame with 'label' and 'touch_time' columns.
    """
    print("Starting triple barrier labeling...")

    print(f"Calculating ATR with a period of {atr_period}...")
    atr = prices.ta.atr(high=prices['high'], low=prices['low'], close=prices['close'], length=atr_period)

    atr = atr.dropna()
    prices = prices.loc[atr.index]

    take_profit_levels = prices['close'] + (atr * tp_multiplier)
    stop_loss_levels = prices['close'] - (atr * sl_multiplier)

    labels = pd.Series(np.nan, index=prices.index)
    touch_times = pd.Series(pd.NaT, index=prices.index)

    print(f"Iterating through {len(prices)} timestamps to find outcomes...")
    for i in range(len(prices)):
        future_path = prices.iloc[i + 1 : i + 1 + holding_period]

        if future_path.empty:
            continue

        tp_hits = future_path[future_path['high'] >= take_profit_levels.iloc[i]]
        sl_hits = future_path[future_path['low'] <= stop_loss_levels.iloc[i]]

        first_tp_time = tp_hits.index[0] if not tp_hits.empty else pd.NaT
        first_sl_time = sl_hits.index[0] if not sl_hits.empty else pd.NaT

        if pd.notna(first_tp_time) and pd.notna(first_sl_time):
            if first_tp_time < first_sl_time:
                labels.iloc[i] = 1
                touch_times.iloc[i] = first_tp_time
            else:
                labels.iloc[i] = -1
                touch_times.iloc[i] = first_sl_time
        elif pd.notna(first_tp_time):
            labels.iloc[i] = 1
            touch_times.iloc[i] = first_tp_time
        elif pd.notna(first_sl_time):
            labels.iloc[i] = -1
            touch_times.iloc[i] = first_sl_time
        else:
            labels.iloc[i] = 0
            touch_times.iloc[i] = future_path.index[-1]

    print("Labeling complete.")
    result_df = pd.DataFrame({'label': labels, 'touch_time': touch_times})
    return result_df.dropna(subset=['label'])

if __name__ == '__main__':
    print("--- Testing Triple Barrier Labeling ---")
    np.random.seed(42)
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

    outcomes = get_triple_barrier_labels(
        sample_prices,
        holding_period=20,
        tp_multiplier=1.5,
        sl_multiplier=1.5,
        atr_period=14
    )

    print("\nGenerated Outcomes (first 10):")
    print(outcomes.head(10))
    print("\nLabel Counts:")
    print(outcomes['label'].value_counts())

    assert 'label' in outcomes.columns
    assert 'touch_time' in outcomes.columns
    assert not outcomes.isnull().values.any()
    print("\nTest passed: DataFrame with 'label' and 'touch_time' columns generated successfully.")
