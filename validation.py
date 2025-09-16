import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def get_purged_walk_forward_splits(
    X: pd.DataFrame,
    touch_times: pd.Series,
    n_splits: int = 5,
    embargo_size: int = 5
):
    """
    Generates purged and embargoed walk-forward cross-validation splits.

    This generator enhances sklearn's TimeSeriesSplit by:
    1. Purging: Removing training samples whose labels are determined by information
       that overlaps with the validation set.
    2. Embargoing: Removing a small number of samples from the start of the
       validation set to prevent serial correlation leakage.

    :param X: The feature DataFrame with a DatetimeIndex.
    :param touch_times: A Series containing the timestamp when each label was determined.
                        Must be aligned with X's index.
    :param n_splits: The number of splits for TimeSeriesSplit.
    :param embargo_size: The number of samples to remove from the start of the validation set.
    :yields: A tuple of (train_indices, validation_indices).
    """
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X must have a DatetimeIndex.")

    print(f"Generating {n_splits} purged and embargoed splits...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for i, (train_idx_loc, val_idx_loc) in enumerate(tscv.split(X)):
        print(f"\n--- Split {i+1}/{n_splits} ---")

        # Get the actual indices from the integer locations
        train_indices = X.index[train_idx_loc]
        val_indices = X.index[val_idx_loc]

        # 1. Determine the purging boundary
        val_start_time = val_indices[0]

        # Get touch times for the training set
        train_touch_times = touch_times.loc[train_indices]

        # Purge training samples that were labeled using data from the validation period
        purged_train_indices = train_touch_times[train_touch_times < val_start_time].index

        # 2. Apply embargo to the validation set
        embargoed_val_indices = val_indices[embargo_size:]

        print(f"Original train size: {len(train_indices)}, Purged train size: {len(purged_train_indices)}")
        print(f"Original val size: {len(val_indices)}, Embargoed val size: {len(embargoed_val_indices)}")

        if len(purged_train_indices) == 0 or len(embargoed_val_indices) == 0:
            print("Warning: A split resulted in an empty train or validation set. Skipping.")
            continue

        yield purged_train_indices, embargoed_val_indices


if __name__ == '__main__':
    print("--- Testing Purged Walk-Forward CV ---")

    # 1. Create sample data
    idx = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    X_sample = pd.DataFrame(np.random.rand(100, 2), index=idx, columns=['feat1', 'feat2'])

    # 2. Create sample touch_times that will cause purging
    # For each sample, the label is decided sometime in the next 10 days
    touch_times_sample = pd.Series(X_sample.index + pd.to_timedelta(np.random.randint(1, 11, 100), 'D'), index=idx)

    print(f"Sample data created with {len(X_sample)} points.")

    # 3. Use the generator
    splits = get_purged_walk_forward_splits(
        X_sample,
        touch_times_sample,
        n_splits=5,
        embargo_size=3
    )

    # 4. Print the splits to verify the logic
    for train_indices, val_indices in splits:
        print(f"Train period: {train_indices.min()} to {train_indices.max()}")
        print(f"Val period:   {val_indices.min()} to {val_indices.max()}")

        # Verification check: The last touch time in the training set must be before the first day of the validation set
        last_train_touch_time = touch_times_sample.loc[train_indices].max()
        first_val_time = val_indices.min()

        print(f"Last train touch time: {last_train_touch_time}, First val time: {first_val_time}")
        assert last_train_touch_time < first_val_time
        print("Purging check passed.")

    print("\n--- Test complete. ---")
