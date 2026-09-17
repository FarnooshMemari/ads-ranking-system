"""Tests for chronological splitting of the Attribution dataset.

Uses a tiny synthetic fixture — no real dataset download required.
"""

import pandas as pd

from src.data.attribution import SECONDS_PER_DAY, split_by_day


def _make_df(day_uid_pairs):
    """Build a minimal synthetic frame: one row per (day, uid), rest filled in."""
    rows = []
    for day, uid, campaign, click in day_uid_pairs:
        rows.append(
            {
                "timestamp": day * SECONDS_PER_DAY + 1,
                "uid": uid,
                "campaign": campaign,
                "click": click,
            }
        )
    return pd.DataFrame(rows)


def test_split_boundaries_are_correct():
    df = _make_df(
        [
            (0, 1, 100, 1),
            (19, 1, 100, 1),
            (20, 1, 100, 1),
            (24, 1, 100, 1),
            (25, 1, 100, 1),
            (30, 1, 100, 1),
        ]
    )
    train, val, test = split_by_day(df, train_end_day=20, val_end_day=25)

    assert sorted(train["timestamp"] // SECONDS_PER_DAY) == [0, 19]
    assert sorted(val["timestamp"] // SECONDS_PER_DAY) == [20, 24]
    assert sorted(test["timestamp"] // SECONDS_PER_DAY) == [25, 30]


def test_splits_are_disjoint_and_cover_all_rows():
    df = _make_df([(d, 1, 100, 0) for d in range(31)])
    train, val, test = split_by_day(df, train_end_day=20, val_end_day=25)

    assert len(train) + len(val) + len(test) == len(df)
    train_idx, val_idx, test_idx = set(train.index), set(val.index), set(test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_no_future_rows_leak_into_train_split():
    """The core temporal-leakage guarantee: a click that only happens in the
    test period must never appear in the train split, even for the same user."""
    df = _make_df(
        [
            (0, 42, 100, 0),   # user 42's only train-period impression: no click
            (26, 42, 200, 1),  # user 42 clicks campaign 200, but only in the test period
        ]
    )
    train, _, test = split_by_day(df, train_end_day=20, val_end_day=25)

    assert list(train["campaign"]) == [100]
    assert list(train["click"]) == [0]
    assert 200 not in set(train["campaign"])
    assert list(test["campaign"]) == [200]
