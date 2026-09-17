"""Tests for cold/warm cohort classification, using tiny synthetic fixtures."""

import pandas as pd

from src.data.attribution import split_by_day
from src.retrieval.cohort import COLD, WARM, classify_users, cohort_counts


def _row(day, uid, campaign, click):
    return {"timestamp": day * 86400 + 1, "uid": uid, "campaign": campaign, "click": click}


def test_user_with_no_clicks_is_cold():
    train_df = pd.DataFrame([_row(0, 1, 100, 0), _row(1, 1, 101, 0)])
    cohort = classify_users(train_df)

    assert cohort.loc[1] == COLD


def test_user_with_one_clicked_campaign_is_cold():
    train_df = pd.DataFrame([_row(0, 1, 100, 1), _row(1, 1, 100, 1)])  # same campaign, 2 clicks
    cohort = classify_users(train_df)

    assert cohort.loc[1] == COLD  # only 1 DISTINCT clicked campaign


def test_user_with_two_distinct_clicked_campaigns_is_warm():
    train_df = pd.DataFrame([_row(0, 1, 100, 1), _row(1, 1, 200, 1)])
    cohort = classify_users(train_df)

    assert cohort.loc[1] == WARM


def test_threshold_is_configurable():
    train_df = pd.DataFrame(
        [_row(0, 1, 100, 1), _row(1, 1, 200, 1), _row(2, 1, 300, 1)]
    )
    assert classify_users(train_df, warm_min_distinct_campaigns=2).loc[1] == WARM
    assert classify_users(train_df, warm_min_distinct_campaigns=4).loc[1] == COLD


def test_all_users_present_are_classified_including_zero_click_users():
    train_df = pd.DataFrame(
        [_row(0, 1, 100, 1), _row(0, 2, 100, 0), _row(0, 3, 100, 1), _row(0, 3, 200, 1)]
    )
    cohort = classify_users(train_df)

    assert set(cohort.index) == {1, 2, 3}
    assert cohort.loc[1] == COLD
    assert cohort.loc[2] == COLD
    assert cohort.loc[3] == WARM


def test_cohort_counts_summary():
    train_df = pd.DataFrame(
        [_row(0, 1, 100, 0), _row(0, 2, 100, 1), _row(0, 2, 200, 1)]
    )
    counts = cohort_counts(classify_users(train_df))

    assert counts == {"cold": 1, "warm": 1}


def test_no_temporal_leakage_end_to_end():
    """A user who would be WARM only because of test-period clicks must be
    classified COLD when cohorts are computed from the correctly-split train set."""
    df = pd.DataFrame(
        [
            _row(0, 7, 100, 1),   # train: 1 distinct clicked campaign
            _row(26, 7, 200, 1),  # test-period only: a second distinct clicked campaign
            _row(27, 7, 300, 1),  # test-period only: a third distinct clicked campaign
        ]
    )
    train_df, _, _ = split_by_day(df, train_end_day=20, val_end_day=25)
    cohort = classify_users(train_df)

    # Using the full (leaked) dataset would have made this user WARM (3 distinct
    # clicked campaigns); using the correctly-split train set, they are COLD.
    assert cohort.loc[7] == COLD
    leaked_cohort = classify_users(df)  # deliberately using unfiltered df to show contrast
    assert leaked_cohort.loc[7] == WARM
