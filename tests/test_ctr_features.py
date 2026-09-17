"""Tests for CTR feature engineering, using tiny synthetic fixtures.

No real Attribution data is downloaded or required.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.attribution import split_by_day
from src.features.attribution_features import (
    EXCLUDED_POST_OUTCOME_COLUMNS,
    FEATURE_COLUMNS,
    add_point_in_time_pair_features,
    add_point_in_time_user_features,
    assert_no_post_outcome_leakage,
    attach_campaign_aggregates,
    build_ctr_features,
    compute_campaign_aggregates,
)


def _row(day, uid, campaign, click, **extra):
    base = {"timestamp": day * 86400 + 1, "uid": uid, "campaign": campaign, "click": click}
    base.update(extra)
    return base


def _with_cats(row, seed=1):
    for i in range(1, 10):
        row[f"cat{i}"] = seed
    row.setdefault("time_since_last_click", -1)
    return row


class TestNoPostOutcomeFeatures:
    def test_excluded_columns_absent_from_feature_list(self):
        assert set(FEATURE_COLUMNS).isdisjoint(EXCLUDED_POST_OUTCOME_COLUMNS)

    def test_assert_no_post_outcome_leakage_passes_for_clean_list(self):
        assert_no_post_outcome_leakage(FEATURE_COLUMNS)  # should not raise

    def test_assert_no_post_outcome_leakage_raises_for_dirty_list(self):
        with pytest.raises(ValueError):
            assert_no_post_outcome_leakage(FEATURE_COLUMNS + ["cost"])


class TestPointInTimeUserFeatures:
    def test_first_impression_has_zero_history(self):
        df = pd.DataFrame([_row(0, 1, 100, 0)])
        out = add_point_in_time_user_features(df)

        assert out["user_impressions_before"].iloc[0] == 0
        assert out["user_clicks_before"].iloc[0] == 0
        assert np.isnan(out["user_click_rate_before"].iloc[0])
        assert out["user_distinct_campaigns_before"].iloc[0] == 0

    def test_second_impression_reflects_only_prior_row(self):
        df = pd.DataFrame([_row(0, 1, 100, 1), _row(1, 1, 200, 0)])
        out = add_point_in_time_user_features(df)
        second = out[out["campaign"] == 200].iloc[0]

        assert second["user_impressions_before"] == 1
        assert second["user_clicks_before"] == 1
        assert second["user_click_rate_before"] == 1.0
        assert second["user_distinct_campaigns_before"] == 1

    def test_no_future_leakage_within_same_user(self):
        """A row must never see clicks/impressions that happen LATER for the
        same user, even later on the same day."""
        df = pd.DataFrame(
            [_row(0, 1, 100, 0, timestamp=10), _row(0, 1, 200, 1, timestamp=20)]
        )
        out = add_point_in_time_user_features(df)
        first = out[out["campaign"] == 100].iloc[0]

        assert first["user_impressions_before"] == 0
        assert first["user_clicks_before"] == 0  # must NOT count the later click on 200

    def test_original_row_order_preserved(self):
        df = pd.DataFrame([_row(5, 1, 100, 0), _row(0, 1, 200, 1), _row(2, 1, 300, 0)])
        out = add_point_in_time_user_features(df)

        assert list(out["campaign"]) == [100, 200, 300]  # unchanged order


class TestPointInTimePairFeatures:
    def test_pair_history_isolated_per_campaign(self):
        df = pd.DataFrame(
            [
                _row(0, 1, 100, 1),
                _row(1, 1, 200, 0),  # different campaign -- no shared pair history
                _row(2, 1, 100, 0),  # same pair as row 0
            ]
        )
        out = add_point_in_time_pair_features(df)
        third = out.iloc[2]

        assert third["campaign"] == 100
        assert third["pair_impressions_before"] == 1
        assert third["pair_clicks_before"] == 1

        second = out.iloc[1]
        assert second["pair_impressions_before"] == 0  # first time (uid=1, campaign=200)


class TestCampaignAggregates:
    def test_computed_only_from_train_period(self):
        full_df = pd.DataFrame(
            [
                _row(0, 1, 100, 1),
                _row(0, 2, 100, 0),
                _row(26, 3, 100, 1),  # test-period click -- must NOT affect the aggregate
                _row(26, 4, 100, 1),
            ]
        )
        train_df, _, test_df = split_by_day(full_df, train_end_day=20, val_end_day=25)
        agg = compute_campaign_aggregates(train_df)

        assert agg.loc[100, "campaign_impressions_train"] == 2
        assert agg.loc[100, "campaign_clicks_train"] == 1
        assert agg.loc[100, "campaign_ctr_train"] == 0.5

    def test_unseen_campaign_falls_back_to_global_train_ctr(self):
        train_df = pd.DataFrame([_row(0, 1, 100, 1), _row(0, 2, 100, 0)])  # global CTR = 0.5
        test_df = pd.DataFrame([_row(26, 3, 999, 1)])  # campaign 999 never in train
        agg = compute_campaign_aggregates(train_df)

        result = attach_campaign_aggregates(test_df, agg, global_train_ctr=0.5)

        assert result["campaign_impressions_train"].iloc[0] == 0
        assert result["campaign_clicks_train"].iloc[0] == 0
        assert result["campaign_ctr_train"].iloc[0] == 0.5


class TestMissingValueHandling:
    def test_count_features_never_missing(self):
        df = pd.DataFrame([_row(0, 1, 100, 0), _row(1, 1, 200, 1)])
        out = add_point_in_time_user_features(df)
        out = add_point_in_time_pair_features(out)

        never_missing = [
            "user_impressions_before",
            "user_clicks_before",
            "user_distinct_campaigns_before",
            "pair_impressions_before",
            "pair_clicks_before",
        ]
        for col in never_missing:
            assert out[col].isna().sum() == 0

    def test_click_rate_is_nan_only_with_no_prior_history(self):
        df = pd.DataFrame([_row(0, 1, 100, 0), _row(1, 1, 200, 1)])
        out = add_point_in_time_user_features(df)

        assert np.isnan(out.iloc[0]["user_click_rate_before"])  # first impression -- no history
        assert not np.isnan(out.iloc[1]["user_click_rate_before"])  # second -- has history


class TestBuildCtrFeaturesEndToEnd:
    def test_temporal_split_integrity_history_spans_splits_correctly(self):
        """The central correctness property: a TEST-period row's user-history
        features must reflect that same user's real TRAIN-period activity,
        not reset to zero at the split boundary."""
        full_df = pd.DataFrame(
            [
                _with_cats(_row(0, 1, 100, 1)),   # train: user 1 clicks campaign 100
                _with_cats(_row(26, 1, 200, 0)),  # test: user 1's second-ever impression
            ]
        )
        train_df, _, test_df = split_by_day(full_df, train_end_day=20, val_end_day=25)
        features = build_ctr_features(full_df, train_df)
        _, _, test_features = split_by_day(features, train_end_day=20, val_end_day=25)

        test_row = test_features.iloc[0]
        assert test_row["user_impressions_before"] == 1  # counts the train-period impression
        assert test_row["user_clicks_before"] == 1

    def test_train_val_test_schema_consistency(self):
        full_df = pd.DataFrame(
            [_with_cats(_row(d, (d % 3) + 1, 100 + (d % 2), d % 2)) for d in range(31)]
        )
        train_df, val_df, test_df = split_by_day(full_df, train_end_day=20, val_end_day=25)
        features = build_ctr_features(full_df, train_df)
        train_f, val_f, test_f = split_by_day(features, train_end_day=20, val_end_day=25)

        assert list(train_f.columns) == list(val_f.columns) == list(test_f.columns)
        for col in FEATURE_COLUMNS:
            assert train_f[col].dtype == val_f[col].dtype == test_f[col].dtype

    def test_categorical_columns_share_consistent_categories_across_splits(self):
        full_df = pd.DataFrame(
            [_with_cats(_row(0, 1, 100, 1), seed=5), _with_cats(_row(26, 2, 200, 0), seed=9)]
        )
        train_df, _, test_df = split_by_day(full_df, train_end_day=20, val_end_day=25)
        features = build_ctr_features(full_df, train_df)
        train_f, _, test_f = split_by_day(features, train_end_day=20, val_end_day=25)

        # both campaign 100 and 200 must be known categories in BOTH slices'
        # underlying dtype, even though each slice only contains one of them.
        assert set(train_f["campaign"].cat.categories) == set(test_f["campaign"].cat.categories)

    def test_all_feature_columns_present_after_build(self):
        full_df = pd.DataFrame([_with_cats(_row(0, 1, 100, 1)), _with_cats(_row(1, 2, 100, 0))])
        train_df, _, _ = split_by_day(full_df, train_end_day=20, val_end_day=25)
        features = build_ctr_features(full_df, train_df)

        for col in FEATURE_COLUMNS:
            assert col in features.columns


class TestDeterministicConfiguration:
    def test_repeated_build_is_identical(self):
        full_df = pd.DataFrame(
            [_with_cats(_row(d, (d % 4) + 1, 100 + (d % 3), d % 2)) for d in range(15)]
        )
        train_df, _, _ = split_by_day(full_df, train_end_day=20, val_end_day=25)

        first = build_ctr_features(full_df, train_df)
        second = build_ctr_features(full_df, train_df)

        pd.testing.assert_frame_equal(
            first[FEATURE_COLUMNS].reset_index(drop=True),
            second[FEATURE_COLUMNS].reset_index(drop=True),
        )
