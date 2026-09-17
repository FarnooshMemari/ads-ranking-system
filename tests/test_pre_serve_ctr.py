"""Tests for the pre-serve (cat1-9-excluded) CTR feature set and model,
using tiny synthetic fixtures. No real Attribution data required.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.attribution import split_by_day
from src.features.attribution_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    EXCLUDED_POST_OUTCOME_COLUMNS,
    FEATURE_COLUMNS,
    IMPRESSION_CONTEXT_ONLY_COLUMNS,
    PRE_SERVE_CATEGORICAL_FEATURE_COLUMNS,
    PRE_SERVE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    assert_no_post_outcome_leakage,
    build_ctr_features,
)
from src.models.ctr_model import CTRModel


def _row(day, uid, campaign, click, **extra):
    base = {"timestamp": day * 86400 + 1, "uid": uid, "campaign": campaign, "click": click}
    base.update(extra)
    return base


def _with_cats(row, seed=1):
    for i in range(1, 10):
        row[f"cat{i}"] = seed
    row.setdefault("time_since_last_click", -1)
    return row


class TestCat1To9AbsentFromPreServeFeatures:
    def test_no_cat_columns_in_pre_serve_feature_list(self):
        cat_columns = {f"cat{i}" for i in range(1, 10)}
        assert cat_columns.isdisjoint(PRE_SERVE_FEATURE_COLUMNS)

    def test_no_cat_columns_in_pre_serve_categorical_list(self):
        cat_columns = {f"cat{i}" for i in range(1, 10)}
        assert cat_columns.isdisjoint(PRE_SERVE_CATEGORICAL_FEATURE_COLUMNS)

    def test_impression_context_only_columns_are_exactly_cat1_to_9(self):
        assert set(IMPRESSION_CONTEXT_ONLY_COLUMNS) == {f"cat{i}" for i in range(1, 10)}

    def test_time_since_last_click_is_retained(self):
        """Unlike cat1-9, time_since_last_click is derived from the user's own
        past history and is knowable before serving -- it must stay."""
        assert "time_since_last_click" in PRE_SERVE_FEATURE_COLUMNS

    def test_campaign_identity_is_retained(self):
        assert "campaign" in PRE_SERVE_FEATURE_COLUMNS
        assert "campaign" in PRE_SERVE_CATEGORICAL_FEATURE_COLUMNS


class TestNoPostOutcomeLeakageInPreServeFeatures:
    def test_excluded_columns_absent(self):
        assert set(PRE_SERVE_FEATURE_COLUMNS).isdisjoint(EXCLUDED_POST_OUTCOME_COLUMNS)

    def test_assert_no_post_outcome_leakage_passes(self):
        assert_no_post_outcome_leakage(PRE_SERVE_FEATURE_COLUMNS)  # must not raise

    def test_assert_no_post_outcome_leakage_raises_if_contaminated(self):
        with pytest.raises(ValueError):
            assert_no_post_outcome_leakage(PRE_SERVE_FEATURE_COLUMNS + ["cpo"])


class TestConsistentPreprocessing:
    def test_pre_serve_is_a_strict_subset_of_full_feature_columns(self):
        """Derived from FEATURE_COLUMNS, not hardcoded separately -- the two
        feature sets cannot silently drift apart."""
        assert set(PRE_SERVE_FEATURE_COLUMNS).issubset(set(FEATURE_COLUMNS))
        assert len(PRE_SERVE_FEATURE_COLUMNS) == len(FEATURE_COLUMNS) - 9

    def test_pre_serve_categorical_is_a_strict_subset_of_full_categorical_columns(self):
        assert set(PRE_SERVE_CATEGORICAL_FEATURE_COLUMNS).issubset(set(CATEGORICAL_FEATURE_COLUMNS))

    def test_build_ctr_features_output_still_contains_pre_serve_columns(self):
        """The same build_ctr_features pipeline is reused unchanged; the
        pre-serve model just selects a column subset from its output."""
        full_df = pd.DataFrame([_with_cats(_row(0, 1, 100, 1)), _with_cats(_row(1, 2, 100, 0))])
        train_df, _, _ = split_by_day(full_df, train_end_day=20, val_end_day=25)
        features = build_ctr_features(full_df, train_df)

        for col in PRE_SERVE_FEATURE_COLUMNS:
            assert col in features.columns


class TestTemporalIntegrity:
    def test_pre_serve_features_respect_point_in_time_discipline(self):
        """Same underlying guarantee as the full-feature model: a test-period
        row's pre-serve history features must reflect real train-period
        activity for that user, not reset at the split boundary, and must
        never see later-than-self information."""
        full_df = pd.DataFrame(
            [
                _with_cats(_row(0, 1, 100, 1)),    # train: user 1 clicks campaign 100
                _with_cats(_row(26, 1, 200, 0)),   # test: user 1's second-ever impression
            ]
        )
        train_df, _, test_df = split_by_day(full_df, train_end_day=20, val_end_day=25)
        features = build_ctr_features(full_df, train_df)
        _, _, test_features = split_by_day(features, train_end_day=20, val_end_day=25)

        test_row = test_features.iloc[0]
        # these are pre-serve columns -- verify they carry the real prior history
        assert test_row["user_impressions_before"] == 1
        assert test_row["user_clicks_before"] == 1
        assert test_row["pair_impressions_before"] == 0  # different campaign than before

    def test_campaign_aggregates_in_pre_serve_set_come_only_from_train(self):
        full_df = pd.DataFrame(
            [
                _with_cats(_row(0, 1, 100, 1)),
                _with_cats(_row(0, 2, 100, 0)),
                _with_cats(_row(26, 3, 100, 1)),  # test-period click on campaign 100
            ]
        )
        train_df, _, _ = split_by_day(full_df, train_end_day=20, val_end_day=25)
        features = build_ctr_features(full_df, train_df)

        # train-only aggregate: 2 impressions, 1 click -> ctr 0.5, unaffected by
        # the test-period row's click even though it's the same campaign.
        assert features.loc[features["campaign"] == 100, "campaign_ctr_train"].iloc[0] == 0.5


def _synthetic_pre_serve_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    campaign = rng.integers(0, 5, size=n)
    click_prob = np.where(campaign == 0, 0.9, 0.1)
    click = rng.binomial(1, click_prob)
    X = pd.DataFrame(
        {
            "campaign": pd.Series(campaign, dtype="category"),
            "user_clicks_before": rng.integers(0, 5, size=n).astype(float),
            "time_since_last_click": rng.integers(-1, 1000, size=n).astype(float),
        }
    )
    return X, click


class TestDeterministicScoring:
    def test_repeated_fit_predict_identical_with_fixed_seed(self):
        X, y = _synthetic_pre_serve_frame()
        params = {"objective": "binary", "verbose": -1, "seed": 42, "num_leaves": 7, "deterministic": True}

        model_a = CTRModel(params, categorical_features=["campaign"], num_boost_round=25,
                            early_stopping_rounds=None).fit(X, y)
        model_b = CTRModel(params, categorical_features=["campaign"], num_boost_round=25,
                            early_stopping_rounds=None).fit(X, y)

        assert np.allclose(model_a.predict(X), model_b.predict(X))

    def test_predict_does_not_require_cat_columns(self):
        """The whole point of the pre-serve model: it must be usable without
        cat1-9 ever being present in the input at all."""
        X, y = _synthetic_pre_serve_frame()
        assert not any(col.startswith("cat") and col != "campaign" for col in X.columns)

        params = {"objective": "binary", "verbose": -1, "seed": 42, "num_leaves": 7}
        model = CTRModel(params, categorical_features=["campaign"], num_boost_round=20,
                          early_stopping_rounds=None).fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert ((preds >= 0) & (preds <= 1)).all()
