"""Tests for the assembled recommendation pipeline (src/pipeline.py), using
tiny synthetic fixtures. No real Attribution data or trained model required
for most tests -- a fake scoring model is injected where only orchestration
correctness is being checked.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.attribution import SECONDS_PER_DAY
from src.features.attribution_features import (
    IMPRESSION_CONTEXT_ONLY_COLUMNS,
    PRE_SERVE_FEATURE_COLUMNS,
    add_point_in_time_pair_features,
    add_point_in_time_user_features,
)
from src.models.ctr_model import CTRModel
from src.pipeline import Recommendation, RecommendationPipeline
from src.retrieval.cohort import COLD, WARM


class _FakeModel:
    """A stand-in scoring model exposing only what RecommendationPipeline
    needs (`predict`), so orchestration tests don't depend on LightGBM
    behavior -- and can assert exactly which columns reached scoring."""

    def __init__(self, score_fn=None):
        # Constant score by default -- combined with recommend()'s stable
        # sort, this preserves retrieval's own candidate order, which is
        # what routing tests (which only care about *which* candidates were
        # retrieved) rely on. Tests of ranking order itself use a real
        # position-dependent score_fn explicitly.
        self.score_fn = score_fn or (lambda df: np.zeros(len(df), dtype=float))
        self.seen_columns = None

    def predict(self, X):
        self.seen_columns = list(X.columns)
        return self.score_fn(X)


def _row(day, uid, campaign, click):
    return {"timestamp": day * SECONDS_PER_DAY + 1, "uid": uid, "campaign": campaign, "click": click}


def _fit_pipeline(full_df, k=10, model=None, train_end_day=20, val_end_day=25):
    pipeline = RecommendationPipeline(k=k)
    pipeline.fit(full_df, train_end_day=train_end_day, val_end_day=val_end_day,
                 pre_serve_model=model or _FakeModel())
    return pipeline


class TestCohortRouting:
    def test_cold_user_routed_to_popularity(self):
        full_df = pd.DataFrame(
            [_row(0, 1, 100, 0), _row(0, 2, 200, 1), _row(0, 2, 300, 1)]  # user 1: cold, user 2: warm
        )
        pipeline = _fit_pipeline(full_df)

        assert pipeline.cohort_for(1) == COLD
        recs = pipeline.recommend(1)
        assert [r.campaign for r in recs] == pipeline.pop_gen_.generate(None, None, k=10)

    def test_warm_user_routed_to_collaborative_filtering(self):
        full_df = pd.DataFrame(
            [_row(0, 2, 200, 1), _row(0, 2, 300, 1), _row(0, 5, 200, 1), _row(0, 5, 300, 1)]
        )
        pipeline = _fit_pipeline(full_df)

        assert pipeline.cohort_for(2) == WARM
        recs = pipeline.recommend(2)
        assert [r.campaign for r in recs] == pipeline.cf_gen_.generate(2, None, k=10, exclude_seen=False)

    def test_unknown_user_defaults_to_cold_popularity(self):
        """A user never seen in training falls back to the cold/popularity
        path -- the coldest possible case, matching the convention already
        established in evaluate_retrieval_by_cohort."""
        full_df = pd.DataFrame([_row(0, 1, 100, 1), _row(0, 1, 200, 1)])
        pipeline = _fit_pipeline(full_df)

        assert pipeline.cohort_for(999) == COLD
        recs = pipeline.recommend(999)
        assert [r.campaign for r in recs] == pipeline.pop_gen_.generate(None, None, k=10)

    def test_warm_user_falls_back_to_popularity_if_cf_has_no_candidates(self):
        full_df = pd.DataFrame(
            [_row(0, 2, 200, 1), _row(0, 2, 300, 1), _row(0, 5, 200, 1), _row(0, 5, 300, 1)]
        )
        pipeline = _fit_pipeline(full_df)
        # sabotage CF so it returns nothing, forcing the documented fallback
        pipeline.cf_gen_.user_affinity_ = {}

        recs = pipeline.recommend(2)
        assert [r.campaign for r in recs] == pipeline.pop_gen_.generate(None, None, k=10)


class TestPreServeOnlyScoring:
    def test_no_post_outcome_or_cat_columns_reach_the_model(self):
        full_df = pd.DataFrame([_row(0, 1, 100, 1), _row(0, 1, 200, 0), _row(0, 2, 100, 1)])
        model = _FakeModel()
        pipeline = _fit_pipeline(full_df, model=model)

        pipeline.recommend(1)

        assert set(model.seen_columns) == set(PRE_SERVE_FEATURE_COLUMNS)
        assert not (set(model.seen_columns) & set(IMPRESSION_CONTEXT_ONLY_COLUMNS))
        for excluded in ["cost", "cpo", "conversion", "click_pos", "click_nb", "conversion_timestamp"]:
            assert excluded not in model.seen_columns

    def test_scoring_uses_only_pre_serve_feature_columns_in_order(self):
        full_df = pd.DataFrame([_row(0, 1, 100, 1), _row(0, 2, 100, 0)])
        model = _FakeModel()
        pipeline = _fit_pipeline(full_df, model=model)

        pipeline.recommend(1)
        assert model.seen_columns == PRE_SERVE_FEATURE_COLUMNS


class TestRankingOrder:
    def test_recommendations_sorted_by_descending_predicted_ctr(self):
        full_df = pd.DataFrame([_row(0, 1, 100, 1), _row(0, 2, 200, 1), _row(0, 3, 300, 1)])

        def score_fn(df):
            # deliberately reverse order vs. campaign id to prove sorting, not
            # coincidental ordering, drives the result
            return -df["campaign"].astype(float).to_numpy()

        pipeline = _fit_pipeline(full_df, model=_FakeModel(score_fn))
        recs = pipeline.recommend(1)

        scores = [r.predicted_ctr for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_predicted_ctr_values_carried_through_correctly(self):
        full_df = pd.DataFrame([_row(0, 1, 100, 1), _row(0, 2, 200, 1)])
        pipeline = _fit_pipeline(full_df, model=_FakeModel(lambda df: np.full(len(df), 0.42)))

        recs = pipeline.recommend(1)
        assert all(r.predicted_ctr == pytest.approx(0.42) for r in recs)


class TestSnapshotEquivalence:
    def test_user_snapshot_matches_per_row_point_in_time_function(self):
        """The fast, aggregated snapshot must agree exactly with the slower,
        already-tested per-row cumulative function, evaluated at a user's
        last row within the same history window."""
        full_df = pd.DataFrame(
            [
                _row(0, 1, 100, 1),
                _row(1, 1, 200, 0),
                _row(2, 1, 100, 0),
                _row(0, 2, 300, 1),
            ]
        )
        pipeline = _fit_pipeline(full_df, train_end_day=20, val_end_day=25)

        # Reference: apply the existing, already-tested per-row function
        # directly to the same history window and take each user's LAST row.
        history = full_df[full_df["timestamp"] < 25 * SECONDS_PER_DAY]
        reference = add_point_in_time_user_features(history)
        last_rows = reference.sort_values("timestamp").groupby("uid").tail(1).set_index("uid")

        for uid in [1, 2]:
            snap = pipeline.user_snapshot_.loc[uid]
            ref = last_rows.loc[uid]
            # reference gives "before this last row"; snapshot gives "before
            # the cutoff", which equals "before this row" PLUS this row itself
            assert snap["user_impressions_before"] == ref["user_impressions_before"] + 1
            assert snap["user_clicks_before"] == ref["user_clicks_before"] + ref["click"]

    def test_pair_snapshot_matches_per_row_point_in_time_function(self):
        full_df = pd.DataFrame(
            [_row(0, 1, 100, 1), _row(1, 1, 100, 0), _row(2, 1, 200, 1)]
        )
        pipeline = _fit_pipeline(full_df, train_end_day=20, val_end_day=25)

        history = full_df[full_df["timestamp"] < 25 * SECONDS_PER_DAY]
        reference = add_point_in_time_pair_features(history)
        last_rows = (
            reference.sort_values("timestamp").groupby(["uid", "campaign"]).tail(1).set_index(["uid", "campaign"])
        )

        for uid, campaign in [(1, 100), (1, 200)]:
            snap = pipeline.pair_snapshot_.loc[(uid, campaign)]
            ref = last_rows.loc[(uid, campaign)]
            assert snap["pair_impressions_before"] == ref["pair_impressions_before"] + 1
            assert snap["pair_clicks_before"] == ref["pair_clicks_before"] + ref["click"]


class TestTemporalIntegrity:
    def test_snapshot_excludes_rows_at_or_after_cutoff(self):
        """A row exactly on or after val_end_day must not count toward the
        as-of-cutoff snapshot -- the same strict point-in-time discipline
        used everywhere else in this project."""
        full_df = pd.DataFrame(
            [_row(0, 1, 100, 1), _row(25, 1, 200, 1)]  # second row is AT the cutoff (day 25)
        )
        pipeline = _fit_pipeline(full_df, train_end_day=20, val_end_day=25)

        assert pipeline.user_snapshot_.loc[1, "user_impressions_before"] == 1  # only the day-0 row
        assert (1, 200) not in pipeline.pair_snapshot_.index  # the cutoff-day row must not appear

    def test_no_future_leakage_into_recommendation_features(self):
        full_df = pd.DataFrame(
            [_row(0, 1, 100, 0), _row(26, 1, 200, 1)]  # test-period click, must not leak backward
        )
        model = _FakeModel()
        pipeline = _fit_pipeline(full_df, model=model, train_end_day=20, val_end_day=25)

        pipeline.recommend(1)
        # user 1's snapshot must reflect ONLY the day-0 row
        assert pipeline.user_snapshot_.loc[1, "user_clicks_before"] == 0


class TestBatchMatchesSingle:
    def test_batch_matches_single_recommend(self):
        full_df = pd.DataFrame(
            [_row(0, 1, 100, 1), _row(0, 2, 200, 1), _row(0, 2, 300, 0), _row(0, 3, 100, 0)]
        )
        model = _FakeModel(lambda df: df["campaign"].astype(float).to_numpy() % 7)
        pipeline = _fit_pipeline(full_df, model=model)

        uids = [1, 2, 3, 999]
        batch_results = pipeline.recommend_batch(uids)
        for uid in uids:
            single = pipeline.recommend(uid)
            batch = batch_results[uid]
            assert [r.campaign for r in single] == [r.campaign for r in batch]
            assert [r.predicted_ctr for r in single] == [r.predicted_ctr for r in batch]
            assert [r.cohort for r in single] == [r.cohort for r in batch]

    def test_batch_handles_empty_uid_list(self):
        full_df = pd.DataFrame([_row(0, 1, 100, 1)])
        pipeline = _fit_pipeline(full_df)

        assert pipeline.recommend_batch([]) == {}


class TestUseBeforeFit:
    def test_recommend_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            RecommendationPipeline().recommend(1)

    def test_cohort_for_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            RecommendationPipeline().cohort_for(1)

    def test_recommend_batch_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            RecommendationPipeline().recommend_batch([1])


class TestRealModelEndToEnd:
    def test_real_ctr_model_integration_smoke_test(self):
        """One end-to-end check with a real, tiny, actually-trained
        CTRModel (not the fake) -- confirms the feature frame this module
        builds is genuinely compatible with CTRModel.predict(), not just
        with a stub."""
        rng = np.random.default_rng(0)
        n = 200
        rows = []
        for i in range(n):
            row = _row(i % 15, i % 20, 100 + (i % 5), int(rng.random() < 0.3))
            # build_ctr_features expects cat1..cat9 to exist (real dataset schema),
            # even though this pipeline never uses them for scoring -- see
            # PRE_SERVE_FEATURE_COLUMNS assertions elsewhere in this file.
            for c in range(1, 10):
                row[f"cat{c}"] = i % 3
            row["time_since_last_click"] = -1
            rows.append(row)
        full_df = pd.DataFrame(rows)

        train_df = full_df[full_df["timestamp"] < 20 * SECONDS_PER_DAY]
        params = {"objective": "binary", "verbose": -1, "seed": 42, "num_leaves": 7}
        model = CTRModel(params, categorical_features=["campaign"], num_boost_round=10,
                          early_stopping_rounds=None)

        # Build a minimal, real pre-serve-shaped training frame directly
        # (independent of the pipeline under test) to fit a real model.
        from src.features.attribution_features import build_ctr_features

        features = build_ctr_features(full_df, train_df)
        model.fit(features.loc[features["timestamp"] < 20 * SECONDS_PER_DAY, PRE_SERVE_FEATURE_COLUMNS],
                  features.loc[features["timestamp"] < 20 * SECONDS_PER_DAY, "click"])

        pipeline = _fit_pipeline(full_df, model=model, train_end_day=20, val_end_day=25)
        recs = pipeline.recommend(uid=1)

        assert isinstance(recs, list)
        for r in recs:
            assert isinstance(r, Recommendation)
            assert 0.0 <= r.predicted_ctr <= 1.0
