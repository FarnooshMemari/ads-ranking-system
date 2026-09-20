"""Assembled retrieval -> pre-serve CTR scoring recommendation pipeline.

Connects, for the first time as a single callable artifact, the components
already built and validated separately: cohort classification
(``src.retrieval.cohort``), retrieval (``src.retrieval.candidate_generator``),
and pre-serve CTR scoring (``src.models.ctr_model.CTRModel`` restricted to
``src.features.attribution_features.PRE_SERVE_FEATURE_COLUMNS``). See
docs/next_phase_design.md for the full design and docs/pipeline_demo.md for
results.

**This is not a live serving system.** ``RecommendationPipeline`` is fit once
from historical data up to a fixed point-in-time cutoff (``as_of_day``, the
existing validation/test boundary — no new cutoff is introduced) and answers
"what would we recommend to this user as of that moment," entirely offline,
against already-logged data. There is no online component, no API, and no
real-time feature computation.

**Only the pre-serve feature set is ever used to score a candidate.**
``cat1``-``cat9`` never reach this module — they describe an impression's
serving context, which does not exist for a candidate that hasn't been
served (docs/ranking_problem_design.md Q5). This is enforced structurally
(only ``PRE_SERVE_FEATURE_COLUMNS`` is ever selected for scoring, and
``IMPRESSION_CONTEXT_ONLY_COLUMNS`` is asserted absent before every score
call), not just documented.

**The user/pair history "snapshot" computed here is not a new feature
definition.** It answers the same question
``src.features.attribution_features.add_point_in_time_user_features`` /
``add_point_in_time_pair_features`` already answer per logged row — "how
many prior impressions/clicks does this user (or user-campaign pair) have
strictly before a given timestamp" — evaluated once at a fixed cutoff for
every user, via a direct aggregation, rather than per-row via a cumulative
scan. The two are mathematically equivalent for a user's *last* row within
the same history window; this equivalence is verified directly in
``tests/test_pipeline.py``. ``time_since_last_click`` is the one field
without an existing point-in-time function to reuse (the raw dataset only
populates it on real logged rows), so it is computed fresh here, using the
same convention (-1 sentinel for "no prior click") verified in
docs/attribution_dataset_verification.md.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.attribution import SECONDS_PER_DAY, split_by_day
from src.features.attribution_features import (
    IMPRESSION_CONTEXT_ONLY_COLUMNS,
    PRE_SERVE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    assert_no_post_outcome_leakage,
    attach_campaign_aggregates,
    compute_campaign_aggregates,
)
from src.models.ctr_model import CTRModel
from src.retrieval.candidate_generator import (
    CollaborativeFilteringCandidateGenerator,
    PopularityCandidateGenerator,
)
from src.retrieval.cohort import COLD, WARM, classify_users

_USER_FEATURE_COLUMNS = [
    "user_impressions_before",
    "user_clicks_before",
    "user_click_rate_before",
    "user_distinct_campaigns_before",
    "time_since_last_click",
]
_NO_HISTORY_USER_DEFAULTS = {
    "user_impressions_before": 0,
    "user_clicks_before": 0,
    "user_click_rate_before": np.nan,
    "user_distinct_campaigns_before": 0,
    "time_since_last_click": -1.0,
}


@dataclass
class Recommendation:
    """One scored, ranked candidate returned by ``RecommendationPipeline.recommend``."""

    campaign: Any
    predicted_ctr: float
    cohort: str


class RecommendationPipeline:
    """Orchestrates cohort classification, retrieval, and pre-serve CTR scoring.

    Contains no new modeling: every substantive computation is a direct call
    into already-tested ``src/retrieval/`` and ``src/features/`` code, or a
    simple aggregation equivalent to what that code already computes (see
    module docstring).
    """

    def __init__(self, k: int = 10, warm_min_distinct_campaigns: int = 2):
        """
        Args:
            k: Number of candidates to retrieve and rank per user.
            warm_min_distinct_campaigns: Cohort threshold, matching
                ``configs/config.yaml``: ``attribution.cohort.warm_min_distinct_clicked_campaigns``.
        """
        self.k = k
        self.warm_min_distinct_campaigns = warm_min_distinct_campaigns
        self.cohorts_: Optional[pd.Series] = None
        self.pop_gen_: Optional[PopularityCandidateGenerator] = None
        self.cf_gen_: Optional[CollaborativeFilteringCandidateGenerator] = None
        self.campaign_agg_: Optional[pd.DataFrame] = None
        self.global_train_ctr_: Optional[float] = None
        self.model_: Optional[CTRModel] = None
        self.user_snapshot_: Optional[pd.DataFrame] = None
        self.pair_snapshot_: Optional[pd.DataFrame] = None
        self.as_of_timestamp_: Optional[int] = None

    def fit(
        self,
        full_df: pd.DataFrame,
        train_end_day: int,
        val_end_day: int,
        pre_serve_model: CTRModel,
    ) -> "RecommendationPipeline":
        """Fit every stage from historical data, exactly as already validated.

        Re-fits the same retrieval components the separate experiment
        scripts already used, on the same data (train-period only). Does
        NOT train a new CTR model — ``pre_serve_model`` must already be
        fit (e.g. loaded from
        ``outputs/models/pre_serve_ctr_model.txt`` via ``CTRModel.load``).

        Args:
            full_df: The complete Attribution dataset (all splits).
            train_end_day: Same split boundary used everywhere else in this
                project (``configs/config.yaml``: ``attribution.split.train_end_day``).
            val_end_day: Same split boundary; also used as this pipeline's
                fixed "as of" cutoff — no new cutoff is introduced.
            pre_serve_model: An already-fit ``CTRModel`` trained on
                ``PRE_SERVE_FEATURE_COLUMNS`` (see ``docs/pre_serve_ctr_model.md``).

        Returns:
            self.
        """
        train_df, _, _ = split_by_day(full_df, train_end_day, val_end_day)

        self.cohorts_ = classify_users(
            train_df, warm_min_distinct_campaigns=self.warm_min_distinct_campaigns
        )
        warm_uids = set(self.cohorts_[self.cohorts_ == WARM].index)

        self.pop_gen_ = PopularityCandidateGenerator().fit(
            train_df, item_col="campaign", positive_col="click"
        )
        warm_train_df = train_df[train_df["uid"].isin(warm_uids)]
        self.cf_gen_ = CollaborativeFilteringCandidateGenerator(similarity_type="jaccard").fit(
            warm_train_df, user_col="uid", item_col="campaign", positive_col="click"
        )

        self.campaign_agg_ = compute_campaign_aggregates(train_df)
        self.global_train_ctr_ = float(train_df[TARGET_COLUMN].mean())

        self.as_of_timestamp_ = val_end_day * SECONDS_PER_DAY
        history_df = full_df[full_df["timestamp"] < self.as_of_timestamp_]
        self.user_snapshot_ = self._build_user_snapshot(history_df)
        self.pair_snapshot_ = self._build_pair_snapshot(history_df)

        self.model_ = pre_serve_model
        return self

    def _build_user_snapshot(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """One row per user: history strictly before ``as_of_timestamp_``."""
        snap = history_df.groupby("uid").agg(
            user_impressions_before=("uid", "size"),
            user_clicks_before=("click", "sum"),
        )
        snap["user_click_rate_before"] = np.where(
            snap["user_impressions_before"] > 0,
            snap["user_clicks_before"] / snap["user_impressions_before"],
            np.nan,
        )
        snap["user_distinct_campaigns_before"] = history_df.groupby("uid")["campaign"].nunique()

        last_click_ts = history_df[history_df["click"] == 1].groupby("uid")["timestamp"].max()
        time_since_last_click = (self.as_of_timestamp_ - last_click_ts).reindex(snap.index)
        snap["time_since_last_click"] = time_since_last_click.fillna(-1.0)
        return snap

    def _build_pair_snapshot(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """One row per (uid, campaign): pair history strictly before ``as_of_timestamp_``."""
        return history_df.groupby(["uid", "campaign"]).agg(
            pair_impressions_before=("uid", "size"),
            pair_clicks_before=("click", "sum"),
        )

    def cohort_for(self, uid: Any) -> str:
        """Cohort for ``uid``, defaulting to ``COLD`` if never seen in training.

        The coldest possible case — matches the convention already
        established in ``src.evaluation.retrieval_metrics.evaluate_retrieval_by_cohort``.
        """
        if self.cohorts_ is None:
            raise RuntimeError("RecommendationPipeline must be fit() before use.")
        return self.cohorts_.get(uid, COLD)

    def _retrieve(self, uid: Any) -> List[Any]:
        cohort = self.cohort_for(uid)
        if cohort == WARM:
            candidates = self.cf_gen_.generate(uid, None, k=self.k, exclude_seen=False)
            if candidates:
                return candidates
        return self.pop_gen_.generate(user=None, context=None, k=self.k)

    def _score(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Score ``(uid, campaign)`` candidate pairs with the pre-serve model.

        Args:
            pairs_df: Must have ``uid`` and ``campaign`` columns.

        Returns:
            Predicted CTR per row, in the same order as ``pairs_df``.
        """
        features = pairs_df[["campaign"]].copy()

        # A user absent from user_snapshot_ (no history at all before the cutoff)
        # reindexes to NaN across every column -- np.where replaces that with the
        # explicit "no history" default for each column (NaN itself, for the rate
        # column, which is the same convention used for a user WHO has history but
        # zero prior impressions -- see add_point_in_time_user_features).
        user_snap_aligned = self.user_snapshot_.reindex(pairs_df["uid"])
        for col in _USER_FEATURE_COLUMNS:
            values = user_snap_aligned[col].to_numpy(dtype=float)
            default = _NO_HISTORY_USER_DEFAULTS[col]
            features[col] = np.where(np.isnan(values), default, values)

        pair_idx = pd.MultiIndex.from_arrays([pairs_df["uid"], pairs_df["campaign"]])
        pair_vals = self.pair_snapshot_.reindex(pair_idx)
        features["pair_impressions_before"] = pair_vals["pair_impressions_before"].fillna(0).to_numpy()
        features["pair_clicks_before"] = pair_vals["pair_clicks_before"].fillna(0).to_numpy()

        features = attach_campaign_aggregates(features, self.campaign_agg_, self.global_train_ctr_)
        features["campaign"] = features["campaign"].astype("category")

        assert_no_post_outcome_leakage(list(features.columns))
        assert not (set(features.columns) & set(IMPRESSION_CONTEXT_ONLY_COLUMNS)), (
            "cat1..cat9 must never reach candidate scoring -- they don't exist for a "
            "not-yet-served candidate (docs/ranking_problem_design.md Q5)."
        )

        return self.model_.predict(features[PRE_SERVE_FEATURE_COLUMNS])

    def recommend(self, uid: Any) -> List[Recommendation]:
        """Retrieve and rank candidates for a single user.

        Args:
            uid: User identifier.

        Returns:
            Candidates ranked by descending predicted CTR. Empty if
            retrieval produces no candidates at all (should not happen in
            practice — the popularity fallback always has output as long as
            the catalog is non-empty).
        """
        if self.model_ is None:
            raise RuntimeError("RecommendationPipeline must be fit() before use.")

        cohort = self.cohort_for(uid)
        candidates = self._retrieve(uid)
        if not candidates:
            return []

        pairs_df = pd.DataFrame({"uid": [uid] * len(candidates), "campaign": candidates})
        scores = self._score(pairs_df)
        order = np.argsort(-scores, kind="stable")

        return [
            Recommendation(campaign=candidates[i], predicted_ctr=float(scores[i]), cohort=cohort)
            for i in order
        ]

    def recommend_batch(self, uids: List[Any]) -> Dict[Any, List[Recommendation]]:
        """Vectorized ``recommend`` for many users — one scoring call for all
        (user, candidate) pairs at once, instead of one call per user.

        Produces identical results to calling ``recommend`` per user (see
        ``tests/test_pipeline.py::test_batch_matches_single_recommend``);
        exists purely for the population-scale consistency check
        (hundreds of thousands of users), where a per-user model.predict()
        call would be far slower.

        Args:
            uids: User identifiers.

        Returns:
            ``{uid: [Recommendation, ...]}``, one entry per input ``uid``
            (empty list if that user's retrieval produced no candidates).
        """
        if self.model_ is None:
            raise RuntimeError("RecommendationPipeline must be fit() before use.")

        cohorts = {u: self.cohort_for(u) for u in uids}
        candidates_per_user = {u: self._retrieve(u) for u in uids}

        rows = [(u, c) for u in uids for c in candidates_per_user[u]]
        if not rows:
            return {u: [] for u in uids}

        pairs_df = pd.DataFrame(rows, columns=["uid", "campaign"])
        scores = self._score(pairs_df)

        results: Dict[Any, List[Recommendation]] = {u: [] for u in uids}
        pos = 0
        for u in uids:
            n = len(candidates_per_user[u])
            if n == 0:
                continue
            user_scores = scores[pos : pos + n]
            user_candidates = candidates_per_user[u]
            order = np.argsort(-user_scores, kind="stable")
            results[u] = [
                Recommendation(campaign=user_candidates[i], predicted_ctr=float(user_scores[i]), cohort=cohorts[u])
                for i in order
            ]
            pos += n
        return results
