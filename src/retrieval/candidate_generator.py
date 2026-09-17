"""Candidate generation (retrieval) — the first stage of the ranking funnel.

Narrows a large ad pool down to a small set of candidates for a given user
and context, before those candidates are scored and ordered by the ranking
stage (``src/models/ranking_model.py``):

    Large ad pool -> Candidate generation/retrieval -> Ranking model

``PopularityCandidateGenerator`` (cold-start baseline) and
``CollaborativeFilteringCandidateGenerator`` (warm-cohort item-based CF) are
implemented — see docs/attribution_modeling_design.md §3B/§4,
docs/popularity_baseline.md, and docs/collaborative_filtering.md.
``EmbeddingCandidateGenerator`` remains a Phase 4 placeholder.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class BaseCandidateGenerator(ABC):
    """Common interface for all candidate generation strategies.

    Concrete retrieval strategies (embedding-based nearest neighbor,
    collaborative filtering, rule-based targeting, ...) implement this
    interface so the ranking stage can consume candidates uniformly
    regardless of how they were retrieved.
    """

    @abstractmethod
    def generate(self, user: Any, context: Any, k: int) -> List[Any]:
        """Retrieve the top-k candidate ads for a user/context.

        Args:
            user: User identifier or feature representation.
            context: Request context (surface, session, timestamp, ...).
            k: Maximum number of candidates to retrieve.

        Returns:
            A list of up to ``k`` candidate ads to pass to the ranking stage.
        """
        raise NotImplementedError


class EmbeddingCandidateGenerator(BaseCandidateGenerator):
    """Retrieves candidates via approximate nearest-neighbor search over
    learned user/ad embeddings.

    Implemented in Phase 4: recommendation / candidate generation.
    """

    def __init__(self, params: dict | None = None):
        """Initialize the embedding-based candidate generator.

        Args:
            params: Retrieval hyperparameters (e.g. embedding dimension,
                similarity metric, ANN index type).
        """
        self.params = params or {}
        self.index = None

    def fit(self, ads: Any) -> "EmbeddingCandidateGenerator":
        """Build the retrieval index over the ad pool.

        Args:
            ads: The full ad inventory to index.

        Returns:
            The fitted generator instance (self).
        """
        raise NotImplementedError("Implemented in Phase 4: candidate generation.")

    def generate(self, user: Any, context: Any, k: int) -> List[Any]:
        raise NotImplementedError("Implemented in Phase 4: candidate generation.")


class PopularityCandidateGenerator(BaseCandidateGenerator):
    """Retrieves the same training-period-popularity-ranked items for every
    user, regardless of their individual history.

    This is **not personalized** — it is the retrieval strategy for the
    cold-start cohort (see docs/attribution_modeling_design.md §3B/§4), where
    94.61% of users have no relative-preference signal for any personalized
    method to learn from. A separate cohort-classification step
    (``src/retrieval/cohort.py``) decides which users this strategy should be
    applied to; this class has no cohort awareness of its own.
    """

    def __init__(self):
        self.popularity_: Optional[pd.Series] = None

    def fit(
        self,
        train_interactions: pd.DataFrame,
        item_col: str = "campaign",
        positive_col: str = "click",
    ) -> "PopularityCandidateGenerator":
        """Compute a training-period popularity ranking.

        Args:
            train_interactions: Training-period interaction rows only — no
                time filtering is performed here (see
                ``src.data.attribution.split_by_day``); passing anything but
                a pre-filtered training split will leak future information
                into the ranking.
            item_col: Column identifying the recommendable item (e.g.
                ``"campaign"``).
            positive_col: Binary column whose sum defines popularity (e.g.
                ``"click"``).

        Returns:
            self, with ``popularity_`` set to a ``pandas.Series`` indexed by
            item id, sorted descending by count. Ties are broken
            deterministically by ascending item id (stable sort on an
            index-sorted Series), so the ranking is reproducible run to run.
        """
        counts = train_interactions.groupby(item_col)[positive_col].sum()
        self.popularity_ = counts.sort_index().sort_values(ascending=False, kind="mergesort")
        return self

    def generate(self, user: Any, context: Any, k: int) -> List[Any]:
        """Return the top-k most popular items — identical for every user.

        Args:
            user: Ignored — this generator is not personalized.
            context: Ignored.
            k: Number of items to retrieve.

        Returns:
            Up to ``k`` item ids, most popular first. Fewer than ``k`` if the
            catalog itself has fewer than ``k`` items (no padding or error).
        """
        if self.popularity_ is None:
            raise RuntimeError("PopularityCandidateGenerator must be fit() before generate().")
        return self.popularity_.index[:k].tolist()


class CollaborativeFilteringCandidateGenerator(BaseCandidateGenerator):
    """Item-based collaborative filtering via co-occurrence similarity.

    This follows the SAR (Simple Algorithm for Recommendation) design pattern
    from Microsoft Recommenders — verified directly from
    ``recommenders/models/sar/sar_singlenode.py``, which depends only on
    numpy/pandas/scipy (no Spark, no GPU) and uses this same similarity
    vocabulary (cooccurrence, jaccard, ...). Self-implemented here (no new
    dependency) rather than importing the `recommenders` package, consistent
    with how the rest of this project's data/retrieval code is built.

    Memory-based, not embedding-based: there is no per-user vector to learn
    from sparse data — a user's score for a candidate campaign is the sum,
    over campaigns they've already clicked, of (their affinity for that
    campaign) x (its similarity to the candidate). This is deliberately
    chosen over latent-factor methods (e.g. BPR) because it does not require
    estimating a compressed per-user representation from very few
    interactions — see docs/collaborative_filtering.md for the full
    comparison and rationale.

    Scoped to the warm cohort only — see
    docs/attribution_modeling_design.md §3B/§4/§9. Not intended to be fit on
    or evaluated against the cold cohort, which has no relative-preference
    signal to learn from.
    """

    VALID_SIMILARITY_TYPES = ("jaccard", "cooccurrence")

    def __init__(self, similarity_type: str = "jaccard"):
        """Initialize the collaborative filtering candidate generator.

        Args:
            similarity_type: ``"jaccard"`` (co-occurring users normalized by
                the union of each campaign's clickers — robust to popularity
                skew) or ``"cooccurrence"`` (raw co-click counts — biased
                toward popular campaigns, kept for comparison).
        """
        if similarity_type not in self.VALID_SIMILARITY_TYPES:
            raise ValueError(
                f"similarity_type must be one of {self.VALID_SIMILARITY_TYPES}, got {similarity_type!r}"
            )
        self.similarity_type = similarity_type
        self.items_: Optional[List[Any]] = None
        self.similarity_: Optional[pd.DataFrame] = None
        self.user_affinity_: Optional[Dict[Any, Dict[Any, float]]] = None

    def fit(
        self,
        train_interactions: pd.DataFrame,
        user_col: str = "uid",
        item_col: str = "campaign",
        positive_col: str = "click",
    ) -> "CollaborativeFilteringCandidateGenerator":
        """Fit item-item similarity and per-user affinity from positive interactions.

        Args:
            train_interactions: Training-period interaction rows only — no
                time filtering is performed here (see
                ``src.data.attribution.split_by_day``); passing anything but
                a pre-filtered, warm-cohort-restricted training split will
                leak information unavailable at recommendation time.
            user_col: Column identifying the user.
            item_col: Column identifying the recommendable item.
            positive_col: Binary column whose presence defines a positive
                interaction (e.g. ``"click"``).

        Returns:
            self, with ``items_`` (the item catalog observed in training),
            ``similarity_`` (an item x item DataFrame), and
            ``user_affinity_`` (``{user: {item: click_count}}``) set.
        """
        positives = train_interactions[train_interactions[positive_col] == 1]
        self.items_ = sorted(positives[item_col].unique().tolist())

        affinity_counts = positives.groupby([user_col, item_col]).size()
        affinity_df = affinity_counts.unstack(fill_value=0).reindex(columns=self.items_, fill_value=0)

        binary = (affinity_df.to_numpy() > 0).astype(float)  # n_users x n_items
        # Verified benign: some BLAS backends (observed with Apple Accelerate on
        # arm64) emit spurious divide-by-zero/overflow/invalid-value
        # RuntimeWarnings on this matmul shape even for clean 0/1 float64 input
        # with no NaN/Inf. Cross-checked the result against a manual dot-product
        # and against independently-computed per-item counts on both synthetic
        # and real data -- always exact, no NaN/Inf in the output. Suppressed
        # here rather than left to alarm anyone reading the output.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cooccurrence = binary.T @ binary  # n_items x n_items, symmetric

        if self.similarity_type == "cooccurrence":
            sim = cooccurrence
        else:  # jaccard
            item_click_user_counts = np.diag(cooccurrence).copy()
            union = item_click_user_counts[:, None] + item_click_user_counts[None, :] - cooccurrence
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(union > 0, cooccurrence / union, 0.0)
        np.fill_diagonal(sim, 0.0)  # an item's similarity to itself is not used for scoring

        self.similarity_ = pd.DataFrame(sim, index=self.items_, columns=self.items_)
        self.user_affinity_ = {
            uid: {item: int(count) for item, count in row.items() if count > 0}
            for uid, row in affinity_df.iterrows()
        }
        return self

    def generate(
        self, user: Any, context: Any, k: int, exclude_seen: bool = False
    ) -> List[Any]:
        """Score all campaigns for ``user`` and return the top-k.

        Args:
            user: A user id present in ``user_affinity_`` (i.e. seen during
                ``fit``). A user absent from training returns an empty list —
                this generator has no signal to personalize for them (they
                should be routed to ``PopularityCandidateGenerator`` instead).
            context: Ignored — this generator uses only training-period
                click history.
            k: Number of items to retrieve.
            exclude_seen: If True, remove campaigns the user already clicked
                in training from the candidate list before ranking — the
                more common real-world serving policy, but not the default,
                since the popularity baseline it's compared against does not
                exclude previously-seen campaigns either (see
                docs/collaborative_filtering.md for the documented tradeoff).

        Returns:
            Up to ``k`` campaign ids with positive similarity-based
            evidence, most-supported first, deterministic under ties
            (ascending campaign id). Empty if the user is unknown or has no
            campaigns with positive score.
        """
        if self.similarity_ is None or self.user_affinity_ is None:
            raise RuntimeError("CollaborativeFilteringCandidateGenerator must be fit() before generate().")

        user_items = self.user_affinity_.get(user)
        if not user_items:
            return []

        weights = pd.Series(user_items, dtype=float).reindex(self.items_, fill_value=0.0)
        scores = weights.dot(self.similarity_)  # Series indexed by candidate item

        if exclude_seen:
            scores = scores.drop(index=[item for item in user_items if item in scores.index])

        scores = scores[scores > 0]
        ranked = scores.sort_index().sort_values(ascending=False, kind="mergesort")
        return ranked.index[:k].tolist()

    def coverage_stats(self) -> Dict[str, int]:
        """Basic fit-time coverage diagnostics.

        Returns:
            ``{"n_items_in_catalog": ..., "n_users_with_affinity": ...}`` —
            the number of distinct campaigns with any training-period click
            evidence, and the number of distinct users with a computed
            affinity vector.
        """
        if self.items_ is None or self.user_affinity_ is None:
            raise RuntimeError("CollaborativeFilteringCandidateGenerator must be fit() before coverage_stats().")
        return {
            "n_items_in_catalog": len(self.items_),
            "n_users_with_affinity": len(self.user_affinity_),
        }


def merge_candidates(*candidate_lists: List[Any], k: int) -> List[Any]:
    """Merge and deduplicate candidates from multiple retrieval strategies.

    Real systems typically blend several candidate generators (e.g.
    embedding-based + rule-based targeting) before ranking.

    Args:
        *candidate_lists: One or more lists of candidate ads.
        k: Maximum number of merged candidates to return.

    Returns:
        A deduplicated list of up to ``k`` candidates.
    """
    raise NotImplementedError("Implemented in Phase 4: candidate generation.")
