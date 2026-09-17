"""Candidate generation (retrieval) — the first stage of the ranking funnel.

Narrows a large ad pool down to a small set of candidates for a given user
and context, before those candidates are scored and ordered by the ranking
stage (``src/models/ranking_model.py``):

    Large ad pool -> Candidate generation/retrieval -> Ranking model

``PopularityCandidateGenerator`` is implemented (the cold-start baseline —
see docs/attribution_modeling_design.md §3B/§4 and docs/popularity_baseline.md).
The embedding- and collaborative-filtering-based generators remain Phase 4
placeholders, deliberately deferred pending the warm-cohort evaluation this
baseline exists to be compared against.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

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
    """Retrieves candidates via collaborative filtering (user-ad interaction
    co-occurrence, matrix factorization, etc.).

    Implemented in Phase 4: recommendation / candidate generation.
    """

    def __init__(self, params: dict | None = None):
        """Initialize the collaborative filtering candidate generator.

        Args:
            params: Retrieval hyperparameters (e.g. number of latent factors).
        """
        self.params = params or {}

    def fit(self, interactions: Any) -> "CollaborativeFilteringCandidateGenerator":
        """Fit the collaborative filtering model on historical interactions.

        Args:
            interactions: Historical user-ad interaction data.

        Returns:
            The fitted generator instance (self).
        """
        raise NotImplementedError("Implemented in Phase 4: candidate generation.")

    def generate(self, user: Any, context: Any, k: int) -> List[Any]:
        raise NotImplementedError("Implemented in Phase 4: candidate generation.")


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
