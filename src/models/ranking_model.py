"""Ranking model that orders candidate ads for a given user/context.

Implemented in Phase 3: ranking system. Planned to combine predicted CTR
(from ``CTRModel``) with bid/value and business constraints to produce a
final ranked list, and/or learn a pairwise/listwise ranking objective
directly (e.g. LambdaMART).
"""

from typing import Any

from src.models.base_model import BaseModel


class RankingModel(BaseModel):
    """Ranks a list of candidate ads for a given user/context."""

    def __init__(self, params: dict | None = None):
        """Initialize the ranking model.

        Args:
            params: Model hyperparameters (see
                ``configs/config.yaml``: ``model.ranking.params``).
        """
        self.params = params or {}
        self.model = None

    def fit(self, X: Any, y: Any) -> "RankingModel":
        raise NotImplementedError("Implemented in Phase 3: ranking system.")

    def predict(self, X: Any) -> Any:
        """Predict ranking scores for the given feature matrix."""
        raise NotImplementedError("Implemented in Phase 3: ranking system.")

    def save(self, path: str) -> None:
        raise NotImplementedError("Implemented in Phase 3: ranking system.")

    def load(self, path: str) -> "RankingModel":
        raise NotImplementedError("Implemented in Phase 3: ranking system.")

    def rank(self, candidates: Any, context: Any) -> Any:
        """Produce a ranked ordering of candidate ads.

        Args:
            candidates: Candidate ads with their features (from the
                candidate generation / retrieval stage, Phase 4).
            context: User/session context used for scoring.

        Returns:
            Candidates ordered by descending expected value/relevance.
        """
        raise NotImplementedError("Implemented in Phase 3: ranking system.")
