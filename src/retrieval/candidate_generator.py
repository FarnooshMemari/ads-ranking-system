"""Candidate generation (retrieval) — the first stage of the ranking funnel.

Narrows a large ad pool down to a small set of candidates for a given user
and context, before those candidates are scored and ordered by the ranking
stage (``src/models/ranking_model.py``):

    Large ad pool -> Candidate generation/retrieval -> Ranking model

Implemented in Phase 4: recommendation / candidate generation.
"""

from abc import ABC, abstractmethod
from typing import Any, List


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
