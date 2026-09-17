"""Base model interface shared by CTR and ranking models."""

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Common interface for all models in the ads ranking system.

    Concrete CTR and ranking models (Phase 2/3) implement this interface so
    that training, prediction, and evaluation code can operate uniformly
    across model types.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseModel":
        """Fit the model on training features/labels.

        Args:
            X: Training feature matrix.
            y: Training labels/targets.

        Returns:
            The fitted model instance (self).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Produce predictions (e.g. click probabilities or scores) for X.

        Args:
            X: Feature matrix to score.

        Returns:
            Model predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to disk.

        Args:
            path: Destination file path.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        """Load a trained model from disk.

        Args:
            path: Source file path.

        Returns:
            The loaded model instance (self).
        """
        raise NotImplementedError
