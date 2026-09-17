"""CTR (click-through-rate) prediction model.

Implemented in Phase 2: CTR prediction. Planned to support a logistic
regression baseline and a gradient-boosted trees model (LightGBM), selected
via ``configs/config.yaml``: ``model.ctr.type``.
"""

from typing import Any

from src.models.base_model import BaseModel


class CTRModel(BaseModel):
    """Predicts the probability that a user clicks a given ad."""

    def __init__(self, params: dict | None = None):
        """Initialize the CTR model.

        Args:
            params: Model hyperparameters (see
                ``configs/config.yaml``: ``model.ctr.params``).
        """
        self.params = params or {}
        self.model = None

    def fit(self, X: Any, y: Any) -> "CTRModel":
        raise NotImplementedError("Implemented in Phase 2: CTR prediction.")

    def predict(self, X: Any) -> Any:
        """Predict click probabilities for the given feature matrix."""
        raise NotImplementedError("Implemented in Phase 2: CTR prediction.")

    def save(self, path: str) -> None:
        raise NotImplementedError("Implemented in Phase 2: CTR prediction.")

    def load(self, path: str) -> "CTRModel":
        raise NotImplementedError("Implemented in Phase 2: CTR prediction.")
