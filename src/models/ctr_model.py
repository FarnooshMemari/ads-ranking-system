"""CTR (click-through-rate) prediction: a global-CTR baseline and a LightGBM classifier.

Implements the first ranking-stage experiment from docs/ranking_problem_design.md:
``P(click | impression)``, trained and evaluated only on real, already-served
impressions — see docs/ctr_model.md for the full experiment writeup.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel


class GlobalCTRBaseline(BaseModel):
    """Predicts the same constant probability (the training-period global CTR)
    for every impression, regardless of any feature.

    This is a **baseline, not a ranking model**: because it ignores every
    feature, it cannot distinguish between any two candidates — every
    impression gets the identical score, so it induces no ordering at all.
    It exists to answer one question: how much of LightGBM's apparent
    performance comes from actually learning something feature-dependent,
    versus just reflecting the dataset's overall click rate. See
    docs/ctr_model.md.
    """

    def __init__(self):
        self.global_ctr_: Optional[float] = None

    def fit(self, X: Any, y: Sequence[int]) -> "GlobalCTRBaseline":
        """Compute the global click rate from ``y``. ``X`` is ignored entirely.

        Args:
            X: Ignored.
            y: Binary click labels for the training set.

        Returns:
            self.
        """
        self.global_ctr_ = float(pd.Series(y).mean())
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the constant global CTR for every row in ``X``.

        Args:
            X: Any object supporting ``len()``.

        Returns:
            An array of length ``len(X)``, every value equal to the training
            global CTR.
        """
        if self.global_ctr_ is None:
            raise RuntimeError("GlobalCTRBaseline must be fit() before predict().")
        return np.full(len(X), self.global_ctr_, dtype=float)

    def save(self, path: str) -> None:
        if self.global_ctr_ is None:
            raise RuntimeError("GlobalCTRBaseline must be fit() before save().")
        with open(path, "w") as f:
            f.write(repr(self.global_ctr_))

    def load(self, path: str) -> "GlobalCTRBaseline":
        with open(path, "r") as f:
            self.global_ctr_ = float(f.read())
        return self


class CTRModel(BaseModel):
    """LightGBM binary classifier: ``P(click = 1 | impression)``.

    Categorical features (``campaign``, ``cat1``–``cat9``) are expected to
    already be pandas ``category`` dtype on the input (see
    ``src.features.attribution_features.build_ctr_features``, which casts
    them consistently across the full dataset before any split, so category
    encodings never drift between train/validation/test).
    """

    def __init__(
        self,
        params: Dict[str, Any],
        categorical_features: Optional[List[str]] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: Optional[int] = 30,
    ):
        """Initialize the CTR model.

        Args:
            params: LightGBM training parameters (see
                ``configs/config.yaml``: ``attribution.ranking.ctr_model.params``).
            categorical_features: Column names to treat as categorical.
            num_boost_round: Maximum number of boosting rounds.
            early_stopping_rounds: Stop early if the validation metric hasn't
                improved for this many rounds (requires ``eval_set`` in
                ``fit``); ``None`` disables early stopping.
        """
        self.params = params
        self.categorical_features = categorical_features or []
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: Optional[lgb.Booster] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[int],
        eval_set: Optional[Tuple[pd.DataFrame, Sequence[int]]] = None,
    ) -> "CTRModel":
        """Train on ``(X, y)``, optionally early-stopping on ``eval_set``.

        Args:
            X: Training feature matrix (see
                ``src.features.attribution_features.FEATURE_COLUMNS``).
            y: Binary click labels.
            eval_set: Optional ``(X_val, y_val)`` — **validation data only**;
                used exclusively for early stopping (a model/configuration
                decision), never for final reporting. The test period must
                never be passed here.

        Returns:
            self.
        """
        self.feature_names_ = list(X.columns)
        train_set = lgb.Dataset(
            X, label=y, categorical_feature=self.categorical_features, free_raw_data=False
        )

        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=0)]
        if eval_set is not None:
            X_val, y_val = eval_set
            val_set = lgb.Dataset(
                X_val[self.feature_names_],
                label=y_val,
                categorical_feature=self.categorical_features,
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets.append(val_set)
            valid_names.append("valid")
            if self.early_stopping_rounds:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict click probabilities.

        Args:
            X: Feature matrix with at least the columns used in ``fit``.

        Returns:
            An array of predicted P(click=1), using the best (early-stopped)
            iteration if one was found.
        """
        if self.booster is None:
            raise RuntimeError("CTRModel must be fit() before predict().")
        best_iter = self.booster.best_iteration if self.booster.best_iteration else None
        return self.booster.predict(X[self.feature_names_], num_iteration=best_iter)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importances, most important first.

        Args:
            importance_type: ``"gain"`` (total split gain contributed by the
                feature) or ``"split"`` (number of times the feature was
                used to split). Gain-based importance reflects how much the
                *trained model* relies on a feature — it is not a causal
                effect estimate and should not be interpreted as one.

        Returns:
            A Series indexed by feature name, sorted descending.
        """
        if self.booster is None:
            raise RuntimeError("CTRModel must be fit() before feature_importance().")
        importances = self.booster.feature_importance(importance_type=importance_type)
        return pd.Series(importances, index=self.booster.feature_name()).sort_values(ascending=False)

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("CTRModel must be fit() before save().")
        self.booster.save_model(path)

    def load(self, path: str) -> "CTRModel":
        self.booster = lgb.Booster(model_file=path)
        self.feature_names_ = self.booster.feature_name()
        return self
