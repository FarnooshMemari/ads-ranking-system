"""Feature engineering for CTR prediction and ranking.

Implementations land starting in Phase 1/2 as the feature set is defined
against a concrete dataset.
"""

from typing import Any, List


def build_features(data: Any, config: dict) -> Any:
    """Build the full model-ready feature matrix from raw/processed data.

    Args:
        data: Input dataset (e.g. a ``pandas.DataFrame``) containing raw
            user, ad, and context columns.
        config: Feature configuration (see ``configs/config.yaml``:
            ``features.categorical_columns`` / ``features.numerical_columns``).

    Returns:
        A feature matrix ready for model training/inference.
    """
    raise NotImplementedError("Implemented in Phase 1/2: feature engineering.")


def encode_categorical_features(data: Any, columns: List[str]) -> Any:
    """Encode categorical columns (e.g. target/hash/embedding encoding).

    Args:
        data: Input dataset.
        columns: Names of categorical columns to encode.

    Returns:
        The dataset with categorical columns encoded numerically.
    """
    raise NotImplementedError("Implemented in Phase 1/2: feature engineering.")


def generate_cross_features(data: Any, feature_pairs: List[tuple]) -> Any:
    """Generate pairwise cross/interaction features (e.g. user_x_ad_category).

    Args:
        data: Input dataset.
        feature_pairs: List of ``(feature_a, feature_b)`` tuples to cross.

    Returns:
        The dataset augmented with cross features.
    """
    raise NotImplementedError("Implemented in Phase 2: CTR prediction.")
