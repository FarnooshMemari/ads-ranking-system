"""Lightweight structural validation for the Criteo sample dataset.

A handful of targeted checks — not a general-purpose data-quality framework.
Intended to catch obvious problems (wrong schema, corrupt download, malformed
rows) before the data reaches feature engineering.
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.data.criteo import CATEGORICAL_COLUMNS, HEADER, LABEL_COLUMN, NUMERICAL_COLUMNS


@dataclass
class ValidationResult:
    """Result of validating a Criteo-shaped DataFrame."""

    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_criteo_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Run lightweight structural checks on a Criteo-shaped DataFrame.

    Checks:
        - the DataFrame is non-empty
        - all expected columns are present (and no unexpected ones)
        - the label column is fully populated and binary (only 0/1)
        - numerical columns are numeric-typed (missing values are allowed)
        - categorical columns are string/object-typed (missing values are allowed)
        - no rows where every feature column is missing

    Args:
        df: A DataFrame expected to follow the Criteo schema (``src.data.criteo.HEADER``).

    Returns:
        A ``ValidationResult``; ``result.is_valid`` is False if any check failed,
        with human-readable messages in ``result.errors``.
    """
    result = ValidationResult()

    if df.empty:
        result.errors.append("Dataset is empty (0 rows).")
        return result

    missing_columns = [c for c in HEADER if c not in df.columns]
    if missing_columns:
        result.errors.append(f"Missing expected columns: {missing_columns}")
        return result  # remaining checks assume the schema is present

    unexpected_columns = [c for c in df.columns if c not in HEADER]
    if unexpected_columns:
        result.errors.append(f"Unexpected columns present: {unexpected_columns}")

    if df[LABEL_COLUMN].isna().any():
        result.errors.append(f"'{LABEL_COLUMN}' contains missing values; it must be fully populated.")
    else:
        invalid_labels = sorted(set(df[LABEL_COLUMN].unique()) - {0, 1})
        if invalid_labels:
            result.errors.append(f"'{LABEL_COLUMN}' contains values outside {{0, 1}}: {invalid_labels}")

    for col in NUMERICAL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            result.errors.append(f"Numerical column '{col}' is not numeric-typed.")

    for col in CATEGORICAL_COLUMNS:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            result.errors.append(f"Categorical column '{col}' is not string-typed.")

    feature_columns = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    fully_empty_rows = int(df[feature_columns].isna().all(axis=1).sum())
    if fully_empty_rows:
        result.errors.append(f"{fully_empty_rows} row(s) have every feature column missing.")

    return result
