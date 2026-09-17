"""Tests for Criteo dataset validation, using a tiny synthetic fixture.

No real Criteo data is downloaded or required for these tests.
"""

import pandas as pd
import pytest

from src.data.criteo import CATEGORICAL_COLUMNS, HEADER, NUMERICAL_COLUMNS
from src.data.validation import validate_criteo_dataframe


def _make_valid_df(n_rows: int = 4) -> pd.DataFrame:
    # Missing values are staggered between numerical and categorical columns so
    # that no single row ends up with every feature column missing.
    data = {"label": [0, 1, 0, 1][:n_rows]}
    for col in NUMERICAL_COLUMNS:
        data[col] = [1, None, 3, 4][:n_rows]
    for col in CATEGORICAL_COLUMNS:
        data[col] = [None, "b2c3d4", "d4e5f6", "a1b2c3"][:n_rows]
    return pd.DataFrame(data, columns=HEADER)


def test_valid_synthetic_dataframe_passes():
    df = _make_valid_df()
    result = validate_criteo_dataframe(df)

    assert result.is_valid
    assert result.errors == []


def test_empty_dataframe_is_invalid():
    df = _make_valid_df(n_rows=0)
    result = validate_criteo_dataframe(df)

    assert not result.is_valid
    assert "empty" in result.errors[0].lower()


def test_missing_columns_detected():
    df = _make_valid_df().drop(columns=["cat00"])
    result = validate_criteo_dataframe(df)

    assert not result.is_valid
    assert any("Missing expected columns" in e for e in result.errors)


def test_invalid_label_values_detected():
    df = _make_valid_df()
    df.loc[0, "label"] = 2
    result = validate_criteo_dataframe(df)

    assert not result.is_valid
    assert any("label" in e and "0, 1" in e for e in result.errors)


def test_missing_label_values_detected():
    df = _make_valid_df()
    df.loc[0, "label"] = None
    result = validate_criteo_dataframe(df)

    assert not result.is_valid
    assert any("missing values" in e for e in result.errors)


def test_non_numeric_numerical_column_detected():
    df = _make_valid_df()
    df["int00"] = ["not", "a", "number", "here"][: len(df)]
    result = validate_criteo_dataframe(df)

    assert not result.is_valid
    assert any("int00" in e and "numeric" in e for e in result.errors)


def test_fully_empty_row_detected():
    df = _make_valid_df()
    df.loc[0, NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS] = None
    result = validate_criteo_dataframe(df)

    assert not result.is_valid
    assert any("feature column" in e and "missing" in e for e in result.errors)


@pytest.mark.parametrize("n_rows", [1, 2, 4])
def test_valid_dataframe_of_various_sizes_passes(n_rows):
    df = _make_valid_df(n_rows=n_rows)
    result = validate_criteo_dataframe(df)

    assert result.is_valid
