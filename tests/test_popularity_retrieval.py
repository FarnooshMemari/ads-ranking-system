"""Tests for PopularityCandidateGenerator, using tiny synthetic fixtures."""

import pandas as pd
import pytest

from src.retrieval.candidate_generator import PopularityCandidateGenerator


def test_generate_before_fit_raises():
    with pytest.raises(RuntimeError):
        PopularityCandidateGenerator().generate(user=None, context=None, k=5)


def test_ranks_by_click_count_descending():
    train_df = pd.DataFrame(
        {
            "campaign": [100, 100, 100, 200, 200, 300],
            "click": [1, 1, 1, 1, 1, 1],
        }
    )
    gen = PopularityCandidateGenerator().fit(train_df)

    assert gen.generate(user=None, context=None, k=3) == [100, 200, 300]


def test_generation_is_identical_for_every_user():
    """The whole point of this baseline: no personalization."""
    train_df = pd.DataFrame({"campaign": [100, 200, 200], "click": [1, 1, 1]})
    gen = PopularityCandidateGenerator().fit(train_df)

    assert gen.generate(user="alice", context=None, k=2) == gen.generate(
        user="bob", context={"foo": "bar"}, k=2
    )


def test_ties_are_broken_deterministically_by_ascending_campaign_id():
    train_df = pd.DataFrame(
        {"campaign": [300, 300, 100, 100, 200, 200], "click": [1, 1, 1, 1, 1, 1]}
    )
    gen = PopularityCandidateGenerator().fit(train_df)

    # all three campaigns tie at 2 clicks each -> ascending campaign id order
    assert gen.generate(user=None, context=None, k=3) == [100, 200, 300]
    # re-fitting should reproduce the same order (determinism, not luck)
    gen2 = PopularityCandidateGenerator().fit(train_df)
    assert gen2.generate(user=None, context=None, k=3) == [100, 200, 300]


def test_unseen_campaign_never_appears():
    """A campaign with zero training-period clicks (or absent entirely from
    train) must never be retrieved -- it simply isn't in the popularity ranking."""
    train_df = pd.DataFrame({"campaign": [100, 100, 200], "click": [1, 1, 1]})
    gen = PopularityCandidateGenerator().fit(train_df)

    result = gen.generate(user=None, context=None, k=10)
    assert 999 not in result  # campaign 999 never appeared in training data
    assert set(result) == {100, 200}


def test_k_larger_than_catalog_returns_all_available_without_padding():
    train_df = pd.DataFrame({"campaign": [100, 200, 300], "click": [1, 1, 1]})
    gen = PopularityCandidateGenerator().fit(train_df)

    result = gen.generate(user=None, context=None, k=1000)
    assert len(result) == 3
    assert set(result) == {100, 200, 300}


def test_custom_item_and_positive_columns():
    train_df = pd.DataFrame({"ad_id": [1, 1, 2], "converted": [1, 1, 1]})
    gen = PopularityCandidateGenerator().fit(train_df, item_col="ad_id", positive_col="converted")

    assert gen.generate(user=None, context=None, k=2) == [1, 2]
