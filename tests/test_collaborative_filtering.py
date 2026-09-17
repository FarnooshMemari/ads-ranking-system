"""Tests for CollaborativeFilteringCandidateGenerator, using tiny synthetic fixtures."""

import pandas as pd
import pytest

from src.data.attribution import split_by_day
from src.retrieval.candidate_generator import CollaborativeFilteringCandidateGenerator


def _row(day, uid, campaign, click):
    return {"timestamp": day * 86400 + 1, "uid": uid, "campaign": campaign, "click": click}


def test_invalid_similarity_type_rejected():
    with pytest.raises(ValueError):
        CollaborativeFilteringCandidateGenerator(similarity_type="nope")


def test_generate_before_fit_raises():
    with pytest.raises(RuntimeError):
        CollaborativeFilteringCandidateGenerator().generate(user=1, context=None, k=5)


def test_unknown_user_returns_empty_list():
    train_df = pd.DataFrame(
        [_row(0, 1, 100, 1), _row(0, 1, 200, 1), _row(0, 2, 100, 1), _row(0, 2, 200, 1)]
    )
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    assert gen.generate(user=999, context=None, k=5) == []


def test_unknown_campaign_never_recommended():
    """A campaign that never appears in training (e.g. only shows up in test
    ground truth) must never be retrieved -- it simply has no similarity row."""
    train_df = pd.DataFrame(
        [_row(0, 1, 100, 1), _row(0, 1, 200, 1), _row(0, 2, 100, 1), _row(0, 2, 200, 1)]
    )
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    result = gen.generate(user=1, context=None, k=10)
    assert 999 not in result  # campaign 999 never appeared in training


def test_users_who_coclick_produce_positive_similarity_and_recommendation():
    # users 1 and 2 both click 100 and 200; user 3 only clicks 100.
    # -> campaign 200 should be recommended to user 3 via co-click with 100.
    train_df = pd.DataFrame(
        [
            _row(0, 1, 100, 1), _row(0, 1, 200, 1),
            _row(0, 2, 100, 1), _row(0, 2, 200, 1),
            _row(0, 3, 100, 1),
        ]
    )
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    result = gen.generate(user=3, context=None, k=5)
    assert 200 in result
    assert 100 not in result  # zero self-similarity by construction


def test_exclude_seen_removes_previously_clicked_campaigns():
    train_df = pd.DataFrame(
        [
            _row(0, 1, 100, 1), _row(0, 1, 200, 1),
            _row(0, 2, 100, 1), _row(0, 2, 200, 1),
            _row(0, 3, 100, 1), _row(0, 3, 200, 1),
        ]
    )
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    without_exclusion = gen.generate(user=1, context=None, k=10, exclude_seen=False)
    with_exclusion = gen.generate(user=1, context=None, k=10, exclude_seen=True)

    assert 200 in without_exclusion  # a campaign the user already clicked
    assert 200 not in with_exclusion


def test_k_larger_than_available_positive_scores_returns_fewer_without_padding():
    train_df = pd.DataFrame([_row(0, 1, 100, 1), _row(0, 2, 100, 1)])  # only 1 item, no co-click pairs
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    # user 1 has affinity only for campaign 100; nothing else has positive similarity to it
    result = gen.generate(user=1, context=None, k=1000)
    assert len(result) <= 1000
    assert 100 not in result  # self-similarity excluded, and nothing else co-occurs


def test_deterministic_across_repeated_calls():
    train_df = pd.DataFrame(
        [
            _row(0, 1, 100, 1), _row(0, 1, 200, 1),
            _row(0, 2, 100, 1), _row(0, 2, 200, 1), _row(0, 2, 300, 1),
        ]
    )
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    first = gen.generate(user=1, context=None, k=5)
    second = gen.generate(user=1, context=None, k=5)
    assert first == second


def test_deterministic_tie_break_by_ascending_campaign_id():
    # user 1 clicks 100; users 2 and 3 each co-click 100 with a distinct campaign,
    # constructed so 200 and 300 tie in jaccard similarity to 100.
    train_df = pd.DataFrame(
        [
            _row(0, 1, 100, 1),
            _row(0, 2, 100, 1), _row(0, 2, 200, 1),
            _row(0, 3, 100, 1), _row(0, 3, 300, 1),
        ]
    )
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    result = gen.generate(user=1, context=None, k=2)
    assert result == [200, 300]  # tie broken by ascending campaign id


def test_train_only_interaction_construction_no_leakage_end_to_end():
    """A co-click that only happens in the test period must not create
    similarity between two campaigns when the generator is fit on the
    correctly-split train set."""
    df = pd.DataFrame(
        [
            _row(0, 1, 100, 1),    # train
            _row(0, 2, 100, 1),    # train
            _row(26, 1, 200, 1),   # test-period only
            _row(26, 2, 200, 1),   # test-period only -- would create 100<->200 co-click if leaked
        ]
    )
    train_df, _, _ = split_by_day(df, train_end_day=20, val_end_day=25)
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    assert 200 not in gen.items_  # never observed in the (correctly restricted) training data
    result = gen.generate(user=1, context=None, k=10)
    assert 200 not in result


def test_cooccurrence_similarity_type_selectable():
    train_df = pd.DataFrame(
        [
            _row(0, 1, 100, 1), _row(0, 1, 200, 1),
            _row(0, 2, 100, 1), _row(0, 2, 200, 1),
            _row(0, 3, 100, 1),
        ]
    )
    gen = CollaborativeFilteringCandidateGenerator(similarity_type="cooccurrence").fit(train_df)

    result = gen.generate(user=3, context=None, k=5)
    assert 200 in result


def test_coverage_stats():
    train_df = pd.DataFrame(
        [_row(0, 1, 100, 1), _row(0, 1, 200, 1), _row(0, 2, 100, 1)]
    )
    gen = CollaborativeFilteringCandidateGenerator().fit(train_df)

    stats = gen.coverage_stats()
    assert stats["n_items_in_catalog"] == 2
    assert stats["n_users_with_affinity"] == 2


def test_coverage_stats_before_fit_raises():
    with pytest.raises(RuntimeError):
        CollaborativeFilteringCandidateGenerator().coverage_stats()
