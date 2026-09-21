"""Tests for the restricted retrieval-candidate ranking check
(src/evaluation/ranking_integration.py), using tiny synthetic fixtures.
"""

import math

import pandas as pd
import pytest

from src.evaluation.ranking_integration import (
    build_exposure_maps,
    classify_test_users,
    find_contrast_users,
    pairwise_accuracy,
    verified_candidates,
)
from src.retrieval.cohort import COLD, WARM


def _test_row(uid, campaign, click):
    return {"uid": uid, "campaign": campaign, "click": click}


class TestClassifyTestUsers:
    """Regression coverage for the population-coverage bug: a user absent
    from the train-period cohort map must default to COLD, never be
    dropped from evaluation entirely."""

    def test_user_with_train_history_keeps_their_assigned_cohort(self):
        cohorts = pd.Series({1: WARM, 2: COLD})

        result = classify_test_users(cohorts, test_uids=[1, 2])

        assert result == {1: WARM, 2: COLD}

    def test_user_absent_from_train_period_defaults_to_cold_not_dropped(self):
        """The exact bug scenario: uid=99 appears in the test period but has
        zero train-period history (absent from `cohorts` entirely) -- must
        be classified COLD, not silently missing from the result."""
        cohorts = pd.Series({1: WARM, 2: COLD})  # uid=99 is NOT in here

        result = classify_test_users(cohorts, test_uids=[1, 2, 99])

        assert 99 in result  # not dropped
        assert result[99] == COLD

    def test_every_test_uid_is_present_in_the_result(self):
        """No user is ever silently excluded, regardless of train-period history."""
        cohorts = pd.Series({1: WARM}, dtype=object)
        test_uids = [1, 2, 3, 4, 5]  # only uid=1 has train history

        result = classify_test_users(cohorts, test_uids)

        assert set(result.keys()) == set(test_uids)
        assert result[1] == WARM
        for uid in [2, 3, 4, 5]:
            assert result[uid] == COLD

    def test_custom_default_cohort_respected(self):
        cohorts = pd.Series({1: WARM}, dtype=object)

        result = classify_test_users(cohorts, test_uids=[1, 2], default_cohort="custom_default")

        assert result[2] == "custom_default"

    def test_empty_cohorts_series_defaults_everyone(self):
        """The extreme case: a train period with no classified users at all
        (e.g. a tiny synthetic fixture) must still classify every test user,
        never raise, never drop anyone."""
        cohorts = pd.Series(dtype=object)

        result = classify_test_users(cohorts, test_uids=[1, 2, 3])

        assert result == {1: COLD, 2: COLD, 3: COLD}


class TestBuildExposureMaps:
    def test_shown_includes_both_clicked_and_not_clicked_rows(self):
        test_df = pd.DataFrame([_test_row(1, 100, 1), _test_row(1, 200, 0)])
        shown, _ = build_exposure_maps(test_df)

        assert shown[1] == {100, 200}

    def test_click_lookup_reflects_real_outcomes(self):
        test_df = pd.DataFrame([_test_row(1, 100, 1), _test_row(1, 200, 0)])
        _, click_lookup = build_exposure_maps(test_df)

        assert click_lookup[(1, 100)] == 1
        assert click_lookup[(1, 200)] == 0

    def test_duplicate_pair_takes_max_click(self):
        """Same (uid, campaign) shown twice in the test period, once
        clicked -- the pair counts as clicked overall."""
        test_df = pd.DataFrame([_test_row(1, 100, 0), _test_row(1, 100, 1)])
        shown, click_lookup = build_exposure_maps(test_df)

        assert shown[1] == {100}
        assert click_lookup[(1, 100)] == 1

    def test_user_absent_from_test_df_has_no_entry(self):
        test_df = pd.DataFrame([_test_row(1, 100, 1)])
        shown, _ = build_exposure_maps(test_df)

        assert 2 not in shown


class TestVerifiedCandidates:
    def test_only_actually_shown_campaigns_survive(self):
        retrieved = [100, 200, 300]
        shown_for_user = {100, 300}  # 200 was retrieved but NEVER shown -- must be dropped

        result = verified_candidates(retrieved, shown_for_user)
        assert result == [100, 300]
        assert 200 not in result

    def test_empty_exposure_set_yields_no_verified_candidates(self):
        """The core constraint, directly enforced: a user with zero real
        test-period impressions contributes zero verified candidates --
        never a fabricated negative for any retrieved campaign."""
        assert verified_candidates([100, 200], set()) == []

    def test_order_preserved(self):
        retrieved = [300, 100, 200]
        shown_for_user = {100, 200, 300}
        assert verified_candidates(retrieved, shown_for_user) == [300, 100, 200]


class TestFindContrastUsers:
    def test_user_with_only_clicks_has_no_contrast(self):
        verified_per_user = {1: [100, 200]}
        click_lookup = {(1, 100): 1, (1, 200): 1}

        assert find_contrast_users(verified_per_user, click_lookup) == {}

    def test_user_with_only_non_clicks_has_no_contrast(self):
        verified_per_user = {1: [100, 200]}
        click_lookup = {(1, 100): 0, (1, 200): 0}

        assert find_contrast_users(verified_per_user, click_lookup) == {}

    def test_user_with_both_has_a_contrast(self):
        verified_per_user = {1: [100, 200, 300]}
        click_lookup = {(1, 100): 1, (1, 200): 0, (1, 300): 0}

        result = find_contrast_users(verified_per_user, click_lookup)
        assert result[1] == ([100], [200, 300])

    def test_missing_click_lookup_entry_defaults_to_not_clicked(self):
        verified_per_user = {1: [100, 200]}
        click_lookup = {(1, 100): 1}  # 200 absent -- must not crash, defaults to 0

        result = find_contrast_users(verified_per_user, click_lookup)
        assert result[1] == ([100], [200])


class TestPairwiseAccuracy:
    def test_perfect_ranking_gives_accuracy_one(self):
        contrast_users = {1: ([100], [200])}
        scores = {(1, 100): 0.9, (1, 200): 0.1}
        acc, n_pairs, n_users = pairwise_accuracy(contrast_users, lambda u, c: scores[(u, c)])

        assert acc == 1.0
        assert n_pairs == 1
        assert n_users == 1

    def test_inverted_ranking_gives_accuracy_zero(self):
        contrast_users = {1: ([100], [200])}
        scores = {(1, 100): 0.1, (1, 200): 0.9}  # clicked item scored LOWER
        acc, n_pairs, n_users = pairwise_accuracy(contrast_users, lambda u, c: scores[(u, c)])

        assert acc == 0.0

    def test_tie_counts_as_incorrect(self):
        contrast_users = {1: ([100], [200])}
        scores = {(1, 100): 0.5, (1, 200): 0.5}
        acc, n_pairs, n_users = pairwise_accuracy(contrast_users, lambda u, c: scores[(u, c)])

        assert acc == 0.0  # strict > required, tie is not a correct ranking

    def test_multiple_clicked_and_not_clicked_produce_cross_product_pairs(self):
        contrast_users = {1: ([100, 101], [200, 201])}
        scores = {(1, 100): 0.9, (1, 101): 0.8, (1, 200): 0.2, (1, 201): 0.1}
        acc, n_pairs, n_users = pairwise_accuracy(contrast_users, lambda u, c: scores[(u, c)])

        assert n_pairs == 4  # 2 clicked x 2 not-clicked
        assert acc == 1.0

    def test_empty_contrast_users_returns_nan_accuracy(self):
        acc, n_pairs, n_users = pairwise_accuracy({}, lambda u, c: 0.5)

        assert math.isnan(acc)
        assert n_pairs == 0
        assert n_users == 0

    def test_missing_score_pair_skipped_gracefully(self):
        """score_fn returning None (e.g. no feature row found) must not
        crash and must not count toward the denominator."""
        contrast_users = {1: ([100], [200]), 2: ([300], [400])}

        def score_fn(uid, campaign):
            if uid == 1:
                return None  # simulate a missing feature row for user 1
            return {(2, 300): 0.9, (2, 400): 0.1}[(uid, campaign)]

        acc, n_pairs, n_users = pairwise_accuracy(contrast_users, score_fn)
        assert n_pairs == 1  # only user 2's pair counted
        assert n_users == 1
        assert acc == 1.0

    def test_never_scores_an_unverified_campaign(self):
        """Structural guarantee: pairwise_accuracy only ever calls score_fn
        with campaigns that came from a contrast_users entry, which itself
        can only contain verified_candidates() output -- there is no code
        path here that could introduce an unobserved campaign."""
        contrast_users = {1: ([100], [200])}
        seen_campaigns = []

        def score_fn(uid, campaign):
            seen_campaigns.append(campaign)
            return 1.0 if campaign == 100 else 0.0

        pairwise_accuracy(contrast_users, score_fn)
        assert set(seen_campaigns) == {100, 200}
