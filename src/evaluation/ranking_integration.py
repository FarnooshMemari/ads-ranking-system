"""Restricted retrieval-candidate ranking check (docs/ranking_integration_design.md Part B).

Pure, model-agnostic logic for: building the "independently confirmed as
observed" exposure map from real test-period rows, intersecting retrieved
candidates against it (never treating an unobserved campaign as a negative),
finding users with a genuine click-vs-non-click contrast among their
verified candidates, and computing pairwise ranking accuracy over that
contrast set via an injected scoring function (decoupling this module from
any specific model or feature representation, so it is fully testable with
synthetic data and no LightGBM dependency).
"""

from collections import defaultdict
from itertools import product
from typing import Any, Callable, Dict, List, Set, Tuple

import pandas as pd


def build_exposure_maps(test_df: pd.DataFrame) -> Tuple[Dict[Any, Set[Any]], Dict[Tuple[Any, Any], int]]:
    """Build the "independently confirmed as observed" exposure map.

    Uses EVERY real test-period row — not just clicked ones — since exposure
    (was this campaign actually shown to this user), not click outcome, is
    what's being verified here.

    Args:
        test_df: Real test-period impression rows (``uid``, ``campaign``,
            ``click`` columns).

    Returns:
        ``(shown, click_lookup)`` — ``shown[uid]`` is the set of campaigns
        genuinely shown to that user in the test period; ``click_lookup[(uid,
        campaign)]`` is 1 if any of that pair's real rows was clicked, else 0.
    """
    shown: Dict[Any, Set[Any]] = defaultdict(set)
    click_lookup: Dict[Tuple[Any, Any], int] = {}
    for uid, campaign, click in zip(test_df["uid"], test_df["campaign"], test_df["click"]):
        shown[uid].add(campaign)
        key = (uid, campaign)
        click_lookup[key] = max(click_lookup.get(key, 0), int(click))
    return dict(shown), click_lookup


def verified_candidates(retrieved: List[Any], shown_for_user: Set[Any]) -> List[Any]:
    """Intersect a retrieved candidate list with a user's real exposure set.

    This is the single enforcement point for the project's core constraint:
    a candidate absent from ``shown_for_user`` is unknown, never a negative
    — it is simply dropped here, never scored as anything.

    Args:
        retrieved: Retrieval-stage output for one user.
        shown_for_user: That user's real, independently-confirmed exposure
            set (from ``build_exposure_maps``).

    Returns:
        The subset of ``retrieved`` that is independently confirmed as
        actually shown to this user (order preserved).
    """
    return [c for c in retrieved if c in shown_for_user]


def find_contrast_users(
    verified_per_user: Dict[Any, List[Any]], click_lookup: Dict[Tuple[Any, Any], int]
) -> Dict[Any, Tuple[List[Any], List[Any]]]:
    """Find users whose verified candidates include a genuine click/non-click contrast.

    A "contrast" (at least one clicked and at least one not-clicked verified
    candidate) is the minimum needed for a pairwise ranking comparison to
    exist at all — this is stricter than merely having 2+ verified
    candidates (which could all share the same outcome).

    Args:
        verified_per_user: user -> list of verified (independently observed)
            candidates (from ``verified_candidates``, one call per user).
        click_lookup: ``(uid, campaign) -> 0/1`` from ``build_exposure_maps``.

    Returns:
        ``{uid: (clicked_campaigns, not_clicked_campaigns)}`` for users with
        both lists non-empty.
    """
    contrast_users = {}
    for uid, campaigns in verified_per_user.items():
        clicked = [c for c in campaigns if click_lookup.get((uid, c), 0) == 1]
        not_clicked = [c for c in campaigns if click_lookup.get((uid, c), 0) == 0]
        if clicked and not_clicked:
            contrast_users[uid] = (clicked, not_clicked)
    return contrast_users


def pairwise_accuracy(
    contrast_users: Dict[Any, Tuple[List[Any], List[Any]]],
    score_fn: Callable[[Any, Any], Any],
) -> Tuple[float, int, int]:
    """Fraction of (clicked, not-clicked) verified pairs scored in the correct order.

    Args:
        contrast_users: Output of ``find_contrast_users``.
        score_fn: ``score_fn(uid, campaign) -> float | None`` — injected so
            this function has no dependency on any specific model or feature
            representation; a test can pass a trivial lookup table.
            Returning ``None`` signals "this pair can't be scored" (e.g. a
            missing feature row) — that pair is skipped, not counted as an
            error.

    Returns:
        ``(accuracy, n_pairs, n_users_used)`` — ``accuracy`` is ``nan`` if
        ``n_pairs == 0``. A pair counts as correct if the clicked
        candidate's score is strictly greater than the not-clicked
        candidate's; ties count as incorrect (conservative).
        ``n_users_used`` counts only users who contributed at least one
        actually-scored pair.
    """
    correct, total = 0, 0
    users_with_scored_pairs = set()
    for uid, (clicked, not_clicked) in contrast_users.items():
        for c, nc in product(clicked, not_clicked):
            cs, ncs = score_fn(uid, c), score_fn(uid, nc)
            if cs is None or ncs is None:
                continue
            total += 1
            users_with_scored_pairs.add(uid)
            if cs > ncs:
                correct += 1
    accuracy = correct / total if total else float("nan")
    return accuracy, total, len(users_with_scored_pairs)
