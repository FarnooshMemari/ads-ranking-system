"""Point-in-time feature engineering for the Attribution dataset's CTR ranking stage.

Implements the feature set from docs/ranking_problem_design.md (Q5, row (a) — offline
scoring of an already-logged impression) minus `cost`/`cpo`. Every feature here is
either:

  - a **count-based, expanding, strictly-prior aggregate** (user/pair history) —
    computed once, over the user's *entire* real history (train+validation+test
    together), because the point-in-time cutoff for any row is that row's own
    timestamp, not the train/val/test split boundary. A test-period row's history
    features correctly include that same user's train-period activity — this is
    real prior information, not leakage. See
    ``build_ctr_features``'s docstring for exactly why this function must be
    called on the full dataset *before* splitting.
  - a **frozen, training-period-only campaign aggregate** — computed once from the
    training split alone and broadcast unchanged onto every row (train, validation,
    and test alike), preventing a campaign's future popularity from leaking into
    predictions for earlier rows (docs/attribution_modeling_design.md §6).
  - a **raw contextual field already on the impression row itself**
    (`cat1`–`cat9`, `time_since_last_click`) — these describe the impression being
    scored, not the future, so they carry no leakage risk for offline scoring of a
    logged impression (see docs/ranking_problem_design.md Q5 for the important
    caveat that this is *not* the same feature set available when scoring a
    not-yet-served retrieval candidate).

Excluded entirely, as inputs, per docs/ranking_problem_design.md Q4/Q8:
`conversion`, `conversion_timestamp`, `conversion_id`, `attribution`, `click_pos`,
`click_nb`, `cost`, `cpo` — all post-outcome or outcome-adjacent fields.
"""

from typing import List

import numpy as np
import pandas as pd

TARGET_COLUMN = "click"

CONTEXTUAL_FEATURE_COLUMNS = [f"cat{i}" for i in range(1, 10)] + ["time_since_last_click"]

USER_HISTORY_FEATURE_COLUMNS = [
    "user_impressions_before",
    "user_clicks_before",
    "user_click_rate_before",
    "user_distinct_campaigns_before",
]

PAIR_HISTORY_FEATURE_COLUMNS = [
    "pair_impressions_before",
    "pair_clicks_before",
]

CAMPAIGN_FEATURE_COLUMNS = [
    "campaign_impressions_train",
    "campaign_clicks_train",
    "campaign_ctr_train",
]

FEATURE_COLUMNS = (
    ["campaign"]
    + CONTEXTUAL_FEATURE_COLUMNS
    + USER_HISTORY_FEATURE_COLUMNS
    + PAIR_HISTORY_FEATURE_COLUMNS
    + CAMPAIGN_FEATURE_COLUMNS
)

CATEGORICAL_FEATURE_COLUMNS = ["campaign"] + [f"cat{i}" for i in range(1, 10)]

# Fields that must never appear in FEATURE_COLUMNS -- checked by a unit test
# (tests/test_ctr_features.py::test_no_post_outcome_columns_in_feature_columns)
# so this list stays authoritative even if features are added later.
EXCLUDED_POST_OUTCOME_COLUMNS = [
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click_pos",
    "click_nb",
    "cost",
    "cpo",
]

# Columns that are cumulative counts and are therefore NEVER missing (0 is the
# correct "no history yet" value, not a placeholder needing imputation).
_NEVER_MISSING_COLUMNS = [
    "user_impressions_before",
    "user_clicks_before",
    "user_distinct_campaigns_before",
    "pair_impressions_before",
    "pair_clicks_before",
]

# Columns that CAN be genuinely missing (undefined ratio, 0/0) and are left as
# NaN deliberately, for LightGBM's native missing-value handling -- see
# docs/ctr_model.md for the documented rationale.
_NATIVE_MISSING_COLUMNS = ["user_click_rate_before"]


def add_point_in_time_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add expanding, strictly-prior user-history features.

    Args:
        df: Must contain the user's *complete* relevant history (not one
            split in isolation) — see module docstring.

    Returns:
        ``df`` with ``user_impressions_before``, ``user_clicks_before``,
        ``user_click_rate_before`` (NaN if no prior impressions),
        ``user_distinct_campaigns_before`` added, original row order
        preserved.
    """
    original_index = df.index
    ordered = df.sort_values("timestamp", kind="mergesort")

    ordered["user_impressions_before"] = ordered.groupby("uid").cumcount()

    cum_clicks = ordered.groupby("uid")["click"].cumsum()
    ordered["user_clicks_before"] = cum_clicks - ordered["click"]

    with np.errstate(invalid="ignore", divide="ignore"):
        ordered["user_click_rate_before"] = np.where(
            ordered["user_impressions_before"] > 0,
            ordered["user_clicks_before"] / ordered["user_impressions_before"],
            np.nan,
        )

    # A row is the first time this (uid, campaign) pair has appeared, in
    # chronological order -- summing that flag cumulatively per user gives
    # "distinct campaigns seen so far including this one"; subtracting the
    # row's own flag gives "distinct campaigns seen strictly before this row".
    # Fully vectorized (no per-group Python-level apply) -- required for this
    # to run in reasonable time over 16M+ rows.
    ordered["_is_first_pair"] = (ordered.groupby(["uid", "campaign"]).cumcount() == 0).astype(int)
    cum_distinct = ordered.groupby("uid")["_is_first_pair"].cumsum()
    ordered["user_distinct_campaigns_before"] = cum_distinct - ordered["_is_first_pair"]
    ordered = ordered.drop(columns=["_is_first_pair"])

    return ordered.reindex(original_index)


def add_point_in_time_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add expanding, strictly-prior user-campaign PAIR history features.

    Args:
        df: Must contain the user's complete relevant history (see module
            docstring); does not need to already have the user-level
            features added.

    Returns:
        ``df`` with ``pair_impressions_before`` and ``pair_clicks_before``
        added, original row order preserved.
    """
    original_index = df.index
    ordered = df.sort_values("timestamp", kind="mergesort")

    ordered["pair_impressions_before"] = ordered.groupby(["uid", "campaign"]).cumcount()
    cum_pair_clicks = ordered.groupby(["uid", "campaign"])["click"].cumsum()
    ordered["pair_clicks_before"] = cum_pair_clicks - ordered["click"]

    return ordered.reindex(original_index)


def compute_campaign_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute frozen, training-period-only campaign aggregates.

    Args:
        train_df: Training-period rows only — see
            docs/attribution_modeling_design.md §6 on why this must never
            include validation/test rows.

    Returns:
        A DataFrame indexed by ``campaign`` with ``campaign_impressions_train``,
        ``campaign_clicks_train``, ``campaign_ctr_train``.
    """
    agg = train_df.groupby("campaign")["click"].agg(["count", "sum"])
    agg = agg.rename(columns={"count": "campaign_impressions_train", "sum": "campaign_clicks_train"})
    agg["campaign_ctr_train"] = agg["campaign_clicks_train"] / agg["campaign_impressions_train"]
    return agg


def attach_campaign_aggregates(
    df: pd.DataFrame, campaign_agg: pd.DataFrame, global_train_ctr: float
) -> pd.DataFrame:
    """Left-join frozen campaign aggregates onto every row.

    A campaign absent from ``campaign_agg`` (never observed in the training
    period at all) gets an explicit, documented fallback: zero prior volume
    and the global training CTR as its rate estimate — a meaningfully
    different situation from a user's missing history (see
    ``_NATIVE_MISSING_COLUMNS`` handling), so it is filled rather than left
    as NaN.

    Args:
        df: Rows to attach features to (any split).
        campaign_agg: Output of ``compute_campaign_aggregates``.
        global_train_ctr: Training-period global click rate (the baseline's
            own prediction — see ``src.models.ctr_model.GlobalCTRBaseline``),
            used as the fallback rate for a campaign with zero training data.

    Returns:
        ``df`` with the three campaign aggregate columns added.
    """
    merged = df.merge(campaign_agg, on="campaign", how="left")
    merged["campaign_impressions_train"] = merged["campaign_impressions_train"].fillna(0)
    merged["campaign_clicks_train"] = merged["campaign_clicks_train"].fillna(0)
    merged["campaign_ctr_train"] = merged["campaign_ctr_train"].fillna(global_train_ctr)
    merged.index = df.index
    return merged


def build_ctr_features(full_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full CTR feature matrix for ``full_df``.

    **Must be called on the complete dataset — train, validation, and test
    rows together — before splitting**, not once per split. A test-period
    row's user-history features must reflect that same user's real
    train-period activity (a genuine part of "what was knowable before this
    impression"); computing history features on an isolated test-only slice
    would incorrectly reset every user's history to zero at the split
    boundary, which is not what "point-in-time" means (see
    docs/ranking_problem_design.md Q5/Q6). Campaign aggregates, by contrast,
    *are* restricted to ``train_df`` only, to prevent a campaign's future
    popularity leaking backward.

    After this function returns, split the result with
    ``src.data.attribution.split_by_day`` as usual — the categorical dtype
    casting performed here (on the full frame) guarantees consistent
    category encodings across whatever splits are taken afterward.

    Args:
        full_df: The complete dataset (all splits).
        train_df: The training-period subset of ``full_df`` (e.g. from
            ``split_by_day``), used only for the frozen campaign aggregates.

    Returns:
        ``full_df`` with every column in ``FEATURE_COLUMNS`` added, plus the
        original columns preserved. ``campaign`` and `cat1`–`cat9`` are cast
        to pandas ``category`` dtype for LightGBM's native categorical
        handling.
    """
    global_train_ctr = float(train_df[TARGET_COLUMN].mean())
    campaign_agg = compute_campaign_aggregates(train_df)

    out = add_point_in_time_user_features(full_df)
    out = add_point_in_time_pair_features(out)
    out = attach_campaign_aggregates(out, campaign_agg, global_train_ctr)

    for col in CATEGORICAL_FEATURE_COLUMNS:
        out[col] = out[col].astype("category")

    return out


def assert_no_post_outcome_leakage(feature_columns: List[str]) -> None:
    """Raise if any excluded post-outcome column has snuck into a feature list.

    Args:
        feature_columns: Column names about to be used as model input.

    Raises:
        ValueError: If any column in ``EXCLUDED_POST_OUTCOME_COLUMNS`` is present.
    """
    leaked = set(feature_columns) & set(EXCLUDED_POST_OUTCOME_COLUMNS)
    if leaked:
        raise ValueError(f"Post-outcome columns must never be used as features: {sorted(leaked)}")
