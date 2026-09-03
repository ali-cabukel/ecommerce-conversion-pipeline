"""Tests for session-table construction and baseline training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.config import FEATURE_COLUMNS, LABEL_COL
from training.dataset import (
    add_product_window_features,
    build_order_grain,
    build_sessions,
    synthesize_olist,
)
from training.train import time_split, train_model


def _sessions(n_orders: int = 400, seed: int = 0, abandon_ratio: float = 1.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tables = synthesize_olist(rng, n_orders=n_orders)
    return build_sessions(tables, rng, abandon_ratio=abandon_ratio)


def test_order_grain_has_no_future_user_history() -> None:
    rng = np.random.default_rng(1)
    grain = build_order_grain(synthesize_olist(rng, n_orders=500))
    firsts = grain.sort_values("order_purchase_timestamp").groupby("customer_unique_id").head(1)
    assert (firsts["user_total_orders"] == 0).all()
    assert firsts["user_avg_order_value"].isna().all()

    repeats = grain[grain["user_total_orders"] > 0]
    assert not repeats.empty
    assert (repeats["user_avg_order_value"] > 0).all()


def test_seller_score_is_point_in_time() -> None:
    tables = {
        "orders": pd.DataFrame(
            {
                "order_id": ["o1", "o2"],
                "customer_id": ["c1", "c2"],
                "order_status": ["delivered", "delivered"],
                "order_purchase_timestamp": pd.to_datetime(
                    ["2018-01-01", "2018-01-20"], utc=True
                ),
            }
        ),
        "items": pd.DataFrame(
            {
                "order_id": ["o1", "o2"],
                "product_id": ["p", "p"],
                "seller_id": ["s", "s"],
                "price": [10.0, 20.0],
            }
        ),
        "customers": pd.DataFrame(
            {
                "customer_id": ["c1", "c2"],
                "customer_unique_id": ["u1", "u2"],
            }
        ),
        "reviews": pd.DataFrame(
            {
                "order_id": ["o1", "o2"],
                "review_score": [2, 5],
                "review_creation_date": pd.to_datetime(
                    ["2018-01-10", "2018-01-25"], utc=True
                ),
            }
        ),
    }
    grain = build_order_grain(tables).sort_values("order_purchase_timestamp")
    assert pd.isna(grain.iloc[0]["seller_avg_review_score"])
    assert grain.iloc[1]["seller_avg_review_score"] == 2.0


def test_training_table_schema_and_both_classes() -> None:
    sessions = _sessions()
    for col in FEATURE_COLUMNS + ["session_id", "customer_id", "product_id", "session_ts", LABEL_COL]:
        assert col in sessions.columns
    assert sessions[LABEL_COL].isin([0, 1]).all()
    assert sessions[LABEL_COL].nunique() == 2
    assert sessions["session_id"].is_unique
    assert sessions["checkout_started"].isin([0, 1]).all()


def test_product_window_excludes_current_session() -> None:
    raw = pd.DataFrame(
        {
            "session_id": ["a", "b", "c"],
            "product_id": ["p1", "p1", "p1"],
            "session_ts": pd.to_datetime(
                ["2018-01-01", "2018-01-08", "2018-01-14"], utc=True
            ),
            LABEL_COL: [1, 0, 1],
            "session_page_views": [1, 1, 1],
        }
    )
    out = add_product_window_features(raw)
    assert list(out["product_view_count_7d"]) == [0, 1, 1]
    assert np.isnan(out.loc[0, "product_conversion_rate_7d"])
    assert out.loc[1, "product_conversion_rate_7d"] == 1.0
    assert out.loc[2, "product_conversion_rate_7d"] == 0.0


def test_time_split_is_chronological() -> None:
    sessions = _sessions(n_orders=250)
    train_df, test_df = time_split(sessions, test_fraction=0.2)
    assert train_df["session_ts"].max() <= test_df["session_ts"].min()
    assert len(train_df) + len(test_df) == len(sessions)


def test_baseline_model_beats_chance() -> None:
    sessions = _sessions(n_orders=800, abandon_ratio=2.0)
    _, metrics, _ = train_model(sessions, seed=0, compute_importance=False)
    assert metrics["test_roc_auc"] > 0.65
    assert metrics["n_train"] > metrics["n_test"]
