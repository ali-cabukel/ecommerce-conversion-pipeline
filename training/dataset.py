"""Build a session-level conversion training table from Olist orders.

Olist is order-level, not clickstream. Each completed order becomes a
converting session, and abandoned sessions are reconstructed from the same
customers, products, and prices so the model sees both classes. User, seller,
and 7-day product features are point-in-time (no future leakage).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.config import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LABEL_COL,
    OLIST_FILES,
    PROCESSED_PATH,
    PURCHASE_STATUSES,
    RANDOM_SEED,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

CONVERT_STAGE_P = np.array([0.20, 0.30, 0.50])
ABANDON_STAGE_P = np.array([0.55, 0.30, 0.15])
STAGES = np.array(["browse", "cart", "checkout"])


def olist_available(raw_dir: Path = RAW_DIR) -> bool:
    return all((raw_dir / name).exists() for name in OLIST_FILES.values())


def load_olist(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    orders = pd.read_csv(raw_dir / OLIST_FILES["orders"])
    items = pd.read_csv(raw_dir / OLIST_FILES["items"])
    customers = pd.read_csv(raw_dir / OLIST_FILES["customers"])
    reviews = pd.read_csv(raw_dir / OLIST_FILES["reviews"])

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], utc=True
    )
    reviews["review_creation_date"] = pd.to_datetime(
        reviews["review_creation_date"], utc=True
    )
    return {
        "orders": orders,
        "items": items,
        "customers": customers,
        "reviews": reviews,
    }


def synthesize_olist(rng: np.random.Generator, n_orders: int = 8_000) -> dict[str, pd.DataFrame]:
    """Generate Olist-shaped tables when the Kaggle dump is not present."""
    n_customers = min(3_000, n_orders)
    n_products = min(1_200, n_orders)
    n_sellers = min(200, n_orders)

    unique_ids = np.array([f"cuniq_{i:05d}" for i in range(n_customers)])
    product_ids = np.array([f"prod_{i:05d}" for i in range(n_products)])
    seller_ids = np.array([f"seller_{i:04d}" for i in range(n_sellers)])

    start = np.datetime64("2017-01-01T00:00:00")
    span_seconds = int((np.datetime64("2018-08-29") - np.datetime64("2017-01-01")) / np.timedelta64(1, "s"))
    purchase_offsets = rng.integers(0, span_seconds, size=n_orders)
    purchase_ts = pd.to_datetime(start) + pd.to_timedelta(purchase_offsets, unit="s")
    purchase_ts = pd.DatetimeIndex(purchase_ts).tz_localize("UTC")

    customer_unique = rng.choice(unique_ids, size=n_orders, replace=True)
    customer_ids = np.array([f"cust_{i:06d}" for i in range(n_orders)])
    order_ids = np.array([f"order_{i:06d}" for i in range(n_orders)])
    statuses = rng.choice(
        np.array(["delivered", "shipped", "canceled"]),
        size=n_orders,
        p=np.array([0.90, 0.06, 0.04]),
    )

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_status": statuses,
            "order_purchase_timestamp": purchase_ts,
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_unique_id": customer_unique,
        }
    )

    items_per_order = rng.choice([1, 2, 3], size=n_orders, p=[0.85, 0.12, 0.03])
    item_order_ids = np.repeat(order_ids, items_per_order)
    n_items = len(item_order_ids)
    prices = np.round(np.clip(rng.lognormal(mean=4.2, sigma=0.7, size=n_items), 5.0, 2_000.0), 2)
    items = pd.DataFrame(
        {
            "order_id": item_order_ids,
            "order_item_id": np.concatenate([np.arange(1, k + 1) for k in items_per_order]),
            "product_id": rng.choice(product_ids, size=n_items, replace=True),
            "seller_id": rng.choice(seller_ids, size=n_items, replace=True),
            "price": prices,
            "freight_value": np.round(rng.uniform(5.0, 40.0, size=n_items), 2),
        }
    )

    reviewed = (rng.random(n_orders) < 0.88) & (statuses != "canceled")
    review_orders = order_ids[reviewed]
    n_reviews = len(review_orders)
    reviews = pd.DataFrame(
        {
            "review_id": np.array([f"rev_{i:06d}" for i in range(n_reviews)]),
            "order_id": review_orders,
            "review_score": rng.choice(
                np.array([1, 2, 3, 4, 5]),
                size=n_reviews,
                p=np.array([0.04, 0.03, 0.08, 0.19, 0.66]),
            ),
            "review_creation_date": pd.to_datetime(purchase_ts[reviewed])
            + pd.to_timedelta(rng.integers(2, 20, size=n_reviews), unit="D"),
        }
    )
    return {"orders": orders, "items": items, "customers": customers, "reviews": reviews}


def build_order_grain(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per purchased order with point-in-time user and seller features."""
    orders = tables["orders"]
    items = tables["items"]
    customers = tables["customers"]
    reviews = tables["reviews"]

    purchased = orders.loc[
        orders["order_status"].isin(PURCHASE_STATUSES)
        & orders["order_purchase_timestamp"].notna()
    ].copy()

    item_agg = (
        items.groupby("order_id", as_index=False)
        .agg(
            order_value=("price", "sum"),
            product_id=("product_id", "first"),
            seller_id=("seller_id", "first"),
        )
    )
    grain = purchased.merge(item_agg, on="order_id", how="inner")
    grain = grain.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )
    grain["customer_unique_id"] = grain["customer_unique_id"].fillna(grain["customer_id"])
    grain = grain.sort_values(["customer_unique_id", "order_purchase_timestamp"])

    grain["user_total_orders"] = grain.groupby("customer_unique_id").cumcount()
    prior_value_sum = (
        grain.groupby("customer_unique_id")["order_value"].cumsum() - grain["order_value"]
    )
    grain["user_avg_order_value"] = prior_value_sum / grain["user_total_orders"].replace(0, np.nan)

    grain = _attach_seller_scores(grain, items, reviews)
    return grain.reset_index(drop=True)


def _attach_seller_scores(
    grain: pd.DataFrame,
    items: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    seller_of_order = items[["order_id", "seller_id"]].drop_duplicates()
    hist = reviews[["order_id", "review_score", "review_creation_date"]].merge(
        seller_of_order, on="order_id", how="inner"
    )
    hist = hist.dropna(subset=["review_creation_date", "seller_id"]).sort_values(
        ["seller_id", "review_creation_date"]
    )
    hist["seller_avg_review_score"] = hist.groupby("seller_id")["review_score"].transform(
        lambda s: s.expanding().mean()
    )

    left = grain.sort_values("order_purchase_timestamp").rename(
        columns={"order_purchase_timestamp": "as_of"}
    )
    right = hist.sort_values("review_creation_date").rename(
        columns={"review_creation_date": "as_of"}
    )
    left["as_of"] = pd.to_datetime(left["as_of"], utc=True)
    right["as_of"] = pd.to_datetime(right["as_of"], utc=True)
    merged = pd.merge_asof(
        left,
        right[["seller_id", "as_of", "seller_avg_review_score"]],
        by="seller_id",
        on="as_of",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged.rename(columns={"as_of": "order_purchase_timestamp"})


def _funnel_features(
    rng: np.random.Generator, order_values: np.ndarray, convert: bool
) -> pd.DataFrame:
    n = len(order_values)
    probs = CONVERT_STAGE_P if convert else ABANDON_STAGE_P
    stages = rng.choice(STAGES, size=n, p=probs)

    browse = stages == "browse"
    cart = stages == "cart"
    checkout = stages == "checkout"

    page_views = np.empty(n, dtype=np.int32)
    page_views[browse] = rng.poisson(2, browse.sum()) + 1
    page_views[cart] = rng.poisson(4, cart.sum()) + 2
    page_views[checkout] = rng.poisson(6, checkout.sum()) + 3

    minutes = np.empty(n, dtype=np.float64)
    minutes[browse] = rng.exponential(6.0, browse.sum())
    minutes[cart] = rng.exponential(4.0, cart.sum())
    minutes[checkout] = rng.exponential(1.5, checkout.sum())
    minutes = np.clip(minutes, 0.05, 45.0)

    cart_value = np.where(browse, 0.0, order_values).astype(np.float64)
    return pd.DataFrame(
        {
            "session_page_views": page_views,
            "session_cart_value": np.round(cart_value, 2),
            "minutes_since_last_event": np.round(minutes, 3),
            "checkout_started": checkout.astype(np.int8),
        }
    )


def _sessions_from_orders(
    grain: pd.DataFrame,
    rng: np.random.Generator,
    abandon_ratio: float,
) -> pd.DataFrame:
    n = len(grain)
    converting = pd.DataFrame(
        {
            "session_id": np.array([f"s-buy-{oid}" for oid in grain["order_id"]]),
            "customer_id": grain["customer_unique_id"].to_numpy(),
            "product_id": grain["product_id"].to_numpy(),
            "seller_id": grain["seller_id"].to_numpy(),
            "session_ts": grain["order_purchase_timestamp"].to_numpy(),
            "user_total_orders": grain["user_total_orders"].to_numpy(),
            "user_avg_order_value": grain["user_avg_order_value"].to_numpy(),
            "seller_avg_review_score": grain["seller_avg_review_score"].to_numpy(),
            LABEL_COL: np.ones(n, dtype=np.int8),
        }
    )
    converting = pd.concat(
        [converting, _funnel_features(rng, grain["order_value"].to_numpy(), convert=True)],
        axis=1,
    )

    n_abandon = int(round(n * abandon_ratio))
    repeat_penalty = 1.0 / (1.0 + grain["user_total_orders"].to_numpy(dtype=np.float64))
    score = grain["seller_avg_review_score"].to_numpy(dtype=np.float64)
    score = np.where(np.isnan(score), 4.0, score)
    quality_penalty = np.clip((6.0 - score) / 5.0, 0.05, 1.0)
    weights = repeat_penalty * quality_penalty
    weights = weights / weights.sum()
    idx = rng.choice(n, size=n_abandon, replace=True, p=weights)
    sampled = grain.iloc[idx].reset_index(drop=True)

    lag_hours = rng.integers(1, 72, size=n_abandon)
    session_ts = sampled["order_purchase_timestamp"] - pd.to_timedelta(lag_hours, unit="h")

    abandoned = pd.DataFrame(
        {
            "session_id": np.array([f"s-drop-{i:07d}" for i in range(n_abandon)]),
            "customer_id": sampled["customer_unique_id"].to_numpy(),
            "product_id": sampled["product_id"].to_numpy(),
            "seller_id": sampled["seller_id"].to_numpy(),
            "session_ts": session_ts.to_numpy(),
            "user_total_orders": sampled["user_total_orders"].to_numpy(),
            "user_avg_order_value": sampled["user_avg_order_value"].to_numpy(),
            "seller_avg_review_score": sampled["seller_avg_review_score"].to_numpy(),
            LABEL_COL: np.zeros(n_abandon, dtype=np.int8),
        }
    )
    abandoned = pd.concat(
        [abandoned, _funnel_features(rng, sampled["order_value"].to_numpy(), convert=False)],
        axis=1,
    )
    sessions = pd.concat([converting, abandoned], ignore_index=True)
    sessions["session_ts"] = pd.to_datetime(sessions["session_ts"], utc=True)
    return sessions


def add_product_window_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """7-day product view count and conversion rate, excluding the current session."""
    sessions = sessions.copy()
    sessions["session_ts"] = pd.to_datetime(sessions["session_ts"], utc=True)
    sessions = sessions.sort_values(["product_id", "session_ts"]).reset_index(drop=True)
    ts = sessions["session_ts"].astype("int64").to_numpy()
    y = sessions[LABEL_COL].to_numpy(dtype=np.float64)
    n = len(sessions)
    view_count = np.zeros(n, dtype=np.int32)
    purchases_7d = np.zeros(n, dtype=np.float64)
    window_ns = np.int64(7 * 24 * 60 * 60 * 1_000_000_000)

    for idx in sessions.groupby("product_id", sort=False).indices.values():
        idx = np.asarray(idx)
        t = ts[idx]
        left = np.searchsorted(t, t - window_ns, side="left")
        rel = np.arange(len(idx))
        view_count[idx] = rel - left
        csum = np.concatenate([[0.0], np.cumsum(y[idx])])
        purchases_7d[idx] = csum[rel] - csum[left]

    sessions["product_view_count_7d"] = view_count
    sessions["product_conversion_rate_7d"] = np.divide(
        purchases_7d,
        view_count,
        out=np.full(n, np.nan),
        where=view_count > 0,
    )
    return sessions


def build_sessions(
    tables: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    max_orders: int | None = None,
    abandon_ratio: float = 2.0,
) -> pd.DataFrame:
    grain = build_order_grain(tables)
    if max_orders is not None and len(grain) > max_orders:
        grain = grain.sort_values("order_purchase_timestamp").tail(max_orders).reset_index(drop=True)
        grain["user_total_orders"] = grain.groupby("customer_unique_id").cumcount()
        prior_value_sum = (
            grain.groupby("customer_unique_id")["order_value"].cumsum() - grain["order_value"]
        )
        grain["user_avg_order_value"] = prior_value_sum / grain["user_total_orders"].replace(
            0, np.nan
        )

    sessions = _sessions_from_orders(grain, rng, abandon_ratio=abandon_ratio)
    sessions = add_product_window_features(sessions)
    columns = ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COL]
    return sessions[columns].sort_values("session_ts").reset_index(drop=True)


def build_training_table(
    raw_dir: Path = RAW_DIR,
    output_path: Path = PROCESSED_PATH,
    seed: int = RANDOM_SEED,
    max_orders: int | None = None,
    abandon_ratio: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    if olist_available(raw_dir):
        source = "olist"
        logger.info("Loading Olist CSVs from %s", raw_dir)
        tables = load_olist(raw_dir)
    else:
        source = "synthetic"
        n_orders = max_orders or 8_000
        logger.warning(
            "Olist CSVs not found in %s — synthesizing %s orders. "
            "Download the real dump with: kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip",
            raw_dir,
            n_orders,
        )
        tables = synthesize_olist(rng, n_orders=n_orders)

    sessions = build_sessions(tables, rng, max_orders=max_orders, abandon_ratio=abandon_ratio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(output_path, index=False)

    meta = {
        "source": source,
        "n_sessions": int(len(sessions)),
        "n_converters": int(sessions[LABEL_COL].sum()),
        "conversion_rate": float(sessions[LABEL_COL].mean()),
        "feature_columns": FEATURE_COLUMNS,
        "label": LABEL_COL,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "path": str(output_path),
    }
    meta_path = output_path.with_name("sessions_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(
        "Wrote %s sessions (conversion_rate=%.3f, source=%s) to %s",
        meta["n_sessions"],
        meta["conversion_rate"],
        source,
        output_path,
    )
    return sessions, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the conversion training table")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-orders", type=int, default=None)
    parser.add_argument("--abandon-ratio", type=float, default=2.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_training_table(
        raw_dir=args.raw_dir,
        output_path=args.output,
        seed=args.seed,
        max_orders=args.max_orders,
        abandon_ratio=args.abandon_ratio,
    )


if __name__ == "__main__":
    main()
