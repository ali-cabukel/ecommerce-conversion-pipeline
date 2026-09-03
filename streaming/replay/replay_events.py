"""Replay reconstructed sessions as Kafka e-commerce events."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.config import PROCESSED_PATH
from training.dataset import build_training_table

logger = logging.getLogger(__name__)


def _events_for_session(row: pd.Series) -> list[dict]:
    ts = pd.Timestamp(row["session_ts"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ts = ts.to_pydatetime()
    base = {
        "session_id": row["session_id"],
        "customer_id": row["customer_id"],
        "product_id": row["product_id"],
        "seller_id": row["seller_id"],
        "price": float(row["session_cart_value"] or 0.0),
    }
    views = max(int(row["session_page_views"]), 1)
    events: list[dict] = []
    for i in range(views):
        events.append(
            {
                **base,
                "event_type": "page_view",
                "event_ts": (ts + timedelta(seconds=i)).isoformat(),
            }
        )
    cursor = views
    if float(row["session_cart_value"] or 0) > 0:
        events.append(
            {
                **base,
                "event_type": "add_to_cart",
                "event_ts": (ts + timedelta(seconds=cursor)).isoformat(),
            }
        )
        cursor += 1
    if int(row["checkout_started"]):
        events.append(
            {
                **base,
                "event_type": "checkout_start",
                "event_ts": (ts + timedelta(seconds=cursor)).isoformat(),
            }
        )
        cursor += 1
    if int(row.get("purchased_within_session", 0)):
        events.append(
            {
                **base,
                "event_type": "purchase",
                "event_ts": (ts + timedelta(seconds=cursor)).isoformat(),
            }
        )
    return events


def iter_events(sessions: pd.DataFrame):
    ordered = sessions.sort_values("session_ts")
    for _, row in ordered.iterrows():
        yield from _events_for_session(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay sessions onto Kafka")
    parser.add_argument("--sessions", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"))
    parser.add_argument("--topic", default=os.getenv("KAFKA_TOPIC_EVENTS", "ecommerce.events"))
    parser.add_argument("--max-sessions", type=int, default=500)
    parser.add_argument("--events-per-second", type=float, default=float(os.getenv("REPLAY_EVENTS_PER_SECOND", "20")))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.sessions.exists():
        build_training_table(output_path=args.sessions)
    sessions = pd.read_parquet(args.sessions)
    if args.max_sessions:
        sessions = sessions.sort_values("session_ts").tail(args.max_sessions)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8"),
    )
    delay = 0.0 if args.events_per_second <= 0 else 1.0 / args.events_per_second
    n = 0
    for event in iter_events(sessions):
        producer.send(args.topic, key=event["session_id"], value=event)
        n += 1
        if delay:
            time.sleep(delay)
    producer.flush()
    logger.info("Published %s events to %s", n, args.topic)


if __name__ == "__main__":
    main()
