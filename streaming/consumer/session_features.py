"""Consume Kafka events and push session features to the Feast Redis online store."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streaming.consumer.aggregator import SessionStore

logger = logging.getLogger(__name__)

REPO_PATH = Path(__file__).resolve().parents[2] / "feast" / "feature_repo"


def _push(store, features: dict) -> None:
    from feast.data_source import PushMode

    df = pd.DataFrame([features])
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    store.push("session_features_push", df, to=PushMode.ONLINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kafka → Feast online session features")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"))
    parser.add_argument("--topic", default=os.getenv("KAFKA_TOPIC_EVENTS", "ecommerce.events"))
    parser.add_argument("--group-id", default="conversion-session-features")
    parser.add_argument("--repo", type=Path, default=REPO_PATH)
    parser.add_argument("--max-events", type=int, default=0, help="Stop after N events (0 = run forever)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from feast import FeatureStore

    store = FeatureStore(repo_path=str(args.repo))
    sessions = SessionStore()
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap.split(","),
        group_id=args.group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    logger.info("Consuming %s at %s", args.topic, args.bootstrap)
    n = 0
    try:
        for message in consumer:
            features = sessions.apply(message.value)
            _push(store, features)
            n += 1
            if n % 50 == 0:
                logger.info("Pushed %s session updates to Redis", n)
            if args.max_events and n >= args.max_events:
                break
    finally:
        consumer.close()
    logger.info("Done. pushed=%s active_sessions=%s", n, len(sessions.sessions))


if __name__ == "__main__":
    main()
