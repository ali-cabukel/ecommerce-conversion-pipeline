"""Publish one reconstructed session to Kafka and score it on /predict."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streaming.replay.replay_events import _events_for_session
from training.config import PROCESSED_PATH

logger = logging.getLogger(__name__)


def _predict(port: str, payload: dict, timeout: float = 15) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish one session to Kafka")
    parser.add_argument("--sessions", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"))
    parser.add_argument("--topic", default=os.getenv("KAFKA_TOPIC_EVENTS", "ecommerce.events"))
    parser.add_argument("--port", default=os.getenv("BENTOML_PORT", "3000"))
    parser.add_argument(
        "--wait",
        type=float,
        default=2.0,
        help="Seconds to wait for the Kafka consumer to push Redis before /predict",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Only publish Kafka events; do not call /predict",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.sessions.exists():
        raise FileNotFoundError(f"{args.sessions} missing. Run: uv run python training/dataset.py")
    sessions = pd.read_parquet(args.sessions)
    if args.session_id:
        match = sessions[sessions["session_id"] == args.session_id]
        if match.empty:
            raise SystemExit(f"session_id {args.session_id} not in {args.sessions}")
        row = match.iloc[0]
    else:
        row = sessions.sort_values("session_ts").iloc[-1]

    events = _events_for_session(row)
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8"),
    )
    for event in events:
        producer.send(args.topic, key=event["session_id"], value=event)
    producer.flush()

    session_id = row["session_id"]
    payload = {
        "session_id": session_id,
        "customer_id": row["customer_id"],
        "product_id": row["product_id"],
        "seller_id": row["seller_id"],
    }
    logger.info("Published %s events for %s", len(events), session_id)
    print(json.dumps(payload, indent=2, default=str))
    curl = (
        "curl -s "
        f"http://localhost:{args.port}/predict "
        "-H 'Content-Type: application/json' "
        f"-d '{json.dumps(payload, default=str)}'"
    )
    if args.no_predict:
        print()
        print(curl)
        return

    if args.wait > 0:
        logger.info("Waiting %.1fs for Feast push → Redis, then calling /predict", args.wait)
        time.sleep(args.wait)
    try:
        result = _predict(args.port, payload)
    except urllib.error.URLError as exc:
        logger.error(
            "/predict failed on port %s (%s). Serve BentoML on that port, or set BENTOML_PORT.",
            args.port,
            exc.reason,
        )
        print()
        print(curl)
        raise SystemExit(1) from exc
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        logger.error("/predict HTTP %s: %s", exc.code, body)
        print()
        print(curl)
        raise SystemExit(1) from exc

    print(json.dumps(result, indent=2, default=str))
    logger.info(
        "Scored on :%s  conversion_probability=%s  will_purchase=%s",
        args.port,
        result.get("conversion_probability"),
        result.get("will_purchase"),
    )


if __name__ == "__main__":
    main()
