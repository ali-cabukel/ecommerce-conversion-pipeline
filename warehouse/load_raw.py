"""Load Olist CSVs and the session table into Postgres `raw` schema for dbt."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.config import PROCESSED_PATH, RAW_DIR
from training.dataset import build_training_table, olist_available

logger = logging.getLogger(__name__)

OLIST_TABLES = {
    "olist_orders": "olist_orders_dataset.csv",
    "olist_order_items": "olist_order_items_dataset.csv",
    "olist_customers": "olist_customers_dataset.csv",
    "olist_order_reviews": "olist_order_reviews_dataset.csv",
}


def warehouse_url() -> str:
    user = os.getenv("POSTGRES_USER", "conversion")
    password = os.getenv("POSTGRES_PASSWORD", "changethis")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "conversion")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _write_table(engine, df: pd.DataFrame, table: str) -> None:
    logger.info("Writing raw.%s (%s rows)", table, len(df))
    # dbt staging views depend on these tables, so never DROP them.
    if inspect(engine).has_table(table, schema="raw"):
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE raw.{table}"))
    else:
        df.head(0).to_sql(table, engine, schema="raw", if_exists="fail", index=False)
    df.to_sql(
        table,
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=2_000,
    )


def load_raw(
    raw_dir: Path = RAW_DIR,
    sessions_path: Path = PROCESSED_PATH,
    rebuild_sessions: bool = False,
) -> None:
    engine = create_engine(warehouse_url())
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS feast"))

    if not olist_available(raw_dir):
        raise FileNotFoundError(
            f"Olist CSVs not found in {raw_dir}. See data/README.md."
        )
    for table, filename in OLIST_TABLES.items():
        path = raw_dir / filename
        df = pd.read_csv(path)
        for col in df.columns:
            if "timestamp" in col or col.endswith("_date"):
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        _write_table(engine, df, table)

    if rebuild_sessions or not sessions_path.exists():
        logger.info("Building session table before warehouse load")
        build_training_table(raw_dir=raw_dir, output_path=sessions_path)

    sessions = pd.read_parquet(sessions_path)
    sessions["session_ts"] = pd.to_datetime(sessions["session_ts"], utc=True)
    _write_table(engine, sessions, "sessions")
    logger.info("Warehouse load complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load raw tables into Postgres")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--sessions", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--rebuild-sessions", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_raw(
        raw_dir=args.raw_dir,
        sessions_path=args.sessions,
        rebuild_sessions=args.rebuild_sessions,
    )


if __name__ == "__main__":
    main()
