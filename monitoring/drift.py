"""Compare current session features to a training reference with Evidently (PSI)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.config import (
    DRIFT_METRICS_PATH,
    DRIFT_REFERENCE_PATH,
    DRIFT_REPORT_PATH,
    FEATURE_COLUMNS,
    LABEL_COL,
    PROCESSED_PATH,
)

logger = logging.getLogger(__name__)
DEFAULT_DRIFT_SHARE = 0.5
DEFAULT_METHOD = "psi"


def feature_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or FEATURE_COLUMNS
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Frame missing feature columns: {missing}")
    frame = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return frame.reset_index(drop=True)


def write_reference(
    sessions: pd.DataFrame,
    path: Path = DRIFT_REFERENCE_PATH,
    max_rows: int = 20_000,
    seed: int = 42,
) -> Path:
    frame = feature_frame(sessions)
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("Wrote drift reference (%s rows) to %s", len(frame), path)
    return path


def time_split(
    sessions: pd.DataFrame, current_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_col = "session_ts" if "session_ts" in sessions.columns else "event_timestamp"
    ordered = sessions.sort_values(time_col).reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - current_fraction))
    if cut <= 0 or cut >= len(ordered):
        raise ValueError("Not enough rows for a time-based drift split")
    return ordered.iloc[:cut], ordered.iloc[cut:]


def run_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    method: str = DEFAULT_METHOD,
    drift_share: float = DEFAULT_DRIFT_SHARE,
) -> Any:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    cols = columns or FEATURE_COLUMNS
    ref = feature_frame(reference, cols)
    cur = feature_frame(current, cols)
    if ref.empty or cur.empty:
        raise ValueError("Reference and current frames must be non-empty")
    report = Report(
        [DataDriftPreset(columns=cols, method=method, drift_share=drift_share)],
        include_tests=True,
    )
    return report.run(cur, ref)


def summarize_eval(evaluation: Any, drift_share: float = DEFAULT_DRIFT_SHARE) -> dict[str, Any]:
    payload = evaluation.dict() if hasattr(evaluation, "dict") else evaluation
    metrics = payload.get("metrics", [])
    tests = payload.get("tests", [])
    drifted_count = 0.0
    drifted_share = 0.0
    column_scores: dict[str, float] = {}
    for metric in metrics:
        name = str(metric.get("metric_id") or metric.get("metric_name") or "")
        config = metric.get("config") or {}
        value = metric.get("value")
        if "DriftedColumnsCount" in name or config.get("type", "").endswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drifted_count = float(value.get("count", 0.0))
                drifted_share = float(value.get("share", 0.0))
        elif "ValueDrift" in name or config.get("type", "").endswith("ValueDrift"):
            column = config.get("column")
            if column is not None and isinstance(value, (int, float)):
                column_scores[str(column)] = float(value)
    failed_tests = [
        {
            "name": test.get("name"),
            "status": test.get("status"),
            "description": test.get("description"),
        }
        for test in tests
        if str(test.get("status", "")).upper() == "FAIL"
    ]
    dataset_drift = drifted_share >= drift_share
    return {
        "method": DEFAULT_METHOD,
        "drift_share_threshold": drift_share,
        "drifted_columns": drifted_count,
        "drifted_share": drifted_share,
        "dataset_drift": dataset_drift,
        "column_scores": column_scores,
        "failed_tests": failed_tests,
        "n_failed_tests": len(failed_tests),
    }


def persist_report(
    evaluation: Any,
    summary: dict[str, Any],
    *,
    metrics_path: Path = DRIFT_METRICS_PATH,
    html_path: Path = DRIFT_REPORT_PATH,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {
        "n_reference": summary.get("n_reference"),
        "n_current": summary.get("n_current"),
        "source": summary.get("source"),
    }
    metrics_path.write_text(json.dumps({**extra, **summary}, indent=2))
    if hasattr(evaluation, "save_html"):
        evaluation.save_html(str(html_path))
    logger.info("Wrote %s and %s", metrics_path, html_path)


def _load_sessions(from_warehouse: bool, processed: Path, max_sessions: int | None) -> tuple[pd.DataFrame, str]:
    if from_warehouse:
        from training.train import load_training_table_from_feast

        sessions, meta = load_training_table_from_feast(max_sessions=max_sessions)
        return sessions, str(meta.get("source", "feast_postgres"))
    if not processed.exists():
        raise FileNotFoundError(f"{processed} missing. Pass --from-warehouse or build sessions.")
    return pd.read_parquet(processed), "parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect feature drift vs the training reference")
    parser.add_argument("--from-warehouse", action="store_true", help="Load sessions from Feast/Postgres")
    parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--reference", type=Path, default=DRIFT_REFERENCE_PATH)
    parser.add_argument("--current", type=Path, default=None, help="Optional current-features parquet")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--current-fraction", type=float, default=0.2)
    parser.add_argument("--time-split", action="store_true", help="Ignore saved reference; split one table by time")
    parser.add_argument("--method", default=os.getenv("CONVERSION_DRIFT_METHOD", DEFAULT_METHOD))
    parser.add_argument(
        "--drift-share",
        type=float,
        default=float(os.getenv("CONVERSION_MAX_DRIFT_SHARE", str(DEFAULT_DRIFT_SHARE))),
    )
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--metrics-path", type=Path, default=DRIFT_METRICS_PATH)
    parser.add_argument("--html-path", type=Path, default=DRIFT_REPORT_PATH)
    args = parser.parse_args()
    if os.getenv("CONVERSION_FAIL_ON_DRIFT", "").lower() in {"1", "true", "yes"}:
        args.fail_on_drift = True

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.current is not None:
        if not args.reference.exists():
            raise FileNotFoundError(f"Reference snapshot missing: {args.reference}")
        reference = pd.read_parquet(args.reference)
        current = pd.read_parquet(args.current)
        source = "parquet_pair"
    elif args.time_split or not args.reference.exists():
        if not args.reference.exists() and not args.time_split:
            logger.info("No drift reference at %s; using a time split", args.reference)
        sessions, source = _load_sessions(args.from_warehouse, args.processed, args.max_sessions)
        reference, current = time_split(sessions, current_fraction=args.current_fraction)
        source = f"{source}_time_split"
    else:
        reference = pd.read_parquet(args.reference)
        sessions, source = _load_sessions(args.from_warehouse, args.processed, args.max_sessions)
        current = sessions
        source = f"{source}_vs_reference"

    evaluation = run_drift(
        reference,
        current,
        method=args.method,
        drift_share=args.drift_share,
    )
    summary = summarize_eval(evaluation, drift_share=args.drift_share)
    summary.update(
        {
            "method": args.method,
            "n_reference": int(len(reference)),
            "n_current": int(len(current)),
            "source": source,
            "label": LABEL_COL,
        }
    )
    persist_report(evaluation, summary, metrics_path=args.metrics_path, html_path=args.html_path)
    logger.info(
        "dataset_drift=%s  drifted_share=%.3f  threshold=%.3f  drifted_columns=%.0f",
        summary["dataset_drift"],
        summary["drifted_share"],
        args.drift_share,
        summary["drifted_columns"],
    )
    print(json.dumps({k: summary[k] for k in ("dataset_drift", "drifted_share", "drifted_columns", "n_reference", "n_current")}, indent=2))
    if args.fail_on_drift and summary["dataset_drift"]:
        raise SystemExit(
            f"Dataset drift: {summary['drifted_share']:.3f} of columns drifted "
            f"(threshold {args.drift_share:.2f})"
        )


if __name__ == "__main__":
    main()
