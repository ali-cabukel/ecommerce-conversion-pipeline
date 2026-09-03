"""Data-drift helpers (Evidently PSI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.drift import feature_frame, run_drift, summarize_eval, time_split, write_reference
from training.config import FEATURE_COLUMNS


def _frame(mean: float, n: int = 180, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(mean, 0.4, n) for col in FEATURE_COLUMNS}
    data["checkout_started"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_feature_frame_coerces_numeric() -> None:
    raw = _frame(1.0)
    raw["user_total_orders"] = raw["user_total_orders"].astype(str)
    frame = feature_frame(raw)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame["user_total_orders"].dtype.kind == "f"


def test_time_split_uses_latest_as_current() -> None:
    sessions = _frame(1.0)
    sessions["session_ts"] = pd.date_range("2018-01-01", periods=len(sessions), freq="h", tz="UTC")
    reference, current = time_split(sessions, current_fraction=0.25)
    assert len(current) == 45
    assert current["session_ts"].min() >= reference["session_ts"].max()


def test_write_reference_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ref.parquet"
    write_reference(_frame(2.0, n=50), path=path, max_rows=20, seed=1)
    stored = pd.read_parquet(path)
    assert len(stored) == 20
    assert list(stored.columns) == FEATURE_COLUMNS


def test_summarize_detects_mean_shift() -> None:
    reference = _frame(1.0, seed=1)
    current = _frame(6.0, seed=2)
    summary = summarize_eval(run_drift(reference, current, drift_share=0.3), drift_share=0.3)
    assert summary["dataset_drift"] is True
    assert summary["drifted_share"] >= 0.3
    assert summary["column_scores"]


def test_summarize_stable_when_same_distribution() -> None:
    reference = _frame(1.0, seed=3)
    current = _frame(1.0, seed=4)
    summary = summarize_eval(run_drift(reference, current, drift_share=0.5), drift_share=0.5)
    assert summary["dataset_drift"] is False
    assert summary["drifted_share"] < 0.5
