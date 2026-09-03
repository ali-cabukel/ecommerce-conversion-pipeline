"""Unit tests for conversion scoring (no Redis / BentoML required)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from serving.predict import ConversionScorer, feature_frame, unwrap_online
from training.config import FEATURE_COLUMNS


class _FakeOnline:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_dict(self) -> dict:
        return self.payload


class _FakeStore:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple] = []

    def get_online_features(self, features, entity_rows):
        self.calls.append((features, entity_rows))
        return _FakeOnline(self.payload)


def _bundle(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(FEATURE_COLUMNS)))
    y = (X[:, 0] + X[:, 5] > 0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0)
    model.fit(X, y)
    path = tmp_path / "conversion_model.joblib"
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, path)
    return path


def test_unwrap_online_takes_first_value() -> None:
    assert unwrap_online({"user_total_orders": [3], "empty": []}) == {
        "user_total_orders": 3,
        "empty": None,
    }


def test_feature_frame_uses_nan_for_missing() -> None:
    frame = feature_frame({"user_total_orders": 2}, ["user_total_orders", "session_page_views"])
    assert frame.iloc[0]["user_total_orders"] == 2.0
    assert np.isnan(frame.iloc[0]["session_page_views"])


def test_predict_from_feature_override(tmp_path: Path) -> None:
    scorer = ConversionScorer(model_path=_bundle(tmp_path), store=_FakeStore({}))
    features = {col: 1.0 for col in FEATURE_COLUMNS}
    result = scorer.predict(session_id="s1", customer_id="c1", features=features)
    assert result["session_id"] == "s1"
    assert result["source"] == "override"
    assert 0.0 <= result["conversion_probability"] <= 1.0
    assert result["will_purchase"] in {True, False}
    assert set(result["features"]) == set(FEATURE_COLUMNS)


def test_predict_from_feast_online(tmp_path: Path) -> None:
    payload = {col: [1.0] for col in FEATURE_COLUMNS}
    payload.update(
        {
            "customer_id": ["c1"],
            "product_id": ["p1"],
            "seller_id": ["s1"],
        }
    )
    store = _FakeStore(payload)
    scorer = ConversionScorer(model_path=_bundle(tmp_path), store=store)
    result = scorer.predict(session_id="sess-1")
    assert result["source"] == "feast_online"
    assert result["customer_id"] == "c1"
    assert result["product_id"] == "p1"
    assert store.calls


def test_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        ConversionScorer(model_path=tmp_path / "missing.joblib")


def test_redis_down_is_runtime_error(tmp_path: Path) -> None:
    class _Down:
        def get_online_features(self, features, entity_rows):
            raise ConnectionError("Error 61 connecting to localhost:6379. Connection refused.")

    scorer = ConversionScorer(model_path=_bundle(tmp_path), store=_Down())
    with pytest.raises(RuntimeError, match="Redis"):
        scorer.predict(session_id="s1", customer_id="c", product_id="p", seller_id="s")


def test_observe_request_records_success_and_error() -> None:
    from prometheus_client import generate_latest

    from serving.metrics import observe_request

    observe_request(
        status="ok",
        source="override",
        error="none",
        latency_seconds=0.02,
        probability=0.81,
        will_purchase=True,
    )
    observe_request(status="error", source="feast_online", error="store", latency_seconds=0.01)
    text = generate_latest().decode()
    assert "conversion_predict_requests_total" in text
    assert 'status="ok"' in text
    assert 'source="override"' in text
    assert "conversion_predict_latency_seconds_sum" in text
    assert "conversion_predict_probability_sum" in text
    assert 'will_purchase="true"' in text


def test_missing_entities_raise(tmp_path: Path) -> None:
    store = _FakeStore({"customer_id": [None], "product_id": [None], "seller_id": [None]})
    scorer = ConversionScorer(model_path=_bundle(tmp_path), store=store)
    with pytest.raises(ValueError, match="customer_id"):
        scorer.predict(session_id="sess-missing")
