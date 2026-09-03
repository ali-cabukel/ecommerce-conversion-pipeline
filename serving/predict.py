"""Score a live session from Feast online features (Redis) + the trained model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from training.config import (
    FEAST_REPO_PATH,
    FEATURE_COLUMNS,
    FEATURE_REFS,
    MODEL_PATH,
)

FEATURE_SERVICE = "conversion_prediction"
SESSION_ENTITY_REFS = [
    "session_features:customer_id",
    "session_features:product_id",
    "session_features:seller_id",
]


def _scalar(value: Any) -> Any:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def unwrap_online(result: dict[str, Any]) -> dict[str, Any]:
    return {key: _scalar(value) for key, value in result.items()}


def _is_store_unavailable(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "connection refused" in message or "connecting to" in message or "error 61" in message


def load_bundle(model_path: Path = MODEL_PATH) -> tuple[Any, list[str]]:
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run: uv run python training/train.py"
        )
    bundle = joblib.load(model_path)
    columns = list(bundle.get("feature_columns") or FEATURE_COLUMNS)
    return bundle["model"], columns


def feature_frame(feature_map: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    values = []
    for col in columns:
        value = feature_map.get(col)
        values.append(np.nan if value is None else float(value))
    return pd.DataFrame([values], columns=columns)


class ConversionScorer:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        repo_path: Path = FEAST_REPO_PATH,
        store: Any | None = None,
    ) -> None:
        self.model, self.feature_columns = load_bundle(model_path)
        self.repo_path = Path(repo_path)
        self._store = store

    @property
    def store(self) -> Any:
        if self._store is None:
            from feast import FeatureStore

            self._store = FeatureStore(repo_path=str(self.repo_path))
        return self._store

    def _online(self, features: Any, entity_rows: list[dict[str, str]]) -> dict[str, Any]:
        try:
            raw = self.store.get_online_features(
                features=features,
                entity_rows=entity_rows,
            ).to_dict()
        except Exception as exc:
            if _is_store_unavailable(exc):
                raise RuntimeError(
                    "Feast online store unavailable (Redis). "
                    "Start Redis and materialize/push features, or pass `features`."
                ) from exc
            raise
        return unwrap_online(raw)

    def resolve_entities(
        self,
        session_id: str,
        customer_id: str | None = None,
        product_id: str | None = None,
        seller_id: str | None = None,
    ) -> tuple[str, str, str, str]:
        if customer_id and product_id and seller_id:
            return session_id, customer_id, product_id, seller_id
        row = self._online(
            features=SESSION_ENTITY_REFS,
            entity_rows=[{"session_id": session_id}],
        )
        customer_id = customer_id or row.get("customer_id")
        product_id = product_id or row.get("product_id")
        seller_id = seller_id or row.get("seller_id")
        missing = [
            name
            for name, value in (
                ("customer_id", customer_id),
                ("product_id", product_id),
                ("seller_id", seller_id),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing "
                + ", ".join(missing)
                + ". Pass them in the request or push the session to Redis first."
            )
        return session_id, str(customer_id), str(product_id), str(seller_id)

    def fetch_features(
        self,
        session_id: str,
        customer_id: str | None = None,
        product_id: str | None = None,
        seller_id: str | None = None,
    ) -> dict[str, Any]:
        session_id, customer_id, product_id, seller_id = self.resolve_entities(
            session_id, customer_id, product_id, seller_id
        )
        entity = {
            "session_id": session_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "seller_id": seller_id,
        }
        try:
            row = self._online(features=FEATURE_SERVICE, entity_rows=[entity])
        except RuntimeError:
            raise
        except Exception:
            row = self._online(features=FEATURE_REFS, entity_rows=[entity])
        row["session_id"] = session_id
        row["customer_id"] = customer_id
        row["product_id"] = product_id
        row["seller_id"] = seller_id
        return row

    def predict(
        self,
        session_id: str,
        customer_id: str | None = None,
        product_id: str | None = None,
        seller_id: str | None = None,
        features: dict[str, Any] | None = None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        if features:
            feature_map = dict(features)
            source = "override"
        else:
            feature_map = self.fetch_features(
                session_id, customer_id, product_id, seller_id
            )
            source = "feast_online"
            customer_id = customer_id or feature_map.get("customer_id")
            product_id = product_id or feature_map.get("product_id")
            seller_id = seller_id or feature_map.get("seller_id")

        frame = feature_frame(feature_map, self.feature_columns)
        probability = float(self.model.predict_proba(frame)[0, 1])
        used = {
            col: (None if pd.isna(frame.iloc[0][col]) else float(frame.iloc[0][col]))
            for col in self.feature_columns
        }
        return {
            "conversion_probability": probability,
            "will_purchase": probability >= threshold,
            "session_id": session_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "seller_id": seller_id,
            "features": used,
            "source": source,
        }
