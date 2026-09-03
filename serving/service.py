"""BentoML service: Feast online features → conversion probability."""

from __future__ import annotations

import time
from typing import Any

import bentoml
from pydantic import BaseModel

from serving.metrics import PREDICT_IN_FLIGHT, observe_request
from serving.predict import ConversionScorer


class PredictResponse(BaseModel):
    conversion_probability: float
    will_purchase: bool
    session_id: str
    customer_id: str | None = None
    product_id: str | None = None
    seller_id: str | None = None
    features: dict[str, float | None]
    source: str


@bentoml.service(name="conversion_prediction", traffic={"timeout": 15})
class ConversionService:
    def __init__(self) -> None:
        self.scorer = ConversionScorer()

    @bentoml.api
    def predict(
        self,
        session_id: str,
        customer_id: str | None = None,
        product_id: str | None = None,
        seller_id: str | None = None,
        features: dict[str, Any] | None = None,
    ) -> PredictResponse:
        started = time.perf_counter()
        source = "override" if features else "feast_online"
        PREDICT_IN_FLIGHT.inc()
        try:
            result = self.scorer.predict(
                session_id=session_id,
                customer_id=customer_id,
                product_id=product_id,
                seller_id=seller_id,
                features=features,
            )
        except FileNotFoundError as exc:
            observe_request(
                status="error",
                source=source,
                error="not_found",
                latency_seconds=time.perf_counter() - started,
            )
            raise bentoml.exceptions.NotFound(str(exc)) from exc
        except ValueError as exc:
            observe_request(
                status="error",
                source=source,
                error="invalid",
                latency_seconds=time.perf_counter() - started,
            )
            raise bentoml.exceptions.InvalidArgument(str(exc)) from exc
        except RuntimeError as exc:
            observe_request(
                status="error",
                source=source,
                error="store",
                latency_seconds=time.perf_counter() - started,
            )
            raise bentoml.exceptions.BentoMLException(str(exc)) from exc
        except Exception:
            observe_request(
                status="error",
                source=source,
                error="other",
                latency_seconds=time.perf_counter() - started,
            )
            raise
        finally:
            PREDICT_IN_FLIGHT.dec()

        observe_request(
            status="ok",
            source=result.get("source", source),
            error="none",
            latency_seconds=time.perf_counter() - started,
            probability=result["conversion_probability"],
            will_purchase=result["will_purchase"],
        )
        return PredictResponse.model_validate(result)
