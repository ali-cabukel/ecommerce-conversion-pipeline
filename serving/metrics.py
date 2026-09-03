"""Prometheus metrics for /predict request and response traffic."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICT_REQUESTS = Counter(
    "conversion_predict_requests_total",
    "Conversion /predict requests by outcome",
    ["status", "source", "error"],
)
PREDICT_IN_FLIGHT = Gauge(
    "conversion_predict_in_flight",
    "In-flight /predict requests",
)
PREDICT_LATENCY = Histogram(
    "conversion_predict_latency_seconds",
    "Conversion /predict latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREDICT_PROBABILITY = Histogram(
    "conversion_predict_probability",
    "Predicted conversion_probability from successful responses",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
PREDICT_WILL_PURCHASE = Counter(
    "conversion_predict_will_purchase_total",
    "Predicted will_purchase class from successful responses",
    ["will_purchase"],
)


def observe_request(
    *,
    status: str,
    source: str,
    error: str,
    latency_seconds: float,
    probability: float | None = None,
    will_purchase: bool | None = None,
) -> None:
    PREDICT_REQUESTS.labels(status=status, source=source, error=error).inc()
    PREDICT_LATENCY.observe(latency_seconds)
    if probability is not None:
        PREDICT_PROBABILITY.observe(max(0.0, min(float(probability), 1.0)))
    if will_purchase is not None:
        PREDICT_WILL_PURCHASE.labels(will_purchase=str(bool(will_purchase)).lower()).inc()
