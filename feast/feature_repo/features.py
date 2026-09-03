"""Feast feature views: Postgres batch marts + Kafka-pushed session features."""

from datetime import timedelta

from feast.types import Float64, Int64, String
from feast.value_type import ValueType

from feast import Entity, FeatureService, FeatureView, Field, PushSource

try:
    from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
        PostgreSQLSource,
    )
except ImportError:  # pragma: no cover - import path varies by Feast version
    from feast.infra.offline_stores.postgres_source import PostgreSQLSource

customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.STRING)
product = Entity(name="product", join_keys=["product_id"], value_type=ValueType.STRING)
seller = Entity(name="seller", join_keys=["seller_id"], value_type=ValueType.STRING)
session = Entity(name="session", join_keys=["session_id"], value_type=ValueType.STRING)

user_source = PostgreSQLSource(
    name="user_features_source",
    query="SELECT * FROM marts.fct_user_features",
    timestamp_field="event_timestamp",
)
product_source = PostgreSQLSource(
    name="product_features_source",
    query="SELECT * FROM marts.fct_product_features",
    timestamp_field="event_timestamp",
)
seller_source = PostgreSQLSource(
    name="seller_features_source",
    query="SELECT * FROM marts.fct_seller_features",
    timestamp_field="event_timestamp",
)
session_batch_source = PostgreSQLSource(
    name="session_features_batch_source",
    query="SELECT * FROM marts.fct_session_features",
    timestamp_field="event_timestamp",
)
session_push_source = PushSource(
    name="session_features_push",
    batch_source=session_batch_source,
)

user_features = FeatureView(
    name="user_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="user_total_orders", dtype=Int64),
        Field(name="user_avg_order_value", dtype=Float64),
    ],
    source=user_source,
    online=True,
)
product_features = FeatureView(
    name="product_features",
    entities=[product],
    ttl=timedelta(days=7),
    schema=[
        Field(name="product_view_count_7d", dtype=Int64),
        Field(name="product_conversion_rate_7d", dtype=Float64),
    ],
    source=product_source,
    online=True,
)
seller_features = FeatureView(
    name="seller_features",
    entities=[seller],
    ttl=timedelta(days=365),
    schema=[
        Field(name="seller_avg_review_score", dtype=Float64),
    ],
    source=seller_source,
    online=True,
)
session_features = FeatureView(
    name="session_features",
    entities=[session],
    ttl=timedelta(hours=6),
    schema=[
        Field(name="customer_id", dtype=String),
        Field(name="product_id", dtype=String),
        Field(name="seller_id", dtype=String),
        Field(name="session_page_views", dtype=Int64),
        Field(name="session_cart_value", dtype=Float64),
        Field(name="minutes_since_last_event", dtype=Float64),
        Field(name="checkout_started", dtype=Int64),
    ],
    source=session_push_source,
    online=True,
)

conversion_prediction = FeatureService(
    name="conversion_prediction",
    features=[user_features, product_features, seller_features, session_features],
)
