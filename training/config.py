"""Shared paths, feature names, and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "sessions.parquet"
PROCESSED_META_PATH = PROCESSED_DIR / "sessions_meta.json"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "conversion_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

EXPERIMENT_NAME = "conversion-prediction"
LABEL_COL = "purchased_within_session"
RANDOM_SEED = 42
TEST_FRACTION = 0.2

FEATURE_COLUMNS = [
    "user_total_orders",
    "user_avg_order_value",
    "product_conversion_rate_7d",
    "product_view_count_7d",
    "seller_avg_review_score",
    "session_page_views",
    "session_cart_value",
    "minutes_since_last_event",
    "checkout_started",
]

ID_COLUMNS = [
    "session_id",
    "customer_id",
    "product_id",
    "seller_id",
    "session_ts",
]

OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
}

PURCHASE_STATUSES = frozenset(
    {"delivered", "shipped", "invoiced", "processing", "approved"}
)
