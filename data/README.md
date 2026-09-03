# Data

## Source

[Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) on Kaggle.

## Download

```bash
# Requires Kaggle API credentials at ~/.kaggle/kaggle.json
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

Expected files in `data/raw/`:

- `olist_customers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`

## Versioning

Raw data is tracked with DVC (not committed to git). After download:

```bash
dvc add data/raw
```

## Training table

`python training/dataset.py` reads the CSVs above and writes:

- `data/processed/sessions.parquet` — one row per reconstructed session
- `data/processed/sessions_meta.json` — row counts, conversion rate, source (`olist` or `synthetic`)

If the Olist dump is missing, the builder synthesizes Olist-shaped tables so training still runs.

**Label:** `purchased_within_session`

**Features:** `user_total_orders`, `user_avg_order_value`, `product_conversion_rate_7d`, `product_view_count_7d`, `seller_avg_review_score`, `session_page_views`, `session_cart_value`, `minutes_since_last_event`, `checkout_started`
