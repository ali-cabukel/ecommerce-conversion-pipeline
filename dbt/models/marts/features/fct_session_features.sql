-- Batch snapshot of in-session features. Live values are pushed from Kafka → Redis.
select
    session_id,
    customer_id,
    product_id,
    seller_id,
    session_ts as event_timestamp,
    session_page_views,
    session_cart_value,
    minutes_since_last_event,
    checkout_started
from {{ ref("stg_sessions") }}
