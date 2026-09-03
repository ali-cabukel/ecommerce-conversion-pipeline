-- 7-day product stats from reconstructed sessions (Olist has no native clickstream).
select
    product_id,
    session_ts as event_timestamp,
    product_view_count_7d,
    product_conversion_rate_7d
from {{ ref("stg_sessions") }}
