select
    session_id::text as session_id,
    customer_id::text as customer_id,
    product_id::text as product_id,
    seller_id::text as seller_id,
    session_ts::timestamptz as session_ts,
    user_total_orders::integer as user_total_orders,
    user_avg_order_value::double precision as user_avg_order_value,
    product_conversion_rate_7d::double precision as product_conversion_rate_7d,
    product_view_count_7d::integer as product_view_count_7d,
    seller_avg_review_score::double precision as seller_avg_review_score,
    session_page_views::integer as session_page_views,
    session_cart_value::double precision as session_cart_value,
    minutes_since_last_event::double precision as minutes_since_last_event,
    checkout_started::integer as checkout_started,
    purchased_within_session::integer as purchased_within_session
from {{ source("raw", "sessions") }}
