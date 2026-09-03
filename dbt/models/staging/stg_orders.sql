select
    order_id::text as order_id,
    customer_id::text as customer_id,
    order_status::text as order_status,
    order_purchase_timestamp::timestamptz as order_purchase_timestamp
from {{ source("raw", "olist_orders") }}
where order_purchase_timestamp is not null
