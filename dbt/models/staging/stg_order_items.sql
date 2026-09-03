select
    order_id::text as order_id,
    order_item_id::integer as order_item_id,
    product_id::text as product_id,
    seller_id::text as seller_id,
    price::double precision as price
from {{ source("raw", "olist_order_items") }}
