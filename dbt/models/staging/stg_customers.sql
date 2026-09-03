select
    customer_id::text as customer_id,
    customer_unique_id::text as customer_unique_id
from {{ source("raw", "olist_customers") }}
