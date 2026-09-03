select
    review_id::text as review_id,
    order_id::text as order_id,
    review_score::integer as review_score,
    review_creation_date::timestamptz as review_creation_date
from {{ source("raw", "olist_order_reviews") }}
where review_creation_date is not null
