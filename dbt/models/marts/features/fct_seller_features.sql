-- Running seller review average using only reviews strictly before this timestamp.
with order_sellers as (
    select distinct
        order_id,
        seller_id
    from {{ ref("stg_order_items") }}
),

seller_reviews as (
    select
        order_sellers.seller_id,
        reviews.review_score,
        reviews.review_creation_date as event_timestamp
    from {{ ref("stg_reviews") }} as reviews
    inner join order_sellers
        on reviews.order_id = order_sellers.order_id
)

select
    seller_id,
    event_timestamp,
    avg(review_score) over (
        partition by seller_id
        order by event_timestamp
        rows between unbounded preceding and 1 preceding
    )::double precision as seller_avg_review_score
from seller_reviews
