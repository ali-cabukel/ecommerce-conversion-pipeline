-- Point-in-time user history from Olist orders (excludes the current order).
with order_value as (
    select
        order_id,
        sum(price) as order_value
    from {{ ref("stg_order_items") }}
    group by 1
),

purchased as (
    select
        customers.customer_unique_id as customer_id,
        orders.order_purchase_timestamp as event_timestamp,
        order_value.order_value
    from {{ ref("stg_orders") }} as orders
    inner join {{ ref("stg_customers") }} as customers
        on orders.customer_id = customers.customer_id
    inner join order_value
        on orders.order_id = order_value.order_id
    where orders.order_status in (
        'delivered', 'shipped', 'invoiced', 'processing', 'approved'
    )
)

select
    customer_id,
    event_timestamp,
    (
        row_number() over (
            partition by customer_id
            order by event_timestamp
        ) - 1
    )::integer as user_total_orders,
    avg(order_value) over (
        partition by customer_id
        order by event_timestamp
        rows between unbounded preceding and 1 preceding
    )::double precision as user_avg_order_value
from purchased
