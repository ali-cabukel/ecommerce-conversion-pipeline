from streaming.consumer.aggregator import SessionState, SessionStore, parse_event_ts


def test_session_aggregates_funnel_in_order() -> None:
    store = SessionStore()
    events = [
        {
            "session_id": "s1",
            "customer_id": "c1",
            "product_id": "p1",
            "seller_id": "sel1",
            "event_type": "page_view",
            "event_ts": "2018-01-01T10:00:00+00:00",
            "price": 0,
        },
        {
            "session_id": "s1",
            "customer_id": "c1",
            "product_id": "p1",
            "seller_id": "sel1",
            "event_type": "page_view",
            "event_ts": "2018-01-01T10:02:00+00:00",
            "price": 0,
        },
        {
            "session_id": "s1",
            "customer_id": "c1",
            "product_id": "p1",
            "seller_id": "sel1",
            "event_type": "add_to_cart",
            "event_ts": "2018-01-01T10:03:00+00:00",
            "price": 49.9,
        },
        {
            "session_id": "s1",
            "customer_id": "c1",
            "product_id": "p1",
            "seller_id": "sel1",
            "event_type": "checkout_start",
            "event_ts": "2018-01-01T10:04:00+00:00",
            "price": 49.9,
        },
    ]
    out = [store.apply(event) for event in events]
    assert out[-1]["session_page_views"] == 2
    assert out[-1]["session_cart_value"] == 49.9
    assert out[-1]["checkout_started"] == 1
    assert out[1]["minutes_since_last_event"] == 2.0


def test_parse_event_ts_requires_timezone() -> None:
    ts = parse_event_ts("2018-01-01T10:00:00Z")
    assert ts.tzinfo is not None
    state = SessionState("s", "c", "p", "sel")
    first = state.apply(
        {
            "event_type": "page_view",
            "event_ts": "2018-01-01T10:00:00+00:00",
            "price": 0,
        }
    )
    assert first["session_page_views"] == 1
    assert first["minutes_since_last_event"] == 0.0
