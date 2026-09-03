"""In-memory session feature aggregation for the Kafka → Redis path."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


def parse_event_ts(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("event_ts must be timezone-aware")
    return ts


@dataclass
class SessionState:
    session_id: str
    customer_id: str
    product_id: str
    seller_id: str
    page_views: int = 0
    cart_value: float = 0.0
    checkout_started: int = 0
    last_event_ts: datetime | None = None

    def apply(self, event: dict) -> dict:
        ts = parse_event_ts(event["event_ts"])
        minutes = (
            0.0
            if self.last_event_ts is None
            else max((ts - self.last_event_ts).total_seconds() / 60.0, 0.0)
        )
        event_type = event["event_type"]
        price = float(event.get("price") or 0.0)

        if event_type == "page_view":
            self.page_views += 1
        elif event_type == "add_to_cart":
            self.page_views = max(self.page_views, 1)
            self.cart_value += price
        elif event_type == "checkout_start":
            self.checkout_started = 1
            if self.cart_value <= 0:
                self.cart_value = price
        elif event_type == "purchase":
            self.checkout_started = 1
            if self.cart_value <= 0:
                self.cart_value = price
        else:
            raise ValueError(f"Unknown event_type: {event_type}")

        self.last_event_ts = ts
        return {
            "session_id": self.session_id,
            "customer_id": self.customer_id,
            "product_id": self.product_id,
            "seller_id": self.seller_id,
            "event_timestamp": ts,
            "session_page_views": int(self.page_views),
            "session_cart_value": float(self.cart_value),
            "minutes_since_last_event": float(minutes),
            "checkout_started": int(self.checkout_started),
        }


@dataclass
class SessionStore:
    sessions: dict[str, SessionState] = field(default_factory=dict)

    def apply(self, event: dict) -> dict:
        session_id = event["session_id"]
        state = self.sessions.get(session_id)
        if state is None:
            state = SessionState(
                session_id=session_id,
                customer_id=event["customer_id"],
                product_id=event["product_id"],
                seller_id=event.get("seller_id") or "",
            )
            self.sessions[session_id] = state
        return state.apply(event)
