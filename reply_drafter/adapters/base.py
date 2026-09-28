"""The adapter contract.

Tools never talk to a platform directly: they call a ``StoreAdapter``. Swap
the mock for Shopify, another platform, or an internal API by implementing
these five methods. Every method returns plain JSON-serialisable data.

``as_of`` makes replays honest: when an eval replays a conversation from
last March, the adapter must hide orders, shipments and tracking events
that happened *after* the customer's message, or the drafter would be
graded with knowledge the staff member didn't have.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol


class StoreAdapter(Protocol):
    def store_profile(self) -> dict:
        """{"name", "voice", "support_signoff", "timezone", "currency", "domain"}"""

    def lookup_orders(self, *, email: str | None = None, order_number: str | None = None,
                      as_of: datetime | None = None, limit: int = 5) -> list[dict]: ...

    def search_products(self, query: str, limit: int = 5) -> list[dict]: ...

    def get_product(self, ref: str) -> dict | None:
        """``ref`` is a SKU, handle or product title."""

    def search_policies(self, query: str, limit: int = 3) -> list[dict]: ...
