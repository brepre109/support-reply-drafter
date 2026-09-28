"""An in-memory store shaped like Shopify data, loaded from JSON.

``data/store.json`` holds a fictional candle shop: products with variants and
inventory, policies, customers, and orders with fulfillments and carrier
tracking events. Everything is synthetic.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..textutil import score


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _visible(ts: str | None, as_of: datetime | None) -> bool:
    return as_of is None or (ts is not None and parse_ts(ts) <= as_of)


def normalize_order_number(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return f"#{digits}" if digits else ""


class MockShopifyAdapter:
    def __init__(self, data: dict):
        self.data = data

    @classmethod
    def from_file(cls, path: str | Path) -> "MockShopifyAdapter":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- profile ---------------------------------------------------------------
    def store_profile(self) -> dict:
        return dict(self.data["shop"])

    # -- orders ------------------------------------------------------------------
    def _shape_order(self, o: dict, as_of: datetime | None) -> dict:
        cancelled = o.get("cancelledAt") if _visible(o.get("cancelledAt"), as_of) and o.get("cancelledAt") else None
        fulfillments = []
        for f in o.get("fulfillments", []):
            if not _visible(f["createdAt"], as_of):
                continue
            events = [e for e in f.get("events", []) if _visible(e["at"], as_of)]
            latest = events[-1] if events else None
            fulfillments.append({
                "shipped_at": f["createdAt"], "carrier": f["carrier"], "tracking_number": f["trackingNumber"],
                "latest_status": latest["status"] if latest else "label_created",
                "latest_event": latest, "events": events,
            })
        if cancelled:
            fulfillment_status = "CANCELLED"
        elif not fulfillments:
            fulfillment_status = "UNFULFILLED"
        elif any(f["latest_status"] == "delivered" for f in fulfillments):
            fulfillment_status = "DELIVERED"
        else:
            fulfillment_status = "SHIPPED"
        financial = o["financialStatus"]
        if financial == "REFUNDED" and not cancelled:
            financial = "PAID"  # the refund happened after as_of
        return {
            "order_number": o["name"], "email": o["email"], "created_at": o["createdAt"],
            "financial_status": financial, "fulfillment_status": fulfillment_status, "cancelled_at": cancelled,
            "shipping_method": o.get("shippingMethod"), "shipping_country": o.get("shippingCountry"),
            "items": [{"sku": li["sku"], "title": li["title"], "quantity": li["quantity"], "price": li["price"]}
                      for li in o["lineItems"]],
            "note": o.get("note"), "fulfillments": fulfillments,
        }

    def lookup_orders(self, *, email=None, order_number=None, as_of=None, limit=5) -> list[dict]:
        wanted = normalize_order_number(order_number) if order_number else None
        out = []
        for o in sorted(self.data["orders"], key=lambda o: o["createdAt"], reverse=True):
            if not _visible(o["createdAt"], as_of):
                continue
            if wanted and o["name"] != wanted:
                continue
            if email and o["email"].lower() != email.lower():
                continue
            out.append(self._shape_order(o, as_of))
            if len(out) >= limit:
                break
        return out

    # -- catalog -----------------------------------------------------------------
    @staticmethod
    def _shape_product(p: dict) -> dict:
        return {
            "title": p["title"], "handle": p["handle"], "status": p["status"], "type": p["productType"],
            "description": p["description"], "details": p.get("details", {}),
            "url": f"/products/{p['handle']}",
            "variants": [{"sku": v["sku"], "title": v["title"], "price": v["price"],
                          "available": v["inventory"] is None or v["inventory"] > 0,
                          "inventory": v["inventory"], **({"burn_time": v["burn_time"]} if "burn_time" in v else {})}
                         for v in p["variants"]],
        }

    def search_products(self, query: str, limit: int = 5) -> list[dict]:
        scored = []
        for p in self.data["products"]:
            s = score(query, p["title"], p["description"], " ".join(p.get("tags", [])), p["productType"],
                      " ".join(v["sku"] for v in p["variants"]), json.dumps(p.get("details", {})))
            if s > 0:
                scored.append((s, p["status"] == "ACTIVE", p))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [self._shape_product(p) for _, _, p in scored[:limit]]

    def get_product(self, ref: str) -> dict | None:
        ref_l = ref.strip().lower()
        for p in self.data["products"]:
            if ref_l in (p["handle"], p["title"].lower()) or any(v["sku"].lower() == ref_l for v in p["variants"]):
                return self._shape_product(p)
        hits = self.search_products(ref, limit=1)
        return hits[0] if hits else None

    # -- policies ----------------------------------------------------------------
    def search_policies(self, query: str, limit: int = 3) -> list[dict]:
        scored = [(score(query, p["topic"], p["title"], p["text"]), p) for p in self.data["policies"]]
        scored = [t for t in scored if t[0] > 0]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [{"topic": p["topic"], "title": p["title"], "text": p["text"]} for _, p in scored[:limit]]
