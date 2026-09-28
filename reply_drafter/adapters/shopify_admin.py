"""Read-only adapter for a live store through the Shopify Admin GraphQL API.

Scopes: ``read_orders`` (plus ``read_all_orders`` for orders older than 60
days), ``read_products``, ``read_legal_policies``. Customer email and name
fields are protected customer data, so the app must be approved for that
access level.

This adapter only reads. The drafter never sends messages, edits orders or
issues refunds, by design: a human reviews and sends every reply.
"""
from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Callable

import requests

from .mock_shopify import _visible, normalize_order_number

API_VERSION = "2026-07"

ORDERS = """
query SupportOrders($query: String!, $first: Int!) {
  orders(first: $first, query: $query, sortKey: CREATED_AT, reverse: true) {
    nodes {
      name email createdAt cancelledAt note
      displayFinancialStatus displayFulfillmentStatus
      shippingLine { title }
      shippingAddress { countryCodeV2 }
      lineItems(first: 30) { nodes { sku name quantity originalUnitPriceSet { shopMoney { amount } } } }
      fulfillments(first: 10) {
        createdAt displayStatus
        trackingInfo(first: 3) { company number }
        events(first: 30) { nodes { happenedAt status city province } }
      }
    }
  }
}
"""

PRODUCTS = """
query SupportProducts($query: String!, $first: Int!) {
  products(first: $first, query: $query) {
    nodes {
      title handle status productType description
      variants(first: 30) { nodes { sku title price inventoryQuantity availableForSale } }
    }
  }
}
"""

POLICIES = """
query SupportPolicies { shop { name shopPolicies { type title body } } }
"""


def strip_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value or "")).replace("\xa0", " ").strip()


class ShopifyAdminAdapter:
    def __init__(self, shop: str, token: str, profile: dict | None = None,
                 post: Callable[..., dict] | None = None, api_version: str = API_VERSION):
        self.url = f"https://{shop}/admin/api/{api_version}/graphql.json"
        self.token = token
        self.profile = profile or {"name": shop, "voice": "Friendly and concise.", "support_signoff": "",
                                   "timezone": "UTC", "currency": "USD", "domain": shop}
        self._post = post or self._requests_post
        self._policies: list[dict] | None = None

    def _requests_post(self, url: str, json: dict, headers: dict) -> dict:
        r = requests.post(url, json=json, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        body = self._post(self.url, json={"query": query, "variables": variables or {}},
                          headers={"X-Shopify-Access-Token": self.token, "Content-Type": "application/json"})
        if body.get("errors"):
            raise RuntimeError("; ".join(e.get("message", "") for e in body["errors"]))
        return body["data"]

    def store_profile(self) -> dict:
        return dict(self.profile)

    def lookup_orders(self, *, email=None, order_number=None, as_of: datetime | None = None, limit=5):
        terms = []
        if order_number:
            terms.append(f"name:{normalize_order_number(order_number)}")
        if email:
            terms.append(f"email:{email}")
        if not terms:
            return []
        if as_of:
            terms.append(f"created_at:<='{as_of.isoformat()}'")
        nodes = self._gql(ORDERS, {"query": " ".join(terms), "first": limit})["orders"]["nodes"]
        return [self._shape_order(o, as_of) for o in nodes if _visible(o["createdAt"], as_of)]

    @staticmethod
    def _shape_order(o: dict, as_of) -> dict:
        fulfillments = []
        for f in o.get("fulfillments") or []:
            if not _visible(f["createdAt"], as_of):
                continue
            events = [{"at": e["happenedAt"], "status": e["status"].lower(),
                       "location": ", ".join(x for x in (e.get("city"), e.get("province")) if x)}
                      for e in (f.get("events") or {}).get("nodes", []) if _visible(e["happenedAt"], as_of)]
            tracking = (f.get("trackingInfo") or [{}])[0] if f.get("trackingInfo") else {}
            fulfillments.append({"shipped_at": f["createdAt"], "carrier": tracking.get("company"),
                                 "tracking_number": tracking.get("number"),
                                 "latest_status": events[-1]["status"] if events else (f.get("displayStatus") or "").lower(),
                                 "latest_event": events[-1] if events else None, "events": events})
        cancelled = o.get("cancelledAt") if _visible(o.get("cancelledAt"), as_of) else None
        return {
            "order_number": o["name"], "email": o.get("email"), "created_at": o["createdAt"],
            "financial_status": o.get("displayFinancialStatus"),
            # Current status from Shopify; in a replay the fulfillments list is the as-of view.
            "fulfillment_status": "CANCELLED" if cancelled else o.get("displayFulfillmentStatus"),
            "cancelled_at": cancelled,
            "shipping_method": (o.get("shippingLine") or {}).get("title"),
            "shipping_country": (o.get("shippingAddress") or {}).get("countryCodeV2"),
            "items": [{"sku": li.get("sku"), "title": li["name"], "quantity": li["quantity"],
                       "price": li["originalUnitPriceSet"]["shopMoney"]["amount"]}
                      for li in o["lineItems"]["nodes"]],
            "note": o.get("note"), "fulfillments": fulfillments,
        }

    def _products(self, query: str, limit: int) -> list[dict]:
        nodes = self._gql(PRODUCTS, {"query": query, "first": limit})["products"]["nodes"]
        return [{
            "title": p["title"], "handle": p["handle"], "status": p["status"], "type": p["productType"],
            "description": p["description"], "details": {}, "url": f"/products/{p['handle']}",
            "variants": [{"sku": v["sku"], "title": v["title"], "price": v["price"],
                          "available": v["availableForSale"], "inventory": v["inventoryQuantity"]}
                         for v in p["variants"]["nodes"]],
        } for p in nodes]

    def search_products(self, query: str, limit: int = 5) -> list[dict]:
        return self._products(query, limit)

    def get_product(self, ref: str) -> dict | None:
        hits = self._products(f"sku:{ref} OR handle:{ref}", 1) or self._products(ref, 1)
        return hits[0] if hits else None

    def search_policies(self, query: str, limit: int = 3) -> list[dict]:
        from ..textutil import score

        if self._policies is None:
            shop = self._gql(POLICIES)["shop"]
            self._policies = [{"topic": p["type"].lower(), "title": p["title"], "text": strip_html(p["body"])}
                              for p in shop["shopPolicies"]]
        scored = sorted(((score(query, p["topic"], p["title"], p["text"]), p) for p in self._policies),
                        key=lambda t: t[0], reverse=True)
        return [p for s, p in scored if s > 0][:limit]
