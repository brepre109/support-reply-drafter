"""Tool schemas (OpenAI function-calling format) and their dispatcher."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from .adapters.base import StoreAdapter

CATEGORIES = [
    "order_status", "delivery_problem", "damaged_or_wrong_item", "returns_refunds", "order_change",
    "product_question", "stock", "discount", "shipping_info", "wholesale", "other",
]


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or [],
                       "additionalProperties": False}}}


TOOLS = [
    _fn("lookup_orders",
        "Look up orders by order number and/or email. With no arguments, returns the current customer's recent "
        "orders. Results include items, payment and fulfillment status, carrier, tracking number and tracking events.",
        {"order_number": {"type": "string", "description": "e.g. '1041' or '#1041'"},
         "email": {"type": "string"}}),
    _fn("search_products", "Keyword search over the catalog: titles, descriptions, scent notes, specs, stock, prices.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("get_product", "Fetch one product by SKU, handle or exact title.",
        {"ref": {"type": "string"}}, ["ref"]),
    _fn("search_policies",
        "Search store policies: shipping, returns, damaged items, order changes, discounts, wholesale, care, gifts.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("submit_reply", "Submit the final draft. Call exactly once, when done.", {
        "reply": {"type": "string", "description": "The message to the customer, ready to send."},
        "category": {"type": "string", "enum": CATEGORIES},
        "confidence": {"type": "number", "description": "0-1: how sure you are the reply is correct and complete."},
        "needs_human": {"type": "boolean",
                        "description": "True if a person must act or decide (refund, replacement, address change, "
                                       "cancellation, an angry or legal message) or a fact is unknowable from tools."},
        "human_reason": {"type": "string", "description": "Why a human is needed (empty if not)."},
        "facts_used": {"type": "array", "items": {"type": "string"},
                       "description": "Each concrete fact the reply relies on, with its source tool."},
    }, ["reply", "category", "confidence", "needs_human"]),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


@dataclass
class ToolBox:
    adapter: StoreAdapter
    customer_email: str | None = None
    as_of: datetime | None = None
    calls: list[dict] = field(default_factory=list)

    def call(self, name: str, args: dict) -> object:
        try:
            if name == "lookup_orders":
                email = args.get("email") or (None if args.get("order_number") else self.customer_email)
                result = self.adapter.lookup_orders(email=email, order_number=args.get("order_number"),
                                                    as_of=self.as_of)
                if not result:
                    result = {"orders": [], "note": "No matching orders. Ask the customer to confirm the order "
                                                   "number or the email used at checkout."}
            elif name == "search_products":
                result = self.adapter.search_products(args.get("query", "")) or {"products": [], "note": "no matches"}
            elif name == "get_product":
                result = self.adapter.get_product(args.get("ref", "")) or {"error": "product not found"}
            elif name == "search_policies":
                result = self.adapter.search_policies(args.get("query", "")) or {"policies": [], "note": "no matches"}
            else:
                result = {"error": f"unknown tool {name}"}
        except Exception as exc:  # a failing data source must not crash the draft
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.calls.append({"tool": name, "args": args, "result": result})
        return result

    @staticmethod
    def dumps(result: object, limit: int = 12000) -> str:
        text = json.dumps(result, ensure_ascii=False, default=str)
        return text if len(text) <= limit else text[:limit] + "...(truncated)"
