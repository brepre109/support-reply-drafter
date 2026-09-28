"""Pluggable data sources the drafter's tools read from."""
from .base import StoreAdapter
from .mock_shopify import MockShopifyAdapter

__all__ = ["StoreAdapter", "MockShopifyAdapter"]
