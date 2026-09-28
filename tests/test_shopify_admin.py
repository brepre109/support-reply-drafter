from conftest import ts
from reply_drafter.adapters.shopify_admin import ShopifyAdminAdapter

ORDER = {
    "name": "#2001", "email": "a@example.com", "createdAt": "2026-05-01T10:00:00Z", "cancelledAt": None, "note": None,
    "displayFinancialStatus": "PAID", "displayFulfillmentStatus": "FULFILLED",
    "shippingLine": {"title": "Standard"}, "shippingAddress": {"countryCodeV2": "US"},
    "lineItems": {"nodes": [{"sku": "SKU-1", "name": "Thing", "quantity": 2,
                             "originalUnitPriceSet": {"shopMoney": {"amount": "9.5"}}}]},
    "fulfillments": [{"createdAt": "2026-05-02T10:00:00Z", "displayStatus": "IN_TRANSIT",
                      "trackingInfo": [{"company": "UPS", "number": "1ZTEST"}],
                      "events": {"nodes": [
                          {"happenedAt": "2026-05-02T12:00:00Z", "status": "IN_TRANSIT", "city": "Reno", "province": "NV"},
                          {"happenedAt": "2026-05-04T12:00:00Z", "status": "DELIVERED", "city": "Austin", "province": "TX"}]}}],
}


def make(responses):
    sent = []

    def post(url, json, headers):
        sent.append({"url": url, "json": json, "headers": headers})
        return responses.pop(0)

    return ShopifyAdminAdapter("your-store.myshopify.com", "tok", post=post), sent


def test_orders_query_and_shaping_with_as_of():
    a, sent = make([{"data": {"orders": {"nodes": [ORDER]}}}])
    (o,) = a.lookup_orders(order_number="2001", as_of=ts("2026-05-03T00:00:00Z"))
    q = sent[0]["json"]["variables"]["query"]
    assert "name:#2001" in q and "created_at:<=" in q
    assert sent[0]["headers"]["X-Shopify-Access-Token"] == "tok"
    assert sent[0]["url"].endswith("/admin/api/2026-07/graphql.json")
    f = o["fulfillments"][0]
    assert f["carrier"] == "UPS" and f["latest_status"] == "in_transit" and len(f["events"]) == 1
    assert o["items"] == [{"sku": "SKU-1", "title": "Thing", "quantity": 2, "price": "9.5"}]


def test_policies_are_fetched_once_and_stripped():
    a, sent = make([{"data": {"shop": {"name": "x", "shopPolicies": [
        {"type": "REFUND_POLICY", "title": "Refund policy", "body": "<p>Returns within&nbsp;30 days.</p>"}]}}}])
    assert a.search_policies("refund returns")[0]["text"] == "Returns within 30 days."
    a.search_policies("refund")
    assert len(sent) == 1


def test_graphql_errors_raise():
    import pytest

    a, _ = make([{"errors": [{"message": "Access denied for orders field"}]}])
    with pytest.raises(RuntimeError, match="Access denied"):
        a.lookup_orders(email="a@example.com")
