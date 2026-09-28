from conftest import ts


def test_lookup_by_number_accepts_any_format(adapter):
    for ref in ("1041", "#1041", "order 1041"):
        (o,) = adapter.lookup_orders(order_number=ref)
        assert o["order_number"] == "#1041"


def test_lookup_by_email_newest_first(adapter):
    orders = adapter.lookup_orders(email="HANNAH.KIM@example.com")
    assert [o["order_number"] for o in orders] == ["#1047"]


def test_as_of_hides_future_tracking_events(adapter):
    (o,) = adapter.lookup_orders(order_number="1047", as_of=ts("2026-03-17T09:00:00Z"))
    assert o["fulfillment_status"] == "SHIPPED"
    f = o["fulfillments"][0]
    assert [e["status"] for e in f["events"]] == ["label_created", "in_transit"]
    (now,) = adapter.lookup_orders(order_number="1047")
    assert now["fulfillment_status"] == "DELIVERED"


def test_as_of_hides_future_orders_and_fulfillments(adapter):
    assert adapter.lookup_orders(order_number="1047", as_of=ts("2026-03-01T00:00:00Z")) == []
    (o,) = adapter.lookup_orders(order_number="1047", as_of=ts("2026-03-15T00:00:00Z"))
    assert o["fulfillment_status"] == "UNFULFILLED" and o["fulfillments"] == []


def test_cancellation_and_refund_respect_as_of(adapter):
    (before,) = adapter.lookup_orders(order_number="1053", as_of=ts("2026-03-27T10:00:00Z"))
    assert before["fulfillment_status"] == "UNFULFILLED" and before["financial_status"] == "PAID"
    (after,) = adapter.lookup_orders(order_number="1053", as_of=ts("2026-03-28T09:00:00Z"))
    assert after["fulfillment_status"] == "CANCELLED" and after["financial_status"] == "REFUNDED"


def test_product_search_and_stock(adapter):
    top = adapter.search_products("lavender diffuser refill")[0]
    assert top["handle"] in {"lavender-fields-diffuser-refill", "lavender-fields-reed-diffuser"}
    cedar = adapter.get_product("FH-CDL-CEDAR-14")
    assert {v["title"]: v["available"] for v in cedar["variants"]} == {"8 oz": True, "14 oz": False}
    assert adapter.get_product("brass-candle-snuffer")["variants"][0]["available"] is False


def test_policy_search(adapter):
    assert adapter.search_policies("can I return a candle")[0]["topic"] == "returns"
    assert adapter.search_policies("do you ship to australia")[0]["topic"] == "shipping"
    assert adapter.search_policies("zzzz") == []
