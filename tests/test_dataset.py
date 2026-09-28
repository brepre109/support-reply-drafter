from reply_drafter.adapters.mock_shopify import parse_ts


def test_sample_dataset_shape(conversations, cases):
    assert len(conversations) == 30
    assert len(cases) == 31  # c03 has two staff turns
    assert len({c["id"] for c in conversations}) == 30
    for conv in conversations:
        assert conv["customer"]["email"].endswith("@example.com")


def test_cut_point_excludes_reference_and_freezes_time(cases):
    c = next(c for c in cases if c.case_id == "c03#3")
    assert [m["role"] for m in c.context] == ["customer", "staff", "customer"]
    assert c.reference.startswith("So glad it made it")
    assert c.as_of == parse_ts("2026-03-11T09:10:00Z")


def test_replay_never_leaks_the_future(cases, adapter):
    """For every case, nothing the tools can return is newer than the cut."""
    for case in cases:
        for o in adapter.lookup_orders(email=case.customer["email"], as_of=case.as_of, limit=50):
            assert parse_ts(o["created_at"]) <= case.as_of, case.case_id
            for f in o["fulfillments"]:
                assert parse_ts(f["shipped_at"]) <= case.as_of
                assert all(parse_ts(e["at"]) <= case.as_of for e in f["events"])


def test_every_referenced_order_exists(conversations, adapter):
    import re

    for conv in conversations:
        for num in re.findall(r"\b(10\d\d)\b", conv["messages"][0]["text"]):
            assert adapter.lookup_orders(order_number=num), f"{conv['id']} mentions unknown order {num}"
