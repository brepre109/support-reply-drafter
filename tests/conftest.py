from pathlib import Path

import pytest

from reply_drafter.adapters.mock_shopify import MockShopifyAdapter, parse_ts
from reply_drafter.dataset import all_cases, load_conversations

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def adapter():
    return MockShopifyAdapter.from_file(DATA / "store.json")


@pytest.fixture
def conversations():
    return load_conversations(DATA / "conversations.jsonl")


@pytest.fixture
def cases(conversations):
    return all_cases(conversations)


def ts(value):
    return parse_ts(value)
