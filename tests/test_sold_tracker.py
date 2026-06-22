"""Tests for the longitudinal sold-price tracker's pure logic."""

from datetime import datetime, timezone, timedelta

import pytest

from src.sold_tracker import SoldTracker, condition_band, _parse


# ── condition_band ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("listing,expected", [
    ({"condition_id": "1000"}, "new"),
    ({"condition_id": "2000"}, "refurb"),
    ({"condition_id": "3000"}, "used"),
    ({"condition_id": "7000"}, "parts"),
    ({"condition": "Nuovo"}, "new"),
    ({"condition": "Ricondizionato"}, "refurb"),
    ({"condition": "Usato"}, "used"),
    ({"condition": "Per ricambi non funzionante"}, "parts"),
    ({}, "used"),
])
def test_condition_band(listing, expected):
    assert condition_band(listing) == expected


# ── build_comp: realized outcome + sold proxy ──────────────────────────────

def _row(first_days_ago: int, last_days_ago: int):
    now = datetime.now(timezone.utc)
    return {
        "id": "v1|123",
        "product_key": "iphone_13_128gb",
        "condition_band": "used",
        "price": 480.0,
        "first_seen": (now - timedelta(days=first_days_ago)).isoformat(),
        "last_seen": (now - timedelta(days=last_days_ago)).isoformat(),
    }


def test_fast_disappearance_is_sold_proxy():
    # Listed 5 days ago, last seen today, then 404 -> sold.
    comp = SoldTracker.build_comp(_row(5, 0), ended_via_404=True, sold_max_days=60)
    assert comp["sold_proxy"] is True
    assert comp["days_active"] == 5
    assert comp["final_price"] == 480.0
    assert comp["product_key"] == "iphone_13_128gb"


def test_long_lived_then_404_is_not_sold():
    # First seen 80 days ago, still visible until ~today, then 404: it sat on
    # the market well beyond the sold window -> not a confident "sold".
    comp = SoldTracker.build_comp(_row(80, 0), ended_via_404=True, sold_max_days=60)
    assert comp["sold_proxy"] is False
    assert comp["days_active"] == 80


def test_ended_without_404_is_never_sold():
    # Finalized because it exceeded the tracking horizon, not a confirmed 404.
    comp = SoldTracker.build_comp(_row(10, 1), ended_via_404=False, sold_max_days=60)
    assert comp["sold_proxy"] is False


def test_parse_handles_z_suffix():
    assert _parse("2026-06-22T00:00:00Z").tzinfo is not None
