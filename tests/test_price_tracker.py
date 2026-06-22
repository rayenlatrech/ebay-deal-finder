"""Tests for the pre-AI classification and baseline robustness layers.

These guard the integrity of the funnel: if grouping or outlier rejection
regresses, the market baseline silently corrupts and every downstream decision
(and AI token) is wasted on the wrong listings.
"""

import pytest

from src.price_tracker import PriceTracker, is_for_parts


@pytest.fixture
def tracker():
    # generate_product_key / static helpers never touch the DB.
    return PriceTracker.__new__(PriceTracker)


# ── Classification: variant precedence ────────────────────────────────────

@pytest.mark.parametrize("title,expected_key", [
    ("Sony PS5 PlayStation Portal Remote Player", "ps5_portal"),
    ("PS5 Console 825GB", "ps5_console_825gb"),
    ("PlayStation 5 Console 825GB", "ps5_console_825gb"),   # same bucket as "PS5"
    ("PlayStation 5 Slim Digital 1TB", "ps5_slim_1tb"),
    ("iPhone 13 Pro Max 256GB", "iphone_13_pro_max_256gb"),
    ("iPhone 14 128GB ottimo stato", "iphone_14_128gb"),
    ("Samsung Galaxy S9 64GB", "galaxy_s9_64gb"),           # single-digit model
    ("MacBook Air M2 256GB", "macbook_air_m2_256gb"),
    ("AirPods Pro 2 originali", "airpods_pro"),
])
def test_device_classification(tracker, title, expected_key):
    key, confidence = tracker.generate_product_key(title)
    assert key == expected_key
    assert confidence >= PriceTracker.MIN_CONFIDENCE


# ── Classification: accessory / bundle disambiguation ─────────────────────

def test_bundled_charger_stays_a_device(tracker):
    # "with charger" => the charger is an included extra, not the product.
    key, confidence = tracker.generate_product_key("MacBook Pro M1 13 con caricatore incluso")
    assert key == "macbook_pro_m1"
    assert confidence >= PriceTracker.MIN_CONFIDENCE


@pytest.mark.parametrize("title", [
    "Caricatore per MacBook Pro USB-C 96W",   # "per" => part FOR a device
    "Cover iPhone 13 Pro silicone",           # accessory leads the title
    "iPhone 13 schermo display ricambio",     # spare part
    "Ricambio batteria per iPhone 12",
])
def test_accessories_are_demoted(tracker, title):
    key, confidence = tracker.generate_product_key(title)
    assert key.endswith("_part")
    assert confidence < PriceTracker.MIN_CONFIDENCE


def test_unparseable_falls_back_to_epid(tracker):
    assert tracker.generate_product_key("Strange unparseable thing") == ("", 0.0)
    assert tracker.generate_product_key("Strange unparseable thing", epid="123456") == ("epid_123456", 0.9)


# ── for-parts detection ───────────────────────────────────────────────────

@pytest.mark.parametrize("listing,expected", [
    ({"condition_id": "7000"}, True),
    ({"condition": "For parts or not working"}, True),
    ({"condition": "Usato - per ricambi"}, True),
    ({"condition": "Used"}, False),
    ({}, False),
])
def test_is_for_parts(listing, expected):
    assert is_for_parts(listing) is expected


# ── Statistical outlier rejection ─────────────────────────────────────────

def test_iqr_drops_part_and_bundle():
    # €150 part and €900 bundle around a cluster of ~€400 consoles.
    kept = PriceTracker._reject_outliers([150, 380, 390, 400, 410, 420, 900])
    assert 150 not in kept and 900 not in kept
    assert set(kept) == {380, 390, 400, 410, 420}


def test_iqr_noop_on_small_samples():
    assert PriceTracker._reject_outliers([100, 900]) == [100, 900]


# ── ePID purity guard ─────────────────────────────────────────────────────

def test_epid_guard_drops_minority_catalog_product():
    # Four "console" ePIDs around €400, one Portal ePID at €190 mis-bucketed here.
    pairs = [
        (400, "ep_console"), (410, "ep_console"),
        (395, "ep_console"), (405, "ep_console"),
        (190, "ep_portal"),
    ]
    prices = PriceTracker._purge_minority_epids(pairs)
    assert 190 not in prices
    assert sorted(prices) == [395, 400, 405, 410]


def test_epid_guard_keeps_unidentified_listings():
    pairs = [
        (400, "ep_console"), (410, "ep_console"),
        (395, "ep_console"), (405, "ep_console"),
        (402, ""),  # no catalog id -> benefit of the doubt
    ]
    prices = PriceTracker._purge_minority_epids(pairs)
    assert 402 in prices


def test_epid_guard_inactive_without_clear_majority():
    # No dominant catalog product => guard must not touch the data.
    pairs = [(400, "a"), (410, "b"), (395, "c"), (405, "d")]
    assert sorted(PriceTracker._purge_minority_epids(pairs)) == [395, 400, 405, 410]
