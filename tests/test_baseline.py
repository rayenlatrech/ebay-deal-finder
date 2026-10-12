"""Window baselines: market from tracked listings, retail as a safety net."""

import pytest

from src.baseline import (
    MarketPrice, compute_window_prices, learn_storage_premiums,
    reference_new_price, resolve_baseline, storage_gb, model_key,
)
from src.price_tracker import PriceTracker


def rows(key, band, market, prices):
    return [{"product_key": key, "condition_band": band, "marketplace": market, "price": p} for p in prices]


def calibration(n_models=6, gap=0.95):
    """Well-sampled models in both markets, so the DE/IT gap can be measured."""
    out = []
    for i in range(n_models):
        base = 300 + 50 * i
        out += rows(f"phone_{i}", "used", "EBAY_IT", [base] * 10)
        out += rows(f"phone_{i}", "used", "EBAY_DE", [base * gap] * 10)
    return out


def test_well_sampled_home_market_ignores_foreign():
    data = rows("iphone_15_128gb", "used", "EBAY_IT", [400] * 10) + \
           rows("iphone_15_128gb", "used", "EBAY_DE", [300] * 50) + calibration()
    prices, _ = compute_window_prices(data, "EBAY_IT", ["EBAY_DE"])
    mp = prices["iphone_15_128gb#used"]
    assert mp.value == 400 and mp.n == 10 and mp.n_foreign == 0


def test_thin_home_is_topped_up_with_rescaled_foreign():
    data = rows("iphone_17_512gb", "used", "EBAY_IT", [850, 860, 870]) + \
           rows("iphone_17_512gb", "used", "EBAY_DE", [817, 817, 817, 817, 817]) + calibration()
    prices, ratios = compute_window_prices(data, "EBAY_IT", ["EBAY_DE"])
    assert ratios[("EBAY_DE", "used")] == pytest.approx(0.95)
    mp = prices["iphone_17_512gb#used"]
    assert mp.n == 8 and mp.n_home == 3 and mp.n_foreign == 5
    assert 850 <= mp.value <= 870          # German prices lifted to Italian level (817/0.95 = 860)


def test_foreign_market_far_from_home_is_ignored():
    # The iPhone 17 Pro case: Germany "cheaper" by a third means fakes.
    data = rows("iphone_17_pro_256gb", "used", "EBAY_IT", [830, 832, 835]) + \
           rows("iphone_17_pro_256gb", "used", "EBAY_DE", [548] * 20) + calibration()
    prices, _ = compute_window_prices(data, "EBAY_IT", ["EBAY_DE"])
    mp = prices["iphone_17_pro_256gb#used"]
    assert mp.n_foreign == 0 and mp.value >= 830


def test_no_home_listings_needs_a_self_sufficient_foreign_sample():
    big = rows("galaxy_s25_512gb", "used", "EBAY_DE", [570] * 9) + calibration()
    small = rows("galaxy_s25_512gb", "used", "EBAY_DE", [570] * 4) + calibration()
    assert compute_window_prices(big, "EBAY_IT", ["EBAY_DE"])[0]["galaxy_s25_512gb#used"].n == 9
    assert "galaxy_s25_512gb#used" not in compute_window_prices(small, "EBAY_IT", ["EBAY_DE"])[0]


def test_gap_needs_enough_models_to_be_measured():
    data = rows("x", "used", "EBAY_IT", [400, 400, 400]) + rows("x", "used", "EBAY_DE", [380] * 5) + \
           calibration(n_models=3)
    prices, ratios = compute_window_prices(data, "EBAY_IT", ["EBAY_DE"], min_ratio_models=5)
    assert ("EBAY_DE", "used") not in ratios
    assert prices["x#used"].n_foreign == 0


def test_unrecorded_marketplace_counts_as_home():
    prices, _ = compute_window_prices(rows("x", "used", None, [300] * 8), "EBAY_IT")
    assert prices["x#used"].n_home == 8


@pytest.mark.parametrize("old,new", [
    ("galaxy_s25_fe_8gb", "galaxy_s25_fe"),          # RAM figure dropped
    ("iphone_16_pro_126gb", "iphone_16_pro_128gb"),  # typo snapped
    ("iphone_16_pro_max_1000gb", "iphone_16_pro_max_1tb"),
    ("iphone_15_128gb", "iphone_15_128gb"),
    ("ps5_slim", "ps5_slim"),
])
def test_stored_keys_are_normalized(old, new):
    from src.baseline import normalize_key
    assert normalize_key(old) == new


def test_old_ram_keys_no_longer_hide_the_base_tier():
    # galaxy_s25_fe_8gb used to make "8GB" the smallest tier, so the S25 FE's
    # retail price never applied to the 128GB bucket and its cap never fired.
    data = rows("galaxy_s25_fe_8gb", "new", "EBAY_IT", [600] * 5) + \
           rows("galaxy_s25_fe_128gb", "new", "EBAY_IT", [661] * 13)
    prices, _ = compute_window_prices(data, "EBAY_IT")
    refs = {"galaxy_s25_fe": {"new_price": 432}}
    assert reference_new_price("galaxy_s25_fe_128gb", refs, prices, {}) == 432
    b = resolve_baseline(prices["galaxy_s25_fe_128gb#new"], 432, band_factor=1.0)
    assert (b.value, b.source) == (432, "new-price cap")


def test_unlisted_marketplaces_are_ignored():
    data = rows("x", "used", "EBAY_NL", [100] * 20) + rows("x", "used", "EBAY_IT", [300] * 8)
    prices, _ = compute_window_prices(data, "EBAY_IT", ["EBAY_DE"])
    assert prices["x#used"].value == 300


# ── storage-aware retail ───────────────────────────────────────────────

def mp(value, n=10):
    return MarketPrice(value, n, n, 0)


@pytest.fixture
def iphone17_market():
    return {
        "iphone_17_256gb#used": mp(760), "iphone_17_512gb#used": mp(860, n=2),
        "iphone_16_128gb#used": mp(560), "iphone_16_256gb#used": mp(616),
        "iphone_15_128gb#used": mp(400), "iphone_15_256gb#used": mp(446),
        "iphone_15_512gb#used": mp(500), "iphone_14_256gb#used": mp(344),
        "iphone_14_512gb#used": mp(386),
    }


def test_storage_helpers():
    assert storage_gb("iphone_17_1tb") == 1024 and storage_gb("iphone_17") is None
    assert model_key("galaxy_s25_ultra_512gb") == "galaxy_s25_ultra"


def test_exact_tier_reference_wins(iphone17_market):
    refs = {"iphone_17_512gb": {"new_price": 1229}, "iphone_17": {"new_price": 979}}
    premiums = learn_storage_premiums(iphone17_market)
    assert reference_new_price("iphone_17_512gb", refs, iphone17_market, premiums) == 1229


def test_model_reference_is_the_base_tier_price(iphone17_market):
    refs = {"iphone_17": {"new_price": 979}}
    premiums = learn_storage_premiums(iphone17_market)
    # 256GB is iPhone 17's smallest tier on the market, so the entry applies as-is.
    assert reference_new_price("iphone_17_256gb", refs, iphone17_market, premiums) == 979


def test_bigger_tier_is_scaled_by_the_learned_premium(iphone17_market):
    refs = {"iphone_17": {"new_price": 979}}
    premiums = learn_storage_premiums(iphone17_market)
    ratio = premiums[("iphone", 256, 512)]
    assert 1.05 < ratio < 1.20
    assert reference_new_price("iphone_17_512gb", refs, iphone17_market, premiums) == pytest.approx(979 * ratio)


def test_unknown_premium_withholds_the_price(iphone17_market):
    refs = {"iphone_17": {"new_price": 979}}
    assert reference_new_price("iphone_17_1tb", refs, iphone17_market, {}) is None


# ── the decision ───────────────────────────────────────────────────────

def test_enough_listings_use_the_market():
    b = resolve_baseline(mp(758, n=55), new_price=979, band_factor=0.75)
    assert (b.value, b.source, b.n) == (758, "market", 55)


def test_used_price_above_new_is_capped():
    b = resolve_baseline(mp(900, n=20), new_price=850, band_factor=0.75)
    assert (b.value, b.source) == (850, "new-price cap")


def test_few_listings_blend_toward_a_lower_retail_estimate():
    b = resolve_baseline(mp(700, n=4), new_price=800, band_factor=0.75)   # estimate 600
    assert b.source == "blend"
    assert b.value == pytest.approx(0.5 * 700 + 0.5 * 600)


def test_retail_never_raises_a_thin_market():
    # The sealed-Galaxy case: list price above the street price must not turn
    # an ordinary listing into a "deal".
    b = resolve_baseline(mp(700, n=4), new_price=1029, band_factor=1.0)
    assert (b.value, b.source) == (700, "thin market")


def test_one_or_two_listings_are_not_a_market():
    # A 2015 iPad Pro at €175 was judged against a single €721 listing.
    assert resolve_baseline(mp(721, n=1), new_price=None, band_factor=0.75) is None
    assert resolve_baseline(mp(700, n=2), new_price=1029, band_factor=1.0) is None


def test_few_listings_without_retail_use_the_thin_market():
    b = resolve_baseline(mp(700, n=4), new_price=None, band_factor=0.75)
    assert (b.value, b.source) == (700, "thin market")


def test_hardly_any_listings_take_a_lower_retail_estimate():
    b = resolve_baseline(mp(700, n=2), new_price=800, band_factor=0.75)
    assert (b.value, b.source) == (600, "retail estimate")


def test_no_market_is_never_judged_on_retail_alone():
    assert resolve_baseline(None, new_price=800, band_factor=0.75) is None
    assert resolve_baseline(None, new_price=None, band_factor=0.75) is None


def test_implausible_storage_premiums_are_dropped():
    market = {
        "switch_oled_64gb#used": mp(250), "switch_oled_128gb#used": mp(220),     # cheaper: noise
        "switch_lite_64gb#used": mp(150), "switch_lite_128gb#used": mp(130),
        "ipad_64gb#used": mp(200), "ipad_128gb#used": mp(380),                   # +90%: mixed gens
        "ipad_air_64gb#used": mp(300), "ipad_air_128gb#used": mp(560),
    }
    premiums = learn_storage_premiums(market)
    assert ("switch", 64, 128) not in premiums
    assert ("ipad", 64, 128) not in premiums


# ── storage parsing ────────────────────────────────────────────────────

@pytest.fixture
def tracker():
    return PriceTracker(None)


@pytest.mark.parametrize("title,expected", [
    ("IPHONE 16 PRO 126GB TITANIO NATURALE", "iphone_16_pro_128gb"),      # typo snapped
    ("Samsung Galaxy S25 265GB", "galaxy_s25_256gb"),
    ("Samsung Galaxy S23 Ultra 12/1024gb Cream", "galaxy_s23_ultra_1tb"),
    ("iPhone 14 Pro 1000GB", "iphone_14_pro_1tb"),
    ("Samsung Galaxy S25 Doppia SIM Android 15 5G 12 GB", "galaxy_s25"),   # RAM ignored
    ("Apple iPhone 13 Bianco 5,12 GB Smartphone Usato", "iphone_13"),
    ("PS5 Console 825GB", "ps5_console_825gb"),
    ("SAMSUNG GALAXY S26 FE 5G 8+512GB GRAPHITE", "galaxy_s26_fe_512gb"),   # "+" is RAM+storage
    ("Samsung Galaxy S26+ 256GB", "galaxy_s26_plus_256gb"),
    ("Samsung Galaxy S25 Plus 512GB", "galaxy_s25_plus_512gb"),
    ("iPhone 15 Plus 128GB", "iphone_15_plus_128gb"),
    ("iPhone 15+ 128GB", "iphone_15_plus_128gb"),
    ("iPhone 15 6+128GB nero", "iphone_15_128gb"),
])
def test_storage_tiers(tracker, title, expected):
    assert tracker.generate_product_key(title)[0] == expected


@pytest.mark.parametrize("title", [
    "Apple iPhone 256 GB Argento iOS usato",     # was iphone_25_256gb
    "Apple iPhone 512 GB Arancione con scatola",  # was iphone_51_512gb
    "Apple iPhone 1TB Bianco",                   # was iphone_1_1tb
    "Apple iPhone 5G 128 GB Nero",               # "5G" is the network
    "Apple iPhone 2G 16gb 1a Generazione",
])
def test_no_phantom_iphone_models(tracker, title):
    assert tracker.generate_product_key(title)[0] == ""
