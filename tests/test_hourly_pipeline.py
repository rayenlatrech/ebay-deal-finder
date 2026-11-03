"""A full scan followed by quick ones, end to end, with every service faked."""

from datetime import datetime, timedelta, timezone

import pytest

import src.ai_analyzer
import src.db
import src.ebay_client
import src.scan_state
import src.sold_tracker
import src.telegram_notifier
from src.main import run_pipeline

ITEM_PRICE = 300.0
MARKET_PRICE = 600.0


def _listing(item_id, price=ITEM_PRICE, **extra):
    return {
        "item_id": item_id, "title": "Apple iPhone 15 128GB", "price": price, "total_price": price,
        "shipping_cost": 0.0, "currency": "EUR", "condition": "Usato", "condition_id": "3000",
        "category": "iPhone", "epid": "", "marketplace": "EBAY_IT", "is_foreign": False,
        "item_location_country": "IT", "seller_username": f"seller{item_id}",
        "seller_feedback_score": 1000, "seller_feedback_percentage": 99.5,
        "image_url": f"https://i.ebayimg.com/images/g/{item_id}/s-l500.jpg", "all_image_urls": [],
        "item_url": f"https://www.ebay.it/itm/{item_id}", **extra,
    }


class World:
    """What the fake services see and do, across scans."""

    def __init__(self):
        self.listings = []
        self.auctions = []
        self.analyses = {}
        self.alerted = set()
        self.calls = {"item": [], "text": [], "image": [], "window": 0, "reconcile": 0, "price_points": 0}
        self.sent = []
        self.reports = []
        self.gemini_keys = ["key"]
        self.image_fails = False
        self.text_verdict = {"condition_assessment": "mint", "red_flags": [], "hidden_defects": [],
                             "is_good_deal": True, "is_complete_item": True}
        self.window_rows = [
            {"id": str(i), "product_key": "iphone_15_128gb", "condition_band": "used",
             "price": MARKET_PRICE, "marketplace": "EBAY_IT", "status": "active", "last_seen": "x"}
            for i in range(20)
        ]


@pytest.fixture
def world(monkeypatch, tmp_path):
    w = World()
    monkeypatch.setattr(src.scan_state, "STATE_PATH", tmp_path / "state.json")

    class FakeDb:
        def __getattr__(self, name):
            # Bookkeeping these tests don't look at.
            return lambda *a, **k: None

        def get_watchlist(self, only_enabled=True):
            return [{"name": "iPhone", "query": "iphone", "category_id": "9355", "min_discount": 10}]

        def get_reference_prices(self):
            return {}

        def get_tracked_since(self, since):
            w.calls["window"] += 1
            return w.window_rows

        def get_tracked_prices(self, ids):
            return {}

        def get_alerted_ids(self, ids):
            return {i for i in ids if i in w.alerted}

        def get_recent_analyses(self, ids, since):
            return {i: w.analyses[i] for i in ids if i in w.analyses}

        def save_analysis(self, analysis):
            w.analyses[analysis["listing_id"]] = analysis["ai_assessment"]

        def mark_alerted(self, item_id):
            w.alerted.add(item_id)

        def count_sellers_for_image(self, image_id):
            return 1

        def save_price_point(self, *a):
            w.calls["price_points"] += 1

        def get_price_history(self, key):
            return []

    class FakeEbay:
        @staticmethod
        def home_item_url(item_id, market):
            return f"https://www.ebay.it/itm/{item_id}"

        def search_all_categories(self, **kwargs):
            return [dict(l) for l in w.listings]

        def search_ending_auctions(self, **kwargs):
            return [dict(a) for a in w.auctions]

        def get_item_details(self, item_id, marketplace=None):
            w.calls["item"].append(item_id)
            return {"short_description": "Ottime condizioni, batteria 90%"}

        def item_is_active(self, item_id, marketplace=None):
            return True

    class FakeAi:
        def __init__(self, db):
            self.gemini_keys = w.gemini_keys

        def analyze_listing(self, listing):
            w.calls["text"].append(listing["item_id"])
            return dict(w.text_verdict)

        def analyze_image(self, image_url, **kwargs):
            w.calls["image"].append(image_url.split("/g/")[1].split("/")[0])
            if w.image_fails:
                return {"image_ai_failed": True, "photo_type": "uncertain"}
            return {"photo_type": "real", "condition_matches_description": True, "visible_damage": []}

        def quick_photo_check(self, listing):
            return {"suspicious": False, "flags": []}

        def extract_image_id(self, url):
            return url

    class FakeTelegram:
        def send_hot_deal_alert(self, listing, analysis, scores):
            w.sent.append(listing["item_id"])
            return True

        def send_deal_summary(self, deals):
            w.sent.extend(d["listing"]["item_id"] for d in deals)
            return True

        def send_scan_report(self, stats):
            return True

        def send_error_alert(self, msg):
            raise AssertionError(f"pipeline crashed: {msg}")

        def send_daily_report(self, report, gaps=None):
            w.reports.append(report)
            return True

    class FakeSoldTracker:
        def __init__(self, *a, **k):
            pass

        def record_sightings(self, listings):
            return len(listings)

        def reconcile_ended(self, ebay):
            w.calls["reconcile"] += 1
            return 0

        def rebuild_resale_stats(self):
            return 0

        def cleanup(self):
            pass

    monkeypatch.setattr(src.db, "Database", FakeDb)
    monkeypatch.setattr(src.ebay_client, "EbayClient", FakeEbay)
    monkeypatch.setattr(src.ai_analyzer, "AIAnalyzer", FakeAi)
    monkeypatch.setattr(src.telegram_notifier, "TelegramNotifier", FakeTelegram)
    monkeypatch.setattr(src.sold_tracker, "SoldTracker", FakeSoldTracker)
    return w


def _rate_poorly(w):
    """The AI's verdict keeps a listing from alerting, whatever its price."""
    w.text_verdict = {"condition_assessment": "poor", "red_flags": ["crepa sullo schermo"],
                      "hidden_defects": ["batteria da sostituire"], "is_good_deal": False,
                      "is_complete_item": True}


def _reset_calls(w):
    for key, value in w.calls.items():
        w.calls[key] = [] if isinstance(value, list) else 0


def _make_next_scan_full():
    state = src.scan_state.load_state()
    state["last_full_at"] = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    src.scan_state.save_state(state)


def test_first_scan_is_full_and_alerts(world):
    world.listings = [_listing("a")]
    run_pipeline()
    assert world.calls["window"] == 1 and world.calls["reconcile"] == 1
    assert world.calls["text"] == ["a"] and world.calls["image"] == ["a"]
    assert world.sent == ["a"]
    # The legacy per-scan price history isn't written in window mode.
    assert world.calls["price_points"] == 0


def test_quick_scan_reviews_a_new_listing_and_nothing_else(world):
    world.listings = [_listing("a")]
    run_pipeline()
    _reset_calls(world)

    world.listings = [_listing("a"), _listing("b")]
    run_pipeline()
    # Market prices reused, no resale maintenance, nothing repeated for "a".
    assert world.calls["window"] == 0 and world.calls["reconcile"] == 0
    assert world.calls["item"] == ["b"] and world.calls["text"] == ["b"]
    assert world.sent == ["a", "b"]


def test_unchanged_listing_that_did_not_alert_costs_nothing_on_a_quick_scan(world):
    _rate_poorly(world)
    world.listings = [_listing("seen")]
    run_pipeline()
    assert world.sent == [] and world.calls["text"] == ["seen"]
    _reset_calls(world)
    run_pipeline()
    assert world.calls["item"] == [] and world.calls["text"] == []


def test_price_drop_gets_reviewed_without_paying_for_ai_again(world):
    _rate_poorly(world)
    world.listings = [_listing("a", price=500.0)]
    run_pipeline()
    assert world.sent == [] and world.calls["text"] == ["a"]
    _reset_calls(world)

    world.listings = [_listing("a", price=300.0)]
    run_pipeline()
    assert world.calls["item"] == ["a"]    # looked at again...
    assert world.calls["text"] == []       # ...using the verdict it already has


def test_photo_checks_stop_at_three_per_scan(world):
    world.listings = [_listing(str(i)) for i in range(6)]
    run_pipeline()
    assert len(world.calls["image"]) == 3
    assert len(world.calls["text"]) == 6


def test_photo_checks_go_to_the_likeliest_hot_deals_first(world):
    # A worse deal found first must not take the only photo checks.
    world.listings = [_listing(str(i), price=500.0) for i in range(3)] + [_listing("best", price=300.0)]
    run_pipeline()
    assert world.calls["image"][0] == "best"


def test_listing_waiting_for_a_photo_gets_one_on_a_later_scan(world):
    world.listings = [_listing(str(i), price=480.0) for i in range(4)]
    run_pipeline()
    checked = set(world.calls["image"])
    assert len(checked) == 3
    _reset_calls(world)

    run_pipeline()
    assert world.calls["text"] == []
    assert world.calls["image"] and not set(world.calls["image"]) & checked


def test_no_photo_attempts_once_the_daily_limit_has_taken_the_keys(world):
    world.gemini_keys.clear()
    world.listings = [_listing("a")]
    run_pipeline()
    assert world.calls["image"] == []
    assert world.calls["text"] == ["a"]


def test_failed_photo_checks_do_not_use_the_allowance_but_are_bounded(world):
    world.image_fails = True
    world.listings = [_listing(str(i)) for i in range(6)]
    run_pipeline()
    assert len(world.calls["image"]) == 3


def test_full_scan_runs_again_after_four_hours(world):
    _rate_poorly(world)
    world.listings = [_listing("a")]
    run_pipeline()
    _make_next_scan_full()
    _reset_calls(world)

    run_pipeline()
    assert world.calls["window"] == 1 and world.calls["reconcile"] == 1
    assert world.calls["item"] == ["a"]    # re-scored against refreshed prices
    assert world.calls["text"] == []       # without paying for AI again


def test_auctions_outside_the_two_hour_window_are_ignored(world):
    now = datetime.now(timezone.utc)
    soon = (now + timedelta(minutes=90)).isoformat()
    later = (now + timedelta(hours=3)).isoformat()
    world.auctions = [
        _listing("soon", price=100.0, is_auction=True, auction_end=soon),
        _listing("later", price=100.0, is_auction=True, auction_end=later),
    ]
    run_pipeline()
    assert world.calls["text"] == ["soon"]


def test_no_report_per_scan_and_lost_state_only_means_a_full_scan(world):
    world.listings = [_listing("a")]
    run_pipeline()
    run_pipeline()
    assert world.reports == []
    assert src.scan_state.load_state()["report"]["scans"] == 2

    src.scan_state.STATE_PATH.unlink()
    _reset_calls(world)
    run_pipeline()
    assert world.calls["window"] == 1
