"""Digests, feedback buttons, scan health, alert outcomes and exit prices."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.deal_scorer import exit_signal, find_exit_mismatches
from src.feedback import handle_feedback, mark_choice, parse_callback
from src.health import build_status, warning_due, warning_text
from src.scan_state import digest_due, slim_deal
from src.sold_tracker import SoldTracker
from src.telegram_notifier import MAX_CALLBACK_BYTES, MAX_DIGEST_ROWS, TelegramNotifier

NOW = datetime(2026, 11, 3, 12, 1, tzinfo=timezone.utc)
HOURS = [7, 12, 18]
ITEM = "v1|335678901234|0"


# ── Digest timing ──────────────────────────────────────────────────────

def _last(**delta):
    return {"last_digest_at": (NOW - timedelta(**delta)).isoformat()}


def test_digest_due_after_a_slot_passes():
    assert digest_due(_last(hours=5), NOW, HOURS)        # sent 07:01, now 12:01
    assert digest_due(_last(minutes=30), NOW, HOURS)     # sent 11:31, 12:00 has passed
    assert not digest_due(_last(seconds=30), NOW, HOURS)  # sent 12:00:30, this slot's done


def test_digest_waits_between_slots_and_catches_up_after_a_missed_one():
    afternoon = NOW.replace(hour=15)
    assert not digest_due({"last_digest_at": NOW.isoformat()}, afternoon, HOURS)
    # Before the first slot of the day, yesterday's last slot counts.
    early = NOW.replace(hour=2)
    assert digest_due({"last_digest_at": (early - timedelta(hours=20)).isoformat()}, early, HOURS)


def test_no_digest_without_a_start_time_or_hours():
    assert not digest_due({}, NOW, HOURS)
    assert not digest_due(_last(days=2), NOW, [])


def test_slim_deal_keeps_what_a_digest_row_needs():
    deal = {"listing": {"item_id": ITEM, "title": "x", "price": 1, "total_price": 2, "item_url": "u",
                        "negotiable": True, "marketplace": "EBAY_IT", "short_description": "long " * 500},
            "analysis": {"photo_type": "stock", "reasoning": "long"}, "scores": {"deal_score": 70}}
    slim = slim_deal(deal)
    assert "short_description" not in slim["listing"]
    assert slim["analysis"] == {"photo_type": "stock"}
    assert slim["scores"]["deal_score"] == 70


# ── Buttons on alerts ──────────────────────────────────────────────────

@pytest.fixture
def notifier(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    n = TelegramNotifier()
    n.sent = []
    n._send_message = lambda text, parse_mode="HTML", reply_markup=None: n.sent.append((text, reply_markup)) or True
    return n


def _deal(item_id, score=70, key=None):
    return {"listing": {"item_id": item_id, "title": f"t{item_id}", "total_price": 100, "item_url": "u"},
            "analysis": {}, "scores": {"deal_score": score, "product_key": key or f"k{item_id}", "discount_pct": 20}}


def test_hot_alert_has_one_row_of_buttons(notifier):
    notifier.send_hot_deal_alert({"item_id": ITEM, "title": "t", "total_price": 100}, {}, {"deal_score": 85})
    rows = notifier.sent[0][1]["inline_keyboard"]
    assert len(rows) == 1
    assert [b["callback_data"] for b in rows[0]] == [f"fb:up:{ITEM}", f"fb:down:{ITEM}", f"fb:bought:{ITEM}"]


def test_digest_rows_match_their_buttons_and_stop_at_the_cap(notifier):
    deals = [_deal(str(i), score=99 - i) for i in range(MAX_DIGEST_ROWS + 5)]
    notifier.send_deal_summary(deals)
    text, markup = notifier.sent[0]
    assert len(markup["inline_keyboard"]) == MAX_DIGEST_ROWS
    assert f"best {MAX_DIGEST_ROWS} shown" in text
    assert markup["inline_keyboard"][0][0]["text"] == "1 👍"
    assert markup["inline_keyboard"][0][0]["callback_data"] == "fb:up:0"
    assert "t0" in text and f"t{MAX_DIGEST_ROWS + 4}" not in text


def test_an_id_too_long_for_a_button_is_left_out(notifier):
    assert notifier._feedback_row("x" * MAX_CALLBACK_BYTES) == []


def test_buttons_only_on_the_last_part_of_a_long_message(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    n = TelegramNotifier()
    payloads = []
    n.session = SimpleNamespace(post=lambda url, json, timeout: payloads.append(json) or SimpleNamespace(status_code=200))
    monkeypatch.setattr("src.telegram_notifier.time.sleep", lambda s: None)
    assert n._send_message("\n".join(["x" * 100] * 80), reply_markup={"inline_keyboard": []})
    assert len(payloads) > 1
    assert "reply_markup" in payloads[-1] and all("reply_markup" not in p for p in payloads[:-1])


# ── Pressing a button ──────────────────────────────────────────────────

class FakeNotifier:
    def __init__(self):
        self.answers, self.edits = [], []

    def answer_callback(self, callback_id, text):
        self.answers.append(text)

    def edit_reply_markup(self, chat_id, message_id, markup):
        self.edits.append(markup)


def _callback(data=f"fb:up:{ITEM}", chat=42, sender=42):
    row = [{"text": "👍", "callback_data": f"fb:up:{ITEM}"}, {"text": "👎", "callback_data": f"fb:down:{ITEM}"}]
    return {"id": "1", "data": data, "from": {"id": sender},
            "message": {"message_id": 5, "chat": {"id": chat}, "reply_markup": {"inline_keyboard": [row]}}}


@pytest.mark.parametrize("data,expected", [
    (f"fb:up:{ITEM}", ("up", ITEM)),
    (f"fb:bought:{ITEM}", ("bought", ITEM)),
    (f"fb:meh:{ITEM}", None),
    ("fb:up:", None),
    ("other", None),
    ("", None),
])
def test_parse_callback(data, expected):
    assert parse_callback(data) == expected


def test_press_is_saved_and_ticked():
    saved, notifier = [], FakeNotifier()
    assert handle_feedback(_callback(), lambda i, r: saved.append((i, r)) or True, notifier, "42")
    assert saved == [(ITEM, "up")]
    assert notifier.answers == ["Saved, thanks"]
    assert notifier.edits[0]["inline_keyboard"][0][0]["text"] == "✅ 👍"


def test_changing_your_mind_moves_the_tick_only_in_that_row():
    other_row = [{"text": "✅ 2 👍", "callback_data": "fb:up:2"}, {"text": "2 👎", "callback_data": "fb:down:2"}]
    first_row = [{"text": "✅ 1 👍", "callback_data": "fb:up:1"}, {"text": "1 👎", "callback_data": "fb:down:1"}]
    out = mark_choice({"inline_keyboard": [first_row, other_row]}, "fb:down:1")["inline_keyboard"]
    assert [b["text"] for b in out[0]] == ["1 👍", "✅ 1 👎"]
    assert [b["text"] for b in out[1]] == ["✅ 2 👍", "2 👎"]


@pytest.mark.parametrize("callback", [_callback(chat=99), _callback(sender=99)])
def test_strangers_cannot_rate(callback):
    saved, notifier = [], FakeNotifier()
    assert not handle_feedback(callback, lambda i, r: saved.append(1) or True, notifier, "42")
    assert saved == [] and notifier.answers == []


def test_failed_save_says_so_and_keeps_the_buttons():
    notifier = FakeNotifier()
    assert not handle_feedback(_callback(), lambda i, r: False, notifier, "42")
    assert "Couldn't save" in notifier.answers[0]
    assert notifier.edits == []


# ── Scan health ────────────────────────────────────────────────────────

def test_no_warning_while_scans_run_and_reminders_every_six_hours():
    assert not warning_due(0.5, NOW)
    assert not warning_due(2.9, NOW)
    assert warning_due(3.2, NOW)
    assert not warning_due(4.2, NOW)
    assert not warning_due(8.9, NOW)
    assert warning_due(9.1, NOW)


def test_exactly_one_warning_per_six_hours_of_hourly_pings():
    ages = [3 + i + 0.5 for i in range(24)]
    assert sum(warning_due(a, NOW) for a in ages) == 4


def test_unreadable_database_warns_every_six_hours_not_every_ping():
    assert warning_due(None, NOW.replace(hour=12))
    assert not warning_due(None, NOW.replace(hour=13))
    assert "can't read" in warning_text(None)
    assert "No scan for 5 hours" in warning_text(5.1)


class StatusDb:
    def __init__(self, last_scan):
        self.last_scan = last_scan

    def get_alerted_since(self, since):
        return {"a": {"deal_score": 85}, "b": {"deal_score": 66}}

    def get_last_scan_time(self):
        return self.last_scan

    def get_key_cooldown(self, key_hash):
        return None

    def get_feedback_counts(self):
        return {"up": 3, "bought": 1}


def test_status_reply(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_1", "k")
    text = build_status(StatusDb(NOW - timedelta(minutes=20)), NOW)
    assert "✅ Last scan: 20 min ago" in text
    assert "2 (1 hot)" in text
    assert "👍 3 👎 0 🛒 1" in text


def test_status_flags_a_stopped_scanner():
    assert "⚠️ Last scan: 5.0 hours ago" in build_status(StatusDb(NOW - timedelta(hours=5)), NOW)
    assert "Can't read" in build_status(StatusDb(None), NOW)


# ── Alerted listings first ─────────────────────────────────────────────

class TrackerDb:
    def __init__(self, alerted, rows, ended=None):
        self.alerted, self.rows, self.ended = alerted, rows, ended or {}
        self.touched = []

    def get_alerted_since(self, since):
        return self.alerted

    def get_tracked_rows(self, ids):
        return [r for r in self.rows if r["id"] in ids]

    def get_ended_times(self, ids):
        return {i: t for i, t in self.ended.items() if i in ids}

    def touch_tracked(self, item_id, ts):
        self.touched.append(item_id)

    def update_tracked_checked(self, item_id, ts):
        pass

    def finalize_tracked(self, item_id, comp):
        self.touched.append(("ended", item_id))


class FakeEbay:
    def item_is_active(self, item_id, marketplace=None):
        return item_id != "gone"


def _ago(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


def _row(item_id, **checked):
    return {"id": item_id, "first_seen": _ago(days=2), "last_seen": _ago(days=1),
            "last_checked": _ago(**checked), "product_key": "k", "price": 1}


def test_only_alerted_listings_that_are_due_get_checked():
    alerted = {"new": {"analyzed_at": _ago(hours=2)}, "new_seen": {"analyzed_at": _ago(hours=2)},
               "old": {"analyzed_at": _ago(days=3)}, "old_recent": {"analyzed_at": _ago(days=3)}}
    rows = [_row("new", hours=4), _row("new_seen", hours=1), _row("old", hours=25), _row("old_recent", hours=5)]
    db = TrackerDb(alerted, rows)
    db.checked = []
    db.update_tracked_checked = lambda item_id, ts: db.checked.append(item_id)
    SoldTracker(db, None, {}).reconcile_alerted(FakeEbay())
    assert db.checked == ["new", "old"]


def test_rechecking_an_alert_does_not_put_it_back_in_the_market_window():
    db = TrackerDb({"new": {"analyzed_at": _ago(hours=2)}}, [_row("new", hours=4)])
    db.checked = []
    db.update_tracked_checked = lambda item_id, ts: db.checked.append(item_id)
    SoldTracker(db, None, {}).reconcile_alerted(FakeEbay())
    assert db.checked == ["new"]
    assert db.touched == []      # touching would reset last_seen


def test_checks_stop_at_the_cap_newest_alerts_first_and_ended_ones_finalize():
    alerted = {"older": {"analyzed_at": _ago(hours=10)}, "gone": {"analyzed_at": _ago(hours=1)}}
    rows = [_row("older", hours=5), _row("gone", hours=5)]
    db = TrackerDb(alerted, rows)
    assert SoldTracker(db, None, {"max_alert_checks": 1}).reconcile_alerted(FakeEbay()) == 1
    assert db.touched == [("ended", "gone")]


def test_outcomes_only_count_settled_tracked_alerts():
    alerted = {
        "fast": {"analyzed_at": _ago(days=4)},
        "day": {"analyzed_at": _ago(days=4)},
        "alive": {"analyzed_at": _ago(days=4)},
        "too_new": {"analyzed_at": _ago(days=1)},
        "auction": {"analyzed_at": _ago(days=4)},   # never tracked, so unknown
    }
    t0 = datetime.now(timezone.utc) - timedelta(days=4)
    ended = {"fast": (t0 + timedelta(hours=3)).isoformat(), "day": (t0 + timedelta(hours=20)).isoformat()}
    db = TrackerDb(alerted, [_row("alive", hours=1), _row("too_new", hours=1)], ended)
    assert SoldTracker(db, None, {}).alert_outcomes() == {
        "alerts": 3, "within_6h": 1, "within_24h": 2, "within_72h": 2}


def test_no_outcomes_when_nothing_has_settled():
    db = TrackerDb({"a": {"analyzed_at": _ago(hours=5)}}, [_row("a", hours=1)])
    assert SoldTracker(db, None, {}).alert_outcomes() is None


# ── Exit prices ────────────────────────────────────────────────────────

def _stat(price, n=10, days=9):
    return {"median_price": price, "sample_size": n, "median_days_to_sell": days}


def test_exit_signal_needs_enough_listings():
    assert exit_signal(None, 500) == {}
    assert exit_signal(_stat(500, n=4), 500) == {}
    assert exit_signal(_stat(None), 500) == {}
    assert exit_signal(_stat(500), 0) == {}
    assert exit_signal(_stat(480), 500) == {"exit_price": 480.0, "exit_n": 10, "exit_days": 9, "exit_mismatch": False}


@pytest.mark.parametrize("price,mismatch", [(370, True), (380, False), (690, False), (710, True)])
def test_exit_mismatch_limits(price, mismatch):
    assert exit_signal(_stat(price), 500)["exit_mismatch"] is mismatch


def test_alert_shows_exit_price_and_warns_on_a_mismatch(notifier):
    listing = {"item_id": ITEM, "title": "t", "total_price": 100}
    notifier.send_hot_deal_alert(listing, {}, {"deal_score": 85, **exit_signal(_stat(480), 500)})
    assert "left eBay at ~€480 (10 listings, typically after 9 days)" in notifier.sent[0][0]
    assert "not a confirmed sale" in notifier.sent[0][0]
    assert "misread" not in notifier.sent[0][0]

    notifier.send_hot_deal_alert(listing, {}, {"deal_score": 85, **exit_signal(_stat(300), 500)})
    assert "misread" in notifier.sent[1][0]


def test_alert_without_exit_data_says_nothing(notifier):
    notifier.send_hot_deal_alert({"item_id": ITEM, "title": "t", "total_price": 100}, {}, {"deal_score": 85})
    assert "left eBay" not in notifier.sent[0][0]


def test_worst_mismatches_come_first_and_thin_markets_are_ignored():
    market = lambda value, n=20: SimpleNamespace(value=value, n=n)   # noqa: E731
    prices = {"a#used": market(100), "b#used": market(100), "c#used": market(100), "d#used": market(100, n=3)}
    stats = {("a", "used"): _stat(70), ("b", "used"): _stat(200), ("c", "used"): _stat(100), ("d", "used"): _stat(10),
             ("e", "used"): _stat(10)}
    found = find_exit_mismatches(prices, stats)
    assert [m["product_key"] for m in found] == ["b", "a"]


def test_daily_report_shows_the_new_sections(notifier):
    notifier.send_daily_report(
        {"scans": 2, "total_scanned": 100, "hot_deals": 1, "ebay_calls": 3000},
        outcomes={"alerts": 10, "within_6h": 2, "within_24h": 5, "within_72h": 6},
        feedback={"up": 1, "down": 2},
        mismatches=[{"product_key": "ps5", "condition_band": "used", "exit_price": 200, "market": 400}],
    )
    text = notifier.sent[0][0]
    assert "eBay calls: 3,000" in text
    assert "Gone within 24h: 5 (50%)" in text
    assert "👍 1 👎 2 🛒 0" in text
    assert "ps5 (used): left at ~€200, market €400" in text


# ── Bot and health endpoint ────────────────────────────────────────────

def _message(text, chat=42):
    return {"message": {"text": text, "chat": {"id": chat}, "date": 1000}}


def test_status_command_is_owner_only():
    from src.deal_check import handle_update

    sent = []
    send = lambda text: sent.append(text) or True   # noqa: E731
    handle_update(_message("/status"), None, send, "42", now=1000, status=lambda: "all good")
    assert sent == ["all good"]
    handle_update(_message("/status", chat=99), None, send, "42", now=1000, status=lambda: "secret")
    assert sent == ["all good"]


def test_health_endpoint_needs_the_secret(monkeypatch):
    import importlib.util
    import pathlib
    import threading
    import urllib.error
    import urllib.request
    from http.server import HTTPServer

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("health_check", root / "api" / "health.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "_check", lambda: calls.append(1) or False)
    httpd = HTTPServer(("127.0.0.1", 0), module.handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}/api/health"

    def get(header=None):
        req = urllib.request.Request(url)
        if header:
            req.add_header("Authorization", header)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, b""

    try:
        monkeypatch.delenv("HEALTH_CHECK_SECRET", raising=False)
        assert get("Bearer anything")[0] == 401      # no secret set: closed, not open
        monkeypatch.setenv("HEALTH_CHECK_SECRET", "s3cret")
        assert get()[0] == 401
        assert get("Bearer wrong")[0] == 401
        assert calls == []
        assert get("Bearer s3cret") == (200, b"stale")
        assert calls == [1]
    finally:
        httpd.shutdown()


# ── Delivery to Italy ──────────────────────────────────────────────────

_DE = {"item_id": ITEM, "title": "t", "price": 102, "total_price": 108, "shipping_cost": 6}


def test_no_delivery_is_flagged_with_the_local_price_only(notifier):
    notifier.send_hot_deal_alert(_DE, {}, {"deal_score": 85, "ships_to_italy": False, "is_foreign": True})
    text = notifier.sent[0][0]
    assert "No delivery to Italy listed" in text and "€102 + €6 local delivery" in text
    assert "friend" in text and "forwarding" in text
    assert "Ships to Italy" not in text and "ships to Italy" not in text


def test_delivery_to_italy_shows_the_sum(notifier):
    listing = {**_DE, "price": 200, "total_price": 230, "shipping_cost": 30}
    notifier.send_hot_deal_alert(listing, {}, {"deal_score": 85, "ships_to_italy": True})
    assert "Ships to Italy: €200 + €30 delivery = <b>€230</b>" in notifier.sent[0][0]


def test_free_delivery_and_unknown_delivery(notifier):
    notifier.send_hot_deal_alert({**_DE, "shipping_cost": 0}, {}, {"deal_score": 85, "ships_to_italy": True})
    assert "🚚 Ships to Italy" in notifier.sent[0][0] and " + €0" not in notifier.sent[0][0]
    # The lookup failed: don't claim either way.
    notifier.send_hot_deal_alert(_DE, {}, {"deal_score": 85, "ships_to_italy": None, "is_foreign": True})
    assert "Ships to Italy" not in notifier.sent[1][0] and "No delivery" not in notifier.sent[1][0]


def test_digest_marks_rows_and_prefers_a_listing_that_ships_on_a_tie(notifier):
    def deal(item_id, ships):
        d = _deal(item_id, key="same")
        d["scores"]["ships_to_italy"] = ships
        return d

    notifier.send_deal_summary([deal("no", False), deal("yes", True)])
    text = notifier.sent[0][0]
    assert "tyes" in text and "tno" not in text
    notifier.send_deal_summary([deal("no", False)])
    assert "🚫" in notifier.sent[1][0].split("OTHER SELLERS")[1].splitlines()[1]


def test_item_lookup_asks_as_a_buyer_in_italy():
    from src.ebay_client import BUYER_CONTEXT, EbayClient

    client = EbayClient.__new__(EbayClient)
    client.marketplace = "EBAY_IT"
    client._get_access_token = lambda: "tok"
    assert "X-EBAY-C-ENDUSERCTX" not in client._get_headers("EBAY_DE")
    assert client._get_headers("EBAY_DE", buyer_in_italy=True)["X-EBAY-C-ENDUSERCTX"] == BUYER_CONTEXT
    assert "country%3DIT" in BUYER_CONTEXT


@pytest.mark.parametrize("options,ships,cost", [
    ([], False, None),
    ([{"shippingCost": {"value": "29.99"}}, {"shippingCost": {"value": "19.99"}}], True, 19.99),
    ([{"shippingCostType": "CALCULATED"}], True, None),
])
def test_item_details_read_delivery_to_italy(options, ships, cost):
    from src.ebay_client import EbayClient

    client = EbayClient.__new__(EbayClient)
    client.marketplace = "EBAY_IT"
    client._api_get = lambda *a, **k: {"shippingOptions": options, "price": {"value": "100"}}
    details = client.get_item_details("x", marketplace="EBAY_DE")
    assert details["ships_to_buyer"] is ships
    assert details["shipping_to_buyer"] == cost


def test_foreign_searches_stay_in_that_country_without_a_delivery_filter():
    from src.ebay_client import EbayClient

    client = EbayClient.__new__(EbayClient)
    client.marketplace = "EBAY_IT"
    sweeps = client._sweeps(["EBAY_DE"], "IT")
    assert sweeps[0] == ("EBAY_IT", "IT", None, False)
    assert sweeps[1] == ("EBAY_DE", "DE", None, True)
