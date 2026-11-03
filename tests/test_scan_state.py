"""What one scan hands to the next: which scans are full, and what gets reviewed."""

from datetime import datetime, timedelta, timezone

from src.scan_state import (
    add_to_report, full_scan_due, load_state, prune_judged, record_verdict,
    restore_report, save_state, select_for_review, slim_window_rows, take_report_if_due,
)

NOW = datetime(2026, 11, 3, 12, 1, tzinfo=timezone.utc)


def _state(hours_ago):
    return {"last_full_at": (NOW - timedelta(hours=hours_ago)).isoformat(), "window_rows": [{"price": 1}]}


def _candidate(item_id, price=100.0):
    return {"listing": {"item_id": item_id}, "total_price": price}


def _ids(candidates):
    return [c["listing"]["item_id"] for c in candidates]


# ── Full or quick ──────────────────────────────────────────────────────

def test_first_scan_and_lost_state_are_full():
    assert full_scan_due({}, NOW, 4)
    assert full_scan_due({"last_full_at": "garbage", "window_rows": [{}]}, NOW, 4)
    # Prices can't be reused without the listings they come from.
    assert full_scan_due({"last_full_at": NOW.isoformat(), "window_rows": []}, NOW, 4)


def test_full_scan_every_four_hours_despite_late_starts():
    assert not full_scan_due(_state(1), NOW, 4)
    assert not full_scan_due(_state(3), NOW, 4)
    # Last one started at 08:02, this one at 12:01: still that slot's full scan.
    assert full_scan_due(_state(3.98), NOW, 4)


def test_state_survives_a_round_trip_and_bad_files_are_ignored(tmp_path):
    path = tmp_path / "nested" / "state.json"
    save_state({"judged": {"a": {"price": 5}}}, path)
    assert load_state(path) == {"judged": {"a": {"price": 5}}}
    assert load_state(tmp_path / "missing.json") == {}
    path.write_text("{broken", encoding="utf-8")
    assert load_state(path) == {}
    path.write_text("[1]", encoding="utf-8")
    assert load_state(path) == {}


def test_window_rows_keep_only_what_prices_need():
    rows = [{"id": "1", "product_key": "k", "condition_band": "used", "price": 5,
             "marketplace": "EBAY_IT", "status": "active", "last_seen": "x"}]
    assert slim_window_rows(rows) == [
        {"product_key": "k", "condition_band": "used", "price": 5, "marketplace": "EBAY_IT"}]


# ── What gets reviewed ─────────────────────────────────────────────────

def test_quick_scan_reviews_only_what_the_ai_has_not_judged():
    judged = {"old": {"price": 100}}
    picked = select_for_review(
        [_candidate("old"), _candidate("new")], judged, full=False, max_total=50, photo_slots=3)
    assert _ids(picked) == ["new"]


def test_price_drop_is_reviewed_again_but_a_rise_is_not():
    judged = {"dropped": {"price": 100}, "raised": {"price": 100}, "same": {"price": 100}}
    picked = select_for_review(
        [_candidate("dropped", 90), _candidate("raised", 110), _candidate("same", 100)],
        judged, full=False, max_total=50, photo_slots=3)
    assert _ids(picked) == ["dropped"]


def test_rising_auction_bid_is_not_reviewed_again():
    judged = {"auction": {"price": 200}}
    assert select_for_review(
        [_candidate("auction", 240)], judged, full=False, max_total=50, photo_slots=3) == []


def test_listings_awaiting_a_photo_are_limited_to_the_photo_slots():
    judged = {i: {"price": 100, "needs_photo": True} for i in ("a", "b", "c", "d")}
    candidates = [_candidate(i) for i in ("a", "b", "c", "d")] + [_candidate("new")]
    assert _ids(select_for_review(candidates, judged, False, 50, photo_slots=3)) == ["new", "a", "b", "c"]
    # Gemini is out for the day: looking at them again would achieve nothing.
    assert _ids(select_for_review(candidates, judged, False, 50, photo_slots=0)) == ["new"]


def test_full_scan_rescores_known_listings_but_new_ones_go_first():
    judged = {"old1": {"price": 100}, "old2": {"price": 100}}
    candidates = [_candidate("old1"), _candidate("old2"), _candidate("new")]
    assert _ids(select_for_review(candidates, judged, True, 50, 3)) == ["new", "old1", "old2"]
    # Old listings used to fill the limit and push out new ones.
    assert _ids(select_for_review(candidates, judged, True, 1, 3)) == ["new"]


def test_lost_verdicts_make_everything_new():
    candidates = [_candidate("a"), _candidate("b")]
    assert _ids(select_for_review(candidates, {}, False, 50, 3)) == ["a", "b"]


def test_verdict_records_the_search_price_and_old_ones_expire():
    judged = {}
    record_verdict(judged, _candidate("a", 99.999), NOW, needs_photo=True)
    assert judged["a"]["price"] == 100.0 and judged["a"]["needs_photo"] is True

    judged["old"] = {"price": 1, "at": (NOW - timedelta(days=31)).isoformat()}
    judged["bad"] = {"price": 1, "at": "garbage"}
    assert set(prune_judged(judged, NOW)) == {"a"}


# ── Daily report ───────────────────────────────────────────────────────

def test_report_adds_up_and_is_sent_once_a_day():
    state = {}
    stats = {"total_scanned": 100, "hot_deals": 1, "good_deals": 2, "photo_checks": 3, "errors": 0}
    add_to_report(state, stats, NOW)
    add_to_report(state, stats, NOW + timedelta(hours=1))
    assert take_report_if_due(state, NOW + timedelta(hours=2), 24) is None

    report = take_report_if_due(state, NOW + timedelta(hours=24), 24)
    assert report["scans"] == 2 and report["total_scanned"] == 200 and report["photo_checks"] == 6
    assert "report" not in state

    # The next period starts over.
    add_to_report(state, stats, NOW + timedelta(hours=25))
    assert state["report"]["scans"] == 1


def test_failed_report_is_kept():
    state = {}
    add_to_report(state, {"hot_deals": 4}, NOW)
    report = take_report_if_due(state, NOW + timedelta(hours=24), 24)
    restore_report(state, report)
    add_to_report(state, {"hot_deals": 1}, NOW + timedelta(hours=24))
    assert state["report"]["hot_deals"] == 5
