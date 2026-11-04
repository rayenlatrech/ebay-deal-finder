"""What one scan hands to the next.

Scans run hourly, but most of a scan's work doesn't need to be redone that
often: market prices barely move in an hour, and a listing already judged at
its current price will get the same verdict. The state file records that work
so a quick scan can skip it. GitHub Actions' cache carries it between runs.

Losing the file is always safe: the next scan is simply a full one.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

STATE_PATH = Path(os.environ.get("SCAN_STATE_FILE", ".scan_state/state.json"))

# A scan starts up to a few minutes late, so a full scan due every 4h must not
# miss its slot because the last one started at 08:01 and this one at 12:00.
FULL_SCAN_SLACK_HOURS = 0.5

# Forget a verdict after this long: analyses are reused for 30 days too.
JUDGED_KEEP_DAYS = 30

REPORT_COUNTERS = (
    "total_scanned", "hot_deals", "good_deals", "ships_to_italy", "no_delivery",
    "photo_checks", "ebay_calls", "failed_searches", "errors",
)

_DIGEST_LISTING_FIELDS = ("item_id", "title", "price", "total_price", "item_url", "negotiable", "marketplace")


def load_state(path: Optional[Path] = None) -> dict[str, Any]:
    """The previous scan's state, or an empty one if it's missing or unreadable."""
    try:
        state = json.loads(Path(path or STATE_PATH).read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError) as e:
        logger.info("No usable scan state (%s) — starting fresh", type(e).__name__)
        return {}


def save_state(state: dict[str, Any], path: Optional[Path] = None) -> None:
    """Write the state for the next scan. A failure only costs that scan's savings."""
    target = Path(path or STATE_PATH)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(state, separators=(",", ":"), default=str), encoding="utf-8")
    except OSError as e:
        logger.error("Failed to save scan state: %s", e)


def _parse(timestamp: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None


def full_scan_due(state: dict[str, Any], now: datetime, every_hours: float) -> bool:
    """True when market prices and resale tracking are due for a refresh."""
    last = _parse(state.get("last_full_at"))
    if last is None or not state.get("window_rows"):
        return True
    return (now - last).total_seconds() / 3600 >= every_hours - FULL_SCAN_SLACK_HOURS


def slim_window_rows(rows: list[dict]) -> list[dict]:
    """Only the columns market prices are computed from."""
    keep = ("product_key", "condition_band", "price", "marketplace")
    return [{k: r.get(k) for k in keep} for r in rows]


def select_for_review(
    candidates: list[dict],
    judged: dict[str, dict],
    full: bool,
    max_total: int,
    photo_slots: int,
) -> list[dict]:
    """Which price-gated candidates to look at this scan, in that order.

    "New" means the AI hasn't judged this listing at this price: never reviewed,
    or cheaper than when it was. That's deliberately not "never seen", so a
    listing that first looked too expensive and later dropped its price still
    gets reviewed, and so does one caught by a crashed scan.

    A quick scan takes only those, plus a few listings that could still alert if
    a photo check confirmed them. A full scan also re-scores everything else,
    because the market prices it just refreshed can change a verdict, but new
    listings go first so old ones can't use up the limit.

    Args:
        candidates: Already ranked best first, each with the ``total_price`` the
            search returned. That price is compared rather than the item lookup's,
            so a small difference between the two can't look like a price drop.
        judged: Verdicts from earlier scans, keyed by item id.
    """
    new, awaiting_photo, known = [], [], []
    for candidate in candidates:
        entry = judged.get(candidate["listing"]["item_id"])
        if entry is None or candidate["total_price"] < float(entry.get("price") or 0) - 0.005:
            new.append(candidate)
        elif entry.get("needs_photo"):
            awaiting_photo.append(candidate)
        else:
            known.append(candidate)

    if full:
        return (new + awaiting_photo + known)[:max_total]
    return new[:max_total] + awaiting_photo[:photo_slots]


def record_verdict(judged: dict[str, dict], candidate: dict, now: datetime, needs_photo: bool = False) -> None:
    """Remember that this listing has been judged at this price."""
    judged[candidate["listing"]["item_id"]] = {
        "price": round(candidate["total_price"], 2),
        "at": now.isoformat(),
        "needs_photo": needs_photo,
    }


def prune_judged(judged: dict[str, dict], now: datetime) -> dict[str, dict]:
    """Drop old verdicts so the file stays small."""
    cutoff = now - timedelta(days=JUDGED_KEEP_DAYS)
    kept = {}
    for item_id, entry in judged.items():
        at = _parse(entry.get("at"))
        if at is not None and at >= cutoff:
            kept[item_id] = entry
    return kept


def slim_deal(deal: dict) -> dict:
    """The parts of a good deal its digest row needs, to wait in the state file."""
    return {
        "listing": {k: deal["listing"].get(k) for k in _DIGEST_LISTING_FIELDS},
        "analysis": {"photo_type": deal.get("analysis", {}).get("photo_type", "")},
        "scores": deal["scores"],
    }


def digest_due(state: dict[str, Any], now: datetime, hours_utc: list[int]) -> bool:
    """True once a digest time has passed since the last digest went out."""
    last = _parse(state.get("last_digest_at"))
    if last is None or not hours_utc:
        return False
    slots = []
    for hour in hours_utc:
        slot = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        slots.append(slot if slot <= now else slot - timedelta(days=1))
    return last < max(slots)


def add_to_report(state: dict[str, Any], stats: dict[str, Any], now: datetime) -> None:
    """Add this scan's numbers to the running totals for the next report."""
    report = state.setdefault("report", {"since": now.isoformat(), "scans": 0})
    report["scans"] = report.get("scans", 0) + 1
    for counter in REPORT_COUNTERS:
        report[counter] = report.get(counter, 0) + int(stats.get(counter, 0) or 0)


def take_report_if_due(state: dict[str, Any], now: datetime, every_hours: float) -> Optional[dict]:
    """The accumulated report if one is due, clearing it from the state.

    Hand it back with restore_report if it couldn't be sent, so a failed send
    doesn't lose the day's numbers.
    """
    report = state.get("report")
    since = _parse((report or {}).get("since"))
    if since is None:
        return None
    # Slack for the same reason as full scans: it would otherwise slip an hour.
    if (now - since).total_seconds() / 3600 < every_hours - FULL_SCAN_SLACK_HOURS:
        return None
    return state.pop("report")


def restore_report(state: dict[str, Any], report: dict) -> None:
    """Put back a report that failed to send."""
    state["report"] = report
