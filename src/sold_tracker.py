"""Longitudinal sold-price tracking — builds the proprietary resale dataset.

eBay's API won't tell us what an item *sold* for, so we infer it: record every
classifiable listing we see, then re-check the stale ones. When a listing
disappears (404), it ended — and for fixed-price (predominantly GTC) listings a
disappearance is a reasonable *proxy* for "sold". We bank the realized price,
the condition band, and how long it took to leave the market (a sell-through /
liquidity signal), then roll those comps up into compact per-bucket stats.

Storage discipline (free Supabase is 500 MB):
    • tracked_listings  — ONE row per listing, upserted (bounded by active set)
    • sold_comps        — finalized outcomes, kept for a rolling retention window
    • resale_stats      — one row per (product_key, condition_band), recomputed

Nothing here is consumed by the deal scorer yet — it's the data foundation that
has to start accruing now because it only matures with calendar time.
"""

import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.price_tracker import (
    PriceTracker, is_for_parts, condition_band, PRICED_BANDS,
)

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str) -> datetime:
    """Parse a Supabase ISO timestamp (handles trailing 'Z')."""
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


class SoldTracker:
    """Records listing sightings and finalizes realized prices over time."""

    def __init__(self, db, price_tracker: PriceTracker, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.db = db
        self.price_tracker = price_tracker
        self.enabled = cfg.get("enabled", True)
        self.stale_hours = cfg.get("stale_hours", 24)
        self.max_reconcile_checks = cfg.get("max_reconcile_checks", 120)
        self.max_track_days = cfg.get("max_track_days", 90)
        self.sold_max_days = cfg.get("sold_max_days", 60)
        self.retention_days = cfg.get("retention_days", 150)
        self.min_comps = cfg.get("min_comps", 3)
        self.stats_refresh_hours = cfg.get("stats_refresh_hours", 20)

    # ── Step A: record what we saw this scan ──────────────────────────

    def record_sightings(self, listings: list[dict]) -> int:
        """Upsert one row per classifiable, non-parts listing. Returns count."""
        if not self.enabled:
            return 0

        now_iso = _now().isoformat()
        records = []
        for it in listings:
            band = condition_band(it)
            # Refurb and parts are excluded from the resale dataset too, so the
            # realized-price stats stay comparable to the deal baselines.
            if is_for_parts(it) or band not in PRICED_BANDS:
                continue
            key, conf = self.price_tracker.generate_product_key(
                it.get("title", ""), it.get("category", ""),
                it.get("epid", ""), it.get("condition_id", ""),
            )
            if not key or conf < PriceTracker.MIN_CONFIDENCE:
                continue
            price = float(it.get("total_price") or it.get("price") or 0)
            if price <= 0:
                continue
            # NOTE: first_seen is intentionally omitted so the column DEFAULT
            # fires on insert and is preserved (not overwritten) on update.
            records.append({
                "id": it["item_id"],
                "product_key": key,
                "condition_band": band,
                "price": round(price, 2),
                "status": "active",
                "last_seen": now_iso,
                "last_checked": now_iso,
            })

        if records:
            self.db.upsert_tracked_listings(records)
        return len(records)

    # ── Step B: finalize listings that have ended ─────────────────────

    @staticmethod
    def build_comp(row: dict, ended_via_404: bool, sold_max_days: int) -> dict:
        """Pure: turn an ended tracked row into a sold_comp record."""
        first = _parse(row["first_seen"])
        last = _parse(row["last_seen"])
        days_active = max((last - first).days, 0)
        # Disappeared (404) within a plausible window => proxy "sold". An item
        # that outlived max_track_days without 404-ing is treated as uncertain.
        sold_proxy = bool(ended_via_404 and days_active <= sold_max_days)
        return {
            "listing_id": row["id"],
            "product_key": row["product_key"],
            "condition_band": row.get("condition_band"),
            "final_price": row.get("price"),
            "days_active": days_active,
            "sold_proxy": sold_proxy,
        }

    def reconcile_ended(self, ebay) -> int:
        """Re-check stale tracked listings; finalize the ones that ended.

        Bounded by max_reconcile_checks to protect the eBay daily quota.
        """
        if not self.enabled:
            return 0

        now = _now()
        stale_before = (now - timedelta(hours=self.stale_hours)).isoformat()
        rows = self.db.get_stale_tracked(stale_before, self.max_reconcile_checks)

        finalized = 0
        for row in rows:
            age_days = (now - _parse(row["first_seen"])).days
            active = ebay.item_is_active(row["id"])

            # Still active and not too old: reset its clock, keep tracking.
            if active is True and age_days < self.max_track_days:
                self.db.touch_tracked(row["id"], now.isoformat())
                continue
            # Transient/unknown and not too old: advance check cursor, retry later.
            if active is None and age_days < self.max_track_days:
                self.db.update_tracked_checked(row["id"], now.isoformat())
                continue

            # Ended (404) or exceeded the tracking horizon -> finalize.
            comp = self.build_comp(row, ended_via_404=(active is False), sold_max_days=self.sold_max_days)
            self.db.finalize_tracked(row["id"], comp)
            finalized += 1

        return finalized

    # ── Step C: roll comps up into compact per-bucket stats ───────────

    def rebuild_resale_stats(self, force: bool = False) -> int:
        """Recompute resale_stats from the retained sold comps (gated ~daily)."""
        if not self.enabled:
            return 0
        if not force:
            age = self.db.get_resale_stats_age_hours()
            if age is not None and age < self.stats_refresh_hours:
                return 0

        since = (_now() - timedelta(days=self.retention_days)).isoformat()
        comps = self.db.get_sold_comps_since(since)  # sold_proxy=True only

        groups: dict[tuple[str, str], list[dict]] = {}
        for c in comps:
            band = c.get("condition_band") or "unknown"
            groups.setdefault((c["product_key"], band), []).append(c)

        written = 0
        for (product_key, band), items in groups.items():
            prices = sorted(float(i["final_price"]) for i in items if i.get("final_price") is not None)
            if len(prices) < self.min_comps:
                continue
            days = [int(i["days_active"]) for i in items if i.get("days_active") is not None]

            if len(prices) >= 4:
                q = statistics.quantiles(prices, n=4)
                p25, p75 = round(q[0], 2), round(q[2], 2)
            else:
                p25, p75 = prices[0], prices[-1]

            self.db.upsert_resale_stat({
                "product_key": product_key,
                "condition_band": band,
                "median_price": round(statistics.median(prices), 2),
                "p25_price": p25,
                "p75_price": p75,
                "sample_size": len(prices),
                "median_days_to_sell": round(statistics.median(days), 1) if days else None,
            })
            written += 1

        logger.info("Rebuilt resale_stats for %d buckets", written)
        return written

    # ── Step D: retention — keep the DB flat under the 500 MB cap ─────

    def cleanup(self) -> None:
        """Delete aged rows so storage stays bounded regardless of runtime."""
        if not self.enabled:
            return
        now = _now()
        cut = lambda days: (now - timedelta(days=days)).isoformat()  # noqa: E731

        # Rolling window for the raw comps (resale_stats is the durable summary).
        self.db.delete_rows_older_than("sold_comps", "ended_at", cut(self.retention_days))
        # Existing append-only tables that previously had no retention at all.
        self.db.delete_rows_older_than("analyses", "analyzed_at", cut(90))
        self.db.delete_rows_older_than("image_sellers", "first_seen_at", cut(90))
        self.db.delete_rows_older_than("price_history", "recorded_at", cut(90))
        self.db.delete_rows_older_than("listings", "last_seen_at", cut(120))
        # Safety net for any tracked row reconcile never got around to.
        self.db.delete_rows_older_than("tracked_listings", "last_checked", cut(self.max_track_days + 30))
