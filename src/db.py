"""Supabase database wrapper for storing listings, analyses, and price history."""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from supabase import create_client, Client

logger = logging.getLogger(__name__)


class Database:
    """Wrapper around Supabase for all database operations."""

    def __init__(self) -> None:
        """Initialize Supabase client from environment variables."""
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        self.client: Client = create_client(url, key)
        logger.info("Database connected to Supabase")

    # ── Listings ──────────────────────────────────────────────────────

    def upsert_listing(self, listing: dict[str, Any]) -> None:
        """Insert or update a listing record."""
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": listing["item_id"],
            "title": listing["title"],
            "price": listing["price"],
            "currency": listing.get("currency", "EUR"),
            "condition": listing.get("condition"),
            "category": listing.get("category"),
            "seller_score": listing.get("seller_feedback_score"),
            "seller_pct": listing.get("seller_feedback_percentage"),
            "negotiable": listing.get("negotiable", False),
            "image_url": listing.get("image_url"),
            "listing_url": listing.get("item_url"),
            "marketplace": listing.get("marketplace", "EBAY_IT"),
            "last_seen_at": now,
        }
        try:
            self.client.table("listings").upsert(
                record, on_conflict="id"
            ).execute()
        except Exception as e:
            logger.error("Failed to upsert listing %s: %s", listing["item_id"], e)

    def get_listing(self, item_id: str) -> Optional[dict]:
        """Get a listing by eBay item ID. Returns None if not found."""
        try:
            resp = (
                self.client.table("listings")
                .select("*")
                .eq("id", item_id)
                .maybe_single()
                .execute()
            )
            return resp.data
        except Exception as e:
            logger.error("Failed to get listing %s: %s", item_id, e)
            return None

    # ── Analyses ─────────────────────────────────────────────────────

    def save_analysis(self, analysis: dict[str, Any]) -> None:
        """Store an AI analysis result for a listing."""
        record = {
            "listing_id": analysis["listing_id"],
            "deal_score": analysis.get("deal_score"),
            "price_score": analysis.get("price_score"),
            "condition_score": analysis.get("condition_score"),
            "seller_score": analysis.get("seller_score"),
            "image_score": analysis.get("image_score"),
            "ai_assessment": analysis.get("ai_assessment"),
            "alerted": analysis.get("alerted", False),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.client.table("analyses").insert(record).execute()
        except Exception as e:
            logger.error("Failed to save analysis for %s: %s", analysis["listing_id"], e)

    def was_alerted(self, item_id: str) -> bool:
        """Check if an alert was already sent for this listing."""
        try:
            resp = (
                self.client.table("analyses")
                .select("alerted")
                .eq("listing_id", item_id)
                .eq("alerted", True)
                .limit(1)
                .execute()
            )
            return len(resp.data) > 0
        except Exception as e:
            logger.error("Failed to check alert status for %s: %s", item_id, e)
            return False

    def get_recent_analyses(self, item_ids: list[str], since_iso: str) -> dict[str, dict]:
        """Recent AI assessments for these listings, keyed by listing id.

        A listing's description and photos don't change, so re-running the AI on
        the same item every scan burns quota for an answer we already have.
        Listings can sit on eBay for weeks, so this is the difference between
        analysing an item once and analysing it dozens of times.
        """
        out: dict[str, dict] = {}
        unique = list({i for i in item_ids if i})
        CHUNK = 200
        for start in range(0, len(unique), CHUNK):
            chunk = unique[start:start + CHUNK]
            try:
                resp = (
                    self.client.table("analyses")
                    .select("listing_id, ai_assessment, analyzed_at")
                    .in_("listing_id", chunk)
                    .gte("analyzed_at", since_iso)
                    .order("analyzed_at", desc=True)
                    .execute()
                )
                for row in (resp.data or []):
                    # Ordered newest-first, so keep the first hit per listing.
                    lid = row["listing_id"]
                    if lid not in out and row.get("ai_assessment"):
                        out[lid] = row["ai_assessment"]
            except Exception as e:
                logger.error("Failed to load cached analyses: %s", e)
        return out

    def get_alerted_ids(self, item_ids: list[str]) -> set[str]:
        """Which of these listings have already been alerted on.

        Batched (chunked) so a scan costs a couple of queries instead of one
        round-trip per listing.
        """
        found: set[str] = set()
        unique = list({i for i in item_ids if i})
        CHUNK = 200
        for start in range(0, len(unique), CHUNK):
            chunk = unique[start:start + CHUNK]
            try:
                resp = (
                    self.client.table("analyses")
                    .select("listing_id")
                    .in_("listing_id", chunk)
                    .eq("alerted", True)
                    .execute()
                )
                found.update(r["listing_id"] for r in (resp.data or []))
            except Exception as e:
                logger.error("Failed to batch-check alert status: %s", e)
        return found

    def mark_alerted(self, item_id: str) -> None:
        """Mark the most recent analysis for a listing as alerted."""
        try:
            self.client.table("analyses").update({"alerted": True}).eq(
                "listing_id", item_id
            ).execute()
        except Exception as e:
            logger.error("Failed to mark alerted for %s: %s", item_id, e)

    # ── Price History ────────────────────────────────────────────────

    def get_price_history(self, product_key: str) -> list[dict]:
        """Get historical price data for a product key."""
        try:
            resp = (
                self.client.table("price_history")
                .select("*")
                .eq("product_key", product_key)
                .order("recorded_at", desc=True)
                .limit(30)
                .execute()
            )
            return resp.data or []
        except Exception as e:
            logger.error("Failed to get price history for %s: %s", product_key, e)
            return []

    def save_price_point(
        self, product_key: str, avg_price: float, sample_size: int
    ) -> None:
        """Record a market price data point."""
        record = {
            "product_key": product_key,
            "avg_price": avg_price,
            "sample_size": sample_size,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.client.table("price_history").insert(record).execute()
        except Exception as e:
            logger.error("Failed to save price point for %s: %s", product_key, e)

    # ── Image Deduplication (Stock Photo Detection) ──────────────────

    def count_sellers_for_image(self, image_id: str) -> int:
        """Count how many different sellers use the same image ID."""
        try:
            resp = (
                self.client.table("image_sellers")
                .select("seller_id")
                .eq("image_id", image_id)
                .execute()
            )
            # Count unique sellers
            sellers = {r["seller_id"] for r in (resp.data or [])}
            return len(sellers)
        except Exception as e:
            logger.error("Failed to count sellers for image %s: %s", image_id, e)
            return 0

    def save_image_seller(self, image_id: str, seller_id: str) -> None:
        """Track which seller uses which image (for stock photo detection)."""
        record = {
            "image_id": image_id,
            "seller_id": seller_id,
        }
        try:
            self.client.table("image_sellers").upsert(
                record, on_conflict="image_id,seller_id"
            ).execute()
        except Exception as e:
            logger.error("Failed to save image-seller mapping: %s", e)

    # ── API Quota Tracking ───────────────────────────────────────────

    def get_key_cooldown(self, key_hash: str) -> Optional[datetime]:
        """Get the time when an exhausted API key will reset. Returns None if key is good."""
        try:
            resp = (
                self.client.table("api_quotas")
                .select("exhausted_until")
                .eq("api_key_hash", key_hash)
                .maybe_single()
                .execute()
            )
            if not resp or not hasattr(resp, 'data') or not resp.data:
                return None
            
            until_str = resp.data.get("exhausted_until")
            if not until_str:
                return None
                
            until_dt = datetime.fromisoformat(until_str.replace('Z', '+00:00'))
            if datetime.now(timezone.utc) < until_dt:
                return until_dt
            
            # Cooldown passed, clear it
            self.client.table("api_quotas").delete().eq("api_key_hash", key_hash).execute()
            return None
            
        except Exception as e:
            logger.error("Failed to check key cooldown for hash %s: %s", key_hash, e)
            return None

    def set_key_exhausted(self, key_hash: str, reset_time: datetime) -> None:
        """Mark an API key as exhausted until the specified reset time."""
        record = {
            "api_key_hash": key_hash,
            "exhausted_until": reset_time.isoformat(),
        }
        try:
            self.client.table("api_quotas").upsert(
                record, on_conflict="api_key_hash"
            ).execute()
            logger.warning("API key (hash: %s) marked as exhausted until %s", key_hash, reset_time.isoformat())
        except Exception as e:
            logger.error("Failed to set key exhausted for hash %s: %s", key_hash, e)

    # ── Watchlist (category config, editable from the web UI) ─────────

    def get_watchlist(self, only_enabled: bool = True) -> list[dict]:
        """Return watchlist rows (category config). Most important first.

        Args:
            only_enabled: If True, return only enabled categories (used by the
                scanner). The web UI passes False to edit everything.
        """
        try:
            query = self.client.table("watchlist").select("*")
            if only_enabled:
                query = query.eq("enabled", True)
            resp = query.order("priority", desc=True).order("name").execute()
            return resp.data or []
        except Exception as e:
            logger.error("Failed to load watchlist: %s", e)
            return []

    def upsert_watchlist_item(self, item: dict[str, Any]) -> None:
        """Insert or update a single watchlist category by name."""
        item = {**item, "updated_at": datetime.now(timezone.utc).isoformat()}
        self.client.table("watchlist").upsert(item, on_conflict="name").execute()

    def delete_watchlist_item(self, name: str) -> None:
        """Remove a category from the watchlist."""
        self.client.table("watchlist").delete().eq("name", name).execute()

    # ── Resale tracking (longitudinal sold-price dataset) ─────────────

    def upsert_tracked_listings(self, records: list[dict[str, Any]]) -> None:
        """Batch upsert tracked-listing rows (one per listing, keyed by id)."""
        try:
            self.client.table("tracked_listings").upsert(records, on_conflict="id").execute()
        except Exception as e:
            logger.error("Failed to upsert %d tracked listings: %s", len(records), e)

    def get_tracked_prices(self, item_ids: list[str]) -> dict[str, dict]:
        """Previously-recorded price info for these listings, keyed by id.

        Used to spot price cuts: a seller dropping their asking price is a
        motivated seller, which is a stronger signal than a listing that merely
        looked cheap the first time we saw it.
        """
        out: dict[str, dict] = {}
        unique = list({i for i in item_ids if i})
        CHUNK = 200
        for start in range(0, len(unique), CHUNK):
            chunk = unique[start:start + CHUNK]
            try:
                resp = (
                    self.client.table("tracked_listings")
                    .select("id, price, first_price")
                    .in_("id", chunk)
                    .execute()
                )
                for row in (resp.data or []):
                    out[row["id"]] = row
            except Exception as e:
                logger.error("Failed to load tracked prices: %s", e)
        return out

    def get_stale_tracked(self, stale_before_iso: str, limit: int) -> list[dict]:
        """Active tracked rows not seen since `stale_before_iso`, oldest-checked first."""
        try:
            resp = (
                self.client.table("tracked_listings")
                .select("*")
                .eq("status", "active")
                .lt("last_seen", stale_before_iso)
                .order("last_checked")
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as e:
            logger.error("Failed to load stale tracked listings: %s", e)
            return []

    def touch_tracked(self, item_id: str, ts_iso: str) -> None:
        """Listing is still live: reset both last_seen and last_checked."""
        try:
            self.client.table("tracked_listings").update(
                {"last_seen": ts_iso, "last_checked": ts_iso}
            ).eq("id", item_id).execute()
        except Exception as e:
            logger.error("Failed to touch tracked listing %s: %s", item_id, e)

    def update_tracked_checked(self, item_id: str, ts_iso: str) -> None:
        """Transient check result: advance only last_checked (keep it stale)."""
        try:
            self.client.table("tracked_listings").update(
                {"last_checked": ts_iso}
            ).eq("id", item_id).execute()
        except Exception as e:
            logger.error("Failed to update tracked listing %s: %s", item_id, e)

    def finalize_tracked(self, item_id: str, comp: dict[str, Any]) -> None:
        """Record the realized outcome and drop the raw tracked row."""
        try:
            self.client.table("sold_comps").insert(comp).execute()
            self.client.table("tracked_listings").delete().eq("id", item_id).execute()
        except Exception as e:
            logger.error("Failed to finalize tracked listing %s: %s", item_id, e)

    def get_sold_comps_since(self, since_iso: str) -> list[dict]:
        """Sold-proxy comps ended on/after `since_iso` (for the resale rollup)."""
        try:
            resp = (
                self.client.table("sold_comps")
                .select("product_key, condition_band, final_price, days_active")
                .gte("ended_at", since_iso)
                .eq("sold_proxy", True)
                .execute()
            )
            return resp.data or []
        except Exception as e:
            logger.error("Failed to load sold comps: %s", e)
            return []

    def upsert_resale_stat(self, record: dict[str, Any]) -> None:
        """Insert or update one (product_key, condition_band) resale summary."""
        record = {**record, "updated_at": datetime.now(timezone.utc).isoformat()}
        try:
            self.client.table("resale_stats").upsert(
                record, on_conflict="product_key,condition_band"
            ).execute()
        except Exception as e:
            logger.error("Failed to upsert resale stat: %s", e)

    def get_resale_stats_age_hours(self) -> Optional[float]:
        """Hours since resale_stats was last rebuilt (None if never)."""
        try:
            resp = (
                self.client.table("resale_stats")
                .select("updated_at")
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
            if not resp.data:
                return None
            last = datetime.fromisoformat(resp.data[0]["updated_at"].replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        except Exception as e:
            logger.error("Failed to read resale_stats freshness: %s", e)
            return None

    def delete_rows_older_than(self, table: str, column: str, cutoff_iso: str) -> None:
        """Retention helper: delete rows whose timestamp column predates cutoff."""
        try:
            self.client.table(table).delete().lt(column, cutoff_iso).execute()
        except Exception as e:
            logger.error("Retention delete failed on %s.%s: %s", table, column, e)

    # ── Seller quality tracking ───────────────────────────────────────

    def record_deal_sellers(self, sellers: list[dict[str, Any]]) -> None:
        """Increment each seller's deal tally.

        Sellers who repeatedly list below market (estate clearers, people
        offloading gear they don't specialise in) are a durable edge: worth
        watching directly rather than waiting for the scanner to find them.
        Read-modify-write is fine here — only a handful of sellers per run.
        """
        if not sellers:
            return
        names = [s["seller_username"] for s in sellers if s.get("seller_username")]
        if not names:
            return
        try:
            resp = (
                self.client.table("seller_stats")
                .select("seller_username, deals_found")
                .in_("seller_username", names)
                .execute()
            )
            existing = {r["seller_username"]: r.get("deals_found") or 0 for r in (resp.data or [])}
        except Exception as e:
            logger.error("Failed to read seller stats: %s", e)
            existing = {}

        now = datetime.now(timezone.utc).isoformat()
        records = []
        for s in sellers:
            name = s.get("seller_username")
            if not name:
                continue
            records.append({
                "seller_username": name,
                "marketplace": s.get("marketplace"),
                "deals_found": existing.get(name, 0) + s.get("count", 1),
                "last_deal_at": now,
            })
        try:
            self.client.table("seller_stats").upsert(
                records, on_conflict="seller_username"
            ).execute()
        except Exception as e:
            logger.error("Failed to record seller stats: %s", e)

    def get_top_deal_sellers(self, limit: int = 10, min_deals: int = 2) -> list[dict]:
        """Sellers who have produced the most deals, best first."""
        try:
            resp = (
                self.client.table("seller_stats")
                .select("*")
                .gte("deals_found", min_deals)
                .order("deals_found", desc=True)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as e:
            logger.error("Failed to load top sellers: %s", e)
            return []

    # ── Reference prices (manually-curated new/street prices) ─────────

    def get_reference_prices(self) -> dict[str, dict]:
        """All reference prices keyed by product_key (used by the scanner)."""
        try:
            resp = self.client.table("reference_prices").select("*").execute()
            return {r["product_key"]: r for r in (resp.data or [])}
        except Exception as e:
            logger.error("Failed to load reference prices: %s", e)
            return {}

    def upsert_reference_price(self, record: dict[str, Any]) -> None:
        """Insert/update one reference price; refreshes updated_at (= verified now)."""
        record = {**record, "updated_at": datetime.now(timezone.utc).isoformat()}
        self.client.table("reference_prices").upsert(
            record, on_conflict="product_key"
        ).execute()

    def delete_reference_price(self, product_key: str) -> None:
        """Remove a reference price."""
        self.client.table("reference_prices").delete().eq("product_key", product_key).execute()

    def get_observed_product_keys(self, limit: int = 1000) -> list[str]:
        """Distinct product_keys currently being tracked (to help fill prices)."""
        try:
            resp = self.client.table("tracked_listings").select("product_key").limit(limit).execute()
            return sorted({r["product_key"] for r in (resp.data or []) if r.get("product_key")})
        except Exception as e:
            logger.error("Failed to load observed product keys: %s", e)
            return []
