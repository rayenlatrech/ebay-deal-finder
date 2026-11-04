"""Main orchestrator — runs the full deal-finding pipeline."""

import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Load .env for local development (ignored in GitHub Actions)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deal_finder")


def _image_score_for(photo_type: str, condition_matches: Optional[bool]) -> int:
    """Map a photo verdict to its image score (shared by fresh and cached paths)."""
    if photo_type == "real":
        if condition_matches is True:
            return 85
        if condition_matches is False:
            return 30
        return 60
    if photo_type == "stock":
        return 20
    return 40


# The most a photo check can add: real photos that match the description.
BEST_IMAGE_SCORE = _image_score_for("real", True)

# A scan makes about 66 searches, so this many lost is more than a blip.
MANY_FAILED_SEARCHES = 10

# Why a listing's photo was or wasn't judged by the image model:
#   checked / checked_earlier — the model looked at the main photo (now / cached)
#   ai_failed            — the model was called but unavailable
#   photos_too_small     — every photo is a thumbnail; nothing to judge
#   skipped_limit        — the run's image-check budget was used up
#   skipped_cannot_alert — even a perfect photo couldn't make it a deal
#   no_photo             — the listing has no image
# Alerts show this, because "uncertain" from a real check and "never looked"
# used to read exactly the same.
PHOTO_REVIEWED = ("checked", "checked_earlier")


def _hours_until(timestamp: str, now: datetime) -> Optional[float]:
    """Hours from `now` until an eBay ISO timestamp, or None if unparseable."""
    if not timestamp:
        return None
    try:
        end = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (end - now).total_seconds() / 3600.0


def load_config(db=None) -> dict:
    """Load configuration.

    Settings always come from settings.yaml. Categories come from the Supabase
    ``watchlist`` table (editable via the Streamlit UI) when a database handle
    is provided and the table is non-empty; otherwise they fall back to
    categories.yaml. Either way each category is merged on top of the shared
    defaults so it's fully populated (priority, min_discount, ...).
    """
    config_dir = Path(__file__).parent.parent / "config"

    with open(config_dir / "categories.yaml", "r", encoding="utf-8") as f:
        categories_raw = yaml.safe_load(f)

    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    defaults = categories_raw.get("defaults", {})

    categories = None
    if db is not None:
        rows = db.get_watchlist(only_enabled=True)
        if rows:
            # DB nulls must not clobber defaults, so drop them before merging.
            categories = [
                {**defaults, **{k: v for k, v in row.items() if v is not None}}
                for row in rows
            ]
            logger.info("Loaded %d categories from Supabase watchlist", len(categories))

    if categories is None:
        categories = [{**defaults, **cat} for cat in categories_raw["categories"]]
        logger.info("Loaded %d categories from categories.yaml (watchlist empty)", len(categories))

    return {
        "categories": categories,
        "settings": settings,
    }


def run_pipeline() -> None:
    """Execute the full deal-finding pipeline.

    Flow:
        1. Search eBay for all configured categories
        2. Get market prices
        3. Score each listing on price
        4. For listings above threshold: run AI text analysis
        5. For top listings: run image analysis (stock detection + condition)
        6. Calculate composite scores
        7. Send alerts for hot and good deals
        8. Report scan statistics

    Every few hours this is a full scan. In between, a quick scan reuses the
    last full scan's market prices, only reviews listings the AI hasn't judged
    at their current price, and skips resale tracking maintenance.
    """
    from src.db import Database
    from src.ebay_client import EbayClient
    from src.ai_analyzer import (
        AIAnalyzer, split_assessment, text_assessment_reusable, image_assessment_reusable,
    )
    from src.deal_scorer import (
        DealScorer, exit_signal, find_exit_mismatches, is_trusted_seller, listing_age,
    )
    from src.price_tracker import (
        PriceTracker, is_for_parts, condition_band, baseline_key,
        PRICED_BANDS,
    )
    from src.baseline import MarketWindow
    from src.scan_state import (
        add_to_report, digest_due, full_scan_due, load_state, prune_judged, record_verdict,
        restore_report, save_state, select_for_review, slim_deal, slim_window_rows,
        take_report_if_due,
    )
    from src.sold_tracker import SoldTracker
    from src.telegram_notifier import TelegramNotifier

    # Initialize components
    logger.info("═══ eBay AI Deal Finder — Starting Scan ═══")
    db = Database()

    # Categories come from the Supabase watchlist (editable in the web UI),
    # falling back to categories.yaml when the table is empty.
    config = load_config(db)
    settings = config["settings"]
    # Per-category config lookup (priority, min_discount, price curve, ...)
    category_config = {cat["name"]: cat for cat in config["categories"]}

    ebay = EbayClient()
    ai = AIAnalyzer(db)
    scorer = DealScorer()
    price_tracker = PriceTracker(db, settings["ebay"].get("marketplace", "EBAY_IT"))
    sold_tracker = SoldTracker(db, price_tracker, settings.get("tracking", {}))
    telegram = TelegramNotifier()

    stats = {
        "total_scanned": 0,
        "hot_deals": 0,
        "good_deals": 0,
        "logged": 0,
        "skipped": 0,
        "errors": 0,
        "photo_checks": 0,
    }

    scan_started = datetime.now(timezone.utc)
    state = load_state()
    judged = prune_judged(state.get("judged", {}), scan_started)
    # Good deals waiting for the next digest. The first one waits for its time.
    pending_digest = state.get("digest", {})
    digest_hours = settings["telegram"].get("digest_hours_utc") or []
    state.setdefault("last_digest_at", scan_started.isoformat())
    use_window_mode = (settings.get("baseline", {}) or {}).get("mode", "legacy") == "window"
    full_scan = not use_window_mode or full_scan_due(
        state, scan_started, settings["scanning"].get("full_scan_every_hours", 4)
    )
    logger.info("Scan type: %s", "full" if full_scan else "quick")

    try:
        # ── Step 1: Search eBay ──────────────────────────────────────
        logger.info("Step 1: Searching eBay Italy...")
        ebay_cfg = settings["ebay"]
        trust_cfg = settings.get("seller_trust", {})
        min_positive_pct = trust_cfg.get("min_positive_pct", 95.0)
        min_sales = trust_cfg.get("min_sales", 10)

        listings = ebay.search_all_categories(
            categories=config["categories"],
            max_pages=ebay_cfg.get("max_pages_per_category", 1),
            results_per_page=ebay_cfg.get("results_per_page", 200),
            foreign_marketplaces=ebay_cfg.get("foreign_marketplaces", []),
            ship_to_country=ebay_cfg.get("ship_to_country", "IT"),
        )

        # Auctions get their own sweep sorted by end time: sorted by newest they
        # never came close enough to ending to judge (0 alerts in 30 runs).
        if ebay_cfg.get("include_auctions", False):
            auctions = ebay.search_ending_auctions(
                categories=config["categories"],
                max_hours=ebay_cfg.get("auction_ending_within_hours", 10),
                foreign_marketplaces=ebay_cfg.get("foreign_marketplaces", []),
                ship_to_country=ebay_cfg.get("ship_to_country", "IT"),
            )
            known = {l["item_id"] for l in listings}
            listings.extend(a for a in auctions if a["item_id"] not in known)

        stats["total_scanned"] = len(listings)
        # A refused search looks just like an empty one, so say when it's a lot of them.
        stats["failed_searches"] = ebay.failed_calls
        if ebay.failed_calls >= MANY_FAILED_SEARCHES:
            telegram.send_error_alert(
                f"eBay refused {ebay.failed_calls} searches, so this scan may have missed "
                "many listings. This often means the daily limit of 5,000 calls was reached."
            )
        logger.info("Found %d listings across all categories", len(listings))

        # Point every alert link at the home marketplace so it opens in the
        # eBay app (foreign domains such as ebay.nl are handed to the browser).
        if ebay_cfg.get("rewrite_urls_to_home", True):
            home_market = ebay_cfg.get("marketplace", "EBAY_IT")
            rewritten = 0
            for listing in listings:
                if listing.get("marketplace") == home_market:
                    continue
                home_url = EbayClient.home_item_url(listing["item_id"], home_market)
                if home_url:
                    listing["original_item_url"] = listing.get("item_url", "")
                    listing["item_url"] = home_url
                    rewritten += 1
            if rewritten:
                logger.info("Rewrote %d foreign listing URL(s) onto the home domain", rewritten)

        # ── Bad-unit title filter ────────────────────────────────────
        # Carrier-locked, iCloud-locked and non-EU stock are bad buys at any
        # price, so they're dropped before anything else looks at them.
        title_blocklist = []
        for pattern in settings["scanning"].get("excluded_title_patterns", []) or []:
            try:
                title_blocklist.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.error("Ignoring invalid excluded_title_pattern %r: %s", pattern, e)

        if title_blocklist:
            before = len(listings)
            listings = [
                l for l in listings
                if not any(p.search(l.get("title", "")) for p in title_blocklist)
            ]
            blocked = before - len(listings)
            if blocked:
                stats["skipped"] += blocked
                logger.info("Dropped %d listing(s) matching bad-unit title patterns", blocked)

        # ── Customs gate: EU-located items only ──────────────────────
        # Foreign sweeps filter on "ships to Italy", which a US or UK seller can
        # satisfy — but the buyer then pays import VAT and customs handling,
        # wiping out the measured discount. Keep purchases inside the EU
        # customs union. Items with no stated location get the benefit of the
        # doubt but are counted, so a large number here would be visible.
        if ebay_cfg.get("eu_only", True):
            eu = {c.upper() for c in ebay_cfg.get("eu_countries", [])}
            kept_eu, non_eu, unknown_loc = [], 0, 0
            for listing in listings:
                country = str(listing.get("item_location_country") or "").upper()
                if not country:
                    unknown_loc += 1
                    kept_eu.append(listing)
                elif country in eu:
                    kept_eu.append(listing)
                else:
                    non_eu += 1
            if non_eu:
                stats["skipped"] += non_eu
                logger.info(
                    "Dropped %d non-EU listing(s) (customs/import VAT would erase the discount); "
                    "%d had no stated location", non_eu, unknown_loc,
                )
            listings = kept_eu

        # ── Cross-border trust gate ──────────────────────────────────
        # Foreign listings are only considered from established, well-rated
        # sellers. Dropped here so they never reach the baselines, the AI
        # budget, or the resale dataset.
        kept: list[dict] = []
        for listing in listings:
            trusted = is_trusted_seller(listing, min_positive_pct, min_sales)
            listing["seller_trusted"] = trusted
            if listing.get("is_foreign") and not trusted:
                stats["skipped"] += 1
                continue
            kept.append(listing)

        dropped_foreign = len(listings) - len(kept)
        if dropped_foreign:
            logger.info(
                "Dropped %d foreign listing(s) from sellers below the trust bar "
                "(>=%.1f%% positive, >=%d sales)",
                dropped_foreign, min_positive_pct, min_sales,
            )
        listings = kept

        if not listings:
            logger.warning("No listings found — exiting")
            telegram.send_scan_report(stats)
            return

        # ── Step 1b: Record sightings for sold-price tracking ────────
        try:
            recorded = sold_tracker.record_sightings(listings)
            logger.info("Recorded %d sightings for resale tracking", recorded)
        except Exception as e:
            logger.error("Sighting recording failed: %s", e)

        # ── Step 2: Market Prices ────────────────────────────────────
        # Prices every model from all listings tracked over the last N days
        # (settings.yaml → baseline.mode: window); see baseline.py. They barely
        # move in an hour, so a quick scan reuses the last full scan's listings.
        logger.info("Step 2: Getting market prices...")
        reference_prices = db.get_reference_prices()
        window = None
        if use_window_mode:
            if full_scan:
                window = MarketWindow.load(db, settings, price_tracker.marketplace_home)
                if window:
                    state["window_rows"] = slim_window_rows(window.rows)
                    state["last_full_at"] = scan_started.isoformat()
                elif state.get("window_rows"):
                    # A few hours old beats the legacy baseline, whose price
                    # history stopped being written. The next scan retries.
                    logger.warning("Market window read failed — using the last full scan's")
                    window = MarketWindow.from_rows(
                        state["window_rows"], reference_prices, settings, price_tracker.marketplace_home
                    )
            else:
                window = MarketWindow.from_rows(
                    state["window_rows"], reference_prices, settings, price_tracker.marketplace_home
                )
            if window:
                logger.info(
                    "Market window: %d listings over %dd -> %d buckets (%d with >=%d listings); "
                    "foreign/home price ratios %s",
                    len(window.rows), window.days, len(window.prices),
                    sum(1 for m in window.prices.values() if m.n >= window.min_listings),
                    window.min_listings,
                    {f"{m}/{b}": round(r, 3) for (m, b), r in window.ratios.items()},
                )
            else:
                logger.warning("Market window unavailable — using the legacy baseline this run")
        use_window = window is not None
        # The per-scan price history only feeds the legacy baseline.
        market_prices = {} if use_window else price_tracker.update_market_prices(listings)

        # Standing cross-border price gaps (a whole market pricing lower is a
        # more durable edge than any single cheap listing).
        try:
            arbitrage = price_tracker.find_marketplace_gaps(listings)
            stats["arbitrage"] = arbitrage[:5]
            for gap in arbitrage[:5]:
                logger.info(
                    "💱 %s (%s): €%.0f on %s vs €%.0f home (%.0f%% cheaper)",
                    gap["product_key"], gap["condition_band"], gap["cheapest_price"],
                    gap["cheapest_market"], gap["home_price"], gap["gap_pct"],
                )
        except Exception as e:
            logger.error("Arbitrage scan failed: %s", e)

        # ── Step 3: Score on Price & Filter ──────────────────────────
        logger.info("Step 3: Scoring listings on price...")
        global_min_discount = settings["scanning"].get("min_discount_for_ai", 15)
        max_ai_analyses = settings["ai"].get("max_text_analyses_per_run", 50)

        # Buckets that must never be alerted on (e.g. Galaxy A-series bar the A57).
        excluded_patterns = []
        for pattern in settings["scanning"].get("excluded_product_patterns", []) or []:
            try:
                excluded_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.error("Ignoring invalid excluded_product_pattern %r: %s", pattern, e)

        # Manually-curated new/street prices — used as a baseline ceiling so an
        # inflated eBay asking-median can't invent a deal (see the A57 case).
        ref_cfg = settings.get("reference", {})
        band_factor = {
            "used": ref_cfg.get("used_factor", 0.75),
            "new": ref_cfg.get("new_factor", 1.00),
        }

        auction_window_h = ebay_cfg.get("auction_ending_within_hours", 2)
        now_utc = datetime.now(timezone.utc)

        candidates = []
        below_threshold = []
        for listing in listings:
            # An auction is only judgeable near its close — before that the bid
            # will keep moving, so a "cheap" auction means nothing yet.
            if listing.get("is_auction"):
                hours_left = _hours_until(listing.get("auction_end"), now_utc)
                if hours_left is None or not (0 < hours_left <= auction_window_h):
                    stats["skipped"] += 1
                    continue
                listing["hours_left"] = hours_left

            # Drop for-parts and refurbished units before they reach the funnel.
            band = condition_band(listing)
            if is_for_parts(listing) or band not in PRICED_BANDS:
                stats["skipped"] += 1
                continue

            product_key, confidence = price_tracker.generate_product_key(
                listing["title"],
                listing.get("category", ""),
                listing.get("epid", ""),
                listing.get("condition_id", ""),
            )
            # Skip items we couldn't classify or that have low confidence
            if not product_key or confidence < PriceTracker.MIN_CONFIDENCE:
                stats["skipped"] += 1
                continue

            # Explicitly excluded buckets never become deals.
            if any(p.search(product_key) for p in excluded_patterns):
                stats["skipped"] += 1
                continue

            # Per-category focus knobs
            cat_cfg = category_config.get(listing.get("category", ""), {})
            priority = cat_cfg.get("priority", 1.0)
            cat_min_discount = cat_cfg.get("min_discount", global_min_discount)
            cat_price_scores = cat_cfg.get("price_scores")  # None => global curve

            # Compare like with like: a used unit against the used baseline,
            # a sealed one against the sealed baseline.
            baseline_n = None
            if use_window:
                resolved = window.judge(product_key, band)
                if not resolved:
                    stats["skipped"] += 1
                    continue
                market = window.prices.get(baseline_key(product_key, band))
                avg_price = market.value if market else None
                new_price = resolved.new_price
                baseline, baseline_source, baseline_n = resolved.value, resolved.source, resolved.n
            else:
                avg_price = market_prices.get(baseline_key(product_key, band))
                if not avg_price:
                    stats["skipped"] += 1
                    continue

                # ── New-price ceiling (legacy) ───────────────────────
                # Priority 1: curated retail price (exact key, then without
                # storage so one entry covers every storage tier).
                ref = reference_prices.get(product_key) or reference_prices.get(
                    re.sub(r"_\d+(gb|tb)$", "", product_key)
                )
                new_price = float(ref["new_price"]) if ref and ref.get("new_price") else None
                ceiling_source = "retail"
                # Priority 2 (free, automatic, optional): eBay's own NEW-condition
                # baseline for this bucket — asking prices, not true retail, so
                # it's only a fallback and can be disabled in settings.yaml.
                if new_price is None and ref_cfg.get("use_ebay_new_fallback", True):
                    new_price = market_prices.get(baseline_key(product_key, "new"))
                    ceiling_source = "eBay new"

                # A used unit can't be worth full retail, so scale the ceiling.
                baseline = avg_price
                baseline_source = f"{band} market"
                if new_price:
                    ceiling = new_price * band_factor.get(band, 1.0)
                    if 0 < ceiling < avg_price:
                        baseline = ceiling
                        baseline_source = f"{ceiling_source} anchor"

            # Calculate preliminary discount using Total Price (Price + Shipping)
            total_price = listing.get("total_price", listing["price"])
            price_score, discount_pct, is_suspicious = scorer.calculate_price_score(
                total_price, baseline, cat_price_scores
            )

            if discount_pct < cat_min_discount:
                below_threshold.append(listing)
                stats["skipped"] += 1
                continue

            # An auction's current bid will keep rising, so the useful number is
            # the most it's still worth paying: the price that would still clear
            # this category's minimum discount, net of shipping. Entered as an
            # eBay max bid, proxy bidding holds that line even if the auction
            # ends long after the alert.
            max_bid = None
            if listing.get("is_auction"):
                shipping = float(listing.get("shipping_cost") or 0)
                max_bid = round(baseline * (1 - cat_min_discount / 100) - shipping)
                if max_bid <= float(listing.get("price") or 0):
                    stats["skipped"] += 1   # already bid past the point of being a deal
                    continue

            candidates.append({
                "max_bid": max_bid,
                "band": band,
                "listing": listing,
                "total_price": float(total_price),
                "product_key": product_key,
                "avg_price": baseline,          # effective baseline used for scoring
                "ebay_avg": avg_price,          # raw eBay asking baseline
                "new_price": new_price,         # curated new price, if any
                "baseline_source": baseline_source,
                "baseline_n": baseline_n,       # listings behind it (window mode)
                "price_score": price_score,
                "discount_pct": discount_pct,
                "is_suspicious": is_suspicious,
                "priority": priority,
                # Focus ranking: a high-priority category with a strong price
                # score wins AI budget over a low-priority one, regardless of
                # raw discount.
                "rank_score": price_score * priority,
            })

        db.upsert_listings(below_threshold)

        # Skip listings already alerted on.
        alerted_ids = db.get_alerted_ids([c["listing"]["item_id"] for c in candidates])
        stats["skipped"] += sum(1 for c in candidates if c["listing"]["item_id"] in alerted_ids)
        candidates = [c for c in candidates if c["listing"]["item_id"] not in alerted_ids]

        # Allocate the limited AI budget to the highest-priority strong deals.
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)
        max_image_analyses = settings["ai"].get("max_image_analyses_per_run", 3)
        reviewing = select_for_review(
            candidates, judged, full_scan, max_ai_analyses,
            photo_slots=max_image_analyses if ai.gemini_keys else 0,
        )
        stats["skipped"] += len(candidates) - len(reviewing)
        logger.info(
            "Step 3 complete: reviewing %d of %d candidates (out of %d listings)",
            len(reviewing), len(candidates), stats["total_scanned"],
        )
        candidates = reviewing

        # ── Step 4: AI Text Analysis ─────────────────────────────────
        logger.info("Step 4: Running AI text analysis...")
        hot_deals: list[dict] = []
        good_deals: list[dict] = []
        send_now: list[dict] = []
        resale_stats = db.get_resale_stats()
        photos_per_check = settings["ai"].get("photos_per_check", 3)
        image_analyses_done = 0
        image_analyses_failed = 0

        # A listing's description and photos never change, so reuse any recent
        # assessment instead of paying for the same answer every scan.
        cache_hours = settings["ai"].get("cache_hours", 720)
        cache_since = (datetime.now(timezone.utc) - timedelta(hours=cache_hours)).isoformat()
        analysis_cache = db.get_recent_analyses(
            [c["listing"]["item_id"] for c in candidates], cache_since
        )
        # The text and photo halves of a stored assessment fail independently,
        # so each is judged on its own. Previously one failed image check marked
        # the whole assessment unusable, discarding a good text analysis and
        # forcing it to be paid for again every scan.
        cache_hits = 0
        image_cache_hits = 0
        image_statuses: dict[str, int] = {}
        score_debug: list[dict] = []

        # Text reviews come first, because photo checks are scarce: the few
        # allowed per scan should go to the listings most likely to be hot deals,
        # and that can't be known until the text is reviewed.
        reviewed: list[dict] = []
        for candidate in candidates:
            listing = candidate["listing"]
            cached_text, cached_image = split_assessment(analysis_cache.get(listing["item_id"]))
            if not text_assessment_reusable(cached_text):
                cached_text = {}
            if not image_assessment_reusable(cached_image):
                cached_image = {}
            # The description hasn't changed, but "not a good deal" was about the
            # old price. Only a cut matters: a higher price stays not a deal.
            earlier = judged.get(listing["item_id"])
            if cached_text and earlier and candidate["total_price"] < float(earlier.get("price") or 0) - 0.005:
                cached_text = {k: v for k, v in cached_text.items() if k != "is_good_deal"}
            try:
                # Get full item details for description (query the listing's own
                # marketplace — a foreign item can 404 against the home site).
                details = ebay.get_item_details(
                    listing["item_id"], marketplace=listing.get("marketplace")
                )
                if details:
                    shipping = listing.get("shipping_cost", 0.0) or 0.0
                    listing.update(details)
                    # get_item_details overwrites `price` but carries no shipping,
                    # so recompute the total to keep the two consistent.
                    listing["shipping_cost"] = shipping
                    listing["total_price"] = float(listing.get("price", 0)) + shipping

                # Enrich listing with price context for the AI prompt
                listing["avg_price"] = candidate["avg_price"]
                listing["discount_pct"] = candidate["discount_pct"]

                # Run AI text analysis (cached result if we've seen this listing)
                if cached_text:
                    text_analysis = cached_text
                    cache_hits += 1
                else:
                    text_analysis = ai.analyze_listing(listing)

                # Instantly drop accessories/boxes/incomplete items caught by Groq
                if not text_analysis.get("is_complete_item", True):
                    logger.info("Skipping %s: Groq flagged as incomplete/accessory", listing["item_id"])
                    stats["skipped"] += 1
                    record_verdict(judged, candidate, now_utc)
                    continue

                if text_analysis.get("condition_assessment") in ["parts_only", "damaged"]:
                    logger.info("Skipping %s: Groq flagged as damaged/parts", listing["item_id"])
                    stats["skipped"] += 1
                    record_verdict(judged, candidate, now_utc)
                    continue

                # Calculate seller score
                seller_score, seller_trust_level = scorer.calculate_seller_score(
                    listing.get("seller_feedback_score", 0),
                    listing.get("seller_feedback_percentage", 0),
                )

                # Map AI condition assessment to numeric score
                condition_map = {
                    "mint": 95, "good": 75, "fair": 55,
                    "poor": 30, "damaged": 15, "parts_only": 5,
                    "unknown": 50,
                }
                condition_score = condition_map.get(
                    text_analysis.get("condition_assessment", "unknown"), 50
                )

                # Calculate listing quality
                listing_quality = scorer.calculate_listing_quality_score(listing)

                # An item first listed weeks ago and still unsold usually has a
                # catch, even when a relist makes it look brand new.
                age_days, relisted = listing_age(listing, now_utc)
                age_penalty = scorer.calculate_age_penalty(age_days)

                # Image checks are capped per run, so don't spend one on a
                # listing that can't alert even with the best possible photo
                # verdict — a photo check can only add BEST_IMAGE_SCORE, and its
                # penalties only subtract. That leaves the budget for listings
                # whose alert actually depends on it.
                best_case = scorer.calculate_composite_score(
                    price_score=candidate["price_score"],
                    condition_score=condition_score,
                    seller_score=seller_score,
                    image_score=BEST_IMAGE_SCORE,
                    listing_quality_score=listing_quality,
                    is_suspicious=candidate["is_suspicious"],
                    ai_penalty=scorer.calculate_ai_penalty(text_analysis, {}),
                    age_penalty=age_penalty,
                )

                reviewed.append({
                    **candidate,
                    "text_analysis": text_analysis,
                    "cached_image": cached_image,
                    "fresh_text": not cached_text,
                    "seller_score": seller_score,
                    "seller_trust_level": seller_trust_level,
                    "condition_score": condition_score,
                    "listing_quality": listing_quality,
                    "age_days": age_days,
                    "relisted": relisted,
                    "age_penalty": age_penalty,
                    "best_case": best_case,
                })

            except Exception as e:
                logger.error("Error analyzing %s: %s", listing.get("item_id", "?"), e)
                stats["errors"] += 1

        reviewed.sort(key=lambda c: c["best_case"], reverse=True)

        for candidate in reviewed:
            listing = candidate["listing"]
            text_analysis = candidate["text_analysis"]
            cached_image = candidate["cached_image"]
            seller_score = candidate["seller_score"]
            seller_trust_level = candidate["seller_trust_level"]
            condition_score = candidate["condition_score"]
            listing_quality = candidate["listing_quality"]
            age_days = candidate["age_days"]
            relisted = candidate["relisted"]
            age_penalty = candidate["age_penalty"]
            could_alert = scorer.classify_deal(candidate["best_case"]) in ("hot_deal", "good_deal")
            try:
                # ── Step 5: Image Analysis (Top Listings Only) ───────
                image_score = 50  # Default neutral
                photo_type = "uncertain"
                stock_penalty = 0
                image_analysis = {}
                image_status = "no_photo"
                fresh_image = False

                # Quick free check first
                quick_check = ai.quick_photo_check(listing)

                # Track image for dedup
                image_url = listing.get("image_url", "")
                if image_url:
                    image_id = ai.extract_image_id(image_url)
                    if image_id:
                        seller_id = listing.get("seller_username", "unknown")
                        db.save_image_seller(image_id, seller_id)

                        # Check dedup
                        seller_count = db.count_sellers_for_image(image_id)
                        if seller_count >= 5:
                            quick_check["suspicious"] = True
                            quick_check["flags"].append(
                                f"image_reused_by_{seller_count}_sellers"
                            )

                # Reuse a genuine stored photo verdict. One that failed, or an
                # image that was never analysed, is retried instead of being
                # locked in as "uncertain" for the whole cache window.
                if cached_image:
                    image_analysis = cached_image
                    image_cache_hits += 1
                    image_status = "checked_earlier"
                    photo_type = cached_image.get("photo_type", "uncertain")
                    image_score = _image_score_for(
                        photo_type, cached_image.get("condition_matches_description")
                    )

                # Full image analysis for top candidates. Only a completed check
                # uses up the allowance: a brief Gemini outage shouldn't cost this
                # scan's few checks. It stops after as many failures, and when
                # the daily limit has taken every key out.
                elif (image_url and could_alert and ai.gemini_keys
                      and image_analyses_done < max_image_analyses
                      and image_analyses_failed < max_image_analyses):
                    fresh_image = True
                    image_analysis = ai.analyze_image(
                        image_url,
                        condition=listing.get("condition", ""),
                        description_excerpt=listing.get("short_description", ""),
                        # The back and edges are usually in photos 2-3; they
                        # ride along in the same request, so cost no extra calls.
                        # The whole gallery is offered so thumbnails can be skipped.
                        extra_image_urls=[u for u in listing.get("all_image_urls", []) if u != image_url],
                        max_photos=photos_per_check,
                    )
                    if image_analysis.get("photos_too_small"):
                        image_status = "photos_too_small"   # no request was spent
                    elif image_analysis.get("image_ai_failed"):
                        image_analyses_failed += 1
                        image_status = "ai_failed"
                    else:
                        image_analyses_done += 1
                        image_status = "checked"

                    photo_type = image_analysis.get("photo_type", "uncertain")
                    image_score = _image_score_for(
                        photo_type, image_analysis.get("condition_matches_description")
                    )

                else:
                    if image_url:
                        image_status = "skipped_limit" if could_alert else "skipped_cannot_alert"
                    if quick_check.get("suspicious"):
                        photo_type = "stock"
                        image_score = 25
                image_statuses[image_status] = image_statuses.get(image_status, 0) + 1

                # Stock photo penalty
                stock_penalty = scorer.get_stock_photo_penalty(
                    photo_type, listing.get("condition", "")
                )

                # ── Step 6: Composite Score ──────────────────────────
                # The model's own findings (red flags, undisclosed defects,
                # visible damage, its deal verdict) now count against the score.
                ai_penalty = scorer.calculate_ai_penalty(text_analysis, image_analysis)

                deal_score = scorer.calculate_composite_score(
                    price_score=candidate["price_score"],
                    condition_score=condition_score,
                    seller_score=seller_score,
                    image_score=image_score,
                    listing_quality_score=listing_quality,
                    stock_photo_penalty=stock_penalty,
                    is_suspicious=candidate["is_suspicious"],
                    ai_penalty=ai_penalty,
                    age_penalty=age_penalty,
                )

                deal_class = scorer.classify_deal(deal_score)

                score_debug.append({
                    "deal": deal_score, "price": candidate["price_score"],
                    "cond": condition_score, "seller": seller_score,
                    "img": image_score, "qual": listing_quality,
                    "ai": ai_penalty, "age": age_penalty, "key": candidate["product_key"],
                })

                # Merge all analysis data
                full_analysis = {
                    **text_analysis,
                    **image_analysis,
                    "photo_type": photo_type,
                    "quick_photo_flags": quick_check.get("flags", []),
                }

                scores = {
                    "deal_score": deal_score,
                    "price_score": candidate["price_score"],
                    "condition_score": condition_score,
                    "seller_score": seller_score,
                    "image_score": image_score,
                    "listing_quality": listing_quality,
                    "stock_penalty": stock_penalty,
                    "avg_price": candidate["avg_price"],
                    "ebay_avg": candidate["ebay_avg"],
                    "new_price": candidate["new_price"],
                    "baseline_source": candidate["baseline_source"],
                    "baseline_n": candidate["baseline_n"],
                    "baseline_days": window.days if use_window else None,
                    "condition_band": candidate["band"],
                    "product_key": candidate["product_key"],
                    "price_drop_pct": listing.get("price_drop_pct"),
                    "original_price": listing.get("original_price"),
                    "is_auction": listing.get("is_auction", False),
                    "bid_count": listing.get("bid_count", 0),
                    "hours_left": listing.get("hours_left"),
                    "max_bid": candidate.get("max_bid"),
                    "seller_trusted": listing.get("seller_trusted", False),
                    "is_foreign": listing.get("is_foreign", False),
                    "marketplace": listing.get("marketplace", ""),
                    "discount_pct": candidate["discount_pct"],
                    "is_suspicious": candidate["is_suspicious"],
                    "seller_trust_level": seller_trust_level,
                    "deal_class": deal_class,
                    # Kept in scores, not the cached assessment: it describes
                    # this run, not the listing.
                    "image_status": image_status,
                    "text_reviewed": not text_analysis.get("text_ai_failed", False),
                    "listing_age_days": round(age_days, 2) if age_days is not None else None,
                    "relisted": relisted,
                    **exit_signal(
                        resale_stats.get((candidate["product_key"], candidate["band"])),
                        candidate["avg_price"],
                    ),
                }

                # Save to database. A new row only when there's something new to
                # keep: one per re-score would pile up rows that say the same thing.
                db.upsert_listing(listing)
                if candidate["fresh_text"] or fresh_image or deal_class in ("hot_deal", "good_deal"):
                    db.save_analysis({
                        "listing_id": listing["item_id"],
                        "deal_score": deal_score,
                        "price_score": candidate["price_score"],
                        "condition_score": condition_score,
                        "seller_score": seller_score,
                        "image_score": image_score,
                        "ai_assessment": full_analysis,
                    })

                # A full scan can re-score a deal already waiting for a digest.
                was_queued = pending_digest.pop(listing["item_id"], None) is not None
                delivered = True

                # ── Step 7: Send Alerts ──────────────────────────────
                if deal_class == "hot_deal":
                    logger.info(
                        "🔥 HOT DEAL: %s — €%.0f (%s%% off) — Score: %d",
                        listing["title"][:40], listing["price"],
                        candidate["discount_pct"], deal_score,
                    )
                    delivered = telegram.send_hot_deal_alert(listing, full_analysis, scores)
                    if delivered:
                        db.mark_alerted(listing["item_id"])
                    else:
                        logger.warning("Hot deal alert failed to send — it stays unjudged for a retry")
                    hot_deals.append({
                        "listing": listing,
                        "analysis": full_analysis,
                        "scores": scores,
                    })
                    stats["hot_deals"] += 1

                elif deal_class == "good_deal":
                    logger.info(
                        "👍 Good deal: %s — €%.0f (%s%% off) — Score: %d",
                        listing["title"][:40], listing["price"],
                        candidate["discount_pct"], deal_score,
                    )
                    deal = {"listing": listing, "analysis": full_analysis, "scores": scores}
                    # An auction would end before the next digest, and a deal only
                    # short of hot for want of a photo check shouldn't wait for one.
                    would_be_hot = (
                        scorer.classify_deal(candidate["best_case"]) == "hot_deal"
                        and image_status in ("skipped_limit", "ai_failed")
                    )
                    if listing.get("is_auction") or would_be_hot or not digest_hours:
                        send_now.append(deal)
                    else:
                        pending_digest[listing["item_id"]] = slim_deal(deal)
                    if not was_queued:
                        good_deals.append(deal)
                        stats["good_deals"] += 1

                elif deal_class == "log_only":
                    stats["logged"] += 1

                else:
                    stats["skipped"] += 1

                # A failed text review or alert stays unjudged so the next scan
                # retries it.
                if delivered and not text_analysis.get("text_ai_failed"):
                    record_verdict(
                        judged, candidate, now_utc,
                        needs_photo=could_alert and image_status in ("skipped_limit", "ai_failed"),
                    )

            except Exception as e:
                logger.error("Error analyzing %s: %s", listing.get("item_id", "?"), e)
                stats["errors"] += 1

        # ── Send Batched Good Deals ──────────────────────────────────
        batching = settings["telegram"].get("batch_good_deals", True)
        if send_now and batching:
            # Only mark as alerted if the summary actually went out, otherwise
            # a failed send would silently bury these listings forever.
            if telegram.send_deal_summary(send_now):
                for deal in send_now:
                    db.mark_alerted(deal["listing"]["item_id"])
            else:
                logger.warning("Deal summary failed to send — not marking as alerted")

        if pending_digest and batching and digest_due(state, scan_started, digest_hours):
            # They may have waited for hours, so leave out any that have gone.
            gone = [
                item_id for item_id, deal in pending_digest.items()
                if ebay.item_is_active(item_id, marketplace=deal["listing"].get("marketplace")) is False
            ]
            for item_id in gone:
                del pending_digest[item_id]
            if not pending_digest or telegram.send_deal_summary(list(pending_digest.values())):
                for item_id in pending_digest:
                    db.mark_alerted(item_id)
                logger.info("Digest sent: %d deal(s), %d already gone", len(pending_digest), len(gone))
                pending_digest.clear()
                state["last_digest_at"] = scan_started.isoformat()
            else:
                logger.warning("Digest failed to send — keeping it for the next scan")

        stats["photo_checks"] = image_analyses_done
        if candidates:
            logger.info(
                "AI cache: text reused %d/%d (%d fresh calls), photo verdict reused %d, "
                "fresh image checks %d (%d failed)",
                cache_hits, len(candidates), len(candidates) - cache_hits,
                image_cache_hits, image_analyses_done, image_analyses_failed,
            )
            logger.info("Photo review: %s", ", ".join(
                f"{k}={v}" for k, v in sorted(image_statuses.items())) or "none")
        # Score spread makes a quiet scan diagnosable: without it a run of all
        # zeros is indistinguishable from "scored 58 and just missed the bar".
        if score_debug:
            score_debug.sort(key=lambda s: s["deal"], reverse=True)
            logger.info("Top scores this run (threshold: good=%d hot=%d):",
                        scorer.thresholds["good_deal"], scorer.thresholds["hot_deal"])
            for s in score_debug[:8]:
                logger.info(
                    "  deal=%-3d price=%-3d cond=%-3d seller=%-3d img=%-3d qual=%-3d ai=%-4d age=%-4d %s",
                    s["deal"], s["price"], s["cond"], s["seller"],
                    s["img"], s["qual"], s["ai"], s["age"], s["key"],
                )

        # ── Record which sellers produced deals ──────────────────────
        try:
            tally: dict[str, dict] = {}
            for deal in hot_deals + good_deals:
                name = deal["listing"].get("seller_username")
                if not name:
                    continue
                entry = tally.setdefault(name, {
                    "seller_username": name,
                    "marketplace": deal["listing"].get("marketplace"),
                    "count": 0,
                })
                entry["count"] += 1
            if tally:
                db.record_deal_sellers(list(tally.values()))
        except Exception as e:
            logger.error("Seller stat recording failed: %s", e)

        # ── Step 9: Resale tracking maintenance (never breaks a scan) ─
        if full_scan:
            try:
                alerted_ended = sold_tracker.reconcile_alerted(ebay)
                ended = alerted_ended + sold_tracker.reconcile_ended(ebay)
                sold_tracker.rebuild_resale_stats()
                sold_tracker.cleanup()
                logger.info("Resale maintenance: finalized %d ended listing(s)", ended)
            except Exception as e:
                logger.error("Resale maintenance failed: %s", e)

        # ── Step 8: Report ───────────────────────────────────────────
        # Totals only, once a day: a message per scan would be 24 a day.
        if settings["telegram"].get("send_scan_report", True):
            stats["ebay_calls"] = ebay.browse_calls
            add_to_report(state, stats, scan_started)
            report = take_report_if_due(
                state, scan_started, settings["telegram"].get("report_every_hours", 24)
            )
            if report:
                outcomes, feedback, mismatches = None, None, None
                try:
                    outcomes = sold_tracker.alert_outcomes()
                    feedback = db.get_feedback_counts()
                    if window:
                        mismatches = find_exit_mismatches(window.prices, resale_stats)
                except Exception as e:     # the totals still go out
                    logger.error("Report extras failed: %s", e)
                if not telegram.send_daily_report(
                    report, stats.get("arbitrage"), outcomes=outcomes,
                    feedback=feedback, mismatches=mismatches,
                ):
                    restore_report(state, report)

        state["judged"] = judged
        state["digest"] = pending_digest
        save_state(state)

        logger.info("═══ Scan Complete ═══")
        logger.info("Stats: %s", stats)

    except Exception as e:
        logger.critical("Pipeline failed: %s", e, exc_info=True)
        try:
            telegram.send_error_alert(f"Pipeline crashed: {e}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
