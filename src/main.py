"""Main orchestrator — runs the full deal-finding pipeline."""

import logging
import os
import sys
from pathlib import Path

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
        2. Update market price history
        3. Score each listing on price
        4. For listings above threshold: run AI text analysis
        5. For top listings: run image analysis (stock detection + condition)
        6. Calculate composite scores
        7. Send alerts for hot and good deals
        8. Report scan statistics
    """
    from src.db import Database
    from src.ebay_client import EbayClient
    from src.ai_analyzer import AIAnalyzer
    from src.deal_scorer import DealScorer
    from src.price_tracker import PriceTracker, is_for_parts
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
    price_tracker = PriceTracker(db)
    sold_tracker = SoldTracker(db, price_tracker, settings.get("tracking", {}))
    telegram = TelegramNotifier()

    stats = {
        "total_scanned": 0,
        "hot_deals": 0,
        "good_deals": 0,
        "logged": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        # ── Step 1: Search eBay ──────────────────────────────────────
        logger.info("Step 1: Searching eBay Italy...")
        listings = ebay.search_all_categories(
            categories=config["categories"],
            max_pages=settings["ebay"].get("max_pages_per_category", 2),
            results_per_page=settings["ebay"].get("results_per_page", 50),
        )
        stats["total_scanned"] = len(listings)
        logger.info("Found %d listings across all categories", len(listings))

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

        # ── Step 2: Update Market Prices ─────────────────────────────
        logger.info("Step 2: Updating market price history...")
        market_prices = price_tracker.update_market_prices(listings)

        # ── Step 3: Score on Price & Filter ──────────────────────────
        logger.info("Step 3: Scoring listings on price...")
        global_min_discount = settings["scanning"].get("min_discount_for_ai", 15)
        max_ai_analyses = settings["ai"].get("max_text_analyses_per_run", 50)

        # Manually-curated new/street prices — used as a baseline ceiling so an
        # inflated eBay asking-median can't invent a deal (see the A57 case).
        reference_prices = db.get_reference_prices()

        candidates = []
        for listing in listings:
            # Drop "for parts / not working" units before they reach the funnel
            if is_for_parts(listing):
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

            avg_price = market_prices.get(product_key)
            if not avg_price:
                stats["skipped"] += 1
                continue

            # Per-category focus knobs
            cat_cfg = category_config.get(listing.get("category", ""), {})
            priority = cat_cfg.get("priority", 1.0)
            cat_min_discount = cat_cfg.get("min_discount", global_min_discount)
            cat_price_scores = cat_cfg.get("price_scores")  # None => global curve

            # Anchor the baseline to the curated new price (a hard ceiling): a
            # used item can't be worth more than its current new price.
            ref = reference_prices.get(product_key)
            new_price = float(ref["new_price"]) if ref and ref.get("new_price") else None
            baseline = avg_price
            baseline_source = "eBay market"
            if new_price and 0 < new_price < avg_price:
                baseline = new_price
                baseline_source = "new price"

            # Calculate preliminary discount using Total Price (Price + Shipping)
            total_price = listing.get("total_price", listing["price"])
            price_score, discount_pct, is_suspicious = scorer.calculate_price_score(
                total_price, baseline, cat_price_scores
            )

            if discount_pct < cat_min_discount:
                db.upsert_listing(listing)
                stats["skipped"] += 1
                continue

            # Skip if already alerted
            if db.was_alerted(listing["item_id"]):
                stats["skipped"] += 1
                continue

            candidates.append({
                "listing": listing,
                "product_key": product_key,
                "avg_price": baseline,          # effective baseline used for scoring
                "ebay_avg": avg_price,          # raw eBay asking baseline
                "new_price": new_price,         # curated new price, if any
                "baseline_source": baseline_source,
                "price_score": price_score,
                "discount_pct": discount_pct,
                "is_suspicious": is_suspicious,
                "priority": priority,
                # Focus ranking: a high-priority category with a strong price
                # score wins AI budget over a low-priority one, regardless of
                # raw discount.
                "rank_score": price_score * priority,
            })

        # Allocate the limited AI budget to the highest-priority strong deals.
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)
        candidates = candidates[:max_ai_analyses]
        logger.info(
            "Step 3 complete: %d candidates for AI analysis (out of %d)",
            len(candidates), stats["total_scanned"],
        )

        # ── Step 4: AI Text Analysis ─────────────────────────────────
        logger.info("Step 4: Running AI text analysis...")
        hot_deals: list[dict] = []
        good_deals: list[dict] = []
        max_image_analyses = settings["ai"].get("max_image_analyses_per_run", 20)
        image_analyses_done = 0

        for candidate in candidates:
            listing = candidate["listing"]
            try:
                # Get full item details for description
                details = ebay.get_item_details(listing["item_id"])
                if details:
                    listing.update(details)

                # Enrich listing with price context for the AI prompt
                listing["avg_price"] = candidate["avg_price"]
                listing["discount_pct"] = candidate["discount_pct"]

                # Run AI text analysis (Groq)
                text_analysis = ai.analyze_listing(listing)

                # Instantly drop accessories/boxes/incomplete items caught by Groq
                if not text_analysis.get("is_complete_item", True):
                    logger.info("Skipping %s: Groq flagged as incomplete/accessory", listing["item_id"])
                    stats["skipped"] += 1
                    continue
                
                if text_analysis.get("condition_assessment") in ["parts_only", "damaged"]:
                    logger.info("Skipping %s: Groq flagged as damaged/parts", listing["item_id"])
                    stats["skipped"] += 1
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

                # ── Step 5: Image Analysis (Top Listings Only) ───────
                image_score = 50  # Default neutral
                photo_type = "uncertain"
                stock_penalty = 0
                image_analysis = {}

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

                # Full image analysis for top candidates
                if image_analyses_done < max_image_analyses and image_url:
                    image_analysis = ai.analyze_image(
                        image_url,
                        condition=listing.get("condition", ""),
                        description_excerpt=listing.get("short_description", ""),
                    )
                    image_analyses_done += 1

                    photo_type = image_analysis.get("photo_type", "uncertain")
                    if photo_type == "real":
                        match = image_analysis.get("condition_matches_description")
                        if match is True:
                            image_score = 85
                        elif match is False:
                            image_score = 30
                        else:
                            image_score = 60
                    elif photo_type == "stock":
                        image_score = 20
                    else:
                        image_score = 40

                elif quick_check.get("suspicious"):
                    photo_type = "stock"
                    image_score = 25

                # Stock photo penalty
                stock_penalty = scorer.get_stock_photo_penalty(
                    photo_type, listing.get("condition", "")
                )

                # ── Step 6: Composite Score ──────────────────────────
                deal_score = scorer.calculate_composite_score(
                    price_score=candidate["price_score"],
                    condition_score=condition_score,
                    seller_score=seller_score,
                    image_score=image_score,
                    listing_quality_score=listing_quality,
                    stock_photo_penalty=stock_penalty,
                )

                deal_class = scorer.classify_deal(deal_score)

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
                    "discount_pct": candidate["discount_pct"],
                    "is_suspicious": candidate["is_suspicious"],
                    "seller_trust_level": seller_trust_level,
                    "deal_class": deal_class,
                }

                # Save to database
                db.upsert_listing(listing)
                db.save_analysis({
                    "listing_id": listing["item_id"],
                    "deal_score": deal_score,
                    "price_score": candidate["price_score"],
                    "condition_score": condition_score,
                    "seller_score": seller_score,
                    "image_score": image_score,
                    "ai_assessment": full_analysis,
                })

                # ── Step 7: Send Alerts ──────────────────────────────
                if deal_class == "hot_deal":
                    logger.info(
                        "🔥 HOT DEAL: %s — €%.0f (%s%% off) — Score: %d",
                        listing["title"][:40], listing["price"],
                        candidate["discount_pct"], deal_score,
                    )
                    telegram.send_hot_deal_alert(listing, full_analysis, scores)
                    db.mark_alerted(listing["item_id"])
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
                    good_deals.append({
                        "listing": listing,
                        "analysis": full_analysis,
                        "scores": scores,
                    })
                    stats["good_deals"] += 1

                elif deal_class == "log_only":
                    stats["logged"] += 1

                else:
                    stats["skipped"] += 1

            except Exception as e:
                logger.error("Error analyzing %s: %s", listing.get("item_id", "?"), e)
                stats["errors"] += 1

        # ── Send Batched Good Deals ──────────────────────────────────
        if good_deals and settings["telegram"].get("batch_good_deals", True):
            telegram.send_deal_summary(good_deals)
            for deal in good_deals:
                db.mark_alerted(deal["listing"]["item_id"])

        # ── Step 9: Resale tracking maintenance (never breaks a scan) ─
        try:
            ended = sold_tracker.reconcile_ended(ebay)
            sold_tracker.rebuild_resale_stats()
            sold_tracker.cleanup()
            logger.info("Resale maintenance: finalized %d ended listing(s)", ended)
        except Exception as e:
            logger.error("Resale maintenance failed: %s", e)

        # ── Step 8: Report ───────────────────────────────────────────
        if settings["telegram"].get("send_scan_report", True):
            telegram.send_scan_report(stats)

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
