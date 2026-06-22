"""Deal scoring engine — calculates composite scores from multiple signals."""

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class DealScorer:
    """Scores listings by combining price, condition, seller, image, and quality signals."""

    def __init__(self, config_path: str = "config/scoring.yaml") -> None:
        """Load scoring configuration from YAML."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.weights = self.config["weights"]
        self.thresholds = self.config["thresholds"]
        self.price_scores = self.config["price_scores"]
        self.stock_penalties = self.config["stock_photo_penalties"]
        self.seller_tiers = self.config["seller_tiers"]
        logger.info("Deal scorer loaded with weights: %s", self.weights)


    def calculate_price_score(
        self,
        price: float,
        avg_market_price: float,
        price_scores: list[dict] | None = None,
    ) -> tuple[int, float, bool]:
        """Calculate price score based on discount percentage.

        Args:
            price: The listing's (total) price.
            avg_market_price: The bucket's market baseline.
            price_scores: Optional per-category discount→score curve. When given,
                it overrides the global curve from scoring.yaml — this is how a
                modest discount on a phone can score as high as a steep discount
                on a low-priority accessory.

        Returns:
            Tuple of (score 0-100, discount_pct, is_suspicious)
        """
        if avg_market_price <= 0:
            return 0, 0.0, False

        discount_pct = ((avg_market_price - price) / avg_market_price) * 100

        if discount_pct < 0:
            return 0, discount_pct, False

        # Map discount to score using the per-category curve if supplied,
        # otherwise the global default.
        tiers = price_scores or self.price_scores
        score = 0
        for tier in tiers:
            if discount_pct <= tier["max_discount"]:
                score = tier["score"]
                break

        # Flag suspiciously low prices (>50% off)
        is_suspicious = discount_pct > 50

        return score, round(discount_pct, 1), is_suspicious

    # ── Seller Score ─────────────────────────────────────────────────

    def calculate_seller_score(
        self, feedback_score: int, feedback_percentage: float
    ) -> tuple[int, str]:
        """Calculate seller trust score.

        Returns:
            Tuple of (score 0-100, trust_level string)
        """
        tiers = self.seller_tiers

        # Check bad reviews first (overrides everything)
        if feedback_percentage < tiers["low_bad"].get("max_positive_pct", 95.0):
            return tiers["low_bad"]["score"], "low_bad"

        # Check from highest tier down
        if (
            feedback_score >= tiers["high"]["min_sales"]
            and feedback_percentage >= tiers["high"]["min_positive_pct"]
        ):
            return tiers["high"]["score"], "high"

        if (
            feedback_score >= tiers["medium"]["min_sales"]
            and feedback_percentage >= tiers["medium"]["min_positive_pct"]
        ):
            return tiers["medium"]["score"], "medium"

        return tiers["low_new"]["score"], "low_new"

    # ── Listing Quality Score ────────────────────────────────────────

    def calculate_listing_quality_score(self, listing: dict[str, Any]) -> int:
        """Score listing quality based on completeness.

        Considers: description length, image count, specifics provided.
        """
        score = 30  # Base score

        # Description quality
        description = listing.get("description", "") or ""
        desc_len = len(description)
        if desc_len > 500:
            score += 25
        elif desc_len > 200:
            score += 15
        elif desc_len > 50:
            score += 5

        # Image count
        image_count = listing.get("image_count", 1)
        if image_count >= 6:
            score += 30
        elif image_count >= 4:
            score += 20
        elif image_count >= 2:
            score += 10
        # Single image: no bonus

        # Has item specifics (brand, model, etc.)
        specifics = listing.get("item_specifics", {})
        if len(specifics) >= 5:
            score += 15
        elif len(specifics) >= 2:
            score += 10

        return min(score, 100)

    # ── Composite Score ──────────────────────────────────────────────

    def calculate_composite_score(
        self,
        price_score: int,
        condition_score: int,
        seller_score: int,
        image_score: int,
        listing_quality_score: int,
        stock_photo_penalty: int = 0,
    ) -> int:
        """Calculate weighted composite deal score (0-100).

        Args:
            price_score: How good the price discount is (0-100)
            condition_score: AI assessment of condition (0-100)
            seller_score: Seller trust rating (0-100)
            image_score: Image verification score (0-100)
            listing_quality_score: Description/photo completeness (0-100)
            stock_photo_penalty: Negative penalty for stock photos
        """
        raw = (
            price_score * self.weights["price"]
            + condition_score * self.weights["condition"]
            + seller_score * self.weights["seller"]
            + image_score * self.weights["image"]
            + listing_quality_score * self.weights["listing_quality"]
        )

        # Apply stock photo penalty
        final = max(0, min(100, int(raw + stock_photo_penalty)))
        return final

    # ── Classification ───────────────────────────────────────────────

    def classify_deal(self, score: int) -> str:
        """Classify a deal based on its composite score.

        Returns: 'hot_deal', 'good_deal', 'log_only', or 'skip'
        """
        if score >= self.thresholds["hot_deal"]:
            return "hot_deal"
        elif score >= self.thresholds["good_deal"]:
            return "good_deal"
        elif score >= self.thresholds["log_only"]:
            return "log_only"
        else:
            return "skip"

    def get_stock_photo_penalty(
        self, photo_type: str, condition: str
    ) -> int:
        """Get the scoring penalty for stock/generic photos.

        Args:
            photo_type: 'real', 'stock', or 'uncertain'
            condition: Item condition string
        """
        if photo_type == "real":
            return 0

        is_used = condition.lower() not in ("new", "new with tags", "new with box")

        if photo_type == "stock":
            if is_used:
                return self.stock_penalties["used_item_stock"]
            else:
                return self.stock_penalties["new_item_stock"]
        elif photo_type == "uncertain":
            return self.stock_penalties["uncertain"]

        return 0

    def is_suspicious(self, price: float, avg_price: float) -> bool:
        """Check if a deal is suspiciously cheap (possible scam)."""
        if avg_price <= 0:
            return False
        discount = ((avg_price - price) / avg_price) * 100
        return discount > 50
