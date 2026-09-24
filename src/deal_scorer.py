"""Deal scoring engine — calculates composite scores from multiple signals."""

import logging
import re
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# AI red flags that describe a genuinely risky unit rather than cosmetic wear.
# The lookbehinds matter: "unlocked" contains "locked", and Italian "sbloccato"
# (unlocked — a selling point) contains "bloccato" (locked).
SEVERE_FLAG_RE = re.compile(
    r"icloud|activation\s*lock|find\s*my|\bfrp\b|google\s*(?:account|lock)|"
    r"(?<!un)locked|(?<!s)bloccat|blacklist|"
    r"not\s*working|non\s*funzion|does(?:n'?t|\s*not)\s*work|no\s*power|"
    r"non\s*si\s*accende|won'?t\s*turn\s*on|"
    r"broken|\brott[oaie]\b|crack|crepat|incrinat|"
    r"water|liquid|\bacqua\b|swollen|gonfi|"
    r"for\s*parts|ricambi|stolen|rubat|replica|counterfeit|\bfake\b|\bfals[oa]\b",
    re.IGNORECASE,
)

# Entries models emit to mean "nothing to report"; they must not be penalised.
_EMPTY_FLAG_RE = re.compile(
    r"^\s*(?:none|n/?a|no|nothing|nessun[oa]?|none detected|no red flags|-)?\s*\.?\s*$",
    re.IGNORECASE,
)


def _real_entries(values: Any) -> list[str]:
    """Non-empty, meaningful strings from an AI list field."""
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if isinstance(v, (str, int, float)) and not _EMPTY_FLAG_RE.match(str(v))]

# Default "trusted seller" bar, overridable via settings.yaml → seller_trust.
DEFAULT_MIN_POSITIVE_PCT = 95.0
DEFAULT_MIN_SALES = 10


def is_trusted_seller(
    listing: dict[str, Any],
    min_positive_pct: float = DEFAULT_MIN_POSITIVE_PCT,
    min_sales: int = DEFAULT_MIN_SALES,
) -> bool:
    """True if the seller is established *and* well-rated.

    "Established" means a real track record of completed sales, not just a
    perfect score from two transactions — a 100% rating over 1 sale carries far
    less information than 98% over 500. Used as a hard gate on cross-border
    listings (where returns and disputes are harder) and to split alerts into
    trusted vs. other sections.
    """
    try:
        pct = float(listing.get("seller_feedback_percentage") or 0)
        sales = int(listing.get("seller_feedback_score") or 0)
    except (TypeError, ValueError):
        return False
    return pct >= min_positive_pct and sales >= min_sales


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
        # Too-good-to-be-true guard: past a certain discount, a listing is more
        # likely a scam than a bargain, so it must stop earning more points.
        self.suspicious = self.config.get("suspicious_price", {})
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

        # Flag suspiciously low prices. Beyond this point a bigger discount is
        # evidence of a scam, not of a better deal, so cap the reward instead of
        # letting the price score keep climbing to 100.
        threshold = self.suspicious.get("threshold_pct", 50)
        is_suspicious = discount_pct > threshold
        if is_suspicious:
            score = min(score, self.suspicious.get("max_price_score", 60))

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
        is_suspicious: bool = False,
        ai_penalty: int = 0,
    ) -> int:
        """Calculate weighted composite deal score (0-100).

        Args:
            price_score: How good the price discount is (0-100)
            condition_score: AI assessment of condition (0-100)
            seller_score: Seller trust rating (0-100)
            image_score: Image verification score (0-100)
            listing_quality_score: Description/photo completeness (0-100)
            stock_photo_penalty: Negative penalty for stock photos
            is_suspicious: Price is implausibly low — apply the scam penalty
            ai_penalty: Negative penalty from the AI's own verdict (red flags,
                hidden defects, visible damage) — see calculate_ai_penalty
        """
        raw = (
            price_score * self.weights["price"]
            + condition_score * self.weights["condition"]
            + seller_score * self.weights["seller"]
            + image_score * self.weights["image"]
            + listing_quality_score * self.weights["listing_quality"]
        )

        penalty = stock_photo_penalty + min(0, ai_penalty)
        if is_suspicious:
            penalty += self.suspicious.get("penalty", -25)

        final = max(0, min(100, int(raw + penalty)))
        return final

    # ── AI verdict ───────────────────────────────────────────────────

    def calculate_ai_penalty(
        self, text_analysis: dict[str, Any], image_analysis: dict[str, Any]
    ) -> int:
        """Turn the AI's own findings into a score penalty (0 or negative).

        The model reports red flags, functional problems buried in the
        description (not in the title or condition grade), visible damage and a
        direct "is this a good deal" verdict. Until now only its
        condition grade reached the score, so a listing it flagged for iCloud
        lock or a cracked screen could still alert on price alone.

        Condition is already scored separately, so cosmetic wear is only lightly
        penalised; severe risks (locks, not working, damage, parts) are heavy.
        A failed analysis contributes nothing — its defaults aren't findings.
        """
        cfg = self.config.get("ai_signals", {})
        total = 0

        if text_analysis and not text_analysis.get("text_ai_failed"):
            flags = _real_entries(text_analysis.get("red_flags"))
            defects = _real_entries(text_analysis.get("hidden_defects"))

            # Severity is a property of the problem, not of which list the model
            # filed it under: "Face ID not working" is serious either way. The
            # same issue named in both lists is counted once.
            severe = {f.strip().lower() for f in flags + defects if SEVERE_FLAG_RE.search(f)}
            mild_flags = [f for f in flags if not SEVERE_FLAG_RE.search(f)]
            mild_defects = [d for d in defects if not SEVERE_FLAG_RE.search(d)]

            flag_penalty = (len(severe) * cfg.get("severe_flag_penalty", -20)
                            + len(mild_flags) * cfg.get("flag_penalty", -5))
            total += max(flag_penalty, cfg.get("flags_cap", -30))
            total += max(len(mild_defects) * cfg.get("hidden_defect_penalty", -8),
                         cfg.get("hidden_defects_cap", -16))

            if text_analysis.get("is_good_deal") is False:
                total += cfg.get("not_good_deal_penalty", -10)

        if (image_analysis and not image_analysis.get("image_ai_failed")
                and image_analysis.get("photo_type") == "real"):
            damage = _real_entries(image_analysis.get("visible_damage"))
            total += max(len(damage) * cfg.get("visible_damage_penalty", -4),
                         cfg.get("visible_damage_cap", -12))

        return int(max(total, cfg.get("total_cap", -40)))

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
