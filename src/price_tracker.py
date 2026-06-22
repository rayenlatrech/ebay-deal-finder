"""Market price tracking — maintains robust baselines and classifies listings.

The market baseline is computed *before* any AI is involved, so the integrity of
the funnel depends entirely on this module:

    1. Classify each listing into a product bucket (epid → regex precedence).
    2. Drop "for parts / not working" units so junk can't poison the baseline.
    3. Reject statistical outliers (IQR) inside each bucket.
    4. Blend the current robust median with recent history for a stable baseline.
"""

import logging
import re
import statistics
from collections import Counter
from typing import Optional, Tuple

from src.db import Database

logger = logging.getLogger(__name__)

# ── Base product family detection ────────────────────────────────────────
FAMILY_PATTERNS = {
    "iphone": re.compile(r"iphone\s*(\d{1,2})", re.IGNORECASE),
    "galaxy": re.compile(r"galaxy\s*(s\d{1,2}|a\d{1,2}|note\s*\d{1,2}|z\s*(?:fold|flip)\s*\d?)", re.IGNORECASE),
    "macbook": re.compile(r"macbook\s*(pro|air)?\s*(m\d(?:\s*pro|\s*max)?|\d{4})?", re.IGNORECASE),
    "ipad": re.compile(r"ipad\s*(pro|air|mini)?\s*(m\d|\d{1,2}th|\d{4})?", re.IGNORECASE),
    "ps5": re.compile(r"(?:ps5|playstation\s*5)", re.IGNORECASE),
    "xbox": re.compile(r"xbox\s*series\s*(x|s)", re.IGNORECASE),
    "switch": re.compile(r"switch", re.IGNORECASE),
    "airpods": re.compile(r"airpods", re.IGNORECASE),
}

# ── Sub-variant classification (modifiers that change the core product) ───
# Evaluated top-to-bottom; first match wins, so more specific variants
# (e.g. "pro max" before "pro") MUST come first.
VARIANT_PATTERNS = {
    "iphone": [
        (re.compile(r"pro\s*max", re.IGNORECASE), "_pro_max"),
        (re.compile(r"\bpro\b", re.IGNORECASE), "_pro"),
        (re.compile(r"\bplus\b|\+", re.IGNORECASE), "_plus"),
        (re.compile(r"\bmini\b", re.IGNORECASE), "_mini"),
    ],
    "galaxy": [
        (re.compile(r"ultra", re.IGNORECASE), "_ultra"),
        (re.compile(r"plus|\+", re.IGNORECASE), "_plus"),
        (re.compile(r"\bfe\b", re.IGNORECASE), "_fe"),
    ],
    "ps5": [
        (re.compile(r"portal", re.IGNORECASE), "_portal"),
        (re.compile(r"(controller|dualsense)", re.IGNORECASE), "_controller"),
        (re.compile(r"\bpro\b", re.IGNORECASE), "_pro"),
        (re.compile(r"(slim|digital)", re.IGNORECASE), "_slim"),
        (re.compile(r"(console|standard)", re.IGNORECASE), "_console"),
    ],
    "switch": [
        (re.compile(r"oled", re.IGNORECASE), "_oled"),
        (re.compile(r"lite", re.IGNORECASE), "_lite"),
        (re.compile(r"joycon|joy-con|controller", re.IGNORECASE), "_controller"),
    ],
    "airpods": [
        (re.compile(r"max", re.IGNORECASE), "_max"),
        (re.compile(r"\bpro\b", re.IGNORECASE), "_pro"),
    ],
}

# Accessory / spare-part nouns. Presence alone is NOT enough — see
# _is_accessory_listing for the position/context logic that avoids nuking
# legitimate bundles like "MacBook Pro M1 with charger".
ACCESSORY_KEYWORDS = re.compile(
    r"(charger|caricabatterie|caricatore|alimentatore|cable|cavo|screen|schermo|"
    r"display|lcd|vetro|glass|battery|batteria|motherboard|scheda|case|cover|"
    r"box|scatola|empty|vuota|vuoto|parts|ricambi|ricambio|adapter|adattatore)",
    re.IGNORECASE,
)

# A bundle word *before* the accessory noun => the accessory is an included
# extra, not the product itself ("MacBook ... with charger").
BUNDLE_CONTEXT = re.compile(r"(\bwith\b|\bcon\b|inclus|compres|completo\s+di|\+)", re.IGNORECASE)

# A "for / replacement" word => the listing is a part FOR a device, not the
# device ("ricambio per iPhone 13", "replacement screen for ...").
FOR_CONTEXT = re.compile(r"(ricambio|ricambi|replacement|compatibile|compatible|\bper\b|\bfor\b)", re.IGNORECASE)

# Storage size extraction
STORAGE_PATTERN = re.compile(r"(\d+)\s*(gb|tb)", re.IGNORECASE)

# eBay conditionId for "For parts or not working"
FOR_PARTS_CONDITION_ID = "7000"


def is_for_parts(listing: dict) -> bool:
    """True if eBay marks the listing as 'for parts / not working'.

    These units are real but priced like scrap, so they must never enter the
    market baseline nor be surfaced as deals.
    """
    if str(listing.get("condition_id", "") or "") == FOR_PARTS_CONDITION_ID:
        return True
    cond = (listing.get("condition", "") or "").lower()
    return any(s in cond for s in ("parts", "non funzionante", "ricambi", "not working"))


class PriceTracker:
    """Tracks market prices and classifies listings into product buckets."""

    # Confidence required for a listing to influence/consume a baseline.
    MIN_CONFIDENCE = 0.8
    # Robust current-run samples needed to trust a brand-new baseline.
    MIN_SAMPLES = 3
    # How many recent saved medians to blend into the baseline.
    HISTORY_WINDOW = 8

    def __init__(self, db: Database) -> None:
        """Initialize with database client."""
        self.db = db

    # ── Baseline retrieval ────────────────────────────────────────────

    def get_market_price(self, product_key: str) -> Optional[float]:
        """Return the most recent stored median for a product, or None."""
        history = self.db.get_price_history(product_key)
        if not history:
            return None
        return float(history[0]["avg_price"])

    def calculate_discount(self, price: float, market_price: float) -> float:
        """Calculate discount percentage of price vs market baseline."""
        if market_price <= 0:
            return 0.0
        return round(((market_price - price) / market_price) * 100, 1)

    # ── Baseline computation ──────────────────────────────────────────

    @staticmethod
    def _reject_outliers(prices: list[float]) -> list[float]:
        """Drop prices outside the 1.5×IQR fence.

        This is the safety net that keeps the baseline honest even when title
        classification misfires: a single mis-bucketed €150 part or €900 bundle
        can no longer drag a €400 console median around.
        """
        if len(prices) < 4:
            return list(prices)
        ordered = sorted(prices)
        q1, _, q3 = statistics.quantiles(ordered, n=4)
        iqr = q3 - q1
        if iqr == 0:
            return ordered
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        kept = [p for p in ordered if low <= p <= high]
        return kept or ordered

    @staticmethod
    def _purge_minority_epids(pairs: list[tuple[float, str]]) -> list[float]:
        """Use eBay's catalog ePID to purge listings the regex mis-bucketed.

        ePID is eBay's own authoritative product identity, so it's the most
        reliable signal we have for free. Within a regex bucket, if the
        catalog-identified listings overwhelmingly point at one product, any
        listing carrying a *different* ePID is almost certainly a different
        product that the title regex grouped here by mistake (e.g. a Portal
        that slipped into the console bucket). We drop those.

        Listings without an ePID are kept — they get the benefit of the doubt
        and still have to survive the IQR filter downstream.
        """
        epids = [e for _, e in pairs if e]
        # Need a meaningful number of catalog-identified items to judge.
        if len(epids) < 4:
            return [p for p, _ in pairs]

        top_epid, top_count = Counter(epids).most_common(1)[0]
        # Only act when there is a clear dominant catalog product.
        if top_count / len(epids) < 0.5:
            return [p for p, _ in pairs]

        return [p for p, e in pairs if e == top_epid or not e]

    def update_market_prices(self, listings: list[dict]) -> dict[str, float]:
        """Compute and store robust median baselines for the current scan.

        Pipeline per bucket: drop for-parts units → reject IQR outliers →
        median → persist → blend with recent history. Only buckets with enough
        support (current samples or prior history) are returned.
        """
        # bucket -> list of (price, epid)
        groups: dict[str, list[tuple[float, str]]] = {}

        for listing in listings:
            if is_for_parts(listing):
                continue
            key, confidence = self.generate_product_key(
                listing.get("title", ""),
                listing.get("category", ""),
                listing.get("epid", ""),
                listing.get("condition_id", ""),
            )
            price = float(listing.get("total_price") or listing.get("price") or 0)
            if key and confidence >= self.MIN_CONFIDENCE and price > 0:
                epid = str(listing.get("epid", "") or "")
                groups.setdefault(key, []).append((price, epid))

        baselines: dict[str, float] = {}
        for key, pairs in groups.items():
            # Layer 1 (structured): purge catalog-confirmed wrong products.
            prices = self._purge_minority_epids(pairs)
            # Layer 2 (statistical): trim price outliers.
            robust = self._reject_outliers(prices)
            current_median = statistics.median(robust) if robust else None

            if current_median is not None:
                self.db.save_price_point(key, round(current_median, 2), len(robust))

            # Blend across recent runs (history already includes the point we
            # just saved) to smooth out thin, noisy single-scan samples.
            history = self.db.get_price_history(key)
            hist_points = [
                float(h["avg_price"]) for h in history[: self.HISTORY_WINDOW]
                if h.get("avg_price") is not None
            ]
            pool = hist_points or ([current_median] if current_median is not None else [])
            if not pool:
                continue
            baseline = statistics.median(pool)

            enough_now = len(robust) >= self.MIN_SAMPLES
            enough_history = len(hist_points) >= 2
            if enough_now or enough_history:
                baselines[key] = round(baseline, 2)
            else:
                logger.debug(
                    "Skipping %s baseline: thin support (n=%d, history=%d)",
                    key, len(robust), len(hist_points),
                )

        logger.info("Computed robust baselines for %d products", len(baselines))
        return baselines

    # ── Classification ────────────────────────────────────────────────

    def _is_accessory_listing(
        self, title_lower: str, fam_start: int, variant_suffix: str
    ) -> Tuple[bool, bool]:
        """Decide whether a listing is really a part/accessory, not the device.

        Returns (is_part, is_recognized_variant).

        Rules (in order):
            • Recognized accessory *variants* (ps5_controller, ps5_portal) are
              legitimate, well-defined product buckets — keep them.
            • Accessory noun that LEADS the title → it's the product ("Cover iPhone 13").
            • "ricambio per / replacement for" context → spare part.
            • Accessory noun introduced by a bundle word ("...with charger") →
              an included extra, the device is still the product.
            • Otherwise an accessory noun trailing the device with no bundle
              word ("iPhone 13 schermo") → spare part.
        """
        acc = ACCESSORY_KEYWORDS.search(title_lower)
        if not acc:
            return False, False

        if variant_suffix in ("_controller", "_portal"):
            return False, True

        acc_pos = acc.start()
        if acc_pos < fam_start:
            return True, False
        if FOR_CONTEXT.search(title_lower):
            return True, False
        if BUNDLE_CONTEXT.search(title_lower[:acc_pos]):
            return False, False
        return True, False

    def generate_product_key(
        self,
        title: str,
        category: str = "",
        epid: str = "",
        condition_id: str = "",
    ) -> Tuple[str, float]:
        """Classify a listing into a product bucket with a confidence score.

        Grouping precedence:
            1. eBay regex family + variant + storage (human-readable, stable).
            2. eBay catalog ePID as a fallback when the title can't be parsed —
               it's eBay's own authoritative classification and is free.

        Returns (product_key, confidence 0.0-1.0).
        """
        title_lower = title.lower().strip()

        # ── Stage 1: Family detection ──
        matched_family = None
        fam_start = len(title_lower)
        base_key_parts: list[str] = []

        for family, pattern in FAMILY_PATTERNS.items():
            match = pattern.search(title_lower)
            if match:
                matched_family = family
                fam_start = match.start()
                base_key_parts.append(family)
                for group in match.groups():
                    if group:
                        cleaned = group.strip().replace(" ", "_")
                        if cleaned and cleaned not in base_key_parts:
                            base_key_parts.append(cleaned)
                break

        if not matched_family:
            # Fall back to eBay's catalog id for listings we can't parse.
            if epid:
                return (f"epid_{epid}", 0.9)
            return ("", 0.0)

        # ── Stage 2: Variant classification ──
        variant_suffix = ""
        if matched_family in VARIANT_PATTERNS:
            for pattern, suffix in VARIANT_PATTERNS[matched_family]:
                if pattern.search(title_lower):
                    variant_suffix = suffix
                    break

        # A bare PS5 with no explicit variant is a standard console.
        if matched_family == "ps5" and not variant_suffix:
            variant_suffix = "_console"

        # ── Storage extraction ──
        storage_suffix = ""
        storage_match = STORAGE_PATTERN.search(title_lower)
        if storage_match:
            size = storage_match.group(1)
            unit = storage_match.group(2).lower()
            storage_suffix = f"_{size}{unit}"

        final_key = "_".join(base_key_parts) + variant_suffix + storage_suffix
        final_key = re.sub(r"_+", "_", final_key).strip("_")

        # ── Accessory / part disambiguation (position + context aware) ──
        is_part, recognized_variant = self._is_accessory_listing(
            title_lower, fam_start, variant_suffix
        )

        if recognized_variant:
            confidence = 0.97
        elif is_part:
            # Isolate spare parts in their own low-trust bucket so they neither
            # pollute the baseline nor get surfaced as "deals".
            final_key += "_part"
            confidence = 0.4
        else:
            confidence = 0.95

        return final_key, confidence

    # ── Price trend (now backed by persisted history) ─────────────────

    def get_price_trend(self, product_key: str, days: int = 30) -> str:
        """Analyze price trend over recent history.

        Returns: 'rising', 'falling', 'stable', or 'unknown'
        """
        history = self.db.get_price_history(product_key)

        if len(history) < 3:
            return "unknown"

        recent = [float(h["avg_price"]) for h in history[:3]]
        older = [float(h["avg_price"]) for h in history[3:6]]

        if not older:
            return "unknown"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        change_pct = ((recent_avg - older_avg) / older_avg) * 100

        if change_pct > 5:
            return "rising"
        elif change_pct < -5:
            return "falling"
        else:
            return "stable"
