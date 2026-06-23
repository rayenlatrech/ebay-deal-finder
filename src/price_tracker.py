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

# ── Family anchors (ordered; first hit wins) ─────────────────────────────
# Detection is split from key-building: each family applies its own variant /
# chip / screen-size logic in the dedicated _*_key builders below, because a
# single regex can't robustly handle e.g. "iPhone Air", "MacBook 15 M4", or
# "Switch 2" where the price-defining token sits anywhere in the title.
FAMILY_ANCHORS = [
    ("iphone", re.compile(r"\biphone\b", re.IGNORECASE)),
    ("galaxy", re.compile(r"\bgalaxy\b", re.IGNORECASE)),
    ("macbook", re.compile(r"\bmacbook\b", re.IGNORECASE)),
    ("ipad", re.compile(r"\bipad\b", re.IGNORECASE)),
    ("ps5", re.compile(r"(?:\bps5\b|playstation\s*5)", re.IGNORECASE)),
    ("xbox", re.compile(r"xbox\s*series", re.IGNORECASE)),
    ("switch", re.compile(r"\bswitch\b", re.IGNORECASE)),
    ("airpods", re.compile(r"\bairpods\b", re.IGNORECASE)),
]

# Apple-silicon chip — the #1 MacBook/iPad price driver. Extracted globally
# from the whole title because sellers put it before OR after the screen size.
CHIP_PATTERN = re.compile(r"\bm([1-5])(?:\s*(pro|max|ultra))?\b", re.IGNORECASE)

# Screen sizes, normalised (Apple rebranded 10.9"->11", 12.9"->13"). The
# negative lookahead stops "16 GB" of RAM/storage being read as a 16" screen.
MAC_SIZE_PATTERN = re.compile(r"\b(13|14|15|16)\b(?!\s*(?:gb|tb))", re.IGNORECASE)
IPAD_SIZE_13 = re.compile(r"\b(?:12\.9|13)\b(?!\s*(?:gb|tb))", re.IGNORECASE)
IPAD_SIZE_11 = re.compile(r"\b(?:10\.9|11)\b(?!\s*(?:gb|tb))", re.IGNORECASE)

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
        t = title.lower().strip()

        family, fam_start = self._detect_family(t)
        if not family:
            if epid:
                return (f"epid_{epid}", 0.9)
            return ("", 0.0)

        core_key, variant_suffix = self._build_key(family, t)
        if not core_key:
            # Recognized the family but couldn't pin the model (e.g. "iPhone"
            # with no number). Fall back to eBay's catalog id if present.
            if epid:
                return (f"epid_{epid}", 0.9)
            return ("", 0.0)

        final_key = re.sub(r"_+", "_", core_key).strip("_")

        # ── Accessory / part disambiguation (position + context aware) ──
        is_part, recognized_variant = self._is_accessory_listing(
            t, fam_start, variant_suffix
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

    # ── Family detection & key building ──────────────────────────────

    @staticmethod
    def _detect_family(t: str) -> Tuple[Optional[str], int]:
        for family, anchor in FAMILY_ANCHORS:
            m = anchor.search(t)
            if m:
                return family, m.start()
        return None, len(t)

    @staticmethod
    def _storage_suffix(t: str) -> str:
        m = STORAGE_PATTERN.search(t)
        return f"_{m.group(1)}{m.group(2).lower()}" if m else ""

    @staticmethod
    def _chip_suffix(t: str) -> str:
        m = CHIP_PATTERN.search(t)
        if not m:
            return ""
        tier = m.group(2)
        return f"_m{m.group(1)}" + (f"_{tier.lower()}" if tier else "")

    def _build_key(self, family: str, t: str) -> Tuple[str, str]:
        """Return (core_key, recognized_variant).

        recognized_variant is only set to '_controller'/'_portal' so the
        accessory check treats those as legitimate product buckets; '' otherwise.
        """
        if family == "ps5":
            return self._ps5_key(t)
        if family == "switch":
            return self._switch_key(t)
        builders = {
            "iphone": self._iphone_key,
            "galaxy": self._galaxy_key,
            "macbook": self._mac_key,
            "ipad": self._ipad_key,
            "xbox": self._xbox_key,
            "airpods": self._airpods_key,
        }
        builder = builders.get(family)
        return (builder(t) if builder else ""), ""

    def _iphone_key(self, t: str) -> str:
        # "iPhone Air" is its own model (no number); unify "iPhone 17 Air" with it.
        if re.search(r"\bair\b", t):
            return "iphone_air" + self._storage_suffix(t)
        m = re.search(r"iphone\s*(\d{1,2})", t)
        if not m:
            return ""
        num = m.group(1)
        base = f"iphone_{num}"
        if re.search(r"pro\s*max", t):
            base += "_pro_max"
        elif re.search(r"\bpro\b", t):
            base += "_pro"
        elif re.search(r"\bplus\b|\+", t):
            base += "_plus"
        elif re.search(r"\bmini\b", t):
            base += "_mini"
        elif re.search(rf"\b{num}e\b", t):  # 16e / 17e budget models (attached 'e')
            base += "_e"
        return base + self._storage_suffix(t)

    def _galaxy_key(self, t: str) -> str:
        m = re.search(
            r"galaxy\s*(s\d{1,2}|a\d{1,2}|note\s*\d{1,2}|z\s*(?:fold|flip)\s*\d?)", t
        )
        if not m:
            return ""
        base = "galaxy_" + re.sub(r"\s+", "_", m.group(1).strip())
        if re.search(r"\bedge\b", t):
            base += "_edge"
        elif re.search(r"ultra", t):
            base += "_ultra"
        elif re.search(r"plus|\+", t):
            base += "_plus"
        elif re.search(r"\bfe\b", t):
            base += "_fe"
        return base + self._storage_suffix(t)

    def _mac_key(self, t: str) -> str:
        m = re.search(r"macbook\s*(pro|air)?", t)
        line = m.group(1) if (m and m.group(1)) else ""
        base = "macbook" + (f"_{line}" if line else "")
        size = MAC_SIZE_PATTERN.search(t)
        size_suffix = f"_{size.group(1)}" if size else ""
        return base + self._chip_suffix(t) + size_suffix + self._storage_suffix(t)

    def _ipad_key(self, t: str) -> str:
        m = re.search(r"ipad\s*(pro|air|mini)?", t)
        line = m.group(1) if (m and m.group(1)) else ""
        extra = self._chip_suffix(t)
        if line in ("pro", "air"):
            if IPAD_SIZE_13.search(t):
                extra += "_13"
            elif IPAD_SIZE_11.search(t):
                extra += "_11"
        elif not line:
            # Bare iPad: the generation number is the price differentiator.
            gm = re.search(r"ipad\s*(\d{1,2})", t)
            if gm:
                extra += f"_gen{gm.group(1)}"
        base = "ipad" + (f"_{line}" if line else "")
        return base + extra + self._storage_suffix(t)

    def _ps5_key(self, t: str) -> Tuple[str, str]:
        if re.search(r"portal", t):
            variant, recognized = "_portal", "_portal"
        elif re.search(r"controller|dualsense", t):
            variant, recognized = "_controller", "_controller"
        elif re.search(r"\bpro\b", t):
            variant, recognized = "_pro", ""
        elif re.search(r"slim|digital", t):
            variant, recognized = "_slim", ""
        else:
            variant, recognized = "_console", ""
        return "ps5" + variant + self._storage_suffix(t), recognized

    def _xbox_key(self, t: str) -> str:
        m = re.search(r"xbox\s*series\s*(x|s)", t)
        if not m:
            return ""
        base = f"xbox_{m.group(1)}"
        if re.search(r"\bdigital\b", t):
            base += "_digital"
        return base + self._storage_suffix(t)

    def _switch_key(self, t: str) -> Tuple[str, str]:
        recognized = ""
        if re.search(r"switch\s*2\b", t):
            variant = "_2"
        elif re.search(r"oled", t):
            variant = "_oled"
        elif re.search(r"lite", t):
            variant = "_lite"
        elif re.search(r"joy-?con|controller", t):
            variant, recognized = "_controller", "_controller"
        else:
            variant = ""
        return "switch" + variant + self._storage_suffix(t), recognized

    def _airpods_key(self, t: str) -> str:
        if re.search(r"\bmax\b", t):
            return "airpods_max"
        if re.search(r"pro\s*3", t):
            return "airpods_pro_3"
        if re.search(r"pro\s*2", t):
            return "airpods_pro_2"
        if re.search(r"\bpro\b", t):
            return "airpods_pro"
        if re.search(r"\b4\b", t):
            anc = "_anc" if re.search(r"anc|noise|cancellaz|cancelling|riduzione", t) else ""
            return "airpods_4" + anc
        if re.search(r"\b3\b", t):
            return "airpods_3"
        if re.search(r"\b2\b", t):
            return "airpods_2"
        return "airpods"

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
