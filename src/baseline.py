"""Market baselines from a rolling window of tracked listings.

The legacy baseline priced a model from only the listings posted since the
previous scan (1-3 per model), smoothed over the last 8 scans (~32 hours), and
then let retail x 0.75 override it whenever lower. On recent phones that put
the baseline 15-25% under the real market, so genuine deals were missed.

Here the market price comes from every listing tracked over the last N days
(each counted once, at its current price). Where the home market is thin it is
topped up from other EU marketplaces, rescaled by their measured price gap.
The retail price becomes a safety net instead of a cap: it fills in when there
are too few listings, and caps a used price that would exceed a new one.
"""

import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from src.price_tracker import PriceTracker, baseline_key

_STORAGE_SUFFIX = re.compile(r"_(\d+)(gb|tb)$")

# Plausible range for a bigger storage tier's price relative to a smaller one.
MIN_PREMIUM = 1.0
MAX_PREMIUM = 1.6


@dataclass
class MarketPrice:
    """A bucket's market price and the evidence behind it."""

    value: float
    n: int              # listings behind the value (home + rescaled foreign)
    n_home: int
    n_foreign: int


def _robust_p35(prices: list[float]) -> Optional[float]:
    """The baseline statistic: IQR outlier rejection, then the 35th percentile.

    The low percentile (not the median) counters asking-price skew: overpriced
    listings sit unsold and accumulate, real trades happen lower.
    """
    if not prices:
        return None
    return PriceTracker._percentile(PriceTracker._reject_outliers(prices), PriceTracker.BASELINE_PERCENTILE)


def storage_gb(product_key: str) -> Optional[int]:
    """Storage tier of a product key in GB, or None if it has none."""
    m = _STORAGE_SUFFIX.search(product_key)
    if not m:
        return None
    return int(m.group(1)) * (1024 if m.group(2) == "tb" else 1)


def model_key(product_key: str) -> str:
    """Product key without its storage tier ("iphone_17_256gb" -> "iphone_17")."""
    return _STORAGE_SUFFIX.sub("", product_key)


def _family(product_key: str) -> str:
    return product_key.split("_", 1)[0]


def normalize_key(product_key: str) -> str:
    """Bring a key stored by an older classifier in line with the current one.

    Tracked rows keep the key they were recorded with, so the window holds
    "_126gb" typos, "_8gb" RAM figures and "_1000gb" next to "_1tb". Re-snapping
    the storage suffix merges them into the real tier (or drops a non-tier),
    so they count toward the right bucket instead of forming phantom ones.
    """
    m = _STORAGE_SUFFIX.search(product_key)
    if not m:
        return product_key
    in_gb = int(m.group(1)) * (1024 if m.group(2) == "tb" else 1)
    tier = PriceTracker._snap_storage(in_gb)
    root = product_key[: m.start()]
    if tier is None:
        return root
    return root + (f"_{tier // 1024}tb" if tier >= 1024 else f"_{tier}gb")


# ── Market prices ────────────────────────────────────────────────────────

def compute_window_prices(
    rows: Iterable[dict[str, Any]],
    home: str,
    topup_markets: Iterable[str] = (),
    min_listings: int = 8,
    max_gap_pct: float = 20.0,
    min_ratio_models: int = 5,
) -> tuple[dict[str, MarketPrice], dict[tuple[str, str], float]]:
    """Market price per bucket (``product_key#band``) from tracked listings.

    Args:
        rows: tracked listings — product_key, condition_band, price, marketplace
            (a missing marketplace is the home one, from before it was recorded).
        home: the marketplace deals are judged against.
        topup_markets: marketplaces allowed to fill in when home is thin.
        min_listings: home listings that make a bucket self-sufficient.
        max_gap_pct: a foreign market whose rescaled price for this bucket is
            further than this from the home price is ignored for it — a gap
            that large means fakes or misread listings, not a cheaper market.
        min_ratio_models: models needed to measure a market's price gap.

    Returns:
        (prices, ratios) — ratios maps (market, band) to foreign/home price.
    """
    topup_markets = tuple(m for m in topup_markets if m != home)
    groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        market = r.get("marketplace") or home
        if market != home and market not in topup_markets:
            continue
        try:
            price = float(r.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if price <= 0 or not r.get("product_key") or not r.get("condition_band"):
            continue
        bucket = baseline_key(normalize_key(r["product_key"]), r["condition_band"])
        groups.setdefault((bucket, market), []).append(price)

    # Each foreign market's typical price relative to home, per band, measured
    # only on buckets where both sides are well sampled.
    ratios: dict[tuple[str, str], float] = {}
    for market in topup_markets:
        for band in ("used", "new"):
            samples = []
            for (bucket, m), prices in groups.items():
                if m != market or not bucket.endswith("#" + band) or len(prices) < min_listings:
                    continue
                home_prices = groups.get((bucket, home), [])
                if len(home_prices) >= min_listings:
                    samples.append(_robust_p35(prices) / _robust_p35(home_prices))
            if len(samples) >= min_ratio_models:
                ratios[(market, band)] = statistics.median(samples)

    buckets = {bucket for bucket, _ in groups}
    result: dict[str, MarketPrice] = {}
    for bucket in buckets:
        band = bucket.rsplit("#", 1)[1]
        home_prices = groups.get((bucket, home), [])
        pool = list(home_prices)
        n_foreign = 0
        if len(home_prices) < min_listings:
            home_p35 = _robust_p35(home_prices) if len(home_prices) >= 3 else None
            for market in topup_markets:
                ratio = ratios.get((market, band))
                foreign = groups.get((bucket, market), [])
                if not ratio or not foreign:
                    continue
                scaled = [p / ratio for p in foreign]
                if home_p35:
                    gap = abs(_robust_p35(scaled) - home_p35) / home_p35 * 100
                    if gap > max_gap_pct:
                        continue
                elif len(foreign) < min_listings:
                    # Nothing at home to check it against: only a foreign
                    # sample big enough to stand on its own is trusted.
                    continue
                pool.extend(scaled)
                n_foreign += len(scaled)
        value = _robust_p35(pool)
        if value:
            result[bucket] = MarketPrice(round(value, 2), len(pool), len(home_prices), n_foreign)
    return result, ratios


# ── Storage-aware reference prices ───────────────────────────────────────

def learn_storage_premiums(
    prices: dict[str, MarketPrice], min_listings: int = 8, min_models: int = 2,
) -> dict[tuple[str, int, int], float]:
    """How much more each storage tier sells for than a smaller one, per family.

    Measured on used buckets where both tiers of the same model are well
    sampled, then the median across models — so one entered base-model retail
    price can be scaled to every tier instead of capping a 512GB unit at the
    128GB price.
    """
    by_model: dict[str, dict[int, float]] = {}
    for bucket, mp in prices.items():
        key, band = bucket.rsplit("#", 1)
        tier = storage_gb(key)
        if band != "used" or tier is None or mp.n < min_listings:
            continue
        by_model.setdefault(model_key(key), {})[tier] = mp.value

    samples: dict[tuple[str, int, int], list[float]] = {}
    for model, tiers in by_model.items():
        ordered = sorted(tiers)
        for i, low in enumerate(ordered):
            for high in ordered[i + 1:]:
                samples.setdefault((_family(model), low, high), []).append(tiers[high] / tiers[low])
    premiums = {k: statistics.median(v) for k, v in samples.items() if len(v) >= min_models}
    # More storage is never cheaper, and never worth +60% on its own; outside
    # that range the "premium" is a generation mix-up in a loosely keyed model.
    return {k: v for k, v in premiums.items() if MIN_PREMIUM <= v <= MAX_PREMIUM}


def reference_new_price(
    product_key: str,
    references: dict[str, dict],
    prices: dict[str, MarketPrice],
    premiums: dict[tuple[str, int, int], float],
    min_base_listings: int = 3,
) -> Optional[float]:
    """Retail price for this exact storage tier, or None if it can't be known.

    An entry for the exact tier wins. Otherwise a model-level entry is taken as
    the price of the model's smallest tier on the market, and scaled up by the
    learned storage premium. Without a premium the price is withheld rather
    than misapplied to a bigger tier.
    """
    exact = references.get(product_key)
    if exact and exact.get("new_price"):
        return float(exact["new_price"])

    model = model_key(product_key)
    ref = references.get(model)
    if not ref or not ref.get("new_price"):
        return None
    price = float(ref["new_price"])
    tier = storage_gb(product_key)
    if tier is None:
        return price

    tiers_seen = sorted({
        storage_gb(b.rsplit("#", 1)[0]) for b, mp in prices.items()
        if model_key(b.rsplit("#", 1)[0]) == model and storage_gb(b.rsplit("#", 1)[0])
        and mp.n_home >= min_base_listings
    })
    base_tier = tiers_seen[0] if tiers_seen else tier
    if tier == base_tier:
        return price
    premium = premiums.get((_family(model), base_tier, tier))
    return price * premium if premium else None


# ── The decision ─────────────────────────────────────────────────────────

@dataclass
class Baseline:
    value: float
    source: str           # market | blend | thin market | retail estimate | new-price cap
    n: int
    new_price: Optional[float]


def resolve_baseline(
    market: Optional[MarketPrice],
    new_price: Optional[float],
    band_factor: float,
    min_listings: int = 8,
    min_blend_listings: int = 3,
) -> Optional[Baseline]:
    """Pick the price a listing is judged against.

    The retail price may only ever LOWER a baseline — cancel a deal, never
    invent one. List prices sit well above street prices (a sealed Galaxy
    sells far under Samsung's list price), so letting retail raise a thin
    market would flag ordinary listings as bargains.

    Enough listings → the market, capped at the new price (a unit can't be
    worth more than new). A few → the market, pulled toward the retail
    estimate only when that is lower, more strongly the thinner the market.
    One or two listings aren't a market: only a lower retail estimate is
    used, otherwise the listing isn't judged (None).
    """
    if not market:
        return None
    estimate = new_price * band_factor if new_price else None
    n = market.n

    if n >= min_listings:
        if new_price and market.value > new_price:
            return Baseline(round(new_price, 2), "new-price cap", n, new_price)
        return Baseline(market.value, "market", n, new_price)

    lower_estimate = estimate if estimate and estimate < market.value else None
    if n >= min_blend_listings:
        if lower_estimate is None:
            return Baseline(market.value, "thin market", n, new_price)
        w = n / min_listings
        value = w * market.value + (1 - w) * lower_estimate
        return Baseline(round(value, 2), "blend", n, new_price)
    if lower_estimate is not None:
        return Baseline(round(lower_estimate, 2), "retail estimate", n, new_price)
    return None
