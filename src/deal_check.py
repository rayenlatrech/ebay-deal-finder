"""Answer "is this a good deal?" messages sent to the Telegram bot.

The user sends a device, condition and (optionally) a price — free text, or an
eBay link. The reply judges it against the same 14-day market the scanner uses
(baseline.MarketWindow), shows the price spread and the cheapest live listings,
and adds an AI review: what to check before buying, and any warning signs.
"""

import html
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import yaml

from src.ai_analyzer import DATE_RULE
from src.baseline import MarketWindow, normalize_key
from src.price_tracker import PriceTracker, condition_band, is_for_parts

logger = logging.getLogger(__name__)

# ── Parsing the request ──────────────────────────────────────────────────

EBAY_ITEM_RE = re.compile(
    r"https?://(?:www\.|m\.)?ebay\.(it|de|fr|es|com|co\.uk|nl|be|at)/itm/(?:[^/\s?]+/)?(\d{9,15})",
    re.IGNORECASE,
)
DOMAIN_MARKETPLACE = {
    "it": "EBAY_IT", "de": "EBAY_DE", "fr": "EBAY_FR", "es": "EBAY_ES", "com": "EBAY_US",
    "co.uk": "EBAY_GB", "nl": "EBAY_NL", "be": "EBAY_BE", "at": "EBAY_AT",
}

# A price written with a currency: "550€", "€ 550", "550 euro".
CURRENCY_PRICE_RE = re.compile(
    r"€\s*(\d{2,5}(?:[.,]\d{1,2})?)|(\d{2,5}(?:[.,]\d{1,2})?)\s*(?:€|eur(?:o|os)?\b)",
    re.IGNORECASE,
)
# A bare number that could be a price: not a storage size, percentage, screen
# size or year. Only used when no number carries a currency.
BARE_NUMBER_RE = re.compile(
    r"(?<![\w.,])(\d{2,5})(?![\w.,%\"']|\s*(?:gb|tb|go|mah|hz|pollici|inch|zoll|mm|cm|%))",
    re.IGNORECASE,
)

LIKE_NEW_RE = re.compile(r"\b(?:come|pari\s+al|like|wie)\s+(?:nuov|new|neu)", re.IGNORECASE)
NEW_RE = re.compile(r"\b(?:nuov[oa]|sigillat[oa]|new|sealed|mai\s+aperto|neu|versiegelt|nib|bnib)\b", re.IGNORECASE)
REFURB_RE = re.compile(r"\b(?:ricondizionat\w*|rigenerat\w*|refurb\w*)", re.IGNORECASE)
PARTS_RE = re.compile(r"\b(?:per\s+ricambi|for\s+parts|non\s+funzionante|not\s+working|rott[oa])\b", re.IGNORECASE)

# Galaxy models are usually written without "Galaxy" ("S24 Ultra 256").
GALAXY_SHORTHAND_RE = re.compile(r"^(?!.*\bgalaxy\b).*?\b(?:s\d{2}|a5\d|z\s*(?:fold|flip)\s*\d)\b", re.IGNORECASE)


@dataclass
class DealRequest:
    text: str
    price: Optional[float]
    band: str                      # used | new | refurb | parts
    ebay_item_id: str = ""         # "v1|123|0" when the message is an eBay link
    ebay_marketplace: str = ""


def parse_price(text: str) -> Optional[float]:
    """The asking price in a message, or None if it doesn't state one."""
    marked = CURRENCY_PRICE_RE.findall(text)
    if marked:
        raw = next(v for v in marked[-1] if v)
        return float(raw.replace(",", "."))
    for raw in reversed(BARE_NUMBER_RE.findall(text)):
        value = int(raw)
        if 2000 <= value <= 2035:          # a year, not a price
            continue
        if value >= 30:
            return float(value)
    return None


def parse_band(text: str) -> str:
    """Condition band from the wording ("come nuovo" is still used)."""
    if PARTS_RE.search(text):
        return "parts"
    if REFURB_RE.search(text):
        return "refurb"
    if LIKE_NEW_RE.search(text):
        return "used"
    if NEW_RE.search(text):
        return "new"
    return "used"


def parse_request(text: str) -> DealRequest:
    text = (text or "").strip()
    m = EBAY_ITEM_RE.search(text)
    if m:
        domain, number = m.group(1).lower(), m.group(2)
        return DealRequest(text=text, price=None, band="used",
                           ebay_item_id=f"v1|{number}|0",
                           ebay_marketplace=DOMAIN_MARKETPLACE.get(domain, "EBAY_IT"))
    return DealRequest(text=text, price=parse_price(text), band=parse_band(text))


def classify(text: str) -> tuple[str, float]:
    """Product key for a free-text device description."""
    if GALAXY_SHORTHAND_RE.search(text):
        text = "galaxy " + text
    key, confidence = PriceTracker(None).generate_product_key(text)
    return (normalize_key(key) if key else ""), confidence


# ── Categories (for the per-category deal threshold) ─────────────────────

FAMILY_CATEGORY = [
    ("galaxy_tab", "Galaxy Tablets"), ("xiaomi_pad", "Xiaomi Tablets"), ("redmi_pad", "Redmi Tablets"),
    ("iphone", "iPhones"), ("galaxy", "Samsung Galaxy"), ("macbook", "MacBooks"),
    ("ipad", "iPads"), ("ps5", "PlayStation 5"), ("xbox", "Xbox Series X"),
    ("switch", "Nintendo Switch"), ("airpods", "AirPods"),
]


def category_for(product_key: str, categories: dict[str, dict]) -> Optional[dict]:
    for prefix, name in FAMILY_CATEGORY:
        if product_key.startswith(prefix):
            return categories.get(name)
    return None


# ── The market picture ───────────────────────────────────────────────────

def _pct(values: list[float], pct: float) -> Optional[float]:
    return PriceTracker._percentile(PriceTracker._reject_outliers(values), pct) if values else None


@dataclass
class MarketView:
    baseline: Any                  # baseline.Baseline or None
    low: Optional[float]           # 25th percentile of home asking prices
    mid: Optional[float]
    high: Optional[float]
    n_home: int
    cheapest_live: list[dict]


def market_view(window: MarketWindow, product_key: str, band: str, home: str, now=None) -> MarketView:
    """Baseline plus the spread and cheapest live listings for one bucket."""
    now = now or datetime.now(timezone.utc)
    rows = [r for r in window.rows
            if normalize_key(r.get("product_key") or "") == product_key and r.get("condition_band") == band]
    home_prices = [float(r["price"]) for r in rows
                   if (r.get("marketplace") or home) == home and r.get("price")]
    baseline = window.judge(product_key, band)

    live_since = (now - timedelta(hours=26)).isoformat()
    floor = baseline.value * 0.5 if baseline else 0     # below half is junk or bait
    live = [r for r in rows
            if r.get("status") == "active" and (r.get("last_seen") or "") >= live_since
            and r.get("price") and float(r["price"]) >= floor]
    live.sort(key=lambda r: float(r["price"]))
    return MarketView(
        baseline=baseline,
        low=_pct(home_prices, 25), mid=_pct(home_prices, 50), high=_pct(home_prices, 75),
        n_home=len(home_prices),
        cheapest_live=live[:3],
    )


# ── Verdict ──────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    label: str
    discount_pct: float
    max_price: float


def verdict(price: float, baseline: float, min_discount: float, suspicious_pct: float = 55) -> Verdict:
    discount = (baseline - price) / baseline * 100
    max_price = baseline * (1 - min_discount / 100)
    if discount > suspicious_pct:
        label = "🚨 Too cheap to be true — high scam risk"
    elif discount >= min_discount:
        label = "✅ Good deal"
    elif discount >= -2:                 # within noise of the market price
        label = "👌 Fair price, not a bargain"
    else:
        label = "❌ Above market"
    return Verdict(label, round(discount, 1), round(max_price))


# ── AI review ────────────────────────────────────────────────────────────

REVIEW_PROMPT = """You help a reseller in Italy judge a second-hand electronics deal.
""" + DATE_RULE + """
The user wrote: "{text}"

Market facts (from eBay listings over the last {days} days):
- Recognised as: {product} ({band})
- Market price: {market}
- Asking price: {price}

Return ONLY valid JSON:
{{
  "note": "one or two sentences: is this worth pursuing and why",
  "check_before_buying": ["3-5 short, model-specific checks (battery health %, iCloud/FRP lock, Face ID, screen burn-in, IMEI blacklist, warranty, original box/receipt...)"],
  "warning_signs": ["anything in the user's text that is a red flag — [] if none"]
}}
Respond in English."""


def ai_review(ai, request: DealRequest, product_key: str, band: str,
              market_text: str, days: int) -> dict:
    from src.ai_analyzer import _parse_ai_json, _today

    prompt = REVIEW_PROMPT.format(
        today=_today(), text=request.text[:600], days=days, product=product_key, band=band,
        market=market_text, price=f"€{request.price:.0f}" if request.price else "not given",
    )
    response = ai._call_ai(prompt)
    return (_parse_ai_json(response) if response else None) or {}


# ── Reply ────────────────────────────────────────────────────────────────

HELP_TEXT = (
    "🔎 <b>Deal check</b>\n"
    "Send me a device, its condition and the price, and I'll compare it with the market.\n\n"
    "Examples:\n"
    "• <code>iPhone 15 Pro 256GB usato 550€</code>\n"
    "• <code>S24 Ultra 512GB come nuovo 600</code>\n"
    "• <code>PS5 slim sigillata 380 euro</code>\n"
    "• an eBay link — I'll also check the description and photos"
)


_NAME_WORDS = {"iphone": "iPhone", "ipad": "iPad", "macbook": "MacBook", "airpods": "AirPods",
               "ps5": "PS5", "fe": "FE", "se": "SE", "oled": "OLED", "xbox": "Xbox"}


def pretty_name(product_key: str) -> str:
    """"iphone_15_pro_256gb" -> "iPhone 15 Pro 256GB"."""
    words = []
    for w in product_key.split("_"):
        if re.fullmatch(r"\d+(gb|tb)", w):
            words.append(w.upper())
        elif re.fullmatch(r"[a-z]\d+", w):        # s24, a57, m3
            words.append(w.upper())
        else:
            words.append(_NAME_WORDS.get(w, w.capitalize()))
    return " ".join(words)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def _item_link(item_id: str) -> str:
    parts = str(item_id).split("|")
    number = parts[1] if len(parts) > 1 else parts[0]
    return f"https://www.ebay.it/itm/{number}" if number.isdigit() else ""


def _baseline_line(b, days: int) -> str:
    if b is None:
        return "not enough recent listings to price it"
    if b.source == "market":
        return f"€{b.value:.0f} ({b.n} listings, {days} days)"
    if b.source == "new-price cap":
        return f"€{b.value:.0f} (new price — used listings ask more)"
    if b.source == "blend":
        return f"~€{b.value:.0f} (only {b.n} listings, blended with retail)"
    if b.source == "thin market":
        return f"~€{b.value:.0f} (only {b.n} listings — rough)"
    return f"~€{b.value:.0f} (retail estimate, too few listings)"


def build_reply(request: DealRequest, product_key: str, band: str, view: MarketView,
                category: Optional[dict], review: dict, days: int,
                listing: Optional[dict] = None, photo_lines: Optional[list[str]] = None) -> str:
    lines = [f"🔎 <b>{_esc(pretty_name(product_key))}</b> — {band}"]
    if listing:
        lines.append(f"📱 {_esc(listing.get('title', '')[:90])}")
        from src.deal_scorer import listing_age
        days, _ = listing_age(listing)
        if days is not None:
            # A single-item lookup has no original listing date, so a relist
            # can't be told apart here — only this listing's own age.
            when = f"{days:.0f} days ago" if days >= 1.5 else ("yesterday" if days >= 1 else
                                                               f"{max(1, round(days * 24))}h ago")
            hint = " — still unsold, ask why" if days >= 30 else ""
            lines.append(f"🗓️ Listed {when}{hint}")

    b = view.baseline
    lines.append(f"📊 Market: {_baseline_line(b, days)}")
    if view.n_home >= 3 and view.low and view.high:
        lines.append(f"   Italy asking range: €{view.low:.0f} – €{view.high:.0f} (middle €{view.mid:.0f})")
    if b and b.new_price:
        lines.append(f"   New price: €{b.new_price:.0f}")

    if request.price and b:
        min_discount = float((category or {}).get("min_discount", 15))
        v = verdict(request.price, b.value, min_discount)
        if abs(v.discount_pct) < 1:
            where = "right at"
        elif v.discount_pct > 0:
            where = f"{v.discount_pct:.0f}% under"
        else:
            where = f"{-v.discount_pct:.0f}% over"
        lines += ["", f"<b>{v.label}</b> — €{request.price:.0f} is {where} market",
                  f"💡 Pay at most <b>€{v.max_price}</b> for it to be a deal ({min_discount:.0f}% under)"]
    elif b:
        min_discount = float((category or {}).get("min_discount", 15))
        lines += ["", f"💡 A deal is anything under <b>€{b.value * (1 - min_discount / 100):.0f}</b> "
                      f"({min_discount:.0f}% under market). Add the price to get a verdict."]

    if photo_lines:
        lines += [""] + photo_lines

    from src.deal_scorer import _real_entries   # drops "None", "N/A", "" entries

    note = review.get("note")
    checks = _real_entries(review.get("check_before_buying"))
    warnings = _real_entries(review.get("warning_signs") or review.get("red_flags"))
    if note or checks or warnings:
        lines.append("")
        if note:
            lines.append(f"🤖 {_esc(note)}")
        if warnings:
            lines.append("⚠️ <b>Warning signs:</b> " + "; ".join(_esc(w) for w in warnings[:4]))
        if checks:
            lines.append("✅ <b>Check before buying:</b>")
            lines += [f"• {_esc(c)}" for c in checks[:5]]
    elif review == {}:
        lines += ["", "🤖 AI review unavailable right now."]

    if view.cheapest_live:
        lines += ["", "🛒 <b>Cheapest live listings now</b> (unchecked):"]
        for r in view.cheapest_live:
            url = _item_link(r.get("id", ""))
            flag = "" if (r.get("marketplace") or "EBAY_IT") == "EBAY_IT" else f" ({r['marketplace'][5:]})"
            label = f"€{float(r['price']):.0f}{flag}"
            lines.append(f"• <a href=\"{_esc(url)}\">{label}</a>" if url else f"• {label}")
    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────

class DealChecker:
    """Holds the clients a check needs; one per webhook call."""

    def __init__(self, db, ai, ebay=None, settings_path="config/settings.yaml",
                 categories_path="config/categories.yaml") -> None:
        self.db, self.ai, self.ebay = db, ai, ebay
        with open(settings_path, encoding="utf-8") as f:
            self.settings = yaml.safe_load(f)
        with open(categories_path, encoding="utf-8") as f:
            self.categories = {c["name"]: c for c in yaml.safe_load(f)["categories"]}
        self.home = self.settings.get("ebay", {}).get("marketplace", "EBAY_IT")

    def check(self, text: str) -> str:
        request = parse_request(text)
        listing, photo_lines = None, None

        if request.ebay_item_id:
            if not self.ebay:
                return "eBay links aren't available right now — send the model, condition and price instead."
            listing = self.ebay.get_item_details(request.ebay_item_id, marketplace=request.ebay_marketplace)
            if not listing:
                return ("Couldn't open that eBay listing (it may have variations or have ended). "
                        "Send the model, condition and price instead.")
            listing["total_price"] = float(listing.get("price") or 0) + float(listing.get("shipping_cost") or 0)
            request.price = listing["total_price"]
            request.band = "parts" if is_for_parts(listing) else condition_band(listing)
            request.text = listing.get("title", "")
            product_key, confidence = classify(listing.get("title", ""))
        else:
            product_key, confidence = classify(request.text)

        if request.band in ("refurb", "parts"):
            return (f"That reads as <b>{request.band}</b>. I only price used and sealed units — "
                    "refurbished and for-parts devices are a different market.")
        if not product_key or confidence < PriceTracker.MIN_CONFIDENCE:
            return ("I couldn't recognise the model. Name it like the listing would, e.g. "
                    "<code>iPhone 15 Pro 256GB usato 550€</code>.")
        if product_key.endswith(("_part", "_multi")):
            return "That looks like an accessory/part or several models at once — I can only price a single device."

        window = MarketWindow.load(self.db, self.settings, self.home)
        if window is None:
            return "Market data is unavailable right now — try again in a minute."
        view = market_view(window, product_key, request.band, self.home)
        if listing:   # don't suggest the very listing being checked
            view.cheapest_live = [r for r in view.cheapest_live if r.get("id") != listing.get("item_id")]
        category = category_for(product_key, self.categories)
        market_text = _baseline_line(view.baseline, window.days)

        if listing:
            # The full listing check the scanner runs: description + 3 photos.
            listing["avg_price"] = view.baseline.value if view.baseline else 0
            listing["discount_pct"] = (
                round((view.baseline.value - request.price) / view.baseline.value * 100, 1)
                if view.baseline and request.price else 0)
            analysis = self.ai.analyze_listing(listing)
            review = {
                "note": analysis.get("reasoning"),
                "warning_signs": (analysis.get("red_flags") or []) + (analysis.get("hidden_defects") or []),
                "check_before_buying": [],
            } if not analysis.get("text_ai_failed") else {}
            photos = self.settings.get("ai", {}).get("photos_per_check", 3)
            image = self.ai.analyze_image(
                listing.get("image_url", ""), listing.get("condition", ""),
                listing.get("short_description", ""),
                extra_image_urls=[u for u in listing.get("all_image_urls", [])
                                  if u != listing.get("image_url")],
                max_photos=photos,
            )
            photo_lines = _photo_summary(image)
        else:
            review = ai_review(self.ai, request, product_key, request.band, market_text, window.days)

        return build_reply(request, product_key, request.band, view, category, review,
                           window.days, listing=listing, photo_lines=photo_lines)


def _photo_summary(image: dict) -> list[str]:
    if image.get("photos_too_small"):
        return ["📷 ⚠️ Every photo is thumbnail-sized — nothing can be judged. Ask for real pictures."]
    if image.get("image_ai_failed"):
        return ["📷 ⚠️ Photos NOT checked (image AI unavailable) — look at them yourself."]
    n = image.get("photos_reviewed") or 1
    damage = [d for d in image.get("visible_damage") or [] if d]
    if image.get("photo_type") == "stock":
        out = [f"📷 ⚠️ Stock/catalog photos only ({n} checked) — condition can't be verified."]
    elif image.get("condition_matches_description") is False:
        out = [f"📸 ⚠️ Real photos, but condition may not match the description ({n} checked)."]
    elif image.get("photo_type") == "real":
        out = [f"📸 ✅ Real photos ({n} checked)."]
    else:
        out = [f"📷 ❓ Couldn't tell if the photos are genuine ({n} checked)."]
    if damage:
        out.append("   Visible: " + "; ".join(_esc(d) for d in damage[:3]))
    return out


# ── Telegram update handling ─────────────────────────────────────────────

MAX_UPDATE_AGE_S = 300


def handle_update(update: dict, checker_factory, send, allowed_chat_id: str, now: Optional[float] = None) -> bool:
    """Process one Telegram update. Returns True if a reply was sent.

    Only the owner's chat is answered, and only fresh messages: after an
    outage Telegram re-delivers old updates, which shouldn't each trigger a
    full check.
    """
    message = update.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    if not message or chat_id != str(allowed_chat_id):
        return False
    if (now or time.time()) - float(message.get("date") or 0) > MAX_UPDATE_AGE_S:
        return False
    text = (message.get("text") or message.get("caption") or "").strip()
    if not text:
        return False
    if text.split()[0].split("@")[0] in ("/start", "/help"):
        return send(HELP_TEXT)
    try:
        reply = checker_factory().check(text)
    except Exception as e:  # never leave the user without an answer
        logger.exception("Deal check failed: %s", type(e).__name__)
        reply = "Something went wrong while checking that deal — try again in a minute."
    return send(reply)
