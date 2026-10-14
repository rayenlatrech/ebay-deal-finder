"""Telegram notification sender — sends rich deal alerts and summaries.

Messages use Telegram's HTML parse mode (not Markdown). HTML is far more
forgiving for the messy, special-character-laden titles eBay produces: only
``<``, ``>`` and ``&`` need escaping, which ``_esc`` handles for every piece of
dynamic content. The previous Markdown approach silently dropped any alert whose
title contained an unbalanced ``_``, ``[`` or ``(``.
"""

import html
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_MESSAGE_LENGTH = 4096


class TelegramNotifier:
    """Sends formatted deal alerts via Telegram Bot API."""

    def __init__(self) -> None:
        """Initialize with bot token and chat ID from environment."""
        self.token = os.environ["TELEGRAM_BOT_TOKEN"]
        self.chat_id = os.environ["TELEGRAM_CHAT_ID"]
        self.session = requests.Session()
        logger.info("Telegram notifier initialized")

    # ── Low-Level Send ───────────────────────────────────────────────

    def send_chat_action(self, action: str = "typing") -> None:
        """Show "typing…" while a deal check runs (best effort)."""
        try:
            self.session.post(
                TELEGRAM_API.format(token=self.token, method="sendChatAction"),
                json={"chat_id": self.chat_id, "action": action}, timeout=5,
            )
        except requests.exceptions.RequestException as e:
            logger.warning("sendChatAction failed: %s", type(e).__name__)

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API.

        Handles message splitting if text exceeds 4096 chars.
        """
        chunks = self._split_message(text)

        for chunk in chunks:
            url = TELEGRAM_API.format(token=self.token, method="sendMessage")
            payload = {
                "chat_id": self.chat_id,
                "text": chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            try:
                resp = self.session.post(url, json=payload, timeout=15)
                if resp.status_code == 429:
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                    logger.warning("Telegram rate limited, waiting %ds", retry_after)
                    time.sleep(retry_after)
                    resp = self.session.post(url, json=payload, timeout=15)

                if resp.status_code != 200:
                    logger.error("Telegram error %d: %s", resp.status_code, resp.text)
                    return False

                # Rate limit: 1 msg/sec
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                # The exception text embeds the request URL, which carries the
                # bot token — log only the error type.
                logger.error("Failed to send Telegram message: %s", type(e).__name__)
                return False

        return True

    def _split_message(self, text: str) -> list[str]:
        """Split a message into chunks that fit Telegram's 4096 char limit."""
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= MAX_MESSAGE_LENGTH:
                chunks.append(text)
                break

            split_at = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
            if split_at == -1:
                split_at = MAX_MESSAGE_LENGTH

            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")

        return chunks

    @staticmethod
    def _esc(text: Any) -> str:
        """Escape dynamic content for Telegram HTML parse mode."""
        return html.escape(str(text), quote=False)

    @staticmethod
    def _link(text: Any, url: str) -> str:
        """Build a safe HTML anchor (URL escaped for the href attribute)."""
        return f'<a href="{html.escape(str(url), quote=True)}">{TelegramNotifier._esc(text)}</a>'

    # ── Marketplace / trust badges ───────────────────────────────────

    MARKETPLACE_FLAGS = {
        "EBAY_IT": "🇮🇹", "EBAY_DE": "🇩🇪", "EBAY_FR": "🇫🇷",
        "EBAY_NL": "🇳🇱", "EBAY_BE": "🇧🇪", "EBAY_ES": "🇪🇸",
    }

    @classmethod
    def _flag(cls, marketplace: str) -> str:
        """Country flag for a marketplace id (blank if unknown)."""
        return cls.MARKETPLACE_FLAGS.get(str(marketplace or "").upper(), "")

    # ── Trust Emoji ──────────────────────────────────────────────────

    @staticmethod
    def _trust_emoji(trust_level: str) -> str:
        """Get emoji for seller trust level."""
        mapping = {
            "high": "🟢 Trusted",
            "medium": "🟡 Medium",
            "low_new": "🟠 New seller",
            "low_bad": "🔴 Low trust",
        }
        return mapping.get(trust_level, "⚪ Unknown")

    # ── Baseline evidence ────────────────────────────────────────────

    def _baseline_text(self, value: float, scores: dict[str, Any]) -> str:
        """The price a deal is judged against, with how much evidence backs it."""
        source = scores.get("baseline_source", "market")
        n = scores.get("baseline_n")
        days = scores.get("baseline_days")
        if n is None:  # legacy baseline: no listing count to report
            return f"€{value:.0f} {self._esc(source)}"
        listings = f"{n} listing{'s' if n != 1 else ''}"
        window = f", {days} days" if days else ""
        if source == "market":
            return f"market €{value:.0f} ({listings}{window})"
        if source == "blend":
            return f"market ~€{value:.0f} (only {listings}{window}, blended with retail)"
        if source == "thin market":
            return f"market ~€{value:.0f} (only {listings}{window} — treat with care)"
        if source == "new-price cap":
            return f"€{value:.0f} new price (used listings ask more than new)"
        if source == "retail estimate":
            return f"€{value:.0f} retail estimate (too few listings to price it)"
        return f"€{value:.0f} {self._esc(source)}"

    # ── AI review status ─────────────────────────────────────────────

    # Why the photo wasn't checked, phrased for the alert.
    _PHOTO_SKIP_REASON = {
        "ai_failed": "image AI unavailable",
        "skipped_limit": "this run's photo-check limit was reached",
        "skipped_cannot_alert": "not prioritised",
    }

    def _photo_lines(self, analysis: dict[str, Any], scores: dict[str, Any]) -> list[str]:
        """Describe what is known about the photos — and whether the AI looked.

        An unchecked photo used to read "authenticity uncertain", identical to a
        photo the model examined and couldn't call. Now the two are distinct.
        """
        status = scores.get("image_status")
        photo_type = analysis.get("photo_type", "")
        reviewed = status is None or status in ("checked", "checked_earlier")

        if status == "no_photo":
            return ["📷 ⚠️ <b>Listing has no photo</b> — ask the seller for real pictures."]

        if not reviewed:
            reason = self._PHOTO_SKIP_REASON.get(status, "not checked")
            lines = [
                f"📷 ⚠️ <b>Photos NOT checked by AI</b> ({reason})",
                "   Look at them yourself before buying.",
            ]
            # The free metadata check can still spot a reused catalog image.
            if photo_type == "stock":
                lines.append("   Likely a stock image: the same picture is used by other listings.")
            return lines

        if photo_type == "stock":
            return [
                "📷 ⚠️ <b>NO REAL PHOTOS</b> — Stock/catalog image detected",
                "   Condition cannot be verified from images.",
            ]
        n = int(analysis.get("photos_reviewed") or 1)
        checked = "AI checked the main photo" if n == 1 else f"AI checked {n} photos"
        if photo_type == "real":
            matches = analysis.get("condition_matches_description")
            if matches is True:
                return [f"📸 ✅ Real photos — match the description ({checked})"]
            if matches is False:
                return [f"📸 ⚠️ Real photos — <b>condition may not match description</b> ({checked})"]
            return [f"📸 Real photos ({checked})"]
        return [f"📷 ❓ {checked} but couldn't tell if they're genuine"]

    @staticmethod
    def _review_markers(scores: dict[str, Any]) -> str:
        """Compact flags for a summary row when the AI skipped part of its review."""
        marks = ""
        status = scores.get("image_status")
        if status is not None and status not in ("checked", "checked_earlier"):
            marks += " 🙈"
        if not scores.get("text_reviewed", True):
            marks += " 📝❌"
        return marks

    # ── Hot Deal Alert ───────────────────────────────────────────────

    def send_hot_deal_alert(
        self,
        listing: dict[str, Any],
        analysis: dict[str, Any],
        scores: dict[str, Any],
    ) -> bool:
        """Send a detailed alert for a hot deal (score >= 80)."""
        deal_score = scores.get("deal_score", 0)
        # Show the total actually paid — the discount is computed on it, so
        # displaying the base price alone made the maths look wrong.
        price = listing.get("total_price") or listing.get("price", 0)
        shipping = listing.get("shipping_cost", 0) or 0
        avg_price = scores.get("avg_price", 0)
        discount_pct = scores.get("discount_pct", 0)
        new_price = scores.get("new_price")
        ebay_avg = scores.get("ebay_avg")
        trust_level = scores.get("seller_trust_level", "unknown")

        flag = self._flag(scores.get("marketplace", ""))
        header = "⏳ <b>AUCTION ENDING SOON!</b>" if scores.get("is_auction") \
            else "🔥 <b>HOT DEAL FOUND!</b>"
        if scores.get("seller_trusted"):
            header += "  ⭐ <b>TRUSTED SELLER</b>"
        if scores.get("is_foreign"):
            header += f"\n🌍 Cross-border {flag} — ships to Italy"

        lines = [
            header,
            "",
            f"📱 {flag} {self._esc(listing.get('title', 'Unknown'))}".strip(),
            f"💰 Price: €{price:.0f} — <b>{discount_pct:.0f}% under</b> {self._baseline_text(avg_price, scores)}",
        ]
        if scores.get("is_auction"):
            hours = scores.get("hours_left")
            bids = scores.get("bid_count", 0)
            when = f"{hours:.1f}h left" if hours is not None else "ending soon"
            bid_text = f"{bids} bid{'s' if bids != 1 else ''}" if bids else "<b>no bids yet</b>"
            lines.append(f"   🔨 Auction — {when}, {bid_text} (price is the current bid)")
            max_bid = scores.get("max_bid")
            if max_bid:
                lines.append(
                    f"   💡 <b>Bid up to €{max_bid:.0f}</b> — still a deal at that price. "
                    f"Enter it as your max bid and eBay bids for you."
                )
        if shipping:
            lines.append(f"   📦 incl. €{shipping:.0f} shipping")
        drop = scores.get("price_drop_pct")
        original = scores.get("original_price")
        if drop and original:
            lines.append(
                f"   📉 <b>Seller cut the price {drop:.0f}%</b> (was €{original:.0f}) — motivated seller"
            )
        band = scores.get("condition_band")
        if band:
            lines.append(f"   🏷️ Compared against <b>{self._esc(band)}</b> listings")
        # When anchored to the new price, show the eBay asking figure too, so an
        # inflated asking-median is visible rather than hidden.
        if new_price and ebay_avg and ebay_avg > new_price * 1.05:
            lines.append(f"   ℹ️ eBay asking avg was €{ebay_avg:.0f} (used new price as the anchor)")

        if listing.get("negotiable"):
            lines.append("🤝 Price may be negotiable (Make Offer available)")

        lines.append(f"📊 Deal Score: {deal_score}/100")

        lines.extend([""] + self._photo_lines(analysis, scores))

        # AI condition assessment
        if scores.get("text_reviewed", True):
            condition_text = str(analysis.get("condition_assessment", "unknown"))
            reasoning = analysis.get("reasoning", "")
            lines.extend(["", f"📝 <b>Condition:</b> {self._esc(condition_text.capitalize())}"])
            if reasoning:
                lines.append(f'   AI: "{self._esc(reasoning)}"')
        else:
            lines.extend([
                "",
                "📝 ⚠️ <b>Description NOT reviewed by AI</b> (text AI unavailable)",
                "   Read the description yourself: condition, defects, locks.",
            ])

        # Seller info
        seller_score = listing.get("seller_feedback_score", 0)
        seller_pct = listing.get("seller_feedback_percentage", 0)
        trust_str = self._trust_emoji(trust_level)
        lines.extend([
            "",
            f"👤 <b>Seller:</b> {self._esc(seller_pct)}% positive ({self._esc(seller_score)} sales) — {trust_str}",
        ])

        # Red flags
        red_flags = analysis.get("red_flags", [])
        if red_flags:
            flag_text = ", ".join(self._esc(f) for f in red_flags)
            lines.extend(["", f"⚠️ <b>Red Flags:</b> {flag_text}"])
        else:
            lines.extend(["", "⚠️ <b>Red Flags:</b> None detected"])

        if scores.get("is_suspicious"):
            lines.extend(["", "🚨 <b>WARNING: Price unusually low — possible scam!</b>"])

        item_url = listing.get("item_url", "")
        if item_url:
            lines.extend(["", f"🔗 {self._link('View on eBay', item_url)}"])

        return self._send_message("\n".join(lines))

    # ── Batched Good Deals Summary ───────────────────────────────────

    @staticmethod
    def collapse_duplicates(deals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse listings of the same product down to the single best one.

        A market-wide price move can put a dozen near-identical listings of one
        model into a single summary, which buries everything else. Deals are
        grouped by product key and only the strongest survives (highest score,
        cheapest as tie-break); the rest are recorded as a `duplicate_count` so
        the row can say "+N similar" instead of repeating itself.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        ungrouped: list[dict[str, Any]] = []
        for deal in deals:
            key = deal.get("scores", {}).get("product_key")
            if key:
                groups.setdefault(key, []).append(deal)
            else:
                ungrouped.append(deal)

        def strength(d: dict[str, Any]) -> tuple:
            s, li = d.get("scores", {}), d.get("listing", {})
            price = li.get("total_price") or li.get("price") or 0
            return (-int(s.get("deal_score", 0)), float(price))

        collapsed: list[dict[str, Any]] = []
        for members in groups.values():
            members.sort(key=strength)
            best = dict(members[0])
            best["duplicate_count"] = len(members) - 1
            collapsed.append(best)

        collapsed.extend(ungrouped)
        collapsed.sort(key=strength)
        return collapsed

    def _deal_line(self, index: int, deal: dict[str, Any]) -> str:
        """Render one deal as a summary row."""
        listing = deal.get("listing", {})
        scores = deal.get("scores", {})
        title = str(listing.get("title", "Unknown"))[:50]
        price = listing.get("total_price") or listing.get("price", 0)
        discount = scores.get("discount_pct", 0)
        score = scores.get("deal_score", 0)
        url = listing.get("item_url", "")
        flag = self._flag(scores.get("marketplace", ""))

        link = self._link(title, url) if url else self._esc(title)
        line = f"{index}. {flag} {link} — €{price:.0f} ({discount:.0f}% off) — Score: {score}"

        if listing.get("negotiable"):
            line += " 🤝"
        if deal.get("analysis", {}).get("photo_type", "") == "stock":
            line += " 📷"
        line += self._review_markers(scores)
        if scores.get("is_auction"):
            hours = scores.get("hours_left")
            line += f" 🔨{hours:.0f}h" if hours is not None else " 🔨"
            if scores.get("max_bid"):
                line += f" · bid ≤€{scores['max_bid']:.0f}"
        if scores.get("price_drop_pct"):
            line += " 📉"
        dupes = deal.get("duplicate_count", 0)
        if dupes:
            line += f"  <i>(+{dupes} similar)</i>"
        return line

    def send_deal_summary(self, deals: list[dict[str, Any]]) -> bool:
        """Send a good-deals summary, split by seller trust.

        Deals from established, well-rated sellers are listed first in their own
        section — those are the ones worth acting on quickly. Everything else
        follows in a second section so it's visible but clearly de-emphasised.
        """
        if not deals:
            return True

        total = len(deals)
        deals = self.collapse_duplicates(deals)
        trusted = [d for d in deals if d.get("scores", {}).get("seller_trusted")]
        others = [d for d in deals if not d.get("scores", {}).get("seller_trusted")]

        headline = f"Found {total} good deal{'s' if total != 1 else ''}"
        if len(deals) < total:
            headline += f" across {len(deals)} product{'s' if len(deals) != 1 else ''}"
        lines = [
            "📊 <b>eBay Deal Summary</b>",
            headline + ":",
        ]

        counter = 1
        if trusted:
            lines.extend(["", "⭐ <b>TRUSTED SELLERS</b> — established, 95%+ positive"])
            for deal in trusted:
                lines.append(self._deal_line(counter, deal))
                counter += 1

        if others:
            lines.extend(["", "🔎 <b>OTHER SELLERS</b> — verify before buying"])
            for deal in others:
                lines.append(self._deal_line(counter, deal))
                counter += 1

        lines.extend([
            "",
            "🤝 = Make Offer | 📷 = Stock photo | 📉 = price cut | 🔨 = auction",
            "🙈 = photos not checked by AI | 📝❌ = description not checked by AI",
        ])
        return self._send_message("\n".join(lines))

    # ── Scan Report ──────────────────────────────────────────────────

    def send_scan_report(self, stats: dict[str, Any]) -> bool:
        """Send a brief scan completion report."""
        lines = [
            "📋 <b>Scan Complete</b>",
            "",
            f"🔍 Listings scanned: {stats.get('total_scanned', 0)}",
            f"🔥 Hot deals: {stats.get('hot_deals', 0)}",
            f"👍 Good deals: {stats.get('good_deals', 0)}",
            f"📊 Logged: {stats.get('logged', 0)}",
            f"⏭️ Skipped: {stats.get('skipped', 0)}",
        ]

        errors = stats.get("errors", 0)
        if errors > 0:
            lines.append(f"❌ Errors: {errors}")

        # Standing cross-border price gaps — worth knowing even on a quiet scan.
        gaps = stats.get("arbitrage") or []
        if gaps:
            lines.extend(["", "💱 <b>Cross-border price gaps</b>"])
            for g in gaps:
                flag = self._flag(g["cheapest_market"])
                lines.append(
                    f"  {self._esc(g['product_key'])} ({self._esc(g['condition_band'])}): "
                    f"{flag} €{g['cheapest_price']:.0f} vs €{g['home_price']:.0f} home "
                    f"— <b>{g['gap_pct']:.0f}% cheaper</b>"
                )

        lines.append("\n⏰ Next scan in ~4 hours")
        return self._send_message("\n".join(lines))

    # ── Error Alert ──────────────────────────────────────────────────

    def send_error_alert(self, error_msg: str) -> bool:
        """Send a critical error notification."""
        message = f"🚨 <b>eBay Deal Finder Error</b>\n\n{self._esc(error_msg)}"
        return self._send_message(message)
