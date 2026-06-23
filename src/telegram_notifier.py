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
                logger.error("Failed to send Telegram message: %s", e)
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

    # ── Hot Deal Alert ───────────────────────────────────────────────

    def send_hot_deal_alert(
        self,
        listing: dict[str, Any],
        analysis: dict[str, Any],
        scores: dict[str, Any],
    ) -> bool:
        """Send a detailed alert for a hot deal (score >= 80)."""
        deal_score = scores.get("deal_score", 0)
        price = listing.get("price", 0)
        avg_price = scores.get("avg_price", 0)
        discount_pct = scores.get("discount_pct", 0)
        trust_level = scores.get("seller_trust_level", "unknown")

        lines = [
            "🔥 <b>HOT DEAL FOUND!</b>",
            "",
            f"📱 {self._esc(listing.get('title', 'Unknown'))}",
            f"💰 Price: €{price:.0f} (Market avg: €{avg_price:.0f}, <b>{discount_pct:.0f}% OFF</b>)",
        ]

        if listing.get("negotiable"):
            lines.append("🤝 Price may be negotiable (Make Offer available)")

        lines.append(f"📊 Deal Score: {deal_score}/100")

        # Stock photo warning
        photo_type = analysis.get("photo_type", "")
        if photo_type == "stock":
            lines.extend([
                "",
                "📷 ⚠️ <b>NO REAL PHOTOS</b> — Stock/catalog image detected",
                "   Condition cannot be verified from images.",
            ])
        elif photo_type == "uncertain":
            lines.extend(["", "📷 ❓ Photo authenticity uncertain"])
        elif photo_type == "real":
            matches = analysis.get("condition_matches_description")
            if matches is True:
                lines.extend(["", "📸 ✅ Real photos — matches description"])
            elif matches is False:
                lines.extend(["", "📸 ⚠️ Real photos — <b>condition may not match description</b>"])

        # AI condition assessment
        condition_text = str(analysis.get("condition_assessment", "unknown"))
        reasoning = analysis.get("reasoning", "")
        lines.extend(["", f"📝 <b>Condition:</b> {self._esc(condition_text.capitalize())}"])
        if reasoning:
            lines.append(f'   AI: "{self._esc(reasoning)}"')

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

    def send_deal_summary(self, deals: list[dict[str, Any]]) -> bool:
        """Send a compact summary of good deals (score 60-79)."""
        if not deals:
            return True

        lines = [
            "📊 <b>eBay Italy Deal Summary</b>",
            f"Found {len(deals)} good deal{'s' if len(deals) != 1 else ''}:",
            "",
        ]

        for i, deal in enumerate(deals, 1):
            listing = deal.get("listing", {})
            scores = deal.get("scores", {})
            title = str(listing.get("title", "Unknown"))[:50]
            price = listing.get("price", 0)
            discount = scores.get("discount_pct", 0)
            score = scores.get("deal_score", 0)
            url = listing.get("item_url", "")

            link = self._link(title, url) if url else self._esc(title)
            line = f"{i}. 👍 {link} — €{price:.0f} ({discount:.0f}% off) — Score: {score}"

            if listing.get("negotiable"):
                line += " 🤝"
            if deal.get("analysis", {}).get("photo_type", "") == "stock":
                line += " 📷"

            lines.append(line)

        lines.extend(["", "🤝 = Make Offer available | 📷 = Stock photo"])
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

        lines.append("\n⏰ Next scan in ~4 hours")
        return self._send_message("\n".join(lines))

    # ── Error Alert ──────────────────────────────────────────────────

    def send_error_alert(self, error_msg: str) -> bool:
        """Send a critical error notification."""
        message = f"🚨 <b>eBay Deal Finder Error</b>\n\n{self._esc(error_msg)}"
        return self._send_message(message)
