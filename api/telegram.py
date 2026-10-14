"""Telegram webhook (Vercel function at /api/telegram): the deal-check bot.

Telegram POSTs every message sent to the bot here. Requests are accepted only
with the secret token registered alongside the webhook, and only the owner's
chat (TELEGRAM_CHAT_ID) is ever answered.
"""

import hmac
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler

# Vercel runs from the project root; make `src` importable regardless.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("telegram_webhook")


def _make_checker():
    from src.ai_analyzer import AIAnalyzer
    from src.db import Database
    from src.deal_check import DealChecker
    from src.ebay_client import EbayClient

    db = Database()
    return DealChecker(db, AIAnalyzer(db), EbayClient())


def _process(update: dict) -> None:
    from src.deal_check import handle_update
    from src.telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier()
    message = update.get("message") or {}
    if str((message.get("chat") or {}).get("id", "")) == notifier.chat_id:
        notifier.send_chat_action("typing")
    handle_update(update, _make_checker, notifier._send_message, notifier.chat_id)


class handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._respond(200, b"deal-check bot is up")

    def do_POST(self) -> None:
        expected = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
        given = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not expected or not hmac.compare_digest(given, expected):
            self._respond(401)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            update = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._respond(200)
            return
        try:
            _process(update)
        except Exception as e:
            logger.exception("Webhook failed: %s", type(e).__name__)
        # Always 200: anything else makes Telegram re-deliver the update.
        self._respond(200)
