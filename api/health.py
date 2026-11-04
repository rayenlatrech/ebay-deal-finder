"""Scan health ping (Vercel function at /api/health), called hourly by cron-job.org.

Warns on Telegram when scans have stopped. Requests need the HEALTH_CHECK_SECRET
in an Authorization header, or anyone could trigger messages to the owner.
"""

import hmac
import logging
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler

# Vercel runs from the project root; make `src` importable regardless.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("health_check")


def _check() -> bool:
    """Warn if a warning is due. Returns True if scans look healthy."""
    from src.db import Database
    from src.health import scan_age_hours, warning_due, warning_text
    from src.telegram_notifier import TelegramNotifier

    now = datetime.now(timezone.utc)
    age = scan_age_hours(Database().get_last_scan_time(), now)
    if warning_due(age, now):
        TelegramNotifier()._send_message(warning_text(age))
    return age is not None and age < 3


class handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        expected = os.environ.get("HEALTH_CHECK_SECRET", "")
        given = self.headers.get("Authorization", "")
        if not expected or not hmac.compare_digest(given, f"Bearer {expected}"):
            self._respond(401)
            return
        try:
            self._respond(200, b"ok" if _check() else b"stale")
        except Exception as e:
            logger.exception("Health check failed: %s", type(e).__name__)
            # A 500 shows up as a failed job on cron-job.org, which can email the owner.
            self._respond(500)
