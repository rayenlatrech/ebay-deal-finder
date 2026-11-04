"""Is the scanner alive? Answered from outside GitHub, because a stopped scan can't report itself.

Per-scan reports used to double as a sign of life. Now that there's one report
a day, silence would look exactly like a quiet day, so an hourly ping from
cron-job.org (api/health.py) warns when scans stop, and /status answers on demand.
"""

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.db import Database

# Scans are hourly, so one miss is normal and this many is not.
MAX_SILENCE_HOURS = 3
# Stateless reminder: an hourly ping lands in each hour once, so this repeats the
# warning every 6 hours instead of every hour.
REMIND_EVERY_HOURS = 6


def scan_age_hours(last_scan: Optional[datetime], now: datetime) -> Optional[float]:
    """Hours since the last scan, or None if that can't be read."""
    if last_scan is None:
        return None
    return (now - last_scan).total_seconds() / 3600


def warning_due(age_hours: Optional[float], now: datetime) -> bool:
    """Should this hourly ping send a warning?"""
    if age_hours is None:
        return now.hour % REMIND_EVERY_HOURS == 0
    if age_hours < MAX_SILENCE_HOURS:
        return False
    return int(age_hours - MAX_SILENCE_HOURS) % REMIND_EVERY_HOURS == 0


def warning_text(age_hours: Optional[float]) -> str:
    if age_hours is None:
        return ("⚠️ <b>Scanner check failed</b>\n"
                "I can't read the database, so I can't tell if scans are running.")
    return (f"⚠️ <b>No scan for {age_hours:.0f} hours</b>\n"
            "Scans should run every hour, so no alerts are coming. Check the Actions tab "
            "on GitHub and the job on cron-job.org.")


def _age_text(age_hours: Optional[float]) -> str:
    if age_hours is None:
        return "⚠️ Can't read the last scan time"
    minutes = age_hours * 60
    when = f"{minutes:.0f} min ago" if minutes < 90 else f"{age_hours:.1f} hours ago"
    mark = "⚠️" if age_hours >= MAX_SILENCE_HOURS else "✅"
    return f"{mark} Last scan: {when}"


def _gemini_keys_available(db: Database) -> tuple[int, int]:
    """(usable, total) Gemini keys: one on cooldown has hit its daily limit."""
    keys = [v for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY_") and v]
    usable = sum(1 for key in keys if not db.get_key_cooldown(hashlib.sha256(key.encode()).hexdigest()))
    return usable, len(keys)


def build_status(db: Database, now: Optional[datetime] = None, hot_score: int = 80) -> str:
    """The reply to /status."""
    now = now or datetime.now(timezone.utc)
    alerted = db.get_alerted_since((now - timedelta(hours=24)).isoformat())
    hot = sum(1 for a in alerted.values() if (a.get("deal_score") or 0) >= hot_score)
    usable, total = _gemini_keys_available(db)
    photos = "available" if usable else "out until 08:00 UTC"

    lines = [
        "📟 <b>Scanner status</b>",
        "",
        _age_text(scan_age_hours(db.get_last_scan_time(), now)),
        f"🔔 Alerts in the last 24h: {len(alerted)} ({hot} hot)",
        f"📸 Photo checks: {photos} ({usable} of {total} keys)",
    ]
    feedback = db.get_feedback_counts()
    if feedback:
        lines.append(f"🗳️ Your ratings: 👍 {feedback.get('up', 0)} 👎 {feedback.get('down', 0)} "
                     f"🛒 {feedback.get('bought', 0)}")
    return "\n".join(lines)
