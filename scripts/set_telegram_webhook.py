"""Point the Telegram bot at the deal-check webhook (run once after deploying).

    python scripts/set_telegram_webhook.py https://<project>.vercel.app/api/telegram

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET from .env. Telegram then
sends that secret with every update, and the webhook rejects anything without
it. `--delete` removes the webhook again.
"""

import os
import sys

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv(".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
    if not token:
        print("TELEGRAM_BOT_TOKEN is missing from .env")
        return 1
    api = f"https://api.telegram.org/bot{token}"

    if sys.argv[1:] == ["--delete"]:
        resp = requests.post(f"{api}/deleteWebhook", timeout=15)
    elif len(sys.argv) == 2 and sys.argv[1].startswith("https://"):
        if not secret:
            print("TELEGRAM_WEBHOOK_SECRET is missing from .env (use the same value as on Vercel)")
            return 1
        resp = requests.post(f"{api}/setWebhook", json={
            "url": sys.argv[1],
            "secret_token": secret,
            "allowed_updates": ["message"],
            "drop_pending_updates": True,
        }, timeout=15)
    else:
        print(__doc__)
        return 1

    # Telegram's reply never contains the token; print its verdict only.
    data = resp.json()
    print("OK" if data.get("ok") else "FAILED", "-", data.get("description", ""))
    if data.get("ok") and sys.argv[1] != "--delete":
        info = requests.get(f"{api}/getWebhookInfo", timeout=15).json().get("result", {})
        print("Webhook:", info.get("url"), "| pending updates:", info.get("pending_update_count"),
              "| last error:", info.get("last_error_message") or "none")
    return 0 if data.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
