"""The 👍 / 👎 / 🛒 buttons under alerts: nothing else measures if alerts are any good."""

import logging

logger = logging.getLogger(__name__)

RATINGS = ("up", "down", "bought")
_PICKED = "✅ "


def parse_callback(data: str) -> tuple[str, str] | None:
    """(rating, item id) from a button's data, or None if it isn't a feedback button."""
    # Item ids contain "|", never ":", but split only twice to be safe.
    parts = (data or "").split(":", 2)
    if len(parts) != 3 or parts[0] != "fb" or parts[1] not in RATINGS or not parts[2]:
        return None
    return parts[1], parts[2]


def mark_choice(markup: dict, data: str) -> dict:
    """The same buttons with the pressed one ticked, and any earlier pick in its row cleared."""
    rows = []
    for row in markup.get("inline_keyboard", []):
        pressed_here = any(b.get("callback_data") == data for b in row)
        new_row = []
        for button in row:
            text = button.get("text", "")
            if pressed_here:
                text = text.removeprefix(_PICKED)
                if button.get("callback_data") == data:
                    text = _PICKED + text
            new_row.append({**button, "text": text})
        rows.append(new_row)
    return {"inline_keyboard": rows}


def handle_feedback(callback: dict, save, notifier, allowed_chat_id: str) -> bool:
    """Record one button press. Returns True if it was saved.

    Unlike a message, an old alert's button is still worth answering: the deal
    may only turn out good or bad days later, so there's no age limit here.
    """
    message = callback.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    from_id = str((callback.get("from") or {}).get("id", ""))
    if chat_id != str(allowed_chat_id) or from_id != str(allowed_chat_id):
        return False

    parsed = parse_callback(callback.get("data", ""))
    if not parsed:
        notifier.answer_callback(callback.get("id"), "Unknown button")
        return False

    rating, item_id = parsed
    if not save(item_id, rating):
        notifier.answer_callback(callback.get("id"), "Couldn't save that, try again in a minute")
        return False

    notifier.answer_callback(callback.get("id"), "Saved, thanks")
    if message.get("reply_markup"):
        notifier.edit_reply_markup(
            message["chat"]["id"], message.get("message_id"),
            mark_choice(message["reply_markup"], callback.get("data", "")),
        )
    return True
