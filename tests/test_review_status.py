"""Alerts must say when the AI did not actually review a listing's photo or text."""

import pytest

from src.deal_scorer import DealScorer
from src.telegram_notifier import TelegramNotifier


@pytest.fixture
def notifier(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    n = TelegramNotifier()
    n.sent = []
    n._send_message = lambda text, parse_mode="HTML": n.sent.append(text) or True
    return n


LISTING = {"title": "iPhone 15 128GB", "total_price": 400, "item_url": "https://www.ebay.it/itm/1"}


def _scores(**extra):
    return {"deal_score": 85, "avg_price": 550, "discount_pct": 27,
            "baseline_source": "used market", "product_key": "iphone_15_128gb", **extra}


def _hot(notifier, analysis, **score_extra):
    notifier.send_hot_deal_alert(LISTING, analysis, _scores(**score_extra))
    return notifier.sent[-1]


@pytest.mark.parametrize("status,reason", [
    ("skipped_limit", "photo-check limit"),
    ("ai_failed", "image AI unavailable"),
])
def test_unchecked_photo_is_called_out(notifier, status, reason):
    msg = _hot(notifier, {"photo_type": "uncertain"}, image_status=status, text_reviewed=True)
    assert "Photos NOT checked by AI" in msg
    assert reason in msg
    # Must not read like a verdict the model actually reached.
    assert "couldn't tell" not in msg


def test_checked_but_undecided_photo_reads_differently(notifier):
    msg = _hot(notifier, {"photo_type": "uncertain"}, image_status="checked", text_reviewed=True)
    assert "NOT checked" not in msg
    assert "AI checked the main photo but couldn't tell" in msg


def test_checked_real_photo(notifier):
    msg = _hot(notifier, {"photo_type": "real", "condition_matches_description": True},
               image_status="checked_earlier", text_reviewed=True)
    assert "match the description" in msg
    assert "NOT checked" not in msg


def test_metadata_stock_hint_survives_when_unchecked(notifier):
    msg = _hot(notifier, {"photo_type": "stock"}, image_status="skipped_limit", text_reviewed=True)
    assert "Photos NOT checked by AI" in msg
    assert "stock image" in msg


def test_thumbnail_only_photos(notifier):
    msg = _hot(notifier, {"photo_type": "uncertain"}, image_status="photos_too_small", text_reviewed=True)
    assert "thumbnail-sized" in msg


def test_no_photo(notifier):
    msg = _hot(notifier, {"photo_type": "uncertain"}, image_status="no_photo", text_reviewed=True)
    assert "no photo" in msg


def test_failed_text_review_is_called_out(notifier):
    msg = _hot(notifier, {"photo_type": "real", "condition_matches_description": True,
                          "condition_assessment": "unknown", "reasoning": "Could not perform AI analysis"},
               image_status="checked", text_reviewed=False)
    assert "Description NOT reviewed by AI" in msg
    assert "Could not perform AI analysis" not in msg


def test_legacy_scores_without_status_keep_old_rendering(notifier):
    msg = _hot(notifier, {"photo_type": "real", "condition_matches_description": True,
                          "condition_assessment": "good"})
    assert "match the description" in msg
    assert "NOT" not in msg


def test_summary_marks_unreviewed_rows(notifier):
    deals = [
        {"listing": {**LISTING, "title": "a"}, "analysis": {},
         "scores": _scores(product_key="a", image_status="skipped_limit", text_reviewed=True)},
        {"listing": {**LISTING, "title": "b"}, "analysis": {},
         "scores": _scores(product_key="b", image_status="checked", text_reviewed=False)},
        {"listing": {**LISTING, "title": "c"}, "analysis": {},
         "scores": _scores(product_key="c", image_status="checked", text_reviewed=True)},
    ]
    notifier.send_deal_summary(deals)
    rows = {line.split(">")[1].split("<")[0]: line for line in notifier.sent[-1].splitlines() if "/itm/" in line}
    assert "🙈" in rows["a"] and "📝❌" not in rows["a"]
    assert "📝❌" in rows["b"] and "🙈" not in rows["b"]
    assert "🙈" not in rows["c"] and "📝❌" not in rows["c"]
    assert "🙈 = photos not checked by AI" in notifier.sent[-1]


def test_best_case_image_score_is_an_upper_bound():
    # The budget gate skips a photo check when even BEST_IMAGE_SCORE can't make
    # the listing a deal — only safe if no photo verdict can ever score higher.
    from src.main import BEST_IMAGE_SCORE, _image_score_for
    for photo_type in ("real", "stock", "uncertain", ""):
        for matches in (True, False, None):
            assert _image_score_for(photo_type, matches) <= BEST_IMAGE_SCORE


def test_photo_findings_can_only_lower_the_ai_penalty():
    scorer = DealScorer()
    text = {"condition_assessment": "good", "red_flags": ["light scratches"], "is_good_deal": True}
    image = {"photo_type": "real", "visible_damage": ["dent"]}
    assert scorer.calculate_ai_penalty(text, image) <= scorer.calculate_ai_penalty(text, {})
