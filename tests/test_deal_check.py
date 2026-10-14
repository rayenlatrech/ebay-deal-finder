"""Telegram deal check: parsing, verdicts, and who gets answered."""

import pytest

from src.baseline import Baseline
from src.deal_check import (
    MarketView, build_reply, classify, handle_update, parse_band, parse_price,
    parse_request, verdict,
)


@pytest.mark.parametrize("text,price", [
    ("iPhone 15 Pro 256GB usato 550€", 550),
    ("iPhone 15 Pro 256GB € 549,90", 549.9),
    ("PS5 slim sigillata 380 euro", 380),
    ("S24 Ultra 512GB come nuovo 600", 600),         # bare number, after the storage
    ("iPhone 13 2021 128GB batteria 89% 300", 300),   # year and percentage skipped
    ("iPhone 15 Pro 256GB", None),                    # no price stated
    ("MacBook Air M2 13 pollici 512GB", None),
])
def test_parse_price(text, price):
    assert parse_price(text) == price


@pytest.mark.parametrize("text,band", [
    ("iPhone 15 usato", "used"),
    ("iPhone 15 come nuovo", "used"),       # "like new" is still used
    ("iPhone 15 nuovo sigillato", "new"),
    ("Galaxy S24 sealed", "new"),
    ("iPhone 14 ricondizionato", "refurb"),
    ("iPhone 12 per ricambi", "parts"),
])
def test_parse_band(text, band):
    assert parse_band(text) == band


def test_ebay_link_is_recognised():
    r = parse_request("guarda questo https://www.ebay.de/itm/Apple-iPhone-15/335678901234?hash=x")
    assert r.ebay_item_id == "v1|335678901234|0"
    assert r.ebay_marketplace == "EBAY_DE"
    r = parse_request("https://www.ebay.it/itm/226012345678")
    assert (r.ebay_item_id, r.ebay_marketplace) == ("v1|226012345678|0", "EBAY_IT")


@pytest.mark.parametrize("text,key", [
    ("S24 Ultra 512GB come nuovo 600", "galaxy_s24_ultra_512gb"),   # "Galaxy" omitted
    ("iphone 15 pro 256 gb", "iphone_15_pro_256gb"),
    ("Galaxy A57 256GB", "galaxy_a57_256gb"),
])
def test_classify(text, key):
    assert classify(text)[0] == key


def test_verdicts():
    assert verdict(600, 800, 12).label.startswith("✅")
    assert verdict(760, 800, 12).label.startswith("👌")
    assert verdict(900, 800, 12).label.startswith("❌")
    assert verdict(552, 551, 12).label.startswith("👌")     # 0.2% over is "at market"
    assert verdict(300, 800, 12).label.startswith("🚨")
    assert verdict(600, 800, 12).max_price == 704


def _view(value=758, n=55, source="market"):
    return MarketView(Baseline(value, source, n, 979), 700, 760, 820, 40,
                      [{"id": "v1|123456789|0", "price": 690, "marketplace": "EBAY_IT"}])


def test_reply_contains_the_verdict_and_evidence():
    r = parse_request("iPhone 17 256GB usato 649€")
    text = build_reply(r, "iphone_17_256gb", "used", _view(), {"min_discount": 12},
                       {"note": "Solid price.", "check_before_buying": ["Battery health"],
                        "warning_signs": []}, 14)
    assert "€758 (55 listings, 14 days)" in text
    assert "Good deal" in text and "14% under" in text
    assert "Pay at most <b>€667</b>" in text
    assert "Battery health" in text
    assert 'href="https://www.ebay.it/itm/123456789"' in text


def test_reply_without_price_gives_the_buy_below_figure():
    r = parse_request("iPhone 17 256GB usato")
    text = build_reply(r, "iphone_17_256gb", "used", _view(), {"min_discount": 12}, {}, 14)
    assert "under <b>€667</b>" in text
    assert "AI review unavailable" in text


def test_placeholder_warnings_are_not_shown():
    r = parse_request("iPhone 17 256GB 649€")
    text = build_reply(r, "iphone_17_256gb", "used", _view(), {"min_discount": 12},
                       {"note": "ok", "warning_signs": ["None"]}, 14)
    assert "Warning signs" not in text


@pytest.mark.parametrize("key,name", [
    ("iphone_15_pro_256gb", "iPhone 15 Pro 256GB"),
    ("galaxy_s24_ultra_1tb", "Galaxy S24 Ultra 1TB"),
    ("macbook_air_m3_13", "MacBook Air M3 13"),
])
def test_pretty_name(key, name):
    from src.deal_check import pretty_name
    assert pretty_name(key) == name


def test_reply_escapes_ai_text():
    r = parse_request("iPhone 17 256GB 649€")
    text = build_reply(r, "iphone_17_256gb", "used", _view(), {"min_discount": 12},
                       {"note": "<b>x</b> & y"}, 14)
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in text


# ── who gets answered ──────────────────────────────────────────────────

class _Checker:
    calls = 0

    def check(self, text):
        _Checker.calls += 1
        return f"checked: {text}"


def _update(chat_id="42", text="iPhone 15 500€", date=1000):
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "text": text, "date": date}}


@pytest.fixture
def sent():
    out = []
    return out


def test_owner_gets_an_answer(sent):
    assert handle_update(_update(), _Checker, lambda t: sent.append(t) or True, "42", now=1010)
    assert sent == ["checked: iPhone 15 500€"]


def test_strangers_are_ignored(sent):
    _Checker.calls = 0
    assert not handle_update(_update(chat_id="999"), _Checker, lambda t: sent.append(t) or True, "42", now=1010)
    assert sent == [] and _Checker.calls == 0


def test_stale_redeliveries_are_ignored(sent):
    assert not handle_update(_update(date=1000), _Checker, lambda t: sent.append(t) or True, "42", now=1000 + 3600)
    assert sent == []


def test_help_command(sent):
    handle_update(_update(text="/start"), _Checker, lambda t: sent.append(t) or True, "42", now=1010)
    assert "Deal check" in sent[0]


def test_a_failing_check_still_answers(sent):
    class Boom:
        def check(self, text):
            raise RuntimeError("db down")
    handle_update(_update(), Boom, lambda t: sent.append(t) or True, "42", now=1010)
    assert "went wrong" in sent[0]
