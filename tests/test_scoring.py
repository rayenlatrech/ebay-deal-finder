"""Tests for price scoring, including per-category discount sensitivity."""

import pytest

from src.deal_scorer import DealScorer, is_trusted_seller


@pytest.fixture
def scorer():
    # Reads config/scoring.yaml relative to the project root (pytest cwd).
    return DealScorer()


def test_global_curve_used_by_default(scorer):
    # 25% off on the default curve maps to the 35-tier score (60).
    score, discount, suspicious = scorer.calculate_price_score(75.0, 100.0)
    assert discount == 25.0
    assert score == 60
    assert suspicious is False


def test_per_category_curve_rewards_modest_discounts(scorer):
    # A "smartphone" curve: 20% off should score high (80), unlike the
    # global curve where 20% only scores 30.
    phone_curve = [
        {"max_discount": 8, "score": 0},
        {"max_discount": 15, "score": 50},
        {"max_discount": 25, "score": 80},
        {"max_discount": 40, "score": 95},
        {"max_discount": 100, "score": 100},
    ]
    phone_score, discount, _ = scorer.calculate_price_score(80.0, 100.0, phone_curve)
    global_score, _, _ = scorer.calculate_price_score(80.0, 100.0)

    assert discount == 20.0
    assert phone_score == 80
    assert global_score == 30
    assert phone_score > global_score


def test_suspiciously_low_price_flagged(scorer):
    _, discount, suspicious = scorer.calculate_price_score(40.0, 100.0)
    assert discount == 60.0
    assert suspicious is True


def test_absurd_discount_does_not_out_score_a_strong_one(scorer):
    # The core scam-guard: past the suspicion threshold the price score is
    # capped, so an 85%-off listing can no longer beat a healthy 40%-off one.
    strong, _, strong_susp = scorer.calculate_price_score(60.0, 100.0)   # 40% off
    absurd, _, absurd_susp = scorer.calculate_price_score(15.0, 100.0)   # 85% off
    assert strong_susp is False
    assert absurd_susp is True
    assert absurd <= strong


def test_suspicious_penalty_applied_to_composite(scorer):
    kwargs = dict(
        price_score=100, condition_score=95, seller_score=70,
        image_score=85, listing_quality_score=60, stock_photo_penalty=0,
    )
    clean = scorer.calculate_composite_score(**kwargs)
    flagged = scorer.calculate_composite_score(**kwargs, is_suspicious=True)
    assert flagged < clean


@pytest.mark.parametrize("pct,sales,expected", [
    (99.0, 500, True),    # established and well rated
    (95.0, 10,  True),    # exactly on the bar
    (100.0, 1,  False),   # perfect score, no track record
    (100.0, 9,  False),   # still too few completed sales
    (94.9, 5000, False),  # huge volume but rating below the bar
    (0, 0, False),
])
def test_is_trusted_seller(pct, sales, expected):
    listing = {"seller_feedback_percentage": pct, "seller_feedback_score": sales}
    assert is_trusted_seller(listing) is expected


def test_is_trusted_seller_handles_missing_fields():
    assert is_trusted_seller({}) is False
    assert is_trusted_seller({"seller_feedback_percentage": None,
                              "seller_feedback_score": None}) is False


def test_collapse_duplicates_keeps_only_the_best_per_product():
    from src.telegram_notifier import TelegramNotifier

    def deal(key, score, price):
        return {
            "listing": {"title": key, "total_price": price},
            "scores": {"product_key": key, "deal_score": score},
        }

    deals = [
        deal("macbook_pro_m1_pro_14", 71, 600),
        deal("macbook_pro_m1_pro_14", 77, 625),   # best score -> survivor
        deal("macbook_pro_m1_pro_14", 72, 534),
        deal("iphone_15_128gb", 68, 400),
    ]
    out = TelegramNotifier.collapse_duplicates(deals)
    assert len(out) == 2

    mac = next(d for d in out if d["scores"]["product_key"] == "macbook_pro_m1_pro_14")
    assert mac["scores"]["deal_score"] == 77
    assert mac["duplicate_count"] == 2

    iph = next(d for d in out if d["scores"]["product_key"] == "iphone_15_128gb")
    assert iph["duplicate_count"] == 0


def test_collapse_duplicates_passes_through_unkeyed_deals():
    from src.telegram_notifier import TelegramNotifier

    deals = [{"listing": {"title": "x", "total_price": 10}, "scores": {"deal_score": 60}}]
    assert len(TelegramNotifier.collapse_duplicates(deals)) == 1


@pytest.mark.parametrize("item_id,expected", [
    ("v1|335678901234|0", "https://www.ebay.it/itm/335678901234"),
    ("226012345678", "https://www.ebay.it/itm/226012345678"),
    ("bad-id", ""),          # unparseable -> caller keeps the original url
    ("", ""),
])
def test_home_item_url_rewrite(item_id, expected):
    # Foreign domains (ebay.nl) don't deep-link into the eBay app, and item
    # numbers are global, so links are rebuilt on the home domain.
    from src.ebay_client import EbayClient
    assert EbayClient.home_item_url(item_id, "EBAY_IT") == expected


def test_home_item_url_unknown_marketplace_is_skipped():
    from src.ebay_client import EbayClient
    assert EbayClient.home_item_url("v1|123456789|0", "EBAY_XX") == ""


def test_overpriced_scores_zero(scorer):
    score, discount, _ = scorer.calculate_price_score(120.0, 100.0)
    assert score == 0
    assert discount < 0
