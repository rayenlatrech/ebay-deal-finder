"""Tests for price scoring, including per-category discount sensitivity."""

import pytest

from src.deal_scorer import DealScorer


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


def test_overpriced_scores_zero(scorer):
    score, discount, _ = scorer.calculate_price_score(120.0, 100.0)
    assert score == 0
    assert discount < 0
