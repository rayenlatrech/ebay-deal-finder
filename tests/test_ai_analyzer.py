"""Tests for AI rate-limit handling and the assessment cache rules.

These guard the failure chain found in production: per-minute rate limits
treated as hard failures, Gemini keys benched for 24h over a per-minute limit,
and one failed image check poisoning a good text analysis in the cache.
"""

import sys
import types

import pytest

import src.ai_analyzer as ai_mod
from src.ai_analyzer import (
    AIAnalyzer,
    TEXT_FAILURE_MARKER,
    image_assessment_reusable,
    is_daily_quota_error,
    parse_retry_seconds,
    split_assessment,
    text_assessment_reusable,
)


# ── Retry-delay parsing ────────────────────────────────────────────────────

@pytest.mark.parametrize("message,expected", [
    ("Please try again in 510ms. Need more tokens?", 0.51),
    ("Please try again in 2.3175s. Need more", 2.3175),
    ("Please try again in 5m55.104s. Need more", 355.104),
    ("Please try again in 112.499999ms.", 0.112499999),
    ("{'retryDelay': '33s'}", 33.0),
    ('"retryDelay": "7s"', 7.0),
    ("no delay here", None),
    ("", None),
])
def test_parse_retry_seconds(message, expected):
    got = parse_retry_seconds(message)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("message,daily", [
    ("Rate limit reached ... on tokens per minute (TPM): Limit 8000", False),
    ("Rate limit reached ... on tokens per day (TPD): Limit 100000", True),
    ("quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier", False),
    ("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier", True),
])
def test_daily_vs_per_minute_quota(message, daily):
    assert is_daily_quota_error(message) is daily


# ── Cache reuse rules ──────────────────────────────────────────────────────

GOOD_TEXT = {"condition_assessment": "good", "seller_reasoning": "established", "red_flags": []}
FAILED_IMAGE = {"image_ai_failed": True, "photo_type": "uncertain", "photo_confidence": 0.0}
REAL_IMAGE = {"photo_type": "real", "photo_confidence": 0.9, "image_analyzed": True}


def test_image_failure_no_longer_blocks_the_text_half():
    # The production bug: a failed image check marked the whole merged
    # assessment unusable, so a good text analysis was paid for again.
    text, image = split_assessment({**GOOD_TEXT, **FAILED_IMAGE})
    assert text_assessment_reusable(text) is True
    assert image_assessment_reusable(image) is False


def test_legacy_combined_flag_row_keeps_its_good_text():
    # Rows written before the split carry "ai_failed" with no hint which half
    # failed. The text half is judged on its own content.
    legacy = {**GOOD_TEXT, "ai_failed": True, "photo_type": "uncertain", "photo_confidence": 0.0}
    text, image = split_assessment(legacy)
    assert "ai_failed" not in text
    assert text_assessment_reusable(text) is True
    assert image_assessment_reusable(image) is False


def test_failed_text_default_is_never_reused():
    assert text_assessment_reusable({
        "text_ai_failed": True, "condition_assessment": "unknown",
        "seller_reasoning": TEXT_FAILURE_MARKER,
    }) is False
    # Legacy failure default with no explicit flag.
    assert text_assessment_reusable({
        "condition_assessment": "unknown", "seller_reasoning": TEXT_FAILURE_MARKER,
    }) is False


def test_image_never_analysed_is_not_locked_in_as_uncertain():
    # Vision budget ran out, so only the "uncertain" placeholder was stored.
    assert image_assessment_reusable({"photo_type": "uncertain"}) is False


def test_real_photo_verdict_is_reused():
    assert image_assessment_reusable(REAL_IMAGE) is True
    # Legacy row: no flag, but a genuine confidence score.
    assert image_assessment_reusable({"photo_type": "real", "photo_confidence": 0.85}) is True


def test_split_assessment_handles_empty():
    assert split_assessment(None) == ({}, {})


# ── Groq: per-minute limits are waited out, daily limits are not ───────────

class _RateLimit(Exception):
    status_code = 429


class _FakeGroq:
    """Raises the queued errors in order, then returns a response."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0
        self.last_kwargs = None
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.errors:
            raise self.errors.pop(0)
        message = types.SimpleNamespace(content='{"condition_assessment": "good"}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _analyzer(**attrs):
    a = AIAnalyzer.__new__(AIAnalyzer)
    a.groq_model = "openai/gpt-oss-120b"
    a.groq_reasoning_effort = "low"
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(ai_mod.time, "sleep", lambda s: slept.append(s))
    return slept


def test_groq_per_minute_limit_is_retried(no_sleep):
    fake = _FakeGroq([_RateLimit("tokens per minute (TPM): Limit 8000 ... try again in 1.1s.")])
    a = _analyzer(groq_client=fake)
    assert a._call_groq("prompt") == '{"condition_assessment": "good"}'
    assert fake.calls == 2
    assert no_sleep and no_sleep[0] == pytest.approx(1.6)   # asked wait + 0.5s margin


def test_groq_daily_limit_is_not_waited_out(no_sleep):
    fake = _FakeGroq([_RateLimit("tokens per day (TPD): Limit 100000 ... try again in 14m23s.")])
    a = _analyzer(groq_client=fake)
    assert a._call_groq("prompt") is None
    assert fake.calls == 1
    assert no_sleep == []


def test_groq_uses_low_reasoning_effort_for_gpt_oss(no_sleep):
    fake = _FakeGroq([])
    _analyzer(groq_client=fake)._call_groq("prompt")
    assert fake.last_kwargs["reasoning_effort"] == "low"
    assert fake.last_kwargs["max_tokens"] <= 700


def test_groq_omits_reasoning_effort_for_other_models(no_sleep):
    fake = _FakeGroq([])
    _analyzer(groq_client=fake, groq_model="llama-3.3-70b-versatile")._call_groq("prompt")
    assert "reasoning_effort" not in fake.last_kwargs


# ── Gemini: per-minute limits keep the key, daily limits bench it ──────────

class _FakeDB:
    def __init__(self):
        self.benched = []

    def set_key_exhausted(self, key_hash, until):
        self.benched.append((key_hash, until))


def _install_fake_genai(monkeypatch, errors):
    """Replace google.genai with a stub that raises the queued errors in order."""
    state = {"calls": 0, "errors": list(errors)}

    class _Models:
        def generate_content(self, **kwargs):
            state["calls"] += 1
            if state["errors"]:
                raise state["errors"].pop(0)
            return types.SimpleNamespace(text='{"photo_type": "real"}')

    class _Client:
        def __init__(self, api_key):
            self.models = _Models()

    genai = types.ModuleType("google.genai")
    genai.Client = _Client
    gtypes = types.ModuleType("google.genai.types")
    for name in ("Content", "Part", "GenerateContentConfig"):
        setattr(gtypes, name, type(name, (), {
            "__init__": lambda self, **k: None,
            "from_bytes": staticmethod(lambda **k: None),
            "from_text": staticmethod(lambda **k: None),
        }))
    genai.types = gtypes
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    return state


def _gemini_analyzer(keys):
    return _analyzer(
        db=_FakeDB(), gemini_keys=list(keys), gemini_model="gemini-2.5-flash",
        _gemini_key_index=0, gemini_min_interval=0.0, _gemini_last_call={},
    )


def test_gemini_per_minute_limit_does_not_bench_the_key(monkeypatch, no_sleep):
    state = _install_fake_genai(monkeypatch, [
        Exception("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier retryDelay: '12s'"),
    ])
    a = _gemini_analyzer(["key-a"])
    assert a._call_gemini("prompt") == '{"photo_type": "real"}'
    assert a.db.benched == []            # previously: benched for 24 hours
    assert a.gemini_keys == ["key-a"]
    assert 12.0 in no_sleep
    assert state["calls"] == 2


def test_gemini_daily_limit_benches_the_key_until_reset(monkeypatch, no_sleep):
    _install_fake_genai(monkeypatch, [
        Exception("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
    ])
    a = _gemini_analyzer(["key-a", "key-b"])
    assert a._call_gemini("prompt") == '{"photo_type": "real"}'   # key-b serves it
    assert len(a.db.benched) == 1
    assert a.gemini_keys == ["key-b"]


def _capture_parts(monkeypatch, analyzer):
    """Record the image parts each Gemini request carries."""
    sent = []
    gtypes = sys.modules["google.genai.types"]
    monkeypatch.setattr(gtypes.Part, "from_bytes",
                        staticmethod(lambda **k: sent.append(k["data"]) or "img"), raising=False)
    calls = []
    original = analyzer._call_gemini
    monkeypatch.setattr(analyzer, "_call_gemini",
                        lambda prompt, image_data=None: calls.append(prompt) or original(prompt, image_data))
    return sent, calls


def test_three_photos_go_in_one_request(monkeypatch, no_sleep):
    state = _install_fake_genai(monkeypatch, [])
    a = _gemini_analyzer(["key-a"])
    sizes = {}
    monkeypatch.setattr(a, "_download_image",
                        lambda url, max_dim=1024: sizes.setdefault(url, max_dim) and url.encode())
    sent, prompts = _capture_parts(monkeypatch, a)

    out = a.analyze_image("https://img/1", extra_image_urls=["https://img/2", "https://img/3"])

    assert state["calls"] == 1                          # one request, not three
    assert sent == [b"https://img/1", b"https://img/2", b"https://img/3"]
    assert sizes == {"https://img/1": 1024, "https://img/2": 768, "https://img/3": 768}
    assert "3 photo(s)" in prompts[0]
    assert out["photos_reviewed"] == 3 and out["image_analyzed"] is True


def test_failed_extra_photo_still_analyses_the_rest(monkeypatch, no_sleep):
    _install_fake_genai(monkeypatch, [])
    a = _gemini_analyzer(["key-a"])
    monkeypatch.setattr(a, "_download_image",
                        lambda url, max_dim=1024: None if url.endswith("2") else url.encode())
    sent, _ = _capture_parts(monkeypatch, a)
    out = a.analyze_image("https://img/1", extra_image_urls=["https://img/2", "https://img/3"])
    assert sent == [b"https://img/1", b"https://img/3"]
    assert out["photos_reviewed"] == 2


def test_no_downloadable_photo_is_a_failed_check(monkeypatch, no_sleep):
    state = _install_fake_genai(monkeypatch, [])
    a = _gemini_analyzer(["key-a"])
    monkeypatch.setattr(a, "_download_image", lambda url, max_dim=1024: None)
    out = a.analyze_image("https://img/1", extra_image_urls=["https://img/2"])
    assert out["image_ai_failed"] is True
    assert state["calls"] == 0


def test_thumbnails_are_skipped_for_later_photos(monkeypatch, no_sleep):
    from src.ai_analyzer import TOO_SMALL
    _install_fake_genai(monkeypatch, [])
    a = _gemini_analyzer(["key-a"])
    tiny = {"https://img/1", "https://img/3"}
    tried = []
    monkeypatch.setattr(a, "_download_image",
                        lambda url, max_dim=1024: tried.append(url) or (TOO_SMALL if url in tiny else url.encode()))
    sent, _ = _capture_parts(monkeypatch, a)
    out = a.analyze_image("https://img/1", extra_image_urls=[f"https://img/{i}" for i in range(2, 8)])
    assert sent == [b"https://img/2", b"https://img/4", b"https://img/5"]   # first 3 usable
    assert tried == [f"https://img/{i}" for i in range(1, 6)]               # stops once it has 3
    assert out["photos_reviewed"] == 3


def test_first_usable_photo_gets_the_detailed_size(monkeypatch, no_sleep):
    from src.ai_analyzer import TOO_SMALL
    _install_fake_genai(monkeypatch, [])
    a = _gemini_analyzer(["key-a"])
    sizes = {}
    monkeypatch.setattr(a, "_download_image", lambda url, max_dim=1024: sizes.setdefault(url, max_dim)
                        and (TOO_SMALL if url.endswith("1") else url.encode()))
    _capture_parts(monkeypatch, a)
    a.analyze_image("https://img/1", extra_image_urls=["https://img/2", "https://img/3"])
    assert sizes["https://img/2"] == 1024 and sizes["https://img/3"] == 768


def test_all_thumbnails_spend_no_request(monkeypatch, no_sleep):
    from src.ai_analyzer import TOO_SMALL, image_assessment_reusable
    state = _install_fake_genai(monkeypatch, [])
    a = _gemini_analyzer(["key-a"])
    monkeypatch.setattr(a, "_download_image", lambda url, max_dim=1024: TOO_SMALL)
    out = a.analyze_image("https://img/1", extra_image_urls=["https://img/2"])
    assert out["photos_too_small"] is True
    assert not out.get("image_ai_failed")
    assert state["calls"] == 0
    assert not image_assessment_reusable(out)    # re-checked if the seller adds photos


def test_photo_attempts_are_capped(monkeypatch, no_sleep):
    from src.ai_analyzer import TOO_SMALL
    _install_fake_genai(monkeypatch, [])
    a = _gemini_analyzer(["key-a"])
    tried = []
    monkeypatch.setattr(a, "_download_image", lambda url, max_dim=1024: tried.append(url) or TOO_SMALL)
    a.analyze_image("https://img/0", extra_image_urls=[f"https://img/{i}" for i in range(1, 30)])
    assert len(tried) == a.MAX_PHOTO_ATTEMPTS


def test_download_flags_thumbnails(monkeypatch):
    import io
    from PIL import Image
    from src.ai_analyzer import TOO_SMALL

    def png(size):
        buf = io.BytesIO(); Image.new("RGB", size).save(buf, format="PNG"); return buf.getvalue()

    class _Resp:
        def __init__(self, content): self.content = content
        def raise_for_status(self): pass

    a = _gemini_analyzer(["key-a"])
    monkeypatch.setattr(ai_mod.requests, "get", lambda url, timeout: _Resp(png((80, 80))))
    assert a._download_image("https://img/x") is TOO_SMALL
    monkeypatch.setattr(ai_mod.requests, "get", lambda url, timeout: _Resp(png((1600, 1200))))
    out = a._download_image("https://img/x", 768)
    assert isinstance(out, bytes) and max(Image.open(io.BytesIO(out)).size) == 768


def test_prompts_carry_todays_date():
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%d %B %Y")
    assert today in AIAnalyzer.IMAGE_ANALYSIS_PROMPT.format(today=today, photo_count=3, condition="x",
                                                          description_excerpt="y")
    text = AIAnalyzer.TEXT_ANALYSIS_PROMPT.format(
        today=today, title="t", price=1, avg_price=1, discount_pct=1, condition="c", buying_options="",
        description="d", feedback_score=1, feedback_pct=99, negotiable="false")
    assert today in text and "non-existent" in text


def test_extra_photos_use_ebays_smaller_rendition():
    from src.ai_analyzer import _ebay_image_size
    url = "https://i.ebayimg.com/images/g/SqEAAeSwTVlqzIhi/s-l1600.jpg"
    assert _ebay_image_size(url, 960) == "https://i.ebayimg.com/images/g/SqEAAeSwTVlqzIhi/s-l960.jpg"
    assert _ebay_image_size("https://example.com/photo.jpg", 960) == "https://example.com/photo.jpg"


def test_photos_reviewed_is_cached_with_the_image_half():
    _, image = split_assessment({"condition_assessment": "good", "photo_type": "real",
                                 "image_analyzed": True, "photos_reviewed": 3})
    assert image["photos_reviewed"] == 3


def test_gemini_reset_time_is_in_the_future():
    from datetime import datetime, timezone
    assert AIAnalyzer._next_gemini_reset() > datetime.now(timezone.utc)
