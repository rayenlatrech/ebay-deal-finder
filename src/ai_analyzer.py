import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Sequence
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger(__name__)


# Seller_reasoning value written by the text default. Older rows carry no
# explicit failure flag, so this marker is how their text half is recognised.
TEXT_FAILURE_MARKER = "AI analysis unavailable"

_RETRY_IN_RE = re.compile(r"try again in\s+([0-9hms.]+)", re.IGNORECASE)
_DURATION_PART_RE = re.compile(r"([\d.]+)(ms|h|m|s)")
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]([\d.]+)s", re.IGNORECASE)


def parse_retry_seconds(message: str) -> Optional[float]:
    """Seconds to wait from a rate-limit message, or None if none is given.

    Handles Groq's "try again in 5m55.104s" / "510ms" and Gemini's
    "retryDelay": "33s".
    """
    text = message or ""
    m = _RETRY_IN_RE.search(text)
    if m:
        total, found = 0.0, False
        for value, unit in _DURATION_PART_RE.findall(m.group(1)):
            try:
                number = float(value)
            except ValueError:
                continue
            total += number * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
            found = True
        if found:
            return total
    m = _RETRY_DELAY_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def is_daily_quota_error(message: str) -> bool:
    """True if a rate-limit error is about a daily quota rather than a
    per-minute one. Only daily exhaustion justifies benching a key for hours."""
    text = (message or "").lower()
    return "perday" in text or "per day" in text or "requests per day" in text


def text_assessment_reusable(assessment: Optional[dict]) -> bool:
    """A stored assessment's text half can be reused unless it's the failure default."""
    if not assessment or not assessment.get("condition_assessment"):
        return False
    if assessment.get("text_ai_failed"):
        return False
    return assessment.get("seller_reasoning") != TEXT_FAILURE_MARKER


def image_assessment_reusable(assessment: Optional[dict]) -> bool:
    """A stored photo verdict can be reused only if a vision model produced it.

    An image that was never analysed (vision budget spent) or whose analysis
    failed must be retried later rather than locked in as "uncertain".
    """
    if not assessment or assessment.get("image_ai_failed"):
        return False
    if assessment.get("image_analyzed"):
        return True
    # Rows written before the explicit flag: a real verdict carries a
    # non-zero photo_confidence, the failure default carries 0.0.
    try:
        return float(assessment.get("photo_confidence") or 0) > 0
    except (TypeError, ValueError):
        return False


# Fields belonging to the photo verdict within a stored (merged) assessment.
# Returned by _download_image for a thumbnail-sized photo.
TOO_SMALL = object()


def _today() -> str:
    """Today's date for prompts — models otherwise judge new products by their
    training cutoff (Gemini called the iPhone 16 Pro Max "a model that does not
    exist", Gemma called a 2-year-old one impossibly old)."""
    return datetime.now(timezone.utc).strftime("%d %B %Y")


# Shared prompt rule: the model's product knowledge ends before today.
DATE_RULE = (
    "Today's date is {today}. Your knowledge of products ends before that: a model "
    "you don't recognise (a newer iPhone, Galaxy, etc.) is real. Never call a model "
    "non-existent, or a stated age or purchase date impossible, from your own knowledge."
)

_EBAY_IMAGE_SIZE_RE = re.compile(r"(i\.ebayimg\.com/.+/s-l)\d+(\.\w+)$")


def _ebay_image_size(url: str, size: int) -> str:
    """The same eBay photo at another rendition size ("s-l1600" → "s-l960")."""
    return _EBAY_IMAGE_SIZE_RE.sub(rf"\g<1>{size}\g<2>", url)


IMAGE_ASSESSMENT_KEYS = frozenset({
    "photo_type", "photo_confidence", "photo_signals", "visible_damage",
    "condition_matches_description", "estimated_condition", "concerns",
    "confidence", "image_analyzed", "image_ai_failed", "photos_reviewed", "photos_too_small",
})


def split_assessment(assessment: Optional[dict]) -> tuple[dict, dict]:
    """Split a stored assessment into (text_part, image_part).

    Text and photo verdicts are stored merged, but they fail independently, so
    each half has to be judged and reused on its own. The legacy combined
    "ai_failed" flag is dropped: it can't say which half failed.
    """
    if not assessment:
        return {}, {}
    text_part = {
        k: v for k, v in assessment.items()
        if k not in IMAGE_ASSESSMENT_KEYS and k not in ("ai_failed", "quick_photo_flags")
    }
    image_part = {k: v for k, v in assessment.items() if k in IMAGE_ASSESSMENT_KEYS}
    return text_part, image_part


def _parse_ai_json(text: str) -> dict:
    """Extract and parse JSON from AI response, handling markdown code blocks."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse AI JSON response: %s\nRaw: %s", e, text[:500])
        return {}


class AIAnalyzer:
    """Multi-provider AI analyzer with Gemini key rotation and Groq fallback."""

    # ── Prompt Templates ─────────────────────────────────────────────

    TEXT_ANALYSIS_PROMPT = """You are an expert at evaluating used electronics listings on eBay Italy.
The description may be in Italian — analyze it regardless of language.
Always respond in English.
""" + DATE_RULE + """

LISTING DATA:
- Title: {title}
- Price: €{price} (Market avg: €{avg_price}, Discount: {discount_pct}%)
- Condition stated: {condition}
- Buying options: {buying_options}
- Description: {description}
- Seller feedback score: {feedback_score} ({feedback_pct}% positive)

ANALYZE and return ONLY valid JSON (no markdown, no explanation):
{{
  "condition_assessment": "mint|good|fair|poor|damaged|parts_only",
  "red_flags": ["list any concerning phrases like 'per ricambi', 'iCloud bloccato', 'schermo rotto', 'no reso' or English equivalents"],
  "hidden_defects": ["functional problems buried in the description that the title/condition don't reveal"],
  "seller_trust": "high|medium|low",
  "seller_reasoning": "brief explanation of trust rating",
  "negotiable": {negotiable},
  "is_complete_item": true,  // MUST BE FALSE if it is just a box, single earbud, cable, or component part!
  "is_good_deal": true,      // MUST BE FALSE if is_complete_item is false or if broken/parts_only
  "reasoning": "brief explanation in English"
}}

Rules:
- A working device sold WITHOUT its box, charger or accessories IS a complete item.
- red_flags are for real risks: locks (iCloud/Google/carrier), not working, broken
  or cracked parts, water damage, swollen battery, blacklisted IMEI, "for parts".
  Normal wear such as light scratches is NOT a red flag — reflect it in
  condition_assessment instead. Use an empty list when there are none.
- hidden_defects are FUNCTIONAL problems only (e.g. Face ID not working,
  battery health below 80%, dead pixels, faulty speaker or camera). A battery
  health of 80% or more, cosmetic wear, or a missing box/charger is NOT a
  defect. Use an empty list when there are none."""

    IMAGE_ANALYSIS_PROMPT = """You are an expert at analyzing eBay product photos for used electronics.
""" + DATE_RULE + """

You are given {photo_count} photo(s) from the same listing; the first is the listing's main photo.

FIRST, determine whether the photos show the specific item for sale or are stock/catalog images.
Stock photo signs: pure white background, perfect studio lighting, commercially polished, no visible wear, looks like a manufacturer's product page.
Real photo signs: home/desk background, ambient lighting, visible wear marks, phone camera quality, personal items in background, reflections.
Sellers often use a catalog image as the main photo and real ones after it: answer "real" if ANY photo clearly shows the actual unit, "stock" only if they are all catalog images.

THEN, assess the item's physical condition from every real photo — check the screen, back and edges for scratches, cracks and dents.

The seller described this item as: "{condition}" condition.
The listing says: "{description_excerpt}"

Return ONLY valid JSON (no markdown, no explanation):
{{
  "photo_type": "real|stock|uncertain",
  "photo_confidence": 0.85,
  "photo_signals": ["list specific visual cues"],
  "visible_damage": ["each scratch, dent, crack or discoloration you see — [] if none"],
  "condition_matches_description": true,
  "estimated_condition": "mint|good|fair|poor|cannot_assess",
  "concerns": ["any red flags from the image"],
  "confidence": 0.85
}}"""



    def __init__(self, db_instance) -> None:
        """Initialize AI clients from environment variables."""
        self.db = db_instance
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        # llama-3.3-70b-versatile was retired by Groq on 2026-08-16; gpt-oss-120b
        # is their recommended replacement and is a larger model.
        self.groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        # Cerebras serves the same 70B class model with a far larger free daily
        # token budget, so it backs Groq up rather than replacing the model.
        self.cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        self.cerebras_model = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")

        # Load all Gemini keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
        self.gemini_keys = []
        for key, value in os.environ.items():
            if key.startswith("GEMINI_API_KEY_") and value:
                # Check DB for cooldown
                key_hash = hashlib.sha256(value.encode()).hexdigest()
                cooldown = self.db.get_key_cooldown(key_hash)
                if cooldown:
                    logger.warning("Gemini key %s is in cooldown until %s", key, cooldown.isoformat())
                else:
                    self.gemini_keys.append(value)
        
        self._gemini_key_index = 0
        # Free tier allows ~10 requests/minute per key. Firing image checks
        # back-to-back tripped it on nearly every run, so each key is paced.
        self.gemini_min_interval = float(os.environ.get("GEMINI_MIN_INTERVAL_S", "6.5"))
        self._gemini_last_call: dict[str, float] = {}

        # gpt-oss is a reasoning model; low effort cuts output ~3x (fewer tokens
        # against the per-minute limit) with no loss for this classification.
        self.groq_reasoning_effort = os.environ.get("GROQ_REASONING_EFFORT", "low")

        # Initialize Groq client. Its built-in retries back off too briefly for
        # a saturated per-minute token budget, so retries are handled here with
        # the wait Groq actually asks for.
        groq_key = os.environ.get("GROQ_API_KEY")
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=groq_key, max_retries=0) if groq_key else None
        except ImportError:
            self.groq_client = None
            logger.warning("Groq package not installed, text-only fallback unavailable")

        logger.info(
            "AI Analyzer initialized: %d Gemini key(s), Groq %s, Cerebras %s",
            len(self.gemini_keys),
            "available" if self.groq_client else "unavailable",
            "available" if self.cerebras_key else "unavailable",
        )

    # ── Gemini Calls with Key Rotation ───────────────────────────────

    def _get_next_gemini_key(self) -> str:
        """Round-robin rotation between Gemini API keys."""
        key = self.gemini_keys[self._gemini_key_index % len(self.gemini_keys)]
        self._gemini_key_index += 1
        return key

    def _call_gemini(
        self, prompt: str, image_data: Optional[bytes | list[bytes]] = None
    ) -> Optional[str]:
        """Call Gemini API with key rotation and retry.

        Args:
            prompt: Text prompt
            image_data: Optional image bytes (or a list of them) for vision
                analysis. Several photos go in ONE request, so they cost one
                request against the free tier's request limits, not several.

        Returns:
            Response text or None if all keys exhausted.
        """
        if not self.gemini_keys:
            return None

        from google import genai
        from google.genai import types

        images = [image_data] if isinstance(image_data, bytes) else list(image_data or [])

        # Each key gets a couple of chances: a per-minute limit clears quickly,
        # so it is waited out rather than treated as the key being used up.
        max_attempts = len(self.gemini_keys) * 2
        for attempt in range(max_attempts):
            if not self.gemini_keys:
                break
            api_key = self._get_next_gemini_key()
            self._pace_gemini(api_key)
            try:
                client = genai.Client(api_key=api_key)

                contents: list[Any] = []

                if images:
                    # Build multimodal content: every photo, then the prompt
                    parts = [types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in images]
                    parts.append(types.Part.from_text(text=prompt))
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    contents.append(prompt)

                response = client.models.generate_content(
                    model=self.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
                return response.text

            except Exception as e:
                message = str(e)
                lowered = message.lower()
                if "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered:
                    if is_daily_quota_error(message):
                        # Genuinely used up for the day: bench it until the
                        # daily reset and stop using it for the rest of this run.
                        logger.warning("Gemini key hit its DAILY quota — benched until reset")
                        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                        self.db.set_key_exhausted(key_hash, self._next_gemini_reset())
                        self.gemini_keys = [k for k in self.gemini_keys if k != api_key]
                        continue
                    # Per-minute limit: wait what Gemini asks, then carry on.
                    # Benching the key for a day here is what used to leave
                    # image analysis dead for most of every day.
                    wait = min(parse_retry_seconds(message) or self.gemini_min_interval, 60.0)
                    logger.info("Gemini per-minute limit, waiting %.1fs", wait)
                    time.sleep(wait)
                    continue
                logger.error("Gemini API error: %s", message[:300])
                return None

        logger.warning("Gemini unavailable for this request (%d key(s) active)", len(self.gemini_keys))
        return None

    def _pace_gemini(self, api_key: str) -> None:
        """Keep each key under the free tier's per-minute request limit."""
        last = self._gemini_last_call.get(api_key)
        if last is not None:
            remaining = self.gemini_min_interval - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._gemini_last_call[api_key] = time.monotonic()

    @staticmethod
    def _next_gemini_reset() -> datetime:
        """Gemini daily quotas reset at midnight Pacific; 08:00 UTC covers both
        PST and PDT (it is an hour late in summer, which errs on the safe side)."""
        now = datetime.now(timezone.utc)
        reset = now.replace(hour=8, minute=0, second=0, microsecond=0)
        return reset if reset > now else reset + timedelta(days=1)

    # ── Groq Fallback ────────────────────────────────────────────────

    def _call_groq(self, prompt: str) -> Optional[str]:
        """Call Groq API for text-only analysis (fallback)."""
        if not self.groq_client:
            logger.error("Groq client not available")
            return None

        kwargs: dict[str, Any] = dict(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            # Requested tokens count against the per-minute budget, so reserve
            # what low-effort answers actually need (~130-250) plus margin.
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        if self.groq_model.startswith("openai/gpt-oss") and self.groq_reasoning_effort:
            kwargs["reasoning_effort"] = self.groq_reasoning_effort

        # Every Groq 429 seen in production was the per-MINUTE token limit with
        # "try again in <2s". Those are waited out; a daily limit is not.
        for attempt in range(4):
            try:
                response = self.groq_client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content:
                    return content
                logger.warning("Groq returned an empty response")
                return None
            except Exception as e:
                message = str(e)
                is_rate_limit = getattr(e, "status_code", None) == 429 or "429" in message
                if is_rate_limit and not is_daily_quota_error(message) and attempt < 3:
                    wait = min((parse_retry_seconds(message) or 2.0) + 0.5, 30.0)
                    logger.info("Groq per-minute limit, waiting %.1fs (retry %d/3)", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                logger.error("Groq API error: %s", message[:300])
                return None
        return None

    # ── Cerebras Fallback ────────────────────────────────────────────

    def _call_cerebras(self, prompt: str) -> Optional[str]:
        """Call Cerebras (OpenAI-compatible) for text-only analysis.

        Same 70B-class model as Groq but with a much larger free daily token
        budget, so it takes over when Groq's quota runs out.
        """
        if not self.cerebras_key:
            return None
        try:
            resp = requests.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.cerebras_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.cerebras_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                    "response_format": {"type": "json_object"},
                },
                timeout=45,
            )
            if resp.status_code != 200:
                logger.error("Cerebras API error %d: %s", resp.status_code, resp.text[:200])
                return None
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Cerebras API error: %s", e)
            return None

    def _call_ai(
        self, prompt: str, image_data: Optional[bytes | list[bytes]] = None
    ) -> Optional[str]:
        """Cascade text through the cheapest available provider; images to Gemini.

        Text order is Groq → Cerebras → Gemini. Gemini is deliberately last for
        text because it is the only provider that can do vision, so its quota is
        reserved for image analysis wherever possible.
        """
        if not image_data:
            result = self._call_groq(prompt)
            if result:
                return result
            logger.info("Groq unavailable, trying Cerebras for text")

            result = self._call_cerebras(prompt)
            if result:
                return result
            logger.info("Cerebras unavailable, falling back to Gemini for text")

        # Images always go to Gemini; text lands here only if both others failed.
        result = self._call_gemini(prompt, image_data)
        if result:
            return result

        logger.warning("AI analysis failed completely")
        return None

    # ── Text Analysis ────────────────────────────────────────────────

    def analyze_listing(self, listing: dict[str, Any]) -> dict[str, Any]:
        """Analyze a listing's description, condition, and seller reputation.

        Args:
            listing: Dict with title, price, condition, description, seller info, etc.

        Returns:
            Dict with AI assessment fields (condition_assessment, red_flags, etc.)
        """
        prompt = self.TEXT_ANALYSIS_PROMPT.format(
            today=_today(),
            title=listing.get("title", ""),
            price=listing.get("price", 0),
            avg_price=listing.get("avg_price", "unknown"),
            discount_pct=listing.get("discount_pct", "unknown"),
            condition=listing.get("condition", "unknown"),
            buying_options=", ".join(listing.get("buying_options", [])),
            description=listing.get("description", "No description")[:2000],
            feedback_score=listing.get("seller_feedback_score", 0),
            feedback_pct=listing.get("seller_feedback_percentage", 0),
            negotiable=str(listing.get("negotiable", False)).lower(),
        )

        response = self._call_ai(prompt)
        if not response:
            return self._default_text_analysis()

        parsed = _parse_ai_json(response)
        if not parsed:
            return self._default_text_analysis()

        return parsed

    def _default_text_analysis(self) -> dict[str, Any]:
        """Return default analysis when AI is unavailable.

        Flagged so its text half is never reused from cache: these defaults score
        a listing as unknown/not-a-deal, and replaying them for days would
        suppress real deals long after the outage that produced them. The flag
        is text-specific so an image failure can't drag a good text result down
        with it when the two are merged into one stored assessment.
        """
        return {
            "text_ai_failed": True,
            "condition_assessment": "unknown",
            "red_flags": [],
            "hidden_defects": [],
            "seller_trust": "unknown",
            "seller_reasoning": TEXT_FAILURE_MARKER,
            "negotiable": False,
            "is_good_deal": False,
            "reasoning": "Could not perform AI analysis",
        }

    # ── Image Analysis ───────────────────────────────────────────────

    # The lead photo keeps more detail; the others are sized to fit a single
    # 768px tile, the cheapest size Gemini bills beyond a thumbnail.
    MAIN_PHOTO_MAX_DIM = 1024
    EXTRA_PHOTO_MAX_DIM = 768
    # Below this on its longest side a photo is a thumbnail: nothing about the
    # condition can be read from it (one seller uploaded only 80x80 images).
    MIN_PHOTO_DIM = 400
    # Photos tried before giving up, so a long gallery can't stall a scan.
    MAX_PHOTO_ATTEMPTS = 8

    def analyze_image(
        self,
        image_url: str,
        condition: str = "",
        description_excerpt: str = "",
        extra_image_urls: Sequence[str] = (),
        max_photos: int = 3,
    ) -> dict[str, Any]:
        """Analyze a listing's photos for stock photo detection and condition.

        The main photo is often the front only (or a catalog shot); the next
        photos usually show the back and edges, where damage actually shows.
        Photos are taken in gallery order, skipping thumbnails, until
        `max_photos` usable ones are found; they all go in ONE request, so
        extras don't cost extra requests.

        Args:
            image_url: URL of the listing's main image
            condition: Seller-stated condition
            description_excerpt: Brief excerpt of listing description
            extra_image_urls: The listing's other photos, in gallery order
            max_photos: Usable photos to send

        Returns:
            Dict with photo_type, visible_damage, condition assessment, etc.
        """
        candidates = [u for u in dict.fromkeys([image_url, *extra_image_urls]) if u]
        images: list[bytes] = []
        too_small = 0
        for attempt, url in enumerate(candidates):
            if len(images) >= max_photos or attempt >= self.MAX_PHOTO_ATTEMPTS:
                break
            if images:
                # eBay serves every photo at 1600px; the 960px rendition is a
                # third of the download and is shrunk to 768px regardless.
                got = self._download_image(_ebay_image_size(url, 960), self.EXTRA_PHOTO_MAX_DIM)
            else:
                got = self._download_image(url, self.MAIN_PHOTO_MAX_DIM)
            if got is TOO_SMALL:
                too_small += 1
            elif got:
                images.append(got)
        if not images:
            # Every photo a thumbnail is a finding, not an AI failure — and no
            # request is spent on it.
            return self._too_small_image_analysis() if too_small else self._default_image_analysis()

        prompt = self.IMAGE_ANALYSIS_PROMPT.format(
            today=_today(),
            photo_count=len(images),
            condition=condition or "not specified",
            description_excerpt=description_excerpt[:500] or "no description",
        )

        response = self._call_ai(prompt, images)
        if not response:
            return self._default_image_analysis()

        parsed = _parse_ai_json(response)
        if not parsed:
            return self._default_image_analysis()

        # Marks a genuine vision verdict so the cache can reuse it — and, just as
        # importantly, so an image that was never analysed isn't reused as one.
        parsed["image_analyzed"] = True
        parsed["photos_reviewed"] = len(images)
        return parsed

    def _download_image(self, url: str, max_dim: int = MAIN_PHOTO_MAX_DIM) -> Any:
        """Download an image as JPEG bytes, at most `max_dim` px.

        Returns None if it can't be fetched, and TOO_SMALL if it is a
        thumbnail (under MIN_PHOTO_DIM), so the caller can try the next photo.
        """
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            # Validate it's actually an image and resize if too large
            img = Image.open(BytesIO(resp.content))
            if max(img.size) < self.MIN_PHOTO_DIM:
                return TOO_SMALL
            # Resize on the longest side to save API tokens
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            # Convert to JPEG bytes
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()

        except Exception as e:
            logger.warning("Failed to download image %s: %s", url[:80], e)
            return None

    @staticmethod
    def _too_small_image_analysis() -> dict[str, Any]:
        """Every photo is a thumbnail: nothing to judge, and nothing to cache."""
        return {
            "photos_too_small": True,
            "photo_type": "uncertain",
            "photo_confidence": 0.0,
            "photo_signals": [],
            "visible_damage": [],
            "condition_matches_description": None,
            "estimated_condition": "cannot_assess",
            "concerns": ["All photos are thumbnail-sized"],
            "confidence": 0.0,
        }

    def _default_image_analysis(self) -> dict[str, Any]:
        """Return default image analysis when unavailable (never reused from cache)."""
        return {
            "image_ai_failed": True,
            "photo_type": "uncertain",
            "photo_confidence": 0.0,
            "photo_signals": [],
            "visible_damage": [],
            "condition_matches_description": None,
            "estimated_condition": "cannot_assess",
            "concerns": ["Image analysis unavailable"],
            "confidence": 0.0,
        }

    # ── Quick Photo Check (Free, No API) ─────────────────────────────

    def quick_photo_check(self, listing: dict[str, Any]) -> dict[str, Any]:
        """Pre-screen for stock photos using metadata (no API calls).

        Checks:
            1. Image count (stock listings often have 1 image)
            2. Catalog-linked listing (has GTIN/ePID)

        Returns:
            Dict with 'suspicious' bool and 'flags' list
        """
        flags: list[str] = []
        image_count = listing.get("image_count", listing.get("additional_images_count", 0) + 1)

        # Single image is suspicious for used items
        if image_count <= 1:
            flags.append("single_image_only")

        # Catalog-linked with few images
        has_epid = bool(listing.get("epid"))
        has_gtin = bool(listing.get("gtin"))
        if (has_epid or has_gtin) and image_count <= 2:
            flags.append("catalog_linked_few_images")

        return {
            "suspicious": len(flags) > 0,
            "flags": flags,
            "image_count": image_count,
        }

    @staticmethod
    def extract_image_id(image_url: str) -> str:
        """Extract the unique image ID from an eBay image URL.

        eBay URLs look like: https://i.ebayimg.com/images/g/UNIQUE_ID/s-l1600.jpg
        """
        match = re.search(r"/images/g/([^/]+)/", image_url)
        return match.group(1) if match else ""
