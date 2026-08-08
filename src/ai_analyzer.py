import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger(__name__)


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
  "hidden_defects": ["issues mentioned in description but not in title/condition"],
  "seller_trust": "high|medium|low",
  "seller_reasoning": "brief explanation of trust rating",
  "negotiable": {negotiable},
  "is_complete_item": true,  // MUST BE FALSE if it is just a box, single earbud, cable, or component part!
  "is_good_deal": true,      // MUST BE FALSE if is_complete_item is false or if broken/parts_only
  "reasoning": "brief explanation in English"
}}"""

    IMAGE_ANALYSIS_PROMPT = """You are an expert at analyzing eBay product photos for used electronics.

FIRST, determine if this is a real photo of the specific item or a stock/catalog image.
Stock photo signs: pure white background, perfect studio lighting, commercially polished, no visible wear, looks like a manufacturer's product page.
Real photo signs: home/desk background, ambient lighting, visible wear marks, phone camera quality, personal items in background, reflections.

THEN, if it's a real photo, assess the item's physical condition.

The seller described this item as: "{condition}" condition.
The listing says: "{description_excerpt}"

Return ONLY valid JSON (no markdown, no explanation):
{{
  "photo_type": "real|stock|uncertain",
  "photo_confidence": 0.85,
  "photo_signals": ["list specific visual cues"],
  "visible_damage": ["list any scratches, dents, cracks, discoloration"],
  "condition_matches_description": true,
  "estimated_condition": "mint|good|fair|poor|cannot_assess",
  "concerns": ["any red flags from the image"],
  "confidence": 0.85
}}"""



    def __init__(self, db_instance) -> None:
        """Initialize AI clients from environment variables."""
        self.db = db_instance
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
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

        # Initialize Groq client
        groq_key = os.environ.get("GROQ_API_KEY")
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=groq_key) if groq_key else None
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
        self, prompt: str, image_data: Optional[bytes] = None
    ) -> Optional[str]:
        """Call Gemini API with key rotation and retry.

        Args:
            prompt: Text prompt
            image_data: Optional image bytes for vision analysis

        Returns:
            Response text or None if all keys exhausted.
        """
        if not self.gemini_keys:
            return None

        from google import genai
        from google.genai import types

        for attempt in range(len(self.gemini_keys)):
            api_key = self._get_next_gemini_key()
            try:
                client = genai.Client(api_key=api_key)

                contents: list[Any] = []

                if image_data:
                    # Build multimodal content
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_bytes(
                                    data=image_data,
                                    mime_type="image/jpeg",
                                ),
                                types.Part.from_text(text=prompt),
                            ],
                        )
                    )
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
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    logger.warning(
                        "Gemini rate limited (key index %d), tracking exhaustion and trying next key",
                        (self._gemini_key_index - 1) % len(self.gemini_keys),
                    )
                    
                    # Store cooldown in DB (Assume 24 hour reset for safety, or next day)
                    # We remove the exhausted key from the active list for this session
                    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                    reset_time = datetime.now(timezone.utc) + timedelta(hours=24)
                    self.db.set_key_exhausted(key_hash, reset_time)
                    
                    time.sleep(1)
                    continue
                else:
                    logger.error("Gemini API error: %s", e)
                    return None

        logger.warning("All Gemini keys exhausted")
        return None

    # ── Groq Fallback ────────────────────────────────────────────────

    def _call_groq(self, prompt: str) -> Optional[str]:
        """Call Groq API for text-only analysis (fallback)."""
        if not self.groq_client:
            logger.error("Groq client not available")
            return None

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Groq API error: %s", e)
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

    def _call_ai(self, prompt: str, image_data: Optional[bytes] = None) -> Optional[str]:
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
        """Return default analysis when AI is unavailable."""
        return {
            "condition_assessment": "unknown",
            "red_flags": [],
            "hidden_defects": [],
            "seller_trust": "unknown",
            "seller_reasoning": "AI analysis unavailable",
            "negotiable": False,
            "is_good_deal": False,
            "reasoning": "Could not perform AI analysis",
        }

    # ── Image Analysis ───────────────────────────────────────────────

    def analyze_image(
        self,
        image_url: str,
        condition: str = "",
        description_excerpt: str = "",
    ) -> dict[str, Any]:
        """Analyze a product image for stock photo detection and condition assessment.

        Args:
            image_url: URL of the product image
            condition: Seller-stated condition
            description_excerpt: Brief excerpt of listing description

        Returns:
            Dict with photo_type, visible_damage, condition assessment, etc.
        """
        # Download image
        image_data = self._download_image(image_url)
        if not image_data:
            return self._default_image_analysis()

        prompt = self.IMAGE_ANALYSIS_PROMPT.format(
            condition=condition or "not specified",
            description_excerpt=description_excerpt[:500] or "no description",
        )

        response = self._call_ai(prompt, image_data)
        if not response:
            return self._default_image_analysis()

        parsed = _parse_ai_json(response)
        if not parsed:
            return self._default_image_analysis()

        return parsed

    def _download_image(self, url: str) -> Optional[bytes]:
        """Download an image and return as bytes."""
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            # Validate it's actually an image and resize if too large
            img = Image.open(BytesIO(resp.content))
            # Resize to max 1024px on longest side to save API tokens
            max_dim = 1024
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

    def _default_image_analysis(self) -> dict[str, Any]:
        """Return default image analysis when unavailable."""
        return {
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
