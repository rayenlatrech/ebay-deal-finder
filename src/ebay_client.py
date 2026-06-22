"""eBay Browse API client for searching and retrieving listings from eBay Italy."""

import base64
import logging
import os
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

BROWSE_API_BASE = "https://api.ebay.com/buy/browse/v1"
AUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
OAUTH_SCOPE = "https://api.ebay.com/oauth/api_scope"


class EbayClient:
    """Client for the eBay Browse API targeting eBay Italy."""

    def __init__(self) -> None:
        """Initialize with credentials from environment variables."""
        self.client_id = os.environ["EBAY_CLIENT_ID"]
        self.client_secret = os.environ["EBAY_CLIENT_SECRET"]
        self.marketplace = os.environ.get("EBAY_MARKETPLACE", "EBAY_IT")
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0
        self.session = requests.Session()
        logger.info("eBay client initialized for marketplace: %s", self.marketplace)

    # ── Authentication ───────────────────────────────────────────────

    def _get_access_token(self) -> str:
        """Get OAuth 2.0 access token using client credentials flow.

        Caches the token until it expires.
        """
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token

        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {credentials}",
        }
        data = {
            "grant_type": "client_credentials",
            "scope": OAUTH_SCOPE,
        }

        resp = self.session.post(AUTH_URL, headers=headers, data=data, timeout=15)
        resp.raise_for_status()
        token_data = resp.json()

        self._access_token = token_data["access_token"]
        # Expire 5 minutes early to avoid edge cases
        self._token_expiry = time.time() + token_data.get("expires_in", 7200) - 300
        logger.info("eBay OAuth token acquired, expires in %ss", token_data.get("expires_in"))
        return self._access_token

    def _get_headers(self) -> dict[str, str]:
        """Build request headers with auth and marketplace."""
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "X-EBAY-C-MARKETPLACE-ID": self.marketplace,
            "Content-Type": "application/json",
        }

    # ── API Calls with Retry ─────────────────────────────────────────

    def _api_get(self, url: str, params: Optional[dict] = None) -> dict:
        """Make a GET request with exponential backoff on rate limits."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.session.get(
                    url, headers=self._get_headers(), params=params, timeout=30
                )
                if resp.status_code == 429:
                    wait = min(2 ** (attempt + 1), 30)
                    logger.warning("Rate limited, waiting %ds (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning("API error: %s, retrying in %ds", e, wait)
                    time.sleep(wait)
                else:
                    logger.error("API call failed after %d retries: %s", max_retries, e)
                    raise
        return {}

    # ── Search ───────────────────────────────────────────────────────

    def search_listings(
        self,
        query: str,
        category_id: str,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        page_offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search eBay listings with filters.

        Returns dict with 'items' list and 'total' count.
        """
        url = f"{BROWSE_API_BASE}/item_summary/search"

        # Build filter string
        filters = ["buyingOptions:{FIXED_PRICE}", "itemLocationCountry:IT"]
        
        if min_price and max_price:
            filters.append(f"price:[{min_price}..{max_price}],priceCurrency:EUR")
        elif max_price:
            filters.append(f"price:[..{max_price}],priceCurrency:EUR")
        elif min_price:
            filters.append(f"price:[{min_price}..],priceCurrency:EUR")

        params = {
            "q": query,
            "category_ids": category_id,
            "filter": ",".join(filters),
            "sort": "newlyListed",
            "limit": str(limit),
            "offset": str(page_offset),
            "fieldgroups": "EXTENDED",
        }

        data = self._api_get(url, params)

        items = []
        for raw_item in data.get("itemSummaries", []):
            item = self._parse_listing_summary(raw_item)
            if item and item["currency"] == "EUR":
                items.append(item)

        total = data.get("total", 0)
        logger.info("Search '%s': found %d items (total: %d)", query[:30], len(items), total)
        return {"items": items, "total": total}

    def _parse_listing_summary(self, raw: dict) -> Optional[dict]:
        """Parse a raw item summary into a clean dict."""
        try:
            price_info = raw.get("price", {})
            seller_info = raw.get("seller", {})
            image_info = raw.get("image", {})
            buying_options = raw.get("buyingOptions", [])
            
            # Extract shipping cost
            shipping_cost = 0.0
            shipping_options = raw.get("shippingOptions", [])
            if shipping_options:
                cost_info = shipping_options[0].get("shippingCost", {})
                shipping_cost = float(cost_info.get("value", 0.0))

            base_price = float(price_info.get("value", 0))

            return {
                "item_id": raw.get("itemId", ""),
                "title": raw.get("title", ""),
                "price": base_price,
                "shipping_cost": shipping_cost,
                "total_price": base_price + shipping_cost,
                "currency": price_info.get("currency", "EUR"),
                "condition": raw.get("condition", ""),
                "condition_id": raw.get("conditionId", ""),
                "seller_feedback_score": seller_info.get("feedbackScore", 0),
                "seller_feedback_percentage": float(
                    seller_info.get("feedbackPercentage", "0")
                ),
                "seller_username": seller_info.get("username", ""),
                "image_url": image_info.get("imageUrl", ""),
                "item_url": raw.get("itemWebUrl", ""),
                "buying_options": buying_options,
                "negotiable": "BEST_OFFER" in buying_options,
                "additional_images_count": len(raw.get("additionalImages", [])),
                "short_description": raw.get("shortDescription", ""),
                "item_group_type": raw.get("itemGroupType", ""),
                "epid": raw.get("epid", ""),
                "marketplace": self.marketplace,
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to parse listing: %s", e)
            return None

    # ── Item Details ─────────────────────────────────────────────────

    def get_item_details(self, item_id: str) -> Optional[dict]:
        """Get full details for a single listing.

        Returns enriched dict with full description, all images, etc.
        """
        url = f"{BROWSE_API_BASE}/item/{item_id}"

        try:
            data = self._api_get(url)
        except Exception as e:
            logger.error("Failed to get item details for %s: %s", item_id, e)
            return None

        try:
            price_info = data.get("price", {})
            seller_info = data.get("seller", {})
            primary_image = data.get("image", {})
            additional_images = data.get("additionalImages", [])
            buying_options = data.get("buyingOptions", [])

            all_image_urls = [primary_image.get("imageUrl", "")]
            all_image_urls.extend(img.get("imageUrl", "") for img in additional_images)
            all_image_urls = [url for url in all_image_urls if url]

            return {
                "item_id": data.get("itemId", item_id),
                "title": data.get("title", ""),
                "price": float(price_info.get("value", 0)),
                "currency": price_info.get("currency", "EUR"),
                "condition": data.get("condition", ""),
                "condition_id": data.get("conditionId", ""),
                "description": data.get("description", ""),
                "short_description": data.get("shortDescription", ""),
                "seller_feedback_score": seller_info.get("feedbackScore", 0),
                "seller_feedback_percentage": float(
                    seller_info.get("feedbackPercentage", "0")
                ),
                "seller_username": seller_info.get("username", ""),
                "image_url": primary_image.get("imageUrl", ""),
                "all_image_urls": all_image_urls,
                "image_count": len(all_image_urls),
                "item_url": data.get("itemWebUrl", ""),
                "buying_options": buying_options,
                "negotiable": "BEST_OFFER" in buying_options,
                "category_path": data.get("categoryPath", ""),
                "epid": data.get("epid", ""),
                "gtin": data.get("gtin", ""),
                "brand": data.get("brand", ""),
                "item_specifics": {
                    nv["name"]: nv.get("value", "")
                    for nv in data.get("localizedAspects", [])
                },
                "marketplace": self.marketplace,
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to parse item details for %s: %s", item_id, e)
            return None

    def item_is_active(self, item_id: str) -> Optional[bool]:
        """Lightweight liveness check for sold-price tracking.

        Returns True if the listing is still live, False if it has ended
        (eBay returns 404 for ended/removed items), or None on a transient
        error (caller should retry later rather than assume it ended).
        """
        url = f"{BROWSE_API_BASE}/item/{item_id}"
        try:
            resp = self.session.get(url, headers=self._get_headers(), timeout=20)
            if resp.status_code == 404:
                return False
            if resp.status_code == 200:
                return True
            return None
        except requests.exceptions.RequestException as e:
            logger.warning("Liveness check failed for %s: %s", item_id, e)
            return None

    # ── Batch Search ─────────────────────────────────────────────────

    def search_all_categories(
        self, categories: list[dict], max_pages: int = 2, results_per_page: int = 50
    ) -> list[dict]:
        """Search across all configured categories.

        Args:
            categories: List of category dicts from categories.yaml
            max_pages: Max pages to fetch per category
            results_per_page: Results per page (max 200)

        Returns:
            List of all listing dicts found across all categories
        """
        all_listings: list[dict] = []
        seen_ids: set[str] = set()

        for cat in categories:
            cat_name = cat["name"]
            logger.info("Searching category: %s", cat_name)

            for page in range(max_pages):
                offset = page * results_per_page
                try:
                    result = self.search_listings(
                        query=cat["query"],
                        category_id=cat["category_id"],
                        max_price=cat.get("max_price"),
                        min_price=cat.get("min_price"),
                        page_offset=offset,
                        limit=results_per_page,
                    )
                except Exception as e:
                    logger.error("Error searching %s page %d: %s", cat_name, page, e)
                    break

                items = result.get("items", [])
                if not items:
                    break

                for item in items:
                    if item["item_id"] not in seen_ids:
                        item["category"] = cat_name
                        all_listings.append(item)
                        seen_ids.add(item["item_id"])

                # Don't fetch more pages if we got fewer results than requested
                if len(items) < results_per_page:
                    break

                # Small delay between pages to be respectful
                time.sleep(0.5)

            # Small delay between categories
            time.sleep(0.3)

        logger.info("Total unique listings found: %d", len(all_listings))
        return all_listings
