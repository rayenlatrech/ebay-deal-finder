"""Empirically test what the eBay search-query exclusions (-case -box ...) do.

For each configured category it runs two live searches against eBay Italy:

    A) BASE  — positive keywords only          (e.g. "iPhone")
    B) FULL  — base + the "-word" exclusions    (e.g. "iPhone -case -box ...")

and reports, using our own classifier, how many results are real devices vs
accessories/parts, whether the eBay Browse API actually honours the "-" operator,
and — most importantly — which *legitimate devices* the exclusions throw away.

Run from the project root (needs EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in .env):

    python test_query_exclusions.py
"""

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.ebay_client import EbayClient
from src.price_tracker import PriceTracker, is_for_parts

load_dotenv()

LIMIT = 50  # one page per query is plenty to see the pattern


def split_query(query: str) -> tuple[str, list[str]]:
    """Split 'iPhone -case -box' into ('iPhone', ['case', 'box'])."""
    positives, negatives = [], []
    for token in query.split():
        if token.startswith("-") and len(token) > 1:
            negatives.append(token[1:].lower())
        else:
            positives.append(token)
    return " ".join(positives), negatives


def classify(pt: PriceTracker, items: list[dict], cat_name: str) -> dict:
    """Bucket a result set into device / accessory-part / unclassified."""
    out = {"device": [], "part": [], "unclassified": []}
    for it in items:
        if is_for_parts(it):
            out["part"].append(it)
            continue
        key, conf = pt.generate_product_key(
            it.get("title", ""), cat_name, it.get("epid", ""), it.get("condition_id", "")
        )
        if not key:
            out["unclassified"].append(it)
        elif key.endswith("_part") or conf < PriceTracker.MIN_CONFIDENCE:
            out["part"].append(it)
        else:
            out["device"].append(it)
    return out


def search(ebay: EbayClient, cat: dict, query: str) -> list[dict]:
    try:
        res = ebay.search_listings(
            query=query,
            category_id=cat["category_id"],
            max_price=cat.get("max_price"),
            min_price=cat.get("min_price"),
            limit=LIMIT,
        )
        return res.get("items", [])
    except Exception as e:  # noqa: BLE001
        print(f"   ! search failed for '{query[:40]}': {e}")
        return []


def main() -> None:
    config_path = Path(__file__).parent / "config" / "categories.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    categories = raw["categories"]

    ebay = EbayClient()
    pt = PriceTracker(None)  # generate_product_key never touches the DB

    print("\n" + "=" * 72)
    print("eBay query exclusion test — BASE (keywords only) vs FULL (with -word)")
    print("=" * 72)

    for cat in categories:
        name = cat["name"]
        full_query = cat["query"]
        base_query, negatives = split_query(full_query)

        print(f"\n■ {name}")
        print(f"   base : {base_query}")
        print(f"   full : {full_query}")

        base_items = search(ebay, cat, base_query)
        full_items = search(ebay, cat, full_query) if negatives else base_items

        base_c = classify(pt, base_items, name)
        full_c = classify(pt, full_items, name)

        def pct(part, whole):
            return f"{(100 * part / whole):.0f}%" if whole else "—"

        print(f"   BASE  n={len(base_items):<3} "
              f"device={len(base_c['device'])} ({pct(len(base_c['device']), len(base_items))})  "
              f"part/acc={len(base_c['part'])}  unclassified={len(base_c['unclassified'])}")
        print(f"   FULL  n={len(full_items):<3} "
              f"device={len(full_c['device'])} ({pct(len(full_c['device']), len(full_items))})  "
              f"part/acc={len(full_c['part'])}  unclassified={len(full_c['unclassified'])}")

        # ── Operator check: does eBay actually honour the "-" exclusions? ──
        if negatives:
            leaked = [
                it for it in full_items
                if any(neg in it.get("title", "").lower() for neg in negatives)
            ]
            if leaked:
                print(f"   ⚠️  '-' operator may NOT be honoured: {len(leaked)} FULL results "
                      f"still contain an excluded word. e.g. {leaked[0]['title'][:60]!r}")
            else:
                print("   ✓ '-' operator honoured (no excluded words in FULL results)")

        # ── Collateral damage: real devices the exclusions removed ──
        full_ids = {it["item_id"] for it in full_items}
        dropped_devices = [it for it in base_c["device"] if it["item_id"] not in full_ids]
        if dropped_devices:
            print(f"   ✂️  exclusions dropped {len(dropped_devices)} listing(s) we'd classify "
                  f"as real devices. examples:")
            for it in dropped_devices[:4]:
                print(f"        - €{it.get('price', 0):.0f}  {it['title'][:62]}")

    print("\n" + "=" * 72)
    print("Read: high BASE 'part/acc' % => exclusions help density. Many 'dropped")
    print("devices' => exclusions are too aggressive and losing real phones.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"Missing environment variable: {e}. Check your .env (EBAY_CLIENT_ID/SECRET).")
        sys.exit(1)
