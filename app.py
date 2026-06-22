"""Streamlit control panel for the eBay AI Deal Finder.

Edits the Supabase ``watchlist`` table that the scanner reads at the start of
every run, so you can pick which devices to focus on, tune how aggressively each
category surfaces deals, and review recent finds — all without touching code.

Run from the project root:

    pip install -r requirements-dev.txt
    streamlit run app.py
"""

import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.db import Database
from src.sold_tracker import SoldTracker

load_dotenv()

st.set_page_config(page_title="eBay Deal Finder", page_icon="🔍", layout="wide")

# Columns shown in the main watchlist editor (price_scores is edited separately).
EDITOR_COLUMNS = [
    "enabled", "name", "query", "category_id",
    "min_price", "max_price", "priority", "min_discount",
]


# ── Helpers ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_db() -> Database:
    """Create the Supabase client once per session."""
    return Database()


def _clean_num(value, cast=float):
    """Turn a possibly-NaN/blank cell into a number or None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def load_yaml_defaults() -> list[dict]:
    """Read categories.yaml (used to seed an empty watchlist)."""
    path = Path(__file__).parent / "config" / "categories.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    return [{**defaults, **cat, "enabled": True} for cat in raw["categories"]]


# ── Connect ───────────────────────────────────────────────────────────────

try:
    db = get_db()
except Exception as exc:  # noqa: BLE001 - surface any config/connection issue
    st.error(
        "Could not connect to Supabase. Make sure `SUPABASE_URL` and "
        f"`SUPABASE_KEY` are set in your `.env`.\n\nDetails: {exc}"
    )
    st.stop()

st.title("🔍 eBay AI Deal Finder — Control Panel")
st.caption("Changes are saved to Supabase and picked up on the scanner's next run.")

tab_watchlist, tab_curves, tab_deals, tab_resale = st.tabs(
    ["⚙️ Watchlist", "📈 Discount curves", "🔥 Recent deals", "📦 Resale data"]
)


# ── Tab 1: Watchlist editor ───────────────────────────────────────────────

with tab_watchlist:
    rows = db.get_watchlist(only_enabled=False)

    if not rows:
        st.info("Your watchlist is empty. Seed it from `config/categories.yaml` to get started.")
        if st.button("🌱 Seed from categories.yaml", type="primary"):
            for item in load_yaml_defaults():
                payload = {k: item.get(k) for k in EDITOR_COLUMNS if k in item}
                if item.get("price_scores") is not None:
                    payload["price_scores"] = item["price_scores"]
                db.upsert_watchlist_item(payload)
            st.success("Seeded the watchlist. Reloading…")
            st.rerun()
        st.stop()

    st.markdown(
        "**Focus knobs** — higher `priority` gets more of the limited AI budget "
        "and ranks higher in alerts; lower `min_discount` surfaces smaller "
        "discounts (good for high-value phones). Per-category discount curves "
        "live in the next tab."
    )

    df = pd.DataFrame(rows)
    # Keep only editor columns, in order; fill any missing.
    for col in EDITOR_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[EDITOR_COLUMNS]
    original_names = set(df["name"].dropna())

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="watchlist_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("On", help="Include in scans"),
            "name": st.column_config.TextColumn("Name", help="Unique label, e.g. iPhones"),
            "query": st.column_config.TextColumn("eBay query", width="large"),
            "category_id": st.column_config.TextColumn("Cat ID"),
            "min_price": st.column_config.NumberColumn("Min €", min_value=0, step=10),
            "max_price": st.column_config.NumberColumn("Max €", min_value=0, step=10),
            "priority": st.column_config.NumberColumn(
                "Priority", min_value=0.0, step=0.1,
                help="Focus weight. 3.0 = primary focus, 0.5 = de-prioritized.",
            ),
            "min_discount": st.column_config.NumberColumn(
                "Min disc %", min_value=0, max_value=100, step=1,
                help="Minimum discount vs market baseline to consider.",
            ),
        },
    )

    if st.button("💾 Save watchlist", type="primary"):
        edited_names = set()
        errors = []
        for _, row in edited.iterrows():
            name = (str(row["name"]).strip() if row["name"] is not None else "")
            if not name or name.lower() == "nan":
                continue
            if not str(row.get("query") or "").strip() or not str(row.get("category_id") or "").strip():
                errors.append(f"'{name}' is missing a query or category id — skipped.")
                continue
            edited_names.add(name)
            db.upsert_watchlist_item({
                "name": name,
                "query": str(row["query"]),
                "category_id": str(row["category_id"]),
                "min_price": _clean_num(row["min_price"]),
                "max_price": _clean_num(row["max_price"]),
                "priority": _clean_num(row["priority"]) or 1.0,
                "min_discount": _clean_num(row["min_discount"]) or 15,
                "enabled": bool(row["enabled"]),
            })
        for gone in original_names - edited_names:
            db.delete_watchlist_item(gone)

        for msg in errors:
            st.warning(msg)
        st.success(f"Saved {len(edited_names)} categories.")
        st.rerun()


# ── Tab 2: Per-category discount curves ───────────────────────────────────

with tab_curves:
    st.markdown(
        "A discount **curve** maps *discount %* → *price score (0-100)*. Steeper "
        "early tiers make modest discounts count more — that's how a 20%-off "
        "phone can score like a 40%-off accessory. Leave a category without a "
        "curve to use the global default from `scoring.yaml`."
    )

    rows = db.get_watchlist(only_enabled=False)
    if not rows:
        st.info("Add categories in the Watchlist tab first.")
    else:
        by_name = {r["name"]: r for r in rows}
        choice = st.selectbox("Category", sorted(by_name))
        current = by_name[choice].get("price_scores")

        if current:
            curve_df = pd.DataFrame(current)
        else:
            st.caption("No custom curve yet — this category uses the global default.")
            curve_df = pd.DataFrame(
                [
                    {"max_discount": 8, "score": 0},
                    {"max_discount": 15, "score": 50},
                    {"max_discount": 25, "score": 80},
                    {"max_discount": 40, "score": 95},
                    {"max_discount": 100, "score": 100},
                ]
            )

        edited_curve = st.data_editor(
            curve_df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"curve_{choice}",
            column_config={
                "max_discount": st.column_config.NumberColumn(
                    "Up to discount %", min_value=0, max_value=100, step=1
                ),
                "score": st.column_config.NumberColumn(
                    "Price score", min_value=0, max_value=100, step=5
                ),
            },
        )

        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Save curve", type="primary"):
                curve = [
                    {"max_discount": int(r["max_discount"]), "score": int(r["score"])}
                    for _, r in edited_curve.iterrows()
                    if _clean_num(r["max_discount"]) is not None
                ]
                curve.sort(key=lambda t: t["max_discount"])
                db.upsert_watchlist_item({"name": choice, "price_scores": curve})
                st.success(f"Saved curve for {choice}.")
                st.rerun()
        with col_clear:
            if st.button("🧹 Use global default (clear curve)"):
                db.upsert_watchlist_item({"name": choice, "price_scores": None})
                st.success(f"{choice} now uses the global curve.")
                st.rerun()


# ── Tab 3: Recent deals dashboard ─────────────────────────────────────────

with tab_deals:
    st.markdown("Most recent analyses (newest first).")
    try:
        resp = (
            db.client.table("analyses")
            .select(
                "deal_score, price_score, condition_score, seller_score, "
                "alerted, analyzed_at, listing_id, "
                "listings(title, price, category, listing_url)"
            )
            .order("analyzed_at", desc=True)
            .limit(50)
            .execute()
        )
        records = []
        for a in resp.data or []:
            listing = a.get("listings") or {}
            records.append({
                "score": a.get("deal_score"),
                "alerted": a.get("alerted"),
                "title": listing.get("title", ""),
                "category": listing.get("category", ""),
                "price €": listing.get("price"),
                "url": listing.get("listing_url", ""),
                "analyzed": a.get("analyzed_at"),
            })

        if not records:
            st.info("No analyses yet — run a scan and check back.")
        else:
            deals_df = pd.DataFrame(records).sort_values("score", ascending=False)
            st.dataframe(
                deals_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn("Link", display_text="open"),
                    "alerted": st.column_config.CheckboxColumn("Alerted"),
                },
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load recent deals: {exc}")


# ── Tab 4: Resale dataset (watch it mature) ───────────────────────────────

def _count(table: str, **filters) -> int:
    """Exact row count with optional equality filters (0 on any error)."""
    try:
        q = db.client.table(table).select("*", count="exact")
        for key, value in filters.items():
            q = q.eq(key, value)
        return q.limit(1).execute().count or 0
    except Exception:  # noqa: BLE001 - table may not exist yet
        return 0


with tab_resale:
    st.markdown(
        "The proprietary sold-price dataset, building over time. `sold_proxy` is "
        "**inferred** (a listing that disappeared via 404 within the sold window) "
        "— not eBay-confirmed — so treat early numbers as directional."
    )

    tracked = _count("tracked_listings")
    comps_total = _count("sold_comps")
    comps_sold = _count("sold_comps", sold_proxy=True)
    buckets = _count("resale_stats")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked (live)", f"{tracked:,}")
    c2.metric("Realized comps", f"{comps_total:,}")
    c3.metric("Proxy-sold", f"{comps_sold:,}")
    c4.metric("Resale buckets", f"{buckets:,}")

    if tracked == 0 and comps_total == 0:
        st.info(
            "No tracking data yet. Make sure you ran the `tracked_listings` / "
            "`sold_comps` / `resale_stats` SQL, then let a few scans run — "
            "realized prices only appear once listings end (days to weeks)."
        )

    # ── Per-bucket resale stats ──
    st.subheader("Resale value by product")
    rebuild = st.button("🔄 Rebuild stats now", help="Recompute from current comps")
    if rebuild:
        try:
            written = SoldTracker(db, None).rebuild_resale_stats(force=True)
            st.success(f"Rebuilt {written} bucket(s).")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Rebuild failed: {exc}")

    try:
        resp = (
            db.client.table("resale_stats")
            .select("*")
            .order("sample_size", desc=True)
            .limit(500)
            .execute()
        )
        stats_rows = resp.data or []
        if not stats_rows:
            st.caption("No buckets meet the minimum sample size yet — keep scanning.")
        else:
            stats_df = pd.DataFrame(stats_rows)[[
                "product_key", "condition_band", "median_price",
                "p25_price", "p75_price", "sample_size",
                "median_days_to_sell", "updated_at",
            ]]
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "product_key": st.column_config.TextColumn("Product"),
                    "condition_band": st.column_config.TextColumn("Cond."),
                    "median_price": st.column_config.NumberColumn("Median €", format="%.0f"),
                    "p25_price": st.column_config.NumberColumn("P25 €", format="%.0f"),
                    "p75_price": st.column_config.NumberColumn("P75 €", format="%.0f"),
                    "sample_size": st.column_config.NumberColumn("n"),
                    "median_days_to_sell": st.column_config.NumberColumn("Days to sell", format="%.0f"),
                },
            )

    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load resale stats: {exc}")

    # ── Recent realized outcomes (raw comps) ──
    st.subheader("Recent realized outcomes")
    try:
        resp = (
            db.client.table("sold_comps")
            .select("product_key, condition_band, final_price, days_active, sold_proxy, ended_at")
            .order("ended_at", desc=True)
            .limit(100)
            .execute()
        )
        comp_rows = resp.data or []
        if not comp_rows:
            st.caption("No listings have ended yet.")
        else:
            comps_df = pd.DataFrame(comp_rows)
            st.dataframe(
                comps_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "product_key": st.column_config.TextColumn("Product"),
                    "condition_band": st.column_config.TextColumn("Cond."),
                    "final_price": st.column_config.NumberColumn("Last price €", format="%.0f"),
                    "days_active": st.column_config.NumberColumn("Days active"),
                    "sold_proxy": st.column_config.CheckboxColumn("Sold?"),
                    "ended_at": st.column_config.TextColumn("Ended"),
                },
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load realized outcomes: {exc}")
