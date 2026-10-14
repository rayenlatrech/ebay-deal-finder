# 🔍 eBay AI Deal Finder

An AI-powered bot that scans **eBay Italy** every 4 hours for underpriced used electronics, builds a robust market baseline for each product, and only spends AI tokens on the listings that look genuinely cheap — then alerts you on Telegram. **Total running cost: $0.**

## ✨ Features

- 🛒 **eBay Italy scanning** — monitors iPhones, Galaxy, PS5, Xbox, MacBooks, iPads, AirPods, Switch
- 🧱 **Robust market baseline** — structured catalog (ePID) + regex bucketing, for-parts filtering, IQR outlier rejection, and cross-run history blending, so one weird listing can't corrupt a product's average
- 🧠 **AI-powered analysis** — Groq (Llama 3.3) reads the description (Italian + English) and flags scams, broken/parts items, and untrustworthy sellers
- 📸 **Gemini Vision** — inspects the photo for stock-vs-real detection and visible damage on the surviving candidates
- 📊 **Composite deal scoring** — weighted score from price, condition, seller, images, and listing quality
- 🤝 **Make Offer detection** — flags negotiable listings with Best Offer available
- 📱 **Telegram alerts** — instant hot-deal notifications + batched good-deal summaries (HTML-formatted, so messy titles never break a message)
- 💬 **Deal-check bot** — message the bot a device, condition and price (or an eBay link) and get a verdict against the market, the most to pay, and an AI review
- ⏰ **Automated via GitHub Actions** — started every 4 hours on the dot by an external cron; tests run on every push
- 🔑 **Multi API key rotation** — supports multiple Gemini keys with DB-backed cooldown tracking
- 🆓 **100% free** — all services used within their free tiers

## 🏗️ Architecture

```
GitHub Actions (cron every 4h)
  → Search eBay Italy via Browse API (newest listings per category)
  → Build a robust market baseline per product bucket   ← see "How the baseline stays clean"
  → Filter: keep only listings ≥15% below their bucket baseline
  → Groq reads the description (scam / parts / seller checks)
  → Gemini Vision verifies the photo (stock vs real, visible damage)
  → Composite score (0-100) from 5 signals
  → Telegram alert for hot deals + batched summary for good deals
  → Store listings, analyses, and price history in Supabase
```

The pipeline is a **funnel**: hundreds of listings come in, cheap free filters
(classification → baseline → discount threshold) cut it down to a few dozen, and
only those reach the paid/rate-limited AI stages.

## 🧱 How the baseline stays clean

The market average is computed **before** any AI runs (to save tokens), so the
integrity of the whole system depends on grouping listings correctly and not
letting junk skew the average. Rather than rely on regex being perfect, the
baseline is defended in layers (`src/price_tracker.py`):

1. **Structured-first classification** — listings are bucketed by a normalized
   regex key (`iphone_13_pro_max_256gb`, `ps5_console`, …). Variant patterns are
   evaluated by precedence, so a `PlayStation Portal` lands in `ps5_portal`, not
   the `ps5_console` bucket. Titles the regex can't parse fall back to eBay's
   catalog **ePID**.
2. **Context-aware accessory handling** — instead of banning keywords (which
   would kill *"MacBook Pro M1 with charger"*), the classifier looks at position
   and context: a charger introduced by *"with / con / incluso"* is a bundled
   extra (still a device), while *"caricatore per MacBook"* or a leading
   *"Cover iPhone"* is demoted to a low-trust `_part` bucket.
3. **For-parts exclusion** — anything eBay marks `For parts or not working`
   (conditionId 7000) is dropped before it can drag the average down.
4. **ePID purity guard** — within a bucket, if the catalog-identified listings
   overwhelmingly point at one product, a listing carrying a *different* ePID is
   treated as a misclassification and purged.
5. **IQR outlier rejection** — remaining prices outside the 1.5×IQR fence are
   trimmed, so a stray part or bundle can't move the median.
6. **History blend + minimum support** — the baseline is the median across
   recent stored medians, and a bucket needs enough samples (or prior history)
   before it's trusted at all.

The classification and statistical layers are covered by `tests/` and run in CI.

## 🚀 Quick Setup

### 1. Get Your API Keys

| Service | Where to Sign Up | What You Get |
|---------|-----------------|-------------|
| **eBay** | [developer.ebay.com](https://developer.ebay.com) | Client ID + Secret |
| **Groq** | [console.groq.com](https://console.groq.com) | API Key (free tier) — primary text engine |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com/apikey) | API Key (free tier) — image analysis |
| **Telegram** | @BotFather on Telegram | Bot Token + Chat ID |
| **Supabase** | [supabase.com](https://supabase.com) | URL + Anon Key |

> ⚠️ **Do NOT enable billing** on your Google Cloud project — billing deactivates the free tier.

### 2. Set Up Supabase Database

Create a new project at [supabase.com](https://supabase.com), then run this SQL in the SQL Editor:

```sql
-- Scanned listings
CREATE TABLE listings (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    price           DECIMAL(10,2),
    currency        TEXT DEFAULT 'EUR',
    condition       TEXT,
    category        TEXT,
    seller_score    INTEGER,
    seller_pct      DECIMAL(5,2),
    negotiable      BOOLEAN DEFAULT FALSE,
    image_url       TEXT,
    listing_url     TEXT,
    marketplace     TEXT DEFAULT 'EBAY_IT',
    first_seen_at   TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ DEFAULT NOW()
);

-- AI analysis results
CREATE TABLE analyses (
    id              SERIAL PRIMARY KEY,
    listing_id      TEXT REFERENCES listings(id),
    deal_score      INTEGER,
    price_score     INTEGER,
    condition_score INTEGER,
    seller_score    INTEGER,
    image_score     INTEGER,
    ai_assessment   JSONB,
    alerted         BOOLEAN DEFAULT FALSE,
    analyzed_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Market price history (one robust median per bucket per run)
CREATE TABLE price_history (
    id              SERIAL PRIMARY KEY,
    product_key     TEXT NOT NULL,
    avg_price       DECIMAL(10,2),
    sample_size     INTEGER,
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Image deduplication (stock photo detection)
CREATE TABLE image_sellers (
    image_id        TEXT NOT NULL,
    seller_id       TEXT NOT NULL,
    first_seen_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (image_id, seller_id)
);

-- Gemini API-key cooldown tracking (rate-limit rotation)
CREATE TABLE api_quotas (
    api_key_hash    TEXT PRIMARY KEY,
    exhausted_until TIMESTAMPTZ
);

-- Indexes for performance
CREATE INDEX idx_price_history_key ON price_history(product_key);
CREATE INDEX idx_analyses_listing ON analyses(listing_id);
CREATE INDEX idx_image_sellers_image ON image_sellers(image_id);
```

### 3. Create Telegram Bot

1. Open Telegram, search for **@BotFather**
2. Send `/newbot`, choose a name and username (must end in `_bot`)
3. Save the **bot token** BotFather gives you
4. Send any message to your new bot (e.g., "Hello")
5. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
6. Find your `chat_id` in the JSON response

### 4. Fork & Configure GitHub Repo

1. **Fork** this repository
2. Go to your fork's **Settings → Secrets and variables → Actions**
3. Add these secrets:

| Secret Name | Value |
|------------|-------|
| `EBAY_CLIENT_ID` | Your eBay Client ID |
| `EBAY_CLIENT_SECRET` | Your eBay Client Secret |
| `GROQ_API_KEY` | Your Groq API key |
| `GEMINI_API_KEY_1` | Gemini API key |
| `GEMINI_API_KEY_2` | Additional Gemini API key (optional) |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase anon key |

4. Go to **Actions** tab → enable workflows
5. Click **"eBay Deal Scanner"** → **"Run workflow"** to test

### 5. Local Development (Optional)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ebay-deal-finder.git
cd ebay-deal-finder

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies (use requirements-dev.txt to also get pytest)
pip install -r requirements-dev.txt

# Copy .env.example and fill in your keys
copy .env.example .env
# Edit .env with your real API keys

# Run the tests
pytest

# Run the scanner
python -m src.main
```

> 🔒 **Never commit secrets.** Keep all keys in `.env` (which is git-ignored).
> Don't keep a second plaintext copy of keys anywhere in the repo.

### 6. On-time scans (cron-job.org)

GitHub's own cron is best-effort: scans start up to ~1.5 h late and some slots
are dropped. A free external cron starts them on time instead; GitHub's
schedule stays as a backup that skips itself when a scan already ran.

1. **GitHub token** — Settings → Developer settings → Personal access tokens →
   *Fine-grained tokens* → Generate. Repository access: *Only select
   repositories* → this repo. Permissions: **Actions → Read and write**
   (nothing else). Set a reminder for its expiry date.
2. **cron-job.org** (free account) → *Create cronjob*:
   - URL: `https://api.github.com/repos/<you>/<repo>/actions/workflows/scan.yml/dispatches`
   - Schedule: every day, hours `0,4,8,12,16,20`, minute `0`, time zone **UTC**
   - Advanced → Request method **POST**, request body `{"ref":"main"}`, headers:
     `Accept: application/vnd.github+json`, `Authorization: Bearer <token>`,
     `X-GitHub-Api-Version: 2022-11-28`
3. *Test run* — expect HTTP **204**, and a new `workflow_dispatch` run in the
   Actions tab.

### 7. Deal-check bot (Vercel, optional)

Message the bot `iPhone 15 Pro 256GB usato 550€` (or paste an eBay link) and it
replies with the market price, a verdict, the most to pay and an AI review. It
runs as one serverless function, `api/telegram.py`, on Vercel's free plan.

1. vercel.com → sign in with GitHub → *Add New → Project* → import this repo,
   Framework Preset **Other**.
2. Environment variables: the same as `.env` (`EBAY_*`, `GROQ_API_KEY`,
   `GEMINI_API_KEY_*`, `SUPABASE_URL`, `SUPABASE_KEY`, `TELEGRAM_BOT_TOKEN`,
   `TELEGRAM_CHAT_ID`) plus **`TELEGRAM_WEBHOOK_SECRET`** — any long random
   string; put the same value in your local `.env`.
3. Deploy, then open `https://<project>.vercel.app/api/telegram` — it should say
   *deal-check bot is up*.
4. Point Telegram at it: `python scripts/set_telegram_webhook.py https://<project>.vercel.app/api/telegram`

Only your chat (`TELEGRAM_CHAT_ID`) is answered, and only requests carrying the
webhook secret are accepted.

## 📁 Project Structure

```
├── .github/workflows/scan.yml   # CI: tests on push, scan every 4h (external cron + backup)
├── api/telegram.py              # Vercel webhook: the deal-check bot
├── scripts/                     # One-off setup helpers (Telegram webhook)
├── src/
│   ├── main.py                  # Pipeline orchestrator
│   ├── ebay_client.py           # eBay Browse API client
│   ├── ai_analyzer.py           # Groq (text) + Gemini (vision) analysis
│   ├── deal_scorer.py           # Composite scoring engine
│   ├── price_tracker.py         # Product classification
│   ├── baseline.py              # 14-day market window + retail safety net
│   ├── deal_check.py            # Answers the bot's "is this a good deal?" messages
│   ├── telegram_notifier.py     # Telegram alert sender (HTML)
│   └── db.py                    # Supabase database wrapper
├── tests/                       # Classifier + baseline robustness tests
├── config/
│   ├── categories.yaml          # Product categories to monitor
│   ├── scoring.yaml             # Score weights & thresholds
│   └── settings.yaml            # General settings
├── .env.example                 # Environment variable template
├── requirements.txt             # Runtime dependencies
└── requirements-dev.txt         # + pytest for local/CI testing
```

## ⚙️ Configuration

### Product Categories (`config/categories.yaml`)
Add or remove products to monitor. Each entry has a search query, eBay category ID, and min/max price.

### Scoring (`config/scoring.yaml`)
Adjust deal scoring weights, alert thresholds, and stock photo penalties.

### Settings (`config/settings.yaml`)
Configure marketplace, AI models, scanning limits, and Telegram preferences.

### Baseline tuning (`src/price_tracker.py`)
`MIN_CONFIDENCE`, `MIN_SAMPLES`, and `HISTORY_WINDOW` control how strict the
classifier is and how much support a bucket needs before its baseline is trusted.

## 💰 Cost: $0/month

| Service | Free Tier | Your Usage | Role |
|---------|-----------|-----------|------|
| eBay Browse API | 5,000/day | ~280/day | Search + item details |
| Groq | 14,400/day | ~200/day | **Primary** text/description analysis |
| Google Gemini (×2 keys) | 3,000/day | ~80/day | Image/condition verification |
| GitHub Actions | Unlimited (public repo) | ~1,800 min/mo | Scans + tests |
| Supabase | 500 MB | < 50 MB | Listings, analyses, price history |
| Telegram | Unlimited | ~20 msgs/day | Alerts |
| Vercel (Hobby) | 300 s/function | a few calls/day | Deal-check bot webhook |
| cron-job.org | Free | 6 calls/day | On-time scan triggers |

## 📄 License

MIT License — free to use, modify, and distribute.
