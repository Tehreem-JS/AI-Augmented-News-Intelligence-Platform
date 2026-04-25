# 🧠 AI-Augmented News Intelligence Platform

An automated pipeline that ingests tech news from 15+ RSS feeds, clusters related articles into deduplicated topic groups using NLP, and generates structured AI-powered daily briefs — all served through a FastAPI REST backend with PostgreSQL persistence.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [How Clustering Works](#how-clustering-works)
- [Scheduler](#scheduler)
- [Resume Metrics](#resume-metrics)

---

## Overview

Most tech news aggregators dump raw articles at you. This platform does the opposite — it reads everything, figures out which articles are about the same story, and produces one clean brief per topic cluster. Think of it as a junior analyst that reads 500 articles a day and hands you a 10-point summary.

The pipeline runs on a schedule:

```
RSS Feeds → Fetch → Deduplicate → Cluster → LLM Brief → Store → Serve via API
```

---

## Features

- **Async RSS ingestion** — fetches from 15 feeds (TechCrunch, MIT Tech Review, OpenAI Blog, Ars Technica, The Verge, Wired, and more) concurrently using `asyncpg` + `feedparser`
- **Two-stage clustering engine** — combines TF-IDF cosine similarity with named-entity overlap and keyword fingerprinting; reduces ~66 raw articles to 10–15 deduplicated topic clusters per cycle
- **LLM-powered brief generation** — per-cluster context injection produces structured daily digests categorized by topic (AI, Hardware, Cybersecurity, Startups, etc.)
- **~70% reduction in LLM API calls** — clustering means the LLM sees one context per topic, not one per article
- **FastAPI REST backend** — 6 endpoints covering dashboard, search, category filtering, and statistics
- **PostgreSQL persistence** — full article + cluster + brief history stored and queryable
- **APScheduler automation** — hourly fetch cycles and daily brief generation, no manual triggers needed

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Scheduler                         │
│         (APScheduler — hourly fetch, daily brief)        │
└───────────────────────┬─────────────────────────────────┘
                        │
            ┌───────────▼───────────┐
            │      NewsFetcher       │
            │  15 RSS feeds via      │
            │  feedparser + asyncpg  │
            └───────────┬───────────┘
                        │  raw articles
            ┌───────────▼───────────┐
            │    ArticleProcessor    │
            │  dedup, clean, store   │
            └───────────┬───────────┘
                        │  clean articles
            ┌───────────▼───────────┐
            │     ClusterEngine      │
            │  Stage 1: fingerprint  │
            │  Stage 2: TF-IDF + NER │
            │  Stage 3: centroid     │
            │          merge         │
            └───────────┬───────────┘
                        │  topic clusters
            ┌───────────▼───────────┐
            │     BriefGenerator     │
            │   LLM API per cluster  │
            └───────────┬───────────┘
                        │  structured briefs
            ┌───────────▼───────────┐
            │       PostgreSQL       │
            │  articles / clusters   │
            │  / briefs / stats      │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │       FastAPI          │
            │   REST endpoints        │
            └───────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Web framework | FastAPI |
| Database | PostgreSQL + asyncpg |
| NLP / ML | scikit-learn (TF-IDF, cosine similarity) |
| LLM integration | Anthropic / OpenAI API |
| Scheduling | APScheduler |
| Feed parsing | feedparser |
| Dependency management | pip + venv |

---

## Project Structure

```
tech-news-analyst/
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── scheduler.py          # APScheduler job configuration
│   ├── fetcher.py            # Async RSS feed fetcher
│   ├── processor.py          # Article cleaning + deduplication
│   ├── cluster_engine.py     # Two-stage clustering engine
│   ├── brief_generator.py    # LLM brief generation
│   ├── database.py           # asyncpg connection pool + queries
│   └── models.py             # Pydantic schemas
├── tests/
│   ├── test_clusters.py      # Cluster engine test harness
│   └── test_api.py           # FastAPI endpoint tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- An Anthropic or OpenAI API key

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/tech-news-analyst.git
cd tech-news-analyst

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install tzdata (required on Windows for timezone support)
pip install tzdata

# 5. Copy and fill in environment variables
cp .env.example .env
```

### Database Setup

```bash
# Create the database
psql -U postgres -c "CREATE DATABASE technews;"

# Run migrations (adjust path as needed)
psql -U postgres -d technews -f backend/schema.sql
```

### Run

```bash
python backend/main.py
```

The API will be live at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

---

## Configuration

Copy `.env.example` to `.env` and fill in:

```env
# Database
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/technews

# LLM
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...

# Scheduler timezone (IANA format)
SCHEDULER_TIMEZONE=America/New_York

# Clustering
MAX_ARTICLES=80
TOP_CLUSTERS_FOR_BRIEF=10
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/dashboard` | Latest clusters + briefs |
| `GET` | `/search?q={query}` | Full-text article search |
| `GET` | `/categories` | All clusters grouped by category |
| `GET` | `/categories/{name}` | Clusters for a specific category |
| `GET` | `/statistics` | Pipeline run stats (articles/day, cluster counts) |

### Example response — `/dashboard`

```json
{
  "generated_at": "2025-04-25T06:00:00Z",
  "cluster_count": 12,
  "clusters": [
    {
      "cluster_id": "cluster_42_8371",
      "main_title": "OpenAI launches GPT-5 with major reasoning improvements",
      "article_count": 5,
      "category": ["AI", "Big Tech"],
      "entities": ["OPENAI", "GPT-5"],
      "brief": "OpenAI released GPT-5 on April 20th...",
      "date_range": {
        "start": "2025-04-20T10:00:00Z",
        "end": "2025-04-20T14:00:00Z"
      }
    }
  ]
}
```

---

## How Clustering Works

The engine runs in three stages to maximize grouping accuracy while minimizing LLM cost.

### Stage 1 — Keyword Fingerprint Pre-clustering

Extracts rare meaningful tokens from each article title and runs Union-Find on pairs sharing 2+ tokens. This catches obvious duplicates (same story across multiple sources) before any ML runs — zero vectorization cost.

```
"OpenAI launches GPT-5 with reasoning improvements"  ┐
"GPT-5 release: what you need to know"               ├─→ same cluster (share: gpt-5, openai)
"Developers react to GPT-5 launch by OpenAI"         ┘
```

### Stage 2 — TF-IDF + Named Entity + Bonus Scoring

For articles that didn't group in Stage 1, TF-IDF cosine similarity is computed on title + summary + content. Two bonus scores are layered on top:

- **Entity overlap bonus (+0.30)** — articles mentioning the same companies/models (NVIDIA, Claude, GPT-4…) get a similarity boost
- **Keyword fingerprint bonus (+0.35)** — shared rare title words further boost the score

`min_df` is dynamically set to 2 when corpus size ≥ 10 documents, forcing the model to ignore terms that only appear once (noise reduction).

### Stage 3 — Centroid Merge

After Stage 2 clustering, the TF-IDF centroid of each cluster is computed. Clusters whose centroids score ≥ 0.25 cosine similarity are merged. This catches "near-miss" clusters that should be one topic.

**Result:** 66 raw articles → 10–15 topic clusters → 10–15 LLM calls (vs 66).

---

## Scheduler

Jobs are configured in `scheduler.py` using APScheduler with IANA timezones (no pytz dependency):

```python
from zoneinfo import ZoneInfo

# Generate daily brief at 6 AM
CronTrigger(hour=6, minute=0, timezone=ZoneInfo("America/New_York"))
```

---

## Resume Metrics

Actual numbers this project can honestly claim:

| Metric | Value |
|---|---|
| RSS feeds integrated | 15 |
| Articles processed per day | 500+ |
| Clustering compression ratio | ~66 articles → 10–15 clusters |
| LLM API call reduction | ~70% vs per-article approach |
| API endpoints | 6 |
| Cluster categories | 10 (AI, Hardware, Cybersecurity, Startups…) |

---

## License

MIT
