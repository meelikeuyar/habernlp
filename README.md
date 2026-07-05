# 📰 HaberNLP — Turkish News Intelligence Platform

Real-time Turkish news aggregation pipeline with transformer-based sentiment analysis, automatic topic modeling, and an interactive analytics dashboard.

## What it does

HaberNLP scrapes 8+ major Turkish news sources every hour, stores articles in a SQLite database, runs NLP analysis (sentiment via fine-tuned BERT, topics via BERTopic), and serves insights through a FastAPI-powered dashboard.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Scraper      │────▶│  SQLite DB   │────▶│  NLP Engine  │────▶│  FastAPI +   │
│  (8 sources)  │     │  (articles)  │     │  BERT / Topic│     │  Dashboard   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       ▲                                                              │
       │              APScheduler (hourly)                            │
       └──────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Scraping | BeautifulSoup4, requests, concurrent.futures |
| Database | SQLite + SQLAlchemy ORM |
| NLP — Sentiment | `dbmdz/bert-base-turkish-cased` (fine-tuned) |
| NLP — Topics | BERTopic + sentence-transformers |
| NLP — Keywords | TF-IDF with Turkish stop words |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS dashboard with Chart.js |
| Scheduling | APScheduler |
| CI/CD | GitHub Actions |
| Deployment | Docker + docker-compose |

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/habernlp.git
cd habernlp

# Option 1: Docker (recommended)
docker-compose up --build

# Option 2: Local
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Then open `http://localhost:8000` for the dashboard.

## Project Structure

```
habernlp/
├── main.py                  # Entry point
├── config/
│   └── settings.py          # All configuration
├── src/
│   ├── scraper/             # Multi-source news scraper
│   ├── database/            # SQLAlchemy models + operations
│   ├── nlp/
│   │   ├── sentiment.py     # BERT-based sentiment analysis
│   │   ├── topics.py        # BERTopic topic modeling
│   │   └── keywords.py      # TF-IDF keyword extraction
│   ├── api/                 # FastAPI routes
│   └── scheduler/           # APScheduler jobs
├── frontend/templates/      # Dashboard HTML
├── tests/                   # pytest test suite
├── notebooks/               # Training & evaluation scripts
├── data/                    # SQLite DB (auto-created)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml # CI pipeline
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/haberler` | List articles (with pagination & filters) |
| GET | `/api/analiz` | Full analysis (sentiment + topics + keywords) |
| GET | `/api/trends` | Daily trend data |
| POST | `/api/guncelle` | Trigger manual scrape |
| GET | `/api/health` | Health check |

## NLP Pipeline

**Sentiment Analysis**: Fine-tuned `dbmdz/bert-base-turkish-cased` on 5,000+ labeled Turkish news headlines. The model classifies headlines as positive, negative, or neutral with an F1 score of ~0.85. Falls back to rule-based analysis if the model is unavailable.

**Topic Modeling**: BERTopic with Turkish sentence-transformers clusters articles into automatically discovered topics. The system tracks how topics evolve over time.

**Keyword Extraction**: TF-IDF with custom Turkish stop words identifies the most significant terms per day and per source.

## Training the Sentiment Model

```bash
python notebooks/train_sentiment.py
```

This script downloads the base model, fine-tunes it on the labeled dataset, evaluates it, and saves the result to `models/sentiment/`.

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## License

MIT
