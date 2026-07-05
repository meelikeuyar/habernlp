# 📰 HaberNLP — Turkish News Intelligence Platform

Real-time Turkish news aggregation from 11 sources with BERT-based sentiment analysis, keyword extraction, and an editorial-style analytics dashboard.

## Features

* **Multi-source scraping** — RSS + HTML hybrid scraper pulling from 11 Turkish news outlets (T24, BBC Türkçe, NTV, Hürriyet, Sözcü, Habertürk, TRT Haber, Euronews TR, DW Türkçe, Independent TR)
* **BERT sentiment analysis** — Fine-tuned `dbmdz/bert-base-turkish-cased` model (F1: 0.97 on test set)
* **Keyword extraction** — TF-IDF and frequency-based analysis on headlines
* **Analytics dashboard** — Streamlit app with NYT/FT-inspired editorial design: sentiment trends, source distribution, word clouds, live ticker
* **FastAPI backend** — REST API with filtering, trend analysis, and manual refresh endpoints
* **MLflow tracking** — Experiment tracking for model training runs
* **Scheduled scraping** — APScheduler-based periodic news fetching

## Tech Stack

|Layer|Technology|
|-|-|
|NLP|Transformers, PyTorch, BERT|
|Backend|FastAPI, SQLAlchemy, SQLite|
|Scraping|Requests, BeautifulSoup, Feedparser|
|Dashboard|Streamlit, Plotly|
|Tracking|MLflow|
|DevOps|Docker, GitHub Actions|

## Quick Start

### Prerequisites

* Python 3.11+
* pip

### Installation

```bash
git clone https://github.com/meelikeuyar/habernlp.git
cd habernlp
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### Usage

**Scrape news and run sentiment analysis:**

```bash
python -c "from src.scraper.scraper import haber\_cek, kaydet; kaydet(haber\_cek())"
python run\_analysis.py
```

**Launch the analytics dashboard:**

```bash
streamlit run streamlit\_app.py
```

**Launch the FastAPI backend:**

```bash
python main.py
```

The API will be available at `http://localhost:8000` and the dashboard at `http://localhost:8501`.

### Docker

```bash
docker-compose up --build
```

## API Endpoints

|Method|Endpoint|Description|
|-|-|-|
|GET|`/api/health`|Health check|
|GET|`/api/haberler`|List articles (filterable by source, sentiment, date)|
|GET|`/api/analiz`|Sentiment distribution, keywords, source stats|
|GET|`/api/trends`|Daily sentiment trends|
|POST|`/api/guncelle`|Trigger manual scrape + analysis|

## Project Structure

```
habernlp/
├── config/
│   └── settings.py          # Centralized configuration
├── src/
│   ├── api/
│   │   └── routes.py        # FastAPI endpoints
│   ├── database/
│   │   └── models.py        # SQLAlchemy models
│   ├── nlp/
│   │   ├── sentiment.py     # BERT sentiment analysis
│   │   └── keywords.py      # TF-IDF keyword extraction
│   ├── scraper/
│   │   └── scraper.py       # RSS + HTML hybrid scraper
│   └── scheduler/
│       └── jobs.py          # APScheduler jobs
├── notebooks/
│   └── train\_sentiment.py   # Model training script
├── frontend/
│   └── templates/           # FastAPI dashboard HTML
├── streamlit\_app.py         # Analytics dashboard
├── main.py                  # Application entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Model Training

The sentiment model was fine-tuned on 964 manually labeled Turkish news headlines with a 70/15/15 train/val/test split.

|Metric|Score|
|-|-|
|F1 (macro)|0.97|
|Accuracy|0.97|
|Precision|0.97|
|Recall|0.97|

## News Sources

|Source|Method|Status|
|-|-|-|
|T24|HTML scraping|✅|
|BBC Türkçe|RSS|✅|
|NTV|RSS|✅|
|Hürriyet|RSS|✅|
|Sözcü|RSS|✅|
|Habertürk|RSS|✅|
|TRT Haber|RSS|✅|
|Euronews TR|RSS|✅|
|DW Türkçe|RSS|✅|
|Independent TR|RSS|✅|

## License

This project is for educational purposes.

