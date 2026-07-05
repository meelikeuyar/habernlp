"""Centralized configuration for HaberNLP."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "sentiment"
DATA_DIR.mkdir(exist_ok=True)

# ── Database ──
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'habernlp.db'}")

# ── Scraper ──
SCRAPE_INTERVAL_MINUTES = int(os.getenv("SCRAPE_INTERVAL", "60"))
REQUEST_TIMEOUT = 15
MAX_WORKERS = 6

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}

# ── Kaynak Tanımları ──
# tip: "rss" → feedparser ile çekilir, "html" → BeautifulSoup ile çekilir
# Birden fazla RSS URL'si olan kaynaklar ayrı ayrı tanımlanır.
KAYNAKLAR = [
    # ── T24 (HTML — zaten çalışıyor) ──
    {
        "ad": "T24",
        "tip": "html",
        "url": "https://t24.com.tr",
        "filtre": ["/haber/", "/yazarlar/"],
    },
    # ── BBC Türkçe ──
    {
        "ad": "BBC Türkçe",
        "tip": "rss",
        "url": "https://feeds.bbci.co.uk/turkce/rss.xml",
    },
    # ── NTV ──
    {
        "ad": "NTV",
        "tip": "rss",
        "url": "https://www.ntv.com.tr/gundem.rss",
    },
    {
        "ad": "NTV",
        "tip": "rss",
        "url": "https://www.ntv.com.tr/dunya.rss",
    },
    {
        "ad": "NTV",
        "tip": "rss",
        "url": "https://www.ntv.com.tr/ekonomi.rss",
    },
    # ── Hürriyet ──
    {
        "ad": "Hürriyet",
        "tip": "rss",
        "url": "https://www.hurriyet.com.tr/rss/gundem",
    },
    {
        "ad": "Hürriyet",
        "tip": "rss",
        "url": "https://www.hurriyet.com.tr/rss/ekonomi",
    },
    # ── Sözcü ──
    {
        "ad": "Sözcü",
        "tip": "rss",
        "url": "https://www.sozcu.com.tr/rss/gundem.xml",
    },
    {
        "ad": "Sözcü",
        "tip": "rss",
        "url": "https://www.sozcu.com.tr/rss/ekonomi.xml",
    },

    # ── Habertürk ──
    {
        "ad": "Habertürk",
        "tip": "rss",
        "url": "https://www.haberturk.com/rss",
    },
    # ── TRT Haber ──
    {
        "ad": "TRT Haber",
        "tip": "rss",
        "url": "https://www.trthaber.com/manset.rss",
    },
    # ── Yeni kaynaklar ──
    {
        "ad": "Euronews TR",
        "tip": "rss",
        "url": "https://tr.euronews.com/rss",
    },
    {
        "ad": "DW Türkçe",
        "tip": "rss",
        "url": "https://rss.dw.com/xml/rss-tur-all",
    },
    {
        "ad": "Independent TR",
        "tip": "rss",
        "url": "https://www.indyturk.com/rss.xml",
    },
]

# ── NLP ──
BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
SENTIMENT_LABELS = ["negatif", "nötr", "pozitif"]
TOPIC_MIN_CLUSTER_SIZE = 5

# ── API ──
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Logging ──
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
