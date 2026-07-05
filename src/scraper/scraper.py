"""Multi-source Turkish news scraper — RSS + HTML hybrid with DB persistence."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
import requests
from bs4 import BeautifulSoup
from loguru import logger
from sqlalchemy.exc import IntegrityError

from config.settings import HEADERS, KAYNAKLAR, REQUEST_TIMEOUT, MAX_WORKERS
from src.database import SessionLocal
from src.database.models import Article


# ─────────────────────── HTML Scraper (eski yöntem) ───────────────────────

def _html_cek(kaynak: dict) -> list[dict]:
    """Scrape a single source via HTML link extraction."""
    logger.info(f"[HTML] {kaynak['ad']} → {kaynak['url']}")
    try:
        resp = requests.get(kaynak["url"], headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"  ✗ {kaynak['ad']} HTML failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    gorulenler: set[str] = set()
    haberler: list[dict] = []

    for link in soup.find_all("a", href=True):
        baslik = link.get_text(strip=True)
        href = link["href"]

        if len(baslik) < 20:
            continue
        if not any(f in href for f in kaynak.get("filtre", [])):
            continue

        if href.startswith("/"):
            tam_url = kaynak["url"].rstrip("/") + href
        elif href.startswith("http"):
            tam_url = href
        else:
            continue

        if baslik in gorulenler:
            continue
        gorulenler.add(baslik)

        haberler.append({
            "baslik": baslik,
            "url": tam_url,
            "kaynak": kaynak["ad"],
            "tarih": datetime.now(),
        })

    logger.info(f"  ✓ {kaynak['ad']}: {len(haberler)} articles (HTML)")
    return haberler


# ─────────────────────── RSS Scraper (yeni yöntem) ────────────────────────

def _rss_cek(kaynak: dict) -> list[dict]:
    """Scrape a single source via RSS feed."""
    logger.info(f"[RSS]  {kaynak['ad']} → {kaynak['url']}")
    try:
        resp = requests.get(kaynak["url"], headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"  ✗ {kaynak['ad']} RSS failed: {e}")
        return []

    feed = feedparser.parse(resp.text)

    if feed.bozo and not feed.entries:
        logger.warning(f"  ⚠ {kaynak['ad']}: malformed feed, 0 entries")
        return []

    haberler: list[dict] = []
    for entry in feed.entries:
        baslik = entry.get("title", "").strip()
        url = entry.get("link", "").strip()

        if not baslik or len(baslik) < 10 or not url:
            continue

        # Tarihi RSS'ten al, yoksa şimdiki zaman
        tarih = datetime.now()
        if entry.get("published"):
            try:
                tarih = parsedate_to_datetime(entry["published"])
                tarih = tarih.replace(tzinfo=None)  # naive datetime
            except Exception:
                pass

        haberler.append({
            "baslik": baslik,
            "url": url,
            "kaynak": kaynak["ad"],
            "tarih": tarih,
        })

    logger.info(f"  ✓ {kaynak['ad']}: {len(haberler)} articles (RSS)")
    return haberler


# ─────────────────────── Dispatcher ───────────────────────────────────────

def _tek_kaynak_cek(kaynak: dict) -> list[dict]:
    """Route to RSS or HTML scraper based on kaynak['tip']."""
    tip = kaynak.get("tip", "html")
    if tip == "rss":
        return _rss_cek(kaynak)
    return _html_cek(kaynak)


def haber_cek(kaynaklar: list[dict] | None = None) -> list[dict]:
    """Scrape all sources concurrently and return combined, deduplicated results."""
    kaynaklar = kaynaklar or KAYNAKLAR
    tum_haberler: list[dict] = []
    gorulen_basliklar: set[str] = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_tek_kaynak_cek, k): k["ad"] for k in kaynaklar
        }
        for future in as_completed(futures):
            try:
                for haber in future.result():
                    if haber["baslik"] not in gorulen_basliklar:
                        gorulen_basliklar.add(haber["baslik"])
                        tum_haberler.append(haber)
            except Exception as e:
                logger.error(f"Thread error for {futures[future]}: {e}")

    logger.info(f"Total scraped: {len(tum_haberler)} unique articles from {len(kaynaklar)} feeds")
    return tum_haberler


def kaydet(haberler: list[dict]) -> int:
    """Persist articles to the database. Returns count of new inserts."""
    if not haberler:
        return 0

    session = SessionLocal()
    inserted = 0
    try:
        for h in haberler:
            article = Article(
                baslik=h["baslik"],
                url=h["url"],
                kaynak=h["kaynak"],
                tarih=h["tarih"],
            )
            try:
                session.add(article)
                session.flush()
                inserted += 1
            except IntegrityError:
                session.rollback()
        session.commit()
    finally:
        session.close()

    logger.info(f"Saved {inserted} new articles ({len(haberler) - inserted} duplicates skipped)")
    return inserted
