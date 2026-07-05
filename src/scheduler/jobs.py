"""Scheduled jobs — periodic scraping and NLP analysis."""

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from config.settings import SCRAPE_INTERVAL_MINUTES
from src.scraper.scraper import haber_cek, kaydet
from src.database import SessionLocal
from src.database.models import Article
from src.nlp.sentiment import analiz


def _scrape_and_analyze():
    """Scrape all sources, save to DB, run sentiment analysis on new articles."""
    logger.info("⏰ Scheduled scrape started")

    haberler = haber_cek()
    inserted = kaydet(haberler)

    # Run sentiment on articles that haven't been analyzed yet
    session = SessionLocal()
    try:
        unanalyzed = session.query(Article).filter(Article.sentiment.is_(None)).all()
        for article in unanalyzed:
            label, score = analiz(article.baslik)
            article.sentiment = label
            article.sentiment_score = score
        session.commit()
        logger.info(f"Sentiment analyzed: {len(unanalyzed)} articles")
    finally:
        session.close()

    logger.info(f"⏰ Scheduled scrape done: {inserted} new articles")


scheduler = BackgroundScheduler()


def start_scheduler():
    """Start the periodic scrape job."""
    scheduler.add_job(
        _scrape_and_analyze,
        "interval",
        minutes=SCRAPE_INTERVAL_MINUTES,
        id="scrape_job",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(f"Scheduler started — scraping every {SCRAPE_INTERVAL_MINUTES} min")


def stop_scheduler():
    """Gracefully shut down."""
    scheduler.shutdown(wait=False)
