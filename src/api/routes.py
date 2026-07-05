"""FastAPI routes for the HaberNLP API."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database import get_db
from src.database.models import Article
from src.scraper.scraper import haber_cek, kaydet
from src.nlp.sentiment import analiz
from src.nlp.keywords import extract_tfidf, extract_frequency
from src.nlp.topics import get_topic_summary

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/haberler")
def haberler(
    kaynak: str | None = None,
    sentiment: str | None = None,
    gun: int = 7,
    limit: int = Query(100, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List articles with optional filters."""
    q = db.query(Article).filter(
        Article.tarih >= datetime.utcnow() - timedelta(days=gun)
    )
    if kaynak:
        q = q.filter(Article.kaynak == kaynak)
    if sentiment:
        q = q.filter(Article.sentiment == sentiment)

    total = q.count()
    articles = q.order_by(Article.tarih.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "articles": [a.to_dict() for a in articles],
    }


@router.get("/analiz")
def analiz_endpoint(gun: int = 1, db: Session = Depends(get_db)):
    """Full analysis: sentiment distribution, keywords, source stats."""
    since = datetime.utcnow() - timedelta(days=gun)
    articles = db.query(Article).filter(Article.tarih >= since).all()

    if not articles:
        return {"hata": "No data available", "toplam": 0}

    basliklar = [a.baslik for a in articles]

    # Sentiment distribution
    sentiments = [a.sentiment for a in articles if a.sentiment]
    duygu = {
        "pozitif": sentiments.count("pozitif"),
        "negatif": sentiments.count("negatif"),
        "notr": sentiments.count("nötr"),
    }
    toplam = len(sentiments) or 1
    duygu["pozitif_oran"] = round(duygu["pozitif"] / toplam * 100)
    duygu["negatif_oran"] = round(duygu["negatif"] / toplam * 100)
    duygu["notr_oran"] = round(duygu["notr"] / toplam * 100)

    # Source distribution
    kaynak_counts = {}
    for a in articles:
        kaynak_counts[a.kaynak] = kaynak_counts.get(a.kaynak, 0) + 1

    return {
        "toplam": len(articles),
        "tarih": datetime.now().strftime("%d %B %Y"),
        "duygu": duygu,
        "kelimeler_tfidf": extract_tfidf(basliklar),
        "kelimeler_frekans": extract_frequency(basliklar),
        "kaynaklar": [{"ad": k, "sayi": v} for k, v in kaynak_counts.items()],
        "topics": [],
    }


@router.get("/trends")
def trends(gun: int = 7, db: Session = Depends(get_db)):
    """Daily sentiment trend over the last N days."""
    since = datetime.utcnow() - timedelta(days=gun)
    articles = db.query(Article).filter(Article.tarih >= since).all()

    daily: dict[str, dict] = {}
    for a in articles:
        day = a.tarih.strftime("%Y-%m-%d") if a.tarih else "unknown"
        if day not in daily:
            daily[day] = {"pozitif": 0, "negatif": 0, "notr": 0, "toplam": 0}
        daily[day]["toplam"] += 1
        if a.sentiment:
            key = a.sentiment.replace("ö", "o")
            if key in daily[day]:
                daily[day][key] += 1

    return {
        "days": [
            {"tarih": k, **v}
            for k, v in sorted(daily.items())
        ]
    }


@router.post("/guncelle")
def guncelle(db: Session = Depends(get_db)):
    """Trigger a manual scrape + sentiment analysis."""
    haberler = haber_cek()
    inserted = kaydet(haberler)

    # Analyze new articles
    unanalyzed = db.query(Article).filter(Article.sentiment.is_(None)).all()
    for article in unanalyzed:
        label, score = analiz(article.baslik)
        article.sentiment = label
        article.sentiment_score = score
    db.commit()

    return {
        "mesaj": f"✓ {inserted} new articles saved, {len(unanalyzed)} analyzed",
        "toplam_yeni": inserted,
        "toplam_analiz": len(unanalyzed),
    }