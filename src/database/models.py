"""SQLAlchemy ORM models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index
from src.database import Base


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    baslik = Column(String(500), nullable=False)
    url = Column(String(1000), unique=True, nullable=False)
    kaynak = Column(String(100), nullable=False)
    tarih = Column(DateTime, default=datetime.utcnow)

    # NLP results (populated after analysis)
    sentiment = Column(String(20), default=None)
    sentiment_score = Column(Float, default=None)
    topic_id = Column(Integer, default=None)
    topic_label = Column(String(200), default=None)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_articles_tarih", "tarih"),
        Index("ix_articles_kaynak", "kaynak"),
        Index("ix_articles_sentiment", "sentiment"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "baslik": self.baslik,
            "url": self.url,
            "kaynak": self.kaynak,
            "tarih": self.tarih.isoformat() if self.tarih else None,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "topic_id": self.topic_id,
            "topic_label": self.topic_label,
        }
