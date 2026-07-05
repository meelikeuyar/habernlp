"""Keyword extraction using TF-IDF with Turkish stop words."""

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = {
    "ve", "ile", "bu", "bir", "da", "de", "mi", "mu", "mü", "için",
    "olan", "çok", "daha", "en", "ne", "ki", "şu", "var", "yok",
    "gibi", "kadar", "sonra", "önce", "ama", "ancak", "nin", "den",
    "dan", "ın", "in", "un", "ün", "nın", "ten", "tan", "ler", "lar",
    "deki", "daki", "nde", "nda", "nden", "ile", "ise", "idi", "değil",
    "her", "ben", "sen", "biz", "siz", "onlar", "nasıl", "neden",
    "hangi", "oldu", "olan", "olarak", "sonra", "haber", "haberi",
    "the", "a", "an", "in", "of", "to", "is", "was", "are", "were",
}


def extract_tfidf(basliklar: list[str], top_n: int = 15) -> list[dict]:
    """Extract top keywords using TF-IDF scoring."""
    if not basliklar:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        token_pattern=r"\b[a-zA-ZğüşıöçĞÜŞİÖÇ]{3,}\b",
        max_features=500,
        lowercase=True,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(basliklar)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    return [{"kelime": k, "skor": round(float(s), 4)} for k, s in ranked[:top_n]]


def extract_frequency(basliklar: list[str], top_n: int = 10) -> list[dict]:
    """Simple frequency-based keyword extraction (for comparison)."""
    tum: list[str] = []
    for baslik in basliklar:
        kelimeler = re.findall(r"\b[a-zA-ZğüşıöçĞÜŞİÖÇ]{3,}\b", baslik.lower())
        tum.extend(k for k in kelimeler if k not in STOP_WORDS)

    return [{"kelime": k, "sayi": s} for k, s in Counter(tum).most_common(top_n)]
