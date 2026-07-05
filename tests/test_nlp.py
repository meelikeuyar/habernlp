"""Tests for NLP modules."""

import pytest
from src.nlp.sentiment import _analiz_kural
from src.nlp.keywords import extract_tfidf, extract_frequency, STOP_WORDS


class TestRuleBasedSentiment:
    def test_negative_keywords(self):
        label, score = _analiz_kural("Depremde büyük kayıplar yaşandı")
        assert label == "negatif"
        assert score < 0

    def test_positive_keywords(self):
        label, score = _analiz_kural("Takım büyük bir zafer kazandı")
        assert label == "pozitif"
        assert score > 0

    def test_neutral_no_keywords(self):
        label, score = _analiz_kural("Cumhurbaşkanı yarın Almanya'ya gidecek")
        assert label == "nötr"
        assert score == 0

    def test_empty_string(self):
        label, score = _analiz_kural("")
        assert label == "nötr"


class TestKeywords:
    def test_tfidf_returns_results(self):
        basliklar = [
            "Türkiye ekonomisi büyüme kaydetti",
            "Ekonomi büyüme hedefini aştı",
            "Spor müsabakası sonuçlandı",
        ]
        result = extract_tfidf(basliklar, top_n=5)
        assert len(result) > 0
        assert "kelime" in result[0]
        assert "skor" in result[0]

    def test_tfidf_empty_input(self):
        assert extract_tfidf([]) == []

    def test_frequency_returns_results(self):
        basliklar = ["Ekonomi ekonomi ekonomi büyüdü"]
        result = extract_frequency(basliklar, top_n=3)
        assert result[0]["kelime"] == "ekonomi"

    def test_stop_words_filtered(self):
        basliklar = ["ve ile bu bir için olan çok daha"]
        result = extract_frequency(basliklar)
        kelimeler = [r["kelime"] for r in result]
        for sw in ["ve", "ile", "bu", "bir"]:
            assert sw not in kelimeler
