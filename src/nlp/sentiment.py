"""Sentiment analysis engine.

Primary:  Fine-tuned BERT (dbmdz/bert-base-turkish-cased)
Fallback: Rule-based keyword matching (when model is unavailable)
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from config.settings import BERT_MODEL_NAME, MODEL_DIR, SENTIMENT_LABELS

# ── Lazy-loaded model cache ──
_model = None
_tokenizer = None
_device = None


def _load_model():
    """Load fine-tuned BERT model. Returns (model, tokenizer, device) or None."""
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_path = MODEL_DIR if MODEL_DIR.exists() else BERT_MODEL_NAME

        logger.info(f"Loading sentiment model from: {model_path}")
        _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        _model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=3
        )
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
        logger.info(f"Model loaded on {_device}")
        return _model, _tokenizer, _device
    except Exception as e:
        logger.warning(f"BERT model unavailable, using fallback: {e}")
        return None


def analiz_bert(baslik: str) -> tuple[str, float]:
    """Classify sentiment using BERT. Returns (label, confidence)."""
    result = _load_model()
    if result is None:
        return _analiz_kural(baslik)

    import torch

    model, tokenizer, device = result
    inputs = tokenizer(
        baslik,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()

    return SENTIMENT_LABELS[pred_idx], round(confidence, 4)


# ── Rule-based fallback ──
_POZITIF = {
    "başarı", "zafer", "iyi", "güzel", "harika", "kazandı", "büyüme",
    "gelişme", "barış", "anlaşma", "çözüm", "olumlu", "yükseldi",
    "rekor", "şampiyon", "kazanç", "artış", "iyileşme", "umut",
    "müjde", "destek", "yenilik", "onay", "kutlama",
}

_NEGATIF = {
    "savaş", "ölüm", "kriz", "tutuklama", "patlama", "saldırı",
    "kaos", "felaket", "trajedi", "düşüş", "kayıp", "tehlike",
    "protesto", "istifa", "skandal", "yasak", "iflas", "darbe",
    "bomba", "yangın", "deprem", "sel", "terör", "suç",
    "cinayet", "gözaltı", "kaza", "çatışma", "soykırım",
}


def _analiz_kural(baslik: str) -> tuple[str, float]:
    """Fallback rule-based sentiment from keyword matching."""
    kelimeler = set(re.findall(r"\b\w+\b", baslik.lower()))
    poz = len(kelimeler & _POZITIF)
    neg = len(kelimeler & _NEGATIF)

    if neg > poz:
        return "negatif", -1.0
    elif poz > neg:
        return "pozitif", 1.0
    return "nötr", 0.0


def analiz(baslik: str) -> tuple[str, float]:
    """Public API — tries BERT first, falls back to rules."""
    return analiz_bert(baslik)
