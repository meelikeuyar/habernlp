"""Topic modeling with BERTopic for automatic article clustering."""

from __future__ import annotations

from loguru import logger
from config.settings import TOPIC_MIN_CLUSTER_SIZE

_topic_model = None


def _get_model():
    """Lazy-load BERTopic model."""
    global _topic_model
    if _topic_model is not None:
        return _topic_model

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

        _topic_model = BERTopic(
            embedding_model=embedding_model,
            language="multilingual",
            min_topic_size=TOPIC_MIN_CLUSTER_SIZE,
            verbose=False,
        )
        logger.info("BERTopic model initialized")
        return _topic_model
    except Exception as e:
        logger.warning(f"BERTopic unavailable: {e}")
        return None


def extract_topics(basliklar: list[str]) -> list[dict]:
    """
    Cluster headlines into topics.

    Returns list of dicts: [{"baslik": str, "topic_id": int, "topic_label": str}, ...]
    """
    if len(basliklar) < TOPIC_MIN_CLUSTER_SIZE:
        logger.info("Not enough articles for topic modeling")
        return [{"baslik": b, "topic_id": -1, "topic_label": "uncategorized"} for b in basliklar]

    model = _get_model()
    if model is None:
        return [{"baslik": b, "topic_id": -1, "topic_label": "uncategorized"} for b in basliklar]

    try:
        topics, probs = model.fit_transform(basliklar)
        topic_info = model.get_topic_info()

        label_map = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            label_map[tid] = row.get("Name", f"topic_{tid}")

        results = []
        for baslik, tid in zip(basliklar, topics):
            results.append({
                "baslik": baslik,
                "topic_id": int(tid),
                "topic_label": label_map.get(tid, "uncategorized"),
            })

        logger.info(f"Discovered {len(set(topics)) - (1 if -1 in topics else 0)} topics")
        return results
    except Exception as e:
        logger.error(f"Topic modeling failed: {e}")
        return [{"baslik": b, "topic_id": -1, "topic_label": "uncategorized"} for b in basliklar]


def get_topic_summary() -> list[dict]:
    """Return summary of discovered topics (for API)."""
    model = _get_model()
    if model is None or not hasattr(model, "get_topic_info"):
        return []

    try:
        info = model.get_topic_info()
        return [
            {
                "topic_id": int(row["Topic"]),
                "count": int(row["Count"]),
                "label": row.get("Name", ""),
            }
            for _, row in info.iterrows()
            if row["Topic"] != -1
        ]
    except Exception:
        return []
