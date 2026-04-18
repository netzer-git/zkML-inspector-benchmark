"""Pluggable similarity interface for finding matching and text comparison.

Ships with a Jaccard baseline. Future backends (TF-IDF, embeddings, LLM-as-judge)
implement the same SimilarityBackend interface so matcher/scorers need no changes.
"""

from abc import ABC, abstractmethod
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SimilarityBackend(ABC):
    """Abstract base for text similarity scoring."""

    @abstractmethod
    def score(self, text_a: str, text_b: str) -> float:
        """Return similarity in [0, 1]. 0 = unrelated, 1 = identical."""


class JaccardSimilarity(SimilarityBackend):
    """Word-level Jaccard similarity. Zero extra dependencies."""

    def score(self, text_a: str, text_b: str) -> float:
        tokens_a = set(normalize_text(text_a).split())
        tokens_b = set(normalize_text(text_b).split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
