"""Shared embedder service — thin wrapper over `src.core.agent.embedders`.

All semantic retrieval in brick.agent (SQL column scoring, OKF concept
retrieval, request routing) goes through this one service so the
embedder model is swappable in one place.
"""

from __future__ import annotations

from typing import Sequence

from src.core.agent.embedders import get_embedder


def embed(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts (L2-normalised vectors)."""
    return get_embedder().embed(list(texts))


def embed_one(text: str) -> list[float]:
    """Embed a single text (L2-normalised vector)."""
    return get_embedder().embed([text])[0]


def model_name() -> str:
    """The active embedder's model id."""
    return get_embedder().model


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product — valid cosine similarity because the vectors are
    L2-normalised by the embedder contract."""
    return sum(x * y for x, y in zip(a, b))


def rank_by_similarity(
    query: str,
    candidates: Sequence[str],
) -> list[tuple[str, float]]:
    """Rank candidates against a query by cosine similarity, desc."""
    vecs = get_embedder().embed([query, *candidates])
    q_vec = vecs[0]
    scored = [
        (c, cosine_similarity(q_vec, v))
        for c, v in zip(candidates, vecs[1:])
    ]
    return sorted(scored, key=lambda t: -t[1])
