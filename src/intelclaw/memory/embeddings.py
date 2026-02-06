"""
Embeddings providers for local retrieval.

This module provides a small abstraction over embedding generation so the rest
of the codebase can swap between:
- Local hashing embeddings (offline, best-effort)
- Hosted Jina embeddings (higher quality, requires JINA_API_KEY)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from loguru import logger

import os

from intelclaw.integrations.jina_embeddings import (
    DEFAULT_JINA_BASE_URL,
    DEFAULT_JINA_MODEL,
    JinaEmbeddingsClient,
    JinaEmbeddingsConfig,
)
from intelclaw.memory.vector_store import HashingEmbeddingFunction


class EmbeddingsProvider(Protocol):
    name: str
    dimension: Optional[int]

    async def embed_passages(self, texts: Sequence[str]) -> List[List[float]]: ...

    async def embed_query(self, text: str) -> List[float]: ...

    async def close(self) -> None: ...


class LocalHashEmbeddings:
    name = "local_hash"

    def __init__(self, dimension: int = 384):
        self.dimension: Optional[int] = int(dimension)
        self._fn = HashingEmbeddingFunction(dimension=int(dimension))

    async def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = self._fn(list(texts))
        out: List[List[float]] = []
        for v in vecs:
            if hasattr(v, "tolist"):
                out.append([float(x) for x in v.tolist()])  # type: ignore[union-attr]
            else:
                out.append([float(x) for x in list(v)])
        return out

    async def embed_query(self, text: str) -> List[float]:
        vecs = await self.embed_passages([text])
        return vecs[0] if vecs else []

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class JinaEmbeddingsSettings:
    model: str = "jina-embeddings-v3"
    query_task: str = "retrieval.query"
    passage_task: str = "retrieval.passage"
    late_chunking: bool = True
    dimensions: Optional[int] = None
    normalized: bool = True
    truncate: Optional[bool] = None


class JinaEmbeddings:
    name = "jina"

    def __init__(self, client: JinaEmbeddingsClient, settings: Optional[JinaEmbeddingsSettings] = None):
        self._client = client
        self._settings = settings or JinaEmbeddingsSettings(model=client._cfg.model)  # type: ignore[attr-defined]
        self.dimension: Optional[int] = self._settings.dimensions

    @classmethod
    def from_env(cls, settings: Optional[JinaEmbeddingsSettings] = None) -> Optional["JinaEmbeddings"]:
        settings = settings or JinaEmbeddingsSettings()
        api_key = os.environ.get("JINA_API_KEY")
        if not api_key:
            return None
        base_url = os.environ.get("JINA_API_BASE_URL", DEFAULT_JINA_BASE_URL) or DEFAULT_JINA_BASE_URL
        client = JinaEmbeddingsClient(JinaEmbeddingsConfig(api_key=api_key, model=settings.model, base_url=base_url))
        if not client:
            return None
        return cls(client, settings=settings)

    async def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return await self._client.embed(
            texts,
            task=self._settings.passage_task,
            late_chunking=self._settings.late_chunking,
            dimensions=self._settings.dimensions,
            normalized=self._settings.normalized,
            truncate=self._settings.truncate,
        )

    async def embed_query(self, text: str) -> List[float]:
        vecs = await self._client.embed(
            [text],
            task=self._settings.query_task,
            late_chunking=False,
            dimensions=self._settings.dimensions,
            normalized=self._settings.normalized,
            truncate=self._settings.truncate,
        )
        return vecs[0] if vecs else []

    async def close(self) -> None:
        await self._client.close()


def embeddings_from_config(cfg: dict) -> EmbeddingsProvider:
    provider = str(cfg.get("provider", "jina") or "jina").lower()

    if provider == "jina":
        settings = JinaEmbeddingsSettings(
            model=str(cfg.get("model", DEFAULT_JINA_MODEL) or DEFAULT_JINA_MODEL),
            query_task=str(cfg.get("query_task", "retrieval.query") or "retrieval.query"),
            passage_task=str(cfg.get("passage_task", "retrieval.passage") or "retrieval.passage"),
            late_chunking=bool(cfg.get("late_chunking", True)),
            dimensions=cfg.get("dimensions", None),
            normalized=bool(cfg.get("normalized", True)),
            truncate=cfg.get("truncate", None),
        )
        emb = JinaEmbeddings.from_env(settings=settings)
        if emb:
            return emb

        logger.warning("Jina embeddings selected but JINA_API_KEY is not set; falling back to local hashing embeddings")
        return LocalHashEmbeddings(dimension=int(cfg.get("fallback_dimension", 384)))

    # Default: local hashing
    return LocalHashEmbeddings(dimension=int(cfg.get("dimension", 384)))
