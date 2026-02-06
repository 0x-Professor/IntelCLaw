"""
Jina AI embeddings integration.

Uses the hosted Jina Embeddings API (OpenAI-compatible request/response shape).

Docs: https://jina.ai/embeddings/
OpenAPI: https://api.jina.ai/openapi.json
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


DEFAULT_JINA_BASE_URL = "https://api.jina.ai/v1/embeddings"
DEFAULT_JINA_MODEL = "jina-embeddings-v3"


class JinaEmbeddingsError(RuntimeError):
    """Raised when the Jina embeddings API returns an error."""


@dataclass(frozen=True)
class JinaEmbeddingsConfig:
    api_key: str
    model: str = DEFAULT_JINA_MODEL
    base_url: str = DEFAULT_JINA_BASE_URL
    timeout_s: float = 30.0


class JinaEmbeddingsClient:
    def __init__(self, config: JinaEmbeddingsConfig):
        self._cfg = config
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    def from_env(
        cls,
        api_key_env: str = "JINA_API_KEY",
        model_env: str = "JINA_EMBEDDINGS_MODEL",
        base_url_env: str = "JINA_API_BASE_URL",
    ) -> Optional["JinaEmbeddingsClient"]:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return None

        model = os.environ.get(model_env, DEFAULT_JINA_MODEL) or DEFAULT_JINA_MODEL
        base_url = os.environ.get(base_url_env, DEFAULT_JINA_BASE_URL) or DEFAULT_JINA_BASE_URL

        return cls(JinaEmbeddingsConfig(api_key=api_key, model=model, base_url=base_url))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._cfg.timeout_s),
                headers={
                    "Authorization": f"Bearer {self._cfg.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def embed(
        self,
        texts: Sequence[str],
        *,
        task: Optional[str] = None,
        late_chunking: Optional[bool] = None,
        dimensions: Optional[int] = None,
        embedding_type: Optional[str] = None,
        normalized: Optional[bool] = True,
        truncate: Optional[bool] = None,
    ) -> List[List[float]]:
        """
        Create embeddings for the given texts.

        Args:
            texts: List of texts to embed.
            task: Optional task string (e.g. "retrieval.query" / "retrieval.passage").
            late_chunking: Enable contextual chunk embeddings (API feature).
            dimensions: Optional output dimension (model-dependent).
            embedding_type: Optional output type ("float" or "binary", model-dependent).
            normalized: Request normalized embeddings when supported.
            truncate: Whether to truncate overlong inputs when supported.
        """
        items = [str(t or "") for t in texts]
        if not items:
            return []

        payload: dict[str, Any] = {"model": self._cfg.model, "input": items}
        if task is not None:
            payload["task"] = task
        if late_chunking is not None:
            payload["late_chunking"] = bool(late_chunking)
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)
        if embedding_type is not None:
            payload["embedding_type"] = str(embedding_type)
        if normalized is not None:
            payload["normalized"] = bool(normalized)
        if truncate is not None:
            payload["truncate"] = bool(truncate)

        client = await self._get_client()
        try:
            resp = await client.post(self._cfg.base_url, json=payload)
        except Exception as e:
            logger.debug(f"Jina embeddings request failed: {e}")
            raise

        if resp.status_code >= 400:
            # Never log API keys; include only status and a small excerpt.
            excerpt = (resp.text or "")[:500]
            raise JinaEmbeddingsError(
                f"Jina embeddings API error (status={resp.status_code}): {excerpt}"
            )

        data = resp.json()
        out = data.get("data")
        if not isinstance(out, list):
            raise JinaEmbeddingsError("Invalid embeddings response: missing 'data' list")

        embeddings_by_index: dict[int, List[float]] = {}
        for i, item in enumerate(out):
            if not isinstance(item, dict):
                continue
            idx = item.get("index", i)
            emb = item.get("embedding")
            if isinstance(idx, int) and isinstance(emb, list):
                embeddings_by_index[idx] = [float(x) for x in emb]

        # Preserve request order. If the API doesn't return indices, fall back to list order.
        if len(embeddings_by_index) == len(items):
            return [embeddings_by_index[i] for i in range(len(items))]

        # Fallback best-effort
        ordered: List[List[float]] = []
        for i in range(len(items)):
            if i in embeddings_by_index:
                ordered.append(embeddings_by_index[i])
        if len(ordered) != len(items):
            raise JinaEmbeddingsError(
                f"Invalid embeddings response: expected {len(items)} vectors, got {len(ordered)}"
            )
        return ordered

