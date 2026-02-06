"""
PageIndex API integration.

Wraps the `pageindex` Python SDK with:
- Safe env-based initialization
- Polling helpers
- Minimal, non-secret logging
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger


class PageIndexUnavailableError(RuntimeError):
    """Raised when the PageIndex SDK or API key is unavailable."""


class PageIndexProcessingError(RuntimeError):
    """Raised when PageIndex reports a failed document processing status."""


@dataclass(frozen=True)
class PageIndexAPI:
    _client: Any
    _api_key_fingerprint: str = field(repr=False, default="")

    @classmethod
    def from_env(cls, env_var: str = "PAGEINDEX_API_KEY") -> Optional["PageIndexAPI"]:
        api_key = (os.getenv(env_var) or "").strip()
        if not api_key:
            return None

        try:
            from pageindex.client import PageIndexClient  # type: ignore
        except Exception as e:
            logger.warning(f"PageIndex SDK not available: {e}")
            return None

        fingerprint = api_key[-4:] if len(api_key) >= 4 else "***"
        client = PageIndexClient(api_key=api_key)
        return cls(_client=client, _api_key_fingerprint=fingerprint)

    def submit_pdf(self, path: str) -> str:
        """Submit a PDF for processing. Returns doc_id."""
        try:
            response: Dict[str, Any] = self._client.submit_document(path)
        except Exception as e:
            raise PageIndexUnavailableError(str(e)) from e

        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise PageIndexUnavailableError("PageIndex submit_document returned no doc_id")

        logger.info(f"PageIndex submitted PDF (doc_id={doc_id})")
        return str(doc_id)

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata (status, name, pageNum, etc)."""
        return dict(self._client.get_document(doc_id))

    def get_tree(self, doc_id: str, node_summary: bool = True) -> Dict[str, Any]:
        """Get tree generation status + result payload."""
        return dict(self._client.get_tree(doc_id, node_summary=node_summary))

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from PageIndex."""
        try:
            self._client.delete_document(doc_id)
        except Exception as e:
            raise PageIndexUnavailableError(str(e)) from e
        logger.info(f"PageIndex deleted document (doc_id={doc_id})")

    def wait_for_completed(
        self,
        doc_id: str,
        timeout_s: int = 900,
        poll_s: float = 5.0,
    ) -> Dict[str, Any]:
        """Poll document metadata until status=completed (or failed/timeout)."""
        deadline = time.time() + timeout_s
        last_status: str = "unknown"

        while time.time() < deadline:
            meta = self.get_document(doc_id)
            status = str(meta.get("status") or meta.get("state") or "unknown").lower()
            last_status = status

            if status in {"completed", "complete", "ready", "done"}:
                logger.info(f"PageIndex processing completed (doc_id={doc_id})")
                return meta
            if status in {"failed", "error"}:
                raise PageIndexProcessingError(
                    f"PageIndex processing failed (doc_id={doc_id}, status={status})"
                )

            time.sleep(poll_s)

        raise TimeoutError(
            f"Timed out waiting for PageIndex completion (doc_id={doc_id}, last_status={last_status})"
        )

