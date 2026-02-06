"""
RAG tools - on-demand ingestion and document management.

These tools allow the agent (and user) to:
- Index PDFs via PageIndex (tree cached locally for retrieval)
- Index local markdown/text files into the hierarchical RAG index
- List and delete indexed documents
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from intelclaw.tools.base import (
    BaseTool,
    ToolCategory,
    ToolDefinition,
    ToolPermission,
    ToolResult,
)

if TYPE_CHECKING:
    from intelclaw.memory.manager import MemoryManager


def _resolve_path(path: str) -> Path:
    raw = (path or "").strip().strip('"').strip("'")
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    # Try relative to CWD
    return (Path.cwd() / p).resolve()


class RagIndexPathTool(BaseTool):
    def __init__(self, memory: Optional["MemoryManager"] = None):
        self._memory = memory

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag_index_path",
            description="Index a local path for retrieval (PDF via PageIndex; .md/.txt locally).",
            category=ToolCategory.PRODUCTIVITY,
            permissions=[ToolPermission.READ, ToolPermission.WRITE, ToolPermission.NETWORK],
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to a local file (.pdf/.md/.txt)."},
                    "title": {"type": "string", "description": "Optional display title for the document."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for the document (stored in local registry where applicable).",
                    },
                },
                "required": ["path"],
            },
            returns="object",
            examples=[{"path": "data/pageindex_inbox/contract.pdf"}],
        )

    async def execute(
        self,
        path: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        memory = self._memory
        rag = getattr(memory, "agentic_rag", None) if memory else None
        if not rag:
            return ToolResult(success=False, error="RAG system is not initialized")

        try:
            p = _resolve_path(path)
            if not p.exists() or not p.is_file():
                return ToolResult(success=False, error=f"File not found: {p}")

            doc_id = await rag.index_path(p, title=title, tags=tags)
            if not doc_id:
                return ToolResult(
                    success=False,
                    error="Indexing skipped or failed. For PDFs, ensure PAGEINDEX_API_KEY is set in .env.",
                )

            info = await asyncio.to_thread(getattr(rag, "get_document_info", lambda _id: {}), doc_id)
            info = info or {}
            info.setdefault("doc_id", doc_id)
            return ToolResult(success=True, data=info)
        except Exception as e:
            logger.debug(f"rag_index_path failed: {e}")
            return ToolResult(success=False, error=str(e))


class RagListDocumentsTool(BaseTool):
    def __init__(self, memory: Optional["MemoryManager"] = None):
        self._memory = memory

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag_list_documents",
            description="List documents currently indexed for RAG retrieval.",
            category=ToolCategory.PRODUCTIVITY,
            permissions=[ToolPermission.READ],
            parameters={"type": "object", "properties": {}},
            returns="object",
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        memory = self._memory
        rag = getattr(memory, "agentic_rag", None) if memory else None
        if not rag:
            return ToolResult(success=False, error="RAG system is not initialized")

        try:
            docs = await asyncio.to_thread(getattr(rag, "list_documents", lambda: []))
            return ToolResult(success=True, data={"documents": docs, "count": len(docs)})
        except Exception as e:
            logger.debug(f"rag_list_documents failed: {e}")
            return ToolResult(success=False, error=str(e))


class RagDeleteDocumentTool(BaseTool):
    def __init__(self, memory: Optional["MemoryManager"] = None):
        self._memory = memory

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag_delete_document",
            description="Delete an indexed document by doc_id (local cache; optionally remote PageIndex delete).",
            category=ToolCategory.PRODUCTIVITY,
            permissions=[ToolPermission.WRITE, ToolPermission.NETWORK],
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID to delete."},
                    "confirm": {
                        "type": "boolean",
                        "description": "Set true to confirm deletion (required).",
                        "default": False,
                    },
                    "delete_remote": {
                        "type": "boolean",
                        "description": "If true and doc is PageIndex-backed, also delete from PageIndex.",
                        "default": False,
                    },
                },
                "required": ["doc_id"],
            },
            returns="object",
            requires_confirmation=True,
        )

    async def execute(
        self,
        doc_id: str,
        confirm: bool = False,
        delete_remote: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not confirm:
            return ToolResult(
                success=False,
                error="Deletion not confirmed. Re-run with confirm=true to delete.",
            )

        memory = self._memory
        rag = getattr(memory, "agentic_rag", None) if memory else None
        if not rag:
            return ToolResult(success=False, error="RAG system is not initialized")

        try:
            result = await rag.delete_document(str(doc_id), delete_remote=bool(delete_remote))
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.debug(f"rag_delete_document failed: {e}")
            return ToolResult(success=False, error=str(e))

