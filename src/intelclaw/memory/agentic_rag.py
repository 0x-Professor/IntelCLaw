"""
Agentic RAG - Reasoning-based Retrieval Augmented Generation.

Inspired by PageIndex, this implements a hierarchical tree-based retrieval
system that uses LLM reasoning instead of vector similarity for more
accurate context retrieval.

Key features:
- Hierarchical document indexing (tree structure like table of contents)
- Reasoning-based retrieval (LLM decides relevance, not vector similarity)
- Session and persona persistence
- Context-aware retrieval with multiple search strategies
- Mem0 Platform integration for user preferences and infinite memory
"""

import asyncio
import json
import os
import re
from hashlib import sha256
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from intelclaw.integrations.pageindex_api import (
    PageIndexAPI,
    PageIndexProcessingError,
    PageIndexUnavailableError,
)
from intelclaw.memory.pageindex_store import PageIndexStore, sha256_file
from intelclaw.memory.tool_docs import ToolDocsIndex
from intelclaw.security.redaction import contains_secret, redact_secrets

if TYPE_CHECKING:
    from intelclaw.memory.long_term import LongTermMemory
    from intelclaw.memory.vector_store import VectorStore
# Try to import Mem0 - supports both local and platform API
try:
    from mem0 import MemoryClient  # Platform API (uses MEM0_API_KEY)
    MEM0_PLATFORM_AVAILABLE = True
except ImportError:
    MEM0_PLATFORM_AVAILABLE = False

try:
    from mem0 import Memory  # Local memory (requires OpenAI key)
    MEM0_LOCAL_AVAILABLE = True
except ImportError:
    MEM0_LOCAL_AVAILABLE = False

MEM0_AVAILABLE = MEM0_PLATFORM_AVAILABLE or MEM0_LOCAL_AVAILABLE


class DocumentNode:
    """A node in the document tree structure."""
    
    def __init__(
        self,
        node_id: str,
        title: str,
        content: str = "",
        summary: str = "",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.title = title
        self.content = content
        self.summary = summary
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.children: List["DocumentNode"] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentNode":
        node = cls(
            node_id=data["node_id"],
            title=data["title"],
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {})
        )
        for child_data in data.get("children", []):
            node.children.append(cls.from_dict(child_data))
        return node


class DocumentTree:
    """
    Hierarchical document tree for reasoning-based retrieval.
    
    Like PageIndex, this creates a "table of contents" structure
    that enables LLMs to reason their way to relevant sections.
    """
    
    def __init__(self, doc_id: str, title: str):
        self.doc_id = doc_id
        self.title = title
        self.root = DocumentNode(
            node_id="root",
            title=title,
            summary=f"Root of document: {title}"
        )
        self.node_index: Dict[str, DocumentNode] = {"root": self.root}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def add_node(
        self,
        node_id: Optional[str],
        title: str,
        content: str,
        summary: str = "",
        parent_id: str = "root",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a node to the tree."""
        node_id = str(node_id or str(uuid4())[:8])
        if node_id in self.node_index:
            base = node_id
            i = 1
            while f"{base}_{i}" in self.node_index:
                i += 1
            node_id = f"{base}_{i}"
        parent = self.node_index.get(parent_id, self.root)
        
        node = DocumentNode(
            node_id=node_id,
            title=title,
            content=content,
            summary=summary or content[:200],
            parent_id=parent_id,
            metadata=metadata
        )
        
        parent.children.append(node)
        self.node_index[node_id] = node
        self.updated_at = datetime.now().isoformat()
        
        return node_id
    
    def get_node(self, node_id: str) -> Optional[DocumentNode]:
        """Get a node by ID."""
        return self.node_index.get(node_id)
    
    def get_toc(self, max_depth: int = 3) -> str:
        """
        Generate a table of contents for LLM reasoning.
        
        Returns a structured view that the LLM can reason over
        to navigate to relevant sections.
        """
        lines = [f"# {self.title}\n"]
        
        def traverse(node: DocumentNode, depth: int):
            if depth > max_depth:
                return
            indent = "  " * depth
            lines.append(f"{indent}- [{node.node_id}] {node.title}")
            if node.summary:
                lines.append(f"{indent}  Summary: {node.summary[:100]}...")
            for child in node.children:
                traverse(child, depth + 1)
        
        for child in self.root.children:
            traverse(child, 0)
        
        return "\n".join(lines)
    
    def get_path_to_node(self, node_id: str) -> List[str]:
        """Get the path from root to a node."""
        path = []
        current = self.node_index.get(node_id)
        
        while current:
            path.insert(0, current.node_id)
            current = self.node_index.get(current.parent_id) if current.parent_id else None
        
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "root": self.root.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentTree":
        tree = cls(data["doc_id"], data["title"])
        tree.root = DocumentNode.from_dict(data["root"])
        tree.created_at = data.get("created_at", tree.created_at)
        tree.updated_at = data.get("updated_at", tree.updated_at)
        
        # Rebuild index
        def index_nodes(node: DocumentNode):
            tree.node_index[node.node_id] = node
            for child in node.children:
                index_nodes(child)
        
        index_nodes(tree.root)
        return tree


class AgenticRAG:
    """
    Agentic RAG system with reasoning-based retrieval.
    
    Combines:
    - PageIndex-style hierarchical tree indexing
    - Mem0 for infinite memory and fact extraction
    - ChromaDB for semantic search backup
    - LLM-based reasoning for context selection
    
    This provides better context retrieval than pure vector search
    by using the LLM to reason about document structure and relevance.
    """
    
    def __init__(
        self,
        user_id: str = "default",
        persist_dir: str = "data/agentic_rag",
        llm_provider: Optional[Any] = None,
        vector_store: Optional["VectorStore"] = None,
        long_term: Optional["LongTermMemory"] = None,
    ):
        """
        Initialize Agentic RAG.
        
        Args:
            user_id: User identifier for memory isolation
            persist_dir: Directory for persistent storage
            llm_provider: LLM provider for reasoning
        """
        self.user_id = user_id
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_provider = llm_provider
        self._vector_store = vector_store
        self._long_term = long_term
        
        # Document trees for hierarchical indexing
        self.document_trees: Dict[str, DocumentTree] = {}
        
        # Session memory
        self.session_memories: List[Dict[str, Any]] = []
        
        # Parsed tool docs (from persona/TOOLS.md)
        self._tool_docs: Optional[ToolDocsIndex] = None

        # PageIndex local cache + lazy API wrapper (PDF ingestion)
        self._pageindex_store = PageIndexStore()
        self._pageindex_api: Optional[PageIndexAPI] = None
        self._pageindex_cfg: Dict[str, Any] = {}
        self._max_nodes_per_doc = 3
        self._redaction_mode = "skip"  # "skip" | "redact"
        
        # Optional Mem0 for legacy setups (disabled by default)
        self._mem0: Optional[Any] = None
        
        self._initialized = False
        self._mem0_is_platform = False  # Track if using Platform API
        
        logger.debug("AgenticRAG created")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the RAG system."""
        logger.info("Initializing Agentic RAG system...")
        
        config = config or {}
        self._pageindex_cfg = config.get("pageindex", {}) if isinstance(config, dict) else {}
        self._max_nodes_per_doc = int(self._pageindex_cfg.get("max_nodes_per_doc", config.get("max_nodes_per_doc", 3)))
        redaction_cfg = config.get("redaction", {}) if isinstance(config, dict) else {}
        self._redaction_mode = str(redaction_cfg.get("on_detect", "skip")).lower()
        
        # Load persisted document trees
        await self._load_trees()
        
        # Mem0 is optional and OFF by default in this architecture
        mem0_config = config.get("mem0", {}) if isinstance(config, dict) else {}
        mem0_enabled = bool(mem0_config.get("enabled", False))
        if mem0_enabled:
            # Priority: 1) Platform API (MEM0_API_KEY), 2) Local with OpenAI
            mem0_api_key = os.environ.get("MEM0_API_KEY")

            if mem0_api_key and MEM0_PLATFORM_AVAILABLE:
                try:
                    self._mem0 = MemoryClient(api_key=mem0_api_key)
                    self._mem0_is_platform = True
                    logger.info("Mem0 Platform API initialized (cloud memory enabled)")
                except Exception as e:
                    logger.warning(f"Mem0 Platform initialization failed: {e}")

            elif MEM0_LOCAL_AVAILABLE:
                try:
                    mem0_settings = {
                        "llm": {
                            "provider": mem0_config.get("llm_provider", "openai"),
                            "config": {
                                "model": mem0_config.get("model", "gpt-4o-mini"),
                                "temperature": 0.1,
                            },
                        },
                        "embedder": {
                            "provider": mem0_config.get("embedder_provider", "openai"),
                            "config": {
                                "model": mem0_config.get(
                                    "embedder_model", "text-embedding-3-small"
                                ),
                            },
                        },
                        "vector_store": {
                            "provider": "chroma",
                            "config": {
                                "collection_name": f"intelclaw_rag_{self.user_id}",
                                "path": str(self.persist_dir / "mem0_vectors"),
                            },
                        },
                        "version": "v1.1",
                    }

                    self._mem0 = Memory.from_config(mem0_settings)
                    self._mem0_is_platform = False
                    logger.info("Mem0 Local initialized for Agentic RAG")
                except Exception as e:
                    logger.warning(f"Mem0 Local initialization failed: {e}")
        
        self._initialized = True
        logger.success("Agentic RAG system initialized")
    
    async def shutdown(self) -> None:
        """Shutdown and persist state."""
        logger.info("Shutting down Agentic RAG...")
        await self._save_trees()
        self._initialized = False
    
    # ==================== Mem0 Helper Methods ====================
    
    async def _mem0_add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add content to Mem0 (works with both Platform and Local API)."""
        if not self._mem0:
            return
        
        try:
            if self._mem0_is_platform:
                # Platform API: add(messages, user_id=..., metadata=...)
                # Messages should be a list of dicts with role/content, or a string
                messages = [{"role": "user", "content": content}]
                await asyncio.to_thread(
                    self._mem0.add,
                    messages,
                    user_id=self.user_id,
                    metadata=metadata or {}
                )
            else:
                # Local API: add(data, user_id, metadata)
                await asyncio.to_thread(
                    self._mem0.add,
                    content,
                    user_id=self.user_id,
                    metadata=metadata or {}
                )
        except Exception as e:
            logger.debug(f"Mem0 add failed: {e}")
    
    async def _mem0_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Mem0 (works with both Platform and Local API)."""
        if not self._mem0:
            return []
        
        try:
            if self._mem0_is_platform:
                # Platform API: search(query, filters={"user_id": ...}, limit=...)
                results = await asyncio.to_thread(
                    self._mem0.search,
                    query,
                    filters={"user_id": self.user_id},
                    limit=limit
                )
                # Platform API returns dict with "results" key
                results_list = results.get("results", []) if isinstance(results, dict) else results
                return [
                    {
                        "id": r.get("id", str(uuid4())),
                        "content": r.get("memory", ""),
                        "score": r.get("score", 0.5),
                        "source": "mem0",
                        "metadata": r.get("metadata", {})
                    }
                    for r in results_list
                ]
            else:
                # Local API returns dict with 'results' key
                results = await asyncio.to_thread(
                    self._mem0.search,
                    query,
                    user_id=self.user_id,
                    limit=limit
                )
                results_list = results.get("results", []) if isinstance(results, dict) else results
                return [
                    {
                        "id": r.get("id", str(uuid4())),
                        "content": r.get("memory", r.get("text", "")),
                        "score": r.get("score", 0.5),
                        "source": "mem0",
                        "metadata": r.get("metadata", {})
                    }
                    for r in results_list
                ]
        except Exception as e:
            logger.debug(f"Mem0 search failed: {e}")
        
        return []
    
    async def store_user_preference(
        self,
        preference_type: str,
        preference_value: str,
        context: Optional[str] = None
    ) -> None:
        """
        Store user preference in Mem0 for future reference.
        
        This is used to remember user preferences, habits, and important
        information that should persist across sessions.
        
        Args:
            preference_type: Type of preference (e.g., 'coding_style', 'language', 'tool')
            preference_value: The preference value
            context: Additional context about when/why this preference was learned
        """
        memory_content = f"User preference - {preference_type}: {preference_value}"
        if context:
            memory_content += f"\nContext: {context}"

        # Prefer local infinite memory backend when available
        if self._long_term:
            await self._long_term.add(
                memory_content,
                metadata={
                    "type": "user_preference",
                    "kind": "user_preference",
                    "preference_type": preference_type,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.info(f"Stored user preference locally: {preference_type}")
            return

        # Legacy Mem0 fallback (only if enabled/initialized)
        if self._mem0:
            await self._mem0_add(
                memory_content,
                metadata={
                    "type": "user_preference",
                    "preference_type": preference_type,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.info(f"Stored user preference in Mem0: {preference_type}")
            return

        # File fallback (rare)
        prefs_file = self.persist_dir / "user_preferences.json"
        prefs: Dict[str, Any] = {}
        if prefs_file.exists():
            prefs = json.loads(prefs_file.read_text())

        prefs[preference_type] = {
            "value": preference_value,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        prefs_file.write_text(json.dumps(prefs, indent=2))
        logger.info(f"Stored user preference locally (file): {preference_type}")
    
    async def get_user_preferences(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve user preferences from Mem0.
        
        Args:
            query: Optional query to filter preferences
            
        Returns:
            List of relevant user preferences
        """
        # Prefer local infinite memory backend when available
        if self._long_term:
            search_query = query or "user preference"
            results = await self._long_term.search(search_query, limit=25)
            return [
                {
                    "id": r.get("id", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "source": "long_term",
                    "metadata": r.get("metadata", {}),
                }
                for r in results
                if (r.get("metadata") or {}).get("type") == "user_preference"
                or (r.get("metadata") or {}).get("kind") == "user_preference"
                or "user preference" in str(r.get("content") or "").lower()
            ]

        if not self._mem0:
            # File fallback
            prefs_file = self.persist_dir / "user_preferences.json"
            if prefs_file.exists():
                prefs = json.loads(prefs_file.read_text())
                return [{"type": k, **v} for k, v in prefs.items()]
            return []
        
        # Search Mem0 for preferences
        search_query = query or "user preference"
        results = await self._mem0_search(search_query, limit=20)
        
        # Filter for preference-type memories
        preferences = [
            r for r in results
            if r.get("metadata", {}).get("type") == "user_preference"
            or "preference" in r.get("content", "").lower()
        ]
        
        return preferences

    # ==================== Document Tree Operations ====================

    async def index_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        doc_type: str = "generic"
    ) -> DocumentTree:
        """
        Create a hierarchical tree index for a document.
        
        This is the PageIndex-style indexing that creates a
        navigable structure for LLM reasoning.
        """
        logger.info(f"Indexing document: {title}")
        
        tree = DocumentTree(doc_id, title)
        tree.root.metadata.update(
            {
                "doc_type": doc_type,
                "backend": "local",
                "doc_id": doc_id,
                "title": title,
            }
        )
        
        # Parse content into a hierarchical tree (markdown heading stack)
        self._index_markdown_into_tree(tree, content, base_metadata={"doc_type": doc_type, "backend": "local"})
        
        self.document_trees[doc_id] = tree

        # Best-effort semantic indexing into the shared VectorStore
        await self._index_tree_into_vector_store(tree, doc_id=doc_id, doc_type=doc_type, backend="local")
        
        # Persist
        await self._save_trees()
        
        return tree

    async def index_path(
        self,
        path: Path,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Index a local file into the RAG system.

        - PDFs: indexed via Hosted PageIndex (tree cached locally for retrieval).
        - Text/Markdown: indexed locally via hierarchical markdown parsing.

        Returns:
            Document ID if indexed, otherwise None.
        """
        if not self._initialized:
            return None

        try:
            if not path.exists() or not path.is_file():
                return None

            suffix = path.suffix.lower()
            if suffix == ".pdf":
                return await self._index_pdf_with_pageindex(path, title=title, tags=tags)

            if suffix in {".md", ".txt"}:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="ignore")
                doc_id = f"file::{path.resolve().as_posix()}"
                await self.index_document(
                    doc_id=doc_id,
                    title=title or path.name,
                    content=content,
                    doc_type="file",
                )
                return doc_id

        except Exception as e:
            logger.warning(f"Failed to index path {path}: {e}")

        return None

    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Return lightweight info for a known doc_id (best-effort)."""
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return {}

        tree = self.document_trees.get(doc_id)
        if tree:
            meta = dict(tree.root.metadata or {})
            info: Dict[str, Any] = {
                "doc_id": doc_id,
                "title": tree.title,
                "doc_type": meta.get("doc_type"),
                "backend": meta.get("backend"),
                "local_path": meta.get("local_path"),
                "page_num": meta.get("page_num"),
                "updated_at": tree.updated_at,
            }

            if meta.get("backend") == "pageindex" and meta.get("local_path"):
                try:
                    reg = self._pageindex_store.get_by_path(str(meta.get("local_path")))
                    if reg:
                        info["status"] = reg.get("status")
                        info["description"] = reg.get("description")
                        info["name"] = reg.get("name")
                except Exception:
                    pass

            return info

        # If the tree isn't loaded, fall back to PageIndex registry if present.
        for entry in self._pageindex_store.list_docs():
            if str(entry.get("doc_id")) == doc_id:
                return {
                    "doc_id": doc_id,
                    "title": entry.get("name") or doc_id,
                    "doc_type": "pdf",
                    "backend": "pageindex",
                    "local_path": entry.get("local_path"),
                    "status": entry.get("status"),
                    "updated_at": entry.get("updated_at"),
                    "page_num": entry.get("page_num"),
                    "description": entry.get("description"),
                }

        return {"doc_id": doc_id}

    def list_documents(self, include_persona: bool = False) -> List[Dict[str, Any]]:
        """List indexed documents (excluding persona by default)."""
        docs: List[Dict[str, Any]] = []

        for doc_id, tree in self.document_trees.items():
            meta = dict(tree.root.metadata or {})
            if not include_persona:
                if meta.get("doc_type") == "persona" or str(doc_id).startswith("persona::"):
                    continue

            docs.append(
                {
                    "doc_id": doc_id,
                    "title": tree.title,
                    "doc_type": meta.get("doc_type"),
                    "backend": meta.get("backend"),
                    "local_path": meta.get("local_path"),
                    "updated_at": tree.updated_at,
                }
            )

        # Include PageIndex registry entries that may not be loaded in-memory
        loaded_ids = {d.get("doc_id") for d in docs}
        for entry in self._pageindex_store.list_docs():
            did = str(entry.get("doc_id") or "").strip()
            if not did or did in loaded_ids:
                continue
            docs.append(
                {
                    "doc_id": did,
                    "title": entry.get("name") or did,
                    "doc_type": "pdf",
                    "backend": "pageindex",
                    "local_path": entry.get("local_path"),
                    "status": entry.get("status"),
                    "updated_at": entry.get("updated_at"),
                }
            )

        docs.sort(key=lambda d: str(d.get("updated_at") or ""), reverse=True)
        return docs

    async def delete_document(self, doc_id: str, delete_remote: bool = False) -> Dict[str, Any]:
        """
        Delete an indexed document by doc_id.

        - Removes local cached tree/index entries.
        - If PageIndex-backed and delete_remote=True, also deletes remote document (best-effort).
        """
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return {"deleted": False, "error": "doc_id is required"}

        tree = self.document_trees.pop(doc_id, None)
        backend = (tree.root.metadata or {}).get("backend") if tree else None

        # Detect PageIndex doc even if the tree is not loaded
        pageindex_entry: Optional[Dict[str, Any]] = None
        try:
            for entry in self._pageindex_store.list_docs():
                if str(entry.get("doc_id") or "") == doc_id:
                    pageindex_entry = entry
                    break
        except Exception:
            pageindex_entry = None

        is_pageindex = backend == "pageindex" or pageindex_entry is not None or self._pageindex_store.load_tree(doc_id) is not None
        if is_pageindex and backend is None:
            backend = "pageindex"

        # Best-effort delete from shared vector store
        vector_deleted = 0
        if self._vector_store and getattr(self._vector_store, "is_available", False):
            try:
                vector_deleted = await self._vector_store.delete_where({"doc_id": doc_id})
            except Exception:
                vector_deleted = 0

        # Delete cached PageIndex artifacts if applicable
        cache_deleted = False
        if is_pageindex:
            try:
                cache_deleted = self._pageindex_store.delete_cached(doc_id)
            except Exception:
                cache_deleted = False

        # Optional remote delete (explicit only)
        remote_deleted = False
        remote_error: Optional[str] = None
        if delete_remote and is_pageindex:
            api = self._get_pageindex_api()
            if not api:
                remote_error = "PAGEINDEX_API_KEY not set or PageIndex unavailable"
            else:
                try:
                    api.delete_document(doc_id)
                    remote_deleted = True
                except Exception as e:
                    remote_error = str(e)

        await self._save_trees()

        return {
            "deleted": bool(tree) or cache_deleted or vector_deleted > 0 or remote_deleted,
            "doc_id": doc_id,
            "backend": backend,
            "cache_deleted": cache_deleted,
            "vector_deleted": vector_deleted,
            "remote_deleted": remote_deleted,
            "remote_error": remote_error,
        }

    def _get_pageindex_api(self) -> Optional[PageIndexAPI]:
        if self._pageindex_api is not None:
            return self._pageindex_api
        self._pageindex_api = PageIndexAPI.from_env()
        return self._pageindex_api

    def _extract_pageindex_tree(self, payload: Any) -> Any:
        """
        PageIndex SDK returns a dict with a status and a tree payload.
        This helper attempts to locate the actual tree object.
        """
        if not isinstance(payload, dict):
            return payload
        if "tree" in payload:
            return payload.get("tree")
        if "data" in payload and isinstance(payload.get("data"), dict) and "tree" in payload["data"]:
            return payload["data"].get("tree")
        return payload

    def _pageindex_tree_to_document_tree(
        self,
        doc_id: str,
        title: str,
        tree_payload: Any,
        local_path: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> DocumentTree:
        tree = DocumentTree(doc_id, title)
        tree.root.metadata.update(
            {
                "doc_type": "pdf",
                "backend": "pageindex",
                "doc_id": doc_id,
                "title": title,
                "local_path": local_path,
                "page_num": (meta or {}).get("pageNum") or (meta or {}).get("page_num"),
            }
        )

        root_obj = self._extract_pageindex_tree(tree_payload)
        base_meta = {"doc_type": "pdf", "backend": "pageindex", "local_path": local_path}

        def children(obj: Any) -> List[Any]:
            if isinstance(obj, dict):
                if isinstance(obj.get("nodes"), list):
                    return obj["nodes"]
                if isinstance(obj.get("children"), list):
                    return obj["children"]
            if isinstance(obj, list):
                return obj
            return []

        def title_for(obj: Dict[str, Any]) -> str:
            for k in ("title", "name", "heading"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            nid = obj.get("node_id") or obj.get("id")
            return f"Node {nid}" if nid is not None else "Node"

        def text_for(obj: Dict[str, Any]) -> str:
            v = obj.get("text") or obj.get("content") or ""
            if isinstance(v, list):
                v = "\n".join(str(x) for x in v)
            if not isinstance(v, str):
                v = str(v)
            v = v.strip()
            return v[:10000] if len(v) > 10000 else v

        def summary_for(obj: Dict[str, Any], text: str) -> str:
            v = obj.get("summary") or obj.get("node_summary") or ""
            if isinstance(v, str) and v.strip():
                return v.strip()
            return text[:200]

        def recurse(obj: Any, parent_id: str) -> None:
            if not isinstance(obj, dict):
                for c in children(obj):
                    recurse(c, parent_id)
                return

            nid = obj.get("node_id") or obj.get("id")
            node_id = str(nid) if nid is not None else None
            node_title = title_for(obj)
            node_text = text_for(obj)
            node_summary = summary_for(obj, node_text)

            page_index = obj.get("page_index")
            node_meta = {**base_meta, "page_index": page_index}
            new_id = tree.add_node(
                node_id=node_id,
                title=node_title,
                content=node_text,
                summary=node_summary,
                parent_id=parent_id,
                metadata=node_meta,
            )

            for c in children(obj):
                recurse(c, new_id)

        # Root payload can be a dict or list of nodes
        for c in children(root_obj):
            recurse(c, "root")

        return tree

    async def _index_pdf_with_pageindex(
        self,
        path: Path,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        api = self._get_pageindex_api()
        if not api:
            logger.warning("PAGEINDEX_API_KEY not set or PageIndex SDK unavailable; skipping PDF indexing")
            return None

        resolved = str(path.resolve())
        try:
            file_hash = sha256_file(path)
        except Exception as e:
            logger.warning(f"Failed to hash PDF for PageIndex indexing: {e}")
            return None

        existing = self._pageindex_store.get_by_path(resolved)
        existing_doc_id = str(existing.get("doc_id")) if existing and existing.get("doc_id") else ""
        existing_ok = bool(
            existing
            and existing.get("file_sha256") == file_hash
            and str(existing.get("status", "")).lower() == "completed"
            and existing_doc_id
        )

        if existing_ok:
            if existing_doc_id not in self.document_trees:
                cached = self._pageindex_store.load_tree(existing_doc_id)
                if cached:
                    dtree = self._pageindex_tree_to_document_tree(
                        doc_id=existing_doc_id,
                        title=title or str(existing.get("name") or path.name),
                        tree_payload=cached,
                        local_path=resolved,
                        meta=None,
                    )
                    self.document_trees[existing_doc_id] = dtree
                    await self._index_tree_into_vector_store(
                        dtree, doc_id=existing_doc_id, doc_type="pdf", backend="pageindex"
                    )
                    await self._save_trees()
            return existing_doc_id

        # Submit + wait + tree fetch
        try:
            doc_id = api.submit_pdf(str(path))
            meta = api.wait_for_completed(
                doc_id,
                timeout_s=int(self._pageindex_cfg.get("tree_timeout_seconds", 900)),
                poll_s=float(self._pageindex_cfg.get("poll_seconds", 5)),
            )
            tree_payload = api.get_tree(doc_id, node_summary=True)
        except (PageIndexUnavailableError, PageIndexProcessingError, TimeoutError) as e:
            logger.warning(f"PageIndex indexing failed for {path}: {e}")
            return None

        try:
            self._pageindex_store.save_tree(doc_id, tree_payload)
        except Exception as e:
            logger.warning(f"Failed to cache PageIndex tree for doc_id={doc_id}: {e}")

        # Update registry (data/pageindex/registry.json)
        entry = {
            "local_path": resolved,
            "file_sha256": file_hash,
            "doc_id": doc_id,
            "name": meta.get("name") if isinstance(meta, dict) else path.name,
            "description": meta.get("description") if isinstance(meta, dict) else "",
            "status": meta.get("status") if isinstance(meta, dict) else "completed",
            "created_at": meta.get("createdAt") if isinstance(meta, dict) else None,
            "page_num": meta.get("pageNum") if isinstance(meta, dict) else None,
            "tags": tags or [],
        }
        try:
            self._pageindex_store.upsert_by_path(entry)
        except Exception:
            pass

        # Index into local tree store for query-time retrieval
        dtree = self._pageindex_tree_to_document_tree(
            doc_id=doc_id,
            title=title or str(meta.get("name") or path.name),
            tree_payload=tree_payload,
            local_path=resolved,
            meta=meta if isinstance(meta, dict) else None,
        )
        self.document_trees[doc_id] = dtree
        await self._index_tree_into_vector_store(dtree, doc_id=doc_id, doc_type="pdf", backend="pageindex")
        await self._save_trees()

        return doc_id
    
    def _index_markdown_into_tree(
        self,
        tree: DocumentTree,
        content: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index markdown text into a DocumentTree using heading stack parent assignment."""
        base_metadata = base_metadata or {}
        heading_stack: List[Tuple[int, str]] = []

        intro_lines: List[str] = []
        current_node_id: Optional[str] = None
        current_lines: List[str] = []

        heading_re = re.compile(r"^(#{1,6})\\s+(.+?)\\s*$")

        def finalize(node_id: str, lines: List[str]) -> None:
            node = tree.get_node(node_id)
            if not node:
                return
            text = "\\n".join(lines).strip()
            node.content = text
            if not node.summary:
                node.summary = text[:200] if text else node.summary

        for line in (content or "").splitlines():
            m = heading_re.match(line)
            if m:
                if current_node_id is not None:
                    finalize(current_node_id, current_lines)
                else:
                    # Flush introduction as soon as we hit the first header
                    if any(l.strip() for l in intro_lines):
                        intro_text = "\\n".join(intro_lines).strip()
                        tree.add_node(
                            node_id=None,
                            title="Introduction",
                            content=intro_text,
                            summary=intro_text[:200],
                            parent_id="root",
                            metadata={**base_metadata, "section_level": 0},
                        )
                    intro_lines = []

                level = len(m.group(1))
                title = m.group(2).strip()

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                parent_id = heading_stack[-1][1] if heading_stack else "root"

                node_id = tree.add_node(
                    node_id=None,
                    title=title,
                    content="",
                    summary="",
                    parent_id=parent_id,
                    metadata={**base_metadata, "section_level": level},
                )
                heading_stack.append((level, node_id))
                current_node_id = node_id
                current_lines = []
                continue

            if current_node_id is None:
                intro_lines.append(line)
            else:
                current_lines.append(line)

        if current_node_id is not None:
            finalize(current_node_id, current_lines)
        elif any(l.strip() for l in intro_lines):
            intro_text = "\\n".join(intro_lines).strip()
            tree.add_node(
                node_id=None,
                title="Introduction",
                content=intro_text,
                summary=intro_text[:200],
                parent_id="root",
                metadata={**base_metadata, "section_level": 0},
            )

    async def _index_tree_into_vector_store(
        self,
        tree: DocumentTree,
        doc_id: str,
        doc_type: str,
        backend: str,
    ) -> None:
        if not self._vector_store or not getattr(self._vector_store, "is_available", False):
            return

        documents: List[Dict[str, Any]] = []
        for node_id, node in tree.node_index.items():
            if node_id == "root":
                continue
            text = (node.content or node.summary or "").strip()
            if not text:
                continue
            documents.append(
                {
                    "id": f"{doc_id}:{node_id}",
                    "content": text[:2000],
                    "metadata": {
                        "backend": backend,
                        "doc_id": doc_id,
                        "node_id": node_id,
                        "doc_type": doc_type,
                        "title": node.title,
                    },
                }
            )

        if not documents:
            return

        try:
            await self._vector_store.add_batch(documents)
        except Exception:
            # Semantic indexing is best-effort; retrieval will still work via lexical/tree matching.
            return
    
    async def index_persona(self, persona_dir: Path) -> None:
        """
        Index persona files for context-aware retrieval.
        
        Indexes all `*.md` files under the persona directory.
        """
        logger.info(f"Indexing persona from: {persona_dir}")

        for filepath in sorted(persona_dir.glob("*.md")):
            if not filepath.exists():
                continue

            try:
                content = filepath.read_text(encoding="utf-8")

                if filepath.name.upper() == "TOOLS.MD":
                    self._tool_docs = ToolDocsIndex.parse_markdown(content)

                # Create tree index for retrieval (snippets only; full persona is injected via system prompt)
                doc_id = f"persona::{filepath.name}"
                await self.index_document(
                    doc_id=doc_id,
                    title=filepath.name,
                    content=content,
                    doc_type="persona",
                )

                logger.debug(f"Indexed persona file: {filepath.name}")

            except Exception as e:
                logger.warning(f"Failed to index persona file {filepath.name}: {e}")
    
    # ==================== Reasoning-Based Retrieval ====================
    
    async def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        max_results: int = 5,
        doc_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context using reasoning-based search.
        
        Strategies:
        - "tree": Use document tree traversal with LLM reasoning
        - "semantic": Use vector similarity (fallback)
        - "hybrid": Combine tree reasoning with semantic backup
        
        Args:
            query: The search query
            strategy: Retrieval strategy
            max_results: Maximum results to return
            doc_types: Filter by document types
            
        Returns:
            List of relevant context items with scores
        """
        if not self._initialized:
            return []

        all_results: List[Dict[str, Any]] = []

        if strategy in ["tree", "hybrid"]:
            all_results.extend(await self._tree_retrieve(query, doc_types))

        if strategy in ["semantic", "hybrid"]:
            all_results.extend(await self._semantic_retrieve(query, max_results, doc_types=doc_types))

        # Local "infinite memory" (SQLite) – best-effort
        if self._long_term:
            try:
                ltm_results = await self._long_term.search(query, limit=max_results)
                for r in ltm_results:
                    meta = r.get("metadata") or {}
                    kind = str(meta.get("kind") or meta.get("type") or "memory")
                    all_results.append(
                        {
                            "id": f"ltm:{r.get('id', '')}",
                            "content": r.get("content", ""),
                            "title": kind,
                            "score": float(r.get("score", 0.5)),
                            "source": "long_term",
                            "metadata": meta,
                        }
                    )
            except Exception as e:
                logger.debug(f"Long-term memory search unavailable: {e}")

        # Optional Mem0 search (legacy setups only)
        if self._mem0:
            mem0_results = await self._mem0_search(query, limit=max_results)
            for r in mem0_results:
                r = dict(r)
                r["id"] = f"mem0:{r.get('id', '')}"
                all_results.append(r)

        # Deduplicate by id; keep the highest score
        dedup: Dict[str, Dict[str, Any]] = {}
        for r in all_results:
            rid = str(r.get("id") or "")
            if not rid:
                continue
            score = float(r.get("score", 0.0) or 0.0)
            existing = dedup.get(rid)
            if not existing or score > float(existing.get("score", 0.0) or 0.0):
                dedup[rid] = r

        results = list(dedup.values())
        results.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return results[:max_results]
    
    async def _tree_retrieve(
        self,
        query: str,
        doc_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using tree traversal.
        
        This mimics how humans navigate documents:
        1. Look at table of contents
        2. Identify relevant sections
        3. Drill down into specific content
        """
        results: List[Dict[str, Any]] = []
        query_lower = (query or "").lower()
        q_tokens = {t for t in re.findall(r"[a-z0-9]+", query_lower) if len(t) > 2}
        if not q_tokens and not query_lower.strip():
            return results
        
        for doc_id, tree in self.document_trees.items():
            # Filter by doc_type if specified
            if doc_types:
                root_meta = tree.root.metadata
                if root_meta.get("doc_type") not in doc_types:
                    continue

            per_doc: List[Dict[str, Any]] = []
            for node_id, node in tree.node_index.items():
                if node_id == "root":
                    continue

                hay = f"{node.title}\n{node.summary}\n{node.content}".lower()
                n_tokens = {t for t in re.findall(r"[a-z0-9]+", hay) if len(t) > 2}

                overlap = 0.0
                if q_tokens:
                    overlap = len(q_tokens & n_tokens) / max(len(q_tokens), 1)

                substring_bonus = 0.2 if query_lower.strip() and query_lower in hay else 0.0
                title_bonus = 0.2 if any(t in node.title.lower() for t in q_tokens) else 0.0

                score = min(overlap + substring_bonus + title_bonus, 1.0)
                if score <= 0.0:
                    continue

                per_doc.append(
                    {
                        "id": f"{doc_id}:{node_id}",
                        "content": node.content or node.summary,
                        "title": node.title,
                        "score": score,
                        "source": "tree",
                        "doc_id": doc_id,
                        "node_id": node_id,
                        "path": tree.get_path_to_node(node_id),
                        "doc_type": tree.root.metadata.get("doc_type"),
                        "metadata": node.metadata or {},
                    }
                )

            per_doc.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            results.extend(per_doc[: self._max_nodes_per_doc])
        
        return results
    
    async def _semantic_retrieve(
        self,
        query: str,
        max_results: int = 5,
        doc_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic vector search fallback."""
        if not self._vector_store or not getattr(self._vector_store, "is_available", False):
            return []

        try:
            docs = await self._vector_store.search(query, limit=max_results)
        except Exception as e:
            logger.debug(f"Semantic search unavailable: {e}")
            return []

        results: List[Dict[str, Any]] = []
        for d in docs or []:
            meta = d.get("metadata") or {}
            if doc_types and meta.get("doc_type") not in doc_types:
                continue
            results.append(
                {
                    "id": d.get("id", ""),
                    "content": d.get("content", ""),
                    "score": float(d.get("score", 0.5) or 0.5),
                    "source": "semantic",
                    "metadata": meta,
                    "doc_id": meta.get("doc_id"),
                    "node_id": meta.get("node_id"),
                    "title": meta.get("title", ""),
                    "doc_type": meta.get("doc_type"),
                }
            )

        return results[:max_results]
    
    # ==================== Session Memory ====================
    
    async def store_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store session conversation for future context."""
        safe_messages: List[Dict[str, Any]] = []
        secret_detected = False
        for m in messages:
            role = str(m.get("role", "unknown"))
            content = str(m.get("content", ""))

            if contains_secret(content):
                secret_detected = True
                if self._redaction_mode == "redact":
                    content = redact_secrets(content)
                else:
                    # Default: do not persist secrets (even in session logs)
                    return

            safe_messages.append({**m, "role": role, "content": content})

        session_data = {
            "session_id": session_id,
            "messages": safe_messages,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id
        }
        
        self.session_memories.append(session_data)
        if secret_detected and self._redaction_mode == "redact":
            logger.debug(f"Stored redacted session (session_id={session_id})")
        
        # Save to disk
        session_file = self.persist_dir / "sessions" / f"{session_id}.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(json.dumps(session_data, indent=2))
    
    async def get_session_context(
        self,
        query: str,
        max_sessions: int = 3
    ) -> List[Dict[str, Any]]:
        """Get relevant past session context for a query."""
        relevant_sessions = []
        
        # Simple keyword matching (in full implementation, use LLM reasoning)
        query_lower = query.lower()
        
        for session in self.session_memories[-50:]:  # Check recent sessions
            messages_text = " ".join([
                m.get("content", "") for m in session.get("messages", [])
            ]).lower()
            
            if any(word in messages_text for word in query_lower.split()):
                relevant_sessions.append(session)
        
        return relevant_sessions[:max_sessions]
    
    # ==================== Persistence ====================
    
    async def _save_trees(self) -> None:
        """Save document trees to disk."""
        trees_file = self.persist_dir / "document_trees.json"
        
        try:
            data = {
                doc_id: tree.to_dict()
                for doc_id, tree in self.document_trees.items()
            }
            trees_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save document trees: {e}")
    
    async def _load_trees(self) -> None:
        """Load document trees from disk."""
        trees_file = self.persist_dir / "document_trees.json"
        
        if not trees_file.exists():
            return
        
        try:
            data = json.loads(trees_file.read_text())
            
            for doc_id, tree_data in data.items():
                self.document_trees[doc_id] = DocumentTree.from_dict(tree_data)
            
            logger.info(f"Loaded {len(self.document_trees)} document trees")
            
        except Exception as e:
            logger.error(f"Failed to load document trees: {e}")
    
    # ==================== RAG Context Building ====================
    
    async def build_context(
        self,
        query: str,
        include_persona: bool = True,
        include_session: bool = True,
        max_context_chars: int = 10000,
        available_tool_names: Optional[set[str]] = None,
    ) -> str:
        """
        Build comprehensive context for a query.
        
        This is the main entry point for the RAG system.
        Returns a formatted context string that can be injected
        into the system prompt.
        """
        context_parts: List[str] = []
        total_chars = 0

        def add(text: str) -> None:
            nonlocal total_chars
            if not text or total_chars >= max_context_chars:
                return
            remaining = max_context_chars - total_chars
            if len(text) > remaining:
                text = text[:remaining]
            context_parts.append(text)
            total_chars += len(text)

        results = await self.retrieve(query, strategy="hybrid", max_results=8)

        # Optionally exclude persona docs at query-time to avoid token bloat
        if not include_persona:
            results = [
                r
                for r in results
                if r.get("doc_type") != "persona" and not str(r.get("doc_id") or "").startswith("persona::")
            ]

        doc_results = [r for r in results if r.get("source") in {"tree", "semantic"} and r.get("content")]

        if doc_results:
            add("## Retrieved Context\n")
            for r in doc_results[:5]:
                meta = r.get("metadata") or {}
                source = r.get("source", "unknown")
                title = str(r.get("title") or meta.get("title") or "Context")
                doc_id = r.get("doc_id") or meta.get("doc_id")
                node_id = r.get("node_id") or meta.get("node_id")
                page = meta.get("page_index") or meta.get("page")

                header = f"### {title} (source={source}"
                if doc_id:
                    header += f", doc={doc_id}"
                if node_id:
                    header += f", node={node_id}"
                if page is not None:
                    header += f", page={page}"
                header += ")\n"

                add(header)

                snippet = str(r.get("content") or "").strip()
                if len(snippet) > 1500:
                    snippet = snippet[:1500] + "..."
                add(snippet + "\n\n")

        # Tool hints from persona/TOOLS.md (structured parse)
        if self._tool_docs:
            hints = self._tool_docs.search(query, top_k=3)
            if hints:
                add("## Tool Hints (from TOOLS.md)\n")
                for h in hints:
                    not_registered = (
                        available_tool_names is not None and h.tool_name not in available_tool_names
                    )
                    reg_note = " (not registered)" if not_registered else ""
                    add(f"- {h.tool_name}{reg_note}: {h.signature}\n")
                    if h.section:
                        add(f"  Section: {h.section}\n")
                    if h.description:
                        desc = h.description.strip()
                        if len(desc) > 500:
                            desc = desc[:500] + "..."
                        add(f"  {desc}\n")
                    add("\n")

        # Local long-term memory (preferences + facts)
        if self._long_term:
            try:
                memory_hits = await self._long_term.search(query, limit=10)
                prefs = [
                    m
                    for m in memory_hits
                    if (m.get("metadata") or {}).get("type") == "user_preference"
                    or (m.get("metadata") or {}).get("kind") == "user_preference"
                ]
                if len(prefs) < 3:
                    more = await self._long_term.search("user preference", limit=10)
                    prefs_ids = {p.get("id") for p in prefs}
                    prefs.extend([m for m in more if m.get("id") not in prefs_ids])

                prefs = prefs[:5]
                other = [m for m in memory_hits if m not in prefs][:5]

                if prefs:
                    add("## Remembered Preferences\n")
                    for p in prefs:
                        add(f"- {str(p.get('content', '')).strip()[:240]}\n")
                    add("\n")

                if other:
                    add("## Relevant Long-Term Memories\n")
                    for m in other:
                        add(f"- {str(m.get('content', '')).strip()[:240]}\n")
                    add("\n")
            except Exception as e:
                logger.debug(f"Long-term memory context build unavailable: {e}")

        # Relevant past sessions (short excerpts)
        if include_session:
            sessions = await self.get_session_context(query, max_sessions=2)
            if sessions:
                add("## Relevant Past Sessions\n")
                for session in sessions[:2]:
                    for m in session.get("messages", [])[-4:]:
                        role = str(m.get("role", "unknown"))
                        content = str(m.get("content", "")).strip()
                        add(f"- {role}: {content[:200]}\n")
                    add("\n")

        return "\n".join(context_parts).strip()
    
    @property
    def is_available(self) -> bool:
        """Check if RAG system is available."""
        return self._initialized
