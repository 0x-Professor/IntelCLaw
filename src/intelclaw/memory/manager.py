"""
Memory Manager - Unified interface for multi-tier memory system.

Coordinates short-term, working, and long-term memory.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from intelclaw.memory.short_term import ShortTermMemory
from intelclaw.memory.working_memory import WorkingMemory
from intelclaw.memory.long_term import LongTermMemory
from intelclaw.memory.vector_store import VectorStore
from intelclaw.memory.agentic_rag import AgenticRAG
from intelclaw.memory.pageindex_watcher import PageIndexWatcher, PageIndexWatcherConfig, WATCHDOG_AVAILABLE, _normalize_exts

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager
    from intelclaw.core.events import EventBus


class MemoryManager:
    """
    Unified memory manager for IntelCLaw.
    
    Three-tier memory architecture:
    
    1. Short-Term Memory (STM):
       - In-memory circular buffer
       - Current conversation context
       - Fast access, ephemeral
       
    2. Working Memory (WM):
       - SQLite with TTL
       - Active tasks and session state
       - 24-48 hour retention
       
    3. Long-Term Memory (LTM):
       - Mem0 + ChromaDB + optional Neo4j
       - User preferences, facts, relationships
       - Persistent with lifecycle management
    """
    
    def __init__(
        self,
        config: "ConfigManager",
        event_bus: "EventBus",
    ):
        """
        Initialize memory manager.
        
        Args:
            config: Configuration manager
            event_bus: Event bus for notifications
        """
        self.config = config
        self.event_bus = event_bus
        
        # Memory tiers
        self.short_term: Optional[ShortTermMemory] = None
        self.working: Optional[WorkingMemory] = None
        self.long_term: Optional[LongTermMemory] = None
        self.vector_store: Optional[VectorStore] = None
        
        # Agentic RAG system (reasoning-based retrieval)
        self.agentic_rag: Optional[AgenticRAG] = None

        # PageIndex auto-ingest watcher (optional)
        self.pageindex_watcher: Optional[PageIndexWatcher] = None
        
        self._initialized = False
        
        logger.debug("MemoryManager created")
    
    async def initialize(self) -> None:
        """Initialize all memory tiers."""
        logger.info("Initializing memory system...")
        
        memory_config = self.config.get("memory", {})
        
        # Initialize short-term memory
        self.short_term = ShortTermMemory(
            max_messages=memory_config.get("max_conversation_history", 50)
        )
        
        # Initialize working memory
        db_path = memory_config.get("working_db_path", "data/working_memory.db")
        self.working = WorkingMemory(db_path=db_path)
        await self.working.initialize()
        
        # Initialize long-term memory
        self.long_term = LongTermMemory(
            user_id=memory_config.get("user_id", "default"),
            config=memory_config,
        )
        await self.long_term.initialize()
        
        # Initialize vector store
        vector_config = memory_config.get("vector_store", {})
        self.vector_store = VectorStore(
            collection_name=vector_config.get("collection", "intelclaw"),
            persist_directory=vector_config.get("path", "data/vector_db"),
        )
        await self.vector_store.initialize()
        
        # Initialize Agentic RAG system (reasoning-based retrieval)
        rag_config = memory_config.get("agentic_rag", {})
        if isinstance(rag_config, dict) and isinstance(memory_config, dict):
            rag_config = {
                **rag_config,
                "pageindex": memory_config.get("pageindex", {}),
                "redaction": memory_config.get("redaction", {}),
            }
        self.agentic_rag = AgenticRAG(
            user_id=memory_config.get("user_id", "default"),
            persist_dir=rag_config.get("path", "data/agentic_rag"),
            vector_store=self.vector_store,
            long_term=self.long_term,
        )
        await self.agentic_rag.initialize(rag_config)
        
        # Index persona files for context-aware retrieval
        persona_dir = Path(__file__).parent.parent.parent.parent / "persona"
        if persona_dir.exists():
            await self.agentic_rag.index_persona(persona_dir)

        # Optional: watch a folder and auto-index PDFs via PageIndex
        pageindex_cfg = memory_config.get("pageindex", {}) if isinstance(memory_config, dict) else {}
        if (
            self.agentic_rag
            and pageindex_cfg.get("enabled", True)
            and pageindex_cfg.get("watch", True)
            and WATCHDOG_AVAILABLE
        ):
            ingest_folder = Path(pageindex_cfg.get("ingest_folder", "data/pageindex_inbox"))
            exts = _normalize_exts(pageindex_cfg.get("extensions", [".pdf"]))
            watcher_cfg = PageIndexWatcherConfig(
                ingest_folder=ingest_folder,
                extensions=exts,
                debounce_seconds=float(pageindex_cfg.get("debounce_seconds", 1.0)),
            )

            async def _on_file(path: Path) -> None:
                if not self.agentic_rag:
                    return
                # Implemented on AgenticRAG; watcher is best-effort.
                await self.agentic_rag.index_path(path)

            self.pageindex_watcher = PageIndexWatcher(watcher_cfg, _on_file)
            await self.pageindex_watcher.start(scan_on_startup=True)
        
        self._initialized = True
        logger.success("Memory system initialized")
    
    async def shutdown(self) -> None:
        """Shutdown all memory tiers."""
        logger.info("Shutting down memory system...")

        if self.pageindex_watcher:
            try:
                await self.pageindex_watcher.stop()
            except Exception as e:
                logger.debug(f"Failed to stop PageIndex watcher: {e}")
            self.pageindex_watcher = None
        
        if self.agentic_rag:
            await self.agentic_rag.shutdown()
        
        if self.vector_store:
            await self.vector_store.shutdown()
        
        if self.long_term:
            await self.long_term.shutdown()
        
        if self.working:
            await self.working.shutdown()
        
        self._initialized = False
        logger.info("Memory system shutdown complete")
    
    # ==================== Short-Term Memory ====================
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to short-term memory."""
        if self.short_term:
            self.short_term.add_message(role, content, metadata)
    
    def get_conversation_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        if self.short_term:
            return self.short_term.get_history(limit)
        return []
    
    def clear_conversation(self) -> None:
        """Clear short-term conversation memory."""
        if self.short_term:
            self.short_term.clear()
    
    # ==================== Working Memory ====================
    
    async def store_task(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        ttl_hours: int = 24
    ) -> None:
        """Store a task in working memory."""
        if self.working:
            await self.working.store(
                key=f"task:{task_id}",
                data=task_data,
                ttl_hours=ttl_hours
            )
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a task from working memory."""
        if self.working:
            return await self.working.retrieve(f"task:{task_id}")
        return None
    
    async def store_session_context(
        self,
        context: Dict[str, Any],
        ttl_hours: int = 48
    ) -> None:
        """Store session context."""
        if self.working:
            await self.working.store(
                key="session:context",
                data=context,
                ttl_hours=ttl_hours
            )
    
    async def get_session_context(self) -> Optional[Dict[str, Any]]:
        """Get session context."""
        if self.working:
            return await self.working.retrieve("session:context")
        return None
    
    # ==================== Long-Term Memory ====================
    
    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory in long-term storage."""
        if self.long_term:
            return await self.long_term.add(content, metadata)
        return ""
    
    async def search_memories(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search long-term memories."""
        if self.long_term:
            return await self.long_term.search(query, limit)
        return []
    
    async def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all long-term memories."""
        if self.long_term:
            return await self.long_term.get_all()
        return []
    
    async def update_memory(self, memory_id: str, content: str) -> bool:
        """Update a specific memory."""
        if self.long_term:
            return await self.long_term.update(memory_id, content)
        return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        if self.long_term:
            return await self.long_term.delete(memory_id)
        return False
    
    # ==================== Vector Store ====================
    
    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Index a document in the vector store."""
        if self.vector_store:
            await self.vector_store.add(doc_id, content, metadata)
    
    async def search_documents(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents by semantic similarity."""
        if self.vector_store:
            return await self.vector_store.search(query, limit, filter_metadata)
        return []
    
    # ==================== Convenience Methods ====================
    
    async def store_interaction(
        self,
        user_message: str,
        agent_response: str,
        tools_used: Optional[List[str]] = None,
    ) -> None:
        """
        Store a complete interaction across memory tiers.
        
        - Adds to short-term conversation history
        - Extracts and stores important facts to long-term
        """
        # Short-term
        self.add_message("user", user_message)
        self.add_message("assistant", agent_response)
        
        # Extract facts for long-term (could use LLM for extraction)
        # For now, just store significant interactions
        if len(user_message) > 50 or tools_used:
            await self.store_memory(
                f"User asked: {user_message[:200]}\nResponse summary: {agent_response[:200]}",
                metadata={
                    "type": "interaction",
                    "tools": tools_used or [],
                    "timestamp": datetime.now().isoformat(),
                }
            )
    
    async def get_relevant_context(
        self,
        query: str,
        include_conversation: bool = True,
        include_memories: bool = True,
        include_documents: bool = True,
    ) -> Dict[str, Any]:
        """
        Get all relevant context for a query.
        
        Combines short-term, long-term, and vector search results.
        """
        context = {}
        
        if include_conversation:
            context["conversation"] = self.get_conversation_history(limit=10)
        
        if include_memories:
            context["memories"] = await self.search_memories(query, limit=5)
        
        if include_documents:
            context["documents"] = await self.search_documents(query, limit=3)
        
        return context
    
    # ==================== Agentic RAG ====================
    
    async def get_rag_context(
        self,
        query: str,
        include_persona: bool = True,
        include_session: bool = True,
        max_context_chars: int = 10000,
        available_tool_names: Optional[set[str]] = None,
    ) -> str:
        """
        Get comprehensive RAG context using reasoning-based retrieval.
        
        This uses the AgenticRAG system for better context retrieval
        with hierarchical tree indexing and LLM reasoning.
        
        Args:
            query: The user query
            include_persona: Include persona files in context
            include_session: Include relevant past sessions
            max_context_chars: Maximum context size
            
        Returns:
            Formatted context string for system prompt injection
        """
        if self.agentic_rag:
            return await self.agentic_rag.build_context(
                query=query,
                include_persona=include_persona,
                include_session=include_session,
                max_context_chars=max_context_chars,
                available_tool_names=available_tool_names,
            )
        return ""
    
    async def rag_retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        max_results: int = 5,
        doc_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context using agentic RAG.
        
        Strategies:
        - "tree": Use document tree traversal with LLM reasoning
        - "semantic": Use vector similarity (fallback)
        - "hybrid": Combine tree reasoning with semantic backup
        """
        if self.agentic_rag:
            return await self.agentic_rag.retrieve(
                query=query,
                strategy=strategy,
                max_results=max_results,
                doc_types=doc_types
            )
        return []
    
    async def rag_index_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        doc_type: str = "generic"
    ) -> None:
        """
        Index a document in the agentic RAG system.
        
        Creates a hierarchical tree index for reasoning-based retrieval.
        """
        if self.agentic_rag:
            await self.agentic_rag.index_document(
                doc_id=doc_id,
                title=title,
                content=content,
                doc_type=doc_type
            )
    
    async def rag_store_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store session conversation for future context retrieval."""
        if self.agentic_rag:
            await self.agentic_rag.store_session(
                session_id=session_id,
                messages=messages,
                metadata=metadata
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        stats = {
            "initialized": self._initialized,
            "short_term_messages": 0,
            "working_items": 0,
            "long_term_memories": 0,
            "vector_documents": 0,
            "rag_available": self.agentic_rag is not None and self.agentic_rag.is_available,
            "rag_document_trees": len(self.agentic_rag.document_trees) if self.agentic_rag else 0,
        }
        
        if self.short_term:
            stats["short_term_messages"] = len(self.short_term.get_history())
        
        return stats
