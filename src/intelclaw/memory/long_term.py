"""
Long-Term Memory - Persistent memory with Mem0.

Integrates with Mem0 for semantic memory storage and retrieval.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    logger.warning("mem0 not available - using fallback storage")


class LongTermMemory:
    """
    Long-term memory using Mem0.
    
    Features:
    - Semantic memory storage
    - Automatic fact extraction
    - Relationship tracking
    - User preference learning
    
    Falls back to simple storage if Mem0 not available.
    """
    
    def __init__(
        self,
        user_id: str = "default",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize long-term memory.
        
        Args:
            user_id: User identifier for memory isolation
            config: Mem0 configuration
        """
        self._user_id = user_id
        self._config = config or {}
        self._memory: Optional[Any] = None
        self._fallback_memories: List[Dict[str, Any]] = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Mem0 or fallback storage."""
        if MEM0_AVAILABLE:
            try:
                # Configure Mem0
                mem0_config = {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4o-mini",
                            "temperature": 0.1,
                        }
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": "text-embedding-3-small",
                        }
                    },
                    "vector_store": {
                        "provider": "chroma",
                        "config": {
                            "collection_name": f"intelclaw_{self._user_id}",
                            "path": self._config.get("vector_path", "data/mem0_vectors"),
                        }
                    },
                    "version": "v1.1",
                }
                
                # Override with user config
                if self._config.get("mem0_config"):
                    mem0_config.update(self._config["mem0_config"])
                
                self._memory = Memory.from_config(mem0_config)
                self._initialized = True
                logger.info("Long-term memory initialized with Mem0")
                
            except Exception as e:
                logger.warning(f"Mem0 initialization failed: {e}, using fallback")
                self._initialized = True
        else:
            self._initialized = True
            logger.info("Long-term memory initialized (fallback mode)")
    
    async def shutdown(self) -> None:
        """Shutdown long-term memory."""
        self._memory = None
        logger.info("Long-term memory shutdown complete")
    
    async def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a memory.
        
        Args:
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        if not self._initialized:
            return ""
        
        memory_id = str(uuid4())
        
        if self._memory and MEM0_AVAILABLE:
            try:
                result = await asyncio.to_thread(
                    self._memory.add,
                    content,
                    user_id=self._user_id,
                    metadata=metadata
                )
                
                # Extract memory ID from result
                if result and "results" in result:
                    for r in result["results"]:
                        if "id" in r:
                            return r["id"]
                
                return memory_id
                
            except Exception as e:
                logger.error(f"Mem0 add failed: {e}")
        
        # Fallback storage
        self._fallback_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        })
        
        return memory_id
    
    async def search(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        if not self._initialized:
            return []
        
        if self._memory and MEM0_AVAILABLE:
            try:
                results = await asyncio.to_thread(
                    self._memory.search,
                    query,
                    user_id=self._user_id,
                    limit=limit
                )
                
                return [
                    {
                        "id": r.get("id", ""),
                        "content": r.get("memory", r.get("text", "")),
                        "score": r.get("score", 0),
                        "metadata": r.get("metadata", {}),
                    }
                    for r in (results.get("results", []) if isinstance(results, dict) else results)
                ]
                
            except Exception as e:
                logger.error(f"Mem0 search failed: {e}")
        
        # Fallback: simple keyword search
        query_lower = query.lower()
        matches = [
            m for m in self._fallback_memories
            if query_lower in m["content"].lower()
        ]
        return matches[:limit]
    
    async def get_all(self) -> List[Dict[str, Any]]:
        """Get all memories for the user."""
        if not self._initialized:
            return []
        
        if self._memory and MEM0_AVAILABLE:
            try:
                results = await asyncio.to_thread(
                    self._memory.get_all,
                    user_id=self._user_id
                )
                
                memories = results.get("results", []) if isinstance(results, dict) else results
                
                return [
                    {
                        "id": r.get("id", ""),
                        "content": r.get("memory", r.get("text", "")),
                        "metadata": r.get("metadata", {}),
                        "created_at": r.get("created_at", ""),
                    }
                    for r in memories
                ]
                
            except Exception as e:
                logger.error(f"Mem0 get_all failed: {e}")
        
        return self._fallback_memories.copy()
    
    async def update(self, memory_id: str, content: str) -> bool:
        """Update a memory by ID."""
        if not self._initialized:
            return False
        
        if self._memory and MEM0_AVAILABLE:
            try:
                await asyncio.to_thread(
                    self._memory.update,
                    memory_id,
                    content
                )
                return True
            except Exception as e:
                logger.error(f"Mem0 update failed: {e}")
        
        # Fallback
        for memory in self._fallback_memories:
            if memory["id"] == memory_id:
                memory["content"] = content
                memory["updated_at"] = datetime.now().isoformat()
                return True
        
        return False
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if not self._initialized:
            return False
        
        if self._memory and MEM0_AVAILABLE:
            try:
                await asyncio.to_thread(
                    self._memory.delete,
                    memory_id
                )
                return True
            except Exception as e:
                logger.error(f"Mem0 delete failed: {e}")
        
        # Fallback
        for i, memory in enumerate(self._fallback_memories):
            if memory["id"] == memory_id:
                self._fallback_memories.pop(i)
                return True
        
        return False
    
    async def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get history of changes for a memory."""
        if not self._initialized:
            return []
        
        if self._memory and MEM0_AVAILABLE:
            try:
                history = await asyncio.to_thread(
                    self._memory.history,
                    memory_id
                )
                return history if isinstance(history, list) else []
            except Exception as e:
                logger.error(f"Mem0 history failed: {e}")
        
        return []
    
    @property
    def is_available(self) -> bool:
        """Check if Mem0 is available."""
        return MEM0_AVAILABLE and self._memory is not None
