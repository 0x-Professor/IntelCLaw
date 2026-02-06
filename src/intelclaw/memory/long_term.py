"""
Long-Term Memory - Persistent memory with Mem0.

Integrates with Mem0 for semantic memory storage and retrieval.
"""

import asyncio
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from intelclaw.security.redaction import contains_secret, redact_secrets

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
    
    Defaults to a local SQLite backend (no cloud dependency).
    Can optionally use Mem0 if configured and available.
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

        # Local backend
        long_term_cfg = self._config.get("long_term", {}) if isinstance(self._config, dict) else {}
        self._backend = str(long_term_cfg.get("backend", "local")).lower()
        self._db_path = Path(str(long_term_cfg.get("db_path", "data/long_term_memory.db")))
        self._conn: Optional[sqlite3.Connection] = None
        self._fts_available = False

        redaction_cfg = self._config.get("redaction", {}) if isinstance(self._config, dict) else {}
        self._redaction_mode = str(redaction_cfg.get("on_detect", "skip")).lower()  # "skip" | "redact"
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Mem0 or fallback storage."""
        # Optional Mem0 backend
        if self._backend == "mem0" and MEM0_AVAILABLE:
            try:
                # Configure Mem0
                mem0_config = {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4o-mini",
                            "temperature": 0.1,
                        },
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": "text-embedding-3-small",
                        },
                    },
                    "vector_store": {
                        "provider": "chroma",
                        "config": {
                            "collection_name": f"intelclaw_{self._user_id}",
                            "path": self._config.get("vector_path", "data/mem0_vectors"),
                        },
                    },
                    "version": "v1.1",
                }

                # Override with user config
                if self._config.get("mem0_config"):
                    mem0_config.update(self._config["mem0_config"])

                self._memory = Memory.from_config(mem0_config)
                self._initialized = True
                logger.info("Long-term memory initialized with Mem0")
                return

            except Exception as e:
                logger.warning(f"Mem0 initialization failed: {e}, falling back to local SQLite")

        # Local backend (default)
        self._backend = "local"
        self._init_sqlite()
        self._initialized = True
        logger.info(f"Long-term memory initialized (local SQLite: {self._db_path})")

    def _init_sqlite(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        cursor = self._conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                kind TEXT,
                importance REAL,
                metadata_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_kind ON memories(user_id, kind)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_created ON memories(user_id, created_at)")

        # Best-effort FTS5 for fast retrieval
        try:
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(user_id UNINDEXED, id UNINDEXED, content, summary, kind)
                """
            )
            self._fts_available = True
        except sqlite3.OperationalError:
            self._fts_available = False

        self._conn.commit()

    def _handle_redaction(self, content: str) -> Optional[str]:
        if not content:
            return None
        if contains_secret(content):
            if self._redaction_mode == "redact":
                return redact_secrets(content)
            return None
        return content
    
    async def shutdown(self) -> None:
        """Shutdown long-term memory."""
        self._memory = None
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
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
        
        safe_content = self._handle_redaction(content)
        if safe_content is None:
            return ""

        memory_id = str(uuid4())
        metadata = metadata or {}
        kind = str(metadata.get("kind") or metadata.get("type") or "")
        summary = metadata.get("summary")
        importance = float(metadata.get("importance", 0.5))
        now = datetime.now().isoformat()
        
        if self._backend == "mem0" and self._memory and MEM0_AVAILABLE:
            try:
                result = await asyncio.to_thread(
                    self._memory.add,
                    safe_content,
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
        
        # Local SQLite storage
        if not self._conn:
            return ""

        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, user_id, content, summary, kind, importance, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    self._user_id,
                    safe_content,
                    str(summary) if summary is not None else None,
                    kind,
                    importance,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )

            if self._fts_available:
                cursor.execute(
                    "INSERT OR REPLACE INTO memories_fts (user_id, id, content, summary, kind) VALUES (?, ?, ?, ?, ?)",
                    (self._user_id, memory_id, safe_content, str(summary) if summary else "", kind),
                )

            self._conn.commit()
            return memory_id
        except Exception as e:
            logger.error(f"Local LTM add failed: {e}")
            return ""
    
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
        
        if self._backend == "mem0" and self._memory and MEM0_AVAILABLE:
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

        if not self._conn:
            return []

        cursor = self._conn.cursor()

        # FTS5 path
        if self._fts_available:
            try:
                rows = cursor.execute(
                    """
                    SELECT id, bm25(memories_fts) AS rank
                    FROM memories_fts
                    WHERE user_id = ? AND memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (self._user_id, query, limit),
                ).fetchall()

                ranked_ids = [(str(r["id"]), float(r["rank"])) for r in rows if r["id"]]
                if ranked_ids:
                    ids = [rid for rid, _ in ranked_ids]
                    placeholders = ",".join(["?"] * len(ids))
                    mem_rows = cursor.execute(
                        f"SELECT * FROM memories WHERE user_id = ? AND id IN ({placeholders})",
                        (self._user_id, *ids),
                    ).fetchall()
                    by_id = {str(r["id"]): r for r in mem_rows}

                    results: List[Dict[str, Any]] = []
                    for mid, rank in ranked_ids:
                        row = by_id.get(mid)
                        if not row:
                            continue
                        score = 1.0 / (1.0 + max(rank, 0.0))
                        results.append(
                            {
                                "id": mid,
                                "content": row["content"],
                                "score": score,
                                "metadata": json.loads(row["metadata_json"] or "{}"),
                            }
                        )
                    return results[:limit]
            except sqlite3.OperationalError:
                # fall through to LIKE matching
                pass

        # LIKE + token overlap fallback
        tokens = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2][:6]
        if not tokens:
            return []

        patterns = [f"%{t}%" for t in tokens]
        where_parts = []
        params: List[Any] = [self._user_id]
        for _ in tokens:
            where_parts.append("(content LIKE ? OR summary LIKE ? OR kind LIKE ?)")
        for p in patterns:
            params.extend([p, p, p])

        sql = f"""
            SELECT * FROM memories
            WHERE user_id = ? AND ({' OR '.join(where_parts)})
            LIMIT ?
        """
        params.append(max(limit * 5, limit))

        rows = cursor.execute(sql, params).fetchall()

        def _overlap_score(text: str) -> float:
            lowered = (text or "").lower()
            hits = sum(1 for t in tokens if t in lowered)
            return hits / max(len(tokens), 1)

        scored: List[Dict[str, Any]] = []
        for row in rows:
            score = _overlap_score(str(row["content"]) + " " + str(row["summary"] or ""))
            scored.append(
                {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "score": score,
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            )

        scored.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return scored[:limit]
    
    async def get_all(self) -> List[Dict[str, Any]]:
        """Get all memories for the user."""
        if not self._initialized:
            return []
        
        if self._backend == "mem0" and self._memory and MEM0_AVAILABLE:
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

        if not self._conn:
            return []

        cursor = self._conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
            (self._user_id,),
        ).fetchall()

        return [
            {
                "id": str(r["id"]),
                "content": r["content"],
                "metadata": json.loads(r["metadata_json"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    
    async def update(self, memory_id: str, content: str) -> bool:
        """Update a memory by ID."""
        if not self._initialized:
            return False
        
        safe_content = self._handle_redaction(content)
        if safe_content is None:
            return False

        if self._backend == "mem0" and self._memory and MEM0_AVAILABLE:
            try:
                await asyncio.to_thread(
                    self._memory.update,
                    memory_id,
                    safe_content
                )
                return True
            except Exception as e:
                logger.error(f"Mem0 update failed: {e}")

        if not self._conn:
            return False

        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE user_id = ? AND id = ?",
                (safe_content, datetime.now().isoformat(), self._user_id, memory_id),
            )
            if self._fts_available:
                cursor.execute(
                    "DELETE FROM memories_fts WHERE user_id = ? AND id = ?",
                    (self._user_id, memory_id),
                )
                cursor.execute(
                    "INSERT OR REPLACE INTO memories_fts (user_id, id, content, summary, kind) "
                    "SELECT user_id, id, content, COALESCE(summary,''), COALESCE(kind,'') "
                    "FROM memories WHERE user_id = ? AND id = ?",
                    (self._user_id, memory_id),
                )
            self._conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Local LTM update failed: {e}")
            return False
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if not self._initialized:
            return False
        
        if self._backend == "mem0" and self._memory and MEM0_AVAILABLE:
            try:
                await asyncio.to_thread(
                    self._memory.delete,
                    memory_id
                )
                return True
            except Exception as e:
                logger.error(f"Mem0 delete failed: {e}")

        if not self._conn:
            return False

        try:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM memories WHERE user_id = ? AND id = ?", (self._user_id, memory_id))
            if self._fts_available:
                cursor.execute(
                    "DELETE FROM memories_fts WHERE user_id = ? AND id = ?",
                    (self._user_id, memory_id),
                )
            self._conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Local LTM delete failed: {e}")
            return False
    
    async def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get history of changes for a memory."""
        if not self._initialized:
            return []
        
        if self._backend == "mem0" and self._memory and MEM0_AVAILABLE:
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
        if not self._initialized:
            return False
        if self._backend == "mem0":
            return MEM0_AVAILABLE and self._memory is not None
        return self._conn is not None
