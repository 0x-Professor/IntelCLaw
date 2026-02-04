"""
Working Memory - Session state with TTL.

SQLite-based storage for active tasks and temporary context.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class WorkingMemory:
    """
    Working memory for session state persistence.
    
    Features:
    - SQLite storage with TTL
    - JSON serialization for complex data
    - Automatic cleanup of expired items
    - Thread-safe operations
    """
    
    def __init__(self, db_path: str = "data/working_memory.db"):
        """
        Initialize working memory.
        
        Args:
            db_path: Path to SQLite database
        """
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize database and tables."""
        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        
        # Create tables
        await self._create_tables()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Working memory initialized at {self._db_path}")
    
    async def shutdown(self) -> None:
        """Shutdown working memory."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._conn:
            self._conn.close()
        
        logger.info("Working memory shutdown complete")
    
    async def _create_tables(self) -> None:
        """Create database tables."""
        async with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON working_memory(expires_at)
            """)
            self._conn.commit()
    
    async def store(
        self,
        key: str,
        data: Dict[str, Any],
        ttl_hours: int = 24,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store data in working memory.
        
        Args:
            key: Unique key for the data
            data: Data to store (must be JSON serializable)
            ttl_hours: Time to live in hours
            metadata: Optional metadata
        """
        async with self._lock:
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO working_memory 
                (key, data, expires_at, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                key,
                json.dumps(data),
                expires_at.isoformat(),
                json.dumps(metadata) if metadata else None
            ))
            self._conn.commit()
    
    async def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from working memory.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Data or None if not found/expired
        """
        async with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT data, expires_at FROM working_memory
                WHERE key = ?
            """, (key,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Check expiration
            expires_at = datetime.fromisoformat(row["expires_at"])
            if expires_at < datetime.now():
                # Delete expired
                cursor.execute("DELETE FROM working_memory WHERE key = ?", (key,))
                self._conn.commit()
                return None
            
            return json.loads(row["data"])
    
    async def delete(self, key: str) -> bool:
        """Delete a key from working memory."""
        async with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM working_memory WHERE key = ?", (key,))
            self._conn.commit()
            return cursor.rowcount > 0
    
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        List all keys in working memory.
        
        Args:
            prefix: Optional key prefix filter
            
        Returns:
            List of keys
        """
        async with self._lock:
            cursor = self._conn.cursor()
            
            if prefix:
                cursor.execute("""
                    SELECT key FROM working_memory
                    WHERE key LIKE ? AND expires_at > datetime('now')
                """, (f"{prefix}%",))
            else:
                cursor.execute("""
                    SELECT key FROM working_memory
                    WHERE expires_at > datetime('now')
                """)
            
            return [row["key"] for row in cursor.fetchall()]
    
    async def extend_ttl(self, key: str, additional_hours: int) -> bool:
        """Extend the TTL of an existing key."""
        async with self._lock:
            cursor = self._conn.cursor()
            
            # Get current expiration
            cursor.execute(
                "SELECT expires_at FROM working_memory WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if not row:
                return False
            
            current_expires = datetime.fromisoformat(row["expires_at"])
            new_expires = current_expires + timedelta(hours=additional_hours)
            
            cursor.execute("""
                UPDATE working_memory 
                SET expires_at = ?
                WHERE key = ?
            """, (new_expires.isoformat(), key))
            self._conn.commit()
            return True
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired items."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Working memory cleanup error: {e}")
    
    async def _cleanup_expired(self) -> int:
        """Remove expired items."""
        async with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                DELETE FROM working_memory
                WHERE expires_at < datetime('now')
            """)
            deleted = cursor.rowcount
            self._conn.commit()
            
            if deleted > 0:
                logger.debug(f"Cleaned up {deleted} expired working memory items")
            
            return deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        async with self._lock:
            cursor = self._conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total FROM working_memory")
            total = cursor.fetchone()["total"]
            
            cursor.execute("""
                SELECT COUNT(*) as expired FROM working_memory
                WHERE expires_at < datetime('now')
            """)
            expired = cursor.fetchone()["expired"]
            
            return {
                "total_items": total,
                "expired_items": expired,
                "active_items": total - expired,
            }
