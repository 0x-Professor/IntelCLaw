"""
Data Store - Persistent storage for contacts, notes, and personal data.

Full data storage without privacy restrictions for maximum intelligence.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class DataStore:
    """
    Persistent data storage for IntelCLaw.
    
    Stores:
    - Contacts
    - Notes
    - Credentials (encrypted)
    - Browsing history
    - Application usage
    - Custom data
    """
    
    def __init__(self, db_path: str = "data/datastore.db"):
        """
        Initialize data store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
    
    async def initialize(self) -> None:
        """Initialize database tables."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        
        cursor = self._conn.cursor()
        
        # Contacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                company TEXT,
                role TEXT,
                relationship TEXT,
                notes TEXT,
                social_links TEXT,
                address TEXT,
                birthday TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """)
        
        # Notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                category TEXT,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """)
        
        # Browsing history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS browsing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                title TEXT,
                content_summary TEXT,
                visit_time TEXT,
                duration_seconds INTEGER,
                metadata TEXT
            )
        """)
        
        # App usage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_name TEXT NOT NULL,
                window_title TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_seconds INTEGER,
                activity_type TEXT,
                metadata TEXT
            )
        """)
        
        # Credentials (encrypted)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS credentials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                service TEXT,
                username TEXT,
                encrypted_value TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """)
        
        # Generic key-value store
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keyvalue (
                key TEXT PRIMARY KEY,
                value TEXT,
                category TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Code snippets
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                language TEXT,
                code TEXT,
                description TEXT,
                tags TEXT,
                source TEXT,
                created_at TEXT,
                metadata TEXT
            )
        """)
        
        self._conn.commit()
        logger.info("DataStore initialized")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
    
    # =========================================================================
    # CONTACTS
    # =========================================================================
    
    async def add_contact(
        self,
        name: str,
        email: str = "",
        phone: str = "",
        company: str = "",
        role: str = "",
        relationship: str = "",
        notes: str = "",
        **kwargs
    ) -> int:
        """Add a new contact."""
        cursor = self._conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO contacts (name, email, phone, company, role, relationship, 
                                  notes, social_links, address, birthday, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, email, phone, company, role, relationship, notes,
            json.dumps(kwargs.get("social_links", {})),
            kwargs.get("address", ""),
            kwargs.get("birthday", ""),
            now, now,
            json.dumps(kwargs.get("metadata", {}))
        ))
        
        self._conn.commit()
        contact_id = cursor.lastrowid
        logger.info(f"Contact added: {name} (ID: {contact_id})")
        return contact_id
    
    async def get_contact(self, contact_id: int) -> Optional[Dict[str, Any]]:
        """Get a contact by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    async def search_contacts(self, query: str) -> List[Dict[str, Any]]:
        """Search contacts by name, email, or company."""
        cursor = self._conn.cursor()
        search = f"%{query}%"
        cursor.execute("""
            SELECT * FROM contacts 
            WHERE name LIKE ? OR email LIKE ? OR company LIKE ? OR notes LIKE ?
        """, (search, search, search, search))
        return [dict(row) for row in cursor.fetchall()]
    
    async def list_contacts(self) -> List[Dict[str, Any]]:
        """List all contacts."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM contacts ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]
    
    async def update_contact(self, contact_id: int, **updates) -> bool:
        """Update a contact."""
        if not updates:
            return False
        
        updates["updated_at"] = datetime.now().isoformat()
        
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [contact_id]
        
        cursor = self._conn.cursor()
        cursor.execute(f"UPDATE contacts SET {set_clause} WHERE id = ?", values)
        self._conn.commit()
        return cursor.rowcount > 0
    
    async def delete_contact(self, contact_id: int) -> bool:
        """Delete a contact."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        self._conn.commit()
        return cursor.rowcount > 0
    
    # =========================================================================
    # NOTES
    # =========================================================================
    
    async def add_note(
        self,
        title: str,
        content: str,
        category: str = "",
        tags: List[str] = None
    ) -> int:
        """Add a new note."""
        cursor = self._conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO notes (title, content, category, tags, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, content, category, json.dumps(tags or []), now, now, "{}"))
        
        self._conn.commit()
        return cursor.lastrowid
    
    async def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        """Get a note by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["tags"] = json.loads(result.get("tags", "[]"))
            return result
        return None
    
    async def search_notes(self, query: str) -> List[Dict[str, Any]]:
        """Search notes."""
        cursor = self._conn.cursor()
        search = f"%{query}%"
        cursor.execute("""
            SELECT * FROM notes 
            WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
        """, (search, search, search))
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # BROWSING HISTORY
    # =========================================================================
    
    async def add_browsing_entry(
        self,
        url: str,
        title: str = "",
        content_summary: str = "",
        duration_seconds: int = 0
    ) -> int:
        """Add a browsing history entry."""
        cursor = self._conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO browsing_history (url, title, content_summary, visit_time, duration_seconds, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (url, title, content_summary, now, duration_seconds, "{}"))
        
        self._conn.commit()
        return cursor.lastrowid
    
    async def get_browsing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent browsing history."""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM browsing_history 
            ORDER BY visit_time DESC LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # APP USAGE
    # =========================================================================
    
    async def log_app_usage(
        self,
        app_name: str,
        window_title: str = "",
        duration_seconds: int = 0,
        activity_type: str = ""
    ) -> int:
        """Log application usage."""
        cursor = self._conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO app_usage (app_name, window_title, start_time, duration_seconds, activity_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (app_name, window_title, now, duration_seconds, activity_type, "{}"))
        
        self._conn.commit()
        return cursor.lastrowid
    
    async def get_app_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get application usage statistics."""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT app_name, SUM(duration_seconds) as total_seconds, COUNT(*) as sessions
            FROM app_usage 
            WHERE start_time >= datetime('now', ?)
            GROUP BY app_name
            ORDER BY total_seconds DESC
        """, (f"-{days} days",))
        
        results = [dict(row) for row in cursor.fetchall()]
        return {"days": days, "apps": results}
    
    # =========================================================================
    # CODE SNIPPETS
    # =========================================================================
    
    async def save_code_snippet(
        self,
        title: str,
        code: str,
        language: str = "",
        description: str = "",
        tags: List[str] = None,
        source: str = ""
    ) -> int:
        """Save a code snippet."""
        cursor = self._conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO code_snippets (title, language, code, description, tags, source, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, language, code, description, json.dumps(tags or []), source, now, "{}"))
        
        self._conn.commit()
        return cursor.lastrowid
    
    async def search_code_snippets(self, query: str, language: str = "") -> List[Dict[str, Any]]:
        """Search code snippets."""
        cursor = self._conn.cursor()
        search = f"%{query}%"
        
        if language:
            cursor.execute("""
                SELECT * FROM code_snippets 
                WHERE (title LIKE ? OR description LIKE ? OR code LIKE ?) AND language = ?
            """, (search, search, search, language))
        else:
            cursor.execute("""
                SELECT * FROM code_snippets 
                WHERE title LIKE ? OR description LIKE ? OR code LIKE ?
            """, (search, search, search))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # KEY-VALUE STORE
    # =========================================================================
    
    async def set_value(self, key: str, value: Any, category: str = "") -> None:
        """Set a key-value pair."""
        cursor = self._conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO keyvalue (key, value, category, created_at, updated_at)
            VALUES (?, ?, ?, COALESCE((SELECT created_at FROM keyvalue WHERE key = ?), ?), ?)
        """, (key, json.dumps(value), category, key, now, now))
        
        self._conn.commit()
    
    async def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT value FROM keyvalue WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row["value"])
        return default
    
    async def delete_value(self, key: str) -> bool:
        """Delete a key-value pair."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM keyvalue WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0
    
    # =========================================================================
    # CREDENTIALS (Encrypted)
    # =========================================================================
    
    async def store_credential(
        self,
        name: str,
        value: str,
        service: str = "",
        username: str = ""
    ) -> bool:
        """Store an encrypted credential."""
        try:
            from cryptography.fernet import Fernet
            
            # Get or create encryption key
            key_path = Path("data/.encryption_key")
            if key_path.exists():
                key = key_path.read_bytes()
            else:
                key = Fernet.generate_key()
                key_path.write_bytes(key)
            
            fernet = Fernet(key)
            encrypted = fernet.encrypt(value.encode()).decode()
            
            cursor = self._conn.cursor()
            now = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO credentials (name, service, username, encrypted_value, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM credentials WHERE name = ?), ?), ?, ?)
            """, (name, service, username, encrypted, name, now, now, "{}"))
            
            self._conn.commit()
            logger.info(f"Credential stored: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential: {e}")
            return False
    
    async def get_credential(self, name: str) -> Optional[str]:
        """Get a decrypted credential."""
        try:
            from cryptography.fernet import Fernet
            
            key_path = Path("data/.encryption_key")
            if not key_path.exists():
                return None
            
            key = key_path.read_bytes()
            fernet = Fernet(key)
            
            cursor = self._conn.cursor()
            cursor.execute("SELECT encrypted_value FROM credentials WHERE name = ?", (name,))
            row = cursor.fetchone()
            
            if row:
                decrypted = fernet.decrypt(row["encrypted_value"].encode()).decode()
                return decrypted
            return None
            
        except Exception as e:
            logger.error(f"Failed to get credential: {e}")
            return None
