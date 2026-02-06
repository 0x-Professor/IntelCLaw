"""
SessionStore - persistent chat sessions with retrieval (no summarization).

Goals:
- Store ALL sessions locally (SQLite)
- Allow switching/loading any session
- Retrieve relevant raw context using hybrid lexical + embeddings search
- Redact secrets before persistence (hard requirement)
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from intelclaw.memory.embeddings import EmbeddingsProvider, embeddings_from_config
from intelclaw.security.redaction import contains_secret, redact_secrets

try:
    import numpy as np  # type: ignore

    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", (text or "").lower())


def _token_overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    if not q:
        return 0.0
    t = set(_tokenize(text))
    if not t:
        return 0.0
    return float(len(q & t)) / float(len(q))


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]

    max_chars = max(int(max_chars), 200)
    overlap_chars = max(int(overlap_chars), 0)

    out: List[str] = []
    start = 0
    n = len(s)
    while start < n:
        end = min(start + max_chars, n)
        chunk = s[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        if overlap_chars <= 0:
            start = end
        else:
            start = max(0, end - overlap_chars)
    return out


def _cosine_similarity(query_vec: Sequence[float], cand_blob: bytes, dim: int) -> float:
    if not query_vec or not cand_blob or dim <= 0:
        return 0.0

    if NUMPY_AVAILABLE:
        q = np.asarray(query_vec, dtype=np.float32)
        c = np.frombuffer(cand_blob, dtype=np.float32, count=int(dim))
        if q.size != c.size or q.size == 0:
            return 0.0
        denom = float(np.linalg.norm(q) * np.linalg.norm(c))
        if denom <= 0:
            return 0.0
        return float(np.dot(q, c) / denom)

    # Pure python fallback
    import struct

    floats = struct.iter_unpack("<f", cand_blob)
    c_list = [x[0] for x in floats]
    if len(c_list) != len(query_vec):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(query_vec, c_list))
    qn = sum(float(a) * float(a) for a in query_vec) ** 0.5
    cn = sum(float(b) * float(b) for b in c_list) ** 0.5
    denom = qn * cn
    return float(dot / denom) if denom > 0 else 0.0


def _serialize_embedding(vec: Sequence[float]) -> Tuple[bytes, int]:
    dim = int(len(vec))
    if dim <= 0:
        return b"", 0
    if NUMPY_AVAILABLE:
        arr = np.asarray([float(x) for x in vec], dtype=np.float32)
        return arr.tobytes(), int(arr.size)
    import struct

    return struct.pack("<" + "f" * dim, *[float(x) for x in vec]), dim


@dataclass(frozen=True)
class SessionStoreConfig:
    db_path: str = "data/sessions.db"
    enabled: bool = True

    # Redaction policy for persisted sessions: "redact" (default) or "skip"
    redaction_mode: str = "redact"

    # Chunking for retrieval index
    chunk_max_chars: int = 900
    chunk_overlap_chars: int = 120

    # Embeddings provider config (see memory/embeddings.py)
    embeddings: Dict[str, Any] = field(default_factory=lambda: {"provider": "jina"})

    # Retrieval tuning
    candidate_pool: int = 200
    top_k_hits: int = 8
    max_windows: int = 3
    window_messages_before: int = 2
    window_messages_after: int = 2
    include_other_sessions: bool = True
    exclude_last_n_messages: int = 20
    max_context_chars: int = 8000

    # Scoring weights
    semantic_weight: float = 0.75
    lexical_weight: float = 0.20
    recency_weight: float = 0.05
    recency_half_life_hours: float = 72.0


class SessionStore:
    def __init__(self, config: Optional[SessionStoreConfig] = None):
        self._cfg = config or SessionStoreConfig()

        self._db_path = Path(self._cfg.db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._fts_available = False
        self._embedder: EmbeddingsProvider = embeddings_from_config(dict(self._cfg.embeddings or {}))

    @property
    def is_enabled(self) -> bool:
        return bool(self._cfg.enabled)

    async def initialize(self) -> None:
        if not self.is_enabled:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        await self._create_tables()
        logger.info(f"Session store initialized at {self._db_path}")

    async def shutdown(self) -> None:
        try:
            await self._embedder.close()
        except Exception:
            pass

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def _create_tables(self) -> None:
        if self._conn is None:
            return

        async with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id, id)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (message_id) REFERENCES session_messages(id)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_chunks_session_id ON session_chunks(session_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_chunks_message_id ON session_chunks(message_id)"
            )

            # Best-effort FTS5 for lexical retrieval.
            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS session_chunks_fts
                    USING fts5(content, session_id UNINDEXED, chunk_id UNINDEXED, message_id UNINDEXED)
                    """
                )
                self._fts_available = True
            except Exception as e:
                self._fts_available = False
                logger.debug(f"FTS5 unavailable for session store: {e}")

            self._conn.commit()

    async def create_session(self, *, title: Optional[str] = None, session_id: Optional[str] = None) -> str:
        sid = (session_id or "").strip() or f"session_{uuid.uuid4().hex[:12]}"
        await self.ensure_session(sid, title=title)
        return sid

    async def ensure_session(self, session_id: str, *, title: Optional[str] = None) -> None:
        if not self.is_enabled or self._conn is None:
            return

        sid = str(session_id or "").strip()
        if not sid:
            return

        now = _utc_now_iso()
        async with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT session_id, title FROM sessions WHERE session_id = ?", (sid,))
            row = cur.fetchone()
            if row:
                # Update title if it is currently missing.
                existing_title = ""
                try:
                    existing_title = str(row["title"] or "")
                except Exception:
                    existing_title = ""

                if title and not existing_title.strip():
                    cur.execute(
                        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                        (title[:200], now, sid),
                    )
                    self._conn.commit()
                return

            cur.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sid, (title or "")[:200] or None, now, now, None),
            )
            self._conn.commit()

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        if not self.is_enabled or self._conn is None:
            return []

        async with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    (SELECT COUNT(1) FROM session_messages m WHERE m.session_id = s.session_id) as message_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (int(limit), int(offset)),
            )
            return [dict(r) for r in cur.fetchall()]

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_enabled or self._conn is None:
            return None

        sid = str(session_id or "").strip()
        if not sid:
            return None

        async with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    (SELECT COUNT(1) FROM session_messages m WHERE m.session_id = s.session_id) as message_count
                FROM sessions s
                WHERE s.session_id = ?
                """,
                (sid,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    async def search_sessions(self, query: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search sessions by title/session_id and (when available) message content.

        - Uses FTS5 over session chunks if available.
        - Falls back to LIKE token search over session_messages.
        """
        if not self.is_enabled or self._conn is None:
            return []

        q = str(query or "").strip()
        if not q:
            return await self.list_sessions(limit=limit, offset=offset)

        like = f"%{q}%"

        async with self._lock:
            cur = self._conn.cursor()

            # Prefer FTS if it exists; fall back gracefully on parse errors.
            if self._fts_available:
                try:
                    cur.execute(
                        """
                        WITH combined AS (
                            SELECT session_id, 0.0 AS rank
                            FROM sessions
                            WHERE (title LIKE ? OR session_id LIKE ?)

                            UNION ALL

                            SELECT session_id, MIN(bm25(session_chunks_fts)) AS rank
                            FROM session_chunks_fts
                            WHERE session_chunks_fts MATCH ?
                            GROUP BY session_id
                        ),
                        best AS (
                            SELECT session_id, MIN(rank) AS best_rank
                            FROM combined
                            GROUP BY session_id
                        )
                        SELECT
                            s.session_id,
                            s.title,
                            s.created_at,
                            s.updated_at,
                            (SELECT COUNT(1) FROM session_messages m WHERE m.session_id = s.session_id) as message_count,
                            b.best_rank AS rank
                        FROM best b
                        JOIN sessions s ON s.session_id = b.session_id
                        ORDER BY b.best_rank ASC, s.updated_at DESC
                        LIMIT ? OFFSET ?
                        """,
                        (like, like, q, int(limit), int(offset)),
                    )
                    rows = cur.fetchall()
                    return [dict(r) for r in rows]
                except Exception as e:
                    logger.debug(f"Session search (FTS) failed; falling back to LIKE: {e}")

            toks = _tokenize(q)[:6]
            params: List[Any] = [like, like]

            sql = """
                WITH matched(session_id) AS (
                    SELECT session_id FROM sessions WHERE (title LIKE ? OR session_id LIKE ?)
            """

            if toks:
                like_clauses = " OR ".join(["content LIKE ?"] * len(toks))
                sql += f"""
                    UNION
                    SELECT DISTINCT session_id FROM session_messages WHERE ({like_clauses})
                """
                params.extend([f"%{t}%" for t in toks])

            sql += """
                )
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    (SELECT COUNT(1) FROM session_messages m WHERE m.session_id = s.session_id) as message_count
                FROM matched x
                JOIN sessions s ON s.session_id = x.session_id
                ORDER BY s.updated_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([int(limit), int(offset)])
            cur.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    async def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.is_enabled or self._conn is None:
            return []

        sid = str(session_id or "").strip()
        if not sid:
            return []

        async with self._lock:
            cur = self._conn.cursor()
            if limit is None:
                cur.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM session_messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (sid,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM session_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (sid, int(limit)),
                )
            rows = cur.fetchall()
            out = [dict(r) for r in rows]
            if limit is not None:
                out.reverse()
            return out

    async def delete_session(self, session_id: str) -> bool:
        if not self.is_enabled or self._conn is None:
            return False

        sid = str(session_id or "").strip()
        if not sid:
            return False

        async with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM session_chunks WHERE session_id = ?", (sid,))
            if self._fts_available:
                try:
                    cur.execute("DELETE FROM session_chunks_fts WHERE session_id = ?", (sid,))
                except Exception:
                    pass
            cur.execute("DELETE FROM session_messages WHERE session_id = ?", (sid,))
            cur.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            changed = cur.rowcount > 0
            self._conn.commit()
            return changed

    async def add_messages(
        self,
        session_id: str,
        messages: Sequence[Dict[str, Any]],
        *,
        title_hint: Optional[str] = None,
    ) -> None:
        for m in messages:
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            await self.add_message(session_id, role=role, content=content, title_hint=title_hint)

    async def add_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        created_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        title_hint: Optional[str] = None,
    ) -> Optional[int]:
        if not self.is_enabled or self._conn is None:
            return None

        sid = str(session_id or "").strip()
        if not sid:
            return None

        safe_content = str(content or "")
        if contains_secret(safe_content):
            if str(self._cfg.redaction_mode).lower() == "skip":
                return None
            safe_content = redact_secrets(safe_content)

        # Best-effort: session title from first user message if missing.
        inferred_title = None
        if title_hint:
            inferred_title = title_hint.strip()[:200]
        elif role.lower() == "user":
            inferred_title = safe_content.strip().splitlines()[0][:80] if safe_content.strip() else None

        await self.ensure_session(sid, title=inferred_title)

        now = _utc_now_iso()
        ts = created_at or now
        meta_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None

        async with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_messages (session_id, role, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sid, str(role or "user"), safe_content, ts, meta_json),
            )
            message_id = int(cur.lastrowid)
            cur.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, sid))
            self._conn.commit()

        # Index chunks (outside lock; embedding calls can be slow).
        chunks = _chunk_text(safe_content, self._cfg.chunk_max_chars, self._cfg.chunk_overlap_chars)
        if not chunks:
            return message_id

        embeddings: List[List[float]] = []
        try:
            embeddings = await self._embedder.embed_passages(chunks)
        except Exception as e:
            logger.debug(f"Session chunk embedding failed (session_id={sid}): {e}")
            embeddings = []

        chunk_rows: List[Tuple[str, str, int, str, str, Optional[bytes], Optional[int]]] = []
        for idx, ch in enumerate(chunks):
            emb_blob: Optional[bytes] = None
            emb_dim: Optional[int] = None
            if embeddings and idx < len(embeddings):
                try:
                    emb_blob, emb_dim = _serialize_embedding(embeddings[idx])
                except Exception:
                    emb_blob, emb_dim = None, None
            chunk_rows.append((uuid.uuid4().hex, sid, message_id, ch, ts, emb_blob, emb_dim))

        async with self._lock:
            cur = self._conn.cursor()
            cur.executemany(
                """
                INSERT OR REPLACE INTO session_chunks
                (chunk_id, session_id, message_id, content, created_at, embedding, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_rows,
            )
            if self._fts_available:
                try:
                    cur.executemany(
                        """
                        INSERT INTO session_chunks_fts (content, session_id, chunk_id, message_id)
                        VALUES (?, ?, ?, ?)
                        """,
                        [(r[3], r[1], r[0], r[2]) for r in chunk_rows],
                    )
                except Exception as e:
                    logger.debug(f"Failed inserting into session_chunks_fts: {e}")
            self._conn.commit()

        return message_id

    async def retrieve_context_windows(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        max_windows: Optional[int] = None,
        exclude_last_n_messages: Optional[int] = None,
        include_other_sessions: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant raw context windows for a query.

        Returns:
            List of {"session_id", "title", "score", "messages": [{"role","content","created_at","id"}, ...]}
        """
        if not self.is_enabled or self._conn is None:
            return []

        q = str(query or "").strip()
        if not q:
            return []

        max_windows = int(max_windows if max_windows is not None else self._cfg.max_windows)
        include_other_sessions = bool(
            include_other_sessions if include_other_sessions is not None else self._cfg.include_other_sessions
        )
        exclude_last_n_messages = int(
            exclude_last_n_messages
            if exclude_last_n_messages is not None
            else self._cfg.exclude_last_n_messages
        )

        # Optional cutoff to avoid repeating the current conversation window.
        cutoff_id: Optional[int] = None
        if session_id and exclude_last_n_messages > 0:
            cutoff_id = await self._get_cutoff_message_id(str(session_id), exclude_last_n_messages)

        candidates = await self._fetch_candidate_chunks(
            q,
            session_id=str(session_id) if session_id else None,
            candidate_pool=int(self._cfg.candidate_pool),
            include_other_sessions=include_other_sessions,
            cutoff_message_id=cutoff_id,
        )
        if not candidates:
            return []

        query_vec: List[float] = []
        try:
            query_vec = await self._embedder.embed_query(q)
        except Exception as e:
            logger.debug(f"Query embedding failed (sessions): {e}")
            query_vec = []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        now_dt = datetime.now(timezone.utc)
        for c in candidates:
            content = str(c.get("content") or "")
            lex = _token_overlap_score(q, content)

            sem = 0.0
            emb_blob = c.get("embedding")
            emb_dim = c.get("embedding_dim")
            if query_vec and isinstance(emb_blob, (bytes, bytearray)) and isinstance(emb_dim, int):
                sem = _cosine_similarity(query_vec, bytes(emb_blob), int(emb_dim))

            # Recency boost (exponential decay)
            rec = 0.0
            created_at = c.get("created_at")
            try:
                if created_at:
                    dt = datetime.fromisoformat(str(created_at))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age_h = max((now_dt - dt).total_seconds() / 3600.0, 0.0)
                    hl = float(self._cfg.recency_half_life_hours or 72.0)
                    rec = math.exp(-age_h / max(hl, 1e-6))
            except Exception:
                rec = 0.0

            score = (
                float(self._cfg.semantic_weight) * float(sem)
                + float(self._cfg.lexical_weight) * float(lex)
                + float(self._cfg.recency_weight) * float(rec)
            )

            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: int(self._cfg.top_k_hits)]

        windows: List[Dict[str, Any]] = []
        seen_ranges: Dict[str, List[Tuple[int, int]]] = {}

        for score, hit in scored:
            sid = str(hit.get("session_id") or "")
            mid = hit.get("message_id")
            if not sid or not isinstance(mid, int):
                continue

            msgs = await self._fetch_window_messages(
                sid,
                mid,
                before=int(self._cfg.window_messages_before),
                after=int(self._cfg.window_messages_after),
            )
            if not msgs:
                continue

            start_id = int(msgs[0]["id"])
            end_id = int(msgs[-1]["id"])
            existing = seen_ranges.get(sid, [])
            if any(start_id <= e and end_id >= s for s, e in existing):
                continue
            existing.append((start_id, end_id))
            seen_ranges[sid] = existing

            title = await self._get_session_title(sid)
            windows.append(
                {
                    "session_id": sid,
                    "title": title,
                    "score": float(score),
                    "messages": msgs,
                }
            )
            if len(windows) >= max_windows:
                break

        return windows

    async def _get_session_title(self, session_id: str) -> str:
        if not self._conn:
            return ""
        async with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT title FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            if not row:
                return ""
            try:
                return str(row["title"] or "")
            except Exception:
                return ""

    async def _get_cutoff_message_id(self, session_id: str, exclude_last_n: int) -> Optional[int]:
        if not self._conn:
            return None
        async with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id FROM session_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, int(exclude_last_n)),
            )
            rows = cur.fetchall()
            if not rows:
                return None
            ids = [int(r["id"]) for r in rows if r.get("id") is not None]
            return min(ids) if ids else None

    async def _fetch_candidate_chunks(
        self,
        query: str,
        *,
        session_id: Optional[str],
        candidate_pool: int,
        include_other_sessions: bool,
        cutoff_message_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        if not self._conn:
            return []

        async with self._lock:
            cur = self._conn.cursor()
            candidates: List[Dict[str, Any]] = []

            # When session_id is provided, prioritize that session but optionally include others.
            session_filter = None
            if session_id and not include_other_sessions:
                session_filter = session_id

            if self._fts_available:
                # FTS query: use raw query string (SQLite will parse).
                where = "WHERE session_chunks_fts MATCH ?"
                params: List[Any] = [query]

                if session_filter:
                    where += " AND c.session_id = ?"
                    params.append(session_filter)

                if session_id and cutoff_message_id is not None:
                    # Exclude most recent messages for the active session only.
                    where += " AND NOT (c.session_id = ? AND c.message_id >= ?)"
                    params.extend([session_id, int(cutoff_message_id)])

                cur.execute(
                    f"""
                    SELECT
                        c.chunk_id,
                        c.session_id,
                        c.message_id,
                        c.content,
                        c.created_at,
                        c.embedding,
                        c.embedding_dim
                    FROM session_chunks_fts f
                    JOIN session_chunks c ON c.chunk_id = f.chunk_id
                    {where}
                    ORDER BY bm25(session_chunks_fts)
                    LIMIT ?
                    """,
                    (*params, int(candidate_pool)),
                )
                candidates = [dict(r) for r in cur.fetchall()]
            else:
                # LIKE fallback: require any token match.
                toks = _tokenize(query)[:6]
                if not toks:
                    return []
                like_clauses = " OR ".join(["content LIKE ?"] * len(toks))
                params = [f"%{t}%" for t in toks]
                sql = f"""
                    SELECT chunk_id, session_id, message_id, content, created_at, embedding, embedding_dim
                    FROM session_chunks
                    WHERE ({like_clauses})
                """
                if session_filter:
                    sql += " AND session_id = ?"
                    params.append(session_filter)
                if session_id and cutoff_message_id is not None:
                    sql += " AND NOT (session_id = ? AND message_id >= ?)"
                    params.extend([session_id, int(cutoff_message_id)])
                sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(int(candidate_pool))
                cur.execute(sql, tuple(params))
                candidates = [dict(r) for r in cur.fetchall()]

            return candidates

    async def _fetch_window_messages(
        self,
        session_id: str,
        hit_message_id: int,
        *,
        before: int,
        after: int,
    ) -> List[Dict[str, Any]]:
        if not self._conn:
            return []

        before = max(int(before), 0)
        after = max(int(after), 0)

        async with self._lock:
            cur = self._conn.cursor()

            # Previous messages
            prev: List[Dict[str, Any]] = []
            if before > 0:
                cur.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM session_messages
                    WHERE session_id = ? AND id < ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, int(hit_message_id), int(before)),
                )
                prev = [dict(r) for r in cur.fetchall()]
                prev.reverse()

            # Hit message
            cur.execute(
                """
                SELECT id, role, content, created_at
                FROM session_messages
                WHERE session_id = ? AND id = ?
                """,
                (session_id, int(hit_message_id)),
            )
            hit_row = cur.fetchone()
            hit = [dict(hit_row)] if hit_row else []

            # Next messages
            nxt: List[Dict[str, Any]] = []
            if after > 0:
                cur.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM session_messages
                    WHERE session_id = ? AND id > ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (session_id, int(hit_message_id), int(after)),
                )
                nxt = [dict(r) for r in cur.fetchall()]

        # Enforce context budget (raw, no summarization)
        all_msgs = prev + hit + nxt
        out: List[Dict[str, Any]] = []
        total = 0
        for m in all_msgs:
            content = str(m.get("content") or "")
            if not content:
                continue
            if total >= int(self._cfg.max_context_chars):
                break
            remaining = int(self._cfg.max_context_chars) - total
            if len(content) > remaining:
                m = {**m, "content": content[:remaining]}
            out.append(m)
            total += len(str(m.get("content") or ""))
        return out
