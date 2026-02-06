"""
Vector Store - Semantic search with ChromaDB.

Provides vector embeddings and similarity search for documents.
"""

import asyncio
import hashlib
import re
import sys
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not available - vector search disabled")

try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


class HashingEmbeddingFunction:
    """
    Lightweight local embedding function (no external model dependencies).

    Chroma's default embedding function requires `onnxruntime`, which is not always
    available (especially on new Python versions). This hashing-based embedder is
    deterministic, fast, and good enough for "best-effort" semantic retrieval.
    """

    def __init__(self, dimension: int = 384, token_limit: int = 2048):
        self._dimension = max(int(dimension), 32)
        self._token_limit = max(int(token_limit), 256)

    def __call__(self, input: Any) -> Any:  # Chroma validates/normalizes
        texts: List[str]
        if isinstance(input, str):
            texts = [input]
        elif isinstance(input, list):
            texts = [str(x or "") for x in input]
        else:
            try:
                texts = [str(x or "") for x in list(input)]
            except Exception:
                texts = [str(input or "")]

        if not NUMPY_AVAILABLE:
            # Fall back to pure-python vectors (should be rare since chromadb depends on numpy).
            out_list: List[List[float]] = []
            for text in texts:
                vec = [0.0] * self._dimension
                tokens = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
                if len(tokens) > self._token_limit:
                    tokens = tokens[: self._token_limit]

                for tok in tokens:
                    digest = hashlib.sha256(tok.encode("utf-8")).digest()
                    idx = int.from_bytes(digest[:4], "little", signed=False) % self._dimension
                    sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                    vec[idx] += sign

                # l2 normalize
                norm = sum(v * v for v in vec) ** 0.5
                if norm > 0:
                    vec = [v / norm for v in vec]
                out_list.append(vec)
            return out_list

        out: List[Any] = []
        for text in texts:
            vec = np.zeros((self._dimension,), dtype=np.float32)
            tokens = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
            if len(tokens) > self._token_limit:
                tokens = tokens[: self._token_limit]

            for tok in tokens:
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little", signed=False) % self._dimension
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                vec[idx] += sign

            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            out.append(vec)

        return out

    def embed_query(self, input: Any) -> Any:
        # Chroma calls embed_query() for queries if present.
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "intelclaw_hashing_v1"

    def get_config(self) -> Dict[str, Any]:
        return {"dimension": self._dimension, "token_limit": self._token_limit}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(
            dimension=int(config.get("dimension", 384)),
            token_limit=int(config.get("token_limit", 2048)),
        )

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def is_legacy(self) -> bool:
        # Chroma uses this to decide how to serialize embedding function config.
        return False

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        return None

    @staticmethod
    def validate_config_update(config: Dict[str, Any]) -> None:
        return None


class VectorStore:
    """
    Vector storage and semantic search using ChromaDB.
    
    Features:
    - Document embedding and indexing
    - Semantic similarity search
    - Metadata filtering
    - Persistent storage
    """
    
    def __init__(
        self,
        collection_name: str = "intelclaw",
        persist_directory: str = "data/vector_db",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_function: Custom embedding function (optional)
        """
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_function = embedding_function or HashingEmbeddingFunction()
        
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._initialized = False
        self._write_disabled_logged = False

    def _writes_allowed(self) -> bool:
        # On Windows, importing PyQt6 can cause Chroma's add/upsert to hard-crash
        # the process. Reading/querying is fine if vectors were written beforehand.
        return "PyQt6" not in sys.modules

    def _log_write_disabled_once(self) -> None:
        if not self._write_disabled_logged:
            self._write_disabled_logged = True
            logger.warning("Vector store writes disabled while PyQt6 is loaded (preventing hard crash)")
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if not CHROMA_AVAILABLE:
            logger.warning("Vector store not available (chromadb not installed)")
            return
        
        try:
            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Get or create collection
            try:
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self._embedding_function,
                )
            except Exception as e:
                # A common case is an embedding function mismatch with an existing persisted collection.
                # Reset and recreate with our embedding function (best-effort).
                logger.warning(f"Vector store collection init failed, resetting: {e}")
                try:
                    await asyncio.to_thread(self._client.delete_collection, self._collection_name)
                except Exception:
                    pass
                self._collection = self._client.create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self._embedding_function,
                )
            
            self._initialized = True
            logger.info(f"Vector store initialized: {self._collection_name}")
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown vector store."""
        self._client = None
        self._collection = None
        logger.info("Vector store shutdown complete")
    
    async def add(
        self,
        doc_id: Optional[str],
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            doc_id: Document ID (generated if not provided)
            content: Document content
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        if not self._initialized or not self._collection:
            return ""

        if not self._writes_allowed():
            self._log_write_disabled_once()
            return ""
        
        doc_id = doc_id or str(uuid4())
        
        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata] if metadata else None
            )
            return doc_id
            
        except Exception as e:
            logger.error(f"Vector store add failed: {e}")
            return ""
    
    async def add_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple documents.
        
        Args:
            documents: List of {"id", "content", "metadata"}
            
        Returns:
            List of document IDs
        """
        if not self._initialized or not self._collection:
            return []

        if not self._writes_allowed():
            self._log_write_disabled_once()
            return []
        
        ids = [d.get("id", str(uuid4())) for d in documents]
        contents = [d["content"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]
        
        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            return ids
            
        except Exception as e:
            logger.error(f"Vector store batch add failed: {e}")
            return []
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents by semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum results
            filter_metadata: Metadata filter (ChromaDB where clause)
            
        Returns:
            List of matching documents with scores
        """
        if not self._initialized or not self._collection:
            return []
        
        try:
            results = await asyncio.to_thread(
                self._collection.query,
                query_texts=[query],
                n_results=limit,
                where=filter_metadata
            )
            
            # Format results
            documents = []
            if results and results.get("ids"):
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = {
                        "id": doc_id,
                        "content": results["documents"][0][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    }
                    if results.get("distances"):
                        doc["score"] = 1 - results["distances"][0][i]  # Convert distance to similarity
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        if not self._initialized or not self._collection:
            return None
        
        try:
            results = await asyncio.to_thread(
                self._collection.get,
                ids=[doc_id]
            )
            
            if results and results.get("ids") and results["ids"]:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0] if results.get("documents") else "",
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {},
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Vector get failed: {e}")
            return None
    
    async def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document."""
        if not self._initialized or not self._collection:
            return False
        
        try:
            update_kwargs = {"ids": [doc_id]}
            if content:
                update_kwargs["documents"] = [content]
            if metadata:
                update_kwargs["metadatas"] = [metadata]
            
            await asyncio.to_thread(
                self._collection.update,
                **update_kwargs
            )
            return True
            
        except Exception as e:
            logger.error(f"Vector update failed: {e}")
            return False
    
    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        if not self._initialized or not self._collection:
            return False
        
        try:
            await asyncio.to_thread(
                self._collection.delete,
                ids=[doc_id]
            )
            return True
            
        except Exception as e:
            logger.error(f"Vector delete failed: {e}")
            return False

    async def delete_where(self, filter_metadata: Dict[str, Any]) -> int:
        """Delete documents matching a metadata filter. Returns deleted count (best-effort)."""
        if not self._initialized or not self._collection:
            return 0

        try:
            existing = await asyncio.to_thread(self._collection.get, where=filter_metadata)
            ids = existing.get("ids") if isinstance(existing, dict) else None
            count = len(ids) if isinstance(ids, list) else 0
            await asyncio.to_thread(self._collection.delete, where=filter_metadata)
            return count
        except Exception as e:
            logger.error(f"Vector delete_where failed: {e}")
            return 0
    
    async def count(self) -> int:
        """Get total document count."""
        if not self._initialized or not self._collection:
            return 0
        
        try:
            return await asyncio.to_thread(self._collection.count)
        except Exception as e:
            logger.error(f"Vector count failed: {e}")
            return 0
    
    async def clear(self) -> bool:
        """Clear all documents from the collection."""
        if not self._initialized or not self._client:
            return False
        
        try:
            await asyncio.to_thread(
                self._client.delete_collection,
                self._collection_name
            )
            self._collection = await asyncio.to_thread(
                self._client.create_collection,
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return True
            
        except Exception as e:
            logger.error(f"Vector clear failed: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if vector store is available."""
        return self._initialized
