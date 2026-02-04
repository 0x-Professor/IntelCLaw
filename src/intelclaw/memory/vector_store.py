"""
Vector Store - Semantic search with ChromaDB.

Provides vector embeddings and similarity search for documents.
"""

import asyncio
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
        self._embedding_function = embedding_function
        
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._initialized = False
    
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
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"}
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
        
        doc_id = doc_id or str(uuid4())
        
        try:
            await asyncio.to_thread(
                self._collection.add,
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
        
        ids = [d.get("id", str(uuid4())) for d in documents]
        contents = [d["content"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]
        
        try:
            await asyncio.to_thread(
                self._collection.add,
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
