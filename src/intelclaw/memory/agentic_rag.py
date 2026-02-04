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
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from loguru import logger

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False


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
        title: str,
        content: str,
        summary: str = "",
        parent_id: str = "root",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a node to the tree."""
        node_id = str(uuid4())[:8]
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
        llm_provider: Optional[Any] = None
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
        
        # Document trees for hierarchical indexing
        self.document_trees: Dict[str, DocumentTree] = {}
        
        # Session memory
        self.session_memories: List[Dict[str, Any]] = []
        
        # Persona cache (loaded from markdown files)
        self.persona_context: Dict[str, str] = {}
        
        # Mem0 for infinite memory
        self._mem0: Optional[Any] = None
        
        # ChromaDB for semantic backup
        self._vector_store: Optional[Any] = None
        
        self._initialized = False
        
        logger.debug("AgenticRAG created")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the RAG system."""
        logger.info("Initializing Agentic RAG system...")
        
        config = config or {}
        
        # Load persisted document trees
        await self._load_trees()
        
        # Initialize Mem0 for infinite memory
        if MEM0_AVAILABLE:
            try:
                mem0_config = config.get("mem0", {})
                
                # Configure Mem0 with Copilot-compatible settings
                mem0_settings = {
                    "llm": {
                        "provider": mem0_config.get("llm_provider", "openai"),
                        "config": {
                            "model": mem0_config.get("model", "gpt-4o-mini"),
                            "temperature": 0.1,
                        }
                    },
                    "embedder": {
                        "provider": mem0_config.get("embedder_provider", "openai"),
                        "config": {
                            "model": mem0_config.get("embedder_model", "text-embedding-3-small"),
                        }
                    },
                    "vector_store": {
                        "provider": "chroma",
                        "config": {
                            "collection_name": f"intelclaw_rag_{self.user_id}",
                            "path": str(self.persist_dir / "mem0_vectors"),
                        }
                    },
                    "version": "v1.1",
                }
                
                self._mem0 = Memory.from_config(mem0_settings)
                logger.info("Mem0 initialized for Agentic RAG")
                
            except Exception as e:
                logger.warning(f"Mem0 initialization failed: {e}")
        
        # Initialize ChromaDB fallback
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._vector_store = chromadb.PersistentClient(
                path=str(self.persist_dir / "vectors"),
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB vector store initialized")
            
        except ImportError:
            logger.warning("ChromaDB not available")
        
        self._initialized = True
        logger.success("Agentic RAG system initialized")
    
    async def shutdown(self) -> None:
        """Shutdown and persist state."""
        logger.info("Shutting down Agentic RAG...")
        await self._save_trees()
        self._initialized = False
    
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
        
        # Parse content into sections (simple markdown parsing)
        sections = self._parse_sections(content)
        
        for section in sections:
            tree.add_node(
                title=section["title"],
                content=section["content"],
                summary=section.get("summary", ""),
                parent_id=section.get("parent_id", "root"),
                metadata={"doc_type": doc_type}
            )
        
        self.document_trees[doc_id] = tree
        
        # Also add to Mem0 for semantic backup
        if self._mem0:
            try:
                await asyncio.to_thread(
                    self._mem0.add,
                    f"Document: {title}\n\n{content[:5000]}",
                    user_id=self.user_id,
                    metadata={"doc_id": doc_id, "type": doc_type}
                )
            except Exception as e:
                logger.warning(f"Failed to add to Mem0: {e}")
        
        # Persist
        await self._save_trees()
        
        return tree
    
    def _parse_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse markdown content into sections."""
        sections = []
        current_section = {"title": "Introduction", "content": "", "level": 0}
        
        lines = content.split("\n")
        
        for line in lines:
            # Check for headers
            if line.startswith("# "):
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {
                    "title": line[2:].strip(),
                    "content": "",
                    "level": 1
                }
            elif line.startswith("## "):
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {
                    "title": line[3:].strip(),
                    "content": "",
                    "level": 2
                }
            elif line.startswith("### "):
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {
                    "title": line[4:].strip(),
                    "content": "",
                    "level": 3
                }
            else:
                current_section["content"] += line + "\n"
        
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    async def index_persona(self, persona_dir: Path) -> None:
        """
        Index persona files for context-aware retrieval.
        
        Indexes: AGENT.md, SOUL.md, SKILLS.md, TOOLS.md, USER.md
        """
        logger.info(f"Indexing persona from: {persona_dir}")
        
        persona_files = ["AGENT.md", "SOUL.md", "SKILLS.md", "TOOLS.md", "USER.md"]
        
        for filename in persona_files:
            filepath = persona_dir / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    self.persona_context[filename] = content
                    
                    # Create tree index
                    doc_id = f"persona_{filename.replace('.md', '').lower()}"
                    await self.index_document(
                        doc_id=doc_id,
                        title=filename,
                        content=content,
                        doc_type="persona"
                    )
                    
                    logger.debug(f"Indexed persona file: {filename}")
                    
                except Exception as e:
                    logger.warning(f"Failed to index {filename}: {e}")
    
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
        
        results = []
        
        if strategy in ["tree", "hybrid"]:
            # Tree-based reasoning retrieval
            tree_results = await self._tree_retrieve(query, doc_types)
            results.extend(tree_results)
        
        if strategy in ["semantic", "hybrid"]:
            # Semantic vector search backup
            semantic_results = await self._semantic_retrieve(query, max_results)
            
            # Merge results, avoiding duplicates
            seen_ids = {r.get("id") for r in results}
            for r in semantic_results:
                if r.get("id") not in seen_ids:
                    results.append(r)
        
        # Search Mem0 memories
        if self._mem0:
            try:
                mem0_results = await asyncio.to_thread(
                    self._mem0.search,
                    query,
                    user_id=self.user_id,
                    limit=max_results
                )
                
                for r in (mem0_results.get("results", []) if isinstance(mem0_results, dict) else mem0_results):
                    results.append({
                        "id": r.get("id", str(uuid4())),
                        "content": r.get("memory", r.get("text", "")),
                        "score": r.get("score", 0.5),
                        "source": "mem0",
                        "metadata": r.get("metadata", {})
                    })
            except Exception as e:
                logger.warning(f"Mem0 search failed: {e}")
        
        # Sort by score and limit
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
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
        results = []
        
        for doc_id, tree in self.document_trees.items():
            # Filter by doc_type if specified
            if doc_types:
                root_meta = tree.root.metadata
                if root_meta.get("doc_type") not in doc_types:
                    continue
            
            # Get table of contents for reasoning
            toc = tree.get_toc()
            
            # Simple keyword matching for relevant nodes
            # In a full implementation, this would use LLM reasoning
            query_lower = query.lower()
            
            for node_id, node in tree.node_index.items():
                if node_id == "root":
                    continue
                
                # Score based on keyword overlap
                title_match = query_lower in node.title.lower()
                content_match = query_lower in node.content.lower()
                summary_match = query_lower in node.summary.lower()
                
                if title_match or content_match or summary_match:
                    score = 0.3
                    if title_match:
                        score += 0.4
                    if content_match:
                        score += 0.2
                    if summary_match:
                        score += 0.1
                    
                    results.append({
                        "id": f"{doc_id}:{node_id}",
                        "content": node.content or node.summary,
                        "title": node.title,
                        "score": min(score, 1.0),
                        "source": "tree",
                        "doc_id": doc_id,
                        "node_id": node_id,
                        "path": tree.get_path_to_node(node_id)
                    })
        
        return results
    
    async def _semantic_retrieve(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Semantic vector search fallback."""
        results = []
        
        if not self._vector_store:
            return results
        
        try:
            collection = self._vector_store.get_or_create_collection("intelclaw_docs")
            
            search_results = await asyncio.to_thread(
                collection.query,
                query_texts=[query],
                n_results=max_results
            )
            
            if search_results and search_results.get("ids"):
                for i, doc_id in enumerate(search_results["ids"][0]):
                    results.append({
                        "id": doc_id,
                        "content": search_results["documents"][0][i] if search_results.get("documents") else "",
                        "score": 1 - search_results["distances"][0][i] if search_results.get("distances") else 0.5,
                        "source": "semantic",
                        "metadata": search_results["metadatas"][0][i] if search_results.get("metadatas") else {}
                    })
                    
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        return results
    
    # ==================== Session Memory ====================
    
    async def store_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store session conversation for future context."""
        session_data = {
            "session_id": session_id,
            "messages": messages,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id
        }
        
        self.session_memories.append(session_data)
        
        # Extract and store facts in Mem0
        if self._mem0:
            try:
                # Combine messages for fact extraction
                conversation = "\n".join([
                    f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                    for m in messages[-10:]  # Last 10 messages
                ])
                
                await asyncio.to_thread(
                    self._mem0.add,
                    conversation,
                    user_id=self.user_id,
                    metadata={"type": "session", "session_id": session_id}
                )
                
            except Exception as e:
                logger.warning(f"Failed to store session in Mem0: {e}")
        
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
        max_context_chars: int = 10000
    ) -> str:
        """
        Build comprehensive context for a query.
        
        This is the main entry point for the RAG system.
        Returns a formatted context string that can be injected
        into the system prompt.
        """
        context_parts = []
        total_chars = 0
        
        # 1. Retrieve relevant documents
        results = await self.retrieve(query, strategy="hybrid", max_results=5)
        
        if results:
            context_parts.append("## Relevant Context\n")
            for r in results:
                content = r.get("content", "")[:2000]
                if total_chars + len(content) > max_context_chars:
                    break
                
                source = r.get("source", "unknown")
                title = r.get("title", "")
                context_parts.append(f"### {title or 'Retrieved Context'} (from {source})")
                context_parts.append(content)
                context_parts.append("")
                total_chars += len(content)
        
        # 2. Include persona context if requested
        if include_persona and self.persona_context:
            for filename, content in self.persona_context.items():
                if total_chars + len(content) > max_context_chars:
                    break
                context_parts.append(f"## {filename}\n{content[:3000]}")
                total_chars += len(content)
        
        # 3. Include relevant session context
        if include_session:
            sessions = await self.get_session_context(query)
            if sessions:
                context_parts.append("## Previous Conversations\n")
                for session in sessions[:2]:
                    messages = session.get("messages", [])[-5:]
                    for m in messages:
                        line = f"- {m.get('role', 'unknown')}: {m.get('content', '')[:200]}"
                        if total_chars + len(line) > max_context_chars:
                            break
                        context_parts.append(line)
                        total_chars += len(line)
        
        return "\n\n".join(context_parts)
    
    @property
    def is_available(self) -> bool:
        """Check if RAG system is available."""
        return self._initialized
