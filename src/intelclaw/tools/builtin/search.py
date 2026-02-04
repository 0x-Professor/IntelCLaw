"""
Search Tools - Web search via Tavily and other providers.
"""

import os
from typing import Any, Dict, List, Optional

from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory, ToolPermission

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


class TavilySearchTool(BaseTool):
    """Web search using Tavily API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key."""
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client: Optional[Any] = None
        
        if TAVILY_AVAILABLE and self._api_key:
            self._client = TavilyClient(api_key=self._api_key)
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="tavily_search",
            description="Search the web for current information using Tavily. Returns relevant web results with snippets.",
            category=ToolCategory.SEARCH,
            permissions=[ToolPermission.NETWORK],
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Search depth (default: basic)",
                        "default": "basic"
                    }
                },
                "required": ["query"]
            },
            returns="list[SearchResult]",
            examples=[
                {
                    "query": "latest AI news 2024",
                    "max_results": 5
                }
            ],
            rate_limit=30  # 30 calls per minute
        )
    
    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        **kwargs
    ) -> ToolResult:
        """Execute web search."""
        if not self._client:
            return ToolResult(
                success=False,
                error="Tavily client not initialized. Set TAVILY_API_KEY."
            )
        
        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
            )
            
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0),
                })
            
            return ToolResult(
                success=True,
                data=results,
                metadata={"query": query, "result_count": len(results)}
            )
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return ToolResult(success=False, error=str(e))
