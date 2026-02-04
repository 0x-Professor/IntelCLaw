"""
Web Tools - Web scraping and content extraction.
"""

import asyncio
from typing import Any, Dict, Optional

from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory, ToolPermission

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class WebScrapeTool(BaseTool):
    """Scrape web page content."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_scrape",
            description="Fetch and extract content from a web page.",
            category=ToolCategory.SEARCH,
            permissions=[ToolPermission.NETWORK],
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Extract text content only",
                        "default": True
                    }
                },
                "required": ["url"]
            },
            returns="string",
            rate_limit=20
        )
    
    async def execute(
        self,
        url: str,
        extract_text: bool = True,
        **kwargs
    ) -> ToolResult:
        """Scrape web page."""
        if not HTTPX_AVAILABLE:
            return ToolResult(success=False, error="httpx not available")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "IntelCLaw/1.0"},
                    follow_redirects=True
                )
                response.raise_for_status()
                
                content = response.text
                
                if extract_text:
                    # Simple text extraction
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(content, "html.parser")
                        
                        # Remove script and style elements
                        for element in soup(["script", "style", "nav", "footer", "header"]):
                            element.decompose()
                        
                        # Get text
                        text = soup.get_text(separator="\n", strip=True)
                        
                        # Clean up whitespace
                        lines = [line.strip() for line in text.split("\n") if line.strip()]
                        content = "\n".join(lines)
                        
                    except ImportError:
                        # Fallback: basic regex cleanup
                        import re
                        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
                        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
                        content = re.sub(r'<[^>]+>', ' ', content)
                        content = re.sub(r'\s+', ' ', content).strip()
                
                return ToolResult(
                    success=True,
                    data=content[:50000],  # Limit size
                    metadata={
                        "url": url,
                        "status_code": response.status_code,
                        "content_length": len(content)
                    }
                )
                
        except Exception as e:
            logger.error(f"Web scrape failed: {e}")
            return ToolResult(success=False, error=str(e))
