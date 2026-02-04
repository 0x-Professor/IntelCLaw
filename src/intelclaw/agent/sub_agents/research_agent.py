"""
Research Agent - Specialized for web search and information gathering.

Uses Tavily and other search tools to find and synthesize information.
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from intelclaw.agent.base import (
    AgentContext,
    AgentResponse,
    AgentStatus,
    AgentThought,
    BaseAgent,
)

if TYPE_CHECKING:
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.tools.registry import ToolRegistry


class ResearchAgent(BaseAgent):
    """
    Agent specialized in research and information gathering.
    
    Capabilities:
    - Web search via Tavily
    - Content summarization
    - Fact-checking
    - Source citation
    """
    
    RESEARCH_KEYWORDS = [
        "search", "find", "look up", "research", "what is", "who is",
        "explain", "tell me about", "information", "news", "article",
        "learn", "discover", "investigate", "definition", "meaning"
    ]
    
    def __init__(
        self,
        memory: Optional["MemoryManager"] = None,
        tools: Optional["ToolRegistry"] = None,
    ):
        """Initialize research agent."""
        super().__init__(
            name="Research Agent",
            description="Specialized in web search, information gathering, and content synthesis",
            memory=memory,
            tools=tools,
        )
        
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a research request.
        
        Flow:
        1. Analyze the research question
        2. Search for relevant information
        3. Synthesize findings
        4. Provide cited response
        """
        start_time = time.time()
        self.clear_thoughts()
        self.status = AgentStatus.THINKING
        
        try:
            # Step 1: Analyze the question
            await self.think(
                f"Analyzing research question: {context.user_message}",
                step=1
            )
            
            # Step 2: Perform search
            search_results = await self._perform_search(context.user_message)
            
            await self.act(
                action="web_search",
                action_input={"query": context.user_message},
                step=2
            )
            await self.observe(f"Found {len(search_results)} results", step=2)
            
            # Step 3: Synthesize response
            await self.think("Synthesizing information from search results", step=3)
            
            response = await self._synthesize_response(
                context.user_message,
                search_results
            )
            
            latency = (time.time() - start_time) * 1000
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                answer=response,
                thoughts=self._current_thoughts.copy(),
                tools_used=["web_search"],
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            self.status = AgentStatus.ERROR
            
            return AgentResponse(
                answer=f"I couldn't complete the research: {str(e)}",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    async def can_handle(self, context: AgentContext) -> float:
        """Determine if this is a research request."""
        message_lower = context.user_message.lower()
        
        # Check for research keywords
        keyword_matches = sum(
            1 for kw in self.RESEARCH_KEYWORDS
            if kw in message_lower
        )
        
        # Higher score for question patterns
        if any(message_lower.startswith(q) for q in ["what", "who", "how", "why", "when", "where"]):
            keyword_matches += 2
        
        confidence = min(keyword_matches / 5.0, 1.0)
        return confidence
    
    async def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using available tools."""
        if not self.tools:
            return []
        
        try:
            result = await self.tools.execute("tavily_search", {"query": query})
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return []
    
    async def _synthesize_response(
        self,
        question: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Synthesize search results into a coherent response."""
        
        # Format search results
        results_text = ""
        for i, result in enumerate(search_results[:5], 1):
            title = result.get("title", "No title")
            content = result.get("content", result.get("snippet", ""))[:500]
            url = result.get("url", "")
            results_text += f"\n{i}. **{title}**\n{content}\nSource: {url}\n"
        
        if not results_text:
            return "I couldn't find relevant information for your query. Could you try rephrasing?"
        
        # Use LLM to synthesize
        system_prompt = """You are a research assistant. Synthesize the search results 
into a clear, accurate response. Always cite sources with [n] notation.
Be concise but comprehensive. If information is uncertain, say so."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}\n\nSearch Results:{results_text}"),
        ]
        
        response = await self._llm.ainvoke(messages)
        return response.content
