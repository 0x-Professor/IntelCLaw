"""
Coding Agent - Specialized for code generation and debugging.

Handles programming tasks, code review, and debugging assistance.
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from intelclaw.agent.base import (
    AgentContext,
    AgentResponse,
    AgentStatus,
    BaseAgent,
)
from intelclaw.integrations.llm_provider import LLMProvider

if TYPE_CHECKING:
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.tools.registry import ToolRegistry


class CodingAgent(BaseAgent):
    """
    Agent specialized in code generation and programming assistance.
    
    Capabilities:
    - Code generation in multiple languages
    - Debugging assistance
    - Code review and optimization
    - Documentation generation
    - Explaining code
    """
    
    CODING_KEYWORDS = [
        "code", "program", "function", "debug", "error", "python",
        "javascript", "typescript", "script", "implement", "fix bug",
        "refactor", "optimize", "class", "method", "algorithm",
        "api", "database", "sql", "html", "css", "react", "test"
    ]
    
    LANGUAGE_PATTERNS = {
        "python": ["def ", "import ", "class ", ".py", "python"],
        "javascript": ["function", "const ", "let ", "var ", ".js", "javascript"],
        "typescript": [".ts", "typescript", "interface ", "type "],
        "sql": ["select", "insert", "update", "delete", "from", "where"],
        "bash": ["#!/", ".sh", "bash", "shell"],
    }
    
    def __init__(
        self,
        memory: Optional["MemoryManager"] = None,
        tools: Optional["ToolRegistry"] = None,
    ):
        """Initialize coding agent."""
        super().__init__(
            name="Coding Agent",
            description="Specialized in code generation, debugging, and programming assistance",
            memory=memory,
            tools=tools,
        )
        
        # LLM will be initialized asynchronously
        self._llm_provider: Optional[LLMProvider] = None
        self._llm = None
    
    async def _ensure_llm(self):
        """Ensure LLM is initialized."""
        if self._llm is None:
            self._llm_provider = LLMProvider({"model": "gpt-4o", "temperature": 0.1})
            await self._llm_provider.initialize()
            self._llm = self._llm_provider.llm
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a coding request.
        
        Flow:
        1. Understand the coding task
        2. Detect programming language
        3. Generate/fix/explain code
        4. Provide formatted response
        """
        start_time = time.time()
        self.clear_thoughts()
        self.status = AgentStatus.THINKING
        
        # Ensure LLM is initialized
        await self._ensure_llm()
        
        try:
            # Detect language from context
            language = self._detect_language(context)
            
            await self.think(
                f"Processing coding request for {language or 'general programming'}",
                step=1
            )
            
            # Build specialized prompt
            system_prompt = self._get_coding_prompt(language)
            
            # Check if there's code in screen context
            screen_code = ""
            if context.screen_context and context.screen_context.get("text"):
                screen_code = f"\n\nCode visible on screen:\n{context.screen_context['text'][:2000]}"
            
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{context.user_message}{screen_code}"),
            ]
            
            response = await self._llm.ainvoke(messages)
            
            latency = (time.time() - start_time) * 1000
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                answer=response.content,
                thoughts=self._current_thoughts.copy(),
                tools_used=["code_generation"],
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Coding agent error: {e}")
            self.status = AgentStatus.ERROR
            
            return AgentResponse(
                answer=f"I encountered an error with the coding task: {str(e)}",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    async def can_handle(self, context: AgentContext) -> float:
        """Determine if this is a coding request."""
        message_lower = context.user_message.lower()
        
        # Check for coding keywords
        keyword_matches = sum(
            1 for kw in self.CODING_KEYWORDS
            if kw in message_lower
        )
        
        # Check for code patterns
        code_pattern_matches = sum(
            1 for patterns in self.LANGUAGE_PATTERNS.values()
            for pattern in patterns
            if pattern in message_lower
        )
        
        # Check for code block indicators
        if "```" in context.user_message or "def " in context.user_message:
            keyword_matches += 3
        
        confidence = min((keyword_matches + code_pattern_matches) / 5.0, 1.0)
        return confidence
    
    def _detect_language(self, context: AgentContext) -> Optional[str]:
        """Detect the programming language from context."""
        message_lower = context.user_message.lower()
        
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            if any(p in message_lower for p in patterns):
                return lang
        
        # Check screen context
        if context.screen_context and context.screen_context.get("text"):
            screen_text = context.screen_context["text"].lower()
            for lang, patterns in self.LANGUAGE_PATTERNS.items():
                if any(p in screen_text for p in patterns):
                    return lang
        
        return None
    
    def _get_coding_prompt(self, language: Optional[str]) -> str:
        """Get specialized prompt for coding tasks."""
        
        base_prompt = """You are an expert programmer and coding assistant.

## Guidelines:
1. Write clean, well-documented, production-ready code
2. Follow best practices and design patterns
3. Include error handling where appropriate
4. Explain complex logic with comments
5. Suggest improvements when relevant

## Response Format:
- Use markdown code blocks with language tags
- Provide brief explanation before code
- Note any assumptions made
- Suggest testing approach if applicable
"""
        
        if language:
            base_prompt += f"\n\n## Primary Language: {language}\n"
            
            lang_specific = {
                "python": "Follow PEP 8 style guide. Use type hints. Prefer f-strings.",
                "javascript": "Use modern ES6+ syntax. Handle async properly.",
                "typescript": "Use strict typing. Define interfaces for complex types.",
                "sql": "Use parameterized queries. Consider performance.",
            }
            
            if language in lang_specific:
                base_prompt += lang_specific[language]
        
        return base_prompt
