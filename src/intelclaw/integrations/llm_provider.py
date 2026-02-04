"""
GitHub Copilot LLM Provider - Use Copilot as the LLM backend.

This module enables IntelCLaw to use GitHub Copilot's language models
directly through VS Code's Copilot extension API.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator

from loguru import logger


class CopilotLLM:
    """
    LLM provider that uses GitHub Copilot.
    
    This allows IntelCLaw to leverage your existing GitHub Copilot
    subscription without needing separate API keys.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize Copilot LLM.
        
        Args:
            model: Model to use (gpt-4o, gpt-4o-mini, claude-3.5-sonnet, o1-preview)
        """
        self.model = model
        self._initialized = False
        self._copilot_token: Optional[str] = None
        self._session_id: Optional[str] = None
    
    async def initialize(self) -> bool:
        """Initialize connection to Copilot."""
        # Try to get Copilot token from VS Code
        token = await self._get_copilot_token()
        if token:
            self._copilot_token = token
            self._initialized = True
            logger.info(f"Copilot LLM initialized with model: {self.model}")
            return True
        
        logger.warning("Could not initialize Copilot LLM - token not found")
        return False
    
    async def _get_copilot_token(self) -> Optional[str]:
        """Get GitHub Copilot token from VS Code."""
        # Check environment variable first
        token = os.environ.get("GITHUB_COPILOT_TOKEN")
        if token:
            return token
        
        # Try to read from VS Code Copilot extension storage
        possible_paths = [
            Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Code" / "User" / "globalStorage" / "github.copilot" / "hosts.json",
            Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Code" / "User" / "globalStorage" / "github.copilot-chat" / "token.json",
            Path(os.path.expanduser("~")) / ".config" / "github-copilot" / "hosts.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        # Extract token from hosts.json format
                        for key, value in data.items():
                            if "github.com" in key and isinstance(value, dict):
                                if "oauth_token" in value:
                                    return value["oauth_token"]
                        # Or direct token
                        if "token" in data:
                            return data["token"]
                except Exception as e:
                    logger.debug(f"Could not read token from {path}: {e}")
        
        # Try GitHub CLI token
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        
        return None
    
    async def ainvoke(self, prompt: str, **kwargs) -> "CopilotResponse":
        """
        Invoke the Copilot LLM.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters
            
        Returns:
            Response object with content
        """
        if not self._initialized:
            await self.initialize()
        
        # Use the Copilot API endpoint
        messages = kwargs.get("messages", [])
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self._call_copilot_api(messages)
            return CopilotResponse(content=response)
        except Exception as e:
            logger.error(f"Copilot API error: {e}")
            # Fallback to local processing or error message
            return CopilotResponse(content=f"Error calling Copilot: {e}")
    
    async def _call_copilot_api(self, messages: List[Dict[str, str]]) -> str:
        """Call the GitHub Copilot API."""
        import aiohttp
        
        # GitHub Copilot Chat API endpoint
        api_url = "https://api.githubcopilot.com/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self._copilot_token}",
            "Content-Type": "application/json",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot-chat/0.12.0",
            "Openai-Organization": "github-copilot",
            "User-Agent": "GitHubCopilotChat/0.12.0",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
            "stream": False,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status == 401:
                    raise Exception("Copilot authentication failed - please ensure you're logged into GitHub Copilot")
                else:
                    error_text = await response.text()
                    raise Exception(f"Copilot API error {response.status}: {error_text}")
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream response from Copilot."""
        # For now, just yield the full response
        response = await self.ainvoke(prompt, **kwargs)
        yield response.content
    
    def bind_tools(self, tools: List[Any]) -> "CopilotLLM":
        """Bind tools for function calling."""
        # Return self for chaining (tools handled separately)
        return self


class CopilotResponse:
    """Response from Copilot LLM."""
    
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self) -> str:
        return self.content


class LLMProvider:
    """
    Unified LLM provider that can use Copilot or direct APIs.
    
    Priority:
    1. GitHub Copilot (if available)
    2. OpenAI API (if key provided)
    3. Anthropic API (if key provided)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM provider."""
        self.config = config or {}
        self._llm = None
        self._provider_name = "none"
    
    async def initialize(self) -> bool:
        """Initialize the best available LLM provider."""
        
        # 1. Try GitHub Copilot first (no API key needed)
        copilot = CopilotLLM(model=self.config.get("model", "gpt-4o"))
        if await copilot.initialize():
            self._llm = copilot
            self._provider_name = "copilot"
            logger.info("Using GitHub Copilot as LLM provider")
            return True
        
        # 2. Try OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY") or self.config.get("openai_api_key")
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.config.get("model", "gpt-4o"),
                    api_key=openai_key,
                    temperature=self.config.get("temperature", 0.1),
                )
                self._provider_name = "openai"
                logger.info("Using OpenAI as LLM provider")
                return True
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
        
        # 3. Try Anthropic
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.get("anthropic_api_key")
        if anthropic_key:
            try:
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(
                    model="claude-3-5-sonnet-20241022",
                    api_key=anthropic_key,
                    temperature=self.config.get("temperature", 0.1),
                )
                self._provider_name = "anthropic"
                logger.info("Using Anthropic as LLM provider")
                return True
            except Exception as e:
                logger.warning(f"Anthropic initialization failed: {e}")
        
        logger.error("No LLM provider available")
        return False
    
    @property
    def llm(self):
        """Get the underlying LLM."""
        return self._llm
    
    @property
    def provider(self) -> str:
        """Get the current provider name."""
        return self._provider_name
    
    @property
    def active_provider(self) -> str:
        """Get the active provider name."""
        return self._provider_name
    
    @property
    def available_providers(self) -> List[str]:
        """List potentially available providers."""
        providers = []
        
        # Check Copilot
        if os.environ.get("GITHUB_COPILOT_TOKEN") or self._provider_name == "copilot":
            providers.append("copilot")
        
        # Check OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            providers.append("openai")
        
        # Check Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            providers.append("anthropic")
        
        return providers if providers else ["copilot"]  # Default to copilot attempt
    
    async def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM."""
        if not self._llm:
            return "LLM not initialized"
        
        response = await self._llm.ainvoke(prompt, **kwargs)
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with the LLM."""
        if not self._llm:
            return "LLM not initialized"
        
        if self._provider_name == "copilot":
            response = await self._llm.ainvoke("", messages=messages, **kwargs)
        else:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            lc_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))
            
            response = await self._llm.ainvoke(lc_messages, **kwargs)
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)
