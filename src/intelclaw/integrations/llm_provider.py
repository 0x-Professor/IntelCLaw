"""
GitHub Models LLM Provider - Use GitHub Models API as the LLM backend.

This module enables IntelCLaw to use GitHub's AI models
through the GitHub Models API (https://github.com/marketplace/models).

This works with any GitHub account and uses the free GitHub Models API.
You need a GitHub Personal Access Token (PAT) with `models:read` permission.

To get started:
1. Visit https://github.com/marketplace/models and accept the terms
2. Go to https://github.com/settings/tokens?type=beta (Fine-grained tokens)
3. Create a new token with 'Models' read permission (under Account permissions)
4. Set the token in .env: GITHUB_TOKEN=your_token_here

Available model families:
- GPT (OpenAI): GPT-4.1, GPT-4o, GPT-5 series
- Claude (Anthropic): Haiku 4.5, Sonnet 4/4.5, Opus 4.5
- Gemini (Google): 2.5 Pro, 3 Flash, 3 Pro
- Other: Grok, Raptor, Llama, Mistral, DeepSeek
"""

import asyncio
import json
import os
import subprocess
import webbrowser
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator
from dotenv import load_dotenv

from loguru import logger

# Load environment variables from .env file
load_dotenv()


# GitHub OAuth App Client ID for Copilot (used for OAuth device flow)
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"

# GitHub Models API endpoint - uses Azure AI inference
GITHUB_MODELS_API_URL = "https://models.inference.ai.azure.com/chat/completions"

# =============================================================================
# MODEL CONFIGURATIONS - Updated February 2026
# =============================================================================

# GitHub Copilot Models - Available through GitHub Models API
GITHUB_MODELS = {
    # =========================================================================
    # OpenAI GPT Models (Latest)
    # =========================================================================
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5": "gpt-5",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.2": "gpt-5.2",
    # GPT-5 Codex Series (Optimized for code)
    "gpt-5-codex": "gpt-5-codex",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    
    # =========================================================================
    # Anthropic Claude Models
    # =========================================================================
    "claude-haiku-4.5": "claude-haiku-4.5",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-sonnet-4.5": "claude-sonnet-4.5",
    "claude-opus-4.5": "claude-opus-4.5",
    
    # =========================================================================
    # Google Gemini Models
    # =========================================================================
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3-pro": "gemini-3-pro",
    
    # =========================================================================
    # Other Models
    # =========================================================================
    "grok-code-fast-1": "grok-code-fast-1",
    "raptor-mini": "raptor-mini",
    
    # =========================================================================
    # Legacy/Fallback Models (Still Available)
    # =========================================================================
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4": "gpt-4",
    "o1-preview": "o1-preview",
    "o1-mini": "o1-mini",
    # Meta Llama Models
    "llama-3.3-70b": "Meta-Llama-3.3-70B-Instruct",
    "llama-3.2-90b": "Llama-3.2-90B-Vision-Instruct",
    "llama-3.1-405b": "Meta-Llama-3.1-405B-Instruct",
    # Mistral Models
    "mistral-large": "Mistral-Large-2411",
    "mistral-small": "Mistral-Small-24B-Instruct-2501",
    # DeepSeek Models
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3": "DeepSeek-V3",
    # Microsoft Phi Models
    "phi-4": "Phi-4",
    "phi-3.5-moe": "Phi-3.5-MoE-instruct",
}

# Model categories for easy selection
MODEL_CATEGORIES = {
    "gpt": ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-5-mini", "gpt-5", "gpt-5.1", "gpt-5.2"],
    "gpt-codex": ["gpt-5-codex", "gpt-5.1-codex", "gpt-5.1-codex-max", "gpt-5.1-codex-mini"],
    "claude": ["claude-haiku-4.5", "claude-sonnet-4", "claude-sonnet-4.5", "claude-opus-4.5"],
    "gemini": ["gemini-2.5-pro", "gemini-3-flash", "gemini-3-pro"],
    "other": ["grok-code-fast-1", "raptor-mini"],
    "open-source": ["llama-3.3-70b", "mistral-large", "deepseek-r1", "phi-4"],
}

# Default model - fast and capable
DEFAULT_MODEL = os.getenv("INTELCLAW_DEFAULT_MODEL", "gpt-4o")

# Heavy task models (for complex reasoning)
HEAVY_TASK_MODELS = ["claude-opus-4.5", "gpt-5.1", "gpt-5.1-codex-max", "gemini-3-pro"]

# Coding-optimized models
CODING_MODELS = ["gpt-5-codex", "gpt-5.1-codex", "gpt-5.1-codex-max", "grok-code-fast-1"]


def get_github_model_id(model: str) -> str:
    """Convert a short model name to GitHub Models format."""
    # If already in provider/model format, return as-is
    if "/" in model:
        return model
    # Look up in mapping
    return GITHUB_MODELS.get(model, f"openai/{model}")


class GitHubAuth:
    """
    GitHub OAuth Device Flow Authentication.
    
    This allows users to authenticate with GitHub to use GitHub Models API
    without needing to manually copy tokens.
    
    The GitHub Models API works with any GitHub account and uses regular
    GitHub OAuth tokens (no special Copilot subscription required for basic usage).
    """
    
    TOKEN_FILE = Path.home() / ".intelclaw" / "github_token.json"
    
    @classmethod
    async def authenticate(cls) -> Optional[str]:
        """
        Perform GitHub OAuth device flow authentication.
        
        Returns:
            Access token if successful, None otherwise
        """
        import aiohttp
        
        # Check for existing token first
        existing_token = cls._load_saved_token()
        if existing_token:
            # Verify token is still valid
            if await cls._verify_token(existing_token):
                return existing_token
            logger.info("Saved token expired, re-authenticating...")
        
        print("\n" + "=" * 60)
        print("🔐 GitHub Models API - Personal Access Token Required")
        print("=" * 60)
        print("\nThe GitHub Models API requires a Personal Access Token (PAT).")
        print("\n📌 To create your token:")
        print("   1. Visit: https://github.com/settings/tokens?type=beta")
        print("   2. Click 'Generate new token'")
        print("   3. Give it a name like 'IntelCLaw'")
        print("   4. Under 'Account permissions', find 'Models' and set to 'Read'")
        print("   5. Click 'Generate token' and copy it")
        print("\n" + "-" * 60)
        
        # Open browser to token page
        token_url = "https://github.com/settings/tokens?type=beta"
        print(f"\n🌐 Opening: {token_url}")
        webbrowser.open(token_url)
        
        # Prompt user for token
        print("\n📋 Paste your token below (input hidden):")
        try:
            import getpass
            token = getpass.getpass("Token: ").strip()
        except Exception:
            # Fallback if getpass doesn't work
            token = input("Token: ").strip()
        
        if not token:
            print("\n❌ No token provided.")
            return None
        
        # Verify the token works
        if await cls._verify_token(token):
            cls._save_token(token)
            print("\n✅ Token verified and saved!")
            print("=" * 60 + "\n")
            return token
        else:
            print("\n❌ Token verification failed. Please ensure:")
            print("   - The token has 'Models: Read' permission")
            print("   - You've accepted the GitHub Models terms at:")
            print("     https://github.com/marketplace/models")
            return None
    
    @classmethod
    def _save_token(cls, token: str) -> None:
        """Save token to file."""
        cls.TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        cls.TOKEN_FILE.write_text(json.dumps({
            "access_token": token,
            "saved_at": time.time()
        }), encoding="utf-8")
        # Set restrictive permissions
        try:
            os.chmod(cls.TOKEN_FILE, 0o600)
        except:
            pass
    
    @classmethod
    def _load_saved_token(cls) -> Optional[str]:
        """Load token from file."""
        if cls.TOKEN_FILE.exists():
            try:
                data = json.loads(cls.TOKEN_FILE.read_text(encoding="utf-8"))
                return data.get("access_token")
            except Exception as e:
                logger.debug(f"Could not load saved token: {e}")
        return None
    
    @classmethod
    async def _verify_token(cls, token: str) -> bool:
        """Verify token is still valid."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json"
                    }
                ) as response:
                    return response.status == 200
        except:
            return False
    
    @classmethod
    def clear_token(cls) -> None:
        """Clear saved token."""
        if cls.TOKEN_FILE.exists():
            cls.TOKEN_FILE.unlink()
        print("✅ GitHub token cleared.")


class CopilotLLM:
    """
    LLM provider that uses GitHub Models API.
    
    This allows IntelCLaw to leverage GitHub's free AI models API
    (https://models.github.ai) which provides access to:
    - OpenAI models (GPT-4o, GPT-4o-mini, GPT-4-turbo)
    - Meta Llama models (Llama 3.3, 3.2, 3.1)
    - Mistral models
    - DeepSeek models
    - Microsoft Phi models
    - And more!
    
    Works with any GitHub account - no Copilot subscription required
    for basic usage (rate limits apply).
    
    Default: gpt-4o-mini (fast, free tier friendly)
    """
    
    def __init__(self, model: str = None):
        """
        Initialize GitHub Models LLM.
        
        Args:
            model: Model to use (gpt-4o-mini default, gpt-4o, llama-3.3-70b, etc.)
        """
        self.model = model or DEFAULT_MODEL
        self._github_model_id = get_github_model_id(self.model)
        self._initialized = False
        self._github_token: Optional[str] = None
        self._session_id: Optional[str] = None
        self._anthropic_fallback = None  # Anthropic client for heavy tasks
        self._use_anthropic_for_heavy = False
    
    async def initialize(self) -> bool:
        """Initialize connection to GitHub Models API."""
        # Get GitHub OAuth token
        github_token = await self._get_github_token()
        if not github_token:
            logger.warning("Could not get GitHub token")
            return False
        
        self._github_token = github_token
        self._session_id = str(time.time_ns())
        
        # Verify the token works with GitHub API
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": f"Bearer {github_token}",
                        "Accept": "application/json",
                        "X-GitHub-Api-Version": "2022-11-28"
                    }
                ) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        logger.info(f"GitHub Models LLM initialized for user: {user_data.get('login', 'unknown')}")
                        logger.info(f"Using model: {self._github_model_id} (default: {DEFAULT_MODEL})")
                        self._initialized = True
                        
                        # Initialize Anthropic fallback if API key is available
                        await self._init_anthropic_fallback()
                        
                        return True
                    else:
                        logger.warning(f"GitHub token validation failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error validating GitHub token: {e}")
            return False
    
    async def _init_anthropic_fallback(self):
        """Initialize Anthropic as fallback for heavy tasks."""
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                self._anthropic_fallback = anthropic.AsyncAnthropic(api_key=anthropic_key)
                self._use_anthropic_for_heavy = True
                logger.info("Anthropic fallback enabled for heavy tasks (Claude 3.5 Sonnet)")
            except ImportError:
                logger.debug("anthropic package not installed, skipping fallback")
            except Exception as e:
                logger.debug(f"Could not initialize Anthropic fallback: {e}")
    
    async def _get_github_token(self) -> Optional[str]:
        """
        Get GitHub OAuth token from various sources.
        
        Priority order:
        1. Environment variables (COPILOT_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN)
        2. IntelCLaw auth profiles (new OpenClaw-style system)
        3. Legacy saved token
        4. VS Code Copilot extension storage
        5. GitHub CLI token
        6. Interactive authentication
        """
        # 1. Check environment variable first (COPILOT_GITHUB_TOKEN preferred, like OpenClaw)
        token = (
            os.environ.get("COPILOT_GITHUB_TOKEN") or 
            os.environ.get("GH_TOKEN") or 
            os.environ.get("GITHUB_TOKEN")
        )
        if token:
            logger.debug("Using token from environment variable")
            return token
        
        # 2. Check new auth profiles (OpenClaw-style)
        try:
            from intelclaw.cli.auth import AuthManager
            auth_manager = AuthManager()
            
            # Try github-models profile first (it's free and always works)
            profile = auth_manager.get_default_profile("github-models")
            if profile and profile.access_token and not profile.is_expired():
                logger.debug("Using token from auth profile: github-models")
                return profile.access_token
            
            # Try github-copilot profile
            profile = auth_manager.get_default_profile("github-copilot")
            if profile and profile.access_token and not profile.is_expired():
                logger.debug("Using token from auth profile: github-copilot")
                return profile.access_token
        except ImportError:
            logger.debug("Auth manager not available, skipping profile check")
        except Exception as e:
            logger.debug(f"Could not check auth profiles: {e}")
        
        # 3. Check legacy saved IntelCLaw token
        saved_token = GitHubAuth._load_saved_token()
        if saved_token:
            logger.debug("Using saved IntelCLaw token")
            return saved_token
        
        # 4. Try to read from VS Code Copilot extension storage
        possible_paths = [
            Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Code" / "User" / "globalStorage" / "github.copilot" / "hosts.json",
            Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Code" / "User" / "globalStorage" / "github.copilot-chat" / "hosts.json",
            Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Code - Insiders" / "User" / "globalStorage" / "github.copilot" / "hosts.json",
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
                                    logger.debug(f"Using token from VS Code: {path}")
                                    return value["oauth_token"]
                        # Or direct token
                        if "token" in data:
                            return data["token"]
                except Exception as e:
                    logger.debug(f"Could not read token from {path}: {e}")
        
        # 5. Try GitHub CLI token
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.debug("Using token from GitHub CLI")
                return result.stdout.strip()
        except Exception:
            pass
        
        # 6. No token found - trigger interactive auth
        logger.info("No GitHub token found, starting interactive authentication...")
        return await GitHubAuth.authenticate()
    
    async def ainvoke(self, input_data, **kwargs):
        """
        Invoke the Copilot LLM.
        
        Args:
            input_data: Can be a string prompt or list of LangChain messages
            **kwargs: Additional parameters
            
        Returns:
            LangChain AIMessage for compatibility with LangGraph
        """
        from langchain_core.messages import AIMessage
        
        if not self._initialized:
            await self.initialize()
        
        # Convert input to messages format
        messages = []
        
        # Check if input is LangChain messages (list of message objects)
        if isinstance(input_data, list):
            for msg in input_data:
                if hasattr(msg, 'content'):
                    # It's a LangChain message object
                    role = "user"
                    if msg.__class__.__name__ == "SystemMessage":
                        role = "system"
                    elif msg.__class__.__name__ == "AIMessage":
                        role = "assistant"
                    elif msg.__class__.__name__ == "HumanMessage":
                        role = "user"
                    messages.append({"role": role, "content": msg.content})
                elif isinstance(msg, dict):
                    messages.append(msg)
        elif isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            # Try to get content from object
            if hasattr(input_data, 'content'):
                messages = [{"role": "user", "content": input_data.content}]
            else:
                messages = [{"role": "user", "content": str(input_data)}]
        
        # Add any additional messages from kwargs
        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                if isinstance(msg, dict):
                    messages.append(msg)
                elif hasattr(msg, 'content'):
                    role = "user"
                    if msg.__class__.__name__ == "SystemMessage":
                        role = "system"
                    elif msg.__class__.__name__ == "AIMessage":
                        role = "assistant"
                    messages.append({"role": role, "content": msg.content})
        
        # Detect if this is a heavy task (complex reasoning required)
        is_heavy_task = kwargs.get("heavy_task", False) or self._detect_heavy_task(messages)
        
        try:
            # Use Anthropic for heavy tasks if available
            if is_heavy_task and self._use_anthropic_for_heavy and self._anthropic_fallback:
                logger.info("Using Anthropic Claude for heavy task")
                response = await self._call_anthropic_api(messages)
                return AIMessage(content=response)
            
            # Otherwise use GitHub Models API
            response = await self._call_github_models_api(messages)
            return AIMessage(content=response)
        except Exception as e:
            logger.error(f"Primary LLM error: {e}")
            
            # Try Anthropic fallback on failure
            if self._use_anthropic_for_heavy and self._anthropic_fallback:
                try:
                    logger.info("Falling back to Anthropic Claude")
                    response = await self._call_anthropic_api(messages)
                    return AIMessage(content=response)
                except Exception as fallback_error:
                    logger.error(f"Anthropic fallback also failed: {fallback_error}")
            
            return AIMessage(content=f"Error calling LLM API: {e}")
    
    def _detect_heavy_task(self, messages: List[Dict[str, str]]) -> bool:
        """Detect if messages indicate a heavy/complex task."""
        # Check for indicators of complex tasks
        heavy_indicators = [
            "analyze", "explain in detail", "write a full", "implement",
            "create a complete", "debug", "refactor", "architecture",
            "complex", "comprehensive", "multi-step", "algorithm"
        ]
        
        text = " ".join(msg.get("content", "").lower() for msg in messages)
        
        # Heavy if text is long (>1000 chars) or contains indicators
        if len(text) > 1000:
            return True
        
        for indicator in heavy_indicators:
            if indicator in text:
                return True
        
        return False
    
    async def _call_anthropic_api(self, messages: List[Dict[str, str]]) -> str:
        """Call Anthropic API for heavy tasks."""
        if not self._anthropic_fallback:
            raise Exception("Anthropic client not initialized")
        
        # Convert messages to Anthropic format
        system_msg = None
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_msg = content
            else:
                anthropic_messages.append({
                    "role": "user" if role in ["user", "human"] else "assistant",
                    "content": content
                })
        
        # Use Claude 3.5 Sonnet for heavy tasks
        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 8192,
            "messages": anthropic_messages
        }
        if system_msg:
            kwargs["system"] = system_msg
        
        response = await self._anthropic_fallback.messages.create(**kwargs)
        return response.content[0].text
    
    async def _call_github_models_api(self, messages: List[Dict[str, str]], include_tools: bool = True) -> Dict[str, Any]:
        """
        Call the GitHub Models API (Azure AI inference endpoint).
        
        Returns:
            Dict with 'content' and optionally 'tool_calls'
        """
        import aiohttp
        
        if not self._github_token:
            raise Exception("GitHub token not available - set GITHUB_TOKEN environment variable")
        
        headers = {
            "Authorization": f"Bearer {self._github_token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self._github_model_id,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        
        # Add tools if bound and requested
        if include_tools and hasattr(self, '_tools_schema') and self._tools_schema:
            payload["tools"] = self._tools_schema
            payload["tool_choice"] = "auto"
        
        logger.debug(f"Calling GitHub Models API with model: {self._github_model_id}")
        if include_tools and hasattr(self, '_tools_schema'):
            logger.debug(f"With {len(self._tools_schema)} tools available")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GITHUB_MODELS_API_URL, 
                headers=headers, 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    data = json.loads(response_text)
                    message = data["choices"][0]["message"]
                    
                    result = {
                        "content": message.get("content", ""),
                        "tool_calls": None
                    }
                    
                    # Parse tool calls if present
                    if "tool_calls" in message and message["tool_calls"]:
                        result["tool_calls"] = []
                        for tc in message["tool_calls"]:
                            tool_call = {
                                "id": tc.get("id", f"call_{time.time_ns()}"),
                                "name": tc["function"]["name"],
                                "args": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                            }
                            result["tool_calls"].append(tool_call)
                        logger.info(f"LLM requested {len(result['tool_calls'])} tool calls")
                    
                    return result
                elif response.status == 401:
                    logger.warning("GitHub Models API returned 401 - unauthorized")
                    logger.debug(f"Response: {response_text}")
                    raise Exception(
                        "GitHub authentication failed. To fix this:\\n"
                        "1. Create a GitHub Personal Access Token at https://github.com/settings/tokens\\n"
                        "2. Set it as environment variable: set GITHUB_TOKEN=your_token_here\\n"
                        "3. Restart IntelCLaw"
                    )
                elif response.status == 403:
                    logger.warning("GitHub Models API returned 403 - forbidden")
                    logger.debug(f"Response: {response_text}")
                    raise Exception(
                        "GitHub Models API access denied. To fix this:\\n"
                        "1. Visit https://github.com/marketplace/models and accept the terms\\n"
                        "2. Ensure your GitHub account has access to GitHub Models\\n"
                        "3. Create a new Personal Access Token if needed"
                    )
                elif response.status == 404:
                    logger.warning(f"Model not found: {self._github_model_id}")
                    # Try falling back to gpt-4o-mini
                    if self._github_model_id != "gpt-4o-mini":
                        logger.info("Falling back to gpt-4o-mini")
                        self._github_model_id = "gpt-4o-mini"
                        payload["model"] = "gpt-4o-mini"
                        # Retry with fallback model
                        async with session.post(
                            GITHUB_MODELS_API_URL,
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=120)
                        ) as retry_response:
                            if retry_response.status == 200:
                                retry_data = await retry_response.json()
                                return retry_data["choices"][0]["message"]["content"]
                    raise Exception(f"Model '{self._github_model_id}' not found on GitHub Models API")
                elif response.status == 429:
                    raise Exception("GitHub Models API rate limit exceeded - try again later")
                else:
                    logger.error(f"GitHub Models API error: {response.status} - {response_text}")
                    raise Exception(f"GitHub Models API error {response.status}: {response_text}")
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream response from Copilot."""
        # For now, just yield the full response
        response = await self.ainvoke(prompt, **kwargs)
        # Response is now an AIMessage
        yield response.content if hasattr(response, 'content') else str(response)
    
    def bind_tools(self, tools: List[Any]) -> "CopilotLLM":
        """Bind tools for function calling - returns self for LangGraph compatibility."""
        # Store tools for use in API calls
        self._bound_tools = tools
        # Convert LangChain tools to OpenAI function format
        self._tools_schema = self._convert_tools_to_schema(tools)
        # Return self for chaining
        return self
    
    def _convert_tools_to_schema(self, tools: List[Any]) -> List[Dict]:
        """Convert LangChain tools to OpenAI function calling schema."""
        functions = []
        for tool in tools:
            try:
                # Get tool schema
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    # Pydantic schema
                    schema = tool.args_schema.schema()
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])
                else:
                    properties = {}
                    required = []
                
                func = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or f"Tool: {tool.name}",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                }
                functions.append(func)
            except Exception as e:
                logger.warning(f"Could not convert tool {tool.name} to schema: {e}")
        return functions


class CopilotResponse:
    """
    Response from Copilot LLM - LangChain compatible.
    
    This class mimics LangChain's AIMessage so it can be used
    seamlessly in LangChain chains and agents.
    """
    
    def __init__(self, content: str):
        self.content = content
        # LangChain compatibility attributes
        self.type = "ai"
        self.response_metadata = {}
        self.additional_kwargs = {}
        self.id = None
        self.name = None
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"CopilotResponse(content='{self.content[:50]}...')" if len(self.content) > 50 else f"CopilotResponse(content='{self.content}')"
    
    # LangChain message protocol methods
    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {"type": "ai", "content": self.content}
    
    @classmethod
    def from_json(cls, data: dict) -> "CopilotResponse":
        """Create from JSON dict."""
        return cls(content=data.get("content", ""))
    
    # Make it behave like LangChain AIMessage
    @property
    def text(self) -> str:
        """Alias for content (backwards compatibility)."""
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
        copilot = CopilotLLM(model=self.config.get("model", "gpt-4o-mini"))
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
