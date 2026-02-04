"""
GitHub Copilot LLM Provider - Use Copilot as the LLM backend.

This module enables IntelCLaw to use GitHub Copilot's language models
directly through VS Code's Copilot extension API.

Key insight from OpenClaw: GitHub OAuth tokens must be EXCHANGED for
Copilot API tokens at https://api.github.com/copilot_internal/v2/token
before they can be used with the Copilot chat API.
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

from loguru import logger


# GitHub OAuth App Client ID for Copilot (same as VS Code uses)
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"

# Copilot API Token Exchange URL (the key discovery from OpenClaw!)
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

# Default Copilot API Base URL
DEFAULT_COPILOT_API_BASE_URL = "https://api.individual.githubcopilot.com"


def derive_copilot_base_url_from_token(token: str) -> Optional[str]:
    """
    Extract the API base URL from a Copilot token.
    
    The Copilot token contains a proxy-ep field that determines
    the correct API endpoint to use.
    
    Example: token;proxy-ep=proxy.individual.githubcopilot.com; 
             -> https://api.individual.githubcopilot.com
    """
    trimmed = token.strip()
    if not trimmed:
        return None
    
    # The token is semicolon-delimited with key=value pairs
    match = re.search(r'(?:^|;)\s*proxy-ep=([^;\s]+)', trimmed, re.IGNORECASE)
    if not match:
        return None
    
    proxy_ep = match.group(1).strip()
    if not proxy_ep:
        return None
    
    # Convert proxy.* to api.* (as OpenClaw does)
    host = re.sub(r'^https?://', '', proxy_ep)
    host = re.sub(r'^proxy\.', 'api.', host, flags=re.IGNORECASE)
    
    return f"https://{host}" if host else None


class GitHubAuth:
    """
    GitHub OAuth Device Flow Authentication.
    
    This allows users to authenticate with GitHub to use Copilot
    without needing to manually copy tokens.
    
    Token flow (based on OpenClaw):
    1. GitHub Device Flow -> GitHub OAuth token (read:user scope)
    2. Exchange at /copilot_internal/v2/token -> Copilot API token
    """
    
    TOKEN_FILE = Path.home() / ".intelclaw" / "github_token.json"
    COPILOT_TOKEN_FILE = Path.home() / ".intelclaw" / "copilot_token.json"
    
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
        print("🔐 GitHub Copilot Authentication Required")
        print("=" * 60)
        
        try:
            # Step 1: Request device code
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://github.com/login/device/code",
                    headers={"Accept": "application/json"},
                    data={
                        "client_id": GITHUB_CLIENT_ID,
                        "scope": "read:user copilot"
                    }
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get device code: {await response.text()}")
                        return None
                    
                    data = await response.json()
            
            device_code = data["device_code"]
            user_code = data["user_code"]
            verification_uri = data["verification_uri"]
            expires_in = data.get("expires_in", 900)
            interval = data.get("interval", 5)
            
            # Step 2: Show instructions to user
            print(f"\n📋 Your authentication code: {user_code}")
            print(f"\n🌐 Opening: {verification_uri}")
            print(f"\n1. Enter the code above on the GitHub page")
            print(f"2. Authorize IntelCLaw to use GitHub Copilot")
            print(f"3. Wait for confirmation here...\n")
            
            # Open browser
            webbrowser.open(verification_uri)
            
            # Step 3: Poll for token
            print("⏳ Waiting for authorization", end="", flush=True)
            
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                while time.time() - start_time < expires_in:
                    await asyncio.sleep(interval)
                    print(".", end="", flush=True)
                    
                    async with session.post(
                        "https://github.com/login/oauth/access_token",
                        headers={"Accept": "application/json"},
                        data={
                            "client_id": GITHUB_CLIENT_ID,
                            "device_code": device_code,
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
                        }
                    ) as response:
                        token_data = await response.json()
                        
                        if "access_token" in token_data:
                            access_token = token_data["access_token"]
                            cls._save_token(access_token)
                            print("\n\n✅ Authentication successful!")
                            print("=" * 60 + "\n")
                            return access_token
                        
                        error = token_data.get("error")
                        if error == "authorization_pending":
                            continue
                        elif error == "slow_down":
                            interval += 5
                        elif error == "expired_token":
                            print("\n\n❌ Authorization expired. Please try again.")
                            return None
                        elif error == "access_denied":
                            print("\n\n❌ Authorization denied by user.")
                            return None
                        else:
                            logger.debug(f"Token poll response: {token_data}")
            
            print("\n\n❌ Authorization timed out. Please try again.")
            return None
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            print(f"\n\n❌ Authentication failed: {e}")
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
        if cls.COPILOT_TOKEN_FILE.exists():
            cls.COPILOT_TOKEN_FILE.unlink()
        print("✅ GitHub and Copilot tokens cleared.")
    
    @classmethod
    async def exchange_for_copilot_token(cls, github_token: str) -> Optional[Dict[str, Any]]:
        """
        Exchange a GitHub OAuth token for a Copilot API token.
        
        This is the key step that makes Copilot API work!
        Based on OpenClaw's implementation.
        
        Args:
            github_token: GitHub OAuth access token
            
        Returns:
            Dict with 'token', 'expires_at', 'base_url' if successful
        """
        import aiohttp
        
        # Check for cached Copilot token first
        cached = cls._load_copilot_token()
        if cached:
            # Check if token is still valid (with 5 min buffer)
            expires_at = cached.get("expires_at", 0)
            if expires_at - time.time() > 300:  # 5 minutes buffer
                logger.debug("Using cached Copilot API token")
                return cached
        
        logger.info("Exchanging GitHub token for Copilot API token...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    COPILOT_TOKEN_URL,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {github_token}",
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Copilot token exchange failed: {response.status} - {error_text}")
                        return None
                    
                    data = await response.json()
                    
                    # Parse response
                    token = data.get("token")
                    expires_at = data.get("expires_at")
                    
                    if not token:
                        logger.error("Copilot token response missing 'token' field")
                        return None
                    
                    # Convert expires_at to timestamp (GitHub returns Unix seconds)
                    if isinstance(expires_at, (int, float)):
                        # Handle both seconds and milliseconds
                        if expires_at > 10_000_000_000:
                            expires_at_ts = expires_at / 1000
                        else:
                            expires_at_ts = expires_at
                    else:
                        # Default to 1 hour from now
                        expires_at_ts = time.time() + 3600
                    
                    # Derive the API base URL from the token
                    base_url = derive_copilot_base_url_from_token(token) or DEFAULT_COPILOT_API_BASE_URL
                    
                    result = {
                        "token": token,
                        "expires_at": expires_at_ts,
                        "base_url": base_url,
                        "updated_at": time.time()
                    }
                    
                    # Cache the token
                    cls._save_copilot_token(result)
                    
                    logger.info(f"Copilot token obtained, expires at {time.ctime(expires_at_ts)}")
                    logger.info(f"API base URL: {base_url}")
                    
                    return result
                    
        except Exception as e:
            logger.error(f"Copilot token exchange error: {e}")
            return None
    
    @classmethod
    def _save_copilot_token(cls, token_data: Dict[str, Any]) -> None:
        """Save Copilot token to file."""
        cls.COPILOT_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        cls.COPILOT_TOKEN_FILE.write_text(json.dumps(token_data), encoding="utf-8")
        try:
            os.chmod(cls.COPILOT_TOKEN_FILE, 0o600)
        except:
            pass
    
    @classmethod
    def _load_copilot_token(cls) -> Optional[Dict[str, Any]]:
        """Load Copilot token from file."""
        if cls.COPILOT_TOKEN_FILE.exists():
            try:
                return json.loads(cls.COPILOT_TOKEN_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug(f"Could not load cached Copilot token: {e}")
        return None


class CopilotLLM:
    """
    LLM provider that uses GitHub Copilot.
    
    This allows IntelCLaw to leverage your existing GitHub Copilot
    subscription without needing separate API keys.
    
    Uses the two-step token flow:
    1. GitHub OAuth token (from device flow)
    2. Copilot API token (exchanged at /copilot_internal/v2/token)
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize Copilot LLM.
        
        Args:
            model: Model to use (gpt-4o, gpt-4o-mini, claude-3.5-sonnet, o1-preview)
        """
        self.model = model
        self._initialized = False
        self._github_token: Optional[str] = None
        self._copilot_token: Optional[str] = None
        self._copilot_base_url: str = DEFAULT_COPILOT_API_BASE_URL
        self._token_expires_at: float = 0
        self._session_id: Optional[str] = None
    
    async def initialize(self) -> bool:
        """Initialize connection to Copilot."""
        # Step 1: Get GitHub OAuth token
        github_token = await self._get_github_token()
        if not github_token:
            logger.warning("Could not get GitHub OAuth token")
            return False
        
        self._github_token = github_token
        
        # Step 2: Exchange for Copilot API token
        copilot_data = await GitHubAuth.exchange_for_copilot_token(github_token)
        if not copilot_data:
            logger.warning("Could not exchange GitHub token for Copilot API token")
            logger.warning("Make sure you have an active GitHub Copilot subscription")
            return False
        
        self._copilot_token = copilot_data["token"]
        self._copilot_base_url = copilot_data.get("base_url", DEFAULT_COPILOT_API_BASE_URL)
        self._token_expires_at = copilot_data.get("expires_at", 0)
        
        self._initialized = True
        logger.info(f"Copilot LLM initialized with model: {self.model}")
        logger.info(f"API endpoint: {self._copilot_base_url}")
        return True
    
    async def _ensure_valid_token(self) -> bool:
        """Ensure the Copilot API token is still valid, refresh if needed."""
        if not self._copilot_token:
            return False
        
        # Check if token is expiring soon (5 min buffer)
        if time.time() > self._token_expires_at - 300:
            logger.info("Copilot token expiring, refreshing...")
            if self._github_token:
                copilot_data = await GitHubAuth.exchange_for_copilot_token(self._github_token)
                if copilot_data:
                    self._copilot_token = copilot_data["token"]
                    self._copilot_base_url = copilot_data.get("base_url", DEFAULT_COPILOT_API_BASE_URL)
                    self._token_expires_at = copilot_data.get("expires_at", 0)
                    return True
            return False
        
        return True
    
    async def _get_github_token(self) -> Optional[str]:
        """Get GitHub OAuth token from various sources."""
        # 1. Check environment variable first (COPILOT_GITHUB_TOKEN preferred, like OpenClaw)
        token = (
            os.environ.get("COPILOT_GITHUB_TOKEN") or 
            os.environ.get("GH_TOKEN") or 
            os.environ.get("GITHUB_TOKEN")
        )
        if token:
            logger.debug("Using token from environment variable")
            return token
        
        # 2. Check saved IntelCLaw token
        saved_token = GitHubAuth._load_saved_token()
        if saved_token:
            logger.debug("Using saved IntelCLaw token")
            return saved_token
        
        # 3. Try to read from VS Code Copilot extension storage
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
        
        # 4. Try GitHub CLI token
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
        
        # 5. No token found - trigger interactive auth
        logger.info("No GitHub token found, starting interactive authentication...")
        return await GitHubAuth.authenticate()
    
    async def ainvoke(self, input_data, **kwargs) -> "CopilotResponse":
        """
        Invoke the Copilot LLM.
        
        Args:
            input_data: Can be a string prompt or list of LangChain messages
            **kwargs: Additional parameters
            
        Returns:
            Response object with content
        """
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
        
        try:
            response = await self._call_copilot_api(messages)
            return CopilotResponse(content=response)
        except Exception as e:
            logger.error(f"Copilot API error: {e}")
            # Fallback to local processing or error message
            return CopilotResponse(content=f"Error calling Copilot: {e}")
    
    async def _call_copilot_api(self, messages: List[Dict[str, str]]) -> str:
        """Call the GitHub Copilot API using the exchanged Copilot token."""
        import aiohttp
        
        # Ensure token is valid
        if not await self._ensure_valid_token():
            raise Exception("Copilot token expired and could not be refreshed")
        
        # Use the dynamically determined base URL
        api_url = f"{self._copilot_base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self._copilot_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Editor-Version": "vscode/1.96.0",
            "Editor-Plugin-Version": "copilot-chat/0.26.0",
            "User-Agent": "GitHubCopilotChat/0.26.0",
            "X-Request-Id": str(time.time_ns()),
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
            "stream": False,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status == 401:
                    # Token might be expired, try to refresh
                    logger.warning("Copilot returned 401, token may be invalid")
                    raise Exception("Copilot authentication failed - token may have expired")
                elif response.status == 403:
                    raise Exception("Copilot access denied - ensure you have an active Copilot subscription")
                else:
                    error_text = await response.text()
                    logger.error(f"Copilot API error: {response.status} - {error_text}")
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
