"""
Auth Manager - Handles authentication profiles and model providers.

Inspired by OpenClaw's auth-profiles system:
- Stores OAuth tokens with expiry/refresh
- Supports multiple auth profiles per provider
- Device-flow OAuth for GitHub Copilot
"""

import asyncio
import json
import os
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import aiohttp
from loguru import logger


# GitHub OAuth Client ID (same as VS Code Copilot)
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"

# Default paths
STATE_DIR = Path.home() / ".intelclaw"
AUTH_PROFILES_FILE = STATE_DIR / "auth-profiles.json"
CREDENTIALS_DIR = STATE_DIR / "credentials"


@dataclass
class AuthProfile:
    """Represents an authentication profile for a provider."""
    provider: str
    profile_id: str
    mode: str  # "oauth", "api_key", "token"
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None
    api_key: Optional[str] = None
    email: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None
    
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def is_expiring_soon(self, buffer_seconds: int = 300) -> bool:
        """Check if the token is expiring soon."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at - buffer_seconds


class AuthManager:
    """
    Manages authentication profiles for model providers.
    
    Supported providers:
    - github-copilot: GitHub Copilot via device OAuth flow
    - openai: OpenAI API key
    - anthropic: Anthropic API key
    - github-models: GitHub Models API (free)
    """
    
    PROVIDERS = {
        "github-copilot": {
            "name": "GitHub Copilot",
            "auth_modes": ["oauth"],
            "models": ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o1-preview"],
            "requires_subscription": True,
        },
        "github-models": {
            "name": "GitHub Models (Free)",
            "auth_modes": ["oauth", "token"],
            "models": [
                # OpenAI GPT Models
                "gpt-4.1", "gpt-4o", "gpt-5-mini", "gpt-5", "gpt-5.1", "gpt-5.2",
                "gpt-5-codex", "gpt-5.1-codex", "gpt-5.1-codex-max", "gpt-5.1-codex-mini",
                # Anthropic Claude Models
                "claude-haiku-4.5", "claude-opus-4.5", "claude-sonnet-4", "claude-sonnet-4.5",
                # Google Gemini Models  
                "gemini-2.5-pro", "gemini-3-flash", "gemini-3-pro",
                # xAI Grok Models
                "grok-code-fast-1",
                # Raptor Models
                "raptor-mini",
            ],
            "requires_subscription": False,
        },
        "openai": {
            "name": "OpenAI",
            "auth_modes": ["api_key"],
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini", "o1-preview"],
            "requires_subscription": False,
        },
        "anthropic": {
            "name": "Anthropic",
            "auth_modes": ["api_key"],
            "models": ["claude-3-opus", "claude-3.5-sonnet", "claude-3-haiku"],
            "requires_subscription": False,
        },
    }
    
    def __init__(self, state_dir: Optional[Path] = None):
        """Initialize the auth manager."""
        self.state_dir = state_dir or STATE_DIR
        self.auth_profiles_file = self.state_dir / "auth-profiles.json"
        self._profiles: Dict[str, AuthProfile] = {}
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load auth profiles from file."""
        if self.auth_profiles_file.exists():
            try:
                data = json.loads(self.auth_profiles_file.read_text(encoding="utf-8"))
                for profile_id, profile_data in data.get("profiles", {}).items():
                    self._profiles[profile_id] = AuthProfile(**profile_data)
                logger.debug(f"Loaded {len(self._profiles)} auth profiles")
            except Exception as e:
                logger.warning(f"Could not load auth profiles: {e}")
    
    def _save_profiles(self) -> None:
        """Save auth profiles to file."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "profiles": {pid: asdict(p) for pid, p in self._profiles.items()},
            "updated_at": time.time(),
        }
        self.auth_profiles_file.write_text(
            json.dumps(data, indent=2), 
            encoding="utf-8"
        )
        # Set restrictive permissions
        try:
            os.chmod(self.auth_profiles_file, 0o600)
        except:
            pass
    
    def get_profile(self, profile_id: str) -> Optional[AuthProfile]:
        """Get an auth profile by ID."""
        return self._profiles.get(profile_id)
    
    def get_profiles_for_provider(self, provider: str) -> List[AuthProfile]:
        """Get all profiles for a provider."""
        return [p for p in self._profiles.values() if p.provider == provider]
    
    def get_default_profile(self, provider: str) -> Optional[AuthProfile]:
        """Get the default profile for a provider."""
        profiles = self.get_profiles_for_provider(provider)
        if profiles:
            # Return the first non-expired profile, or the first one
            for p in profiles:
                if not p.is_expired():
                    return p
            return profiles[0]
        return None
    
    def save_profile(self, profile: AuthProfile) -> None:
        """Save an auth profile."""
        profile.updated_at = time.time()
        if profile.created_at is None:
            profile.created_at = time.time()
        self._profiles[profile.profile_id] = profile
        self._save_profiles()
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete an auth profile."""
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            self._save_profiles()
            return True
        return False
    
    async def login_github_copilot(self, profile_id: Optional[str] = None) -> Optional[AuthProfile]:
        """
        Login to GitHub Copilot using device OAuth flow.
        
        This runs the GitHub device flow, then exchanges the GitHub token
        for a Copilot API token.
        """
        profile_id = profile_id or "github-copilot:default"
        
        print("\n🦞 GitHub Copilot Authentication")
        print("=" * 40)
        
        # Step 1: Start device flow
        github_token = await self._github_device_flow()
        if not github_token:
            print("❌ GitHub authentication failed")
            return None
        
        # Step 2: Verify Copilot access and get token
        copilot_data = await self._exchange_for_copilot_token(github_token)
        if not copilot_data:
            print("\n❌ Could not get Copilot API access.")
            print("   Make sure you have an active GitHub Copilot subscription.")
            # Fall back to GitHub Models API
            print("\n💡 Tip: You can use GitHub Models API (free) instead:")
            print("   intelclaw models auth login --provider github-models")
            return None
        
        # Create and save profile
        profile = AuthProfile(
            provider="github-copilot",
            profile_id=profile_id,
            mode="oauth",
            access_token=copilot_data["token"],
            expires_at=copilot_data.get("expires_at"),
            extra={
                "github_token": github_token,
                "copilot_base_url": copilot_data.get("base_url"),
            }
        )
        self.save_profile(profile)
        
        print("\n✅ GitHub Copilot authentication successful!")
        print(f"   Profile: {profile_id}")
        if profile.expires_at:
            expires = datetime.fromtimestamp(profile.expires_at)
            print(f"   Token expires: {expires.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return profile
    
    async def login_github_models(self, profile_id: Optional[str] = None) -> Optional[AuthProfile]:
        """
        Login to GitHub Models API using device OAuth flow.
        
        This is the FREE alternative to Copilot - works with any GitHub account.
        """
        profile_id = profile_id or "github-models:default"
        
        print("\n🦞 GitHub Models API Authentication")
        print("=" * 40)
        print("This is FREE and works with any GitHub account!")
        print()
        
        # Get GitHub token via device flow
        github_token = await self._github_device_flow()
        if not github_token:
            print("❌ GitHub authentication failed")
            return None
        
        # Create and save profile (no Copilot token exchange needed)
        profile = AuthProfile(
            provider="github-models",
            profile_id=profile_id,
            mode="oauth",
            access_token=github_token,
            # GitHub tokens don't expire unless revoked
            expires_at=None,
        )
        self.save_profile(profile)
        
        print("\n✅ GitHub Models API authentication successful!")
        print(f"   Profile: {profile_id}")
        print("   Available models: gpt-4o, claude-3.5-sonnet, llama-3.3-70b, deepseek-r1")
        
        return profile
    
    async def _github_device_flow(self) -> Optional[str]:
        """Run the GitHub device OAuth flow."""
        async with aiohttp.ClientSession() as session:
            # Request device code
            async with session.post(
                "https://github.com/login/device/code",
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "scope": "read:user"
                },
                headers={"Accept": "application/json"}
            ) as response:
                if response.status != 200:
                    logger.error(f"Device code request failed: {response.status}")
                    return None
                
                data = await response.json()
            
            device_code = data["device_code"]
            user_code = data["user_code"]
            verification_uri = data["verification_uri"]
            expires_in = data.get("expires_in", 900)
            interval = data.get("interval", 5)
            
            # Show instructions to user
            print(f"\n📋 Please visit: {verification_uri}")
            print(f"   Enter code: {user_code}")
            print()
            
            # Try to open browser
            try:
                webbrowser.open(verification_uri)
                print("   (Browser opened automatically)")
            except:
                pass
            
            print("\n⏳ Waiting for authorization...")
            
            # Poll for token
            start_time = time.time()
            while time.time() - start_time < expires_in:
                await asyncio.sleep(interval)
                
                async with session.post(
                    "https://github.com/login/oauth/access_token",
                    data={
                        "client_id": GITHUB_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
                    },
                    headers={"Accept": "application/json"}
                ) as response:
                    token_data = await response.json()
                    
                    if "access_token" in token_data:
                        return token_data["access_token"]
                    
                    error = token_data.get("error")
                    if error == "authorization_pending":
                        continue
                    elif error == "slow_down":
                        interval += 5
                    elif error == "expired_token":
                        print("\n❌ Authorization expired. Please try again.")
                        return None
                    elif error == "access_denied":
                        print("\n❌ Authorization denied by user.")
                        return None
            
            print("\n❌ Authorization timed out.")
            return None
    
    async def _exchange_for_copilot_token(self, github_token: str) -> Optional[Dict[str, Any]]:
        """Exchange GitHub token for Copilot API token."""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.96.0",
                "Editor-Plugin-Version": "copilot-chat/0.26.0",
                "User-Agent": "GitHubCopilotChat/0.26.0",
            }
            
            async with session.get(
                "https://api.github.com/copilot_internal/v2/token",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "token": data.get("token"),
                        "expires_at": data.get("expires_at"),
                        "base_url": data.get("endpoints", {}).get("api"),
                    }
                elif response.status == 401:
                    logger.warning("GitHub token not authorized for Copilot")
                elif response.status == 403:
                    logger.warning("Copilot access forbidden - check subscription")
                else:
                    text = await response.text()
                    logger.warning(f"Copilot token exchange failed: {response.status} - {text}")
                
                return None
    
    def save_api_key(self, provider: str, api_key: str, profile_id: Optional[str] = None) -> AuthProfile:
        """Save an API key for a provider."""
        profile_id = profile_id or f"{provider}:default"
        profile = AuthProfile(
            provider=provider,
            profile_id=profile_id,
            mode="api_key",
            api_key=api_key,
        )
        self.save_profile(profile)
        return profile
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current auth status for all providers."""
        status = {}
        for provider, info in self.PROVIDERS.items():
            profiles = self.get_profiles_for_provider(provider)
            default = self.get_default_profile(provider)
            
            status[provider] = {
                "name": info["name"],
                "profiles": len(profiles),
                "default": default.profile_id if default else None,
                "authenticated": bool(default and not default.is_expired()),
                "expiring_soon": default.is_expiring_soon() if default else False,
            }
        return status
    
    def print_status(self) -> None:
        """Print the current auth status."""
        status = self.get_status()
        
        print("\n🔐 Authentication Status")
        print("=" * 50)
        
        for provider, info in status.items():
            auth_icon = "✅" if info["authenticated"] else "❌"
            expiry_warning = " ⚠️ (expiring soon)" if info["expiring_soon"] else ""
            
            print(f"\n{auth_icon} {info['name']}")
            if info["authenticated"]:
                print(f"   Profile: {info['default']}{expiry_warning}")
            else:
                print(f"   Not authenticated")
                print(f"   Run: intelclaw models auth login --provider {provider}")
