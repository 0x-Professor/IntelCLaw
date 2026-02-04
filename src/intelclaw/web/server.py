"""
Web Server - FastAPI-based web interface for IntelCLaw.

Provides a modern chat interface accessible via browser with WebSocket support.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

if TYPE_CHECKING:
    from intelclaw.core.app import IntelCLawApp


# Get the static files directory
STATIC_DIR = Path(__file__).parent / "static"


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, [])
    
    def add_to_session(self, session_id: str, message: Dict[str, Any]):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message)


class WebServer:
    """
    FastAPI web server for IntelCLaw.
    
    Features:
    - Real-time chat via WebSocket
    - REST API for agent interaction
    - Beautiful web UI with static files
    - Settings management
    - Multiple session support
    """
    
    def __init__(self, app: Optional["IntelCLawApp"] = None, host: str = "127.0.0.1", port: int = 8765):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self._app = app
        self.host = host
        self.port = port
        
        # Determine default model based on provider
        provider = os.getenv("INTELCLAW_PROVIDER", "github-models")
        default_model = os.getenv("INTELCLAW_DEFAULT_MODEL", "gpt-4o-mini")
        
        if provider == "github-copilot" and default_model == "gpt-4o-mini":
            # Use a better default for Copilot subscribers
            default_model = "gpt-4o"
        
        self.current_model = default_model
        self.current_provider = provider
        
        self.fastapi = FastAPI(
            title="IntelCLaw",
            description="Autonomous AI Agent Web Interface",
            version="0.1.0"
        )
        self.manager = ConnectionManager()
        self._setup_routes()
        self._server = None
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        
        # Mount static files
        if STATIC_DIR.exists():
            self.fastapi.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
            logger.info(f"Mounted static files from: {STATIC_DIR}")
        else:
            logger.warning(f"Static directory not found: {STATIC_DIR}")
        
        @self.fastapi.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Serve the main chat interface."""
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file), media_type="text/html")
            else:
                # Fallback to inline HTML if static files don't exist
                return HTMLResponse(content=self._get_fallback_html())
        
        @self.fastapi.get("/api/status")
        async def status():
            """Get agent status."""
            return {
                "status": "running",
                "agent": self._app.agent.status.value if self._app and self._app.agent else "ready",
                "model": self.current_model,
                "llm_provider": self._get_llm_provider(),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.fastapi.get("/api/models")
        async def models():
            """Get available models dynamically from the provider."""
            import os
            import time as time_module
            from dotenv import load_dotenv
            load_dotenv()
            
            provider = os.getenv("INTELCLAW_PROVIDER", "github-models")
            copilot_github_token = os.getenv("COPILOT_GITHUB_TOKEN")
            github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
            has_copilot = bool(copilot_github_token) or provider == "github-copilot"
            
            model_list = []
            copilot_models_fetched = False
            
            # Try to fetch Copilot models dynamically
            if has_copilot:
                try:
                    from intelclaw.integrations.llm_provider import (
                        fetch_copilot_models,
                        DEFAULT_COPILOT_API_BASE_URL,
                    )
                    
                    copilot_base_url = os.getenv("COPILOT_API_BASE_URL", DEFAULT_COPILOT_API_BASE_URL)
                    
                    # Try to get Copilot API token from multiple sources:
                    # 1. First try auth-profiles.json (most up-to-date after re-auth)
                    # 2. Fall back to copilot_token.json cache
                    copilot_api_token = None
                    auth_profiles_file = Path.home() / ".intelclaw" / "auth-profiles.json"
                    cache_file = Path.home() / ".intelclaw" / "copilot_token.json"
                    
                    # Try auth-profiles.json first (updated on re-authentication)
                    if auth_profiles_file.exists():
                        try:
                            profiles_data = json.loads(auth_profiles_file.read_text(encoding="utf-8"))
                            copilot_profile = profiles_data.get("profiles", {}).get("github-copilot:default")
                            if copilot_profile:
                                expires_at = copilot_profile.get("expires_at", 0)
                                if expires_at and time_module.time() < expires_at - 300:
                                    copilot_api_token = copilot_profile.get("access_token")
                                    extra = copilot_profile.get("extra", {})
                                    if extra.get("copilot_base_url"):
                                        copilot_base_url = extra.get("copilot_base_url")
                                    logger.info(f"Using Copilot token from auth profile (expires: {expires_at})")
                        except Exception as e:
                            logger.debug(f"Failed to read auth profiles: {e}")
                    
                    # Fall back to copilot_token.json cache
                    if not copilot_api_token and cache_file.exists():
                        try:
                            cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
                            expires_at = cache_data.get("expires_at", 0)
                            
                            # Check if token is still valid (with 5 min buffer)
                            if expires_at and time_module.time() < expires_at - 300:
                                copilot_api_token = cache_data.get("token")
                                if cache_data.get("base_url"):
                                    copilot_base_url = cache_data.get("base_url")
                                logger.debug(f"Using cached Copilot API token (expires: {expires_at})")
                        except Exception as e:
                            logger.debug(f"Failed to read Copilot token cache: {e}")
                    
                    if copilot_api_token:
                        copilot_models = await fetch_copilot_models(copilot_api_token, copilot_base_url)
                        if copilot_models:
                            model_list.extend(copilot_models)
                            copilot_models_fetched = True
                            logger.info(f"Fetched {len(copilot_models)} Copilot models dynamically")
                    else:
                        logger.debug("No valid cached Copilot API token found")
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch Copilot models: {e}")
            
            # Always add static Copilot models if has_copilot but dynamic fetch failed
            if has_copilot and not copilot_models_fetched:
                logger.debug("Using static Copilot model list")
                model_list.extend([
                    {"id": "gpt-4o", "name": "GPT-4o", "provider": "github-copilot", "category": "OpenAI (Copilot)"},
                    {"id": "gpt-4.1", "name": "GPT-4.1", "provider": "github-copilot", "category": "OpenAI (Copilot)"},
                    {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini", "provider": "github-copilot", "category": "OpenAI (Copilot)"},
                    {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano", "provider": "github-copilot", "category": "OpenAI (Copilot)"},
                    {"id": "o1", "name": "o1 (Reasoning)", "provider": "github-copilot", "category": "OpenAI (Copilot)"},
                    {"id": "o1-mini", "name": "o1 Mini", "provider": "github-copilot", "category": "OpenAI (Copilot)"},
                    {"id": "o3-mini", "name": "o3 Mini", "provider": "github-copilot", "category": "OpenAI (Copilot)"},
                    {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "provider": "github-copilot", "category": "Anthropic (Copilot)"},
                    {"id": "claude-3.7-sonnet", "name": "Claude 3.7 Sonnet", "provider": "github-copilot", "category": "Anthropic (Copilot)"},
                    {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "provider": "github-copilot", "category": "Anthropic (Copilot)"},
                    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "github-copilot", "category": "Google (Copilot)"},
                    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "github-copilot", "category": "Google (Copilot)"},
                ])
            
            # Add GitHub Models API (FREE tier) as fallback options
            if github_token or not model_list:
                model_list.extend([
                    {"id": "gpt-4o-mini", "name": "GPT-4o mini (Free)", "provider": "github-models", "category": "OpenAI (Free)"},
                    {"id": "gpt-4o", "name": "GPT-4o (Free)", "provider": "github-models", "category": "OpenAI (Free)"},
                    {"id": "llama-3.3-70b", "name": "Llama 3.3 70B", "provider": "github-models", "category": "Meta Llama (Free)"},
                    {"id": "llama-3.1-405b", "name": "Llama 3.1 405B", "provider": "github-models", "category": "Meta Llama (Free)"},
                    {"id": "mistral-large", "name": "Mistral Large", "provider": "github-models", "category": "Mistral (Free)"},
                    {"id": "deepseek-r1", "name": "DeepSeek R1", "provider": "github-models", "category": "DeepSeek (Free)"},
                    {"id": "deepseek-v3", "name": "DeepSeek V3", "provider": "github-models", "category": "DeepSeek (Free)"},
                ])
            
            return {
                "models": model_list,
                "current": self.current_model,
                "provider": provider,
                "has_copilot": has_copilot,
                "dynamic": copilot_models_fetched
            }
        
        @self.fastapi.post("/api/chat")
        async def chat(request: Request):
            """REST endpoint for chat (non-WebSocket)."""
            data = await request.json()
            message = data.get("message", "")
            
            if not message:
                return JSONResponse({"error": "No message provided"}, status_code=400)
            
            response = await self._process_message(message)
            return {"response": response, "model": self.current_model}
        
        @self.fastapi.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time chat."""
            await self.manager.connect(websocket)
            current_session_id = None
            
            # Send initial state
            await self.manager.send_message({
                "type": "state",
                "model": self.current_model,
                "tasks": [],
                "timestamp": datetime.now().isoformat()
            }, websocket)
            
            try:
                while True:
                    data = await websocket.receive_json()
                    msg_type = data.get("type", "chat")
                    
                    if msg_type == "chat":
                        await self._handle_chat(data, websocket, current_session_id)
                    
                    elif msg_type == "set_model":
                        new_model = data.get("model", self.current_model)
                        new_provider = data.get("provider", None)
                        self.current_model = new_model
                        
                        # Update the LLM model if available
                        model_updated = False
                        
                        # Try updating via agent._llm
                        if self._app and self._app.agent and hasattr(self._app.agent, '_llm'):
                            if hasattr(self._app.agent._llm, 'set_model'):
                                self._app.agent._llm.set_model(new_model)
                                model_updated = True
                                logger.info(f"Model changed to: {new_model} (via agent._llm)")
                        
                        # Try updating via agent._llm_provider
                        if self._app and self._app.agent and hasattr(self._app.agent, '_llm_provider'):
                            llm_provider = self._app.agent._llm_provider
                            if hasattr(llm_provider, 'set_model'):
                                llm_provider.set_model(new_model)
                                model_updated = True
                                logger.info(f"Model changed to: {new_model} (via _llm_provider)")
                            
                            # Update provider if specified
                            if new_provider and hasattr(llm_provider, 'active_provider'):
                                llm_provider.active_provider = new_provider
                                self.current_provider = new_provider
                                logger.info(f"Provider changed to: {new_provider}")
                        
                        if not model_updated:
                            logger.warning(f"Could not update model to {new_model} - no LLM provider found")
                        
                        await self.manager.send_message({
                            "type": "state",
                            "model": self.current_model,
                            "provider": getattr(self, 'current_provider', 'github-copilot'),
                            "model_updated": model_updated
                        }, websocket)
                    
                    elif msg_type == "new_session":
                        current_session_id = data.get("session_id")
                        logger.info(f"New session: {current_session_id}")
                    
                    elif msg_type == "get_state":
                        await self.manager.send_message({
                            "type": "state",
                            "model": self.current_model,
                            "tasks": [],
                            "timestamp": datetime.now().isoformat()
                        }, websocket)
                        
            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                import traceback
                traceback.print_exc()
                self.manager.disconnect(websocket)
    
    async def _handle_chat(self, data: dict, websocket: WebSocket, session_id: Optional[str]):
        """Handle chat message."""
        message = data.get("message", "")
        model = data.get("model", self.current_model)
        settings = data.get("settings", {})
        stream = settings.get("stream", True)
        
        if not message:
            return
        
        # Update current model if changed
        if model != self.current_model:
            self.current_model = model
            # Update the LLM model
            if self._app and self._app.agent and hasattr(self._app.agent, '_llm'):
                if hasattr(self._app.agent._llm, 'set_model'):
                    self._app.agent._llm.set_model(model)
                    logger.info(f"Model changed to: {model}")
        
        # Store user message
        if session_id:
            self.manager.add_to_session(session_id, {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
        
        # Process message through agent
        try:
            if stream and hasattr(self, '_process_message_stream'):
                # Streaming response
                async for chunk in self._process_message_stream(message, settings):
                    await self.manager.send_message({
                        "type": "chat_stream",
                        "delta": chunk,
                        "model": self.current_model
                    }, websocket)
                
                await self.manager.send_message({
                    "type": "chat_complete",
                    "model": self.current_model,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
            else:
                # Non-streaming response
                response = await self._process_message(message, settings)
                
                # Store assistant message
                if session_id:
                    self.manager.add_to_session(session_id, {
                        "role": "assistant",
                        "content": response,
                        "model": self.current_model,
                        "timestamp": datetime.now().isoformat()
                    })
                
                await self.manager.send_message({
                    "type": "chat_response",
                    "content": response,
                    "model": self.current_model,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            import traceback
            traceback.print_exc()
            await self.manager.send_message({
                "type": "error",
                "message": str(e)
            }, websocket)
    
    async def _process_message(self, message: str, settings: Optional[dict] = None) -> str:
        """Process a user message through the agent."""
        if not self._app or not self._app.agent:
            # Demo mode - return helpful response
            return self._get_demo_response(message)
        
        try:
            from intelclaw.agent.base import AgentContext
            
            context = AgentContext(
                user_message=message,
                screen_context=None,
                active_window=None,
                user_preferences=settings or {}
            )
            
            response = await self._app.agent.process(context)
            return response.answer if response else "I couldn't process that request."
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            import traceback
            traceback.print_exc()
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _get_demo_response(self, message: str) -> str:
        """Return a demo response when agent is not available."""
        message_lower = message.lower()
        
        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! I'm IntelCLaw, your autonomous AI assistant. How can I help you today?"
        elif "help" in message_lower:
            return """I can help you with:
            
- **File Operations**: Create, read, edit, and organize files
- **Code Tasks**: Write, review, and debug code
- **Shell Commands**: Execute system commands safely
- **Research**: Search the web and analyze information
- **Automation**: Create scripts and automate repetitive tasks

Just tell me what you need!"""
        elif "python" in message_lower or "code" in message_lower:
            return """I can help you with Python! Here's what I can do:

- Write new Python scripts
- Debug existing code
- Explain code concepts
- Create project structures
- Install packages

What would you like me to help you with?"""
        elif "how" in message_lower and "work" in message_lower:
            return """**How IntelCLaw Works:**

1. **REACT Pattern**: I use a Reasoning + Acting loop to solve tasks
2. **Tool Execution**: I can run shell commands, edit files, and browse the web
3. **Context Awareness**: I understand your workspace and system
4. **Learning**: I adapt to your preferences over time

I'm powered by advanced language models from GitHub Models API!"""
        else:
            return f"""I received your message: "{message}"

I'm currently running in demo mode. To enable full functionality:

1. Run `intelclaw onboard` to complete setup
2. Configure your GitHub Models API token
3. Restart the web server

Once configured, I'll be able to:
- Execute code and commands
- Manage files on your system
- Search the web
- And much more!"""
    
    def _get_llm_provider(self) -> str:
        """Get the current LLM provider name."""
        if self._app and self._app.agent and hasattr(self._app.agent, '_llm_provider'):
            return self._app.agent._llm_provider.active_provider
        return self.current_provider
    
    def _get_fallback_html(self) -> str:
        """Return fallback HTML if static files are missing."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IntelCLaw - Setup Required</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0b;
            color: #fafafa;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            text-align: center;
            padding: 40px;
            max-width: 500px;
        }
        .icon {
            font-size: 64px;
            margin-bottom: 24px;
        }
        h1 {
            font-size: 28px;
            margin-bottom: 16px;
        }
        p {
            color: #a1a1aa;
            margin-bottom: 24px;
            line-height: 1.6;
        }
        code {
            background: #27272a;
            padding: 12px 20px;
            border-radius: 8px;
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            color: #6366f1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">🦅</div>
        <h1>IntelCLaw</h1>
        <p>Static files are missing. Please ensure the frontend is properly built.</p>
        <code>uv run python -m intelclaw gateway</code>
    </div>
</body>
</html>'''
    
    async def start(self):
        """Start the web server."""
        config = uvicorn.Config(
            self.fastapi,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        self._server = uvicorn.Server(config)
        
        logger.info(f"🦅 IntelCLaw Web Server starting on http://{self.host}:{self.port}")
        await self._server.serve()
    
    async def stop(self):
        """Stop the web server."""
        if self._server:
            self._server.should_exit = True
    
    def run(self):
        """Run the web server (blocking)."""
        import asyncio
        asyncio.run(self.start())


def run_server(app: Optional["IntelCLawApp"] = None, host: str = "127.0.0.1", port: int = 8765):
    """Convenience function to run the web server."""
    server = WebServer(app=app, host=host, port=port)
    server.run()


if __name__ == "__main__":
    run_server()
