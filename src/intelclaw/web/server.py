"""
Web Server - FastAPI-based web interface for IntelCLaw.

Provides a chat interface accessible via browser.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
import uvicorn

if TYPE_CHECKING:
    from intelclaw.core.app import IntelCLawApp


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
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


class WebServer:
    """
    FastAPI web server for IntelCLaw.
    
    Features:
    - Real-time chat via WebSocket
    - REST API for agent interaction
    - Beautiful web UI
    - Settings management
    """
    
    def __init__(self, app: Optional["IntelCLawApp"] = None, host: str = "127.0.0.1", port: int = 8765):
        self._app = app
        self.host = host
        self.port = port
        self.fastapi = FastAPI(
            title="IntelCLaw",
            description="Autonomous AI Agent Web Interface",
            version="0.1.0"
        )
        self.manager = ConnectionManager()
        self._setup_routes()
        self._server = None
        self._conversation_history: List[Dict[str, Any]] = []
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        
        @self.fastapi.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Serve the main chat interface."""
            return HTMLResponse(content=self._get_chat_html())
        
        @self.fastapi.get("/api/status")
        async def status():
            """Get agent status."""
            return {
                "status": "running",
                "agent": self._app.agent.status.value if self._app and self._app.agent else "unknown",
                "llm_provider": self._app.agent._llm_provider.active_provider if self._app and self._app.agent and hasattr(self._app.agent, '_llm_provider') else "unknown",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.fastapi.get("/api/history")
        async def history():
            """Get conversation history."""
            return {"history": self._conversation_history[-50:]}  # Last 50 messages
        
        @self.fastapi.post("/api/chat")
        async def chat(request: Request):
            """REST endpoint for chat (non-WebSocket)."""
            data = await request.json()
            message = data.get("message", "")
            
            if not message:
                return JSONResponse({"error": "No message provided"}, status_code=400)
            
            response = await self._process_message(message)
            return {"response": response}
        
        @self.fastapi.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time chat."""
            await self.manager.connect(websocket)
            
            # Send welcome message
            await self.manager.send_message({
                "type": "system",
                "content": "Connected to IntelCLaw! How can I help you?",
                "timestamp": datetime.now().isoformat()
            }, websocket)
            
            try:
                while True:
                    data = await websocket.receive_json()
                    message = data.get("message", "")
                    
                    if message:
                        # Store user message
                        user_msg = {
                            "type": "user",
                            "content": message,
                            "timestamp": datetime.now().isoformat()
                        }
                        self._conversation_history.append(user_msg)
                        
                        # Send typing indicator
                        await self.manager.send_message({
                            "type": "typing",
                            "content": "Thinking...",
                            "timestamp": datetime.now().isoformat()
                        }, websocket)
                        
                        # Process message
                        response = await self._process_message(message)
                        
                        # Store and send response
                        agent_msg = {
                            "type": "agent",
                            "content": response,
                            "timestamp": datetime.now().isoformat()
                        }
                        self._conversation_history.append(agent_msg)
                        
                        await self.manager.send_message(agent_msg, websocket)
                        
            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.manager.disconnect(websocket)
    
    async def _process_message(self, message: str) -> str:
        """Process a user message through the agent."""
        if not self._app or not self._app.agent:
            return "Agent not available. Please ensure IntelCLaw is properly initialized."
        
        try:
            from intelclaw.agent.base import AgentContext
            
            context = AgentContext(
                user_message=message,
                screen_context=None,
                active_window=None,
                user_preferences={}
            )
            
            response = await self._app.agent.process(context)
            return response.answer if response else "I couldn't process that request."
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _get_chat_html(self) -> str:
        """Return the OpenClaw-style chat interface HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IntelCLaw - Control UI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0a0b;
            --bg-secondary: #111113;
            --bg-tertiary: #18181b;
            --bg-hover: #1f1f23;
            --accent: #6366f1;
            --accent-soft: rgba(99, 102, 241, 0.1);
            --accent-hover: #4f46e5;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --text-primary: #fafafa;
            --text-secondary: #a1a1aa;
            --text-tertiary: #71717a;
            --border: #27272a;
            --border-light: #3f3f46;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
            --sidebar-width: 280px;
            --header-height: 56px;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }
        
        /* Layout */
        .app-container {
            display: flex;
            height: 100vh;
        }
        
        /* Sidebar */
        .sidebar {
            width: var(--sidebar-width);
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease;
        }
        
        .sidebar-header {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .logo {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            box-shadow: 0 0 20px var(--accent-glow);
        }
        
        .logo-text {
            font-size: 1.125rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text-primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Navigation */
        .sidebar-nav {
            flex: 1;
            padding: 0.75rem;
            overflow-y: auto;
        }
        
        .nav-section {
            margin-bottom: 1.5rem;
        }
        
        .nav-section-title {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-tertiary);
            padding: 0 0.75rem;
            margin-bottom: 0.5rem;
        }
        
        .nav-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.625rem 0.75rem;
            border-radius: 0.5rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
            font-size: 0.875rem;
        }
        
        .nav-item:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }
        
        .nav-item.active {
            background: var(--accent-soft);
            color: var(--accent);
        }
        
        .nav-item svg {
            width: 18px;
            height: 18px;
            opacity: 0.7;
        }
        
        .nav-item.active svg {
            opacity: 1;
        }
        
        .nav-badge {
            margin-left: auto;
            background: var(--accent);
            color: white;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.125rem 0.5rem;
            border-radius: 9999px;
        }
        
        /* Sidebar Footer */
        .sidebar-footer {
            padding: 1rem 1.25rem;
            border-top: 1px solid var(--border);
        }
        
        .status-card {
            background: var(--bg-tertiary);
            border-radius: 0.75rem;
            padding: 0.875rem;
        }
        
        .status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.8rem;
        }
        
        .status-row + .status-row {
            margin-top: 0.5rem;
        }
        
        .status-label {
            color: var(--text-tertiary);
        }
        
        .status-value {
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }
        
        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--success);
        }
        
        .status-dot.warning { background: var(--warning); }
        .status-dot.error { background: var(--error); }
        
        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        
        /* Header */
        .header {
            height: var(--header-height);
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            padding: 0 1.5rem;
            gap: 1rem;
        }
        
        .header-title {
            font-size: 0.9rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .session-selector {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.5rem 0.75rem;
            color: var(--text-primary);
            font-size: 0.8rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .header-actions {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .header-btn {
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.5rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .header-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-color: var(--border-light);
        }
        
        .header-btn svg {
            width: 18px;
            height: 18px;
        }
        
        .model-selector {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.5rem 0.875rem;
            color: var(--text-primary);
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
        }
        
        .model-selector:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        /* Chat Area */
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }
        
        .message {
            display: flex;
            gap: 1rem;
            max-width: 900px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.875rem;
            flex-shrink: 0;
        }
        
        .message.user .message-avatar {
            background: var(--accent);
        }
        
        .message.agent .message-avatar {
            background: linear-gradient(135deg, #8b5cf6, var(--accent));
        }
        
        .message.system .message-avatar {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
        }
        
        .message-body {
            flex: 1;
            min-width: 0;
        }
        
        .message-header {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            margin-bottom: 0.375rem;
        }
        
        .message-sender {
            font-weight: 600;
            font-size: 0.875rem;
        }
        
        .message-time {
            font-size: 0.7rem;
            color: var(--text-tertiary);
        }
        
        .message-content {
            font-size: 0.9rem;
            line-height: 1.7;
            color: var(--text-secondary);
        }
        
        .message-content p { margin-bottom: 0.75rem; }
        .message-content p:last-child { margin-bottom: 0; }
        
        .message-content code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85em;
            background: var(--bg-tertiary);
            padding: 0.125rem 0.375rem;
            border-radius: 0.25rem;
            border: 1px solid var(--border);
        }
        
        .message-content pre {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1rem;
            overflow-x: auto;
            margin: 0.75rem 0;
        }
        
        .message-content pre code {
            background: none;
            border: none;
            padding: 0;
        }
        
        /* Tool Calls */
        .tool-call {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            margin: 0.75rem 0;
            overflow: hidden;
        }
        
        .tool-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.625rem 0.875rem;
            background: var(--bg-hover);
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .tool-icon {
            color: var(--warning);
        }
        
        .tool-name {
            color: var(--accent);
            font-weight: 500;
        }
        
        .tool-status {
            margin-left: auto;
            font-size: 0.7rem;
            padding: 0.125rem 0.5rem;
            border-radius: 9999px;
            background: var(--accent-soft);
            color: var(--accent);
        }
        
        .tool-status.success {
            background: rgba(34, 197, 94, 0.1);
            color: var(--success);
        }
        
        .tool-content {
            padding: 0.875rem;
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-tertiary);
            max-height: 200px;
            overflow-y: auto;
        }
        
        /* Typing Indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border-radius: 1rem;
            width: fit-content;
        }
        
        .typing-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--text-tertiary);
            animation: typing 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 100% { opacity: 0.3; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1); }
        }
        
        /* Input Area */
        .input-area {
            padding: 1rem 1.5rem 1.5rem;
            background: linear-gradient(to top, var(--bg-primary) 50%, transparent);
        }
        
        .input-container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .input-box {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 0.875rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        
        .input-box:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }
        
        .input-main {
            display: flex;
            align-items: flex-end;
            gap: 0.75rem;
        }
        
        #message-input {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 0.9rem;
            font-family: inherit;
            resize: none;
            outline: none;
            min-height: 24px;
            max-height: 200px;
            line-height: 1.5;
        }
        
        #message-input::placeholder {
            color: var(--text-tertiary);
        }
        
        .input-actions {
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }
        
        .input-btn {
            background: transparent;
            border: none;
            padding: 0.5rem;
            color: var(--text-tertiary);
            cursor: pointer;
            border-radius: 0.5rem;
            transition: all 0.15s ease;
        }
        
        .input-btn:hover {
            background: var(--bg-hover);
            color: var(--text-secondary);
        }
        
        .input-btn svg {
            width: 18px;
            height: 18px;
        }
        
        .send-btn {
            background: var(--accent);
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            color: white;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }
        
        .send-btn:hover {
            background: var(--accent-hover);
        }
        
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .send-btn svg {
            width: 16px;
            height: 16px;
        }
        
        .input-hint {
            font-size: 0.7rem;
            color: var(--text-tertiary);
            text-align: center;
        }
        
        .input-hint kbd {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            padding: 0.125rem 0.375rem;
            border-radius: 0.25rem;
            font-family: inherit;
            font-size: 0.65rem;
        }
        
        /* Right Panel */
        .right-panel {
            width: 320px;
            background: var(--bg-secondary);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease;
        }
        
        .right-panel.collapsed {
            transform: translateX(100%);
            position: absolute;
            right: 0;
        }
        
        .panel-header {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .panel-title {
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .panel-close {
            background: transparent;
            border: none;
            color: var(--text-tertiary);
            cursor: pointer;
            padding: 0.25rem;
        }
        
        .panel-tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
        }
        
        .panel-tab {
            flex: 1;
            padding: 0.75rem;
            background: transparent;
            border: none;
            color: var(--text-tertiary);
            font-size: 0.8rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.15s ease;
        }
        
        .panel-tab:hover {
            color: var(--text-secondary);
        }
        
        .panel-tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }
        
        .panel-content {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }
        
        /* Settings Form */
        .form-group {
            margin-bottom: 1.25rem;
        }
        
        .form-label {
            display: block;
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        
        .form-input {
            width: 100%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.625rem 0.875rem;
            color: var(--text-primary);
            font-size: 0.8rem;
            font-family: inherit;
        }
        
        .form-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .form-select {
            width: 100%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.625rem 0.875rem;
            color: var(--text-primary);
            font-size: 0.8rem;
            cursor: pointer;
        }
        
        .toggle-group {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .toggle {
            width: 40px;
            height: 22px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 11px;
            position: relative;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .toggle.active {
            background: var(--accent);
            border-color: var(--accent);
        }
        
        .toggle::after {
            content: '';
            position: absolute;
            width: 16px;
            height: 16px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.2s ease;
        }
        
        .toggle.active::after {
            transform: translateX(18px);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
        
        /* Mobile */
        @media (max-width: 1024px) {
            .sidebar { 
                position: absolute;
                z-index: 100;
                transform: translateX(-100%);
            }
            .sidebar.open { transform: translateX(0); }
            .right-panel { display: none; }
        }
        
        /* Empty State */
        .empty-state {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            text-align: center;
        }
        
        .empty-icon {
            width: 80px;
            height: 80px;
            background: var(--bg-tertiary);
            border-radius: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }
        
        .empty-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .empty-text {
            color: var(--text-tertiary);
            font-size: 0.9rem;
            max-width: 400px;
            line-height: 1.6;
        }
        
        .suggestion-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1.5rem;
            justify-content: center;
        }
        
        .suggestion-chip {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 9999px;
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
        }
        
        .suggestion-chip:hover {
            background: var(--bg-hover);
            border-color: var(--accent);
            color: var(--text-primary);
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">🦞</div>
                <span class="logo-text">IntelCLaw</span>
            </div>
            
            <nav class="sidebar-nav">
                <div class="nav-section">
                    <div class="nav-section-title">Chat</div>
                    <div class="nav-item active" data-view="chat">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
                        Chat
                    </div>
                    <div class="nav-item" data-view="sessions">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        Sessions
                    </div>
                </div>
                
                <div class="nav-section">
                    <div class="nav-section-title">Tools</div>
                    <div class="nav-item" data-view="tools">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>
                        Tools
                        <span class="nav-badge">12</span>
                    </div>
                    <div class="nav-item" data-view="skills">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                        Skills
                    </div>
                </div>
                
                <div class="nav-section">
                    <div class="nav-section-title">System</div>
                    <div class="nav-item" data-view="models">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                        Models
                    </div>
                    <div class="nav-item" data-view="config">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
                        Config
                    </div>
                    <div class="nav-item" data-view="logs">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
                        Logs
                    </div>
                </div>
            </nav>
            
            <div class="sidebar-footer">
                <div class="status-card">
                    <div class="status-row">
                        <span class="status-label">Gateway</span>
                        <span class="status-value">
                            <span class="status-dot" id="gateway-status"></span>
                            <span id="gateway-text">Connected</span>
                        </span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Model</span>
                        <span class="status-value" id="current-model">gpt-4o</span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Auth</span>
                        <span class="status-value">
                            <span class="status-dot" id="auth-status"></span>
                            <span id="auth-text">GitHub</span>
                        </span>
                    </div>
                </div>
            </div>
        </aside>
        
        <!-- Main Content -->
        <main class="main-content">
            <header class="header">
                <button class="header-btn" id="menu-toggle" style="display: none;">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
                </button>
                
                <div class="header-title">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
                    Chat
                </div>
                
                <div class="session-selector">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
                    main
                </div>
                
                <div class="header-actions">
                    <select class="model-selector" id="model-select">
                        <optgroup label="GitHub Models (Free)">
                            <option value="gpt-4o" selected>gpt-4o</option>
                            <option value="gpt-4o-mini">gpt-4o-mini</option>
                            <option value="gpt-5">gpt-5</option>
                            <option value="gpt-5-mini">gpt-5-mini</option>
                            <option value="o3-mini">o3-mini</option>
                            <option value="o4-mini">o4-mini</option>
                            <option value="phi-4">Phi-4</option>
                            <option value="deepseek-r1">DeepSeek-R1</option>
                            <option value="grok-3">Grok-3</option>
                        </optgroup>
                    </select>
                    
                    <button class="header-btn" id="clear-btn" title="Clear chat">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
                    </button>
                    
                    <button class="header-btn" id="settings-toggle" title="Settings">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
                    </button>
                </div>
            </header>
            
            <div class="chat-area">
                <div class="messages" id="messages">
                    <!-- Empty State -->
                    <div class="empty-state" id="empty-state">
                        <div class="empty-icon">🦞</div>
                        <h2 class="empty-title">Welcome to IntelCLaw</h2>
                        <p class="empty-text">
                            Your autonomous AI agent is ready. Start a conversation or try one of these suggestions:
                        </p>
                        <div class="suggestion-chips">
                            <button class="suggestion-chip" data-prompt="What can you do?">What can you do?</button>
                            <button class="suggestion-chip" data-prompt="Help me with coding">Help me code</button>
                            <button class="suggestion-chip" data-prompt="Search the web for latest AI news">Search the web</button>
                            <button class="suggestion-chip" data-prompt="Analyze my screen">Analyze screen</button>
                        </div>
                    </div>
                </div>
                
                <div class="input-area">
                    <div class="input-container">
                        <div class="input-box">
                            <div class="input-main">
                                <textarea id="message-input" placeholder="Send a message..." rows="1"></textarea>
                                <div class="input-actions">
                                    <button class="input-btn" title="Attach file">
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
                                    </button>
                                    <button class="send-btn" id="send-btn">
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                                        Send
                                    </button>
                                </div>
                            </div>
                        </div>
                        <p class="input-hint"><kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line · Type <kbd>/</kbd> for commands</p>
                    </div>
                </div>
            </div>
        </main>
        
        <!-- Right Panel (Settings) -->
        <aside class="right-panel collapsed" id="right-panel">
            <div class="panel-header">
                <span class="panel-title">Settings</span>
                <button class="panel-close" id="panel-close">
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
                </button>
            </div>
            
            <div class="panel-tabs">
                <button class="panel-tab active" data-tab="general">General</button>
                <button class="panel-tab" data-tab="model">Model</button>
                <button class="panel-tab" data-tab="auth">Auth</button>
            </div>
            
            <div class="panel-content" id="panel-content">
                <div class="form-group">
                    <label class="form-label">Gateway URL</label>
                    <input type="text" class="form-input" value="ws://127.0.0.1:8765/ws" id="gateway-url">
                </div>
                
                <div class="form-group">
                    <label class="form-label">Session Key</label>
                    <input type="text" class="form-input" value="main" id="session-key">
                </div>
                
                <div class="form-group toggle-group">
                    <label class="form-label">Streaming</label>
                    <div class="toggle active" id="streaming-toggle"></div>
                </div>
                
                <div class="form-group toggle-group">
                    <label class="form-label">Show tool calls</label>
                    <div class="toggle active" id="tools-toggle"></div>
                </div>
                
                <div class="form-group toggle-group">
                    <label class="form-label">Dark mode</label>
                    <div class="toggle active" id="dark-toggle"></div>
                </div>
            </div>
        </aside>
    </div>
    
    <script>
        // State
        const state = {
            ws: null,
            connected: false,
            messages: [],
            model: 'gpt-4o',
            session: 'main',
            settings: {
                streaming: true,
                showTools: true,
                darkMode: true
            }
        };
        
        // DOM Elements
        const elements = {
            messages: document.getElementById('messages'),
            input: document.getElementById('message-input'),
            sendBtn: document.getElementById('send-btn'),
            modelSelect: document.getElementById('model-select'),
            emptyState: document.getElementById('empty-state'),
            gatewayStatus: document.getElementById('gateway-status'),
            gatewayText: document.getElementById('gateway-text'),
            currentModel: document.getElementById('current-model'),
            rightPanel: document.getElementById('right-panel'),
            settingsToggle: document.getElementById('settings-toggle'),
            panelClose: document.getElementById('panel-close'),
            clearBtn: document.getElementById('clear-btn')
        };
        
        // WebSocket Connection
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            state.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            state.ws.onopen = () => {
                state.connected = true;
                updateStatus(true);
                console.log('Connected to IntelCLaw');
            };
            
            state.ws.onclose = () => {
                state.connected = false;
                updateStatus(false);
                setTimeout(connect, 3000);
            };
            
            state.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            state.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }
        
        function updateStatus(connected) {
            elements.gatewayStatus.className = 'status-dot' + (connected ? '' : ' error');
            elements.gatewayText.textContent = connected ? 'Connected' : 'Disconnected';
        }
        
        // Message Handling
        function handleMessage(data) {
            removeTypingIndicator();
            
            if (data.type === 'typing') {
                showTypingIndicator();
                return;
            }
            
            if (data.type === 'system' && state.messages.length === 0) {
                // Skip initial system message if empty state is showing
                return;
            }
            
            addMessage(data.type, data.content, data.timestamp);
        }
        
        function addMessage(type, content, timestamp = null) {
            elements.emptyState.style.display = 'none';
            
            const time = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
            const sender = type === 'user' ? 'You' : (type === 'agent' ? 'IntelCLaw' : 'System');
            const avatar = type === 'user' ? '👤' : (type === 'agent' ? '🦞' : 'ℹ️');
            
            const msgEl = document.createElement('div');
            msgEl.className = `message ${type}`;
            msgEl.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-body">
                    <div class="message-header">
                        <span class="message-sender">${sender}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-content">${formatContent(content)}</div>
                </div>
            `;
            
            elements.messages.appendChild(msgEl);
            elements.messages.scrollTop = elements.messages.scrollHeight;
            
            state.messages.push({ type, content, timestamp: time });
        }
        
        function formatContent(content) {
            // Escape HTML
            content = content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            // Code blocks
            content = content.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
            // Inline code
            content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
            // Bold
            content = content.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            // Italic
            content = content.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            // Paragraphs
            content = content.split('\n\n').map(p => `<p>${p}</p>`).join('');
            return content;
        }
        
        function showTypingIndicator() {
            if (document.querySelector('.typing-indicator')) return;
            
            const indicator = document.createElement('div');
            indicator.className = 'message agent';
            indicator.innerHTML = `
                <div class="message-avatar">🦞</div>
                <div class="message-body">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
            indicator.id = 'typing';
            elements.messages.appendChild(indicator);
            elements.messages.scrollTop = elements.messages.scrollHeight;
        }
        
        function removeTypingIndicator() {
            const indicator = document.getElementById('typing');
            if (indicator) indicator.remove();
        }
        
        // Send Message
        function sendMessage(text = null) {
            const message = text || elements.input.value.trim();
            if (!message || !state.connected) return;
            
            addMessage('user', message);
            state.ws.send(JSON.stringify({ message }));
            
            elements.input.value = '';
            elements.input.style.height = 'auto';
        }
        
        // Event Listeners
        elements.sendBtn.addEventListener('click', () => sendMessage());
        
        elements.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        elements.input.addEventListener('input', () => {
            elements.input.style.height = 'auto';
            elements.input.style.height = Math.min(elements.input.scrollHeight, 200) + 'px';
        });
        
        elements.modelSelect.addEventListener('change', (e) => {
            state.model = e.target.value;
            elements.currentModel.textContent = state.model;
        });
        
        elements.settingsToggle.addEventListener('click', () => {
            elements.rightPanel.classList.toggle('collapsed');
        });
        
        elements.panelClose.addEventListener('click', () => {
            elements.rightPanel.classList.add('collapsed');
        });
        
        elements.clearBtn.addEventListener('click', () => {
            state.messages = [];
            elements.messages.innerHTML = '';
            elements.emptyState.style.display = 'flex';
            elements.messages.appendChild(elements.emptyState);
        });
        
        // Suggestion chips
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                sendMessage(chip.dataset.prompt);
            });
        });
        
        // Toggle switches
        document.querySelectorAll('.toggle').forEach(toggle => {
            toggle.addEventListener('click', () => {
                toggle.classList.toggle('active');
            });
        });
        
        // Nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
            });
        });
        
        // Panel tabs
        document.querySelectorAll('.panel-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
            });
        });
        
        // Initialize
        connect();
        elements.input.focus();
    </script>
</body>
</html>'''
    
    async def start(self):
        """Start the web server."""
        config = uvicorn.Config(
            self.fastapi,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False
        )
        self._server = uvicorn.Server(config)
        
        logger.info(f"Starting web server at http://{self.host}:{self.port}")
        
        await self._server.serve()
    
    async def stop(self):
        """Stop the web server."""
        if self._server:
            self._server.should_exit = True
            logger.info("Web server stopped")


def create_standalone_server(host: str = "127.0.0.1", port: int = 8765):
    """Create a standalone web server without the full app."""
    return WebServer(app=None, host=host, port=port)
