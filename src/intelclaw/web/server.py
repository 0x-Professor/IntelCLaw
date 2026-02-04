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
        """Return the chat interface HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IntelCLaw - AI Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #252525;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-primary: #ffffff;
            --text-secondary: #a1a1aa;
            --border: #333;
            --success: #22c55e;
            --error: #ef4444;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 1rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }
        
        .logo-text {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Chat Container */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message {
            max-width: 80%;
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: var(--accent);
            align-self: flex-end;
            border-bottom-right-radius: 0.25rem;
        }
        
        .message.agent {
            background: var(--bg-tertiary);
            align-self: flex-start;
            border-bottom-left-radius: 0.25rem;
        }
        
        .message.system {
            background: transparent;
            border: 1px solid var(--border);
            align-self: center;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .message.typing {
            background: var(--bg-tertiary);
            align-self: flex-start;
            color: var(--text-secondary);
        }
        
        .message-content {
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .message-content code {
            background: rgba(0,0,0,0.3);
            padding: 0.125rem 0.375rem;
            border-radius: 0.25rem;
            font-family: 'Consolas', monospace;
            font-size: 0.875em;
        }
        
        .message-content pre {
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin: 0.5rem 0;
        }
        
        .message-content pre code {
            background: none;
            padding: 0;
        }
        
        .message-time {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
            opacity: 0.7;
        }
        
        /* Input Area */
        .input-container {
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            padding: 1rem 1.5rem;
        }
        
        .input-wrapper {
            display: flex;
            gap: 0.75rem;
            max-width: 900px;
            margin: 0 auto;
        }
        
        .input-field {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 0.875rem 1.25rem;
            color: var(--text-primary);
            font-size: 1rem;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .input-field:focus {
            border-color: var(--accent);
        }
        
        .input-field::placeholder {
            color: var(--text-secondary);
        }
        
        .send-button {
            background: var(--accent);
            border: none;
            border-radius: 0.75rem;
            padding: 0.875rem 1.5rem;
            color: white;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .send-button:hover {
            background: var(--accent-hover);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .send-button svg {
            width: 20px;
            height: 20px;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--border);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .message {
                max-width: 90%;
            }
            
            .input-wrapper {
                flex-direction: column;
            }
            
            .send-button {
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">
            <div class="logo-icon">🤖</div>
            <span class="logo-text">IntelCLaw</span>
        </div>
        <div class="status">
            <div class="status-dot"></div>
            <span id="status-text">Connected</span>
        </div>
    </header>
    
    <main class="chat-container" id="chat">
        <!-- Messages will be inserted here -->
    </main>
    
    <footer class="input-container">
        <div class="input-wrapper">
            <input 
                type="text" 
                class="input-field" 
                id="message-input" 
                placeholder="Type your message..." 
                autocomplete="off"
            >
            <button class="send-button" id="send-button">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                </svg>
                Send
            </button>
        </div>
    </footer>
    
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-button');
        const statusText = document.getElementById('status-text');
        
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                statusText.textContent = 'Connected';
                document.querySelector('.status-dot').style.background = '#22c55e';
                reconnectAttempts = 0;
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                statusText.textContent = 'Disconnected';
                document.querySelector('.status-dot').style.background = '#ef4444';
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connect, 2000 * reconnectAttempts);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }
        
        function handleMessage(data) {
            // Remove typing indicator if present
            const typingMsg = document.querySelector('.message.typing');
            if (typingMsg && data.type !== 'typing') {
                typingMsg.remove();
            }
            
            if (data.type === 'typing') {
                // Show typing indicator
                if (!document.querySelector('.message.typing')) {
                    addMessage(data.content, 'typing');
                }
                return;
            }
            
            addMessage(data.content, data.type, data.timestamp);
        }
        
        function addMessage(content, type, timestamp = null) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${type}`;
            
            // Format content (basic markdown-like formatting)
            let formattedContent = formatContent(content);
            
            let html = `<div class="message-content">${formattedContent}</div>`;
            
            if (timestamp && type !== 'typing') {
                const time = new Date(timestamp).toLocaleTimeString();
                html += `<div class="message-time">${time}</div>`;
            }
            
            msgDiv.innerHTML = html;
            chat.appendChild(msgDiv);
            chat.scrollTop = chat.scrollHeight;
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
            
            return content;
        }
        
        function sendMessage() {
            const message = input.value.trim();
            if (!message || ws.readyState !== WebSocket.OPEN) return;
            
            // Add user message to chat
            addMessage(message, 'user', new Date().toISOString());
            
            // Send to server
            ws.send(JSON.stringify({ message }));
            
            // Clear input
            input.value = '';
            input.focus();
        }
        
        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Connect on load
        connect();
        
        // Focus input on load
        input.focus();
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
