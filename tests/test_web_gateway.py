#!/usr/bin/env python3
"""
IntelCLaw Web Gateway Comprehensive Test Suite

Tests the web gateway at http://127.0.0.1:8765/ including:
- REST API endpoints (status, models, chat)
- WebSocket connectivity and messaging
- Model switching between different providers
- Tool invocations
- Streaming responses

Usage:
    uv run python tests/test_web_gateway.py
    
Or run specific tests:
    uv run python tests/test_web_gateway.py --test models
    uv run python tests/test_web_gateway.py --test chat
    uv run python tests/test_web_gateway.py --test websocket
"""

import asyncio
import json
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    import aiohttp
    import websockets
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp", "websockets"], check=True)
    import aiohttp
    import websockets


# Configuration
BASE_URL = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765/ws"
TIMEOUT = 30

# Test models to try (will be filtered based on availability)
TEST_MODELS = [
    # Copilot models (prioritized)
    {"id": "gpt-4.1", "provider": "github-copilot", "priority": 1},
    {"id": "claude-sonnet-4", "provider": "github-copilot", "priority": 2},
    {"id": "gemini-2.5-pro", "provider": "github-copilot", "priority": 3},
    {"id": "gpt-4o", "provider": "github-copilot", "priority": 4},
    {"id": "claude-sonnet-4.5", "provider": "github-copilot", "priority": 5},
    # Free tier models (fallback)
    {"id": "gpt-4o-mini", "provider": "github-models", "priority": 10},
    {"id": "llama-3.3-70b", "provider": "github-models", "priority": 11},
]

# Test prompts for various capabilities
TEST_PROMPTS = [
    {"name": "simple", "prompt": "What is 2+2? Answer briefly."},
    {"name": "reasoning", "prompt": "Explain why the sky appears blue in one sentence."},
    {"name": "code", "prompt": "Write a Python one-liner that prints 'Hello World'"},
    {"name": "creative", "prompt": "Make up a 5-word story."},
]


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.details: List[Dict[str, Any]] = []
    
    def add(self, name: str, status: str, message: str = "", duration: float = 0):
        self.details.append({
            "name": name,
            "status": status,
            "message": message,
            "duration": duration
        })
        if status == "PASSED":
            self.passed += 1
        elif status == "FAILED":
            self.failed += 1
        else:
            self.skipped += 1
    
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total: {total} | Passed: {self.passed} | Failed: {self.failed} | Skipped: {self.skipped}")
        print("-" * 60)
        for detail in self.details:
            status_icon = "✓" if detail["status"] == "PASSED" else "✗" if detail["status"] == "FAILED" else "○"
            duration_str = f"({detail['duration']:.2f}s)" if detail['duration'] > 0 else ""
            print(f"  {status_icon} {detail['name']}: {detail['status']} {duration_str}")
            if detail["message"]:
                print(f"      → {detail['message'][:100]}")
        print("=" * 60)
        return self.failed == 0


results = TestResults()


def log(msg: str, level: str = "INFO"):
    """Print log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ", "OK": "✓", "WARN": "⚠", "ERROR": "✗", "TEST": "🧪"}
    print(f"[{timestamp}] {icons.get(level, '•')} {msg}")


async def test_server_status():
    """Test that the server is running and responsive"""
    log("Testing server status...", "TEST")
    start = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/status", timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    duration = time.time() - start
                    log(f"Server status: {data.get('status', 'unknown')} - Model: {data.get('model', 'unknown')}", "OK")
                    results.add("Server Status", "PASSED", f"Status: {data.get('status')}", duration)
                    return True
                else:
                    results.add("Server Status", "FAILED", f"HTTP {resp.status}")
                    return False
    except Exception as e:
        results.add("Server Status", "FAILED", str(e))
        log(f"Server not reachable: {e}", "ERROR")
        return False


async def test_models_endpoint():
    """Test the /api/models endpoint returns valid models"""
    log("Testing models endpoint...", "TEST")
    start = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/models", timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                if resp.status != 200:
                    results.add("Models Endpoint", "FAILED", f"HTTP {resp.status}")
                    return []
                
                data = await resp.json()
                duration = time.time() - start
                models = data.get("models", [])
                has_copilot = data.get("has_copilot", False)
                is_dynamic = data.get("dynamic", False)
                current = data.get("current", "")
                
                log(f"Found {len(models)} models | Copilot: {has_copilot} | Dynamic: {is_dynamic} | Current: {current}", "OK")
                
                # Categorize models
                categories = {}
                for model in models:
                    cat = model.get("category", "Other")
                    categories[cat] = categories.get(cat, 0) + 1
                
                for cat, count in categories.items():
                    log(f"  - {cat}: {count} models", "INFO")
                
                results.add("Models Endpoint", "PASSED", f"{len(models)} models loaded (dynamic: {is_dynamic})", duration)
                return models
                
    except Exception as e:
        results.add("Models Endpoint", "FAILED", str(e))
        log(f"Failed to get models: {e}", "ERROR")
        return []


async def test_model_switch(model_id: str, provider: str = "github-copilot"):
    """Test switching to a specific model"""
    log(f"Testing model switch to: {model_id} ({provider})...", "TEST")
    start = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"model": model_id, "provider": provider}
            async with session.post(
                f"{BASE_URL}/api/set_model",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT)
            ) as resp:
                duration = time.time() - start
                if resp.status == 200:
                    data = await resp.json()
                    new_model = data.get("model", model_id)
                    log(f"Switched to model: {new_model}", "OK")
                    results.add(f"Switch to {model_id}", "PASSED", f"Now using {new_model}", duration)
                    return True
                else:
                    text = await resp.text()
                    results.add(f"Switch to {model_id}", "FAILED", f"HTTP {resp.status}: {text[:50]}")
                    return False
    except Exception as e:
        results.add(f"Switch to {model_id}", "FAILED", str(e))
        log(f"Model switch failed: {e}", "ERROR")
        return False


async def test_rest_chat(prompt: str, name: str = "chat"):
    """Test REST chat endpoint"""
    log(f"Testing REST chat ({name})...", "TEST")
    start = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"message": prompt}
            async with session.post(
                f"{BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                duration = time.time() - start
                if resp.status == 200:
                    data = await resp.json()
                    response = data.get("response", "")
                    model = data.get("model", "unknown")
                    
                    # Truncate for display
                    display_response = response[:100] + "..." if len(response) > 100 else response
                    log(f"Response from {model}: {display_response}", "OK")
                    results.add(f"REST Chat ({name})", "PASSED", f"Got response ({len(response)} chars)", duration)
                    return response
                else:
                    text = await resp.text()
                    results.add(f"REST Chat ({name})", "FAILED", f"HTTP {resp.status}: {text[:50]}")
                    return None
    except Exception as e:
        results.add(f"REST Chat ({name})", "FAILED", str(e))
        log(f"REST chat failed: {e}", "ERROR")
        return None


async def test_websocket_connection():
    """Test WebSocket connection and basic messaging"""
    log("Testing WebSocket connection...", "TEST")
    start = time.time()
    
    try:
        async with websockets.connect(WS_URL, ping_timeout=30) as ws:
            duration = time.time() - start
            log("WebSocket connected", "OK")
            
            # Wait for initial state message
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                msg_type = data.get("type", "unknown")
                log(f"Received initial message type: {msg_type}", "INFO")
                results.add("WebSocket Connect", "PASSED", f"Connected, got {msg_type} message", duration)
                return True
            except asyncio.TimeoutError:
                results.add("WebSocket Connect", "PASSED", "Connected (no initial message)", duration)
                return True
                
    except Exception as e:
        results.add("WebSocket Connect", "FAILED", str(e))
        log(f"WebSocket connection failed: {e}", "ERROR")
        return False


async def test_websocket_chat(prompt: str):
    """Test chat via WebSocket"""
    log("Testing WebSocket chat...", "TEST")
    start = time.time()
    
    try:
        async with websockets.connect(WS_URL, ping_timeout=30) as ws:
            # Skip initial state messages
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                    data = json.loads(msg)
                    if data.get("type") != "state":
                        break
            except asyncio.TimeoutError:
                pass
            
            # Send chat message
            chat_msg = json.dumps({
                "type": "message",
                "content": prompt,
                "streaming": False
            })
            await ws.send(chat_msg)
            log(f"Sent: {prompt[:50]}...", "INFO")
            
            # Collect response
            full_response = ""
            response_type = None
            
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=60)
                    data = json.loads(msg)
                    msg_type = data.get("type", "")
                    
                    if msg_type == "response":
                        full_response = data.get("content", "")
                        response_type = "complete"
                        break
                    elif msg_type == "stream_start":
                        response_type = "streaming"
                    elif msg_type == "stream_chunk":
                        full_response += data.get("content", "")
                    elif msg_type == "stream_end":
                        break
                    elif msg_type == "error":
                        raise Exception(data.get("error", "Unknown error"))
                        
            except asyncio.TimeoutError:
                pass
            
            duration = time.time() - start
            
            if full_response:
                display = full_response[:80] + "..." if len(full_response) > 80 else full_response
                log(f"WebSocket response ({response_type}): {display}", "OK")
                results.add("WebSocket Chat", "PASSED", f"Got {len(full_response)} chars ({response_type})", duration)
                return full_response
            else:
                results.add("WebSocket Chat", "FAILED", "No response received")
                return None
                
    except Exception as e:
        results.add("WebSocket Chat", "FAILED", str(e))
        log(f"WebSocket chat failed: {e}", "ERROR")
        return None


async def test_tools_endpoint():
    """Test the tools listing endpoint"""
    log("Testing tools endpoint...", "TEST")
    start = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/tools", timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                duration = time.time() - start
                if resp.status == 200:
                    data = await resp.json()
                    tools = data.get("tools", [])
                    log(f"Found {len(tools)} tools available", "OK")
                    
                    # List some tools
                    for tool in tools[:5]:
                        name = tool.get("name", tool) if isinstance(tool, dict) else tool
                        log(f"  - {name}", "INFO")
                    if len(tools) > 5:
                        log(f"  ... and {len(tools) - 5} more", "INFO")
                    
                    results.add("Tools Endpoint", "PASSED", f"{len(tools)} tools found", duration)
                    return tools
                elif resp.status == 404:
                    results.add("Tools Endpoint", "SKIPPED", "Endpoint not implemented")
                    return []
                else:
                    results.add("Tools Endpoint", "FAILED", f"HTTP {resp.status}")
                    return []
    except Exception as e:
        results.add("Tools Endpoint", "SKIPPED", f"Not available: {e}")
        return []


async def test_multi_model_chat(available_models: List[Dict]):
    """Test chat with multiple different models"""
    log("Testing multi-model chat...", "TEST")
    
    # Filter to models that exist in available_models
    available_ids = {m.get("id") for m in available_models}
    models_to_test = []
    
    for test_model in TEST_MODELS:
        if test_model["id"] in available_ids:
            models_to_test.append(test_model)
            if len(models_to_test) >= 3:  # Test up to 3 models
                break
    
    if not models_to_test:
        log("No matching test models found, using first 3 available", "WARN")
        models_to_test = [{"id": m["id"], "provider": m.get("provider", "github-copilot")} 
                         for m in available_models[:3]]
    
    for model_info in models_to_test:
        model_id = model_info["id"]
        provider = model_info.get("provider", "github-copilot")
        
        # Switch model
        if await test_model_switch(model_id, provider):
            # Test chat with this model
            prompt = "Say 'Hello' and your model name in one line."
            await test_rest_chat(prompt, f"{model_id}")
            await asyncio.sleep(1)  # Rate limit


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("IntelCLaw Web Gateway Test Suite")
    print(f"Target: {BASE_URL}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Test 1: Server status
    if not await test_server_status():
        log("Server not running! Please start with: uv run python -m intelclaw gateway", "ERROR")
        results.summary()
        return False
    
    # Test 2: Models endpoint
    available_models = await test_models_endpoint()
    if not available_models:
        log("No models available, some tests will be skipped", "WARN")
    
    # Test 3: Tools endpoint
    await test_tools_endpoint()
    
    # Test 4: WebSocket connection
    await test_websocket_connection()
    
    # Test 5: Basic REST chat
    await test_rest_chat("Hello, please respond with 'Hi!'", "basic")
    
    # Test 6: WebSocket chat
    await test_websocket_chat("What is 1+1? Just the number please.")
    
    # Test 7: Multi-model testing (if models available)
    if available_models:
        await test_multi_model_chat(available_models)
    
    # Summary
    return results.summary()


async def run_specific_test(test_name: str):
    """Run a specific test"""
    if test_name == "status":
        await test_server_status()
    elif test_name == "models":
        await test_models_endpoint()
    elif test_name == "tools":
        await test_tools_endpoint()
    elif test_name == "websocket":
        await test_websocket_connection()
        await test_websocket_chat("Hello from test!")
    elif test_name == "chat":
        await test_rest_chat("Hello, this is a test!", "test")
    elif test_name == "multi":
        models = await test_models_endpoint()
        await test_multi_model_chat(models)
    else:
        log(f"Unknown test: {test_name}", "ERROR")
        return False
    
    return results.summary()


def main():
    global BASE_URL, WS_URL
    
    parser = argparse.ArgumentParser(description="IntelCLaw Web Gateway Test Suite")
    parser.add_argument("--test", "-t", type=str, help="Run specific test (status, models, tools, websocket, chat, multi)")
    parser.add_argument("--url", type=str, default=None, help="Server URL (default: http://127.0.0.1:8765)")
    args = parser.parse_args()
    
    if args.url:
        BASE_URL = args.url
        WS_URL = args.url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
    
    if args.test:
        success = asyncio.run(run_specific_test(args.test))
    else:
        success = asyncio.run(run_all_tests())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
