"""
Test script for IntelCLaw Web Chat Interface.

This script tests the web server and chat functionality.
"""

import asyncio
import aiohttp
import sys


async def test_server_status(base_url: str):
    """Test if server is running and responding."""
    print(f"\n🔍 Testing server status at {base_url}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/status", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Server is running!")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Agent: {data.get('agent')}")
                    print(f"   LLM Provider: {data.get('llm_provider')}")
                    return True
                else:
                    print(f"❌ Server returned status {response.status}")
                    return False
    except aiohttp.ClientConnectorError:
        print(f"❌ Cannot connect to server at {base_url}")
        print("   Make sure to start the server first: uv run python main.py --web")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_rest_chat(base_url: str, message: str = "Hello, what can you do?"):
    """Test REST API chat endpoint."""
    print(f"\n🔍 Testing REST chat endpoint...")
    print(f"   Message: {message}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/chat",
                json={"message": message},
                timeout=60
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Got response!")
                    print(f"   Response: {data.get('response', '')[:200]}...")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Server returned status {response.status}: {text}")
                    return False
    except asyncio.TimeoutError:
        print(f"❌ Request timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_websocket_chat(base_url: str, message: str = "What's 2+2?"):
    """Test WebSocket chat."""
    print(f"\n🔍 Testing WebSocket chat...")
    print(f"   Message: {message}")
    
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=10) as ws:
                # Wait for welcome message
                welcome = await asyncio.wait_for(ws.receive_json(), timeout=5)
                print(f"✅ Connected! Welcome: {welcome.get('content', '')[:100]}")
                
                # Send message
                await ws.send_json({"message": message})
                print("   Message sent, waiting for response...")
                
                # Wait for typing indicator and response
                while True:
                    try:
                        response = await asyncio.wait_for(ws.receive_json(), timeout=60)
                        msg_type = response.get("type")
                        content = response.get("content", "")
                        
                        if msg_type == "typing":
                            print(f"   💭 {content}")
                        elif msg_type == "agent":
                            print(f"✅ Got response!")
                            print(f"   Response: {content[:200]}...")
                            return True
                        elif msg_type == "error":
                            print(f"❌ Error: {content}")
                            return False
                    except asyncio.TimeoutError:
                        print(f"❌ Response timed out after 60 seconds")
                        return False
                        
    except aiohttp.WSServerHandshakeError as e:
        print(f"❌ WebSocket handshake failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_home_page(base_url: str):
    """Test if home page loads."""
    print(f"\n🔍 Testing home page...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, timeout=5) as response:
                if response.status == 200:
                    text = await response.text()
                    if "IntelCLaw" in text:
                        print(f"✅ Home page loads correctly!")
                        return True
                    else:
                        print(f"❌ Home page content unexpected")
                        return False
                else:
                    print(f"❌ Home page returned status {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def run_all_tests(base_url: str = "http://127.0.0.1:8765"):
    """Run all tests."""
    print("=" * 60)
    print("🧪 IntelCLaw Web Chat Test Suite")
    print("=" * 60)
    print(f"Target: {base_url}")
    
    results = []
    
    # Test 1: Server status
    results.append(("Server Status", await test_server_status(base_url)))
    
    if not results[0][1]:
        print("\n❌ Server not running. Aborting remaining tests.")
        return False
    
    # Test 2: Home page
    results.append(("Home Page", await test_home_page(base_url)))
    
    # Test 3: REST chat
    results.append(("REST Chat", await test_rest_chat(base_url)))
    
    # Test 4: WebSocket chat
    results.append(("WebSocket Chat", await test_websocket_chat(base_url)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n   Total: {passed}/{len(results)} passed")
    print("=" * 60)
    
    return failed == 0


async def interactive_chat(base_url: str = "http://127.0.0.1:8765"):
    """Interactive chat mode via WebSocket."""
    print("=" * 60)
    print("💬 IntelCLaw Interactive Chat")
    print("=" * 60)
    print("Type your messages. Type 'quit' to exit.\n")
    
    ws_url = base_url.replace("http://", "ws://") + "/ws"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # Wait for welcome
                welcome = await ws.receive_json()
                print(f"🤖 {welcome.get('content', 'Connected!')}\n")
                
                while True:
                    # Get user input
                    try:
                        user_input = input("You: ").strip()
                    except EOFError:
                        break
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("\nGoodbye! 👋")
                        break
                    
                    # Send message
                    await ws.send_json({"message": user_input})
                    
                    # Wait for response
                    while True:
                        response = await ws.receive_json()
                        msg_type = response.get("type")
                        content = response.get("content", "")
                        
                        if msg_type == "typing":
                            print(f"   💭 {content}", end="\r")
                        elif msg_type == "agent":
                            print(f"\n🤖 {content}\n")
                            break
                        elif msg_type == "error":
                            print(f"\n❌ {content}\n")
                            break
                            
    except aiohttp.ClientConnectorError:
        print(f"❌ Cannot connect to {base_url}")
        print("   Make sure the server is running: uv run python main.py --web")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IntelCLaw Web Chat")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8765",
        help="Server URL (default: http://127.0.0.1:8765)"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test - only check server status"
    )
    
    args = parser.parse_args()
    
    if args.chat:
        asyncio.run(interactive_chat(args.url))
    elif args.quick:
        asyncio.run(test_server_status(args.url))
    else:
        success = asyncio.run(run_all_tests(args.url))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
