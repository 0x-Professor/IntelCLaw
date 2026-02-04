"""
IntelCLaw Comprehensive Test Suite

This module tests all major components:
1. Environment and configuration loading
2. LLM Provider (GitHub Models API)
3. Tools (file ops, search, shell, code execution)
4. Web server basics
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TestResult:
    """Simple test result container."""
    
    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f" - {self.message}" if self.message else ""
        return f"{status} | {self.name}{msg}"


class IntelClawTestSuite:
    """Comprehensive test suite for IntelCLaw."""
    
    def __init__(self):
        self.results: list[TestResult] = []
        self.passed = 0
        self.failed = 0
    
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.results.append(result)
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1
        print(result)
    
    # =========================================================================
    # Environment Tests
    # =========================================================================
    
    async def test_env_github_token(self) -> TestResult:
        """Test GitHub token is configured."""
        token = os.getenv("GITHUB_TOKEN")
        if token and token.startswith("github_pat_"):
            return TestResult("GitHub Token", True, "Token found and valid format")
        elif token:
            return TestResult("GitHub Token", True, "Token found (may need verification)")
        return TestResult("GitHub Token", False, "GITHUB_TOKEN not set in .env")
    
    async def test_env_tavily_key(self) -> TestResult:
        """Test Tavily API key is configured."""
        key = os.getenv("TAVILY_API_KEY")
        if key and key.startswith("tvly-"):
            return TestResult("Tavily API Key", True, "Key found and valid format")
        elif key:
            return TestResult("Tavily API Key", True, "Key found (may need verification)")
        return TestResult("Tavily API Key", False, "TAVILY_API_KEY not set in .env")
    
    async def test_env_dotenv_loading(self) -> TestResult:
        """Test .env file loading."""
        env_file = Path(".env")
        if env_file.exists():
            content = env_file.read_text()
            if "GITHUB_TOKEN" in content:
                return TestResult("DotEnv Loading", True, ".env file exists and has GITHUB_TOKEN")
        return TestResult("DotEnv Loading", False, ".env file missing or incomplete")
    
    # =========================================================================
    # LLM Provider Tests
    # =========================================================================
    
    async def test_llm_models_config(self) -> TestResult:
        """Test LLM models configuration."""
        try:
            from intelclaw.integrations.llm_provider import GITHUB_MODELS
            
            required_models = ["gpt-4o", "gpt-4o-mini", "llama-3.3-70b", "mistral-large"]
            missing = [m for m in required_models if m not in GITHUB_MODELS]
            
            if missing:
                return TestResult("LLM Models Config", False, f"Missing models: {missing}")
            
            return TestResult("LLM Models Config", True, f"{len(GITHUB_MODELS)} models configured")
        except ImportError as e:
            return TestResult("LLM Models Config", False, f"Import error: {e}")
    
    async def test_llm_provider_init(self) -> TestResult:
        """Test LLM provider initialization."""
        try:
            from intelclaw.integrations.llm_provider import CopilotLLM, DEFAULT_MODEL
            
            llm = CopilotLLM(model=DEFAULT_MODEL)
            
            # Check if token is available
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                return TestResult("LLM Provider Init", False, "No GitHub token available")
            
            return TestResult("LLM Provider Init", True, f"Using model: {DEFAULT_MODEL}")
        except Exception as e:
            return TestResult("LLM Provider Init", False, f"Error: {e}")
    
    async def test_llm_api_connection(self) -> TestResult:
        """Test LLM API connection (simple call)."""
        try:
            from intelclaw.integrations.llm_provider import CopilotLLM
            
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                return TestResult("LLM API Connection", False, "No GitHub token")
            
            llm = CopilotLLM(model="gpt-4o-mini")
            initialized = await llm.initialize()
            
            if initialized:
                return TestResult("LLM API Connection", True, "API connection successful")
            return TestResult("LLM API Connection", False, "Failed to initialize LLM")
        except Exception as e:
            return TestResult("LLM API Connection", False, f"Error: {e}")
    
    # =========================================================================
    # Tool Tests
    # =========================================================================
    
    async def test_tool_file_read(self) -> TestResult:
        """Test file read tool."""
        try:
            from intelclaw.tools.builtin.file_ops import FileReadTool
            
            tool = FileReadTool()
            
            # Read this test file itself
            result = await tool.execute(path=__file__)
            
            if result.success and "IntelCLaw" in result.data:
                return TestResult("File Read Tool", True, f"Read {len(result.data)} chars")
            return TestResult("File Read Tool", False, f"Error: {result.error}")
        except Exception as e:
            return TestResult("File Read Tool", False, f"Error: {e}")
    
    async def test_tool_file_write(self) -> TestResult:
        """Test file write tool."""
        try:
            from intelclaw.tools.builtin.file_ops import FileWriteTool
            
            tool = FileWriteTool()
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                temp_path = f.name
            
            test_content = "IntelCLaw test file content"
            result = await tool.execute(path=temp_path, content=test_content)
            
            # Verify
            if result.success:
                content = Path(temp_path).read_text()
                os.unlink(temp_path)
                if content == test_content:
                    return TestResult("File Write Tool", True, "Write and verify successful")
            
            return TestResult("File Write Tool", False, f"Error: {result.error}")
        except Exception as e:
            return TestResult("File Write Tool", False, f"Error: {e}")
    
    async def test_tool_tavily_search(self) -> TestResult:
        """Test Tavily search tool."""
        try:
            from intelclaw.tools.builtin.search import TavilySearchTool
            
            tool = TavilySearchTool()
            
            if not tool._client:
                return TestResult("Tavily Search Tool", False, "Tavily client not initialized - check API key")
            
            result = await tool.execute(query="Python programming language", max_results=2)
            
            if result.success and result.data:
                return TestResult("Tavily Search Tool", True, f"Found {len(result.data)} results")
            return TestResult("Tavily Search Tool", False, f"Error: {result.error}")
        except Exception as e:
            return TestResult("Tavily Search Tool", False, f"Error: {e}")
    
    async def test_tool_shell_command(self) -> TestResult:
        """Test shell command tool."""
        try:
            from intelclaw.tools.builtin.shell import ShellCommandTool
            
            tool = ShellCommandTool()
            
            # Simple echo command
            if sys.platform == "win32":
                cmd = "echo Hello from IntelCLaw"
            else:
                cmd = "echo 'Hello from IntelCLaw'"
            
            result = await tool.execute(command=cmd)
            
            if result.success and "Hello" in result.data.get("stdout", ""):
                return TestResult("Shell Command Tool", True, "Command executed successfully")
            return TestResult("Shell Command Tool", False, f"Error: {result.error}")
        except Exception as e:
            return TestResult("Shell Command Tool", False, f"Error: {e}")
    
    async def test_tool_code_execution(self) -> TestResult:
        """Test code execution tool."""
        try:
            from intelclaw.tools.builtin.shell import CodeExecutionTool
            
            tool = CodeExecutionTool()
            
            code = "print('IntelCLaw code execution test')\nprint(2 + 2)"
            result = await tool.execute(code=code)
            
            if result.success and "IntelCLaw" in result.data.get("output", ""):
                return TestResult("Code Execution Tool", True, "Code executed successfully")
            return TestResult("Code Execution Tool", False, f"Error: {result.error}")
        except Exception as e:
            return TestResult("Code Execution Tool", False, f"Error: {e}")
    
    async def test_tool_security_blocked(self) -> TestResult:
        """Test shell command security blocking."""
        try:
            from intelclaw.tools.builtin.shell import ShellCommandTool
            
            tool = ShellCommandTool()
            
            # This should be blocked
            result = await tool.execute(command="rm -rf /")
            
            if not result.success and "blocked" in result.error.lower():
                return TestResult("Shell Security", True, "Dangerous command blocked")
            return TestResult("Shell Security", False, "Dangerous command NOT blocked!")
        except Exception as e:
            return TestResult("Shell Security", False, f"Error: {e}")
    
    # =========================================================================
    # Registry Tests
    # =========================================================================
    
    async def test_tool_registry(self) -> TestResult:
        """Test tool registry initialization."""
        try:
            # Mock config and security managers
            class MockConfig:
                def get(self, key, default=None):
                    return default or {}
            
            class MockSecurity:
                async def has_permission(self, perm):
                    return True
            
            from intelclaw.tools.registry import ToolRegistry
            
            registry = ToolRegistry(MockConfig(), MockSecurity())
            await registry.initialize()
            
            tools = registry.list_tools()
            expected_tools = ["tavily_search", "file_read", "file_write", "shell_command", "execute_code"]
            
            registered_names = [t.name for t in tools]
            found = [t for t in expected_tools if t in registered_names]
            
            if len(found) >= 5:
                return TestResult("Tool Registry", True, f"Registered {len(tools)} tools")
            return TestResult("Tool Registry", False, f"Missing tools. Found: {registered_names}")
        except Exception as e:
            return TestResult("Tool Registry", False, f"Error: {e}")
    
    # =========================================================================
    # Config Tests
    # =========================================================================
    
    async def test_config_yaml(self) -> TestResult:
        """Test config.yaml exists and is valid."""
        try:
            import yaml
            
            config_path = Path("config.yaml")
            if not config_path.exists():
                return TestResult("Config YAML", False, "config.yaml not found")
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            if config and isinstance(config, dict):
                return TestResult("Config YAML", True, f"Config loaded with {len(config)} sections")
            return TestResult("Config YAML", False, "Config file is empty or invalid")
        except Exception as e:
            return TestResult("Config YAML", False, f"Error: {e}")
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    async def run_all(self) -> None:
        """Run all tests."""
        print("\n" + "=" * 70)
        print("🦅 IntelCLaw Comprehensive Test Suite")
        print("=" * 70 + "\n")
        
        # Environment tests
        print("📁 Environment Tests")
        print("-" * 40)
        self.add_result(await self.test_env_dotenv_loading())
        self.add_result(await self.test_env_github_token())
        self.add_result(await self.test_env_tavily_key())
        print()
        
        # LLM tests
        print("🤖 LLM Provider Tests")
        print("-" * 40)
        self.add_result(await self.test_llm_models_config())
        self.add_result(await self.test_llm_provider_init())
        self.add_result(await self.test_llm_api_connection())
        print()
        
        # Tool tests
        print("🔧 Tool Tests")
        print("-" * 40)
        self.add_result(await self.test_tool_file_read())
        self.add_result(await self.test_tool_file_write())
        self.add_result(await self.test_tool_tavily_search())
        self.add_result(await self.test_tool_shell_command())
        self.add_result(await self.test_tool_code_execution())
        self.add_result(await self.test_tool_security_blocked())
        print()
        
        # Registry tests
        print("📦 Registry Tests")
        print("-" * 40)
        self.add_result(await self.test_tool_registry())
        print()
        
        # Config tests
        print("⚙️ Config Tests")
        print("-" * 40)
        self.add_result(await self.test_config_yaml())
        print()
        
        # Summary
        print("=" * 70)
        print(f"📊 Test Summary: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        if self.failed > 0:
            print("\n❌ Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"   - {result.name}: {result.message}")
        
        print()
        
        # Return exit code
        return self.failed == 0


async def main():
    """Main entry point for tests."""
    suite = IntelClawTestSuite()
    success = await suite.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
