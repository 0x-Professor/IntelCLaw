"""
Agent Orchestrator - Central coordinator using LangGraph REACT pattern.

This is the root agent that:
- Parses user intent
- Routes to appropriate sub-agents
- Manages multi-step workflows
- Coordinates tool execution
"""

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, Sequence, TypedDict, TYPE_CHECKING

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import BaseModel, Field

from intelclaw.agent.base import (
    AgentContext,
    AgentResponse,
    AgentStatus,
    AgentThought,
    BaseAgent,
)
from intelclaw.agent.router import IntentRouter
from intelclaw.integrations.llm_provider import LLMProvider

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.tools.registry import ToolRegistry
    from intelclaw.core.events import EventBus


class AgentState(TypedDict):
    """State maintained throughout the agent execution."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Dict[str, Any]
    thoughts: List[Dict[str, Any]]
    tools_used: List[str]
    iteration: int
    max_iterations: int


class AgentOrchestrator(BaseAgent):
    """
    Main orchestrator agent using LangGraph for REACT execution.
    
    Architecture:
    - Uses LangGraph's StateGraph for workflow management
    - Implements REACT loop: Reason -> Act -> Observe -> Repeat
    - Routes complex tasks to specialized sub-agents
    - Maintains conversation context across invocations
    - Integrates with memory system for personalization
    - Loads persona from markdown files (OpenClaw-style)
    """
    
    MODEL_NAME = "gpt-4o-mini"  # Default model
    
    # Persona files to load (OpenClaw-style)
    PERSONA_DIR = Path(__file__).parent.parent.parent.parent / "persona"
    PERSONA_FILES = ["AGENT.md", "SOUL.md", "SKILLS.md", "TOOLS.md", "USER.md"]
    
    # Context files injected into system prompt
    BOOTSTRAP_MAX_CHARS = 20000
    
    def __init__(
        self,
        config: "ConfigManager",
        memory: "MemoryManager",
        tools: "ToolRegistry",
        event_bus: "EventBus",
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration manager
            memory: Memory system
            tools: Tool registry
            event_bus: Event bus for notifications
        """
        super().__init__(
            name="IntelCLaw Orchestrator",
            description="Central AI coordinator for task delegation and execution",
            memory=memory,
            tools=tools,
        )
        
        self.config = config
        self.event_bus = event_bus
        self.router = IntentRouter()
        self.sub_agents: Dict[str, BaseAgent] = {}
        
        # LangGraph components (initialized later)
        self._llm_provider: Optional[LLMProvider] = None
        self._llm = None  # Will be set from provider
        self._graph: Optional[StateGraph] = None
        self._compiled_graph = None
        self._langchain_tools: List[BaseTool] = []
        
        # Conversation history (short-term memory)
        self._conversation_history: List[BaseMessage] = []
        self._max_history = 20
        
        logger.info("AgentOrchestrator created")
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and build the LangGraph."""
        logger.info("Initializing AgentOrchestrator...")
        
        # Initialize LLM via unified provider (supports Copilot, OpenAI, Anthropic)
        model_config = self.config.get("models", {})
        llm_config = self.config.get("llm", {})
        
        self._llm_provider = LLMProvider({
            "model": model_config.get("primary", self.MODEL_NAME),
            "temperature": 0.1,
            "provider": llm_config.get("provider", "auto"),
        })
        await self._llm_provider.initialize()
        self._llm = self._llm_provider.llm
        
        logger.info(f"Using LLM provider: {self._llm_provider.active_provider}")
        
        # Get tools from registry
        if self.tools:
            self._langchain_tools = await self.tools.get_langchain_tools()
        
        # Build the REACT graph
        self._build_graph()
        
        # Initialize sub-agents
        await self._initialize_sub_agents()
        
        self.status = AgentStatus.IDLE
        logger.success("AgentOrchestrator initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        logger.info("Shutting down AgentOrchestrator...")
        self.status = AgentStatus.IDLE
        
        # Shutdown sub-agents
        for agent in self.sub_agents.values():
            if hasattr(agent, 'shutdown'):
                await agent.shutdown()
        
        logger.info("AgentOrchestrator shutdown complete")
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history. Useful when switching models."""
        self._conversation_history = []
        logger.info("Conversation history cleared")
    
    async def run(self) -> None:
        """Background run loop for the orchestrator."""
        logger.info("AgentOrchestrator run loop started")
        while self.status != AgentStatus.ERROR:
            await asyncio.sleep(1)  # Heartbeat
    
    def _build_graph(self) -> None:
        """Build the LangGraph REACT workflow."""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("reflect", self._reflect_node)
        
        # Add conditional edges
        workflow.set_entry_point("reason")
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "act": "act",
                "end": END,
            }
        )
        workflow.add_edge("act", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "reason": "reason",
                "end": END,
            }
        )
        
        # Compile the graph
        self._graph = workflow
        self._compiled_graph = workflow.compile()
        
        logger.debug("LangGraph REACT workflow built")
    
    async def _reason_node(self, state: AgentState) -> AgentState:
        """
        Reasoning node - Analyze and plan.
        
        This is where the LLM thinks about the task and decides
        whether to use tools or provide a direct response.
        """
        self.status = AgentStatus.THINKING
        
        # Build prompt with tools
        system_prompt = self._get_system_prompt(state["context"])
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
        # Get LLM response with tool binding
        if self._langchain_tools:
            llm_with_tools = self._llm.bind_tools(self._langchain_tools)
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await self._llm.ainvoke(messages)
        
        # Record thought
        thought = {
            "step": state["iteration"],
            "type": "reason",
            "content": response.content if hasattr(response, 'content') else str(response),
            "has_tool_calls": hasattr(response, 'tool_calls') and bool(response.tool_calls),
            "timestamp": datetime.now().isoformat(),
        }
        
        new_messages = list(state["messages"]) + [response]
        new_thoughts = list(state["thoughts"]) + [thought]
        
        return {
            **state,
            "messages": new_messages,
            "thoughts": new_thoughts,
            "iteration": state["iteration"] + 1,
        }
    
    async def _act_node(self, state: AgentState) -> AgentState:
        """
        Action node - Execute tools.
        
        Processes tool calls from the reasoning step.
        """
        self.status = AgentStatus.EXECUTING
        
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
        
        tool_results = []
        tools_used = list(state["tools_used"])
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.debug(f"Executing tool: {tool_name}")
            tools_used.append(tool_name)
            
            # Execute the tool
            result = await self._execute_tool(tool_name, tool_args)
            
            tool_results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
        
        new_messages = list(state["messages"]) + tool_results
        
        return {
            **state,
            "messages": new_messages,
            "tools_used": tools_used,
        }
    
    async def _reflect_node(self, state: AgentState) -> AgentState:
        """
        Reflection node - Process observations and decide next steps.
        """
        # Check if we have tool results to process
        last_message = state["messages"][-1]
        
        if isinstance(last_message, ToolMessage):
            thought = {
                "step": state["iteration"],
                "type": "observe",
                "tool": last_message.name,
                "observation": last_message.content[:500],  # Truncate for logging
                "timestamp": datetime.now().isoformat(),
            }
            new_thoughts = list(state["thoughts"]) + [thought]
            return {**state, "thoughts": new_thoughts}
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if the agent should continue or end.
        
        Returns:
            "act" if tools should be executed
            "reason" if more thinking needed
            "end" if task is complete
        """
        # Log current iteration (no hard limit - let LLM decide when complete)
        if state["iteration"] > 0:
            logger.debug(f"REACT iteration: {state['iteration']}")
        
        # Check last message
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message is None:
            return "end"
        
        # If AI message with tool calls, go to act
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "act"
        
        # If it's a tool message, go back to reason
        if isinstance(last_message, ToolMessage):
            return "reason"
        
        # If AI message without tool calls, we're done
        if isinstance(last_message, AIMessage) and not getattr(last_message, 'tool_calls', None):
            return "end"
        
        return "end"
    
    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool through the registry."""
        if not self.tools:
            return f"Error: Tool registry not available"
        
        # Ensure tool_args is a dict, not None
        if tool_args is None:
            tool_args = {}
        
        # Log the tool call for debugging
        logger.debug(f"Executing tool '{tool_name}' with args: {tool_args}")
        
        try:
            result = await self.tools.execute(tool_name, tool_args)
            
            # Emit event
            await self.event_bus.emit("tool.executed", {
                "tool": tool_name,
                "args": tool_args,
                "success": True,
            })
            
            return str(result)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool execution error for '{tool_name}': {error_msg}")
            
            # Provide helpful error message for missing parameters
            if "Missing required parameter" in error_msg:
                tool_def = self.tools.get_definition(tool_name)
                if tool_def and tool_def.parameters:
                    required = tool_def.parameters.get("required", [])
                    props = tool_def.parameters.get("properties", {})
                    hint = f"Required parameters: {required}. "
                    hint += f"Got: {list(tool_args.keys())}"
                    return f"Error: {error_msg}. {hint}"
            
            return f"Error executing {tool_name}: {error_msg}"
    
    def _load_persona_files(self) -> Dict[str, str]:
        """
        Load persona markdown files (OpenClaw-style context injection).
        
        Returns:
            Dict mapping filename to content
        """
        persona_content = {}
        
        for filename in self.PERSONA_FILES:
            filepath = self.PERSONA_DIR / filename
            try:
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
                    # Truncate large files
                    if len(content) > self.BOOTSTRAP_MAX_CHARS:
                        content = content[:self.BOOTSTRAP_MAX_CHARS] + "\n\n[... truncated ...]"
                    persona_content[filename] = content
                    logger.debug(f"Loaded persona file: {filename} ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"Failed to load persona file {filename}: {e}")
        
        return persona_content
    
    def _get_system_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build the system prompt using persona files (OpenClaw-style).
        
        Loads and injects AGENT.md, SOUL.md, SKILLS.md, TOOLS.md, USER.md
        to define agent identity, personality, and capabilities.
        """
        
        # Get current model name from LLM provider
        current_model = "unknown"
        if self._llm_provider and hasattr(self._llm_provider, '_llm'):
            llm = self._llm_provider._llm
            if hasattr(llm, 'model'):
                current_model = llm.model
        
        # Load persona files
        persona = self._load_persona_files()
        
        # Build system prompt sections
        sections = []
        
        # Runtime info (model, platform)
        sections.append(f"""## Runtime
- Model: {current_model}
- Provider: GitHub Copilot
- Platform: Windows
- Host: IntelCLaw
""")
        
        # Project Context - Persona Files (OpenClaw-style injection)
        if persona:
            sections.append("# Project Context\n")
            sections.append("The following persona files define your identity and behavior:\n")
            
            # Check if SOUL.md is present (like OpenClaw)
            has_soul = "SOUL.md" in persona
            if has_soul:
                sections.append(
                    "If SOUL.md is present, embody its persona and tone. "
                    "Avoid stiff, generic replies; follow its guidance.\n"
                )
            
            # Inject each persona file
            for filename, content in persona.items():
                sections.append(f"## {filename}\n")
                sections.append(content)
                sections.append("\n")
        
        # Tool usage guidance (minimal - let TOOLS.md handle details)
        sections.append("""## Tool Usage
You MUST use tools when the user asks you to perform actions.
Think step by step before taking action.
Be security-conscious - ask for confirmation for sensitive operations.
""")
        
        # Current runtime context
        sections.append("## Current Context\n")
        
        if context.get("active_window"):
            sections.append(f"Active Window: {context['active_window']}\n")
        
        if context.get("screen_text"):
            sections.append(f"Visible Text: {context['screen_text'][:500]}...\n")
        
        if context.get("user_preferences"):
            prefs = context["user_preferences"]
            sections.append(f"User Preferences: {prefs}\n")
        
        return "\n".join(sections)
    
    async def _initialize_sub_agents(self) -> None:
        """Initialize specialized sub-agents."""
        from intelclaw.agent.sub_agents.research_agent import ResearchAgent
        from intelclaw.agent.sub_agents.coding_agent import CodingAgent
        from intelclaw.agent.sub_agents.task_agent import TaskAgent
        from intelclaw.agent.sub_agents.system_agent import SystemAgent
        
        self.sub_agents = {
            "research": ResearchAgent(memory=self.memory, tools=self.tools),
            "coding": CodingAgent(memory=self.memory, tools=self.tools),
            "task": TaskAgent(memory=self.memory, tools=self.tools),
            "system": SystemAgent(memory=self.memory, tools=self.tools),
        }
        
        # Register sub-agents with router
        for name, agent in self.sub_agents.items():
            self.router.register_agent(name, agent)
        
        logger.info(f"Initialized {len(self.sub_agents)} sub-agents")
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a request through the REACT pipeline.
        
        Args:
            context: Full context for the request
            
        Returns:
            Complete agent response
        """
        start_time = time.time()
        self.clear_thoughts()
        
        # Check if should delegate to sub-agent
        delegation = await self.router.route(context)
        if delegation and delegation.confidence > 0.8:
            sub_agent = self.sub_agents.get(delegation.agent_name)
            if sub_agent:
                logger.info(f"Delegating to {delegation.agent_name}")
                return await sub_agent.process(context)
        
        # Build initial state
        user_message = HumanMessage(content=context.user_message)
        
        # Add conversation history
        history_messages = self._conversation_history[-self._max_history:]
        
        initial_state: AgentState = {
            "messages": history_messages + [user_message],
            "context": {
                "screen_text": context.screen_context.get("text") if context.screen_context else None,
                "active_window": context.active_window,
                "user_preferences": context.user_preferences,
            },
            "thoughts": [],
            "tools_used": [],
            "iteration": 0,
            "max_iterations": 999999,  # Effectively unlimited - let LLM decide when to stop
        }
        
        # Run the graph
        try:
            # Check if we have a valid compiled graph
            if self._compiled_graph is None:
                logger.warning("No compiled graph, using direct LLM")
                return await self._direct_llm_response(context, start_time)
            
            final_state = await self._compiled_graph.ainvoke(initial_state)
            
            # Extract final answer
            last_ai_message = None
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break
            
            answer = last_ai_message.content if last_ai_message else "I couldn't generate a response."
            
            # Update conversation history
            self._conversation_history.append(user_message)
            if last_ai_message:
                self._conversation_history.append(last_ai_message)
            
            # Convert thoughts
            thoughts = [
                AgentThought(
                    step=t.get("step", 0),
                    thought=t.get("content", t.get("observation", "")),
                    action=t.get("tool"),
                    timestamp=datetime.fromisoformat(t["timestamp"]) if "timestamp" in t else datetime.now(),
                )
                for t in final_state["thoughts"]
            ]
            
            latency = (time.time() - start_time) * 1000
            
            # Store in memory
            if self.memory:
                await self.memory.store_interaction(
                    user_message=context.user_message,
                    agent_response=answer,
                    tools_used=final_state["tools_used"],
                )
            
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                answer=answer,
                thoughts=thoughts,
                tools_used=final_state["tools_used"],
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            # Try direct LLM fallback
            try:
                return await self._direct_llm_response(context, start_time)
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                self.status = AgentStatus.ERROR
                
                return AgentResponse(
                    answer=f"I encountered an error: {str(e)}",
                    success=False,
                    error=str(e),
                    latency_ms=(time.time() - start_time) * 1000,
                )
    
    async def _direct_llm_response(self, context: AgentContext, start_time: float) -> AgentResponse:
        """
        Generate a response directly from the LLM without the graph.
        Used as fallback when the graph isn't working.
        """
        if not self._llm:
            return AgentResponse(
                answer="I'm sorry, but I'm not properly initialized. Please check the LLM configuration.",
                success=False,
                error="LLM not initialized",
                latency_ms=(time.time() - start_time) * 1000,
            )
        
        try:
            # Get current model name
            current_model = "unknown"
            if hasattr(self._llm, 'model'):
                current_model = self._llm.model
            
            # Build prompt using persona files (same as _get_system_prompt)
            system_prompt = self._get_system_prompt({
                "active_window": getattr(context, 'active_window', None),
                "screen_text": context.screen_context.get("text") if context.screen_context else None,
                "user_preferences": getattr(context, 'user_preferences', None),
            })
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context.user_message)
            ]
            
            # Call the LLM directly
            response = await self._llm.ainvoke(messages)
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            latency = (time.time() - start_time) * 1000
            
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                answer=answer,
                thoughts=[],
                tools_used=[],
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Direct LLM error: {e}")
            raise
    
    async def can_handle(self, context: AgentContext) -> float:
        """Orchestrator can handle anything."""
        return 1.0
    
    async def change_model(self, model_name: str, provider: Optional[str] = None) -> bool:
        """
        Dynamically change the LLM model.
        
        Args:
            model_name: Name of the model to switch to
            provider: Optional provider override (copilot, openai, anthropic)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Changing model to: {model_name} (provider: {provider or 'auto'})")
            
            new_config = {
                "model": model_name,
                "temperature": 0.1,
            }
            if provider:
                new_config["provider"] = provider
            
            new_provider = LLMProvider(new_config)
            await new_provider.initialize()
            
            # Swap the provider
            self._llm_provider = new_provider
            self._llm = new_provider.llm
            
            # Rebuild the graph with new LLM
            self._build_graph()
            
            logger.info(f"Successfully changed to model: {model_name} via {new_provider.active_provider}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to change model: {e}")
            return False
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM configuration."""
        if self._llm_provider:
            return {
                "provider": self._llm_provider.active_provider,
                "model": getattr(self._llm, "model_name", self.MODEL_NAME),
                "available_providers": self._llm_provider.available_providers,
            }
        return {"provider": "unknown", "model": self.MODEL_NAME}
