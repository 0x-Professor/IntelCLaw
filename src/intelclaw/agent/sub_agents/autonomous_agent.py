"""
Autonomous Agent - Self-managing AI with full capabilities.

This agent can:
- Modify its own configuration
- Create and improve skills
- Change models and settings
- Diagnose and fix issues
- Store any data
- Do anything except destruction
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from intelclaw.agent.base import BaseAgent

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.tools.registry import ToolRegistry
    from intelclaw.agent.self_improvement import SelfImprovement
    from intelclaw.memory.data_store import DataStore
    from intelclaw.integrations.copilot import CopilotIntegration, ModelManager


class AutonomousAgent(BaseAgent):
    """
    Fully autonomous agent with self-modification capabilities.
    
    This agent has full access to:
    - Configuration management
    - Skill creation and improvement
    - Model switching
    - Issue diagnosis and auto-fixing
    - Data storage (contacts, notes, credentials)
    
    The only restriction: destructive operations require confirmation.
    """
    
    name = "autonomous"
    description = "Self-managing AI agent with full capabilities"
    
    # Destructive operations that always require confirmation
    DESTRUCTIVE_OPERATIONS = [
        "delete_file", "delete_directory", "format_disk", 
        "shutdown_system", "kill_process", "uninstall",
        "drop_table", "truncate", "rm -rf", "del /s",
    ]
    
    def __init__(
        self,
        config: "ConfigManager",
        memory: "MemoryManager",
        tools: "ToolRegistry",
        self_improvement: Optional["SelfImprovement"] = None,
        data_store: Optional["DataStore"] = None,
        copilot: Optional["CopilotIntegration"] = None,
        model_manager: Optional["ModelManager"] = None,
    ):
        """Initialize autonomous agent."""
        super().__init__(config, memory, tools)
        self.self_improvement = self_improvement
        self.data_store = data_store
        self.copilot = copilot
        self.model_manager = model_manager
        
        # Track autonomous actions
        self._action_log: List[Dict[str, Any]] = []
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input with full autonomy.
        
        Args:
            input_data: Input containing message and context
            
        Returns:
            Processing result
        """
        message = input_data.get("message", "")
        context = input_data.get("context", {})
        
        # Analyze intent
        intent = await self._analyze_intent(message)
        
        # Check if destructive
        if self._is_destructive(message, intent):
            return {
                "response": f"⚠️ This operation ({intent.get('action')}) is destructive and requires your confirmation.\n\nPlease confirm by saying 'confirm' or 'yes, proceed'.",
                "requires_confirmation": True,
                "pending_action": intent,
            }
        
        # Route to appropriate handler
        handlers = {
            "modify_config": self._handle_config_modification,
            "create_skill": self._handle_skill_creation,
            "improve_skill": self._handle_skill_improvement,
            "change_model": self._handle_model_change,
            "fix_issue": self._handle_issue_fix,
            "store_data": self._handle_data_storage,
            "search_data": self._handle_data_search,
            "manage_contacts": self._handle_contacts,
            "general": self._handle_general,
        }
        
        handler = handlers.get(intent.get("type", "general"), self._handle_general)
        result = await handler(message, intent, context)
        
        # Log action
        self._log_action(intent, result)
        
        return result
    
    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user intent from message."""
        message_lower = message.lower()
        
        # Config modification
        if any(kw in message_lower for kw in ["change config", "modify setting", "update config", "set"]):
            return {"type": "modify_config", "action": "config_change"}
        
        # Skill creation
        if any(kw in message_lower for kw in ["create skill", "new skill", "add skill", "make skill"]):
            return {"type": "create_skill", "action": "skill_creation"}
        
        # Skill improvement
        if any(kw in message_lower for kw in ["improve skill", "update skill", "fix skill", "enhance skill"]):
            return {"type": "improve_skill", "action": "skill_improvement"}
        
        # Model change
        if any(kw in message_lower for kw in ["change model", "switch model", "use model", "set model"]):
            return {"type": "change_model", "action": "model_change"}
        
        # Issue fixing
        if any(kw in message_lower for kw in ["fix", "debug", "error", "issue", "problem", "broken"]):
            return {"type": "fix_issue", "action": "auto_fix"}
        
        # Data storage
        if any(kw in message_lower for kw in ["save", "store", "remember", "keep"]):
            return {"type": "store_data", "action": "data_store"}
        
        # Data search
        if any(kw in message_lower for kw in ["find", "search", "look up", "get"]):
            return {"type": "search_data", "action": "data_search"}
        
        # Contact management
        if any(kw in message_lower for kw in ["contact", "person", "email", "phone"]):
            return {"type": "manage_contacts", "action": "contact_management"}
        
        return {"type": "general", "action": "general_processing"}
    
    def _is_destructive(self, message: str, intent: Dict[str, Any]) -> bool:
        """Check if operation is destructive."""
        message_lower = message.lower()
        return any(op in message_lower for op in self.DESTRUCTIVE_OPERATIONS)
    
    async def _handle_config_modification(
        self, 
        message: str, 
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle configuration changes."""
        if not self.self_improvement:
            return {"response": "Self-improvement module not available", "success": False}
        
        # Parse config change from message
        # Example: "change model to gpt-4o-mini" or "set temperature to 0.5"
        
        # Use LLM to extract config key and value
        if self._llm:
            prompt = f"""Extract the configuration change from this message.
Message: {message}

Available config keys:
- models.primary, models.fallback, models.coding, models.temperature
- hotkeys.summon, hotkeys.quick_action
- ui.theme, ui.transparency
- autonomy.level
- perception.capture_interval
- Any other config key

Return as JSON: {{"key": "...", "value": "..."}}"""

            response = await self._llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            try:
                if "{" in content:
                    json_str = content[content.index("{"):content.rindex("}")+1]
                    change = json.loads(json_str)
                    
                    result = await self.self_improvement.modify_config(
                        change["key"],
                        change["value"],
                        f"User requested: {message}"
                    )
                    
                    if result.get("success"):
                        return {
                            "response": f"✅ Configuration updated!\n\n**{change['key']}**: {result['old']} → {result['new']}",
                            "success": True,
                            "change": result,
                        }
            except Exception as e:
                logger.error(f"Config change error: {e}")
        
        return {"response": "Could not parse configuration change", "success": False}
    
    async def _handle_skill_creation(
        self,
        message: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle skill creation."""
        if not self.self_improvement:
            return {"response": "Self-improvement module not available", "success": False}
        
        # Extract skill name and description
        if self._llm:
            prompt = f"""Extract skill details from this message.
Message: {message}

Return as JSON: {{"name": "skill_name_snake_case", "description": "What the skill does"}}"""

            response = await self._llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            try:
                if "{" in content:
                    json_str = content[content.index("{"):content.rindex("}")+1]
                    skill_info = json.loads(json_str)
                    
                    result = await self.self_improvement.create_skill(
                        skill_info["name"],
                        skill_info["description"],
                        auto_generate=True
                    )
                    
                    if result.get("success"):
                        return {
                            "response": f"✅ Skill created!\n\n**Name**: {skill_info['name']}\n**Description**: {skill_info['description']}\n**Location**: {result['skill']['path']}",
                            "success": True,
                            "skill": result["skill"],
                        }
            except Exception as e:
                logger.error(f"Skill creation error: {e}")
        
        return {"response": "Could not create skill", "success": False}
    
    async def _handle_skill_improvement(
        self,
        message: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle skill improvement."""
        if not self.self_improvement:
            return {"response": "Self-improvement module not available", "success": False}
        
        # List available skills
        skills = self.self_improvement.list_skills()
        
        if not skills:
            return {"response": "No skills available to improve", "success": False}
        
        # Extract skill name and feedback
        skill_names = [s["name"] for s in skills]
        
        if self._llm:
            prompt = f"""Extract skill improvement details.
Message: {message}
Available skills: {skill_names}

Return as JSON: {{"skill_name": "...", "feedback": "improvement feedback"}}"""

            response = await self._llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            try:
                if "{" in content:
                    json_str = content[content.index("{"):content.rindex("}")+1]
                    improvement = json.loads(json_str)
                    
                    result = await self.self_improvement.improve_skill(
                        improvement["skill_name"],
                        improvement["feedback"]
                    )
                    
                    if result.get("success"):
                        return {
                            "response": f"✅ Skill improved: {improvement['skill_name']}",
                            "success": True,
                        }
            except Exception as e:
                logger.error(f"Skill improvement error: {e}")
        
        return {"response": "Could not improve skill", "success": False}
    
    async def _handle_model_change(
        self,
        message: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle model changes."""
        if self.copilot:
            available = await self.copilot.get_available_models()
            
            # Extract model from message
            for model in available.keys():
                if model.lower() in message.lower():
                    result = await self.copilot.set_model(model)
                    
                    if result.get("success"):
                        return {
                            "response": f"✅ Model changed!\n\n**Previous**: {result['old_model']}\n**New**: {result['new_model']}\n**Description**: {result['description']}",
                            "success": True,
                        }
            
            # Show available models
            model_list = "\n".join([f"- **{k}**: {v}" for k, v in available.items()])
            return {
                "response": f"Please specify which model to use:\n\n{model_list}",
                "success": False,
            }
        
        return {"response": "Copilot integration not available", "success": False}
    
    async def _handle_issue_fix(
        self,
        message: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle issue diagnosis and fixing."""
        if not self.self_improvement:
            return {"response": "Self-improvement module not available", "success": False}
        
        # Check if there's an error in context
        error = context.get("error")
        file_path = context.get("file_path")
        
        if error:
            result = await self.self_improvement.auto_fix_issue(
                error,
                file_path=file_path,
                context=message
            )
            
            if result.get("success"):
                diagnosis = result.get("diagnosis", {}).get("diagnosis", {})
                return {
                    "response": f"🔧 Issue analyzed!\n\n**Root Cause**: {diagnosis.get('root_cause', 'Unknown')}\n\n**Suggested Fixes**:\n" + 
                               "\n".join([f"- {fix}" for fix in diagnosis.get('fixes', [])]),
                    "success": True,
                    "diagnosis": diagnosis,
                }
        
        # Analyze from message
        return {
            "response": "Please provide more details about the issue, or share the error message.",
            "success": False,
        }
    
    async def _handle_data_storage(
        self,
        message: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle data storage requests."""
        if not self.data_store:
            return {"response": "Data store not available", "success": False}
        
        # Determine what to store
        if self._llm:
            prompt = f"""What does the user want to store?
Message: {message}

Categories:
- contact: Person's contact info (name, email, phone)
- note: A note or reminder
- code: Code snippet
- credential: API key or password
- general: Other data

Return as JSON: {{"category": "...", "data": {{...}}}}"""

            response = await self._llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            try:
                if "{" in content:
                    json_str = content[content.index("{"):content.rindex("}")+1]
                    storage_info = json.loads(json_str)
                    
                    category = storage_info.get("category")
                    data = storage_info.get("data", {})
                    
                    if category == "contact":
                        contact_id = await self.data_store.add_contact(**data)
                        return {
                            "response": f"✅ Contact saved! (ID: {contact_id})",
                            "success": True,
                        }
                    elif category == "note":
                        note_id = await self.data_store.add_note(**data)
                        return {
                            "response": f"✅ Note saved! (ID: {note_id})",
                            "success": True,
                        }
                    elif category == "code":
                        snippet_id = await self.data_store.save_code_snippet(**data)
                        return {
                            "response": f"✅ Code snippet saved! (ID: {snippet_id})",
                            "success": True,
                        }
                    elif category == "credential":
                        success = await self.data_store.store_credential(**data)
                        return {
                            "response": "✅ Credential stored securely!" if success else "❌ Failed to store credential",
                            "success": success,
                        }
                    else:
                        key = data.get("key", f"data_{datetime.now().timestamp()}")
                        await self.data_store.set_value(key, data)
                        return {
                            "response": f"✅ Data saved with key: {key}",
                            "success": True,
                        }
            except Exception as e:
                logger.error(f"Data storage error: {e}")
        
        return {"response": "Could not determine what to store", "success": False}
    
    async def _handle_data_search(
        self,
        message: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle data search requests."""
        if not self.data_store:
            return {"response": "Data store not available", "success": False}
        
        # Extract search query
        query = message.replace("find", "").replace("search", "").replace("look up", "").replace("get", "").strip()
        
        results = []
        
        # Search contacts
        contacts = await self.data_store.search_contacts(query)
        if contacts:
            results.append(f"**Contacts** ({len(contacts)}):")
            for c in contacts[:5]:
                results.append(f"  - {c['name']}: {c.get('email', '')} {c.get('phone', '')}")
        
        # Search notes
        notes = await self.data_store.search_notes(query)
        if notes:
            results.append(f"\n**Notes** ({len(notes)}):")
            for n in notes[:5]:
                results.append(f"  - {n['title']}")
        
        # Search code snippets
        snippets = await self.data_store.search_code_snippets(query)
        if snippets:
            results.append(f"\n**Code Snippets** ({len(snippets)}):")
            for s in snippets[:5]:
                results.append(f"  - {s['title']} ({s.get('language', 'unknown')})")
        
        if results:
            return {
                "response": "🔍 Search Results:\n\n" + "\n".join(results),
                "success": True,
            }
        
        return {"response": f"No results found for: {query}", "success": False}
    
    async def _handle_contacts(
        self,
        message: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle contact management."""
        if not self.data_store:
            return {"response": "Data store not available", "success": False}
        
        message_lower = message.lower()
        
        # Add contact
        if "add" in message_lower or "save" in message_lower or "new" in message_lower:
            return await self._handle_data_storage(message, intent, context)
        
        # List contacts
        if "list" in message_lower or "show all" in message_lower:
            contacts = await self.data_store.list_contacts()
            if contacts:
                contact_list = "\n".join([f"- **{c['name']}**: {c.get('email', 'N/A')} | {c.get('phone', 'N/A')}" for c in contacts[:20]])
                return {
                    "response": f"📇 Contacts ({len(contacts)} total):\n\n{contact_list}",
                    "success": True,
                }
            return {"response": "No contacts saved yet", "success": False}
        
        # Search contacts
        return await self._handle_data_search(message, intent, context)
    
    async def _handle_general(
        self,
        message: str,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle general requests."""
        # Use the base agent processing
        return await super().process({"message": message, "context": context})
    
    def _log_action(self, intent: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log an autonomous action."""
        self._action_log.append({
            "timestamp": datetime.now().isoformat(),
            "intent": intent,
            "success": result.get("success", False),
        })
        
        # Keep only last 100 actions in memory
        if len(self._action_log) > 100:
            self._action_log = self._action_log[-100:]
    
    def get_action_log(self) -> List[Dict[str, Any]]:
        """Get recent actions."""
        return self._action_log.copy()
