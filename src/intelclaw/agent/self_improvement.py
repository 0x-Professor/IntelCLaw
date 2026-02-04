"""
Self-Improvement Module - Autonomous skill creation and issue fixing.

This module enables IntelCLaw to:
- Create new skills autonomously
- Modify its own configuration
- Diagnose and fix issues
- Improve over time
"""

import asyncio
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager


class SelfImprovement:
    """
    Self-improvement capabilities for IntelCLaw.
    
    Enables the agent to:
    - Create and modify skills
    - Update configuration
    - Fix detected issues
    - Learn from interactions
    """
    
    def __init__(self, config: "ConfigManager"):
        """
        Initialize self-improvement module.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.skills_dir = Path("skills")
        self.skills_dir.mkdir(exist_ok=True)
        self.improvement_log: List[Dict[str, Any]] = []
        self._llm = None
    
    async def initialize(self, llm) -> None:
        """Initialize with LLM access."""
        self._llm = llm
        logger.info("Self-improvement module initialized")
    
    # =========================================================================
    # SKILL CREATION
    # =========================================================================
    
    async def create_skill(
        self,
        name: str,
        description: str,
        code: Optional[str] = None,
        auto_generate: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new skill.
        
        Args:
            name: Skill name
            description: What the skill does
            code: Optional skill code (generated if not provided)
            auto_generate: Whether to auto-generate code
            
        Returns:
            Skill metadata
        """
        logger.info(f"Creating skill: {name}")
        
        skill_path = self.skills_dir / f"{name}.py"
        
        # Generate code if not provided
        if not code and auto_generate and self._llm:
            code = await self._generate_skill_code(name, description)
        
        if not code:
            return {"success": False, "error": "No code provided and generation failed"}
        
        # Save skill file
        skill_path.write_text(code, encoding="utf-8")
        
        # Create metadata
        metadata = {
            "name": name,
            "description": description,
            "created": datetime.now().isoformat(),
            "path": str(skill_path),
            "version": "1.0.0",
            "auto_generated": auto_generate,
        }
        
        # Save metadata
        meta_path = self.skills_dir / f"{name}.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        
        # Log improvement
        self._log_improvement("skill_created", {
            "name": name,
            "description": description,
        })
        
        logger.info(f"Skill created: {skill_path}")
        return {"success": True, "skill": metadata}
    
    async def _generate_skill_code(self, name: str, description: str) -> str:
        """Generate skill code using LLM."""
        prompt = f"""Create a Python skill module for IntelCLaw agent.

Skill Name: {name}
Description: {description}

Requirements:
1. Create a class that extends from a base skill pattern
2. Include an async `execute` method
3. Add proper docstrings and type hints
4. Handle errors gracefully
5. Return structured results

Template:
```python
\"\"\"
Skill: {name}
{description}
\"\"\"

from typing import Any, Dict
from loguru import logger


class {name.title().replace('_', '')}Skill:
    \"\"\"
    {description}
    \"\"\"
    
    name = "{name}"
    description = "{description}"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        \"\"\"Execute the skill.\"\"\"
        try:
            # Implementation here
            result = {{}}
            return {{"success": True, "result": result}}
        except Exception as e:
            logger.error(f"Skill error: {{e}}")
            return {{"success": False, "error": str(e)}}
```

Generate the complete skill implementation:"""

        if self._llm:
            response = await self._llm.ainvoke(prompt)
            # Extract code from response
            code = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up code block markers
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            return code.strip()
        
        return ""
    
    async def improve_skill(self, name: str, feedback: str) -> Dict[str, Any]:
        """
        Improve an existing skill based on feedback.
        
        Args:
            name: Skill name
            feedback: Improvement feedback
            
        Returns:
            Update result
        """
        skill_path = self.skills_dir / f"{name}.py"
        
        if not skill_path.exists():
            return {"success": False, "error": f"Skill {name} not found"}
        
        current_code = skill_path.read_text(encoding="utf-8")
        
        if self._llm:
            prompt = f"""Improve this Python skill based on feedback.

Current Code:
```python
{current_code}
```

Feedback: {feedback}

Provide the improved code:"""

            response = await self._llm.ainvoke(prompt)
            new_code = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up
            if "```python" in new_code:
                new_code = new_code.split("```python")[1].split("```")[0]
            elif "```" in new_code:
                new_code = new_code.split("```")[1].split("```")[0]
            
            # Backup old version
            backup_path = self.skills_dir / f"{name}.py.bak"
            backup_path.write_text(current_code, encoding="utf-8")
            
            # Save new version
            skill_path.write_text(new_code.strip(), encoding="utf-8")
            
            self._log_improvement("skill_improved", {
                "name": name,
                "feedback": feedback,
            })
            
            return {"success": True, "message": f"Skill {name} improved"}
        
        return {"success": False, "error": "LLM not available"}
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List all available skills."""
        skills = []
        for meta_file in self.skills_dir.glob("*.json"):
            try:
                metadata = json.loads(meta_file.read_text(encoding="utf-8"))
                skills.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load skill metadata: {e}")
        return skills
    
    # =========================================================================
    # CONFIGURATION MODIFICATION
    # =========================================================================
    
    async def modify_config(
        self,
        key: str,
        value: Any,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Modify a configuration value.
        
        Args:
            key: Configuration key (dot notation)
            value: New value
            reason: Reason for change
            
        Returns:
            Result of modification
        """
        # Check if allowed
        if not self.config.get("autonomy.can_modify_config", False):
            return {"success": False, "error": "Config modification not allowed"}
        
        # Get old value
        old_value = self.config.get(key)
        
        # Set new value
        self.config.set(key, value)
        
        # Save config
        await self.config.save()
        
        # Log change
        self._log_improvement("config_modified", {
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "reason": reason,
        })
        
        logger.info(f"Config modified: {key} = {value} (was: {old_value})")
        return {"success": True, "key": key, "old": old_value, "new": value}
    
    async def change_model(self, model_type: str, new_model: str) -> Dict[str, Any]:
        """
        Change the AI model being used.
        
        Args:
            model_type: Type (primary, fallback, coding)
            new_model: New model name
            
        Returns:
            Result
        """
        if not self.config.get("autonomy.can_change_models", False):
            return {"success": False, "error": "Model changes not allowed"}
        
        key = f"models.{model_type}"
        return await self.modify_config(key, new_model, f"Switching {model_type} model to {new_model}")
    
    # =========================================================================
    # ISSUE DETECTION & FIXING
    # =========================================================================
    
    async def diagnose_issue(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Diagnose an issue and suggest fixes.
        
        Args:
            error: The exception that occurred
            context: Additional context
            
        Returns:
            Diagnosis and suggested fixes
        """
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context,
        }
        
        if self._llm:
            prompt = f"""Diagnose this error and suggest fixes.

Error Type: {error_info['type']}
Error Message: {error_info['message']}
Context: {error_info['context']}

Traceback:
{error_info['traceback']}

Provide:
1. Root cause analysis
2. Suggested fixes (code changes if applicable)
3. Prevention strategies

Format as JSON:
{{"root_cause": "...", "fixes": ["..."], "prevention": ["..."]}}"""

            response = await self._llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            try:
                # Try to parse JSON from response
                if "{" in content:
                    json_str = content[content.index("{"):content.rindex("}")+1]
                    diagnosis = json.loads(json_str)
                else:
                    diagnosis = {"raw_analysis": content}
            except:
                diagnosis = {"raw_analysis": content}
            
            return {"success": True, "error_info": error_info, "diagnosis": diagnosis}
        
        return {"success": False, "error_info": error_info, "diagnosis": None}
    
    async def auto_fix_issue(
        self,
        error: Exception,
        file_path: Optional[str] = None,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Automatically fix an issue.
        
        Args:
            error: The exception
            file_path: File where error occurred
            context: Additional context
            
        Returns:
            Fix result
        """
        if not self.config.get("autonomy.can_fix_issues", False):
            return {"success": False, "error": "Auto-fix not allowed"}
        
        # First diagnose
        diagnosis = await self.diagnose_issue(error, context)
        
        if not diagnosis.get("success"):
            return {"success": False, "error": "Diagnosis failed"}
        
        # If we have a file path and the diagnosis suggests code fixes
        if file_path and Path(file_path).exists():
            current_code = Path(file_path).read_text(encoding="utf-8")
            
            if self._llm:
                prompt = f"""Fix this code based on the error diagnosis.

Current Code:
```
{current_code}
```

Error: {str(error)}
Diagnosis: {json.dumps(diagnosis.get('diagnosis', {}))}

Provide the fixed code only, no explanations:"""

                response = await self._llm.ainvoke(prompt)
                fixed_code = response.content if hasattr(response, 'content') else str(response)
                
                # Clean up
                if "```" in fixed_code:
                    parts = fixed_code.split("```")
                    if len(parts) >= 2:
                        fixed_code = parts[1]
                        if fixed_code.startswith("python"):
                            fixed_code = fixed_code[6:]
                
                # Backup and save
                backup_path = Path(file_path + ".bak")
                backup_path.write_text(current_code, encoding="utf-8")
                Path(file_path).write_text(fixed_code.strip(), encoding="utf-8")
                
                self._log_improvement("auto_fix_applied", {
                    "error": str(error),
                    "file": file_path,
                })
                
                return {
                    "success": True,
                    "diagnosis": diagnosis,
                    "file": file_path,
                    "backup": str(backup_path),
                }
        
        return {
            "success": True,
            "diagnosis": diagnosis,
            "auto_fix": False,
            "message": "Diagnosis complete but no auto-fix available",
        }
    
    # =========================================================================
    # LEARNING & IMPROVEMENT
    # =========================================================================
    
    def _log_improvement(self, action: str, details: Dict[str, Any]) -> None:
        """Log an improvement action."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        }
        self.improvement_log.append(entry)
        
        # Persist to file
        log_path = Path("data/improvement_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze agent performance and suggest improvements."""
        # Load improvement log
        log_path = Path("data/improvement_log.jsonl")
        if not log_path.exists():
            return {"success": False, "message": "No improvement data available"}
        
        entries = []
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except:
                    pass
        
        # Analyze patterns
        analysis = {
            "total_improvements": len(entries),
            "skills_created": len([e for e in entries if e["action"] == "skill_created"]),
            "skills_improved": len([e for e in entries if e["action"] == "skill_improved"]),
            "config_changes": len([e for e in entries if e["action"] == "config_modified"]),
            "auto_fixes": len([e for e in entries if e["action"] == "auto_fix_applied"]),
        }
        
        return {"success": True, "analysis": analysis}
    
    async def suggest_improvements(self) -> List[str]:
        """Suggest potential improvements based on usage patterns."""
        suggestions = []
        
        # Check for missing skills
        common_skills = ["web_scraper", "email_handler", "calendar_manager", "note_taker"]
        existing = [s["name"] for s in self.list_skills()]
        
        for skill in common_skills:
            if skill not in existing:
                suggestions.append(f"Consider creating '{skill}' skill for common tasks")
        
        # Check configuration
        if not self.config.get("copilot.enabled"):
            suggestions.append("Enable GitHub Copilot integration for better coding assistance")
        
        if self.config.get("models.primary") == "gpt-4o-mini":
            suggestions.append("Consider upgrading to gpt-4o for better reasoning")
        
        return suggestions
