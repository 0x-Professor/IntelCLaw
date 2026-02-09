"""
TeamRegistry - construct and manage team members (core + skill specialists).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from intelclaw.teams.members import TeamMember

if TYPE_CHECKING:
    from intelclaw.integrations.llm_provider import LLMProvider
    from intelclaw.tools.registry import ToolRegistry
    from intelclaw.skills.manager import SkillManager
    from intelclaw.memory.manager import MemoryManager


class TeamRegistry:
    def __init__(
        self,
        *,
        llm_provider: "LLMProvider",
        tools: "ToolRegistry",
        skills: "SkillManager",
        memory: Optional["MemoryManager"] = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._tools = tools
        self._skills = skills
        self._memory = memory

        self._members: Dict[str, TeamMember] = {}
        self._skill_deps: Dict[str, List[str]] = {}

    def get_member(self, member_id: str) -> Optional[TeamMember]:
        return self._members.get(str(member_id or "").strip())

    def list_member_ids(self) -> List[str]:
        return sorted(self._members.keys())

    async def refresh(self) -> None:
        # Core members
        self._members["general"] = TeamMember(
            member_id="general",
            display_name="General",
            agent_instructions="General purpose agent for coordinating tasks.",
            llm_provider=self._llm_provider,
            tools=self._tools,
            memory=self._memory,
        )
        self._members["research"] = TeamMember(
            member_id="research",
            display_name="Research",
            agent_instructions="Focus on web research and synthesis. Prefer tavily_search and web_scrape.",
            llm_provider=self._llm_provider,
            tools=self._tools,
            memory=self._memory,
        )
        self._members["coding"] = TeamMember(
            member_id="coding",
            display_name="Coding",
            agent_instructions="Focus on coding and file changes. Prefer file_* and code_execute tools.",
            llm_provider=self._llm_provider,
            tools=self._tools,
            memory=self._memory,
        )
        self._members["system"] = TeamMember(
            member_id="system",
            display_name="System",
            agent_instructions="Focus on system automation. Prefer powershell, shell_command, and windows tools.",
            llm_provider=self._llm_provider,
            tools=self._tools,
            memory=self._memory,
        )

        # Skill specialists
        skills_list = await self._skills.list_skills()
        self._skill_deps = {}
        for s in skills_list:
            sid = str(s.get("id") or "").strip()
            if not sid:
                continue
            self._skill_deps[sid] = [str(x) for x in (s.get("depends_on") or []) if str(x).strip()]
            instructions = await self._skills.get_agent_instructions(sid)
            self._members[f"skill:{sid}"] = TeamMember(
                member_id=f"skill:{sid}",
                display_name=str(s.get("name") or sid),
                agent_instructions=instructions or f"Specialist agent for skill '{sid}'.",
                llm_provider=self._llm_provider,
                tools=self._tools,
                memory=self._memory,
            )

    def _deps_closure(self, skill_id: str) -> List[str]:
        sid = str(skill_id or "").strip()
        if not sid:
            return []
        out: List[str] = []
        seen: set[str] = set()

        def visit(cur: str) -> None:
            if cur in seen:
                return
            seen.add(cur)
            for d in self._skill_deps.get(cur, []):
                if d:
                    visit(d)
            out.append(cur)

        visit(sid)
        return out  # deps first, then sid

    def allowlist_for_member(self, member_id: str, *, skill_id: Optional[str] = None) -> Set[str]:
        """
        Compute the tool allowlist for a member.

        - Core members get a curated subset of built-in tools.
        - Skill members get all built-in tools + MCP tools for the skill and its dependency chain.
        """
        mid = str(member_id or "").strip()

        # Base allowlists for core members.
        core: Dict[str, Set[str]] = {
            "research": {"tavily_search", "web_scrape", "file_read", "file_search", "list_directory", "get_cwd"},
            "coding": {
                "file_read",
                "file_write",
                "file_delete",
                "file_copy",
                "file_move",
                "file_search",
                "list_directory",
                "get_cwd",
                "code_execute",
                "pip_install",
            },
            "system": {
                "shell_command",
                "powershell",
                "system_info",
                "screenshot",
                "clipboard",
                "launch_app",
                "file_read",
                "file_write",
                "file_search",
                "list_directory",
                "get_cwd",
            },
            "general": set(),
        }

        if mid in core and core[mid]:
            allow = set(core[mid])
        else:
            # Default: all non-MCP tools
            allow = {d.name for d in self._tools.list_tools() if not d.name.startswith("mcp_")}

        if mid.startswith("skill:"):
            sid = skill_id or mid.split("skill:", 1)[1]
            for dep in self._deps_closure(sid):
                allow |= set(self._tools.get_mcp_tool_names_for_skill(dep))

        return allow
