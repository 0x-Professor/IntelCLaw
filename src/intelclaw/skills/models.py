"""
Skills models (Pydantic).

These models define the on-disk skill manifest schema (skill.yaml / skill.json).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SkillTriggers(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    regex: List[str] = Field(default_factory=list)


class SkillAgentConfig(BaseModel):
    instructions_file: str = "AGENT.md"


class MCPServerConfig(BaseModel):
    id: str
    transport: Literal["stdio"] = "stdio"
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    tool_namespace: str = "default"
    tool_allowlist: List[str] = Field(default_factory=list)
    tool_denylist: List[str] = Field(default_factory=list)

    @field_validator("tool_namespace")
    @classmethod
    def _validate_tool_namespace(cls, v: str) -> str:
        v = (v or "").strip()
        return v or "default"


class SkillManifest(BaseModel):
    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    icon: str = ""

    enabled_by_default: bool = False
    depends_on: List[str] = Field(default_factory=list)

    triggers: SkillTriggers = Field(default_factory=SkillTriggers)
    agent: SkillAgentConfig = Field(default_factory=SkillAgentConfig)
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)

    # Runtime metadata (not part of manifest file)
    source_dir: Optional[Path] = None
    source_file: Optional[Path] = None
    source_kind: Optional[Literal["builtin", "user", "synthetic"]] = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("skill id cannot be empty")
        return v

