"""
Skills subsystem - MCP-backed skill packs and routing.

Skills are discovered from:
- Built-in skills: ./skills/<id>/skill.yaml
- User skills:     ./data/skills/<id>/skill.yaml
"""

from __future__ import annotations

from intelclaw.skills.manager import SkillManager
from intelclaw.skills.router import SkillRouter

__all__ = ["SkillManager", "SkillRouter"]

