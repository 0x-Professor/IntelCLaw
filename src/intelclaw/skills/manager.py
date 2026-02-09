"""
SkillManager - discover, install, enable/disable skills.

Skill packs are MCP-server–backed capabilities. Each skill has a manifest
(`skill.yaml` or `skill.json`) and optional agent instructions (`AGENT.md`).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from intelclaw.skills.models import MCPServerConfig, SkillManifest

try:
    import yaml  # type: ignore

    YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    YAML_AVAILABLE = False


@dataclass(frozen=True)
class SkillRuntimeState:
    enabled: bool
    healthy: bool
    last_error: Optional[str] = None


class SkillManager:
    """
    Manage skills on disk and their enabled/disabled state.

    Discovery order:
    1) built-in: <repo>/skills/<id>/skill.yaml
    2) user:     <repo>/data/skills/<id>/skill.yaml (overrides built-in by id)
    """

    def __init__(
        self,
        config: Any,
        event_bus: Any,
        *,
        builtin_dir: Optional[Path] = None,
        user_dir: Optional[Path] = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus

        repo_root = Path(__file__).resolve().parent.parent.parent.parent

        cfg_builtin = None
        cfg_user = None
        cfg_state = None
        try:
            if config is not None and hasattr(config, "get"):
                cfg_builtin = config.get("skills.builtin_dir")
                cfg_user = config.get("skills.user_dir")
                cfg_state = config.get("skills.state_path")
        except Exception:
            pass

        self._builtin_dir = builtin_dir or Path(cfg_builtin or (repo_root / "skills"))
        self._user_dir = user_dir or Path(cfg_user or (repo_root / "data" / "skills"))
        self._state_path = state_path or Path(cfg_state or (repo_root / "data" / "skills_state.json"))

        self._lock = asyncio.Lock()
        self._manifests: Dict[str, SkillManifest] = {}
        self._enabled: Dict[str, bool] = {}
        self._runtime: Dict[str, SkillRuntimeState] = {}
        self._loaded = False

    @property
    def state_path(self) -> Path:
        return self._state_path

    async def initialize(self) -> None:
        await self.reload()

    async def reload(self) -> None:
        async with self._lock:
            manifests = self._discover_manifests()
            enabled = self._load_state_enabled()

            changed = False
            for sid, manifest in manifests.items():
                if sid not in enabled:
                    enabled[sid] = bool(getattr(manifest, "enabled_by_default", False))
                    changed = True

            self._manifests = manifests
            self._enabled = enabled
            self._loaded = True

            if changed or not self._state_path.exists():
                self._persist_state_enabled(enabled)

        await self._emit_changed()

    def _discover_manifests(self) -> Dict[str, SkillManifest]:
        builtin = self._load_manifests_from_dir(self._builtin_dir, source_kind="builtin")
        user = self._load_manifests_from_dir(self._user_dir, source_kind="user")

        manifests = dict(builtin)
        manifests.update(user)  # user overrides by id

        # Optional legacy config-based MCP servers
        legacy = self._build_legacy_manifest()
        if legacy and legacy.id not in manifests:
            manifests[legacy.id] = legacy

        return manifests

    def _load_manifests_from_dir(
        self, root: Path, *, source_kind: str
    ) -> Dict[str, SkillManifest]:
        manifests: Dict[str, SkillManifest] = {}
        if not root.exists():
            return manifests

        candidates = []
        for suffix in ("yaml", "yml", "json"):
            candidates.extend(root.glob(f"*/skill.{suffix}"))

        for manifest_path in candidates:
            try:
                data = self._read_manifest_file(manifest_path)
                manifest = SkillManifest.model_validate(data)
                manifest.source_dir = manifest_path.parent
                manifest.source_file = manifest_path
                manifest.source_kind = source_kind  # type: ignore[assignment]
                manifests[manifest.id] = manifest
            except Exception as e:
                logger.warning(f"Failed to load skill manifest {manifest_path}: {e}")

        return manifests

    def _read_manifest_file(self, path: Path) -> Dict[str, Any]:
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(content)
        else:
            if not YAML_AVAILABLE:
                raise RuntimeError("PyYAML not available to read skill.yaml")
            data = yaml.safe_load(content)  # type: ignore[union-attr]
        if not isinstance(data, dict):
            raise ValueError("skill manifest must be a mapping/object")
        return data

    def _build_legacy_manifest(self) -> Optional[SkillManifest]:
        try:
            servers = self._config.get("mcp.servers", []) if self._config else []
        except Exception:
            servers = []

        if not isinstance(servers, list) or not servers:
            return None

        mcp_servers: List[MCPServerConfig] = []
        for i, item in enumerate(servers):
            if not isinstance(item, dict):
                continue
            try:
                sid = str(item.get("id") or item.get("name") or f"server_{i}")
                transport = str(item.get("transport") or "stdio")
                if transport != "stdio":
                    continue
                command = str(item.get("command") or "").strip()
                if not command:
                    continue
                args = item.get("args") or []
                if not isinstance(args, list):
                    args = [str(args)]
                env = item.get("env") or {}
                if not isinstance(env, dict):
                    env = {}
                tool_ns = str(item.get("tool_namespace") or sid)
                mcp_servers.append(
                    MCPServerConfig(
                        id=sid,
                        transport="stdio",
                        command=command,
                        args=[str(a) for a in args],
                        env={str(k): str(v) for k, v in env.items()},
                        cwd=item.get("cwd"),
                        tool_namespace=tool_ns,
                        tool_allowlist=list(item.get("tool_allowlist") or []),
                        tool_denylist=list(item.get("tool_denylist") or []),
                    )
                )
            except Exception as e:
                logger.debug(f"Skipping legacy MCP server config due to parse error: {e}")

        if not mcp_servers:
            return None

        return SkillManifest(
            id="legacy_mcp",
            name="Legacy MCP",
            version="0.1.0",
            description="MCP servers configured via config.yaml:mcp.servers (legacy compatibility)",
            icon="🧩",
            enabled_by_default=True,
            depends_on=[],
            triggers={},
            agent={"instructions_file": "AGENT.md"},
            mcp_servers=mcp_servers,
            source_kind="synthetic",
        )

    def _load_state_enabled(self) -> Dict[str, bool]:
        path = self._state_path
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                enabled = data.get("enabled") if isinstance(data, dict) else None
                if isinstance(enabled, dict):
                    return {str(k): bool(v) for k, v in enabled.items()}
        except Exception as e:
            logger.warning(f"Failed to read skills state file {path}: {e}")
        return {}

    def _persist_state_enabled(self, enabled: Dict[str, bool]) -> None:
        path = self._state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"enabled": dict(sorted(enabled.items(), key=lambda kv: kv[0]))}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to persist skills state file {path}: {e}")

    async def _emit_changed(self) -> None:
        if not self._event_bus:
            return
        try:
            await self._event_bus.emit(
                "skills.changed",
                {"skills": self.list_skills_sync()},
                source="skills",
            )
        except Exception as e:
            logger.debug(f"Failed to emit skills.changed: {e}")

    def list_skills_sync(self) -> List[Dict[str, Any]]:
        """Sync helper for emitting payloads; use list_skills() for async callers."""
        skills = []
        for sid, manifest in sorted(self._manifests.items(), key=lambda kv: kv[0]):
            skills.append(
                {
                    "id": sid,
                    "name": manifest.name,
                    "version": manifest.version,
                    "description": manifest.description,
                    "icon": manifest.icon,
                    "enabled": bool(self._enabled.get(sid, False)),
                    "depends_on": list(manifest.depends_on or []),
                    "source_kind": manifest.source_kind,
                }
            )
        return skills

    async def list_skills(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return self.list_skills_sync()

    async def get_manifest(self, skill_id: str) -> Optional[SkillManifest]:
        sid = str(skill_id or "").strip()
        if not sid:
            return None
        async with self._lock:
            return self._manifests.get(sid)

    async def is_enabled(self, skill_id: str) -> bool:
        sid = str(skill_id or "").strip()
        if not sid:
            return False
        async with self._lock:
            return bool(self._enabled.get(sid, False))

    async def enabled_skill_manifests(self) -> List[SkillManifest]:
        async with self._lock:
            return [m for sid, m in self._manifests.items() if self._enabled.get(sid)]

    async def enable(self, skill_id: str) -> Dict[str, Any]:
        sid = str(skill_id or "").strip()
        if not sid:
            return {"success": False, "error": "missing skill id"}

        async with self._lock:
            if sid not in self._manifests:
                return {"success": False, "error": f"skill not found: {sid}"}

            changed = False
            to_enable = self._dependency_chain(sid)
            for dep in to_enable:
                if not self._enabled.get(dep, False):
                    self._enabled[dep] = True
                    changed = True

            if changed:
                self._persist_state_enabled(self._enabled)

        if changed:
            await self._emit_event("skills.skill_enabled", {"skill_id": sid})
            await self._emit_changed()
        return {"success": True, "skill_id": sid, "enabled": True}

    async def disable(self, skill_id: str) -> Dict[str, Any]:
        sid = str(skill_id or "").strip()
        if not sid:
            return {"success": False, "error": "missing skill id"}

        async with self._lock:
            if sid not in self._manifests:
                return {"success": False, "error": f"skill not found: {sid}"}

            # Block disabling if any enabled skill depends on this (transitively).
            dependents = self._enabled_dependents_of(sid)
            if dependents:
                return {
                    "success": False,
                    "error": f"Cannot disable '{sid}' because enabled skill(s) depend on it: {', '.join(sorted(dependents))}",
                    "blocked_by": sorted(dependents),
                }

            if not self._enabled.get(sid, False):
                return {"success": True, "skill_id": sid, "enabled": False}

            self._enabled[sid] = False
            self._persist_state_enabled(self._enabled)

        await self._emit_event("skills.skill_disabled", {"skill_id": sid})
        await self._emit_changed()
        return {"success": True, "skill_id": sid, "enabled": False}

    def _dependency_chain(self, skill_id: str) -> List[str]:
        """Return [deps..., skill_id] in dependency order (deps first)."""
        out: List[str] = []
        seen: set[str] = set()

        def visit(sid: str) -> None:
            if sid in seen:
                return
            seen.add(sid)
            manifest = self._manifests.get(sid)
            if manifest:
                for dep in manifest.depends_on or []:
                    visit(dep)
            out.append(sid)

        visit(skill_id)
        return out

    def _enabled_dependents_of(self, skill_id: str) -> set[str]:
        """Return enabled skills that (transitively) depend on skill_id."""
        target = str(skill_id or "").strip()
        if not target:
            return set()

        enabled_skills = {sid for sid, v in self._enabled.items() if v}
        dependents: set[str] = set()

        # Precompute adjacency: skill -> deps
        deps_map: Dict[str, List[str]] = {}
        for sid, manifest in self._manifests.items():
            deps_map[sid] = list(manifest.depends_on or [])

        def depends_on(sid: str, dep: str) -> bool:
            stack = list(deps_map.get(sid, []))
            seen: set[str] = set()
            while stack:
                cur = stack.pop()
                if cur == dep:
                    return True
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(deps_map.get(cur, []))
            return False

        for sid in enabled_skills:
            if sid != target and depends_on(sid, target):
                dependents.add(sid)

        return dependents

    async def install_from_yaml(
        self,
        yaml_text: str,
        *,
        agent_md_text: Optional[str] = None,
        enable: bool = False,
    ) -> Dict[str, Any]:
        if not yaml_text or not str(yaml_text).strip():
            return {"success": False, "error": "manifest_yaml is required"}
        if not YAML_AVAILABLE:
            return {"success": False, "error": "PyYAML not available"}

        try:
            data = yaml.safe_load(yaml_text)  # type: ignore[union-attr]
        except Exception as e:
            return {"success": False, "error": f"Invalid YAML: {e}"}

        if not isinstance(data, dict):
            return {"success": False, "error": "manifest_yaml must parse to a mapping/object"}

        try:
            manifest = SkillManifest.model_validate(data)
        except Exception as e:
            return {"success": False, "error": f"Invalid manifest: {e}"}

        sid = manifest.id
        skill_dir = self._user_dir / sid
        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "skill.yaml").write_text(str(yaml_text).strip() + "\n", encoding="utf-8")
            if agent_md_text and str(agent_md_text).strip():
                (skill_dir / "AGENT.md").write_text(str(agent_md_text).strip() + "\n", encoding="utf-8")
        except Exception as e:
            return {"success": False, "error": f"Failed to write skill files: {e}"}

        await self.reload()

        if enable:
            await self.enable(sid)

        return {"success": True, "skill_id": sid, "path": str(skill_dir)}

    async def _emit_event(self, name: str, data: Dict[str, Any]) -> None:
        if not self._event_bus:
            return
        try:
            await self._event_bus.emit(name, data, source="skills")
        except Exception as e:
            logger.debug(f"Failed to emit event {name}: {e}")

    async def get_agent_instructions(self, skill_id: str) -> str:
        """Return AGENT.md content for the skill (best-effort)."""
        sid = str(skill_id or "").strip()
        if not sid:
            return ""

        async with self._lock:
            manifest = self._manifests.get(sid)

        if not manifest or not manifest.source_dir:
            return ""

        rel = getattr(manifest.agent, "instructions_file", "AGENT.md") or "AGENT.md"
        path = (manifest.source_dir / rel).resolve()
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception:
            return ""
        return ""

    async def get_enabled_mcp_server_specs(self) -> List[Tuple[str, MCPServerConfig, Path]]:
        """
        Returns a list of (skill_id, server_config, skill_dir) for enabled skills.
        """
        async with self._lock:
            out: List[Tuple[str, MCPServerConfig, Path]] = []
            for sid, manifest in self._manifests.items():
                if not self._enabled.get(sid, False):
                    continue
                if not manifest.source_dir:
                    continue
                for server in manifest.mcp_servers or []:
                    out.append((sid, server, manifest.source_dir))
            return out

