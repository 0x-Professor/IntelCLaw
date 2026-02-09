"""
SkillRouter - choose which skill should handle a user request.

Selection order:
1) Explicit directive: "@skill <id>" or "/skill <id>"
2) Keyword/regex triggers across enabled skills
3) Optional LLM tie-breaker when ambiguous (best-effort)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from intelclaw.skills.manager import SkillManager
from intelclaw.skills.models import SkillManifest


@dataclass(frozen=True)
class SkillRouteDecision:
    skill_id: Optional[str]
    confidence: float
    reason: str
    cleaned_message: str


class SkillRouter:
    def __init__(self, skill_manager: SkillManager, llm_provider: Any = None) -> None:
        self._skills = skill_manager
        self._llm_provider = llm_provider

    @staticmethod
    def _extract_directive(message: str) -> Tuple[Optional[str], str]:
        text = str(message or "")
        m = re.search(r"(?:^|\s)(?:@skill|/skill)\s+([a-zA-Z0-9_-]+)\b", text)
        if not m:
            return None, text
        sid = m.group(1).strip()
        cleaned = (text[: m.start()] + " " + text[m.end() :]).strip()
        return sid, cleaned

    async def route(self, message: str) -> SkillRouteDecision:
        requested, cleaned = self._extract_directive(message)
        if requested:
            return SkillRouteDecision(
                skill_id=requested,
                confidence=1.0,
                reason="explicit directive",
                cleaned_message=cleaned,
            )

        # Enabled skills only for auto-detect
        enabled_manifests = await self._skills.enabled_skill_manifests()
        if not enabled_manifests:
            return SkillRouteDecision(
                skill_id=None, confidence=0.0, reason="no enabled skills", cleaned_message=cleaned
            )

        scored: List[Tuple[str, float, str]] = []
        msg_lower = cleaned.lower()
        for manifest in enabled_manifests:
            score = 0.0
            reasons = []

            keywords = [k for k in (manifest.triggers.keywords or []) if isinstance(k, str) and k.strip()]
            if keywords:
                hits = sum(1 for k in keywords if k.lower() in msg_lower)
                if hits:
                    score += min(hits / max(len(keywords), 1), 1.0) * 0.7
                    reasons.append(f"keyword_hits={hits}")

            regexes = [r for r in (manifest.triggers.regex or []) if isinstance(r, str) and r.strip()]
            if regexes:
                rhits = 0
                for pat in regexes[:10]:
                    try:
                        if re.search(pat, cleaned, flags=re.IGNORECASE):
                            rhits += 1
                    except re.error:
                        continue
                if rhits:
                    score += min(rhits / max(len(regexes), 1), 1.0) * 0.3
                    reasons.append(f"regex_hits={rhits}")

            if score > 0:
                scored.append((manifest.id, score, ", ".join(reasons) or "matched"))

        if not scored:
            return SkillRouteDecision(
                skill_id=None, confidence=0.0, reason="no trigger matches", cleaned_message=cleaned
            )

        scored.sort(key=lambda t: t[1], reverse=True)
        best_id, best_score, best_reason = scored[0]

        # Clear winner?
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        if best_score >= 0.6 and (best_score - second_score) >= 0.2:
            return SkillRouteDecision(
                skill_id=best_id, confidence=best_score, reason=best_reason, cleaned_message=cleaned
            )

        # Ambiguous: best-effort LLM tie-breaker
        skill_id = await self._llm_pick(cleaned, enabled_manifests)
        if skill_id:
            return SkillRouteDecision(
                skill_id=skill_id,
                confidence=0.55,
                reason="llm_tiebreak",
                cleaned_message=cleaned,
            )

        return SkillRouteDecision(
            skill_id=best_id,
            confidence=best_score,
            reason=f"ambiguous_fallback: {best_reason}",
            cleaned_message=cleaned,
        )

    async def _llm_pick(self, message: str, skills: List[SkillManifest]) -> Optional[str]:
        llm = getattr(self._llm_provider, "llm", None) if self._llm_provider else None
        if llm is None:
            return None

        items = [{"id": s.id, "name": s.name, "description": s.description} for s in skills]
        prompt = {
            "task": "Pick the single best skill_id for the user request, or null if none apply.",
            "skills": items,
            "request": message,
            "output_json_schema": {"skill_id": "string|null"},
        }

        try:
            resp = await llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You are an intent router for skill packs. "
                            "Return ONLY valid JSON matching the schema."
                        )
                    ),
                    HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
                ]
            )
            text = resp.content if hasattr(resp, "content") else str(resp)
            # best-effort JSON extraction
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                sid = data.get("skill_id")
                if sid is None:
                    return None
                sid = str(sid).strip()
                return sid or None
        except Exception as e:
            logger.debug(f"SkillRouter LLM pick failed: {e}")
            return None

        return None

