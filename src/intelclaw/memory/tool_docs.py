"""
Tool docs index.

Parses `persona/TOOLS.md` into structured tool documentation chunks and
supports lightweight keyword search for "tool hints".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")
_FENCE_RE = re.compile(r"^```")
_SIG_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(.*\)\s*->\s*.+$")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


@dataclass(frozen=True)
class ToolDoc:
    tool_name: str
    signature: str
    section: str
    description: str


class ToolDocsIndex:
    def __init__(self, docs: List[ToolDoc]):
        self._docs = docs

    @classmethod
    def parse_markdown(cls, text: str) -> "ToolDocsIndex":
        heading_stack: List[str] = []
        in_code = False
        code_lines: List[str] = []

        pending_docs: List[ToolDoc] = []
        post_buffer: List[str] = []

        docs: List[ToolDoc] = []

        def current_section() -> str:
            return " > ".join(heading_stack)

        def flush_pending() -> None:
            nonlocal pending_docs, post_buffer, docs
            if not pending_docs:
                post_buffer = []
                return
            desc = "\n".join(post_buffer).strip()
            for d in pending_docs:
                docs.append(
                    ToolDoc(
                        tool_name=d.tool_name,
                        signature=d.signature,
                        section=d.section,
                        description=desc,
                    )
                )
            pending_docs = []
            post_buffer = []

        lines = (text or "").splitlines()
        for line in lines:
            if not in_code:
                m = _HEADING_RE.match(line)
                if m:
                    flush_pending()
                    level = len(m.group(1))
                    title = m.group("title").strip()
                    # Normalize stack to this level
                    heading_stack = heading_stack[: max(level - 1, 0)]
                    heading_stack.append(title)
                    continue

            if _FENCE_RE.match(line):
                if in_code:
                    # end code block
                    in_code = False
                    flush_pending()
                    extracted: List[ToolDoc] = []
                    for cl in code_lines:
                        cl = cl.strip()
                        sig = cl
                        ms = _SIG_RE.match(sig)
                        if not ms:
                            continue
                        name = ms.group("name")
                        extracted.append(
                            ToolDoc(tool_name=name, signature=sig, section=current_section(), description="")
                        )
                    pending_docs = extracted
                    code_lines = []
                else:
                    # start code block
                    in_code = True
                    code_lines = []
                continue

            if in_code:
                code_lines.append(line)
                continue

            if pending_docs:
                post_buffer.append(line)

        flush_pending()
        return cls(docs)

    def list_tool_names(self) -> List[str]:
        return sorted({d.tool_name for d in self._docs})

    def search(self, query: str, top_k: int = 3) -> List[ToolDoc]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return []

        scored: List[tuple[float, ToolDoc]] = []
        for d in self._docs:
            hay = f"{d.tool_name}\n{d.signature}\n{d.section}\n{d.description}"
            h_tokens = _tokens(hay)
            overlap = len(q_tokens & h_tokens) / max(len(q_tokens), 1)
            bonus = 0.2 if d.tool_name.lower() in (query or "").lower() else 0.0
            score = overlap + bonus
            if score <= 0:
                continue
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]


def merge_indexes(indexes: Iterable[ToolDocsIndex]) -> ToolDocsIndex:
    docs: List[ToolDoc] = []
    for idx in indexes:
        docs.extend(idx._docs)  # intentionally internal to merge quickly
    return ToolDocsIndex(docs)

