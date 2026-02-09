from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Optional

from intelclaw.contacts.models import ContactEntry


def normalize_phone(value: str) -> str:
    """
    Normalize a phone number for WhatsApp usage.

    - If the value looks like a JID (contains '@'), return the user part.
    - Otherwise strip all non-digits (e.g., '+', spaces, hyphens).
    """
    s = str(value or "").strip()
    if "@" in s:
        return s.split("@", 1)[0]
    return re.sub(r"\D+", "", s)


def _parse_bool(value: str) -> bool:
    v = str(value or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


class ContactsStore:
    """
    Markdown-backed contacts store.

    Canonical format:

    | name | phone | whatsapp_jid | inbound_allowed | notes |
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> List[ContactEntry]:
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8", errors="replace")
        return self._parse_markdown(text)

    def save(self, entries: Iterable[ContactEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized = list(entries)
        normalized.sort(key=lambda e: (e.name or "").lower())
        self.path.write_text(self._render_markdown(normalized), encoding="utf-8")

    def lookup(self, query: str, *, inbound_only: bool = False) -> List[ContactEntry]:
        q = str(query or "").strip().lower()
        if not q:
            return []
        out: List[ContactEntry] = []
        for e in self.load():
            if inbound_only and not e.inbound_allowed:
                continue
            hay = " ".join([e.name, e.phone, e.whatsapp_jid or "", e.notes]).lower()
            if q in hay:
                out.append(e)
        return out

    def upsert(
        self,
        *,
        name: str,
        phone: str,
        whatsapp_jid: Optional[str] = None,
        inbound_allowed: Optional[bool] = None,
        notes: Optional[str] = None,
    ) -> ContactEntry:
        n = str(name or "").strip()
        if not n:
            raise ValueError("name is required")
        phone_norm = normalize_phone(phone)
        if not phone_norm:
            raise ValueError("phone is required")

        existing = self.load()
        updated: List[ContactEntry] = []
        found: Optional[ContactEntry] = None
        for e in existing:
            if (e.name or "").strip().lower() == n.lower():
                found = e
                continue
            updated.append(e)

        if found is None:
            entry = ContactEntry(
                name=n,
                phone=phone_norm,
                whatsapp_jid=whatsapp_jid.strip() if isinstance(whatsapp_jid, str) and whatsapp_jid.strip() else None,
                inbound_allowed=bool(inbound_allowed) if inbound_allowed is not None else False,
                notes=str(notes or "").strip(),
            )
        else:
            entry = found
            entry = replace(entry, name=n, phone=phone_norm)
            if whatsapp_jid is not None:
                entry = replace(
                    entry,
                    whatsapp_jid=whatsapp_jid.strip() if isinstance(whatsapp_jid, str) and whatsapp_jid.strip() else None,
                )
            if inbound_allowed is not None:
                entry = replace(entry, inbound_allowed=bool(inbound_allowed))
            if notes is not None:
                entry = replace(entry, notes=str(notes or "").strip())

        updated.append(entry)
        self.save(updated)
        return entry

    @staticmethod
    def _parse_markdown(text: str) -> List[ContactEntry]:
        lines = [ln.rstrip("\n") for ln in (text or "").splitlines()]
        # Find the first markdown table header
        header_idx = -1
        for i, ln in enumerate(lines):
            if ln.strip().startswith("|") and "name" in ln.lower() and "phone" in ln.lower():
                header_idx = i
                break
        if header_idx < 0:
            return []

        header_cols = [c.strip().lower() for c in lines[header_idx].strip().strip("|").split("|")]
        col_map = {name: idx for idx, name in enumerate(header_cols) if name}

        def _get(row: List[str], key: str) -> str:
            idx = col_map.get(key)
            if idx is None:
                return ""
            if idx >= len(row):
                return ""
            return row[idx].strip()

        out: List[ContactEntry] = []
        for ln in lines[header_idx + 1 :]:
            s = ln.strip()
            if not s.startswith("|"):
                # End table on first non-row line
                break
            cells = [c.strip() for c in s.strip().strip("|").split("|")]
            # Skip separator row
            if all(set(c) <= {"-"} for c in cells if c):
                continue

            name = _get(cells, "name")
            phone = _get(cells, "phone")
            if not name or not phone:
                continue
            jid = _get(cells, "whatsapp_jid") or None
            inbound = _parse_bool(_get(cells, "inbound_allowed"))
            notes = _get(cells, "notes")
            out.append(
                ContactEntry(
                    name=name,
                    phone=normalize_phone(phone),
                    whatsapp_jid=jid.strip() if isinstance(jid, str) and jid.strip() else None,
                    inbound_allowed=inbound,
                    notes=str(notes or "").strip(),
                )
            )
        return out

    @staticmethod
    def _render_markdown(entries: List[ContactEntry]) -> str:
        header = [
            "# Contacts",
            "",
            "| name | phone | whatsapp_jid | inbound_allowed | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
        rows: List[str] = []
        for e in entries:
            inbound = "yes" if e.inbound_allowed else "no"
            jid = e.whatsapp_jid or ""
            notes = (e.notes or "").replace("\n", " ").strip()
            rows.append(f"| {e.name} | {e.phone} | {jid} | {inbound} | {notes} |")
        return "\n".join(header + rows + [""])

