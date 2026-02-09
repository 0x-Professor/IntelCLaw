from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ContactEntry:
    name: str
    phone: str
    whatsapp_jid: Optional[str] = None
    inbound_allowed: bool = False
    notes: str = ""

