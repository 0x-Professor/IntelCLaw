"""
Secret redaction utilities.

Used to prevent persisting credentials/tokens in local memory stores.
This is heuristic-based and intentionally conservative.
"""

from __future__ import annotations

import re


_STRONG_PATTERNS: list[re.Pattern[str]] = [
    # OpenAI
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    # Anthropic
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}\b"),
    # GitHub tokens
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # Tavily
    re.compile(r"\btvly-[A-Za-z0-9_-]{20,}\b"),
    # Slack
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    # PEM blocks
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]

_CONTEXT_WORDS = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|secret|password|passcode|bearer)\b"
)

_ASSIGNMENT = re.compile(
    r"(?im)^(?P<prefix>\s*(?:export\s+)?[A-Z0-9_]*(?:KEY|TOKEN|SECRET)\s*=\s*)(?P<value>[^\s#]+)"
)

_BEARER = re.compile(r"(?i)\bbearer\s+(?P<tok>[A-Za-z0-9._=-]{12,})")

_HEX_LONG = re.compile(r"\b[a-f0-9]{32,64}\b", re.IGNORECASE)
_TOKEN_LONG = re.compile(r"\b[A-Za-z0-9_\-]{20,}\b")


def contains_secret(text: str) -> bool:
    if not text:
        return False

    for pat in _STRONG_PATTERNS:
        if pat.search(text):
            return True

    if _ASSIGNMENT.search(text):
        return True

    if _BEARER.search(text):
        return True

    # Contextual heuristic: only treat long random-looking tokens as secrets when
    # the surrounding text suggests keys/tokens.
    if _CONTEXT_WORDS.search(text):
        if _HEX_LONG.search(text) or _TOKEN_LONG.search(text):
            return True

    return False


def redact_secrets(text: str) -> str:
    if not text:
        return text

    # Redact KEY=VALUE style lines
    text = _ASSIGNMENT.sub(r"\g<prefix>[REDACTED]", text)

    # Redact explicit Bearer tokens
    text = _BEARER.sub("Bearer [REDACTED]", text)

    # Redact strong patterns unconditionally
    for pat in _STRONG_PATTERNS:
        text = pat.sub("[REDACTED]", text)

    # If the text looks like it's discussing secrets, redact long token candidates too.
    if _CONTEXT_WORDS.search(text):
        text = _HEX_LONG.sub("[REDACTED]", text)
        text = _TOKEN_LONG.sub("[REDACTED]", text)

    return text

