"""
IntelCLaw - Production-grade Windows AI Agent System

A fully autonomous AI agent with transparent overlay UI, screen monitoring,
and autonomous task execution capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"
__author__ = "IntelCLaw Team"

if TYPE_CHECKING:
    from intelclaw.core.app import IntelCLawApp as IntelCLawApp

__all__ = ["IntelCLawApp", "__version__"]


def __getattr__(name: str):
    # Lazy import to avoid heavy side effects when importing submodules like `intelclaw.memory.*`.
    if name == "IntelCLawApp":
        from intelclaw.core.app import IntelCLawApp  # local import

        return IntelCLawApp
    raise AttributeError(name)
