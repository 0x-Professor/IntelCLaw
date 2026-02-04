"""
IntelCLaw CLI - Command Line Interface

Provides OpenClaw-style commands for onboarding, auth, and configuration.
"""

from .onboard import run_onboard
from .auth import AuthManager

__all__ = ["run_onboard", "AuthManager"]
