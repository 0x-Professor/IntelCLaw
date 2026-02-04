"""Perception module - Screen capture, OCR, and activity monitoring."""

from intelclaw.perception.manager import PerceptionManager
from intelclaw.perception.screen_capture import ScreenCapture
from intelclaw.perception.ocr import OCRProcessor
from intelclaw.perception.ui_automation import UIAutomation
from intelclaw.perception.activity_monitor import ActivityMonitor
from intelclaw.perception.context_builder import ContextBuilder

__all__ = [
    "PerceptionManager",
    "ScreenCapture",
    "OCRProcessor",
    "UIAutomation",
    "ActivityMonitor",
    "ContextBuilder",
]
