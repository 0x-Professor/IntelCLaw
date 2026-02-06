"""
PageIndex Watcher - auto-ingest PDFs dropped into a folder.

Uses watchdog to watch a folder and enqueue files for indexing.
The actual indexing implementation is provided via an async callback.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional

from loguru import logger

try:
    from watchdog.events import FileSystemEventHandler  # type: ignore
    from watchdog.observers import Observer  # type: ignore

    WATCHDOG_AVAILABLE = True
except Exception:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # type: ignore
    Observer = object  # type: ignore


AsyncPathHandler = Callable[[Path], Awaitable[None]]


def _normalize_exts(exts: Iterable[str]) -> set[str]:
    normalized = set()
    for e in exts:
        e = (e or "").strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        normalized.add(e)
    return normalized


@dataclass
class PageIndexWatcherConfig:
    ingest_folder: Path
    extensions: set[str]
    debounce_seconds: float = 1.0
    stable_checks: int = 3
    stable_interval_s: float = 0.5


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: "PageIndexWatcher"):
        self._watcher = watcher

    def on_created(self, event):  # type: ignore[override]
        self._watcher._on_fs_event(getattr(event, "src_path", None))

    def on_modified(self, event):  # type: ignore[override]
        self._watcher._on_fs_event(getattr(event, "src_path", None))

    def on_moved(self, event):  # type: ignore[override]
        self._watcher._on_fs_event(getattr(event, "dest_path", None))


class PageIndexWatcher:
    def __init__(self, config: PageIndexWatcherConfig, on_file: AsyncPathHandler):
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog is not available; install watchdog to enable folder watching")

        self._config = config
        self._on_file = on_file
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._queue: "asyncio.Queue[Path]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._observer: Optional[Observer] = None

        self._last_enqueued: dict[str, float] = {}
        self._running = False

    async def start(self, scan_on_startup: bool = True) -> None:
        self._config.ingest_folder.mkdir(parents=True, exist_ok=True)
        self._loop = asyncio.get_running_loop()
        self._running = True

        # Start worker
        self._worker_task = asyncio.create_task(self._worker(), name="pageindex_watcher_worker")

        # Start watchdog observer in background thread
        handler = _Handler(self)
        observer = Observer()
        observer.schedule(handler, str(self._config.ingest_folder), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer

        logger.info(f"PageIndex watcher started: {self._config.ingest_folder}")

        if scan_on_startup:
            await self._scan_existing()

    async def stop(self) -> None:
        self._running = False

        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as e:
                logger.debug(f"PageIndex watcher observer stop failed: {e}")
            self._observer = None

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        logger.info("PageIndex watcher stopped")

    async def _scan_existing(self) -> None:
        for path in self._config.ingest_folder.rglob("*"):
            if path.is_file() and self._is_allowed(path):
                await self._queue.put(path)

    def _is_allowed(self, path: Path) -> bool:
        return path.suffix.lower() in self._config.extensions

    def _on_fs_event(self, src_path: Optional[str]) -> None:
        if not self._running or not src_path:
            return
        path = Path(src_path)
        if not path.exists() or not path.is_file() or not self._is_allowed(path):
            return

        now = time.time()
        key = str(path.resolve())
        last = self._last_enqueued.get(key, 0.0)
        if now - last < self._config.debounce_seconds:
            return
        self._last_enqueued[key] = now

        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, path)

    async def _wait_for_stable(self, path: Path) -> bool:
        last_size = -1
        last_mtime = -1.0
        stable = 0

        for _ in range(max(self._config.stable_checks, 1) * 4):
            try:
                stat = path.stat()
            except FileNotFoundError:
                return False

            size = stat.st_size
            mtime = stat.st_mtime
            if size == last_size and mtime == last_mtime:
                stable += 1
                if stable >= self._config.stable_checks:
                    return True
            else:
                stable = 0
                last_size, last_mtime = size, mtime

            await asyncio.sleep(self._config.stable_interval_s)

        return stable >= self._config.stable_checks

    async def _worker(self) -> None:
        while True:
            path = await self._queue.get()
            try:
                if not path.exists():
                    continue

                ok = await self._wait_for_stable(path)
                if not ok:
                    continue

                await self._on_file(path)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"PageIndex watcher failed to process {path}: {e}")
            finally:
                self._queue.task_done()

