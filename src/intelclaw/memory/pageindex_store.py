"""
PageIndex Local Store.

Persists a lightweight registry and cached PageIndex trees locally (under data/),
so query-time retrieval can be served from disk without remote calls.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


def _now_iso() -> str:
    return datetime.now().isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(str(tmp_path), str(path))


@dataclass
class PageIndexStore:
    base_dir: Path = Path("data/pageindex")

    @property
    def registry_path(self) -> Path:
        return self.base_dir / "registry.json"

    @property
    def trees_dir(self) -> Path:
        return self.base_dir / "trees"

    def load_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load registry keyed by local_path.

        Registry schema (per file):
        - local_path, file_sha256, doc_id, name, status, created_at, updated_at, page_num, description
        """
        if not self.registry_path.exists():
            return {}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to read PageIndex registry: {e}")
        return {}

    def save_registry(self, registry: Dict[str, Dict[str, Any]]) -> None:
        _atomic_write_text(self.registry_path, json.dumps(registry, indent=2, ensure_ascii=False))

    def upsert_by_path(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        local_path = str(entry.get("local_path") or "").strip()
        if not local_path:
            raise ValueError("PageIndexStore upsert requires local_path")

        registry = self.load_registry()
        existing = dict(registry.get(local_path, {}))

        merged = {**existing, **entry}
        merged.setdefault("created_at", existing.get("created_at") or _now_iso())
        merged["updated_at"] = _now_iso()

        registry[local_path] = merged
        self.save_registry(registry)
        return merged

    def get_by_path(self, local_path: str) -> Optional[Dict[str, Any]]:
        return self.load_registry().get(str(local_path))

    def list_docs(self) -> List[Dict[str, Any]]:
        registry = self.load_registry()
        docs = list(registry.values())
        docs.sort(key=lambda d: str(d.get("updated_at") or ""), reverse=True)
        return docs

    def save_tree(self, doc_id: str, tree_json: Dict[str, Any]) -> Path:
        self.trees_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.trees_dir / f"{doc_id}.json"
        _atomic_write_text(out_path, json.dumps(tree_json, indent=2, ensure_ascii=False))
        return out_path

    def load_tree(self, doc_id: str) -> Optional[Dict[str, Any]]:
        path = self.trees_dir / f"{doc_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load cached tree for doc_id={doc_id}: {e}")
            return None

    def delete_cached(self, doc_id: str) -> bool:
        """
        Delete cached tree and registry entries referencing doc_id.
        Returns True if anything was removed.
        """
        changed = False

        # Delete tree file
        tree_path = self.trees_dir / f"{doc_id}.json"
        if tree_path.exists():
            try:
                tree_path.unlink()
                changed = True
            except Exception as e:
                logger.warning(f"Failed to delete tree cache {tree_path}: {e}")

        # Remove registry entries
        registry = self.load_registry()
        to_delete = [p for p, entry in registry.items() if str(entry.get("doc_id")) == str(doc_id)]
        for p in to_delete:
            registry.pop(p, None)
            changed = True
        if to_delete:
            self.save_registry(registry)

        return changed

