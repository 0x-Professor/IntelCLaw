import sys
from pathlib import Path


def pytest_configure():
    # Ensure `src/` is on sys.path so `import intelclaw` works without an editable install.
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if src.exists():
        p = str(src)
        if p not in sys.path:
            sys.path.insert(0, p)

