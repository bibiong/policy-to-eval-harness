"""Minimal .env loader.

API keys belong in a gitignored file, not in shell history and not in a config
file that gets committed. This reads `.env` from the repo root into the
environment if present.

Deliberately dependency-free (no python-dotenv): it handles the three forms that
actually appear in a key file — `KEY=value`, `export KEY=value`, and quoted
values — and ignores everything else rather than pretending to be a shell parser.

Existing environment variables always win, so an explicitly exported key
overrides the file rather than being silently replaced by it.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load KEY=value pairs from `path`. Returns the names of keys that were set."""
    path = Path(path)
    if not path.exists():
        return []

    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        if key not in os.environ:  # a real env var beats the file
            os.environ[key] = value
            loaded.append(key)
    return loaded
