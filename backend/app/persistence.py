"""World persistence: JSON save/load into the data/ directory."""
from __future__ import annotations

import json
import re
from typing import List

from . import config
from .world_state import WorldState

_SAFE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _path(name: str):
    if not _SAFE.match(name):
        raise ValueError(f"invalid world name: {name!r}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return config.DATA_DIR / f"{name}.json"


def save_world(state: WorldState, name: str = "default") -> str:
    path = _path(name)
    path.write_text(json.dumps(state.to_dict(), separators=(",", ":")),
                    encoding="utf-8")
    return str(path)


def load_world(name: str = "default") -> WorldState:
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(f"saved world not found: {name}")
    return WorldState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_worlds() -> List[str]:
    if not config.DATA_DIR.exists():
        return []
    return sorted(p.stem for p in config.DATA_DIR.glob("*.json"))
