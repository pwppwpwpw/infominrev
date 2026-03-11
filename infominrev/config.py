import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], path: str) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get(config: Dict[str, Any], *keys: str, default=None):
    cur: Any = config
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
