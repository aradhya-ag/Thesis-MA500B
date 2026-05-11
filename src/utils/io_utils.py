from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project/


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return self.data

    def get_dotted(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_dotted(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def resolve_path(self, dotted_key: str) -> Path:
        raw = self.get_dotted(dotted_key)
        if raw is None:
            raise KeyError(f"Path config key not found: {dotted_key}")
        return self._resolve(raw)

    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        base = self.config_path.parent.parent if self.config_path else PROJECT_ROOT
        return (base / p).resolve()


def load_config(path: str | os.PathLike, overrides: Optional[Dict[str, Any]] = None) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data=data, config_path=path)
    if overrides:
        for k, v in overrides.items():
            cfg.set_dotted(k, v)
    return cfg


def parse_cli_overrides(items: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def save_json(obj: Any, path: str | os.PathLike, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
