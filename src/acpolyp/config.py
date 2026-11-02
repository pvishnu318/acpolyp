"""YAML configuration with dotted CLI overrides (Phase XIII, step 52)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS


def _coerce(value: str) -> Any:
    """Parse an override value with YAML rules so '1e-4', 'true', '[1,2]' behave as expected."""
    v = yaml.safe_load(value)
    if isinstance(v, str):  # YAML 1.1 reads '1e-3' as a string; accept plain float notation too
        try:
            return float(v)
        except ValueError:
            pass
    return v


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load configs/default.yaml, merge an optional experiment YAML, then apply key=value overrides."""
    cfg = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
    if path is not None:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = CONFIGS / p
        cfg = deep_update(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b.c=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return cfg
