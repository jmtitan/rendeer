"""Immutable, context-local configuration; importing never changes rcParams."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Config:
    theme: str = "paper"
    palette: str = "default"
    font: str = "DejaVu Sans"
    strict: bool = False
    dpi: int = 300
    usetex: bool = False
    colors: tuple[tuple[str, str], ...] = (
        ("Ours", "primary"),
        ("Baseline", "secondary"),
        ("Ablation", "tertiary"),
    )
    rc: tuple[tuple[str, Any], ...] = ()


_DEFAULT = Config()
_CONFIG: ContextVar[Config] = ContextVar("rendeer_config", default=_DEFAULT)


def get_config() -> Config:
    return _CONFIG.get()


def configure(**changes: Any) -> Config:
    """Set defaults for future figures in this execution context, not Matplotlib."""
    from .colors import PALETTES, resolve_color
    from .theme import THEMES

    if "colors" in changes and isinstance(changes["colors"], Mapping):
        changes["colors"] = tuple(changes["colors"].items())
    if "rc" in changes and isinstance(changes["rc"], Mapping):
        changes["rc"] = tuple(
            (k, tuple(v) if isinstance(v, list) else v) for k, v in changes["rc"].items()
        )
    result = replace(get_config(), **changes)
    if result.theme not in THEMES or result.palette not in PALETTES:
        raise ValueError("Unknown theme or categorical palette")
    if not isinstance(result.dpi, int) or result.dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    if not result.font.strip():
        raise ValueError("font must not be empty")
    for _, value in result.colors:
        resolve_color(value)
    _CONFIG.set(result)
    return result


@contextmanager
def context(**changes: Any) -> Iterator[Config]:
    """Temporarily configure rendeer, restoring defaults even on exceptions."""
    token = _CONFIG.set(get_config())
    try:
        yield configure(**changes)
    finally:
        _CONFIG.reset(token)


def set_colors(mapping: Mapping[str, str], *, replace_existing: bool = False) -> Config:
    """Register category -> token/hex mappings; save them for cross-project reuse."""
    merged = {} if replace_existing else dict(get_config().colors)
    merged.update({str(k): v for k, v in mapping.items()})
    return configure(colors=merged)
