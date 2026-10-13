"""Semantic colors with deterministic fallback identities and contrast screening."""

from hashlib import sha256
from types import MappingProxyType

import numpy as np
from matplotlib.colors import to_hex, to_rgb

from .config import Config, get_config

TOKENS = MappingProxyType(
    {
        "primary": "#4678AE",
        "secondary": "#D8765A",
        "tertiary": "#4AA292",
        "quaternary": "#C06C84",
        "purple": "#927BB5",
        "ochre": "#B88A44",
        "neutral": "#7A8594",
        "text": "#252525",
        "grid": "#E5E5E5",
        "background": "#FFFFFF",
        "uncertainty": "#4678AE",
    }
)
PALETTES = MappingProxyType(
    {
        # Restrained categorical hues for white paper backgrounds; not a venue standard.
        "default": tuple(
            TOKENS[name]
            for name in (
                "primary",
                "secondary",
                "tertiary",
                "purple",
                "ochre",
                "quaternary",
                "neutral",
            )
        ),
        "binary": (TOKENS["primary"], TOKENS["secondary"]),
        "comparison": (TOKENS["primary"], TOKENS["neutral"]),
        "grayscale": ("#252525", "#777777", "#AAAAAA", "#444444", "#999999"),
    }
)
COLORMAPS = MappingProxyType({"sequential": "cividis", "diverging": "RdBu_r", "grayscale": "Greys"})
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES = ("-", "--", "-.", ":")


def resolve_color(value: str) -> str:
    return to_hex(TOKENS.get(value, value))


def identity_index(label: str) -> int:
    known = {"Ours": 0, "Baseline": 1, "Ablation": 2, "Observed": 0}
    return known.get(label, int.from_bytes(sha256(label.encode("utf-8")).digest()[:8], "big"))


def color_for(label: str, config: Config | None = None) -> str:
    config = config or get_config()
    palette = PALETTES["grayscale" if config.theme == "grayscale" else config.palette]
    assigned = dict(config.colors).get(str(label))
    if assigned is not None and config.theme != "grayscale" and config.palette != "grayscale":
        return resolve_color(assigned)
    return to_hex(palette[identity_index(str(label)) % len(palette)])


def contrast_ratio(foreground: str, background: str = "white") -> float:
    def luminance(color: str) -> float:
        c = np.asarray(to_rgb(color))
        linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(linear @ np.array([0.2126, 0.7152, 0.0722]))

    a, b = sorted((luminance(foreground), luminance(background)))
    return (b + 0.05) / (a + 0.05)
