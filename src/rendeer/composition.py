"""Convenient small multiples composed from the same axes-level primitives."""

import math
from inspect import signature
from typing import Any

import pandas as pd

from .figure import PLOTS, Figure, figure


def facet(
    data: Any,
    *,
    col: str,
    kind: str = "line",
    wrap: int = 3,
    sharex: bool = True,
    sharey: bool = True,
    **kwargs: Any,
) -> Figure:
    """One named panel per first-appearance category; caption must map panel letters."""
    frame = pd.DataFrame(data)
    if col not in frame or frame.empty or frame[col].isna().any():
        raise ValueError("Facet column must exist and have nonmissing categories")
    if kind not in PLOTS or not isinstance(wrap, int) or wrap < 1:
        raise ValueError("Unknown plot kind or invalid wrap")
    categories = list(pd.unique(frame[col]))
    names = [str(c) for c in categories]
    if len(set(names)) != len(names) or "." in names:
        raise ValueError("Facet names must be distinct strings and cannot be '.'")
    columns = min(wrap, len(names))
    rows = math.ceil(len(names) / columns)
    names_padded = names + ["."] * (columns * rows - len(names))
    layout = [names_padded[i : i + columns] for i in range(0, len(names_padded), columns)]
    result = figure(layout=layout, width="double", sharex=sharex, sharey=sharey, panel_labels=True)
    if "hue" in signature(PLOTS[kind]).parameters:
        kwargs.setdefault("hue", col)
    for category, name in zip(categories, names, strict=True):
        result[name]._draw(PLOTS[kind], frame[frame[col] == category], **kwargs)
    result.legend()
    result.records.append(
        {
            "panel": "*",
            "kind": "facet",
            "column": col,
            "panel_categories": names,
            "shared_x": sharex,
            "shared_y": sharey,
        }
    )
    return result


def pairplot(data: Any, *, columns: list[str] | None = None, hue: str | None = None) -> Figure:
    """Pairwise scatter grid and diagonal histograms; at most four explicit variables."""
    frame = pd.DataFrame(data)
    columns = (
        list(frame.columns.drop(hue) if hue is not None else frame.columns)
        if columns is None
        else columns
    )
    if not 2 <= len(columns) <= 4 or len(set(columns)) != len(columns):
        raise ValueError("Select two to four distinct numeric columns for readable pairs")
    if not set(columns).issubset(frame.columns) or any(
        not pd.api.types.is_numeric_dtype(frame[c]) for c in columns
    ):
        raise ValueError("Pairwise variables must be numeric")
    n = len(columns)
    result = figure(
        layout=[[f"{i},{j}" for j in range(n)] for i in range(n)], width="double", height=2.1 * n
    )
    for i, y in enumerate(columns):
        for j, x in enumerate(columns):
            panel = result[f"{i},{j}"]
            if i == j:
                panel.hist(frame, x=x, hue=hue)
            else:
                panel.scatter(frame, x=x, y=y, hue=hue)
    if hue is not None:
        result.legend()
    return result
