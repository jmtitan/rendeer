"""Matrix plots distinguish missing values from zero and record normalization."""

from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex

from ..colors import contrast_ratio
from ..figure import PLOTS, Panel, plot


@plot
def heatmap(
    panel: Panel,
    data: Any,
    *,
    xticklabels: Any = None,
    yticklabels: Any = None,
    xlabel: str = "Column",
    ylabel: str = "Row",
    cbar_label: str = "Value",
    cmap: Any = "cividis",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    annot: bool = False,
    fmt: str = ".2g",
    colorbar: bool = True,
    title: str | None = None,
) -> None:
    """Cell-centered image, masked missing cells, optional contrast-aware numeric labels."""
    if isinstance(data, pd.DataFrame):
        xticklabels = list(data.columns) if xticklabels is None else xticklabels
        yticklabels = list(data.index) if yticklabels is None else yticklabels
    values = np.ma.asarray(data, dtype=float).filled(np.nan)
    if values.ndim != 2 or values.size == 0 or np.isnan(values).all() or np.isinf(values).any():
        raise ValueError("heatmap requires a nonempty numeric matrix with observed finite values")
    rows, cols = values.shape
    xticklabels = list(range(cols)) if xticklabels is None else list(xticklabels)
    yticklabels = list(range(rows)) if yticklabels is None else list(yticklabels)
    if len(xticklabels) != cols or len(yticklabels) != rows:
        raise ValueError("ticklabels must match the matrix dimensions")
    vmin = float(np.nanmin(values)) if vmin is None else float(vmin)
    vmax = float(np.nanmax(values)) if vmax is None else float(vmax)
    if not np.isfinite([vmin, vmax]).all() or vmin > vmax:
        raise ValueError("Invalid color limits")
    if vmin == vmax:
        delta = max(abs(vmin) * 0.01, 0.5)
        vmin, vmax = vmin - delta, vmax + delta
    norm: Normalize = Normalize(vmin, vmax)
    if center is not None:
        if not vmin < center < vmax:
            raise ValueError("diverging center must be strictly inside color limits")
        norm = TwoSlopeNorm(center, vmin=vmin, vmax=vmax)
    colors = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(bad="#D9D9D9")
    image = panel.ax.imshow(
        np.ma.masked_invalid(values), norm=norm, cmap=colors, interpolation="nearest", aspect="auto"
    )
    panel.mappable = image
    panel.ax.set_xticks(range(cols), [str(x) for x in xticklabels])
    panel.ax.set_yticks(range(rows), [str(y) for y in yticklabels])
    panel.ax.tick_params(length=0)
    for spine in panel.ax.spines.values():
        spine.set_visible(False)
    if annot:
        for row in range(rows):
            for col in range(cols):
                value = values[row, col]
                if not np.isfinite(value):
                    continue
                background = to_hex(colors(norm(value)))
                foreground = (
                    "white"
                    if contrast_ratio("white", background) > contrast_ratio("#252525", background)
                    else "#252525"
                )
                text = panel.ax.text(
                    col, row, format(value, fmt), ha="center", va="center", color=foreground
                )
                text.set_gid("rendeer-cell")
    if colorbar:
        panel.owner.mpl.colorbar(image, ax=panel.ax, label=cbar_label, shrink=0.9, pad=0.03)
    panel.finish(
        "heatmap",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap=colors.name,
        missing=int(np.isnan(values).sum()),
        saturated=int(((values < vmin) | (values > vmax)).sum()),
        cbar_label=cbar_label,
    )


@plot
def correlation(
    panel: Panel,
    data: Any,
    *,
    method: str = "pearson",
    missing: str = "raise",
    annot: bool = True,
    **kwargs: Any,
) -> None:
    """Correlation matrix; missingness is rejected unless pairwise or complete is explicit."""
    frame = pd.DataFrame(data)
    if (
        frame.shape[0] < 2
        or frame.shape[1] < 2
        or not all(pd.api.types.is_numeric_dtype(v) for v in frame.dtypes)
    ):
        raise ValueError("correlation requires at least two numeric columns and observations")
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Unknown correlation method")
    if missing not in {"raise", "pairwise", "complete"}:
        raise ValueError("missing must be raise, pairwise or complete")
    if frame.isna().any().any() and missing == "raise":
        raise ValueError("correlation has missing data; specify a missing policy")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("correlation does not accept infinite values")
    if missing == "complete":
        frame = frame.dropna()
    matrix = frame.corr(method=method, min_periods=2)
    PLOTS["heatmap"](
        panel,
        matrix,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xlabel="Variable",
        ylabel="Variable",
        cbar_label=f"{method.capitalize()} r",
        annot=annot,
        **kwargs,
    )
    panel.owner.records[-1].update(
        kind="correlation",
        missing_policy=missing,
        method=method,
        pairwise_n=(frame.notna().astype(int).T @ frame.notna().astype(int)).to_numpy().tolist(),
    )


@plot
def confusion(
    panel: Panel,
    data: Any,
    *,
    labels: Any = None,
    normalize: str | None = None,
    annot: bool = True,
    **kwargs: Any,
) -> None:
    """Observed count matrix, optionally row/column/global normalized; empty margins stay missing."""
    counts = np.asarray(data, dtype=float)
    if (
        counts.ndim != 2
        or counts.shape[0] != counts.shape[1]
        or counts.size == 0
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("confusion requires a square matrix of nonnegative integer counts")
    if normalize not in {None, "true", "pred", "all"}:
        raise ValueError("normalize must be None, true, pred or all")
    values = counts.copy()
    if normalize:
        denominator = (
            counts.sum(axis=1, keepdims=True)
            if normalize == "true"
            else (counts.sum(axis=0, keepdims=True) if normalize == "pred" else counts.sum())
        )
        values = np.divide(
            counts, denominator, out=np.full_like(counts, np.nan), where=denominator != 0
        )
    PLOTS["heatmap"](
        panel,
        values,
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        cbar_label="Fraction" if normalize else "Count",
        annot=annot,
        fmt=".2f" if normalize else ".0f",
        vmin=0,
        vmax=1 if normalize else None,
        **kwargs,
    )
    panel.owner.records[-1].update(kind="confusion", normalization=normalize, n=int(counts.sum()))
