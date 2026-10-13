"""Validation and explicitly requested summaries, independent of rendering."""

import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd


def vector(data: Any, value: Any, name: str, *, numeric: bool = False) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, str):
        if data is None:
            raise ValueError(f"{name} names a column but no data was supplied")
        try:
            value = data[value]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Column {value!r} is missing") from exc
    result = np.asanyarray(value)
    if np.ma.isMaskedArray(result):
        result = np.ma.asarray(result, dtype=float).filled(np.nan)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if numeric:
        try:
            result = result.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if np.isinf(result).any():
            raise ValueError(f"{name} cannot contain infinity")
        if np.isnan(result).all():
            raise ValueError(f"{name} has no observed values")
    return result


def xy_frame(data: Any, x: Any, y: Any, hue: Any = None) -> pd.DataFrame:
    xx = vector(data, x, "x")
    yy = vector(data, y, "y", numeric=True)
    hh = vector(data, hue, "hue") if hue is not None else np.full(len(xx), "Observed")
    if not len(xx) == len(yy) == len(hh):
        raise ValueError("x, y and hue must have equal lengths")
    if pd.isna(xx).any() or pd.isna(hh).any():
        raise ValueError("x and hue cannot be missing; y may contain gaps")
    if np.issubdtype(xx.dtype, np.number) and not np.isfinite(xx).all():
        raise ValueError("x must be finite")
    return pd.DataFrame({"x": xx, "y": yy, "hue": hh})


def groups(frame: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    for key, group in frame.groupby("hue", sort=False, observed=True):
        yield str(key), group


def bounds(
    data: Any, lower: Any, upper: Any, y: np.ndarray, uncertainty: str | None
) -> tuple[np.ndarray, np.ndarray]:
    if not uncertainty:
        raise ValueError("Explicit uncertainty definition is required for bounds")
    lo = vector(data, lower, "lower", numeric=True)
    hi = vector(data, upper, "upper", numeric=True)
    if len(lo) != len(y) or len(hi) != len(y):
        raise ValueError("bounds must have the same length as y")
    if ((lo > y) | (hi < y) | (lo > hi)).any():
        raise ValueError("bounds must enclose the estimate: lower <= y <= upper")
    if not np.array_equal(np.isnan(lo), np.isnan(hi)):
        raise ValueError("both bounds must be missing together")
    return lo, hi


def summarize(
    data: Any, *, x: str, y: str, hue: str | None = None, unit: str | None = None, error: str = "sd"
) -> pd.DataFrame:
    """Mean and SD/SE/pointwise Student-t 95% CI across independent units.

    At most one observation per unit/x/hue is allowed. Missing y values are
    excluded from that point's summary only; n and missing counts are returned.
    ci95 assumes independent, approximately normally distributed unit values.
    This is not a simultaneous band or an analysis of within-unit dependence.
    """
    if unit is None:
        raise ValueError("unit must identify independent replicates")
    if error not in {"sd", "se", "ci95", "range"}:
        raise ValueError("error must be sd, se, ci95 or range")
    frame = pd.DataFrame(data).copy()
    keys = [x] + ([hue] if hue else [])
    required = [*keys, y, unit]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError("Summary requires nonempty data and all named columns")
    if frame[[*keys, unit]].isna().any().any():
        raise ValueError("Grouping columns and unit cannot be missing")
    vector(frame, y, "y", numeric=True)
    if frame.duplicated([*keys, unit]).any():
        raise ValueError("duplicate unit within x/hue; aggregate at the independent unit first")
    records = []
    small = False
    for key, group in frame.groupby(keys, sort=False, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group[y].dropna().to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        lo = hi = np.nan
        if n >= 2:
            sd = float(values.std(ddof=1))
            half = sd if error == "sd" else sd / np.sqrt(n)
            if error == "ci95":
                try:
                    from scipy.stats import t
                except ImportError as exc:
                    raise ImportError("ci95 requires rendeer[stats]") from exc
                half *= t.ppf(0.975, n - 1)
            lo, hi = (
                (float(values.min()), float(values.max()))
                if error == "range"
                else (mean - half, mean + half)
            )
        else:
            small = True
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "estimate": mean,
                "lower": lo,
                "upper": hi,
                "n": n,
                "missing": len(group) - n,
            }
        )
    if small:
        warnings.warn(
            "Interval unavailable for groups with fewer than two observed units",
            UserWarning,
            stacklevel=2,
        )
    result = pd.DataFrame(records)
    result.attrs.update(
        uncertainty=error,
        unit=unit,
        estimator="mean",
        missing_policy="exclude per point and report counts",
    )
    return result
