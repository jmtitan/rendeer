"""PNG by default; explicit PDF only. Fixed dimensions and no implicit overwrite."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

from .theme import rc_params
from .validation import QualityError

if TYPE_CHECKING:
    from .figure import Figure


def save(
    figure: Figure,
    path: str | Path,
    *,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    overwrite: bool = False,
    manifest: bool = True,
    transparent: bool = False,
) -> list[Path]:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlink")
    suffix = path.suffix.lstrip(".").lower()
    chosen = list(formats) if formats is not None else [suffix or "png"]
    if not chosen or any(fmt not in {"png", "pdf"} for fmt in chosen):
        raise ValueError("Only PNG and PDF exports are supported")
    if len(set(chosen)) != len(chosen):
        raise ValueError("Duplicate export formats")
    if suffix and suffix not in {"png", "pdf"}:
        raise ValueError("Only PNG and PDF filename extensions are supported")
    if formats is not None and suffix and chosen != [suffix]:
        raise ValueError("Use an extensionless path for multi-format export")
    dpi = figure.config.dpi if dpi is None else dpi
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    base = path.with_suffix("") if suffix else path
    destinations = [base.with_suffix(f".{fmt}").absolute() for fmt in chosen]
    manifest_path = base.with_suffix(".json").absolute()
    all_paths = destinations + ([manifest_path] if manifest else [])
    for destination in all_paths:
        if destination.is_symlink() or (destination.exists() and not overwrite):
            raise FileExistsError(f"Refusing to overwrite {destination}")
    report = figure.validate()
    if not report.passed:
        if figure.config.strict:
            raise QualityError(report)
        warnings.warn(str(QualityError(report)), UserWarning, stacklevel=2)
    for destination in all_paths:
        destination.parent.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    records = []
    try:
        with mpl.rc_context({**rc_params(figure.config), "savefig.bbox": None}):
            for destination, fmt in zip(destinations, chosen, strict=True):
                fd, name = tempfile.mkstemp(
                    prefix=".rendeer-", suffix=f".{fmt}", dir=destination.parent
                )
                os.close(fd)
                temporary = Path(name)
                staged.append((temporary, destination))
                metadata: dict[str, Any] = (
                    {"Software": "rendeer"}
                    if fmt == "png"
                    else {"Creator": "rendeer", "CreationDate": None, "ModDate": None}
                )
                figure.mpl.savefig(
                    temporary,
                    format=fmt,
                    dpi=dpi,
                    bbox_inches=None,
                    transparent=transparent,
                    metadata=metadata,
                )
                records.append(
                    {
                        "path": str(destination),
                        "format": fmt,
                        "dpi": dpi,
                        "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
                    }
                )
        if manifest:
            fd, name = tempfile.mkstemp(
                prefix=".rendeer-", suffix=".json", dir=manifest_path.parent
            )
            os.close(fd)
            temporary = Path(name)
            staged.append((temporary, manifest_path))
            record = {
                **figure.provenance(),
                "exports": records,
                "quality": report.to_dict(),
                "transparent": transparent,
            }
            temporary.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        for temporary, destination in staged:
            if overwrite:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)
                temporary.unlink()
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return destinations
