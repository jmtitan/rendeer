# Themes, colors, export and quality checks

`rd.configure(**changes)` changes defaults for **future rendeer figures** in
the current execution context. It never calls a persistent Matplotlib style
update. Existing figures keep their configuration snapshot.

```python
rd.configure(theme="paper", palette="default", font="DejaVu Sans", strict=True, dpi=300)
with rd.context(theme="nature", font="Arial", rc={"lines.linewidth": 1.1}):
    fig = rd.line(x=[0, 1], y=[2, 3], xlabel="Time (s)", ylabel="Speed (m/s)")
```

`get_config()` returns a frozen `Config`. `context()` restores even after an
exception. `usetex=True` opts into an external TeX installation; mathtext,
Greek letters and Unicode work without it when the configured font has the
glyphs. CJK requires an installed font with the appropriate coverage (for
example Noto Sans CJK); no proprietary or other font files are bundled.

| Theme | Single / double width (inches) | Body / minimum text (pt) |
| --- | --- | --- |
| paper | 3.5 / 7.2 | 9 / 7 |
| nature | 89/25.4 / 183/25.4 | 7 / 5 |
| ml | 3.25 / 6.75 | 9 / 7 |
| ieee | 3.5 / 7.16 | 8 / 6 |
| slides | 6.4 / 12.8 | 16 / 12 |
| report | 4 / 8 | 10 / 8 |
| grayscale | 3.5 / 7.2 | 9 / 7 |

`width`/`height` override physical size on each figure. `rc` overrides fonts,
line sizes, spacing and other Matplotlib parameters. Nature's panel letters
use 8 pt. The minimum is a validation threshold, not a command to shrink
labels. Presets and font fallbacks do not certify publisher compliance.

## Semantic colors

The default categorical palette uses restrained blue, coral, teal, lavender,
ochre, rose and slate on white. This is a rendeer design choice for ML paper
figures, not an official NeurIPS/ICLR palette.

| Token | Color | Default role |
| --- | --- | --- |
| `primary` | `#4678AE` | Blue; Ours |
| `secondary` | `#D8765A` | Coral; Baseline |
| `tertiary` | `#4AA292` | Teal; Ablation |
| `purple` | `#927BB5` | Lavender; additional category |
| `ochre` | `#B88A44` | Ochre; additional category |
| `quaternary` | `#C06C84` | Rose; additional category |
| `neutral` | `#7A8594` | Slate; explicit de-emphasis |

`TOKENS` also provides text, grid, background and uncertainty. `PALETTES` exposes
`default`, `binary`, `comparison`, and `grayscale`. `COLORMAPS` gives sequential
`cividis`, diverging `RdBu_r`, and `Greys`. The uncertainty fill uses the
series' own semantic color at alpha 0.16, rather than another category.

```python
rd.set_colors({"Ours": "primary", "Baseline": "secondary", "Ablation": "tertiary"})
blue = rd.color_for("Ours")
ratio = rd.contrast_ratio(blue, "white")
```

All seven opaque categorical colors exceed the 3:1 white-background contrast
screen. Thin antialiased marks and translucent fills still need visual review;
this does not certify color-vision accessibility. Markers and line styles
provide redundant method identities. Use `rd.set_colors({"Baseline": "neutral"})`
when deliberately de-emphasizing a comparator.

Mappings accept token names, Matplotlib color names and hex strings; merge by
default (`replace_existing=True` replaces the mapping). Fallback identities
use SHA-256, so changing input order, Python hash seeds or processes does not
change a label's color/marker/style. A small finite palette cannot uniquely
encode arbitrarily many names. Register explicit colors for a known method
roster, especially across projects; the sidecar records the effective map.
Grayscale intentionally converts all identities to the grayscale palette,
with line styles/markers and grouped-bar hatching. It supersedes color overrides.

## Export

```python
fig.save(output_dir / "figure.png")                 # default 300 DPI
fig.save(output_dir / "figure", dpi=600)            # still PNG only
# Only when PDF is explicitly wanted:
fig.save(output_dir / "figure.pdf")
fig.save(output_dir / "figure", formats=["png", "pdf"])
```

Only PNG and PDF are accepted; no SVG, EPS, JPEG or TIFF export exists.
Physical page dimensions are preserved (`bbox_inches=None`) across formats.
PDF uses Type 42 font embedding where supported by Matplotlib. Dense scatter
may opt into `rasterized=True`; text/axes remain vector in PDF. PNG remains
lossless; changing DPI never changes the data or the inch dimensions.

Existing paths fail unless `overwrite=True`. The exporter preflights all
destinations and rejects symlink targets, renders temporary files in the
destination directory, then atomically publishes each file. A multi-file
save is **not** a filesystem-wide transaction. `transparent=True` is explicit;
the default white background is more predictable in print. No dates are
embedded in PDF metadata; byte identity across fonts/backend versions is not
promised. Notebook `_repr_png_()` uses a 150 DPI in-memory PNG preview.

By default, a matching `.json` sidecar records configuration, physical size,
semantic colors, plot transforms, Matplotlib/NumPy versions, output SHA-256
hashes and the measured quality report. `manifest=False` disables it. Supply
source-data/analysis provenance in your own project alongside the figure;
the library does not infer your experiment's sampling assumptions.

## QA and its limits

```python
report = fig.validate()
print(report.passed, report.to_dict())
```

Constrained layout reserves margins and colorbars. Three deterministic
draw/refinement passes rotate crowded categorical ticks, reduce numeric tick
density and reduce legend columns. Annotation labels try ten fixed offsets
and retain leader lines. Data, label text and font sizes are never deleted or
shrunk to force a pass.

Errors include measured text overlap, text outside the canvas, missing axis
labels, text below the preset minimum, invalid limits, empty registered panels,
truncated bar/area/histogram baselines and inconsistent registered colors.
Warnings include potentially ambiguous colors, low contrast and legends over
data. A report always states its coverage limits. Non-strict export warns on
errors; strict export/preview raises `QualityError` before writing an image.

Bounding boxes are conservative and can report false positives, especially
rotated text, mathtext or closely interleaved artists. Arbitrary patches,
images, data-mark occlusion, glyph availability, full color-vision simulation,
and the scientific truth of input data are not established by this check.
Custom native edits can exceed the registered metadata; always inspect the
final artwork and preserve source data. Very dense text has no automatic
solution at a fixed physical size: use fewer panels, more space, shorter
human-reviewed labels, or a caption. Never treat warnings or a passing report
as a claim of accessibility or journal acceptance.

## Troubleshooting / Not now

- Missing SciPy: install `[stats]` for KDE and Student-t intervals.
- Missing labels: arrays carry no units; pass `xlabel` and `ylabel`.
- Duplicate x: separate replicate curves with a hue, or use `unit`/`error`.
- Layout failure: inspect `report.errors`, then widen the figure or divide
  dense panels. Strict mode deliberately does not hide the problem.
- Matplotlib-native custom figures: use `ax=`; rendering is not thread-safe.
- Not now: 3D effects, animation, interactive web charts, arbitrary-artist
  collision proofs, automatic inferential tests, smoothing, multiclass ROC
  averaging and additional rendering backends. They are not advertised APIs.
