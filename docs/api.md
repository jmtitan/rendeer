# API and scientific semantics

All examples are executable in [gallery.py](../examples/gallery.py) and the
repository's `notebooks/test/test_rendeer.ipynb`. Plot functions return a
`rendeer.Figure`; the same named methods on `fig['a']` return a `Panel`.
Only `facet` and `pairplot` create complete grids rather than single panels.

## Shared input conventions

Use `rd.line(frame, x='step', y='accuracy', hue='method')` for a DataFrame or
column mapping, and `rd.line(x=array_x, y=array_y, ...)` for NumPy/sequence data.
String arguments name columns, not scalar data. Arrays require explicit
`xlabel`/`ylabel` for strict validation; table inputs infer labels from column
names. Include units in overrides. `title=` is optional and never inferred.

Standalone primitives accept `ax=None`, `width='single'|'double'|inches`, and
`height=inches`. `ax=` attaches the returned wrapper to that existing Axes and
preserves its ownership. Its existing fonts are not retroactively changed.
Figure properties: `fig.mpl`, `fig.ax` for a single panel, `fig['name'].ax` for
named panels. `fig.render()` refines layout; `fig.validate()` returns a
`QualityReport`; `fig.save(...)` returns the exported image paths.

## Comparison

| Function | Main keyword arguments and grammar |
| --- | --- |
| `line(data=None, *, x, y, hue=None, ...)` | `label`, `color`, `sort=False`, `xscale`, `yscale`, `legend`. Preserves input observation order and NaN gaps. Duplicate x within a series raises unless explicit summarization is requested. Native `Axes.plot` styling kwargs are accepted. |
| `scatter(data=None, *, x, y, hue=None, ...)` | `label`, `color`, `xscale`, `yscale`, `legend`, `rasterized=False`, plus native scatter kwargs. Markers are semantic; `s` denotes point area, not radius. Missing y values are not drawn and their count is recorded. |
| `bar(data=None, *, x, y, hue=None, ...)` | `stacked=False`, `orient='vertical'|'horizontal'`, `order`, `highlight`, `label`, `color`, `legend`, native bar kwargs. x is always category and y is always value, including horizontal plots. One value per category/hue; categories retain first appearance unless a complete explicit order is supplied. Positive and negative stacks have separate zero baselines. Incomplete stacks raise. Missing grouped bars are omitted, not zero-imputed. |
| `interval(data=None, *, x, y, lower, upper, uncertainty, ...)` | Absolute bounds enclosing each point; required uncertainty description. Optional `hue`, `label`, `legend`, labels/title and native errorbar kwargs. A point-range plot is the default; `linestyle='-'` explicitly joins points. |
| `area(data=None, *, x, y, ...)` | Single observed series filled to zero. Optional `label`, `color`, labels and native fill kwargs. NaN creates a gap. |

Line uncertainty has two mutually exclusive routes:

```python
# Compute across independent replicate IDs; no row-wise pseudoreplication.
fig = rd.line(runs, x="step", y="accuracy", hue="method", unit="seed", error="sd")

# Supply precomputed absolute bounds, with their actual meaning.
fig = rd.line(summary, x="step", y="mean", lower="lo", upper="hi",
              uncertainty="95% subject-level bootstrap CI, seed=7")
```

`rd.summarize(data, *, x, y, hue=None, unit, error='sd')` returns grouping
columns plus `estimate`, `lower`, `upper`, `n`, `missing`. Each unit contributes
at most once to each group. Definitions: sample SD (`ddof=1`), SE=SD/sqrt(n),
`ci95`=mean ± t(0.975,n−1)×SE, and `range`=[minimum,maximum]. Only `ci95`
requires SciPy. Fewer than two observed units produces NaN bounds and a warning,
including range, to avoid implying reliable uncertainty from one observation.
All-missing groups are retained as gaps. Missing y values are excluded only
from that point's summary and reported in counts. CI assumptions must be
appropriate for the user's experiment; repeated steps do not increase n.

Learning/training-validation curves, ablations, sensitivity analyses,
hyperparameter sweeps, scaling curves, rankings, time-series comparisons and
accuracy-versus-latency/memory use these same grammars. A horizontal ranking
requires an explicit `order`; no method is silently sorted or hidden.
Use `xscale='log'`, `yscale='log'`, or `'symlog'` when justified. Log inputs
and interval bounds must be positive. Symlog uses Matplotlib's default linear
threshold (2); customize the native axis explicitly if another threshold is
scientifically appropriate, then inspect and document it.

## Distributions

Each function takes `data=None, *, x, hue=None, missing='raise', xlabel=None`.
`missing='drop'` is explicit and records the count excluded. Empty groups fail.
No function suppresses outliers.

| Function | Specific arguments and meaning |
| --- | --- |
| `hist` | `bins='auto'` or explicit edges/count, `density=False`, `ylabel`, `title`, `legend`. One pooled bin grid is used across groups; supplied edges must contain all values. Single samples use bars; comparisons use step outlines. Exact edges are recorded. |
| `kde` | `bandwidth='scott'|'silverman'|float`, `ylabel='Density'`, `title`, `legend`. Optional SciPy Gaussian KDE with 256 evaluation points on each group's observed range. The float is a covariance bandwidth factor, not an absolute bandwidth. Tails are clipped and not renormalized. Two distinct observations are required; estimates on tiny samples warrant caution. |
| `ecdf` | `ylabel='Cumulative probability'`, `title`, `legend`. Exact cumulative counts, tied values share a jump; no smoothing or bandwidth. |
| `box` | `whis=1.5`, `ylabel='Group'`, `title`. Horizontal Tukey box (quartiles, median, whiskers within 1.5 IQR, and visible outliers). Singleton boxes are allowed; they are not uncertainty intervals. |
| `violin` | `bandwidth='scott'|'silverman'|float`, `ylabel='Group'`, `title`. Horizontal kernel-density silhouette with median and extrema; widths normalized within each group, not proportional to sample count. Two distinct observations required. |

For distribution shifts, supply a condition column as `hue`, or compare ECDF
panels with shared limits; no divergence statistic is silently calculated.

## Matrices

- `heatmap(data, *, xticklabels=None, yticklabels=None, xlabel='Column',
  ylabel='Row', cbar_label='Value', cmap='cividis', vmin=None, vmax=None,
  center=None, annot=False, fmt='.2g', colorbar=True, title=None)` accepts a
  2D array or DataFrame. NaNs are gray masked cells, never zero. Explicit
  `center` uses a diverging normalization; select an appropriate diverging
  map. Numeric cell text chooses dark/white by contrast. Saturation counts
  are recorded when limits exclude values.
- `correlation(data, *, method='pearson', missing='raise', annot=True, **heatmap_kwargs)`
  requires numeric columns. `missing='complete'` drops incomplete rows;
  `'pairwise'` uses available pairs and records the pairwise sample counts.
  The scale is fixed at −1 to 1. Spearman/Kendall may require SciPy through
  pandas. Constant columns yield masked undefined correlations.
- `confusion(counts, *, labels=None, normalize=None, annot=True, **heatmap_kwargs)`
  takes a square nonnegative integer count matrix. `normalize='true'|'pred'|'all'`
  divides by row, column or total counts. Empty normalized margins remain
  missing. The entirely unobserved normalized matrix is rejected.

## ML evaluation

- `pareto(data=None, *, x, y, minimize_x=True, maximize_y=True, hue=None,
  xlabel=None, ylabel=None, labels=None, legend=True)` keeps every observed
  point and joins only the 2D nondominated frontier. Lines guide the eye;
  they do not assert interpolated performance. Optional point labels retain
  leader lines. Complexity is O(n log n), excluding annotation/QA costs.
- `roc(*, y_true, y_score, label='Ours', reference=True, legend=True)` requires
  both binary classes, with 1 positive. Tied scores form one threshold. AUC
  uses trapezoidal integration, recorded in provenance. The diagonal is a
  chance reference, without an invented uncertainty interval.
- `precision_recall(...)` uses the same inputs and a step curve. The initial
  no-prediction endpoint is recall 0, precision 1. The optional horizontal
  reference is prevalence. Recorded average precision is the recall-weighted
  step integral, not trapezoidal PR AUC.
- `calibration(*, y_true, y_prob, bins=10, label='Ours', legend=True)` validates
  probabilities in [0,1], uses equal-width bins, and plots each bin at its
  **mean prediction**, not its midpoint. Empty bins stay gaps. Counts and bin
  edges are recorded; error bars are not inferred.

## Composition and annotations

`figure(*, layout=[['a']], width='single', height=None, sharex=False,
sharey=False, width_ratios=None, height_ratios=None, panel_labels=False)` uses
native rectangular subplot mosaics. Repeated names span cells; `'.'` leaves
space empty. Width and height are inches. `width='double'` uses the preset.

- `fig.legend(ncols=None)` deduplicates labels, replaces local legends and
  reserves top space. Different series sharing a label should first be given
  meaningful distinct names.
- `fig.colorbar(panels=['a','b'], label='F1')` requires heatmaps with a common
  scale and map. Create these heatmaps with `colorbar=False`.
- `fig.share_limits(panels=None, axis='y')` applies a common union of limits;
  incompatible scales and inverted axes are rejected.
- `panel.annotate(labels, *, x, y)` retains all labels, uses leader lines,
  and searches ten deterministic offsets after rendering. Unresolved
  overlaps are reported. For significance comparisons, add your independently
  computed statistic/test description with native `panel.ax.annotate`; no
  significance test or stars are invented by the library.
- `panel.dual_axis(*, justification, ylabel)` creates a second y-axis only
  with a nonempty scientific reason. It shares x and records the reason;
  the QA report warns that axis scaling still needs review.
- `facet(data, *, col, kind='line', wrap=3, sharex=True, sharey=True, **plot_kwargs)`
  creates named panels in first-appearance order. No headings are inserted;
  **the caption must map panel letters to the category order** exposed in
  `fig.panels` and provenance. For grammars with hue support, `hue` defaults
  to the facet column to preserve category identities; explicitly use
  `hue=None` to disable this mapping or name another grouping column.
- `pairplot(data, *, columns=None, hue=None)` supports two to four explicitly
  numeric variables, scatter off-diagonal and common-bin histograms on the
  diagonal. The four-variable limit preserves readable panel sizes.

## Migration / customization

```python
# Existing Matplotlib axes can be reused incrementally.
wrapped = rd.scatter(frame, x="latency", y="accuracy", ax=ax)
wrapped.ax.set_xlabel("Latency (ms)")
report = wrapped.validate()

# Optional colormap library; no hard dependency or copied maps.
from cmcrameri import cm
fig = rd.heatmap(values, cmap=cm.batlow, cbar_label="Energy (J)")
```

No backend plugin framework is exposed in 0.1. Native Axes is the extension
boundary. Internal plot registration is not a stable public plugin API.
