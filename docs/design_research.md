# Design research

Sources reviewed on 2026-10-13. These are design inputs, not a claim that an
automatic preset satisfies a publisher's current production checks.

## Official guidance versus library conventions

| Source | Evidence and decision |
| --- | --- |
| [Nature figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) | Label axes and units, use accessible colors and legible sans-serif text, avoid overlapping text and decorative gridlines; embed editable TrueType PDF text. The `nature` preset uses 7 pt body text, a 5 pt validation floor and Type 42 PDF fonts. |
| [Nature panel construction](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) | Design at final size. The preset provides 89 mm and 183 mm widths. Fallback fonts are disclosed; their presence does not establish production compliance. |
| [Nature Methods: Color blindness](https://www.nature.com/articles/nmeth.1618) | Color alone is insufficient for method identities. Retain redundant shape and line encodings when choosing restrained categorical hues. The default is a custom palette, not the Okabe–Ito/Wong palette. |
| [Science initial-manuscript instructions](https://www.science.org/content/page/instructions-preparing-initial-manuscript) | Live retrieval returned HTTP 403. No new Science-specific dimensions, fonts or acceptance claims are derived from the inaccessible page. There is no `science` preset. |
| [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions), [ICLR 2026 author guide](https://iclr.cc/Conferences/2026/AuthorGuide), [ICLR official template](https://github.com/ICLR/Master-Template/blob/master/iclr2026/iclr2026_conference.tex) | Templates determine manuscript space; do not infer a universal ML column width or an official color palette. The general `ml` preset's 3.25/6.75 inch widths are adjustable conventions, not conference requirements. |
| [NeurIPS style-files page](https://neurips.cc/Conferences/2026/PaperInformation/StyleFiles) | The queried 2026 endpoint was unavailable. `ml` is deliberately not a verified NeurIPS 2026 preset. Verify the author's actual template before submission. |
| [JMLR author information](https://jmlr.org/author-info.html) | Follow the journal template and keep captions outside the artwork. No JMLR-specific color system is asserted. |
| [IEEE two-column graphics guidance](https://eds.ieee.org/images/files/Publications/JMEMS_Editors/2columnformatguide.pdf) | Common widths are 3.5 and 7.16 inches; these motivate `ieee`. Check the specific IEEE journal since column sizes can differ. |
| [ACM SIGACCESS figure descriptions](https://www.sigaccess.org/welcome-to-sigaccess/resources/describing-figures/) | A legible PNG alone is not an accessible publication. Examples include prose descriptions and source tables; authors must supply appropriate captions/alternative text. |

The default `paper` design—9 pt labels, 1.4 pt lines, no titles, no grid,
open top/right spines, external legends—is a rendeer convention. It is not a
publisher mandate. Nature's official correct/incorrect graph examples informed
axis, legend and text choices. We do not claim a systematic survey of all
representative figures at every venue.

## Reference implementations reviewed

| Project and implementation | Useful architectural idea | Decision / limitation |
| --- | --- | --- |
| [Matplotlib constrained layout](https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_constrained_layout.py) | GridSpec constraints account for axes decorations and colorbars. | Use native mosaic/axes and rendered bounding boxes. Do not combine `tight_layout` with constrained layout. |
| [Seaborn relational plots](https://github.com/mwaskom/seaborn/blob/master/seaborn/relational.py) | Separate data semantics from artist rendering; distinguish axes-level composition. | Implement explicit summaries outside plotting. Preserve NaN gaps rather than dropping rows and bridging observations. Seaborn is not required. |
| [Seaborn color-palette guide](https://seaborn.pydata.org/tutorial/color_palettes.html) | Moderated saturation can improve aesthetics, but category discrimination still needs review. | Custom blue/coral/teal defaults, with lavender/ochre/rose/slate extensions, white-background contrast screening and redundant markers/styles. No Seaborn dependency or official venue-palette claim. |
| [SciencePlots](https://github.com/garrettj403/SciencePlots) | Small style files make consistent defaults inexpensive. | Use scoped rc settings; no style registration/import side effects or mandatory TeX. |
| [TUEplots bundles](https://github.com/pnkraemer/tueplots/blob/main/tueplots/bundles.py) | Physical dimensions and fonts should vary by output destination. | Immutable theme tokens with rc overrides, not a proliferation of conference-color presets. |
| [adjustText implementation](https://github.com/Phlya/adjustText/blob/master/adjustText/__init__.py) | Measure display-space boxes after drawing; bound refinement iterations. | A small deterministic candidate search handles rendeer annotations. It has finite capacity and reports unresolved collisions; no claim of global optimality. |
| [ggrepel geom](https://github.com/slowkow/ggrepel/blob/master/R/geom-text-repel.R) | Label displacement should retain the anchor via leader lines; stochastic placement needs a seed. | Keep original labels and anchors, use no random placement or automatic label dropping. No R dependency or copied code. |
| [cmcrameri colormap loader](https://github.com/callumrollo/cmcrameri/blob/main/cmcrameri/cm.py) | Distinguish categorical, sequential, diverging and cyclic maps. | Default to built-in cividis / RdBu_r; accept a Matplotlib Colormap object for optional cmcrameri integration. Avoid global registration during import. |
| [tol_colors](https://github.com/Descanonge/tol_colors) | Discrete categorical identities and perceptually meaningful continuous maps solve different tasks. | Central tokens, explicit mapping, grayscale mode, and measurable contrast warnings. Do not certify color-vision accessibility from palette membership. |

Reference code was reviewed for concepts; no implementation, fonts or artwork
were copied into this package. Dependencies keep their own licenses. The new
package source is MIT licensed. SciPy is optional and used for Student-t
quantiles and KDE; NumPy, pandas and Matplotlib are the core dependencies.

## Measurable design decisions

- Preserve the requested page dimensions; never use a tight bounding-box crop.
- Use opaque white output and 300 DPI PNG by default. This repository permits
  PNG and explicitly requested PDF only, so SVG is intentionally unsupported.
- Use restrained categorical hues, with blue/coral/teal for the three default
  method identities. Reserve slate for explicit de-emphasis. The palette
  preview in the acceptance notebook calculates white-background contrast;
  every opaque default swatch must clear 3:1. Keep dark bar edges and redundant
  markers/styles. This screen does not certify translucent fills, grayscale
  separation, color-vision accessibility or publisher acceptance.
- Color lookup uses explicit mappings first and SHA-256 fallback indexes,
  independent of Python hash randomization, category order and process state.
  Finite palettes can collide: warn and recommend an explicit mapping.
- Mean/SD/SE/95% Student-t intervals require the independent sampling unit;
  duplicate unit/step/method rows fail. No automatic smoothing or bootstrapping.
- Check the rendered canvas in a bounded number of passes. Severe measurable
  defects fail strict export; arbitrary artist occlusion remains a human task.
- Configuration uses frozen dataclasses and a ContextVar. Matplotlib itself is
  not thread-safe: concurrent rendering should use processes, not threads.

## Architecture

`config` / `theme` / `colors` define defaults without mutating Matplotlib.
`data` validates and summarizes independent observations. `plots` contains
axes-level grammars registered once for both `rd.line(...)` and
`fig['a'].line(...)`. `Figure` owns native artists and a configuration snapshot;
`Panel` owns an Axes. `layout` refines, `validation` measures, and `export`
writes the physical-size image plus provenance. The native `fig.mpl` and
`panel.ax` remain available; edits through them require another validation.

## Procedural reference

The scientific-visualization skill informed integrity, scoped styling and
rendered-output review. Verified arXiv metadata (current revision at retrieval):
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner
(2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents](https://doi.org/10.48550/arXiv.2609.00065).
