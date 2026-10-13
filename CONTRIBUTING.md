# Contributing

Use Python 3.10+ and a uv-managed environment. The initial compatibility matrix
targets Python 3.10 / Matplotlib 3.8 and Python 3.12 / Matplotlib 3.10. Versions
outside the actual validation record are supported by the declared dependency
range, but are not claimed to have been exercised locally.
The older Matplotlib 3.8 job pins pyparsing 3.2.5 because newer pyparsing emits
deprecation warnings from Matplotlib's own font parser, which the strict
warnings-as-errors test policy correctly exposes. The 3.10 job uses pyparsing 3.3.3.

```bash
uv venv /path/to/scratch/env/rendeer-dev
uv pip install --python /path/to/scratch/env/rendeer-dev/bin/python -e '.[dev]'
```

In this workspace, use the scratch-only shell launcher from this directory:

```bash
bash scripts/check.sh -m pytest -p no:cacheprovider \
  --basetemp=/scratch/mingyan_root/mingyan/jmtang/outputs/LLM4Policy/synthetic/rendeer/tmp/pytest \
  --junitxml=/scratch/mingyan_root/mingyan/jmtang/outputs/LLM4Policy/synthetic/rendeer/pytest.xml
bash scripts/check.sh -m ruff check src tests examples --no-cache
bash scripts/check.sh -m ruff format --check src tests examples --no-cache
bash scripts/check.sh -m mypy --cache-dir=/scratch/mingyan_root/mingyan/jmtang/outputs/LLM4Policy/synthetic/rendeer/cache/mypy
bash scripts/check.sh -m build --no-isolation \
  --outdir /scratch/mingyan_root/mingyan/jmtang/outputs/LLM4Policy/synthetic/rendeer/dist
```

The runner sets Matplotlib/IPython caches, Jupyter runtime, temporary files,
and `PYTHONDONTWRITEBYTECODE` to avoid generated files in the checkout. Source
Python belongs in `src`, examples in `examples`; `scripts` contains shell
launchers only. `RENDEER_PYTHON` selects another uv environment.

Tests derive numerical expectations from hand-computed cases, not from the
implementation. Add a failing semantic or rendering case before changing
behavior. Strict warnings catch Matplotlib deprecations. The reviewed
`tests/reference/learning.png` is a final curated test fixture, generated from
the fixed synthetic example at 300 DPI with NumPy 1.26.4 / Matplotlib 3.10.8 /
DejaVu Sans. Visual comparison permits a small pixel RMS tolerance for backend
variation; do not update it merely to silence an unintended visual change.
`learning-mpl311.png` is a separately inspected Matplotlib 3.11.2 / NumPy 2.5.4
reference because font/layout metrics changed slightly; it uses the same data
and physical size. CI includes this current dependency combination as well.

After a figure change, run the one-item export smoke first, then the complete
gallery, and inspect the exported PNGs at final physical dimensions:

```bash
bash scripts/check.sh examples/gallery.py --only line --output /path/to/scratch/smoke
bash scripts/check.sh examples/gallery.py --output /path/to/scratch/gallery
```

The Jupyter notebook executes the same public APIs, displays figures and
structured failures, and runs the package tests. Its committed output is a
curated walkthrough, not a substitute for re-execution after code changes.
PNG is the only automatically generated figure format. Test or export PDF
only when explicitly requested; never add automatic SVG/JPEG/etc. exports.

The CI workflow is for the **standalone package repository**. Nested workflows
under `3rd_party/Rendeer/.github` do not automatically run in the parent repo.
Check both package builds and a wheel-only installation before releasing.
Do not treat the parent repository's unrelated test failures as a package pass.
