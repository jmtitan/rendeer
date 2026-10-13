# Release preparation

The package version is 0.1.0 (initial/alpha API). The GitHub repository target is
[`jmtitan/rendeer`](https://github.com/jmtitan/rendeer). No distribution has been
uploaded to PyPI. GitHub source upload and PyPI distribution publication are
separate operations.

The source can be used as a standalone checkout or inside LLM4Policy at
`3rd_party/Rendeer`. The parent `.gitignore` ignores `3rd_party`; keep the package's
Git history independent from unrelated parent-project changes. The executed
walkthrough is included at `notebooks/test_rendeer.ipynb` and also maintained
at `notebooks/test/test_rendeer.ipynb` in the parent workspace.

Read-only name checks on 2026-10-13:

- `https://pypi.org/pypi/rendeer/json` returned HTTP 404. This indicates no
  distribution was returned at that moment; it does **not** establish that
  the name is unreserved or that a future upload will be accepted.
- GitHub repository search returned existing names, including `cstef7/rendeer`,
  `lesterchan/rendeer`, and `Shimmen/Rendeer`. Repository names are owner-scoped.
  No target owner was specified, so availability for the intended account
  remains unverified.
- Raw responses are in the local run artifact
  `/scratch/mingyan_root/mingyan/jmtang/outputs/LLM4Policy/synthetic/rendeer/name-availability.json`.

## Publish only after explicit authorization

1. Choose and verify an owner with `gh api repos/OWNER/rendeer`; distinguish a
   real 404 from missing authentication. Recheck the PyPI name and publisher
   permissions. Do not infer authorization from the instructions in this file.
2. Use this standalone package directory, or copy it to a clean checkout on
   scratch. Review the MIT copyright notice and metadata, verify repository
   URLs, and retain the notebook/test fixtures/docs/examples. Do not include
   unrelated parent-project files, credentials, caches or run logs.
3. Install `.[dev]`, run the tests/lint/type checks and gallery review described
   in `CONTRIBUTING.md`. Rebuild with `python -m build --outdir SCRATCH_DIST`.
   Run `python -m twine check SCRATCH_DIST/*` and install the wheel in a fresh
   environment outside the source tree.
4. In that standalone directory, after permission to publish:

   ```bash
   git init -b main
   git add .
   git commit -m "Initial rendeer scientific visualization package"
   # Choose visibility explicitly; --private is a conservative initial setting.
   gh repo create OWNER/rendeer --private --source=. --remote=origin --push
   git tag v0.1.0
   git push origin v0.1.0
   ```

5. Configure PyPI Trusted Publishing in the actual repository/account. Review
   a protected release job before enabling uploads. Alternatively, after
   explicit authorization, use `python -m twine upload SCRATCH_DIST/*` with
   credentials provided by the approved external credential mechanism.
   Never embed tokens in source, workflows, notebooks or logs.

The supplied workflow runs tests and builds only; it does not publish. A local
pass is not a claim that hosted GitHub Actions has run successfully. Additional
OS/font stacks, CJK/TeX environments and real journal production pipelines
require their own validation. See the scratch validation artifacts for what
was actually run in this environment.
