# Release checklist

## One-time public-release checks

- Confirm every contributor or employer with a copyright interest has approved
  the MIT release.
- Confirm the Reconstrura logo permission covers public repository, source
  archive, wheel, and official binary redistribution. Remove or replace the
  logo if it does not.
- Review the complete Git history for credentials, private paths, customer
  names, scene data, large generated files, and third-party code or assets.
- Add source, issue-tracker, and changelog URLs to `[project.urls]` in
  `pyproject.toml` after the public repository URL is final.
- Enable continuous integration for every supported Python version and target
  operating system, or narrow the classifiers to the tested support matrix.

## Every release

1. Update the version in `pyproject.toml` and `src/splat_animator/__init__.py`.
2. Update `CHANGELOG.md`, dependency constraints, `uv.lock`, and
   `THIRD_PARTY_NOTICES.md`.
3. Run the local verification suite:

   ```bash
   uv lock --check
   uv sync --extra dev --locked
   uv run ruff check .
   QT_QPA_PLATFORM=offscreen uv run pytest -q
   uv pip check
   ```

4. Audit exact locked dependencies for known vulnerabilities and license
   changes with maintained tools and upstream metadata.
5. Build into an empty directory and inspect both archives:

   ```bash
   uv build --out-dir dist
   uvx twine check dist/*
   tar -tzf dist/*.tar.gz
   unzip -l dist/*.whl
   ```

6. Install the wheel in a clean environment, run the CLI help command and test
   suite, and perform at least one real GPU preview and FFmpeg render on each
   supported release platform.
7. Upload to TestPyPI first, install that exact artifact, then publish the same
   verified files to PyPI.
8. Create a signed version tag and publish checksums for release artifacts.

Do not bundle PySide6/Qt or FFmpeg into a desktop installer without performing
a new license audit of the exact binary contents. Source-package conclusions do
not automatically cover a frozen application bundle.
