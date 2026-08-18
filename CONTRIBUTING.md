# Contributing

Thank you for contributing to Reconstrura Splat Animator.

## Development setup

Python 3.10 or newer and FFmpeg are required. Create the development
environment and run the checks with:

```bash
uv sync --extra dev
uv run ruff check .
QT_QPA_PLATFORM=offscreen uv run pytest -q
```

Keep changes focused, add or update tests for behavior changes, and update the
README when user-facing behavior changes.

Contributions created with AI assistance are welcome, provided they are
carefully reviewed and tested. Contributors remain responsible for the
quality, correctness, and licensing of everything they submit.

## Contribution license

By submitting a contribution, you agree that it is licensed under the project's
MIT License and that you have the right to submit it under those terms. Do not
submit proprietary code, confidential information, generated output containing
private customer data, or assets whose redistribution terms are unclear.
