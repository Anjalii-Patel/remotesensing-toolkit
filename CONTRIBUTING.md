# Contributing to rs-toolkit

## Setup

```bash
git clone https://github.com/<your-username>/remote-sensing-toolkit.git
cd remote-sensing-toolkit
pip install -e ".[dev]"
```

## Run tests

```bash
pytest -v
```

## Code style

- Keep functions concise and NumPy-idiomatic.
- All arrays follow `(C, H, W)` convention (bands-first).
- Use NaN for nodata — never use sentinel values like -9999 in outputs.
- Add docstrings for public functions.

## Adding a new spectral index

1. Add the function to `src/rs_toolkit/indices.py`
2. Register it in `INDEX_REGISTRY`
3. Add a test in `tests/test_toolkit.py`
4. Update `README.md` formulae section

## Pull requests

- One feature per PR.
- Include tests for new functionality.
- Keep commits atomic and well-described.
