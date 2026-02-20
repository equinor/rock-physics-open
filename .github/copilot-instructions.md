# Copilot Coding Agent Instructions

> Trust these instructions. Only search the repo if something here is incomplete or wrong.

## Repository Overview

**rock-physics-open** is a Python library of rock physics models for quantitative seismic analysis, published on PyPI. Functions take NumPy arrays (or occasionally Pandas DataFrames) as inputs/outputs. ~114 source files, ~17 k lines of Python, 49 test files, 145 tests.

**Python version:** 3.11 (pinned in `.python-version`; CI tests 3.11–3.14).
**Package manager:** [uv](https://docs.astral.sh/uv/). All commands must be prefixed with `uv run` (e.g., `uv run pytest`).
**Build system:** setuptools + setuptools_scm (version derived from git tags, written to `src/rock_physics_open/version.py` — this file is git-ignored and auto-generated).

## Setup & Install

Always run this first after cloning or pulling:

```sh
uv sync --locked --all-extras --dev
```

This installs the package in editable mode plus all dev dependencies into `.venv/`. The `--locked` flag ensures `uv.lock` is respected exactly — **do not run `uv lock`** unless intentionally updating dependencies.

## Validation Commands (CI-Equivalent)

CI runs two jobs on every PR: **lint-and-format** and **test**. Replicate them locally:

### 1. Full pre-commit suite (lint + format + type check)

```sh
uv run pre-commit run --all-files
```

This runs, in order:
- Standard hooks (check-ast, trailing-whitespace, end-of-file-fixer, etc.)
- `ruff check --fix` (linter)
- `ruff format` (formatter)
- `basedpyright` (static type checker)

### 2. Tests

```sh
uv run pytest
```

Tests complete in ~5 seconds. There are no slow or network-dependent tests. **Always run the full suite** — do not skip tests or use `-k` filters unless specifically debugging.

### Running individual checks

```sh
uv run ruff check              # lint only
uv run ruff check --fix        # lint with auto-fix
uv run ruff format             # format in-place
uv run ruff format --check     # format check (no changes)
uv run basedpyright            # type checking
```

### Recommended workflow after making changes

```sh
uv run ruff check --fix && uv run ruff format && uv run basedpyright && uv run pytest
```

Run all four in sequence. Fix any errors before committing.

## Project Layout

```
pyproject.toml              # All config: dependencies, ruff, pytest, basedpyright
uv.lock                     # Locked dependency versions (do not regenerate casually)
.pre-commit-config.yaml     # Pre-commit hook definitions
.python-version             # 3.11

src/rock_physics_open/      # Main package (editable install)
├── equinor_utilities/      # General utilities
│   ├── classification_functions/   # ML classification
│   ├── gen_utilities/              # Input filtering, conversions
│   ├── machine_learning_utilities/ # Pressure models
│   ├── optimisation_utilities/     # Curve fitting helpers
│   ├── snapshot_test_utilities/    # Snapshot testing helpers
│   ├── std_functions/              # Core equations (Gassmann, Hertz-Mindlin, Hashin-Shtrikman, etc.)
│   └── various_utilities/          # Misc (reflectivity, VRH, time shift)
├── fluid_models/           # Brine, gas, oil property models (Batzle-Wang)
├── sandstone_models/       # Cemented, friable, patchy cement models
├── shale_models/           # DEM, SCA, Kuster-Toksöz models
├── span_wagner/            # CO₂ properties (Span-Wagner equations)
├── t_matrix_models/        # T-matrix effective medium models (uses C/Fortran .dll/.so)
└── ternary_plots/          # Ternary diagram utilities

tests/                      # Mirrors src structure with *_tests/ directories
├── conftest.py             # Session-scoped fixtures (testdata dir, data_dir)
├── data/                   # Test data files (CSV, PKL) and snapshots/
└── <module>_tests/         # Test directories per module

.github/workflows/
├── on-pull-request.yaml    # Triggers lint-and-format + test on PRs
├── on-push-main.yaml       # Same + release-please + publish
├── lint-and-format.yaml    # Runs `uv run pre-commit run --all-files`
├── test.yaml               # Runs `uv run pytest` on matrix (3 OS × 4 Python versions)
└── release-please.yaml     # Automated releases via conventional commits
```

## Key Conventions

- **Conventional commits:** Required. Format: `type(scope): description`. Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`. Release-please uses these for changelogs.
- **Strict markers:** `pytest` uses `--strict-markers`. If you add a custom marker, register it in `pyproject.toml` under `[tool.pytest.ini_options]` → `markers`.
- **Type checking:** basedpyright in `recommended` mode. Libraries lacking stubs are listed in `allowedUntypedLibraries` in `pyproject.toml`. Add new untyped libraries there if needed.
- **Ruff config:** Rules and ignores are in `pyproject.toml` under `[tool.ruff]`. Line length is 88. Import sorting uses `combine-as-imports = true`.
- **No display/GUI in tests:** Tests must run headless (CI has no display server). The ternary plot tests already handle this.

## Common Pitfalls

1. **Do not use `pip install`** — always use `uv sync` and `uv run`. The project uses uv for dependency management.
2. **Do not modify `uv.lock` manually** — if you need to add a dependency, add it to `pyproject.toml` and run `uv lock` then `uv sync`.
3. **`version.py` is auto-generated** — never edit it; it is created by setuptools_scm from git tags.
4. **The `t_matrix_models` directory contains compiled binaries** (`.dll`, `.so`) — these are pre-built C/Fortran extensions. Do not modify or rebuild them.
5. **Test data files in `tests/data/snapshots/`** use `.log` extension and are tracked via `.gitattributes` — preserve LF line endings.
6. **All source code is under `src/`** — imports use `from rock_physics_open.…` (not `from src.…`).
