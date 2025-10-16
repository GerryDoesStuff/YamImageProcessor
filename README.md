# YamImageProcessor

[![CI](https://github.com/YamLabs/YamImageProcessor/actions/workflows/ci.yml/badge.svg)](https://github.com/YamLabs/YamImageProcessor/actions/workflows/ci.yml)

The YamImageProcessor project provides a modular foundation for building a microscopic image processing desktop application. The
system is split into clear layers covering core bootstrapping, processing workflows, UI presentation, and plugin discovery in l
ine with the architectural design documents.

## Package layout

```
yam_processor/
├── core/        # Logging, settings, threading, and application bootstrap
├── processing/  # Image processing pipelines (to be implemented)
├── ui/          # Qt-based UI components (to be implemented)
└── plugins/     # Built-in and third-party extensions
```

The `AppCore` class coordinates the foundational services:

* **Logging** – centralised configuration with rotation and optional developer diagnostics.
* **Settings** – unified QSettings-based manager supporting JSON import/export for reproducible configurations.
* **Threading** – background task controller built on `ThreadPoolExecutor` with cooperative cancellation.
* **Plugins** – discovery of modules in the configured plugin packages that expose a `register_module(app_core)` function.

## Quick start

```python
from yam_processor import AppCore

app_core = AppCore()
app_core.bootstrap()
# Application services (settings, logging, threading, plugins) are now ready.
```

This repository currently focuses on establishing the foundation; additional processing pipelines and UI components can be buil
t on top of the provided scaffolding.

## Version metadata

The installed build number is exposed via `yam_processor.__version__` and
`yam_processor.get_version()`. When package metadata is unavailable—such as
running from a source checkout—the helper returns `"0.0.0"` so tooling still
receives a stable string.

## Update checks and telemetry

Update polling and telemetry are opt-in features controlled on
`AppConfiguration`. Both remain disabled by default:

```python
from yam_processor import AppConfiguration, AppCore

config = AppConfiguration(
    enable_update_checks=True,
    telemetry_opt_in=True,
)
app_core = AppCore(config)
app_core.bootstrap()
```

When telemetry is opted in, the flag is persisted through the settings manager
under the `telemetry/opt_in` key. Provide an explicit developer- or user-facing
toggle before enabling telemetry so consent is always captured.

## Localisation

Strings in the Qt user interface are translation-ready.  See
[`docs/TRANSLATIONS.md`](docs/TRANSLATIONS.md) for instructions on generating and
packaging language packs with Qt Linguist tools.

## Developer documentation

- [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) – guidance for authoring
  new processing modules, integrating with `AppCore`, and meeting logging and UI
  accessibility conventions.

## Development tooling

This repository standardises formatting, linting, and type checking so
contributors have a consistent baseline:

- Install the toolchain with `pip install -r requirements-dev.txt`.
- Format and lint the project via `scripts/format.sh`, which runs Black,
  Flake8, and mypy with the configuration defined in `pyproject.toml` and
  `setup.cfg`.

`pyproject.toml` configures Black (88 character lines, targeting Python 3.10)
and applies strict mypy defaults suitable for the codebase. Flake8 mirrors the
same line length and enables `flake8-bugbear` for additional checks to keep the
codebase healthy.

## Continuous integration

Automated checks run through [GitHub Actions](.github/workflows/ci.yml) on every
push and pull request. The workflow provisions Python 3.10, restores a cached
`.venv`, installs runtime dependencies from `requirements.txt` alongside the
tooling in `requirements-dev.txt`, and runs the same quality gates that
contributors use locally:

- `black --check` ensures formatting stays consistent.
- `flake8` enforces style and bug-finding lint rules.
- `mypy` performs static type analysis.
- `pytest` (with `pytest-qt`) exercises the test suite, including the Qt UI
  components in headless mode.

All jobs must pass before changes are merged. If any command fails locally,
resolve the issue before pushing to avoid blocking CI.
