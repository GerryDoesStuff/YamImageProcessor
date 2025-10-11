# YamImageProcessor

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
