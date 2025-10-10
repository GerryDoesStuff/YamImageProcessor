## Design Document: Modular Microscopic Image Processing App (YamImageProcessor)

### 1. Purpose / Core Functionality

The Modular Microscopic Image Processing App (MMIPA) is a cross-platform, research-oriented desktop application providing a unified framework for preprocessing, segmentation, and analysis of microscopic images. It aims to be modular, reproducible, and extensible—designed for scientists and developers who value performance, transparency, and robustness.

The app’s core principles are:

* Transparency and reproducibility of workflows.
* Modular design with isolated, reusable components.
* Multithreaded performance with non-blocking UI.
* Accessibility and cross-platform compliance.
* Data integrity and graceful failure handling.

---

### 2. Architectural Overview

The application has four primary layers:

**Core Layer:** Handles app lifecycle, settings, logging, and thread management.

**Processing Layer:** Defines pipelines and data transformations.

**UI Layer:** Manages visualization, user interaction, and feedback.

**Plugin Layer:** Enables runtime discovery and integration of modular extensions.

Core design follows MVC-like separation with dependency inversion and thread safety.

---

### 3. Software Design Principles

1. Loose coupling and interface-driven design.
2. Thread-safe background processing.
3. User operations are undoable, atomic, and logged.
4. Fail-safe defaults: invalid parameters auto-sanitized.
5. Non-blocking feedback with responsive UI.
6. Transparent parameter persistence.
7. Portable, readable data formats.

---

### 4. Core Components

**AppCore:** Initializes application, logging, preferences, and modules.

**SettingsManager:** Unified configuration storage via `QSettings`, with JSON import/export.

**PipelineManager:** Manages ordered transformations, supports reordering, enabling/disabling, and undo/redo.

**ThreadController:** Runs computational steps asynchronously with progress signals and cancellation.

**Logger:** Centralized multi-level logger with rotating files and optional developer diagnostics.

**ErrorDialog:** Displays contextual errors with traceback copy capability.

---

### 5. User Interface Design

* PyQt-based main window with resizable, DPI-aware design.
* Dockable panels for pipeline, logs, and properties.
* Status bar and progress indicators.
* Keyboard shortcuts and right-click context menus.

**Menus:** File, Edit, View, Modules, and Help.
**Dialogs:** Non-blocking, real-time preview for parameter adjustments.
**Tooltips:** All non-obvious elements documented.

---

### 6. Data Flow

1. User loads image.
2. Pipeline constructed or imported.
3. Processing runs asynchronously.
4. Output displayed, logged, and optionally saved.

Intermediate states are stored for reproducibility.

---

### 7. Good Practices Implemented

* Parameter validation before execution.
* Isolation of failing modules.
* Non-destructive operations.
* Atomic file writes.
* Color consistency between libraries.
* Cross-platform file handling using `pathlib`.
* Consistent undo/redo model.

---

### 8. Plugin Architecture

Modules in `/modules` are auto-discovered via importlib. Each must implement `register_module(app_core)` and subclass `ModuleBase`.

**ModuleBase:** Defines metadata, setup, and pipeline registration methods.

Module authors should:

* Handle exceptions internally.
* Provide safe defaults.
* Include documentation and tooltips.
* Avoid global state.

---

### 9. Performance Optimization

* ThreadPoolExecutor for parallel processing.
* Cached intermediate results.
* In-place NumPy operations to minimize memory.
* OpenCV/scikit-image preferred for speed.
* GPU acceleration hooks for future expansion.

---

### 10. Persistence and File I/O

* Supports `.png`, `.jpg`, `.tiff`, `.bmp`, `.npy`.
* Metadata stored in JSON alongside results.
* Auto-backup before overwriting files.
* Autosave interval configurable.

---

### 11. Accessibility and Localization

* Full keyboard support.
* High-contrast themes.
* Scalable icons and fonts.
* Language packs via Qt translation files.

---

### 12. Security and Privacy

* Input sanitization and isolated temp directories.
* No hidden network activity.
* Logs anonymized.
* Crash recovery enabled with safe restore.

---

### 13. Diagnostics and Logging

* Logs stored in user directory.
* In-app diagnostics panel for logs and thread monitoring.
* Opt-in telemetry for debug mode only.

---

### 14. Updates and Maintenance

* Version metadata and update checker.
* Changelog previews before updates.
* Secure module updates with signature validation.

---

### 15. Scalability

* Handles large images with tiling and lazy loading.
* Memory-efficient caching.
* Progressive preview rendering for large datasets.

---

### 16. Testing and Quality Control

* Unit tests for pipelines and UI.
* GUI testing via `pytest-qt`.
* Code formatting (Black) and linting (Flake8).
* Static typing (mypy).
* CI/CD integration for automated validation.

---

### 17. Future Enhancements

* Real-time camera input.
* Machine learning inference engine.
* Cloud-based processing interface.
* Plugin marketplace.
* Support for 3D and hyperspectral images.
* App integration into a wider app network (i.e., taking images from a microscopy app for more in-depth processing).
---

### 18. Directory Structure

```
microscopic_app/
├── core/
├── ui/
├── modules/
├── resources/
├── tests/
├── docs/
└── main.py
```

---

### 19. Summary

The MMIPA architecture balances modularity, reproducibility, and performance while maintaining usability and security. It provides a foundation for building specialized microscopy tools and advanced analytical extensions with minimal rework.

Note to agent - use the extraction, prprocessing, and segmentation Python scripts as examples of what minimal functionality needs to be implemented in the app
